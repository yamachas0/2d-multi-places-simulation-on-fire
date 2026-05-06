"""【Phase B】 教室Phase の出力を FWPhase に持ち越すための「ハンドオフ情報」を抽出する。

各 agent (触媒は除く) について、教室シミュの最後の状態を Gemini に与え、3つを引き出す:
- intent      : 「これから現地でどう過ごしたいか」を1-2文 (behavior intent)
- future_image: 「座学で描いた未来像」を1-2文 (どんな品川の未来像を描いたか)
- key_memories: 自分の記憶のうち、FW phase に持ち越したいキー (3つくらい)

Phase B の build_shinagawa_field_config.py が <run_dir>/fw_handoff.jsonl を読んで
Agent の initial_memory に injection する。

Usage:
  python tools/extract_fw_handoff.py --run simulations/<classroom_run_dir> [--workers 8]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from render_ab_comparison import (  # noqa: E402
    load_run, get_catalyst_info, is_participant, per_agent_history, _strip_prompt_leak,
)


SYSTEM_HANDOFF = (
    "あなたは『たった今、品川というまちについて他の参加者と座学で議論し終えた一人の参加者』です。\n"
    "これから席を立って、実際にその品川駅前を歩き回って確かめに行くところ。\n\n"
    "次の3つを答えてください:\n\n"
    "1. **intent** (1-2文): 「これから現地でどう過ごしたいか」を、自分の言葉で。\n"
    "   - 具体的でも抽象的でもいい。「特に決めてない」もOK。\n"
    "   - 「○○さんと一緒に行く」「○○エリアを見たい」「人の流れを見ているだけでいい」 など、自然に出る一言。\n"
    "   - 自分は、と主語を立てて書く。他人の発言を借りる伝聞調 (「みんなが〜と言ってた」) は避ける。\n\n"
    "2. **future_image** (1-2文): 座学を通じて自分の中で固まりつつある「品川の未来像」。\n"
    "   - 具体的になっていなければ「まだ漠然としている」程度でもよい。\n"
    "   - 自分が描いた像を率直に。理想を盛らない、自分の言葉。\n\n"
    "3. **key_memories** (2-3個): 座学で交わされた話のうち、現地に持って行くべき自分にとっての要点。\n"
    "   - 他の人が言ってた重要な指摘でもOK (引用形式)。\n"
    "   - 自分が気になって離れない論点でもOK。\n"
    "   - 1つあたり40字程度の短い文。\n\n"
    "JSON のみで返答 (前置き禁止、コードブロック禁止):\n"
    '{ "intent": "...", "future_image": "...", "key_memories": ["...", "...", "..."] }'
)


def build_user_prompt(persona: dict, thoughts: list[str], sent: list[dict],
                       recv: list[dict], town_name: str) -> str:
    name = persona.get("name", "?")
    age = persona.get("age", "?")
    occupation = persona.get("occupation", "")
    bg_short = (persona.get("background", "") or "").split("──")[0].strip()[:300]
    catch = persona.get("catchphrase", "") or ""
    ext = persona.get("temperament_extroversion", "?")
    opt = persona.get("temperament_optimism", "?")
    cur = persona.get("temperament_curiosity", "?")

    sent_block = "\n".join(f"[step{m.get('step')}→{m.get('to_name','?')}] {m.get('message','')}" for m in sent[:30])
    recv_block = "\n".join(f"[step{m.get('step')} from {m.get('from_name','?')}] {m.get('message','')}" for m in recv[:30])
    thoughts_block = "\n".join(thoughts[:30])

    return (
        f"## あなた\n"
        f"名前: {name} ({age})\n"
        f"立場: {occupation}\n"
        f"背景 (要約): {bg_short}\n"
        f"気質: 外向性={ext} / 楽天性={opt} / 好奇心={cur}\n"
        f"口癖: 「{catch}」\n\n"
        f"## 議論した対象のまち: {town_name}\n\n"
        f"## あなたの座学中の思考記録 (時系列、step順)\n{thoughts_block or '(なし)'}\n\n"
        f"## あなた自身の発話 (時系列)\n{sent_block or '(なし)'}\n\n"
        f"## あなたが受け取った発話 (時系列)\n{recv_block or '(なし)'}\n\n"
        f"## 課題\n"
        f"上記すべて踏まえて、システムプロンプトの3項目を JSON で出力する。"
    )


def parse_json_output(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = _strip_prompt_leak(text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--town", default=None, help="町名 override (default: config metadata の town_name)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    run_dir = Path(args.run)
    data = load_run(run_dir)
    cat = get_catalyst_info(data)
    participants = [p for p in data["personas"] if is_participant(p, cat["axis_id"])]

    town_name = args.town or data.get("cfg", {}).get("metadata", {}).get("town_name") or "このまち"
    print(f"[info] run={run_dir}, town={town_name}, participants={len(participants)}")

    client = GeminiClient(
        model="gemini-2.5-flash-lite",
        temperature=0.5, max_tokens=900,
        enable_cache=False, enable_structured_output=False,
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def extract_one(persona: dict) -> dict:
        thoughts, sent, recv = per_agent_history(data, persona["id"])
        user_p = build_user_prompt(persona, thoughts, sent, recv, town_name)
        text = client.generate(SYSTEM_HANDOFF, user_p, temperature=0.5, max_tokens=900)
        parsed = parse_json_output(text or "")
        intent = ""; future_image = ""; key_memories = []
        if parsed:
            intent = (parsed.get("intent") or "").strip()
            future_image = (parsed.get("future_image") or "").strip()
            kms = parsed.get("key_memories") or []
            if isinstance(kms, list):
                key_memories = [k.strip() for k in kms if isinstance(k, str) and k.strip()]
        return {
            "axis_id": persona.get("axis_id"),
            "agent_id": persona.get("id"),
            "name": persona.get("name"),
            "intent": intent,
            "future_image": future_image,
            "key_memories": key_memories[:3],
        }

    print(f"[info] extracting fw_handoff (workers={args.workers}) ...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(extract_one, p): p for p in participants}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"  [{r['axis_id']}] {r['name']}: intent={r['intent'][:40]!r}")

    results.sort(key=lambda r: r.get("axis_id") or "")
    out_path = run_dir / "fw_handoff.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_path} ({len(results)} entries, {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
