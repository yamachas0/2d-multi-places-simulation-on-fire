"""シミュ後の追加 LLM 抽出: 各参加者が「自分が座学で考えた結果としてフィールドワークで確かめたいこと」を引き出す。

各参加者の memory + 自身の発話 + 受信メッセージ を Gemini に渡して、
「あなたが自分の頭で考えた結論として、現地で確かめたいこと 1-3つ」を JSON で出させる。
他人の発言を借りる伝聞調 (「〇〇さんが言ってた…」) は禁止。

出力: <run_dir>/field_questions.jsonl  (1 行 = 1 agent)
  {axis_id, agent_id, name, questions: [...], one_liner}

次のフィールドワーク版シミュで agent.memory に注入できる形を想定。

Usage:
  ./venv/Scripts/python.exe tools/extract_field_questions.py \
    --run simulations/<run_dir> \
    [--town shinagawa]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from render_ab_comparison import (  # noqa: E402
    load_run, get_catalyst_info, is_participant, per_agent_history, _strip_prompt_leak,
)


SYSTEM_FIELD_Q = (
    "あなたは『たった今、あるまちについて他の参加者と座学で議論し終えた一人の参加者』として答える。"
    "次にあなたは **実際にそのまちを歩く** ことになっている。"
    "**重要なルール:**\n"
    "1. これはあなた自身が座学で自分の頭で考えた結論を述べる場。"
    " 他人の発言を借りる伝聞調 (「〇〇さんが言ってた」「みんなが話してた」) は使わない。"
    "**「自分は」「私は」を主語にして、自分の問題意識として書く。**\n"
    "2. 抽象論ではなく、現地で五感で確かめられそうな具体的なこと (場所・モノ・人の動き・看板・空気感・時間帯) に落とし込む。\n"
    "3. 1〜3 個に厳選。多すぎる場合は本当に気になる方を選ぶ。\n"
    "4. 1個あたり1-2文で短く。\n"
    "5. 推測語 (「たぶん」「だろう」) は使わず、断定的に「自分が確かめたいこと」を書く。\n\n"
    "JSON のみで返答 (前置き禁止、コードブロック禁止):\n"
    '{ "questions": ["...", "..."], "one_liner": "<この人がフィールドワークで一番気にしていることを1文で>" }'
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
        f"上記をすべて踏まえて、**あなたがこの座学を通じて自分の頭で考えた結果**、"
        f"次に実際に{town_name}を歩くときに **自分の目・耳・足で確かめたいこと** を1-3つ、"
        f"システムプロンプトの JSON フォーマット通りに出力する。"
    )


def parse_json_output(text: str) -> dict | None:
    if not text:
        return None
    # コードブロック除去
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
    ap.add_argument("--run", required=True, help="run dir (e.g. simulations/<...>)")
    ap.add_argument("--town", default=None, help="まち名 (override)。未指定なら config metadata から読む")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    run_dir = Path(args.run)
    data = load_run(run_dir)

    # 触媒は除外して参加者のみ抽出
    cat = get_catalyst_info(data)
    participants = [p for p in data["personas"] if is_participant(p, cat["axis_id"])]

    town_name = args.town or data.get("cfg", {}).get("metadata", {}).get("town_name") or "このまち"
    print(f"[info] run={run_dir}, town={town_name}, participants={len(participants)}")

    client = GeminiClient(
        model="gemini-2.5-flash-lite",
        temperature=0.5, max_tokens=600,
        enable_cache=False, enable_structured_output=False,
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []

    def extract_one(persona: dict) -> dict:
        thoughts, sent, recv = per_agent_history(data, persona["id"])
        user_p = build_user_prompt(persona, thoughts, sent, recv, town_name)
        text = client.generate(SYSTEM_FIELD_Q, user_p, temperature=0.5, max_tokens=600)
        parsed = parse_json_output(text or "")
        questions = []
        one_liner = ""
        if parsed:
            questions = [q for q in (parsed.get("questions") or []) if isinstance(q, str)]
            one_liner = (parsed.get("one_liner") or "").strip()
        return {
            "axis_id": persona.get("axis_id"),
            "agent_id": persona.get("id"),
            "name": persona.get("name"),
            "questions": questions[:3],
            "one_liner": one_liner,
            "_raw": (text or "")[:300],  # debug用
        }

    print(f"[info] extracting field questions ({args.workers} parallel)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(extract_one, p): p for p in participants}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  [{r['axis_id']}] {r['name']}: {len(r['questions'])} questions / one_liner={r['one_liner'][:40]}")

    # axis_id 順に並べ替え
    results.sort(key=lambda r: r.get("axis_id") or "")

    out_path = run_dir / "field_questions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_path} ({len(results)} entries, {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
