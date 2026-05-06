"""【えびねこ評価軸 / 今回限定実装】

シミュログ (messages.jsonl + memory_reasoning.jsonl) を読み、各エントリに
A/R/E/I/P/B の6軸マルチラベルタグを Gemini で付与する。

  A: Awareness (気づき)         — 場・他者・自分の状態に気づく発言
  R: Reflection (内省)          — 自分の経験・前提・価値観を振り返る発言
  E: Engagement (関与)          — 議論やテーマに関わろうとする姿勢
  I: Interaction (相互作用)     — 他者の発言を受けて反応・応答する
  P: Perspective Shift (視点変容)— 自分の見方が変わった/組み替わった瞬間
  B: Behavior Change (行動変容) — これからの行動・選択を変えると示唆する

出力: <run_dir>/tagged_log.jsonl
  {step, type: "msg"|"memory", id, name, text, tags: ["A","R"], quote_short}

Usage:
  python tools/tag_arepib.py --run simulations/<run_dir>
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
from render_ab_comparison import load_run, get_catalyst_info  # noqa: E402


SYSTEM_TAG = (
    "あなたは観察者として、対話シミュレーションのログを読み、各エントリに『えびねこ評価軸6つ』のうち"
    "**該当するすべてのコード** をマルチラベルで付ける。\n\n"
    "## 6軸の定義\n"
    "- **A** Awareness (気づき): 場・他者・自分の状態に気づく発言、「あれ？」「そういえば」「〜だな」のような気づきの言語化\n"
    "- **R** Reflection (内省): 自分の経験・前提・価値観・過去を振り返る、「自分は今まで〜と思ってきた」「〜だから自分は」のような自己言及\n"
    "- **E** Engagement (関与): テーマや議論に乗る・参加する姿勢、「やろう」「考えてみたい」「気になる」のような前向き表明\n"
    "- **I** Interaction (相互作用): 他者の発言や存在を受けて応答する、引用する、相手の立場を借りる、対話のキャッチボールが成立\n"
    "- **P** Perspective Shift (視点変容): 自分の見方が変わった/組み替わった瞬間、「今までは〜と思ってたけど」「〇〇って言われて見方が」のような転換\n"
    "- **B** Behavior Change (行動変容): これから自分の行動や選択を変える/変えたい、「実際に〜してみる」「次は〜しよう」「〜を確かめに行く」のような未来志向の意図\n\n"
    "## ルール\n"
    "- 1 entry に **複数タグOK** (例: 「みんなの話を聞いて、自分も今度〜してみようかな」 → R + I + B)\n"
    "- 該当しなければ空配列 `[]` でよい (例: 雑談・あいさつ・何も意味がない発話)\n"
    "- 推測しない。**そのテキストから直接読み取れる** ものだけタグする\n"
    "- 「Skipped」「should_speak=False」のような技術メタは空配列\n\n"
    "## 出力フォーマット\n"
    "JSON 配列のみ (前置き・コードブロック禁止):\n"
    '[{"i": <入力のi>, "tags": ["A","R"]}, ...]\n'
    "入力の i (index) を必ず一致させる。"
)


def build_user_prompt(batch: list[dict]) -> str:
    lines = ["以下のエントリそれぞれに、6軸のうち該当するタグをマルチラベルで付ける。"]
    for entry in batch:
        text = (entry.get("text") or "").replace("\n", " ").strip()
        if len(text) > 280:
            text = text[:280] + "…"
        lines.append(f'i={entry["i"]}: ({entry["type"]}/{entry.get("name","?")}) "{text}"')
    lines.append("")
    lines.append("JSON 配列のみで返答。")
    return "\n".join(lines)


def tag_batch(client, batch: list[dict]) -> dict[int, list[str]]:
    text = client.generate(SYSTEM_TAG, build_user_prompt(batch), temperature=0.2, max_tokens=1500)
    if not text:
        return {}
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    m = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    valid_tags = {"A", "R", "E", "I", "P", "B"}
    for r in arr:
        if not isinstance(r, dict): continue
        i = r.get("i")
        tags = r.get("tags") or []
        if isinstance(i, int):
            out[i] = [t for t in tags if t in valid_tags]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    run_dir = Path(args.run)
    data = load_run(run_dir)

    cat = get_catalyst_info(data)
    cat_id = cat["persona"]["id"] if cat["is_present"] else None

    # 入力エントリ作成: messages + memory_reasoning
    name_by_id = {p["id"]: p["name"] for p in data["personas"]}
    entries: list[dict] = []
    i = 0
    for m in data["msgs"]:
        # 触媒 (UMA等) の発話も含める
        text = (m.get("message") or "").strip()
        if not text: continue
        entries.append({
            "i": i,
            "step": m.get("step", 0),
            "type": "msg",
            "id": m.get("from"),
            "name": name_by_id.get(m.get("from"), "?"),
            "to_name": name_by_id.get(m.get("to"), "?"),
            "text": text,
        })
        i += 1
    for d in data["mr"]:
        text = (d.get("memory") or "").strip()
        reason = (d.get("reasoning") or "").strip()
        body = text
        if reason and "Skipped" not in reason and "should_speak" not in reason:
            body = (text + " | reason: " + reason).strip(" |")
        if not body: continue
        if "Skipped" in body or "should_speak" in body: continue
        entries.append({
            "i": i,
            "step": d.get("step", 0),
            "type": "memory",
            "id": d.get("id"),
            "name": d.get("name") or name_by_id.get(d.get("id"), "?"),
            "text": body,
        })
        i += 1
    print(f"[info] run={run_dir.name}, entries={len(entries)}")

    client = GeminiClient(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.2, max_tokens=1500,
        enable_cache=False, enable_structured_output=False,
    )

    # バッチ分割
    batches = [entries[k:k+args.batch] for k in range(0, len(entries), args.batch)]
    print(f"[info] {len(batches)} batches × ~{args.batch} entries (workers={args.workers})")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    results: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(tag_batch, client, b): bi for bi, b in enumerate(batches)}
        for fut in as_completed(futs):
            results.update(fut.result())
    print(f"[info] tagged {len(results)}/{len(entries)} in {time.time()-t0:.1f}s")

    # 出力
    out_path = run_dir / "tagged_log.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            tags = results.get(e["i"], [])
            row = {
                "step": e["step"],
                "type": e["type"],
                "id": e["id"],
                "name": e["name"],
                "text": e["text"][:300],
                "tags": tags,
            }
            if e["type"] == "msg":
                row["to_name"] = e.get("to_name")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_path}")

    # 簡易集計
    import collections
    tag_count = collections.Counter()
    per_agent = collections.defaultdict(lambda: collections.Counter())
    for e in entries:
        tags = results.get(e["i"], [])
        for t in tags:
            tag_count[t] += 1
            per_agent[e["name"]][t] += 1
    print("\n=== 全体タグ分布 ===")
    for t in ["A","R","E","I","P","B"]:
        print(f"  {t}: {tag_count[t]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
