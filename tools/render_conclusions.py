"""Generate conclusions.html — シミュ終了後、各 agent に「最終的な説明」を書かせ、
全体共通理解を合成して HTML 出力する。

Usage:
    python tools/render_conclusions.py simulations/<run_dir>
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_run(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    personas = cfg["agents"]["personas"]
    question = personas[0].get("current_goal", "")

    # memory_reasoning: per-step memory + reasoning per agent
    mr_path = run_dir / "memory_reasoning.jsonl"
    mr_by_agent: dict[int, list] = {}
    if mr_path.exists():
        for ln in mr_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln)
            aid = d.get("agent_id")
            if aid is None:
                continue
            mr_by_agent.setdefault(aid, []).append(d)

    # messages: from/to per step
    msgs_path = run_dir / "messages.jsonl"
    msgs: list = []
    if msgs_path.exists():
        msgs = [json.loads(l) for l in msgs_path.read_text(encoding="utf-8").splitlines()]
    msgs_by_agent: dict[int, list] = {}
    for m in msgs:
        msgs_by_agent.setdefault(m.get("from", -1), []).append(m)

    return {
        "config": cfg,
        "personas": personas,
        "question": question,
        "mr_by_agent": mr_by_agent,
        "msgs": msgs,
        "msgs_by_agent": msgs_by_agent,
    }


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------
def _build_persona_history_prompt(persona: dict, mr_list: list, sent_msgs: list) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for asking persona to write final explanation."""
    name = persona.get("name", "?")
    age = persona.get("age", "?")
    occ = persona.get("occupation", "?")
    speech = persona.get("speech_style", "")
    bg = persona.get("background", "")

    system = (
        f"あなたは『{name}』({age}歳・{occ})です。\n"
        f"話し方: {speech}\n"
        f"あなたの背景: {bg}\n\n"
        "あなたはこれまで教室で『目の前の知覚も言語化もできない何か』について考え、"
        "クラスメイトと議論してきました。今、以下の2点を**端的に**書いてください。\n\n"
        "出力フォーマット (この見出しを必ず使う、それ以外の前置きや締めは不要):\n"
        "  ## 当初の見立て\n"
        "  最初に『何か』に気づいた時、自分が何だと思ったかを 1〜2文 で。\n\n"
        "  ## 議論を経た最終結論\n"
        "  クラスメイトとのやり取りを経て、最終的に何だと結論づけたかを 1〜2文 で。\n\n"
        "- 各項目とも1〜2文 (合計 80〜140 字程度に収める)\n"
        "- 自分の話し方を保つ\n"
        "- 抽象論で逃げず、自分の言葉で書く\n"
        "- マークダウンの見出し記号 (##) はそのまま使ってよい"
    )

    # 全 step の思考と発言を時系列で出す (初期 vs 後期 の差を LLM に判別させたいので)
    thoughts = []
    for d in mr_list:
        mem = (d.get("memory") or "").strip()
        if mem:
            thoughts.append(f"  [step{d.get('step')}] {mem}")

    sent_quotes = []
    for m in sent_msgs:
        sent_quotes.append(f"  [step{m.get('step')}→{m.get('to_name')}] {m.get('message')}")

    user = (
        "==== あなたが残してきた思考の記録 (全step、時系列) ====\n"
        + ("\n".join(thoughts) if thoughts else "  (記録なし)")
        + "\n\n==== あなたが他の生徒に話したこと (時系列) ====\n"
        + ("\n".join(sent_quotes) if sent_quotes else "  (発言なし)")
        + "\n\n以上の **時系列の記録** を読んで、初期と最終の落差が見えるよう、"
        "上記フォーマットで「当初の見立て」と「議論を経た最終結論」を端的に書いてください。"
    )
    return system, user


def _split_initial_final(text: str) -> tuple[str, str]:
    """LLM が ## 当初の見立て / ## 議論を経た最終結論 で出した文字列を 2 つに分割。"""
    if not text:
        return "", ""
    lines = text.split("\n")
    cur = "intro"
    bucket = {"initial": [], "final": []}
    for ln in lines:
        s = ln.strip()
        if s.startswith("##"):
            tag = s.lstrip("#").strip()
            if "当初" in tag or "最初" in tag or "Initial" in tag:
                cur = "initial"
                continue
            if "最終" in tag or "結論" in tag or "Final" in tag:
                cur = "final"
                continue
        if cur in bucket:
            bucket[cur].append(ln)
    return ("\n".join(bucket["initial"]).strip(),
            "\n".join(bucket["final"]).strip())


def _build_synthesis_prompt(question: str, conclusions: list[dict]) -> tuple[str, str]:
    """Build prompt to synthesize collective conclusion from individual ones."""
    system = (
        "あなたは観察者です。教室の生徒たちがそれぞれ、『目の前の知覚も言語化もできない何か』に"
        "ついて、当初の見立てと議論を経た最終結論を書き残しました。\n"
        "その記録を読んで、教室全体としてどう変化したかを、観察者として端的に要約してください。\n\n"
        "出力フォーマット (この順で、見出し付き、各項目2-4文):\n"
        "  ### 当初の集合的見立て\n"
        "  生徒たちが最初に『何か』を捉えようとしたとき、共通して何と思っていたか。\n\n"
        "  ### 議論を経た最終的到達点\n"
        "  クラス全体で議論を重ねた結果、どんな共通理解にたどり着いたか。\n\n"
        "  ### 分岐点\n"
        "  生徒間で意見が分かれた／表現が割れたポイントを端的に。\n\n"
        "  ### 教室の集合知としての最良の言い当て\n"
        "  1文で、最も核心に近いと思える表現を挙げる。\n\n"
        "マークダウンの見出し記号 (###) は出力に含めて構わない。それ以外の装飾は不要。"
    )
    blocks = []
    for c in conclusions:
        p = c["persona"]
        blocks.append(
            f"--- {p.get('name')}({p.get('age')}・{p.get('occupation')}) ---\n"
            f"[当初] {c.get('initial', '?')}\n"
            f"[最終] {c.get('final', '?')}"
        )
    user = (
        f"問い: {question}\n\n"
        + "\n\n".join(blocks)
        + "\n\n以上の生徒たちの当初の見立てと最終結論を踏まえて、上記フォーマットで要約してください。"
    )
    return system, user


def _call_gemini(client: GeminiClient, system: str, user: str, max_tokens: int = 600) -> str:
    """Call Gemini and return plain text (strip JSON wrapping if structured output kicks in)."""
    resp = client.generate(system, user, temperature=0.7, max_tokens=max_tokens)
    if not resp:
        return ""
    # If structured output returned JSON, try to extract a sensible string field
    s = resp.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            j = json.loads(s)
            for key in ("text", "message", "memory", "reasoning", "answer"):
                if key in j and isinstance(j[key], str):
                    return j[key].strip()
            return json.dumps(j, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return s


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>知覚不可能な何か - 結論 ({run_name})</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
         max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.7; color: #222; }}
  h1 {{ font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2 {{ font-size: 1.2em; color: #444; margin-top: 2em; border-left: 4px solid #888; padding-left: 0.6em; }}
  .question {{ background: #f4f1ea; padding: 1em 1.4em; border-radius: 8px;
               font-style: italic; color: #444; }}
  .synthesis {{ background: #fff7e6; padding: 1.2em 1.6em; border-radius: 10px;
                border: 1px solid #ddc88c; white-space: pre-wrap; }}
  .agent-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1em 1.4em;
                 margin: 1em 0; background: #fafafa; }}
  .agent-name {{ font-weight: bold; font-size: 1.05em; color: #333; }}
  .agent-meta {{ color: #888; font-size: 0.9em; margin-bottom: 0.8em; }}
  .phase-label {{ display: inline-block; font-size: 0.75em; padding: 0.15em 0.5em;
                  border-radius: 3px; margin-top: 0.6em; margin-bottom: 0.2em;
                  font-weight: bold; letter-spacing: 0.05em; }}
  .phase-initial {{ background: #e8f0f8; color: #4a6b8a; }}
  .phase-final   {{ background: #f8edd8; color: #8a6a3a; }}
  .phase-text {{ margin: 0 0 0.4em 0; padding: 0.2em 0.4em; }}
  .quotes {{ margin-top: 0.8em; padding-top: 0.6em; border-top: 1px dashed #ccc;
              font-size: 0.85em; color: #666; }}
  .quote {{ margin: 0.3em 0; padding-left: 0.8em; border-left: 2px solid #ccc; }}
  .meta {{ color: #aaa; font-size: 0.8em; margin-top: 3em; text-align: center; }}
</style>
</head>
<body>
<h1>知覚不可能な何か — 結論</h1>
<p class="question">問い: {question_html}</p>

<h2>教室全体としての結論</h2>
<div class="synthesis">{synthesis_html}</div>

<h2>各生徒の最終的な説明</h2>
{cards_html}

<p class="meta">{run_name} · agents={n_agents} · steps={n_steps} · messages={n_msgs}</p>
</body>
</html>
"""

CARD_TEMPLATE = """<div class="agent-card">
  <div class="agent-name">{name}</div>
  <div class="agent-meta">{age}歳 · {occupation}</div>
  <div class="phase-label phase-initial">当初の見立て</div>
  <div class="phase-text">{initial_html}</div>
  <div class="phase-label phase-final">議論を経た最終結論</div>
  <div class="phase-text">{final_html}</div>
  {quotes_html}
</div>
"""


def _quotes_block(sent_msgs: list, max_n: int = 2) -> str:
    if not sent_msgs:
        return ""
    picked = sent_msgs[-max_n:]
    qs = "\n".join(
        f'<div class="quote">step{html.escape(str(m.get("step")))} → {html.escape(str(m.get("to_name","?")))}: {html.escape(m.get("message",""))}</div>'
        for m in picked
    )
    return f'<div class="quotes"><b>シミュ中の発言（抜粋）</b>\n{qs}</div>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--no-synthesis", action="store_true",
                    help="個別結論のみ生成、集合的synthesisをスキップ")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[err] run_dir not found: {run_dir}", file=sys.stderr)
        return 1

    data = _load_run(run_dir)
    personas = data["personas"]
    cfg = data["config"]

    # Gemini client (plain text mode)
    llm_cfg = cfg.get("llm", {})
    client = GeminiClient(
        base_url=llm_cfg.get("base_url"),
        model=llm_cfg.get("model", "gemini-3.1-flash-lite-preview"),
        temperature=0.7,
        max_tokens=600,
        enable_cache=False,  # 各 agent 用のpromptは個別なのでcache効果ほぼなし
        enable_structured_output=False,
    )

    print(f"[info] generating per-agent conclusions for {len(personas)} agents...")
    results = [None] * len(personas)

    def _gen(i: int):
        p = personas[i]
        mr_list = data["mr_by_agent"].get(p["id"], [])
        sent = data["msgs_by_agent"].get(p["id"], [])
        sys_p, user_p = _build_persona_history_prompt(p, mr_list, sent)
        text = _call_gemini(client, sys_p, user_p, max_tokens=500)
        return i, {"persona": p, "conclusion": text or "(LLM応答なし)", "sent_msgs": sent}

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_gen, i) for i in range(len(personas))]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out
            print(f"  [ok] agent{i} ({out['persona'].get('name')}) - {len(out['conclusion'])}chars")

    # 各 agent の text を 当初/最終 に分割 (synthesis 前に行う)
    for r in results:
        ini, fin = _split_initial_final(r["conclusion"])
        r["initial"] = ini if ini else "(取り出せず)"
        r["final"] = fin if fin else r["conclusion"]  # 失敗時は raw を fallback

    # Synthesis (initial/final を活用するので、分割後に呼ぶ)
    if args.no_synthesis:
        synthesis = "(synthesis skipped)"
    else:
        print("[info] generating collective synthesis...")
        sys_s, user_s = _build_synthesis_prompt(data["question"], results)
        synthesis = _call_gemini(client, sys_s, user_s, max_tokens=900) or "(synthesis empty)"

    # HTML
    cards_html = "\n".join(
        CARD_TEMPLATE.format(
            name=html.escape(r["persona"].get("name", "?")),
            age=html.escape(str(r["persona"].get("age", "?"))),
            occupation=html.escape(r["persona"].get("occupation", "?")),
            initial_html=html.escape(r["initial"]).replace("\n", "<br>"),
            final_html=html.escape(r["final"]).replace("\n", "<br>"),
            quotes_html=_quotes_block(r["sent_msgs"]),
        )
        for r in results
    )

    out_html = HTML_TEMPLATE.format(
        run_name=html.escape(run_dir.name),
        question_html=html.escape(data["question"]),
        synthesis_html=html.escape(synthesis).replace("\n", "<br>"),
        cards_html=cards_html,
        n_agents=len(personas),
        n_steps=cfg.get("simulation", {}).get("duration", "?"),
        n_msgs=len(data["msgs"]),
    )

    out_path = run_dir / "conclusions.html"
    out_path.write_text(out_html, encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
