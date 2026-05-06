"""Generate phaseB_report.html — 品川FW (Phase B) シミュ単独条件用レポート。

各参加者について Gemini に以下を分析させる:
  - 座学(Phase A)で描いた未来像 (handoff から取得)
  - FW中の特徴的な気づき・観察 (1-3個)
  - FW後の未来像更新 / 持ち帰り
  - 一行サマリ

Usage:
  python tools/render_phaseB_report.py <fw_run_dir> --classroom-run <classroom_run_dir> [--out <html>]
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402


def load_run(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    personas = cfg["agents"]["personas"]
    mr_path = run_dir / "memory_reasoning.jsonl"
    msgs_path = run_dir / "messages.jsonl"
    sd_path = run_dir / "simulation_data.json"
    mr = [json.loads(l) for l in mr_path.read_text(encoding="utf-8").splitlines()] if mr_path.exists() else []
    msgs = [json.loads(l) for l in msgs_path.read_text(encoding="utf-8").splitlines()] if msgs_path.exists() else []
    timeline = []
    if sd_path.exists():
        sd = json.loads(sd_path.read_text(encoding="utf-8"))
        timeline = sd.get("timeline") or []
    return {"cfg": cfg, "personas": personas, "mr": mr, "msgs": msgs, "timeline": timeline, "run_dir": run_dir}


def build_full_history_rows(data: dict, agent_id: int, id_to_name: dict | None = None) -> list[dict]:
    """各 step のフルログを返す。timeline (simulation_data.json) を主、msgs を join。
    companions = 同 step に同じ current_place 内にいる他の agent 名 list."""
    by_step_msg_sent = collections.defaultdict(list)
    by_step_msg_recv = collections.defaultdict(list)
    for m in data["msgs"]:
        if m.get("from") == agent_id:
            by_step_msg_sent[m.get("step")].append(m)
        if m.get("to") == agent_id:
            by_step_msg_recv[m.get("step")].append(m)
    rows = []
    prev_xy = None
    for f in data.get("timeline") or []:
        ag = next((a for a in (f.get("agents") or []) if a.get("id") == agent_id), None)
        if not ag:
            continue
        step = f.get("step")
        time_label = f.get("time", "")
        x, y = ag.get("x"), ag.get("y")
        place = ag.get("current_place") or ""
        layer = ag.get("layer") or ""
        moved = (prev_xy is not None and (x, y) != prev_xy)
        prev_xy = (x, y)
        # 同行者: 同じ current_place かつ近接 (5cell以内) の他 agent
        companions = []
        for other in (f.get("agents") or []):
            if other.get("id") == agent_id:
                continue
            ox, oy = other.get("x"), other.get("y")
            same_place = bool(place) and other.get("current_place") == place
            try:
                close = (ox is not None and oy is not None and abs(ox - x) <= 5 and abs(oy - y) <= 5)
            except Exception:
                close = False
            if same_place or close:
                nm = (id_to_name or {}).get(other.get("id"), f"#{other.get('id')}")
                companions.append(nm)
        rows.append({
            "step": step,
            "time": time_label,
            "x": x, "y": y, "place": place, "layer": layer,
            "moved": moved,
            "companions": companions,
            "memory": (ag.get("memory") or "").strip(),
            "reasoning": (ag.get("reasoning") or "").strip(),
            "sent": by_step_msg_sent.get(step, []),
            "recv": by_step_msg_recv.get(step, []),
        })
    return rows


def load_handoffs(classroom_run_dir: Path) -> dict:
    """fw_handoff.jsonl と field_questions.jsonl を axis_id 別に読む。"""
    out = {"handoff_by_axis": {}, "fq_by_axis": {}}
    h_path = classroom_run_dir / "fw_handoff.jsonl"
    if h_path.exists():
        for line in h_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            ax = d.get("axis_id") or d.get("agent_axis_id")
            if ax:
                out["handoff_by_axis"][ax] = d
    fq_path = classroom_run_dir / "field_questions.jsonl"
    if fq_path.exists():
        for line in fq_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            ax = d.get("axis_id") or d.get("agent_axis_id")
            if ax:
                out["fq_by_axis"][ax] = d
    return out


def per_agent_history(data: dict, agent_id: int):
    thoughts = []
    for d in data["mr"]:
        if d.get("id") == agent_id:
            mem = (d.get("memory") or "").strip()
            reason = (d.get("reasoning") or "").strip()
            if mem or reason:
                thoughts.append(f"[step{d.get('step')}] mem: {mem} | reason: {reason[:200]}")
    sent = [m for m in data["msgs"] if m.get("from") == agent_id]
    recv = [m for m in data["msgs"] if m.get("to") == agent_id]
    return thoughts, sent, recv


SYSTEM_ANALYSIS = (
    "あなたは観察者。フィールドワーク (FW) 記録を読んで、参加者 1 人の変化を分析する。"
    "参加者は午前に座学で品川の未来像を議論し、午後 13 時から品川駅前を歩いて、"
    "見た光景・出会った人・気づきから未来像をアップデートする。\n\n"
    "**応答ルール (絶対守る):**\n"
    "1. **すべて日本語**で書く。英語禁止 (固有名詞除く)。\n"
    "2. **「ご要望に従い」「以下の通り」のような前置き禁止**。\n"
    "3. 出力の **1文字目は必ず `### 座学で描いた未来像 (FW開始前)` の '#'** から始める。\n"
    "4. 4 つの見出しを順に書き、それぞれの本文は記録から実際に観察できた事実のみ。\n\n"
    "出力テンプレ (このまま開始する):\n\n"
    "### 座学で描いた未来像 (FW開始前)\n"
    "（handoff からの要約 2-3 文）\n\n"
    "### 現地で得た気づき (FW中)\n"
    "- (気づき 1: 具体的に)\n"
    "- (気づき 2)\n"
    "- (気づき 3)\n\n"
    "### 未来像のアップデート (FW後)\n"
    "（FW を通じてどう変わったか 2-4 文）\n\n"
    "### 一行で言うと\n"
    "（座学→FW での変化を 1 文で）"
)


SYSTEM_ANALYSIS_HOST = (
    "あなたは観察者。FW (フィールドワーク) 受け入れ担当 = 企業 host 1人の活動記録を読んで、"
    "その日の動きと気づきを整理する。host は自社拠点に立って学生の訪問を受け、"
    "学生に自社・まちの話を語りつつ、若い人の視点・違和感を引き出す立場。\n\n"
    "**応答ルール (絶対守る):**\n"
    "1. **すべて日本語**で書く。英語禁止 (固有名詞除く)。\n"
    "2. 前置き禁止。出力の **1文字目は `### 現地で得た気づき (FW中)` の '#'** から始める。\n"
    "3. 2 つの見出しを順に書き、本文は記録から実際に観察できた事実のみ。盛らない。\n"
    "4. 学生がほとんど来なかった / 一方的に語っただけ / 質問が浅かった等の事実があれば包み隠さず書く。\n\n"
    "出力テンプレ (このまま開始する):\n\n"
    "### 現地で得た気づき (FW中)\n"
    "- (気づき 1: 具体的に — 誰とどう話して何が見えたか)\n"
    "- (気づき 2)\n"
    "- (気づき 3)\n\n"
    "### 一行で言うと\n"
    "（今日のFWで見えたこと・気づいたこと、もしくは見えなかったこと を 1 文で）"
)


def build_phaseB_host_prompt(h: dict, sent: list[dict], recv: list[dict], thoughts: list[str]) -> str:
    bg = (h.get("background") or "").strip()[:300]
    name = h.get("name", "?")
    occ = h.get("occupation", "?")
    place = h.get("initial_place", "?")
    sent_block = "\n".join(f"[step{m.get('step')}→{m.get('to_name')}] {(m.get('message') or '')[:200]}" for m in sent[:30]) or "(発話なし)"
    recv_block = "\n".join(f"[step{m.get('step')} from {m.get('from_name')}] {(m.get('message') or '')[:200]}" for m in recv[:30]) or "(受信なし)"
    th_block = "\n".join(thoughts[:20]) or "(思考記録なし)"
    return (
        f"## 受け入れ担当 persona\n"
        f"- 名前: {name}\n"
        f"- 所属: {occ}\n"
        f"- 配置拠点: {place}\n"
        f"- 背景: {bg}\n\n"
        f"## FW中の自分の発話 (時系列)\n{sent_block}\n\n"
        f"## FW中の受信 (時系列)\n{recv_block}\n\n"
        f"## FW中の自分の memory + reasoning (時系列、抜粋)\n{th_block}\n\n"
        f"## 課題\n上のシステム指示のテンプレ通りに書いてください。"
    )


def build_phaseB_prompt(persona: dict, handoff: dict | None, fq: dict | None,
                         thoughts: list[str], sent: list[dict], recv: list[dict]) -> str:
    sent_block = "\n".join(f"[step{m.get('step')}→{m.get('to_name')}] {m.get('message')}" for m in sent[:30])
    recv_block = "\n".join(f"[step{m.get('step')} from {m.get('from_name')}] {m.get('message')}" for m in recv[:30])
    thoughts_block = "\n".join(thoughts[:40])
    handoff_block = ""
    if handoff:
        fi = handoff.get("future_image", "").strip()
        intent = handoff.get("intent", "").strip()
        kms = handoff.get("key_memories") or []
        handoff_block = (
            f"## 座学(Phase A)からの引き継ぎ\n"
            f"- 描いた未来像: {fi}\n"
            f"- これからどう過ごしたいか: {intent}\n"
            f"- 気になっていたこと: " + " / ".join(kms[:3])
        )
    fq_block = ""
    if fq:
        qs = fq.get("questions") or []
        ol = (fq.get("one_liner") or "").strip()
        fq_block = (
            "## 座学で立てた現地で確かめたいこと\n"
            + "\n".join(f"- {q}" for q in qs[:3])
            + (f"\n- (テーマ) {ol}" if ol else "")
        )
    temp = (
        f"外向性={persona.get('temperament_extroversion','?')} / "
        f"楽天性={persona.get('temperament_optimism','?')} / "
        f"好奇心={persona.get('temperament_curiosity','?')}"
    )
    return (
        f"## 参加者: {persona.get('name')} ({persona.get('axis_id')}: 気質={temp})\n\n"
        f"{handoff_block}\n\n{fq_block}\n\n"
        f"## FW中の思考記録 (時系列)\n{thoughts_block or '(なし)'}\n\n"
        f"## FW中の発言 (時系列)\n{sent_block or '(なし)'}\n\n"
        f"## FW中に受信した発言 (時系列)\n{recv_block or '(なし)'}\n\n"
        "以上を踏まえ、上記フォーマット通りに**日本語で**分析する。前置き・締めの挨拶は不要。"
        "見出しも本文もすべて日本語で書く。英語は使わない。"
    )


_LEAK_PATTERNS = [
    r"^(以下|下記|それでは|では).*?(分析|報告)します[。:.]?\s*",
    r"研究員として[、,]?",
    r"^(承知|了解)しました[。:.]?\s*",
]


def _strip_prompt_leak(text: str) -> str:
    if not text:
        return text
    s = text
    for pat in _LEAK_PATTERNS:
        s = re.sub(pat, "", s, flags=re.MULTILINE | re.DOTALL)
    return s.strip()


def call_gemini_md(client: GeminiClient, system: str, user: str, max_tokens: int = 1500) -> str:
    """Markdown テキスト (### 見出し付き) を返す。失敗時は空文字。"""
    resp = client.generate(system, user, temperature=0.4, max_tokens=max_tokens)
    if not resp:
        return ""
    return _strip_prompt_leak(resp.strip())


def _md_inline(s: str) -> str:
    """インライン markdown (太字/斜体/コード) → HTML。html.escape を内側で使う。"""
    s = html.escape(s)
    # **bold**
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    # *italic* (太字残骸を避けるため、隣接 * を除外)
    s = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', s)
    # `code`
    s = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', s)
    return s


def render_md_to_html(md: str) -> str:
    """軽量 Markdown → HTML 変換 (見出し # / ## / ### / #### / リスト '- ' / 段落 / **太字**)。"""
    if not md or not md.strip():
        return "<p class='empty'>(分析失敗)</p>"
    lines = md.split("\n")
    out = []
    in_p = False
    in_ul = False
    def close_p():
        nonlocal in_p
        if in_p: out.append("</p>"); in_p = False
    def close_ul():
        nonlocal in_ul
        if in_ul: out.append("</ul>"); in_ul = False
    for ln in lines:
        s = ln.rstrip()
        if s.startswith("#### "):
            close_p(); close_ul()
            out.append(f"<h5>{_md_inline(s[5:].strip())}</h5>")
        elif s.startswith("### "):
            close_p(); close_ul()
            out.append(f"<h4>{_md_inline(s[4:].strip())}</h4>")
        elif s.startswith("## "):
            close_p(); close_ul()
            out.append(f"<h3>{_md_inline(s[3:].strip())}</h3>")
        elif s.startswith("# "):
            close_p(); close_ul()
            out.append(f"<h2>{_md_inline(s[2:].strip())}</h2>")
        elif s.startswith("- ") or s.startswith("* "):
            close_p()
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{_md_inline(s[2:].strip())}</li>")
        elif not s.strip():
            close_p(); close_ul()
        else:
            close_ul()
            if not in_p: out.append("<p>"); in_p = True
            else: out.append("<br>")
            out.append(_md_inline(s))
    close_p(); close_ul()
    return "\n".join(out)


HTML_TPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="color-scheme" content="dark">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; }}
body {{ font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
        max-width: 1200px; margin: 0 auto; padding: 1.5em 1em 4em; line-height: 1.7;
        color: #e8e8e8; background: #15171a; }}
h1 {{ font-size: 1.7em; border-bottom: 2px solid #4a8ad8; padding-bottom: 0.3em;
      color: #f5f5ef; margin-top: 0; }}
h1 .subtitle {{ font-size: 0.55em; color: #888; font-weight: normal; }}
h2 {{ font-size: 1.2em; color: #cfd0c8; margin-top: 2em; border-left: 4px solid #4a8ad8;
      padding-left: 0.6em; }}
h3 {{ font-size: 1.05em; margin-top: 1.6em; color: #c8dbef; }}
h4 {{ font-size: 0.98em; margin-top: 1em; color: #ffd85f; border-bottom: 1px dotted #444;
      padding-bottom: 0.2em; }}
.exp-meta {{ background: #1d2025; padding: 0.8em 1.2em; border-radius: 8px;
             font-size: 0.95em; border: 1px solid #2c2c2c; }}
.exp-meta li {{ margin: 0.25em 0; }}
.intro-question {{ background: #28230f; padding: 1em 1.4em; border-radius: 8px;
                    border: 1px solid #5a4f1c; margin: 1em 0; font-size: 1.05em;
                    color: #f5e9b8; }}
.personas-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                   gap: 0.7em; }}
.persona-card {{ border: 1px solid #2c2c2c; border-radius: 6px; padding: 0.6em 0.9em;
                 background: #1d2025; font-size: 0.85em; line-height: 1.5; }}
.persona-card.male   {{ border-left: 4px solid #4ea8ff; }}
.persona-card.female {{ border-left: 4px solid #e85a9b; }}
.persona-card .name {{ font-weight: 600; color: #f5f5ef; }}
.persona-card .meta {{ color: #888; font-size: 0.85em; }}
.analysis-card {{ border: 1px solid #2c2c2c; border-radius: 8px; padding: 1.2em 1.5em;
                  background: #1d2025; margin-bottom: 1.4em; }}
.analysis-card .agent-name {{ font-weight: 600; color: #ffd85f; font-size: 1.1em;
                              margin-bottom: 0.3em; }}
.analysis-card .agent-meta {{ color: #888; font-size: 0.85em; margin-bottom: 0.8em; }}
.analysis-card p {{ margin: 0.4em 0; color: #d8d8d8; }}
.analysis-card.male   {{ border-left: 4px solid #4ea8ff; }}
.analysis-card.female {{ border-left: 4px solid #e85a9b; }}
.empty {{ color: #555; font-style: italic; }}
.api-meta {{ background: #1d2025; border: 1px solid #2c2c2c; border-radius: 8px;
             padding: 1em 1.2em; font-size: 0.92em; }}
.api-meta dl {{ margin: 0; display: grid; grid-template-columns: max-content 1fr;
                column-gap: 1em; row-gap: 0.3em; }}
.api-meta dt {{ color: #888; }}
.api-meta dd {{ margin: 0; color: #e8e8e8; }}
.api-table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; }}
.api-table th, .api-table td {{ padding: 0.5em 0.7em; border: 1px solid #2c2c2c; text-align: right; color: #d8d8d8; }}
.api-table th {{ background: #25282d; color: #cfd0c8; text-align: left; }}
.api-table td:first-child, .api-table th:first-child {{ text-align: left; }}
.tlx-wrap {{ background: #1f1f1f; border-radius: 6px; padding: 0.6em; margin: 0.6em 0; }}
.tlx-controls {{ display: flex; align-items: center; gap: 1em; padding: 0.4em 0.6em; color: #ddd; font-size: 0.92em; }}
.tlx-controls input[type=range] {{ flex: 1; }}
.tlx-controls .step-cur {{ color: #ffd85f; font-weight: bold; min-width: 12em; }}
.tlx-canvas {{ position: relative; overflow-x: auto; }}
.tlx-canvas svg {{ display: block; min-width: 1100px; }}
.tlx-bubble-layer {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
.tlx-bubble {{ position: absolute; width: 360px; max-width: 360px; padding: 10px 14px;
              background: #2a2d33; color: #e8e8e8; border-radius: 8px; font-size: 13px; line-height: 1.5;
              box-shadow: 0 6px 18px rgba(0,0,0,0.6); pointer-events: auto;
              border: 1px solid #555; word-wrap: break-word; }}
.tlx-bubble .close {{ position: absolute; top: 4px; right: 8px; cursor: pointer;
                      color: #888; font-size: 14px; user-select: none; }}
.tlx-bubble .close:hover {{ color: #e8e8e8; }}
.tlx-bubble .meta {{ font-size: 10.5px; color: #888; margin-bottom: 4px; padding-right: 16px; }}
.tlx-cursor {{ position: absolute; top: 30px; bottom: 20px; width: 2px; background: #ffd85f; pointer-events: none; opacity: 0.7; }}
.tlx-step-label {{ position: absolute; top: 4px; padding: 2px 6px; background: #ffd85f; color: #1f1f1f;
                   font-size: 10px; font-weight: bold; border-radius: 3px; transform: translateX(-50%); }}
.hl-grid {{ display: grid; gap: 0.6em; }}
.hl-item {{ background: #1d2025; border: 1px solid #2c2c2c; border-left: 4px solid #ffd85f;
            padding: 0.7em 1em; border-radius: 6px; }}
.hl-step {{ font-size: 0.85em; color: #a0a0a0; margin-bottom: 0.3em; }}
.hl-body {{ color: #e8e8e8; }}
details.history {{ margin-top: 0.8em; border: 1px solid #2c2c2c; border-radius: 6px; }}
details.history summary {{ cursor: pointer; padding: 0.5em 0.8em; color: #a0a0a0;
                           background: #25282d; border-radius: 6px 6px 0 0; font-size: 0.88em; }}
details.history[open] summary {{ border-bottom: 1px solid #2c2c2c; }}
details.history .hbox {{ padding: 0.6em 1em; max-height: 360px; overflow-y: auto;
                          background: #15171a; border-radius: 0 0 6px 6px; }}
details.history h5 {{ margin: 0.6em 0 0.3em; color: #c8dbef; font-size: 0.92em; }}
details.history .ev-line {{ font-size: 0.85em; color: #c8c8c8; padding: 0.25em 0;
                             border-bottom: 1px dotted #2c2c2c; }}
table.full-history {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
table.full-history th {{ background: #25282d; color: #cfd0c8; padding: 0.4em 0.5em;
                          border: 1px solid #2c2c2c; text-align: left; position: sticky; top: 0; }}
table.full-history td {{ padding: 0.4em 0.5em; border: 1px solid #2c2c2c; vertical-align: top; color: #d8d8d8; }}
table.full-history .t-step {{ width: 3em; color: #ffd85f; font-family: monospace; }}
table.full-history .t-time {{ width: 4em; color: #888; font-family: monospace; }}
table.full-history .t-loc  {{ width: 13em; color: #b0c4de; font-size: 0.92em; }}
table.full-history .t-comp {{ width: 9em; color: #cdb8e8; font-size: 0.85em; }}
table.full-history .t-thought {{ font-size: 0.9em; line-height: 1.45; }}
table.full-history .t-thought .th-mem {{ color: #d8d8d8; padding: 0.1em 0; }}
table.full-history .t-thought .th-reason {{ color: #a0a0a0; margin-top: 0.3em; padding: 0.1em 0; }}
table.full-history .t-thought .th-label {{ display: inline-block; font-size: 0.75em; color: #ffd85f; margin-right: 0.4em; padding: 0.05em 0.4em; background: #2a2d33; border-radius: 3px; }}
table.full-history .t-thought .th-reason .th-label {{ color: #66ccee; }}
table.full-history .t-sent .ev-out {{ color: #ff9f43; padding: 0.15em 0; }}
table.full-history .t-recv .ev-in {{ color: #66ccee; padding: 0.15em 0; }}
table.full-history .empty {{ color: #444; }}
/* tlx-bubble: クリックで前面に */
.tlx-bubble {{ z-index: 1; }}
.tlx-bubble.front {{ z-index: 100; }}
</style></head><body>

<h1>品川 フィールドワーク シミュ レポート <span class="subtitle">(Phase B / {variant_label})</span></h1>

<h2>実験設定</h2>
<div class="exp-meta">
<ul>
<li>シナリオ: 品川駅前FW (Phase B、座学を経た20人の午後5時間)</li>
<li>条件: 単独 (触媒・特殊介入なし、純粋に現地観察)</li>
<li>{n_agents}人 × {duration}step ({duration_min}分 / {start_time}スタート → {end_time})</li>
<li>variant: {variant_label} / seed: {seed}</li>
<li>LLM: {llm_model}</li>
<li>座学run: <code>{classroom_label}</code></li>
</ul>
</div>

<div class="intro-question">
<div class="label" style="font-size:0.8em;color:#a89e80">中心問い + FW前提 (config の current_goal そのまま)</div>
{goal_html}
{goal_note_html}
</div>

<h2>参加者一覧</h2>
<div class="personas-grid">
{personas_html}
</div>

<h2>全体活動タイムライン — スライダー / ◀▶ボタン / 矢印キーで step 移動</h2>
<div class="note" style="font-size:0.85em;color:#888;margin-bottom:0.5em">
<span style="color:#888">●</span> 思考のみ &nbsp;
<span style="color:#ff9f43">●</span> 児童間の会話 &nbsp;
スライダーを動かすと、その step に発話した人の点から吹き出しで発言が出る
</div>
{interactive_timeline_html}

<h2>特徴的な出来事タイムライン</h2>
{highlights_html}

<h2>各参加者の言語化分析 (座学→FWの変化)</h2>
{analysis_html}

<h2>API消費・所要時間</h2>
{api_table_html}
<div class="api-meta" style="margin-top:0.8em">
<dl>
<dt>分析対象 agent 数</dt><dd>{n_analyzed}</dd>
<dt>FW シミュ run</dt><dd><code>{run_label}</code></dd>
<dt>レポート生成日時</dt><dd>{generated_at}</dd>
</dl>
</div>

</body></html>
"""


def fmt_duration(sec: int) -> str:
    if sec <= 0:
        return "-"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}時間{m}分{s}秒"
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


# Sim log parse + cost estimation (ab_comparison.py からそのまま)
TOKEN_PAT = re.compile(r"Token usage \(gemini(?:-chat)?\): input=(\d+), cache_read=(\d+), output=(\d+)")
TS_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", re.MULTILINE)
PRICE_INPUT_PER_M = 0.10
PRICE_OUTPUT_PER_M = 0.40
PRICE_CACHE_PER_M = 0.025
USD_TO_JPY = 150


def parse_sim_log(log_path):
    if not log_path or not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    calls = TOKEN_PAT.findall(text)
    sum_in = sum(int(a) for a, _, _ in calls)
    sum_cache = sum(int(b) for _, b, _ in calls)
    sum_out = sum(int(c) for _, _, c in calls)
    timestamps = TS_PAT.findall(text)
    duration_sec = 0
    if len(timestamps) >= 2:
        try:
            t0 = datetime.datetime.fromisoformat(timestamps[0].replace(" ", "T"))
            t1 = datetime.datetime.fromisoformat(timestamps[-1].replace(" ", "T"))
            duration_sec = int((t1 - t0).total_seconds())
        except Exception:
            pass
    return {
        "n_calls": len(calls),
        "input_tokens": sum_in,
        "cache_read_tokens": sum_cache,
        "uncached_input": max(0, sum_in - sum_cache),
        "output_tokens": sum_out,
        "duration_sec": duration_sec,
    }


def estimate_cost_usd(stats):
    if not stats:
        return 0.0
    return (
        stats.get("uncached_input", 0) * PRICE_INPUT_PER_M / 1_000_000
        + stats.get("cache_read_tokens", 0) * PRICE_CACHE_PER_M / 1_000_000
        + stats.get("output_tokens", 0) * PRICE_OUTPUT_PER_M / 1_000_000
    )


def render_api_table(sim_stats, analysis_dur_sec, n_analyzed, sim_label: str = "FW シミュ"):
    rows = []
    sections = [(sim_label, sim_stats), ("分析・レポート生成", {"n_calls": n_analyzed, "duration_sec": analysis_dur_sec})]
    total = {"n_calls": 0, "input_tokens": 0, "cache_read_tokens": 0, "uncached_input": 0,
             "output_tokens": 0, "duration_sec": 0, "cost_usd": 0.0}
    for label, s in sections:
        s = s or {}
        cost = estimate_cost_usd(s)
        total["n_calls"] += s.get("n_calls", 0) or 0
        total["input_tokens"] += s.get("input_tokens", 0) or 0
        total["cache_read_tokens"] += s.get("cache_read_tokens", 0) or 0
        total["uncached_input"] += s.get("uncached_input", 0) or 0
        total["output_tokens"] += s.get("output_tokens", 0) or 0
        total["duration_sec"] += s.get("duration_sec", 0) or 0
        total["cost_usd"] += cost
        cache_rate = (s.get("cache_read_tokens", 0) / s.get("input_tokens", 1) * 100) if s.get("input_tokens") else 0
        rows.append(
            f"<tr><th>{label}</th>"
            f"<td>{s.get('n_calls', 0):,}</td>"
            f"<td>{s.get('input_tokens', 0):,}</td>"
            f"<td>{s.get('output_tokens', 0):,}</td>"
            f"<td>{cache_rate:.1f}%</td>"
            f"<td>{fmt_duration(s.get('duration_sec', 0))}</td>"
            f"<td>${cost:.3f}</td></tr>"
        )
    cache_rate_total = (total["cache_read_tokens"] / total["input_tokens"] * 100) if total["input_tokens"] else 0
    rows.append(
        f"<tr style='border-top:2px solid #888;font-weight:bold'><th>合計</th>"
        f"<td>{total['n_calls']:,}</td>"
        f"<td>{total['input_tokens']:,}</td>"
        f"<td>{total['output_tokens']:,}</td>"
        f"<td>{cache_rate_total:.1f}%</td>"
        f"<td>{fmt_duration(total['duration_sec'])}</td>"
        f"<td>${total['cost_usd']:.3f} (≒¥{total['cost_usd']*USD_TO_JPY:.0f})</td></tr>"
    )
    return (
        f'<table class="api-table">'
        f'<thead><tr><th></th><th>API呼出</th><th>入力token</th><th>出力token</th>'
        f'<th>cache hit率</th><th>所要時間</th><th>推定コスト</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'<p style="font-size:0.82em;color:#888;margin-top:0.6em">'
        f'モデル: gemini-2.5-flash-lite / 料金 input ${PRICE_INPUT_PER_M}, '
        f'output ${PRICE_OUTPUT_PER_M}, cache_read ${PRICE_CACHE_PER_M} per 1M tokens / ¥/USD≒{USD_TO_JPY}'
        f'</p>'
    )


# 「特徴的なシーン」抽出: messages.jsonl から insight marker や受信反応を見て上位 K 個を Gemini に選ばせる
INSIGHT_MARKERS = [
    "気づい", "そうか", "なるほど", "実は", "あー", "やっと", "本当は", "もしかして",
    "今わかった", "つまり", "結局", "わかった", "気がする", "腑に落ち",
    "見えてきた", "つかめ", "おもしろ", "確信", "発見", "そうなんだ",
    "一緒に", "行こう", "見に行", "歩いて", "驚い", "感じ",
]


def extract_highlight_candidates(data, max_pre=40):
    """messages.jsonl から insight marker でフィルタした候補を返す。"""
    cand = []
    for m in data["msgs"]:
        msg = (m.get("message") or "").strip()
        if any(k in msg for k in INSIGHT_MARKERS):
            cand.append({
                "step": m.get("step"),
                "from_name": m.get("from_name"),
                "to_name": m.get("to_name"),
                "message": msg[:240],
            })
    return cand[:max_pre]


HIGHLIGHT_SYS = (
    "あなたはフィールドワークの観察者。下記の発話候補から、"
    "**FW中の発見・気づきが起きた特徴的な瞬間** を 6〜8 件選び、"
    "JSON 配列で返す。出力は `[` で始め `]` で終わる JSON のみ。前置き禁止。\n"
    '形式: [{"step": 12, "actors": "誰→誰", "summary": "ここで何が起きたか日本語1-2文"}, ...]\n'
    "値はすべて日本語で。"
)


def select_highlights(client, candidates):
    if not candidates:
        return []
    user = (
        "発話候補:\n" +
        "\n".join(
            f'- step{c["step"]} {c["from_name"]}→{c["to_name"]}: 「{c["message"]}」'
            for c in candidates
        )
    )
    resp = client.generate(HIGHLIGHT_SYS, user, temperature=0.5, max_tokens=2000)
    if not resp:
        return []
    s = resp.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


import collections


def _phaseB_timeline_layout(data, participants, total_steps):
    # 生徒を上、企業担当者を下にまとめてソート (axis_id 内では昇順)
    def _sort_key(p):
        is_host = bool(p.get("is_host") or str(p.get("axis_id", "")).startswith("Host_"))
        return (1 if is_host else 0, str(p.get("axis_id", "")))
    ordered = sorted(participants, key=_sort_key)
    row_h = 22; left_pad = 100; right_pad = 20; top_pad = 30; bot_pad = 20
    step_w = max(8, min(20, int(1200 / max(total_steps, 1))))
    width = left_pad + step_w * total_steps + right_pad
    height = top_pad + row_h * len(ordered) + bot_pad
    agent_pos = {}
    for i, p in enumerate(ordered):
        y = top_pad + i * row_h + row_h / 2
        agent_pos[p["id"]] = {"row": i, "name": p.get("name", "?"), "y_px": y, "axis": p.get("axis_id", "")}
    return {
        "ordered": ordered, "agent_pos": agent_pos, "row_h": row_h,
        "left_pad": left_pad, "right_pad": right_pad,
        "top_pad": top_pad, "bot_pad": bot_pad, "step_w": step_w,
        "width": width, "height": height, "total_steps": total_steps,
    }


def build_phaseB_timeline_svg(data, participants, total_steps):
    layout = _phaseB_timeline_layout(data, participants, total_steps)
    msgs = data["msgs"]
    mr = data["mr"]
    talked = collections.defaultdict(list)
    for m in msgs:
        talked[(m.get("from"), m.get("step"))].append(m)
    thought = collections.defaultdict(list)
    for d in mr:
        if (d.get("memory") or "").strip():
            thought[(d.get("id"), d.get("step"))].append(d)

    parts = [f'<svg class="timeline-svg" width="{layout["width"]}" height="{layout["height"]}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<rect x="0" y="0" width="{layout["width"]}" height="{layout["height"]}" fill="#1f1f1f"/>')

    # bands: total_steps を 5 等分して時間帯ラベル
    bands_phaseB = [
        (1,                              max(2, int(total_steps * 0.10)), "集合・出発"),
        (max(3, int(total_steps * 0.10)) + 1, int(total_steps * 0.30),    "序盤の探索 (積極移動)"),
        (int(total_steps * 0.30) + 1,    int(total_steps * 0.55),         "現地観察 (中盤)"),
        (int(total_steps * 0.55) + 1,    int(total_steps * 0.80),         "深い対話・気づき"),
        (int(total_steps * 0.80) + 1,    total_steps,                     "帰路・解散"),
    ]
    band_colors = ["#252a30", "#1f262d", "#252a30", "#1f262d", "#252a30"]
    for (s1, s2, label), c in zip(bands_phaseB, band_colors):
        x = layout["left_pad"] + (s1 - 1) * layout["step_w"]
        w = (s2 - s1 + 1) * layout["step_w"]
        parts.append(f'<rect x="{x}" y="{layout["top_pad"]}" width="{w}" height="{layout["row_h"] * len(layout["ordered"])}" fill="{c}"/>')

    step_label_intv = max(5, total_steps // 10)
    for s in range(0, total_steps + 1, step_label_intv):
        x = layout["left_pad"] + s * layout["step_w"]
        parts.append(f'<line x1="{x}" y1="{layout["top_pad"]}" x2="{x}" y2="{layout["top_pad"] + layout["row_h"] * len(layout["ordered"])}" stroke="#333" stroke-width="0.5"/>')
        parts.append(f'<text x="{x + 2}" y="{layout["top_pad"] + layout["row_h"] * len(layout["ordered"]) + 12}" font-size="10" fill="#aaa" font-family="sans-serif">{s if s>0 else 1}</text>')

    for i, p in enumerate(layout["ordered"]):
        y = layout["top_pad"] + i * layout["row_h"] + layout["row_h"] / 2
        name = p.get("name", "?")
        # 色分け: 企業担当者=緑、生徒男=青、生徒女=ピンク、生徒その他=黄
        is_host = bool(p.get("is_host") or str(p.get("axis_id", "")).startswith("Host_"))
        if is_host:
            label_color = "#3ec96e"
        elif p.get("gender") == "male":
            label_color = "#4ea8ff"
        elif p.get("gender") == "female":
            label_color = "#e85a9b"
        else:
            label_color = "#f2d23f"
        parts.append(f'<text x="{layout["left_pad"] - 6}" y="{y + 4}" font-size="11" fill="{label_color}" font-family="sans-serif" text-anchor="end">{html.escape(name)}</text>')
        parts.append(f'<line x1="{layout["left_pad"]}" y1="{y}" x2="{layout["left_pad"] + total_steps * layout["step_w"]}" y2="{y}" stroke="#2a2a2a" stroke-width="1"/>')
        for step in range(1, total_steps + 1):
            x = layout["left_pad"] + (step - 1) * layout["step_w"] + layout["step_w"] / 2
            has_talk = (p["id"], step) in talked
            has_thought = (p["id"], step) in thought
            if has_talk:
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="4.5" fill="#ff9f43" '
                    f'style="cursor:pointer" onclick="tlxJump_phaseB({step})">'
                    f'<title>step{step} {html.escape(name)} 発話</title></circle>'
                )
            elif has_thought:
                parts.append(f'<circle cx="{x}" cy="{y}" r="1.6" fill="#888" opacity="0.65"/>')
    parts.append('</svg>')
    return "\n".join(parts), layout


def build_phaseB_interactive_timeline(data, participants, total_steps, start_hour=13, mins_per_step=3, phase_id="phaseB", start_minute=0):
    """interactive タイムライン (slider + bubble)。phase_id で複数 phase を同 page で並置可。
    start_minute: スタート時刻の分数 (例: 09:00 なら 0、09:30 なら 30)
    """
    svg, layout = build_phaseB_timeline_svg(data, participants, total_steps)
    # SVG 内の onclick=tlxJump_phaseB(...) を phase_id に置き換える
    if phase_id != "phaseB":
        svg = svg.replace("tlxJump_phaseB(", f"tlxJump_{phase_id}(")
    msgs = data["msgs"]
    msgs_for_js = []
    for m in msgs:
        from_id = m.get("from")
        if from_id is None:
            continue
        ap = layout["agent_pos"].get(from_id)
        if not ap:
            continue
        x = layout["left_pad"] + (m.get("step", 0) - 1) * layout["step_w"] + layout["step_w"] / 2
        msgs_for_js.append({
            "step": m.get("step", 0),
            "from_name": m.get("from_name", "?"),
            "to_name": m.get("to_name", ""),
            "msg": m.get("message", ""),
            "x": x,
            "y": ap["y_px"],
        })
    msgs_json = json.dumps(msgs_for_js, ensure_ascii=False)
    pid = phase_id
    sm_str = f"{start_minute:02d}"
    return f'''
<div class="tlx-wrap" id="tlx-{pid}">
  <div class="tlx-controls">
    <button onclick="tlxStep_{pid}(-1)" style="padding:4px 10px;border-radius:4px;border:1px solid #555;background:#333;color:#ddd;cursor:pointer">◀</button>
    <input type="range" min="1" max="{total_steps}" value="1" id="tlx-slider-{pid}" step="1">
    <button onclick="tlxStep_{pid}(1)" style="padding:4px 10px;border-radius:4px;border:1px solid #555;background:#333;color:#ddd;cursor:pointer">▶</button>
    <span class="step-cur" id="tlx-cur-{pid}">step 1 / {total_steps} ({start_hour:02d}:{sm_str})</span>
  </div>
  <div class="tlx-canvas" id="tlx-canvas-{pid}">
    {svg}
    <div class="tlx-bubble-layer" id="tlx-bubbles-{pid}"></div>
    <div class="tlx-cursor" id="tlx-cursor-{pid}" style="left:{layout["left_pad"] + layout["step_w"] / 2}px"></div>
    <div class="tlx-step-label" id="tlx-steplabel-{pid}" style="left:{layout["left_pad"] + layout["step_w"] / 2}px">step 1</div>
  </div>
</div>
<script>
(function() {{
  const messages = {msgs_json};
  const stepW = {layout["step_w"]};
  const leftPad = {layout["left_pad"]};
  const totalSteps = {total_steps};
  const startHour = {start_hour};
  const startMinute = {start_minute};
  const minsPerStep = {mins_per_step};
  const sliderEl = document.getElementById('tlx-slider-{pid}');
  const curEl = document.getElementById('tlx-cur-{pid}');
  const cursorEl = document.getElementById('tlx-cursor-{pid}');
  const labelEl = document.getElementById('tlx-steplabel-{pid}');
  const bubbleLayer = document.getElementById('tlx-bubbles-{pid}');
  function clockOf(step) {{
    const total = (step - 1) * minsPerStep + startMinute;
    const h = startHour + Math.floor(total / 60);
    return String(h).padStart(2,'0') + ':' + String(total % 60).padStart(2,'0');
  }}
  function update() {{
    const step = parseInt(sliderEl.value, 10);
    const x = leftPad + (step - 1) * stepW + stepW / 2;
    cursorEl.style.left = x + 'px';
    labelEl.style.left = x + 'px';
    labelEl.textContent = 'step ' + step;
    curEl.textContent = 'step ' + step + ' / ' + totalSteps + ' (' + clockOf(step) + ')';
    bubbleLayer.innerHTML = '';
    const here = messages.filter(m => m.step === step);
    const bubbleW = 360, bubbleH = 90, gapY = 8;
    here.forEach((m, idx) => {{
      const b = document.createElement('div');
      b.className = 'tlx-bubble';
      const canvasW = document.getElementById('tlx-canvas-{pid}').scrollWidth;
      let left = m.x - bubbleW / 2;
      if (left < 4) left = 4;
      if (left + bubbleW > canvasW - 4) left = canvasW - bubbleW - 4;
      const top = Math.max(2, m.y - bubbleH - 12 - idx * (bubbleH + gapY));
      b.style.left = left + 'px';
      b.style.top = top + 'px';
      b.onclick = (e) => {{
        if (e.target === close) return;
        document.querySelectorAll('#tlx-bubbles-{pid} .tlx-bubble').forEach(el => el.classList.remove('front'));
        b.classList.add('front');
      }};
      const close = document.createElement('span');
      close.className = 'close';
      close.textContent = '×';
      close.onclick = (e) => {{ e.stopPropagation(); b.remove(); }};
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = 'step ' + m.step + ' / ' + m.from_name + ' → ' + (m.to_name || '?');
      const text = document.createElement('div');
      text.textContent = '「' + m.msg + '」';
      b.appendChild(close); b.appendChild(meta); b.appendChild(text);
      bubbleLayer.appendChild(b);
    }});
  }}
  window.tlxStep_{pid} = function(delta) {{
    let v = parseInt(sliderEl.value, 10) + delta;
    if (v < 1) v = 1; if (v > totalSteps) v = totalSteps;
    sliderEl.value = v;
    update();
  }};
  window.tlxJump_{pid} = function(step) {{
    if (step < 1) step = 1; if (step > totalSteps) step = totalSteps;
    sliderEl.value = step;
    update();
  }};
  sliderEl.addEventListener('input', update);
  update();
}})();
</script>
'''


def render_highlights_section(hls):
    if not hls:
        return '<p class="empty">(特徴的な瞬間が抽出できませんでした)</p>'
    items = []
    for h in hls:
        items.append(
            f'<div class="hl-item">'
            f'<div class="hl-step">step {html.escape(str(h.get("step","?")))}'
            f' / {html.escape(str(h.get("actors","")))}</div>'
            f'<div class="hl-body">{html.escape(str(h.get("summary","")))}</div>'
            f'</div>'
        )
    return '<div class="hl-grid">' + "".join(items) + '</div>'


TEMPERAMENT_LABELS = {
    "extroversion": {"high": "社交的", "mid": "—", "low": "内向的"},
    "optimism":     {"high": "楽天的", "mid": "—", "low": "心配性"},
    "curiosity":    {"high": "好奇心旺盛", "mid": "—", "low": "慎重"},
}


def _temp_str(p: dict, compact: bool = False) -> str:
    """temperament 3項目をスラッシュ区切り文字列に。
    compact=True で全項目が — (= mid or 未設定) のとき空文字を返す。"""
    ext = TEMPERAMENT_LABELS["extroversion"].get(p.get("temperament_extroversion", ""), "—")
    opt = TEMPERAMENT_LABELS["optimism"].get(p.get("temperament_optimism", ""), "—")
    cur = TEMPERAMENT_LABELS["curiosity"].get(p.get("temperament_curiosity", ""), "—")
    if compact and ext == "—" and opt == "—" and cur == "—":
        return ""
    return f"{ext} / {opt} / {cur}"


# gender 表記: 内部値 (male/female/other) → 日本語 (男/女/その他)
GENDER_JP = {"male": "男", "female": "女", "other": "その他"}


def _gender_jp(p: dict) -> str:
    g = (p.get("gender") or "").strip().lower()
    return GENDER_JP.get(g, g or "?")


def _agent_display_label(p: dict) -> str:
    """smoke22: agent name + 所属 (host) or 特徴的背景 (学生) を 1 行ラベルで返す。
    レポート全体で「誰がどこの人か / 特徴は何か」が見えるように。"""
    name = str(p.get("name", "?"))
    tags = []
    is_host_p = bool(p.get("is_host")) or str(p.get("axis_id", "")).startswith("Host_")
    if is_host_p:
        place = (p.get("initial_place", "") or "").strip()
        occ = (p.get("occupation", "") or "").strip()
        company = occ.split()[0] if occ else ""
        # 表示優先度: 会社名と拠点が異なる → "会社名 / 拠点"。
        # 同じ場合は重複させず、occupation 全文 (役職含む) を出す方が情報量多い。
        if company and place and company != place:
            tags.append(f"{company} / {place}")
        elif occ:
            tags.append(occ)
        elif place:
            tags.append(place)
    else:
        if (p.get("school_fit") or "") == "不適応":
            tags.append("学校不適応")
        if p.get("nationality") in ("western", "asian"):
            tags.append("外国籍")
        if p.get("gender") == "other":
            tags.append("ジェンダーレス")
        if p.get("mobility") == "wheelchair":
            tags.append("ハンディキャップ")
    if tags:
        return f"{name} ({' / '.join(tags)})"
    return name


def render_persona_card(p: dict) -> str:
    gender_cls = p.get("gender", "")
    is_host = bool(p.get("is_host"))
    bg = (p.get("background", "") or "").strip().split("──")[0]  # 切り出し
    catch = p.get("catchphrase", "") or ""
    bg_short = bg[:200] + ("…" if len(bg) > 200 else "")
    cls = "host" if is_host else gender_cls
    role_label = "企業担当者" if is_host else (p.get("occupation") or "")
    school_fit = p.get("school_fit") or ""
    interest = p.get("interest_tag") or ""
    tags = []
    if school_fit and school_fit != "適応":
        tags.append(f"学校適応: {school_fit}")
    if interest:
        tags.append(f"興味: {interest}")
    tag_html = (" / ".join(html.escape(t) for t in tags)) if tags else ""
    temp = _temp_str(p, compact=True)
    parts = [
        f'<div class="persona-card {cls}">',
        f'<div class="name">{html.escape(_agent_display_label(p))}（{p.get("age", "?")}歳・{html.escape(_gender_jp(p))}）</div>',
        f'<div class="meta">{html.escape(role_label)}</div>' if role_label else "",
        f'<div class="meta">{tag_html}</div>' if tag_html else "",
        f'<div class="meta">気質: {html.escape(temp)}</div>' if temp else "",
        f'<div class="meta" style="margin-top:0.3em;color:#bbb;font-size:0.9em">背景: {html.escape(bg_short)}</div>' if bg_short else "",
        f'<div class="meta" style="color:#bbb;font-size:0.9em">口癖: 「{html.escape(catch[:70])}」</div>' if catch else "",
        '</div>',
    ]
    return "".join(p for p in parts if p)


def render_history_details(rows: list[dict], show_movement: bool = True) -> str:
    """全 step のテーブル: step | 時刻 | (場所/移動 | 同行者) | 💭記憶 / 🧠理由 | 発話 | 受信。
    show_movement=False で「場所/移動」「同行」列を非表示 (Phase A 用)。"""
    if not rows:
        return ""
    body_rows = []
    for r in rows:
        place = html.escape(str(r.get("place", "")))
        moved = "→移動" if r.get("moved") else ""
        loc_cell = f'{place} ({r.get("x", "?")},{r.get("y", "?")}) {moved}'
        comp = r.get("companions") or []
        if comp:
            comp_cell = "<br>".join(html.escape(c) for c in comp[:6])
            if len(comp) > 6:
                comp_cell += f'<br><span class="empty">...+{len(comp)-6}</span>'
        else:
            comp_cell = '<span class="empty">—</span>'
        thought_parts = []
        if r["memory"]:
            thought_parts.append(f'<div class="th-mem"><span class="th-label">💭記憶</span>{html.escape(r["memory"])}</div>')
        if r["reasoning"] and r["reasoning"] != r["memory"]:
            thought_parts.append(f'<div class="th-reason"><span class="th-label">🧠理由</span>{html.escape(r["reasoning"])}</div>')
        thought_cell = "".join(thought_parts) or '<span class="empty">—</span>'
        sent_cell = "".join(
            f'<div class="ev-out">→{html.escape(str(m.get("to_name", "")))}: 「{html.escape(str(m.get("message", "")))}」</div>'
            for m in r["sent"]
        ) or '<span class="empty">—</span>'
        recv_cell = "".join(
            f'<div class="ev-in">{html.escape(str(m.get("from_name", "")))} → 「{html.escape(str(m.get("message", "")))}」</div>'
            for m in r["recv"]
        ) or '<span class="empty">—</span>'
        if show_movement:
            body_rows.append(
                f'<tr>'
                f'<td class="t-step">{r["step"]}</td>'
                f'<td class="t-time">{html.escape(str(r.get("time", "")))}</td>'
                f'<td class="t-loc">{loc_cell}</td>'
                f'<td class="t-comp">{comp_cell}</td>'
                f'<td class="t-thought">{thought_cell}</td>'
                f'<td class="t-sent">{sent_cell}</td>'
                f'<td class="t-recv">{recv_cell}</td>'
                f'</tr>'
            )
        else:
            body_rows.append(
                f'<tr>'
                f'<td class="t-step">{r["step"]}</td>'
                f'<td class="t-time">{html.escape(str(r.get("time", "")))}</td>'
                f'<td class="t-thought">{thought_cell}</td>'
                f'<td class="t-sent">{sent_cell}</td>'
                f'<td class="t-recv">{recv_cell}</td>'
                f'</tr>'
            )
    if show_movement:
        thead = '<thead><tr><th>step</th><th>時刻</th><th>場所/移動</th><th>同行</th><th>思考</th><th>発話</th><th>受信</th></tr></thead>'
        note = ('思考列の <strong>💭記憶</strong> = その step の memory / '
                '<strong>🧠理由</strong> = reasoning。'
                '同行列 = 同じ施設内 or 半径5cell 以内にいた他の agent (上位6名まで)。')
    else:
        thead = '<thead><tr><th>step</th><th>時刻</th><th>思考</th><th>発話</th><th>受信</th></tr></thead>'
        note = ('思考列の <strong>💭記憶</strong> = その step の memory / '
                '<strong>🧠理由</strong> = reasoning。')
    return (
        '<div class="history-block">'
        '<h5 style="margin:0.6em 0 0.3em">全 step ログ (思考 / 発話 / 受信' + ('/ 移動 / 同行' if show_movement else '') + ')</h5>'
        '<div class="hbox">'
        f'<p style="font-size:0.82em;color:#888;margin:0.2em 0 0.6em">{note}</p>'
        '<table class="full-history">'
        f'{thead}'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
        '</div></div>'
    )


def _build_activity_summary(full_rows: list, host_id_set: set | None = None,
                            persona: dict | None = None) -> str:
    """各人の概要まとめを rule-based で生成: (学生のみ) 課された行先 / 移動経路 / 主な会話相手。
    主な気づきは Gemini 分析 (md) 側で書かれるためここでは出さない。"""
    if not full_rows:
        return ""
    host_id_set = host_id_set or set()

    # 訪問場所のシーケンス (連続重複は折りたたみ)
    seq = []
    for r in full_rows:
        place = (r.get("place") or "").strip()
        if not place:
            continue
        if not seq or seq[-1] != place:
            seq.append(place)
    visit_route = " → ".join(seq[:10])
    if len(seq) > 10:
        visit_route += f" → … ({len(seq)-10}か所略)"

    # 主な会話相手 (top 4)。生徒/企業担当者ラベル付与。
    counter: dict[str, dict] = {}
    for r in full_rows:
        for m in (r.get("sent") or []):
            n = (m.get("to_name") or "").strip()
            tid = m.get("to")
            if n:
                d = counter.setdefault(n, {"count": 0, "is_host": tid in host_id_set})
                d["count"] += 1
        for m in (r.get("recv") or []):
            n = (m.get("from_name") or "").strip()
            fid = m.get("from")
            if n:
                d = counter.setdefault(n, {"count": 0, "is_host": fid in host_id_set})
                d["count"] += 1
    top_partners = sorted(counter.items(), key=lambda kv: -kv[1]["count"])[:4]
    partners_html_parts = []
    for n, d in top_partners:
        role = "企業" if d["is_host"] else "生徒"
        color = "#3ec96e" if d["is_host"] else "#7aa9ff"
        partners_html_parts.append(
            f'<span style="margin-right:0.8em">'
            f'<span style="color:{color};font-size:0.8em;border:1px solid {color};border-radius:3px;padding:0 0.3em;margin-right:0.2em">{role}</span>'
            f'{html.escape(n)} ({d["count"]})'
            f'</span>'
        )
    partners_html = "".join(partners_html_parts) if partners_html_parts else "（会話なし）"

    # 課された行先 (学生のみ): 移動経路の前に表示し、実際の動きと対比できるようにする
    assigned_html = ""
    if persona and not persona.get("is_host"):
        assigned = persona.get("assigned_hosts") or []
        if assigned:
            parts = []
            for a in assigned:
                place = (a.get("place") or "").strip()
                hname = (a.get("name") or "").strip()
                if place and hname:
                    parts.append(f"{html.escape(place)}（{html.escape(hname)}）")
                elif place:
                    parts.append(html.escape(place))
            if parts:
                assigned_html = (
                    '<div style="margin-bottom:0.2em">'
                    '<span style="color:#888">課された行先（FW で訪問必須の2拠点）:</span> '
                    + " / ".join(parts)
                    + '</div>'
                )

    # host は移動より「誰とどういう会話をしたか」中心の表示にする
    if persona and persona.get("is_host"):
        convo_items = []
        for n, d in top_partners:
            # 各相手とのやり取りを「相手から (←)」と「自分から (→)」で **両方** 拾う。
            # 「話しかけられた」だけで終わらず、host が **どう返したか / 自分から何を投げたか** を見せるのが目的。
            recv_sample = ""
            sent_sample = ""
            recv_count = 0
            sent_count = 0
            for r in full_rows:
                for m in (r.get("recv") or []):
                    if (m.get("from_name") or "").strip() == n:
                        recv_count += 1
                        if not recv_sample:
                            recv_sample = (m.get("message") or "").strip()[:120]
                for m in (r.get("sent") or []):
                    if (m.get("to_name") or "").strip() == n:
                        sent_count += 1
                        if not sent_sample:
                            sent_sample = (m.get("message") or "").strip()[:120]
            lines = []
            if recv_sample:
                lines.append(
                    f'<div style="color:#aaa;margin-left:1em">'
                    f'<span style="color:#7aa9ff">←</span> {html.escape(n)}さんから ({recv_count}回): '
                    f'「{html.escape(recv_sample)}」</div>'
                )
            if sent_sample:
                lines.append(
                    f'<div style="color:#aaa;margin-left:1em">'
                    f'<span style="color:#3ec96e">→</span> {html.escape(n)}さんへ ({sent_count}回): '
                    f'「{html.escape(sent_sample)}」</div>'
                )
            if not lines:
                lines.append('<div style="color:#666;margin-left:1em">（具体的な発話の抽出に失敗）</div>')
            sample_html = "".join(lines)
            role_label = "企業" if d["is_host"] else "生徒"
            color = "#3ec96e" if d["is_host"] else "#7aa9ff"
            convo_items.append(
                '<li style="margin-bottom:0.4em">'
                f'<span style="color:{color};font-size:0.8em;border:1px solid {color};border-radius:3px;padding:0 0.3em;margin-right:0.3em">{role_label}</span>'
                f'<strong>{html.escape(n)}</strong>'
                f'<span style="color:#888"> (合計 {d["count"]}回)</span>'
                f'{sample_html}'
                '</li>'
            )
        convo_html = "<ul style=\"margin:0.2em 0 0.2em 1em;padding-left:0.5em\">" + "".join(convo_items) + "</ul>" if convo_items else "（会話なし）"
        return (
            '<div class="activity-summary" style="font-size:0.88em;color:#ccc;margin:0.3em 0 0.5em">'
            '<div style="margin-bottom:0.2em"><span style="color:#888">誰とどういう会話をしたか:</span></div>'
            + convo_html
            + '</div>'
        )

    return (
        '<div class="activity-summary" style="font-size:0.88em;color:#ccc;margin:0.3em 0 0.5em">'
        + assigned_html
        + '<div style="margin-bottom:0.2em"><span style="color:#888">移動経路:</span> '
        + f'{html.escape(visit_route) if visit_route else "（移動なし）"}</div>'
        + '<div><span style="color:#888">主な会話相手:</span> '
        + f'{partners_html}</div>'
        + '</div>'
    )


def render_analysis_card(p: dict, md: str, full_rows: list, include_history: bool = False,
                         host_id_set: set | None = None) -> str:
    """analysis-card を生成。
    include_history=True にすると全 step ログも展開する (バックデータ用)。
    本文 (レポート) では False で、「現地で得た気づき」セクション内に
    「移動経路 + 主な会話相手」(rule-based) と「気づき本文」(Gemini md) を並べる。"""
    gender_cls = p.get("gender", "")
    is_host = bool(p.get("is_host"))
    cls = "host" if is_host else gender_cls
    role_label = "企業担当者" if is_host else (p.get("occupation") or "")
    temp = _temp_str(p, compact=True)
    school_fit = p.get("school_fit") or ""
    interest = p.get("interest_tag") or ""
    meta_parts = []
    if role_label:
        meta_parts.append(html.escape(role_label))
    if school_fit and school_fit != "適応":
        meta_parts.append(f"学校適応: {html.escape(school_fit)}")
    if interest:
        meta_parts.append(f"興味: {html.escape(interest)}")
    if temp:
        meta_parts.append(f"気質: {html.escape(temp)}")
    meta_line = " / ".join(meta_parts)
    summary_html = _build_activity_summary(full_rows, host_id_set=host_id_set, persona=p)
    history_html = render_history_details(full_rows) if include_history else ""
    md_html = render_md_to_html(md) if md else ""
    # md 内の「現地で得た気づき (FW中)」見出し直後に summary (移動経路+会話相手) を挿入。
    # ─ 「移動 → 会話 → 気づき」の流れを 1 セクションにまとめる。
    inserted = False
    for h_level in (3, 4, 2):
        marker_open = f'<h{h_level}>現地で得た気づき'
        idx = md_html.find(marker_open)
        if idx < 0:
            continue
        close_tag = f'</h{h_level}>'
        close_idx = md_html.find(close_tag, idx)
        if close_idx >= 0:
            after_close = close_idx + len(close_tag)
            md_html = md_html[:after_close] + summary_html + md_html[after_close:]
            inserted = True
            break
    if not inserted:
        # 見出しが見つからない (host 等で md が空) なら、独立 h4 セクションで出す
        md_html = (
            '<h4 style="margin-top:0.6em;margin-bottom:0.3em">現地で得た気づき (FW中)</h4>'
            + summary_html
            + md_html
        )
    insight_section = md_html
    return (
        f'<div class="analysis-card {cls}">'
        f'<div class="agent-name">{html.escape(_agent_display_label(p))}（{p.get("age","?")}歳・{html.escape(_gender_jp(p))}）</div>'
        + (f'<div class="agent-meta">{meta_line}</div>' if meta_line else "")
        + insight_section
        + history_html
        + '</div>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="FW (Phase B) run directory")
    ap.add_argument("--classroom-run", type=str, required=True,
                    help="座学 (Phase A) run directory (fw_handoff.jsonl 必須)")
    ap.add_argument("--out", type=str, default=None,
                    help="output HTML path (default: <run_dir>/phaseB_report.html)")
    ap.add_argument("--log", type=str, default=None,
                    help="FW sim log file (token usage parse)。未指定なら _run_v2_100step.log を auto-detect")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    fw = load_run(Path(args.run_dir))
    classroom_dir = Path(args.classroom_run)
    handoffs = load_handoffs(classroom_dir)

    # 参加者 (axis_id を持つ非・触媒)
    participants = [p for p in fw["personas"] if p.get("axis_id") and p.get("axis_id") not in ("Sato", "AIRobo", "AIGod", "UMA")]
    print(f"[info] participants: {len(participants)}")
    print(f"[info] handoff covered: {sum(1 for p in participants if handoffs['handoff_by_axis'].get(p['axis_id']))}")

    # structured output を切らないと message/reasoning schema が強制される (汎用テキスト用途では NG)
    client = GeminiClient(enable_structured_output=False, enable_cache=False)

    id_to_name = {a["id"]: a.get("name", f"#{a['id']}") for a in fw["personas"]}

    def analyze_one(p):
        ax = p.get("axis_id")
        thoughts, sent, recv = per_agent_history(fw, p["id"])
        full_rows = build_full_history_rows(fw, p["id"], id_to_name=id_to_name)
        prompt = build_phaseB_prompt(
            p, handoffs["handoff_by_axis"].get(ax), handoffs["fq_by_axis"].get(ax),
            thoughts, sent, recv,
        )
        try:
            md = call_gemini_md(client, SYSTEM_ANALYSIS, prompt, max_tokens=1500)
        except Exception as e:
            print(f"[warn] {p.get('name')} analysis failed: {e}")
            md = ""
        return p, md, full_rows

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_one, p) for p in participants]
        for f in as_completed(futures):
            p, md, fr = f.result()
            print(f"  done: {p.get('name')} ({len(md)}ch, full_rows={len(fr)})")
            results.append((p, md, fr))
    analysis_dur_sec = int(time.time() - t0)
    order = {p["id"]: i for i, p in enumerate(participants)}
    results.sort(key=lambda r: order.get(r[0]["id"], 999))

    # 特徴的な出来事タイムライン抽出
    print("[info] extracting highlights...")
    cand = extract_highlight_candidates(fw, max_pre=40)
    highlights = select_highlights(client, cand) if cand else []
    print(f"     highlights: {len(highlights)}")

    # 設定
    cfg = fw["cfg"]
    meta = cfg.get("metadata", {})
    sim = cfg.get("simulation", {})
    ts = sim.get("time_scale", {})
    duration = sim.get("duration", "?")
    minutes_per_step = ts.get("step_duration_minutes", "?")
    start_time = ts.get("start_time", "?")
    end_time = "?"
    try:
        sh, sm = map(int, str(start_time).split(":"))
        total_min = int(duration) * int(minutes_per_step)
        eh = (sh + total_min // 60) % 24
        em = (sm + total_min % 60) % 60
        end_time = f"{eh:02d}:{em:02d}"
    except Exception:
        pass
    n_agents = len(participants)
    variant = meta.get("variant", "elementary")
    variant_label = {"high": "高校生版", "adult": "社会人版", "elementary": "小学生版"}.get(variant, variant)
    seed = sim.get("seed", "?")
    llm_model = (cfg.get("llm", {}) or {}).get("model", "?")
    classroom_label = classroom_dir.name
    goal = ""
    if participants:
        goal = (participants[0].get("current_goal") or "").strip()
    goal_html = html.escape(goal).replace("\n", "<br>") if goal else "(設定なし)"

    personas_html = "\n".join(render_persona_card(p) for p in participants)
    analysis_html = "\n".join(render_analysis_card(p, md, fr) for p, md, fr in results)
    highlights_html = render_highlights_section(highlights)

    # Interactive timeline (slider + dot + bubble)
    try:
        sh = int(str(start_time).split(":")[0])
    except Exception:
        sh = 13
    try:
        mps = int(minutes_per_step)
    except Exception:
        mps = 3
    interactive_timeline_html = build_phaseB_interactive_timeline(
        fw, participants, int(duration) if isinstance(duration, int) or str(duration).isdigit() else 30,
        start_hour=sh, mins_per_step=mps,
    )

    # 中心問いに付ける v2 ルールベース注釈 (current_goal が長いプロンプトを抱えてるとき)
    goal_note_html = ""
    if goal and ("再訪" in goal or "一緒に行こう" in goal or "**前提**" in goal):
        goal_note_html = (
            '<p style="margin-top:0.7em;font-size:0.82em;color:#a89e80;line-height:1.5">'
            '<strong>備考</strong>: この current_goal には「中心問い (未来像のアップデート)」だけでなく、'
            'v2 sim 時点で実装していた "再訪抑制" "一緒に行こう同期" "13時集合の前提" '
            'などのルール文も同梱されている。次の v3 sim では、純粋な中心問いを 1 文に絞り、'
            '前提や行動指針は別フィールドに分離する予定。</p>'
        )

    # sim log auto-detect
    sim_log_path = args.log
    if not sim_log_path:
        for cand_log in [ROOT / "_run_v2_100step.log", ROOT / "_run_v2_smoke.log", ROOT / "_run108.log"]:
            if cand_log.exists():
                sim_log_path = str(cand_log)
                break
    sim_stats = parse_sim_log(sim_log_path) if sim_log_path else None
    api_table_html = render_api_table(sim_stats, analysis_dur_sec, len(results))

    body = HTML_TPL.format(
        title=f"FW phaseB レポート ({variant_label})",
        variant_label=variant_label,
        n_agents=n_agents,
        duration=duration,
        duration_min=int(duration) * int(minutes_per_step) if isinstance(duration, int) else "?",
        start_time=start_time,
        end_time=end_time,
        seed=seed,
        llm_model=llm_model,
        classroom_label=html.escape(classroom_label),
        goal_html=goal_html,
        personas_html=personas_html,
        analysis_html=analysis_html,
        highlights_html=highlights_html,
        interactive_timeline_html=interactive_timeline_html,
        goal_note_html=goal_note_html,
        api_table_html=api_table_html,
        n_analyzed=len(results),
        run_label=html.escape(Path(args.run_dir).name),
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    out_path = Path(args.out) if args.out else (Path(args.run_dir) / "phaseB_report.html")
    out_path.write_text(body, encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    print(f"     {len(results)} agents analyzed in {fmt_duration(analysis_dur_sec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
