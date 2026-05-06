"""3 variant (elementary / high / adult) のベースラインA条件を並列比較するレポート。

Usage:
  python tools/render_baseline_overview.py \
    --elem  simulations/<elem_run_dir>  --elem-log  _baseline20_elem.log \
    --high  simulations/<high_run_dir>  --high-log  _baseline20_high.log \
    --adult simulations/<adult_run_dir> --adult-log _baseline20_adult.log \
    --out   simulations/<out_dir>/baseline_overview_v1.html
"""
from __future__ import annotations

import argparse
import collections
import html
import io
import json
import logging
import os
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

# render_ab_comparison から再利用するヘルパ
sys.path.insert(0, str(ROOT / "tools"))
from render_ab_comparison import (  # noqa: E402
    load_run, get_catalyst_info, is_participant,
    parse_sim_log, parse_logbuf, estimate_cost_usd, fmt_duration,
    TEMPERAMENT_LABELS, USD_TO_JPY, _strip_prompt_leak,
)


VARIANT_LABELS = {
    "elementary": {"title": "小学生版", "icon": "🎈", "role": "児童", "role_full": "小学生"},
    "high":       {"title": "高校生版", "icon": "🎒", "role": "生徒", "role_full": "高校2年生"},
    "adult":      {"title": "社会人版", "icon": "💼", "role": "参加者", "role_full": "シンギュラボ参加者"},
}


def call_gemini(client, system_prompt, user_prompt, max_tokens=2500):
    try:
        return client.generate(system_prompt, user_prompt, temperature=0.4, max_tokens=max_tokens)
    except Exception as e:
        return f"(LLM error: {e})"


def stats_for_run(data: dict) -> dict:
    msgs = data["msgs"]
    personas = data["personas"]
    sent = collections.Counter(m["from"] for m in msgs)
    n_silent = sum(1 for p in personas if p["id"] not in sent)
    name_by_id = {p["id"]: p["name"] for p in personas}
    top_speakers = sent.most_common(5)
    # 平均/中央値
    counts = [sent.get(p["id"], 0) for p in personas]
    counts.sort()
    n = len(counts)
    median = counts[n//2]
    return {
        "n_msgs": len(msgs),
        "n_personas": len(personas),
        "n_silent": n_silent,
        "min_per_p": min(counts),
        "max_per_p": max(counts),
        "median_per_p": median,
        "top_speakers": [(name_by_id.get(aid, "?"), n) for aid, n in top_speakers],
    }


def sample_messages(data: dict, k: int = 8) -> list[dict]:
    """会話ログから k 件をピックアップ (序盤/中盤/終盤分散 + 長文優先)。"""
    msgs = sorted(data["msgs"], key=lambda m: m.get("step", 0))
    if not msgs:
        return []
    n = len(msgs)
    if n <= k:
        return msgs
    # 序盤/中盤/終盤ぞれぞれから3件、間 1 件
    picks = []
    for start, end in [(0, n//3), (n//3, 2*n//3), (2*n//3, n)]:
        seg = msgs[start:end]
        seg.sort(key=lambda m: -len(m.get("message","") or ""))
        picks.extend(seg[:max(1, k // 3)])
    return picks[:k]


SYSTEM_OVERVIEW = (
    "3つの世代 (小学生 / 高校生 / 社会人) で同じ問いを A条件 (触媒なし、参加者のみ) で回した結果を **データに即して** 比較する。"
    "物語化や美化を一切しない。失敗していたら失敗と書く。\n\n"
    "出力フォーマット (前置き・自己紹介・お辞儀的締め禁止、いきなり最初の見出しから):\n\n"
    "### 共通の問い\n"
    "本実験で全 variant に投げられた中心問いを1文で。\n\n"
    "### 世代ごとに何が起きたか\n"
    "小学生 / 高校生 / 社会人 それぞれで、参加者だけのときに何が起きたかを 各3-5文。"
    "**いいところ・悪いところとして何が出たか / 未来像として何を描いたか / 議論がまとまったかバラけたか** を率直に。\n\n"
    "### 世代差の構造的な違い\n"
    "3世代で **何が決定的に違ったか** を3-5文。例: 議論の深さ、未来像の現実性、共有のされ方、何にフォーカスが集まりやすかったか。"
    "ログから直接読み取れる事実だけを使う。\n\n"
    "### 気質3次元・国籍・職業背景と動きの関係\n"
    "気質3軸 (外向性/楽天性/好奇心)、外国人ルーツ、ホワイト/ブルーカラー、観光客 などの背景属性が、"
    "話題の入りやすさ・拾われ方にどう影響していたか観察。3-5文。\n\n"
    "### 印象的な発話 (各世代1-2件)\n"
    "ログから直接引用できる『実際に起きた瞬間』を世代ごとに1-2件、引用形式で。創作禁止。\n\n"
    "重要: 「研究員として〜以下の通り要約します」のような前置きや締めは絶対書かない。## や ### の見出しから直接本文に入る。"
)


SYSTEM_TEAM_VISIONS = (
    "あるまちについて20人が議論した会話ログを読み、自然発生した小グループ (チーム) 単位で"
    "**そのチームが議論を通じて到達した未来像** を抽出する。\n\n"
    "ルール:\n"
    "- 同じ未来像をめぐって何度かやり取りした人たち をひとつのチームとみなす。"
    "1人だけで完結しているなら『1人の構想』として書いてよい。\n"
    "- 各チームは 2-7人 程度。20人全員が必ずしも分類できる必要はない (浮いてた人がいたなら触れない)。\n"
    "- **未来像は「〇〇な品川になる」「〇〇が増える」「〇〇が共存する」のような状態描写**。"
    "「〇〇しよう」のような提案ではなく、**たどり着いた共通像**を書く。\n"
    "- ログから直接引用できる情報のみ使う。創作・補間・推測禁止。\n"
    "- 描いていなければ「未到達」と書く。\n\n"
    "JSON 配列のみ返す (前置き・コードブロック禁止):\n"
    '[{"team_label":"...", "members":["氏名1","氏名2",...], "future_vision":"...", "evidence":"...(引用1個)"}, ...]'
)


def extract_team_visions(data: dict, client, variant_label: str) -> list[dict]:
    """1 variant の発話を全部 Gemini に渡して、チーム単位の未来像を抽出させる。"""
    msgs = data.get("msgs", [])
    if not msgs:
        return []
    name_by_id = {p["id"]: p["name"] for p in data["personas"]}
    # Step順、最大250件くらいに圧縮 (token ケア)
    msgs = sorted(msgs, key=lambda m: m.get("step", 0))
    if len(msgs) > 250:
        msgs = msgs[:250]
    log_lines = []
    for m in msgs:
        fr = name_by_id.get(m.get("from"), "?")
        to = name_by_id.get(m.get("to"), "?")
        body = (m.get("message") or "")[:200]
        log_lines.append(f"[step{m.get('step'):03d}] {fr} → {to}: {body}")
    user_p = (
        f"## variant: {variant_label}\n"
        f"## 全発話ログ ({len(msgs)} 件)\n"
        + "\n".join(log_lines)
        + "\n\n## 課題\n上記会話ログから、自然に形成された小グループとそれぞれが到達した未来像を抽出。JSON 配列で返す。"
    )
    text = client.generate(SYSTEM_TEAM_VISIONS, user_p, temperature=0.4, max_tokens=2000)
    if not text:
        return []
    text = _strip_prompt_leak(text)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    m = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def build_synthesis_prompt(d_elem, d_high, d_adult) -> str:
    blocks = []
    for label, d in [("小学生", d_elem), ("高校生", d_high), ("社会人", d_adult)]:
        s = stats_for_run(d)
        # 中心問い (参加者の current_goal から)
        goal = ""
        for p in d["personas"]:
            if p.get("axis_id") and p.get("axis_id") not in ("Sato", "UMA", "AIRobo"):
                goal = (p.get("current_goal") or "")
                break
        blocks.append(
            f"=== {label} ===\n"
            f"問い: {goal}\n"
            f"統計: 全メッセージ={s['n_msgs']}, 参加者={s['n_personas']}人, silent={s['n_silent']}人, "
            f"発話/人 中央値={s['median_per_p']}, 最大={s['max_per_p']}, 最小={s['min_per_p']}\n"
            f"発話量 top5: {', '.join(f'{n}({c})' for n, c in s['top_speakers'])}\n"
        )
        # 気質分布
        ext_dist = collections.Counter(p.get("temperament_extroversion") for p in d["personas"])
        opt_dist = collections.Counter(p.get("temperament_optimism") for p in d["personas"])
        cur_dist = collections.Counter(p.get("temperament_curiosity") for p in d["personas"])
        blocks.append(
            f"気質分布: 外向性 {dict(ext_dist)} / 楽天性 {dict(opt_dist)} / 好奇心 {dict(cur_dist)}\n"
        )
        # サンプルメッセージ (8件)
        sample = sample_messages(d, k=8)
        msg_lines = []
        name_by_id = {p["id"]: p["name"] for p in d["personas"]}
        for m in sample:
            fr = name_by_id.get(m["from"], "?")
            to = name_by_id.get(m.get("to"), "?")
            msg_lines.append(f"[step{m.get('step'):03d}] {fr} → {to}: {m.get('message','')[:160]}")
        blocks.append("会話サンプル (序盤/中盤/終盤、各 + 長文):\n" + "\n".join(msg_lines) + "\n")
    return "\n".join(blocks) + "\n\n以上のデータから、上記フォーマット通り比較レポートを書け。"


def render_persona_pills(personas: list[dict]) -> str:
    """20 personas を細かいタグで一行ずつ。"""
    rows = []
    for p in sorted(personas, key=lambda x: x.get("axis_id","")):
        ext = TEMPERAMENT_LABELS["extroversion"].get(p.get("temperament_extroversion",""),"—")
        opt = TEMPERAMENT_LABELS["optimism"].get(p.get("temperament_optimism",""),"—")
        cur = TEMPERAMENT_LABELS["curiosity"].get(p.get("temperament_curiosity",""),"—")
        gender = p.get("gender","")
        cls = "male" if gender=="male" else ("female" if gender=="female" else "")
        rows.append(
            f'<tr class="p-row {cls}">'
            f'<td>{html.escape(p.get("axis_id",""))}</td>'
            f'<td>{html.escape(p.get("name",""))}</td>'
            f'<td>{p.get("age","?")}</td>'
            f'<td>{html.escape(gender)}</td>'
            f'<td>{html.escape(ext)} / {html.escape(opt)} / {html.escape(cur)}</td>'
            f'<td style="font-size:0.85em;color:#666">{html.escape((p.get("background","") or "")[:80])}</td>'
            f'<td style="font-size:0.85em;color:#666">「{html.escape(p.get("catchphrase","") or "")}」</td>'
            f'</tr>'
        )
    return f'<table class="p-table">' \
           f'<thead><tr><th>id</th><th>名前</th><th>年齢</th><th>性別</th>' \
           f'<th>気質 (外向/楽天/好奇)</th><th>背景 (要約)</th><th>口癖</th></tr></thead>' \
           f'<tbody>{"".join(rows)}</tbody></table>'


def render_stats_table(stats_by_v: dict) -> str:
    rows = []
    for vkey, label in [("elementary","小学生"), ("high","高校生"), ("adult","社会人")]:
        s = stats_by_v[vkey]
        rows.append(
            f'<tr><th>{label}</th>'
            f'<td>{s["n_msgs"]:,}</td>'
            f'<td>{s["n_personas"]}</td>'
            f'<td>{s["n_silent"]}</td>'
            f'<td>{s["min_per_p"]} / {s["median_per_p"]} / {s["max_per_p"]}</td>'
            f'</tr>'
        )
    return f'<table class="stats-table"><thead><tr><th></th>' \
           f'<th>全メッセージ</th><th>参加者数</th><th>silent</th>' \
           f'<th>発話/人 (min/中央値/max)</th></tr></thead>' \
           f'<tbody>{"".join(rows)}</tbody></table>'


def render_team_visions(visions: list[dict], variant_label: str) -> str:
    if not visions:
        return f'<p style="color:#888">({variant_label}: チーム単位の未来像は抽出されなかった)</p>'
    cards = []
    for v in visions:
        members = " / ".join(html.escape(m) for m in (v.get("members") or []))
        ev = (v.get("evidence") or "").strip()
        cards.append(
            f'<div class="vision">'
            f'<div class="vision-label">{html.escape(v.get("team_label","?"))}</div>'
            f'<div class="vision-members">メンバー: {members}</div>'
            f'<div class="vision-future">{html.escape(v.get("future_vision",""))}</div>'
            + (f'<div class="vision-evidence">引用: {html.escape(ev)}</div>' if ev else '')
            + f'</div>'
        )
    return "".join(cards)


def render_field_questions(run_dir: Path) -> str:
    fq_path = run_dir / "field_questions.jsonl"
    if not fq_path.exists():
        return f'<p style="color:#888">(field_questions.jsonl 未生成)</p>'
    rows = []
    for line in fq_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        qs = r.get("questions") or []
        qs_html = "<ul>" + "".join(f"<li>{html.escape(q)}</li>" for q in qs) + "</ul>" if qs else "<i>(空)</i>"
        rows.append(
            f'<tr>'
            f'<td>{html.escape(r.get("axis_id",""))}</td>'
            f'<td>{html.escape(r.get("name",""))}</td>'
            f'<td style="font-size:0.88em">{qs_html}</td>'
            f'<td style="font-size:0.85em;color:#666;font-style:italic">{html.escape(r.get("one_liner","") or "")}</td>'
            f'</tr>'
        )
    return f'<table class="fq-table">' \
           f'<thead><tr><th>id</th><th>名前</th><th>確かめたいこと (1-3)</th><th>一行サマリ</th></tr></thead>' \
           f'<tbody>{"".join(rows)}</tbody></table>'


def render_sample_messages(d: dict, k: int = 12) -> str:
    name_by_id = {p["id"]: p["name"] for p in d["personas"]}
    msgs = sample_messages(d, k=k)
    rows = []
    for m in msgs:
        fr = name_by_id.get(m["from"], "?")
        to = name_by_id.get(m.get("to"), "?")
        rows.append(
            f'<div class="msg">'
            f'<div class="msg-meta">step{m.get("step"):03d} · {html.escape(fr)} → {html.escape(to)}</div>'
            f'<div class="msg-body">「{html.escape(m.get("message",""))}」</div>'
            f'</div>'
        )
    return "".join(rows)


def render_api_section(stats_list, analysis_stats, analysis_dur):
    rows = []
    total = {"n_calls":0, "input_tokens":0, "cache_read_tokens":0,
             "uncached_input":0, "output_tokens":0, "duration_sec":0, "cost_usd":0.0}
    for label, s in stats_list:
        s = s or {}
        cost = estimate_cost_usd(s) if s else 0.0
        total["n_calls"] += s.get("n_calls", 0)
        total["input_tokens"] += s.get("input_tokens", 0)
        total["cache_read_tokens"] += s.get("cache_read_tokens", 0)
        total["uncached_input"] += s.get("uncached_input", 0)
        total["output_tokens"] += s.get("output_tokens", 0)
        total["duration_sec"] += s.get("duration_sec", 0)
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
        f"<tr style='border-top:2px solid #888;font-weight:bold'><th>シミュ合計</th>"
        f"<td>{total['n_calls']:,}</td>"
        f"<td>{total['input_tokens']:,}</td>"
        f"<td>{total['output_tokens']:,}</td>"
        f"<td>{cache_rate_total:.1f}%</td>"
        f"<td>{fmt_duration(total['duration_sec'])}</td>"
        f"<td>${total['cost_usd']:.3f}</td></tr>"
    )
    if analysis_stats:
        cost_an = estimate_cost_usd(analysis_stats)
        cr_an = (analysis_stats.get("cache_read_tokens", 0) / analysis_stats.get("input_tokens", 1) * 100) if analysis_stats.get("input_tokens") else 0
        rows.append(
            f"<tr><th>分析 (このレポート)</th>"
            f"<td>{analysis_stats.get('n_calls',0):,}</td>"
            f"<td>{analysis_stats.get('input_tokens',0):,}</td>"
            f"<td>{analysis_stats.get('output_tokens',0):,}</td>"
            f"<td>{cr_an:.1f}%</td>"
            f"<td>{fmt_duration(analysis_dur)}</td>"
            f"<td>${cost_an:.3f}</td></tr>"
        )
        total["cost_usd"] += cost_an
    rows.append(
        f"<tr style='border-top:2px solid #444;font-weight:bold;background:#f8f6f0'><th>総合計 (≒¥)</th>"
        f"<td colspan='5' style='text-align:right'></td>"
        f"<td>${total['cost_usd']:.3f} (≒¥{total['cost_usd']*USD_TO_JPY:.0f})</td></tr>"
    )
    return (
        f'<table class="api-table">'
        f'<thead><tr><th></th><th>API呼び出し</th><th>入力token</th><th>出力token</th>'
        f'<th>cache hit率</th><th>所要時間</th><th>推定コスト</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


HTML_TPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>ベースライン比較レポート (3 variant × 100step × A条件)</title>
<style>
body {{ font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif; max-width: 1080px; margin: 2em auto; padding: 0 1.4em; color: #222; line-height: 1.65; }}
h1 {{ font-size: 1.7em; border-bottom: 3px solid #4a8ad8; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.3em; margin-top: 1.6em; border-left: 5px solid #4a8ad8; padding-left: 0.6em; }}
h3 {{ font-size: 1.1em; margin-top: 1.2em; }}
.subtitle {{ color: #777; font-size: 0.85em; font-weight: normal; }}
.synthesis {{ background: #fff7e6; padding: 1.2em 1.6em; border-radius: 10px; border: 1px solid #ddc88c; white-space: pre-wrap; }}
.stats-table {{ border-collapse: collapse; margin: 1em 0; }}
.stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 0.5em 0.9em; text-align: right; }}
.stats-table th:first-child, .stats-table tbody th {{ text-align: left; background: #fafafa; }}
.stats-table thead th {{ background: #f0f0f0; }}
.p-table {{ border-collapse: collapse; width: 100%; font-size: 0.86em; margin: 0.6em 0 1.4em; }}
.p-table th, .p-table td {{ border: 1px solid #ddd; padding: 0.35em 0.6em; text-align: left; vertical-align: top; }}
.p-table thead th {{ background: #f0f0f0; }}
.p-row.male {{ background: #f6fafe; }}
.p-row.female {{ background: #fcf4f8; }}
.var-section {{ border: 2px solid #ddd; border-radius: 10px; padding: 1em 1.4em; margin: 1.5em 0; background: #fafafa; }}
.var-section h2 {{ margin-top: 0; }}
.msg {{ border-left: 3px solid #d8a040; background: #fff8ef; padding: 0.5em 0.9em; margin: 0.4em 0; font-size: 0.9em; border-radius: 4px; }}
.msg-meta {{ color: #888; font-size: 0.78em; margin-bottom: 0.2em; }}
.msg-body {{ color: #333; }}
.api-table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; margin: 0.8em 0; }}
.api-table th, .api-table td {{ border: 1px solid #ccc; padding: 0.5em 0.8em; text-align: right; }}
.api-table th {{ background: #f0f0f0; font-weight: 500; }}
.api-table th:first-child, .api-table tbody th {{ text-align: left; background: #fafafa; }}
.vision {{ background: #f4f7fb; border-left: 4px solid #4a8ad8; padding: 0.6em 1em; margin: 0.5em 0; border-radius: 5px; }}
.vision-label {{ font-weight: bold; color: #2a4a7c; font-size: 0.95em; margin-bottom: 0.2em; }}
.vision-members {{ font-size: 0.82em; color: #666; margin-bottom: 0.3em; }}
.vision-future {{ font-size: 0.95em; color: #222; line-height: 1.55; }}
.vision-evidence {{ font-size: 0.78em; color: #888; font-style: italic; margin-top: 0.4em; padding-top: 0.4em; border-top: 1px dashed #ccc; }}
.fq-table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 0.5em 0 1em; }}
.fq-table th, .fq-table td {{ border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: left; vertical-align: top; }}
.fq-table thead th {{ background: #f0f0f0; }}
.fq-table ul {{ margin: 0; padding-left: 1.2em; }}
.fq-table li {{ margin-bottom: 0.2em; }}
.meta {{ color: #aaa; font-size: 0.8em; margin-top: 3em; text-align: center; }}
</style></head><body>
<h1>ベースライン比較レポート <span class="subtitle">— 3 variant × 100step × A条件 (触媒なし)</span></h1>
<p style="color:#666">同一問い・同一 seed (42) の下で、小学生 / 高校生 / 社会人 のそれぞれ20人 が触媒なし条件でどう振る舞ったかを並べて見る。</p>

<h2>定量比較</h2>
{stats_table}

<h2>クロス世代 synthesis (Gemini)</h2>
<div class="synthesis">{synth_html}</div>

<h2>API消費・コスト・所要時間</h2>
{api_section}

{var_sections}

<div class="meta">Generated by tools/render_baseline_overview.py</div>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elem", required=True, help="elementary run dir")
    ap.add_argument("--high", required=True, help="high run dir")
    ap.add_argument("--adult", required=True, help="adult run dir")
    ap.add_argument("--elem-log", default=None)
    ap.add_argument("--high-log", default=None)
    ap.add_argument("--adult-log", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d_elem = load_run(Path(args.elem))
    d_high = load_run(Path(args.high))
    d_adult = load_run(Path(args.adult))

    stats_by_v = {
        "elementary": stats_for_run(d_elem),
        "high":       stats_for_run(d_high),
        "adult":      stats_for_run(d_adult),
    }

    # 分析 LLM コール (synthesis)
    analysis_buf = io.StringIO()
    analysis_handler = logging.StreamHandler(analysis_buf)
    analysis_handler.setLevel(logging.INFO)
    analysis_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("llm_client_factory").addHandler(analysis_handler)
    logging.getLogger("llm_client_factory").setLevel(logging.INFO)
    analysis_t_start = time.time()

    client = GeminiClient(
        model="gemini-3.1-flash-lite-preview", temperature=0.4, max_tokens=2500,
        enable_cache=False, enable_structured_output=False,
    )
    user_p = build_synthesis_prompt(d_elem, d_high, d_adult)
    print("[info] generating cross-variant synthesis ...")
    synth = call_gemini(client, SYSTEM_OVERVIEW, user_p, max_tokens=2500) or "(synthesis失敗)"
    synth = _strip_prompt_leak(synth)

    analysis_dur = int(time.time() - analysis_t_start)
    analysis_stats = parse_logbuf(analysis_buf.getvalue())

    # API消費
    sim_stats = []
    for label, log in [("小学生 sim", args.elem_log), ("高校生 sim", args.high_log), ("社会人 sim", args.adult_log)]:
        st = parse_sim_log(Path(log)) if log else None
        sim_stats.append((label, st))
    api_section = render_api_section(sim_stats, analysis_stats, analysis_dur)

    # variant ごとに team visions を抽出 (3 LLM call)
    print("[info] extracting team visions per variant ...")
    visions_by_v = {}
    for vkey, d, label in [("elementary", d_elem, "小学生"), ("high", d_high, "高校生"), ("adult", d_adult, "社会人")]:
        visions_by_v[vkey] = extract_team_visions(d, client, label)
        print(f"  [{vkey}] visions={len(visions_by_v[vkey])}")

    # variant ごとのセクション
    var_section_html = []
    for vkey, d, run_dir_path in [
        ("elementary", d_elem, Path(args.elem)),
        ("high",       d_high, Path(args.high)),
        ("adult",      d_adult, Path(args.adult)),
    ]:
        meta = VARIANT_LABELS[vkey]
        s = stats_by_v[vkey]
        scene_label = d.get("cfg",{}).get("metadata",{}).get("scene_label","?")
        scene_text = d.get("cfg",{}).get("metadata",{}).get("scene_text","")
        town_name = d.get("cfg",{}).get("metadata",{}).get("town_name","")
        visions_html = render_team_visions(visions_by_v[vkey], meta["title"])
        fq_html = render_field_questions(run_dir_path)
        var_section_html.append(
            f'<div class="var-section">'
            f'<h2>{meta["icon"]} {meta["title"]} ({meta["role_full"]} {s["n_personas"]}人 / 場所={scene_label} / 議論対象={town_name})</h2>'
            f'<p style="color:#666">{html.escape(scene_text)}</p>'
            f'<h3>参加者</h3>'
            f'{render_persona_pills(d["personas"])}'
            f'<h3>各チームが描いた未来像</h3>'
            f'{visions_html}'
            f'<h3>各人のフィールドワーク確認事項 (現地で確かめたいこと)</h3>'
            f'{fq_html}'
            f'<h3>会話サンプル (12件、序盤/中盤/終盤分散・長文優先)</h3>'
            f'{render_sample_messages(d, k=12)}'
            f'</div>'
        )

    out_html = HTML_TPL.format(
        stats_table=render_stats_table(stats_by_v),
        synth_html=html.escape(synth).replace("\n", "<br>"),
        api_section=api_section,
        var_sections="\n".join(var_section_html),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
