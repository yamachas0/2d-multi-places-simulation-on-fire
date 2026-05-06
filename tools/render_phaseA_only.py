"""Phase A (座学) 単独レポート — Phase B/C 無しでも見れる軽量レポート。

使い道: smoke 検証で「同じ会話を繰り返してないか」「silent はあるか」「発話の多様性」を
PDF/HTML で目視レビューする。

含まれるセクション:
  1. 表紙 (タイトル / 通し番号 / 走査日時 / scale)
  2. ペルソナ一覧テーブル
  3. 発話頻度サマリ (各人の sent / silent / 主な相手)
  4. 各人の発話全件 (step 順、相手別、同じテーマループ可視化)
  5. Phase A interactive timeline (既存 build_phaseB_interactive_timeline 流用)
  6. 各人の全 step ログ (memory + reasoning + 発話 + 受信)
  7. API 消費 (sim.log)

Usage:
  python tools/render_phaseA_only.py --run <run_dir> [--out <html>]
"""
from __future__ import annotations
import argparse, datetime, html, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
load_dotenv(ROOT / ".env")

from render_phaseB_report import (  # noqa: E402
    load_run, build_full_history_rows, render_history_details,
    build_phaseB_interactive_timeline, _gender_jp, _temp_str,
    parse_sim_log, render_api_table, fmt_duration,
)


HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="color-scheme" content="dark">
<title>{title}</title>
<style>
@page {{ size: 13.33in 7.5in; margin: 10mm; }}
:root {{ color-scheme: dark; }}
body {{ font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
        max-width: 1200px; margin: 0 auto; padding: 1.5em 1em 4em; line-height: 1.7;
        color: #e8e8e8; background: #15171a; }}
h1 {{ color: #ffd85f; border-bottom: 2px solid #4a8ad8; padding-bottom: 0.3em; margin-top: 2em; }}
h2 {{ color: #ffd85f; border-left: 4px solid #ffd85f; padding-left: 0.6em; margin-top: 1.3em; }}
h3 {{ color: #62e08a; border-left: 3px solid #62e08a; padding-left: 0.5em; margin-top: 1em; }}
h4 {{ color: #ffaf3c; margin-top: 0.8em; }}
section {{ margin-top: 2em; }}
.page-break {{ page-break-before: always; }}
.cover {{ text-align: center; padding: 4em 1em; }}
.cover h1 {{ font-size: 2em; border: none; }}
.cover .meta {{ color: #aaa; font-size: 1em; line-height: 2; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #2c2c2c; padding: 0.4em 0.6em; vertical-align: top; }}
th {{ background: #232830; color: #aaa; text-align: left; font-weight: normal; }}
td {{ background: #1a1d22; }}
table.persona-table td.male {{ background: #1a2230; }}
table.persona-table td.female {{ background: #2a1c25; }}
table.persona-table td.other {{ background: #2a2820; }}
.summary-card {{ background: #1d2025; border: 1px solid #2c2c2c; border-radius: 8px;
                padding: 0.8em 1em; margin: 0.4em 0; }}
.summary-card .agent-name {{ color: #ffd85f; font-weight: 600; font-size: 1em; margin-bottom: 0.3em; }}
.summary-card .stats {{ color: #aaa; font-size: 0.88em; }}
.msg-block {{ background: #16191e; border: 1px solid #2c2c2c; border-radius: 8px;
              padding: 0.7em 0.9em; margin: 0.6em 0 1.4em; page-break-inside: auto; }}
.msg-block .agent-name {{ color: #ffd85f; font-weight: 600; font-size: 1em; margin-bottom: 0.4em; }}
.msg-row {{ display: flex; gap: 0.5em; padding: 0.25em 0; border-bottom: 1px dashed #2a2d33; }}
.msg-row:last-child {{ border-bottom: none; }}
.msg-row .step {{ color: #888; width: 4em; flex-shrink: 0; font-size: 0.85em; }}
.msg-row .partner {{ color: #4ea8ff; width: 9em; flex-shrink: 0; font-size: 0.85em; }}
.msg-row .text {{ color: #ddd; font-size: 0.88em; line-height: 1.45; word-break: break-word; }}
.msg-row.silent .text {{ color: #666; font-style: italic; }}
.note {{ color: #aaa; font-size: 0.88em; line-height: 1.5; }}
.api-meta {{ background: #1d2025; padding: 0.8em 1em; border-radius: 8px; font-size: 0.9em; }}
.api-meta dl {{ display: grid; grid-template-columns: 12em 1fr; gap: 0.3em 0.6em; margin: 0; }}
.api-meta dt {{ color: #aaa; }}
@media print {{
  body {{ background: #fff; color: #222; max-width: none; }}
  h1 {{ color: #000; border-bottom-color: #4a8ad8; }}
  h2 {{ color: #333; border-left-color: #4a8ad8; }}
  h3 {{ color: #333; }}
  h4 {{ color: #b08000; }}
  td {{ background: #fff; color: #222; border-color: #ddd; }}
  th {{ background: #ececec; color: #333; }}
  .summary-card, .msg-block, .api-meta {{ background: #fafafa; color: #222; border-color: #ccc; }}
  .summary-card .agent-name, .msg-block .agent-name {{ color: #b08000; }}
  .msg-row .partner {{ color: #2a5d8a; }}
  .msg-row .text {{ color: #222; }}
  .msg-row.silent .text {{ color: #888; }}
  .note, .api-meta dt {{ color: #555; }}
  table.persona-table td.male {{ background: #eff5ff; }}
  table.persona-table td.female {{ background: #fff0f5; }}
  table.persona-table td.other {{ background: #fffbe8; }}
}}
</style></head><body>
"""

FOOT = "</body></html>\n"


def render_persona_table(personas: list[dict]) -> str:
    """ペルソナ一覧テーブル。性別で色分け。"""
    rows = []
    for p in personas:
        gender = p.get("gender", "")
        gender_jp = _gender_jp(p)
        cls = gender if gender in ("male", "female", "other") else ""
        meta_parts = []
        if p.get("school_fit") and p.get("school_fit") != "適応":
            meta_parts.append(f"学校適応:{p['school_fit']}")
        if p.get("interest_tag"):
            meta_parts.append(f"興味:{p['interest_tag']}")
        temp = _temp_str(p, compact=True)
        if temp:
            meta_parts.append(f"気質:{temp}")
        meta = " / ".join(meta_parts)
        rows.append(
            f'<tr>'
            f'<td class="{cls}">{html.escape(p.get("name","?"))}</td>'
            f'<td>{p.get("age","?")}</td>'
            f'<td>{html.escape(gender_jp)}</td>'
            f'<td>{html.escape(p.get("axis_id","") or "")}</td>'
            f'<td style="font-size:0.85em;color:#aaa">{html.escape(meta)}</td>'
            f'</tr>'
        )
    return (
        '<table class="persona-table">'
        '<thead><tr><th>名前</th><th>年齢</th><th>性別</th><th>軸ID</th><th>メタ</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def render_speech_summary(personas: list[dict], msgs: list[dict]) -> str:
    """各 agent の発話統計 (sent / silent / 主な相手) を要約。"""
    by_sender = defaultdict(list)
    for m in msgs:
        by_sender[m.get("from")].append(m)
    cards = []
    for p in personas:
        aid = p.get("id")
        sent = by_sender.get(aid, [])
        silent = sum(1 for m in sent if not (m.get("message") or "").strip())
        partners = Counter(m.get("to_name") or "?" for m in sent)
        partner_str = ", ".join(f"{n}({c})" for n, c in partners.most_common(3))
        cards.append(
            f'<div class="summary-card">'
            f'<div class="agent-name">{html.escape(p.get("name","?"))}</div>'
            f'<div class="stats">sent: {len(sent)} / silent: {silent} / 主な相手: {html.escape(partner_str) or "—"}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def render_messages_per_agent(personas: list[dict], msgs: list[dict]) -> str:
    """各 agent の発話を step 順に全件表示。同じテーマループの目視チェック用。"""
    by_sender = defaultdict(list)
    for m in msgs:
        by_sender[m.get("from")].append(m)
    parts = []
    for p in personas:
        aid = p.get("id")
        sent = sorted(by_sender.get(aid, []), key=lambda m: m.get("step", 0))
        if not sent:
            continue
        rows = []
        for m in sent:
            text = (m.get("message") or "").strip()
            cls = "silent" if not text else ""
            shown = text or "(silent)"
            rows.append(
                f'<div class="msg-row {cls}">'
                f'<span class="step">step{m.get("step","?"):>2}</span>'
                f'<span class="partner">→ {html.escape(m.get("to_name","?"))}</span>'
                f'<span class="text">{html.escape(shown)}</span>'
                f'</div>'
            )
        parts.append(
            f'<div class="msg-block">'
            f'<div class="agent-name">{html.escape(p.get("name","?"))} ({len(sent)} 発話)</div>'
            f'{"".join(rows)}'
            f'</div>'
        )
    return "\n".join(parts)


def _summary_for_compare(run: Path) -> dict:
    """別 run の結果サマリ (発話数/silent/API消費) を取得して比較表用に返す。"""
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) if (run / "config.yaml").exists() else {}
    msgs_path = run / "messages.jsonl"
    msgs = []
    if msgs_path.exists():
        for l in msgs_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                msgs.append(json.loads(l))
    n_msgs = len(msgs)
    silent = sum(1 for m in msgs if not (m.get("message") or "").strip())
    log_path = run / "sim.log"
    api = None
    if log_path.exists():
        try:
            api = parse_sim_log(log_path)
        except Exception:
            api = None
    sim = cfg.get("simulation", {})
    return {
        "run_dir": str(run),
        "run_num": (lambda parts: parts[2] if len(parts) >= 3 and parts[2].isdigit() else "")(run.name.split("_")),
        "duration": sim.get("duration", "?"),
        "n_msgs": n_msgs,
        "silent": silent,
        "api": api,
        "global_channel": sim.get("global_channel", False),
        "chat_thread_mode": sim.get("chat_thread_mode", False),
    }


def render_compare_table(this_summary: dict, other_summary: dict) -> str:
    """2 run 比較表 (API消費 + 発話数 + 設定)。"""
    def _row(label, a, b):
        return f'<tr><th style="text-align:left">{html.escape(label)}</th><td>{a}</td><td>{b}</td></tr>'
    def _api_kv(api, key):
        if not api:
            return "—"
        v = api.get(key)
        if isinstance(v, (int, float)):
            if "yen" in key or "cost" in key:
                return f"¥{v:.2f}" if v < 100 else f"¥{v:.0f}"
            if "duration" in key:
                return fmt_duration(int(v))
            return f"{v:,}" if v >= 1000 else f"{v}"
        return html.escape(str(v))
    rows = []
    rows.append(_row("通し番号", f"#{this_summary['run_num']}", f"#{other_summary['run_num']}"))
    rows.append(_row("duration (steps)", this_summary['duration'], other_summary['duration']))
    rows.append(_row("global_channel", "ON" if this_summary['global_channel'] else "OFF", "ON" if other_summary['global_channel'] else "OFF"))
    rows.append(_row("chat_thread_mode", "ON" if this_summary['chat_thread_mode'] else "OFF", "ON" if other_summary['chat_thread_mode'] else "OFF"))
    rows.append(_row("総 messages", this_summary['n_msgs'], other_summary['n_msgs']))
    rows.append(_row("silent (空文字応答)", this_summary['silent'], other_summary['silent']))
    if this_summary["api"] and other_summary["api"]:
        # gemini-2.5-flash-lite 単価想定 (USD per M tokens). 概算用。
        # 入力 $0.10, cached $0.025, 出力 $0.40 (代表値、preview/GA で揺れあり)
        def _cost_yen(api):
            if not api:
                return 0
            usd = (api.get("uncached_input", 0)*0.10 + api.get("cache_read_tokens", 0)*0.025
                   + api.get("output_tokens", 0)*0.40) / 1_000_000
            return usd * 150  # 1 USD = 150 JPY

        labels = [
            ("n_calls", "LLM呼出回数"),
            ("input_tokens", "input トークン"),
            ("cache_read_tokens", "cache hit"),
            ("uncached_input", "uncached input"),
            ("output_tokens", "output トークン"),
            ("duration_sec", "所要時間"),
        ]
        for k, lab in labels:
            rows.append(_row(lab, _api_kv(this_summary["api"], k), _api_kv(other_summary["api"], k)))
        rows.append(_row("コスト概算 (¥)", f"¥{_cost_yen(this_summary['api']):.2f}", f"¥{_cost_yen(other_summary['api']):.2f}"))
    return (
        '<table style="width:auto;min-width:24em">'
        '<thead><tr><th>項目</th><th>this run</th><th>compare</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Phase A run dir")
    ap.add_argument("--out", default=None, help="HTML out path (default: <run>/v3_report_<NN>.html)")
    ap.add_argument("--compare-with", default=None, help="比較対象の別 run dir (例: smoke8 vs smoke11 比較表を出す)")
    args = ap.parse_args()

    run = Path(args.run)
    data = load_run(run)
    personas = data["personas"]
    msgs = data["msgs"]
    cfg = data["cfg"]
    sim = cfg.get("simulation", {})
    meta = cfg.get("metadata", {})
    duration = sim.get("duration", "?")
    seed = sim.get("seed", "?")
    variant = meta.get("variant", "")
    condition = meta.get("condition", "")

    # 通し番号抽出
    def _extract_run_number(p: Path) -> str:
        parts = p.name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            return parts[2]
        return ""
    run_num = _extract_run_number(run)
    suffix = f"_{run_num}" if run_num else ""

    # API meta — シンプルな自前テーブル (render_api_table は v3_report.py 側で他の signature)
    log_path = run / "sim.log"
    api_table_html = ""
    if log_path.exists():
        try:
            api = parse_sim_log(log_path)
            if api:
                rate = (api.get("cache_read_tokens", 0) / api.get("input_tokens", 1) * 100) if api.get("input_tokens") else 0
                api_table_html = (
                    '<table style="width:auto;min-width:24em">'
                    '<thead><tr><th>項目</th><th>値</th></tr></thead><tbody>'
                    f"<tr><th>LLM呼出回数</th><td>{api.get('n_calls', 0):,}</td></tr>"
                    f"<tr><th>input トークン (合計)</th><td>{api.get('input_tokens', 0):,}</td></tr>"
                    f"<tr><th>cache hit (cache_read)</th><td>{api.get('cache_read_tokens', 0):,} ({rate:.1f}%)</td></tr>"
                    f"<tr><th>uncached input</th><td>{api.get('uncached_input', 0):,}</td></tr>"
                    f"<tr><th>output トークン</th><td>{api.get('output_tokens', 0):,}</td></tr>"
                    f"<tr><th>所要時間</th><td>{fmt_duration(api.get('duration_sec', 0))}</td></tr>"
                    '</tbody></table>'
                )
        except Exception as e:
            api_table_html = f'<p style="color:#888">(sim.log 解析失敗: {html.escape(str(e))})</p>'

    # 比較表 (--compare-with 指定時のみ)
    compare_html = ""
    if args.compare_with:
        compare_run = Path(args.compare_with)
        if compare_run.exists():
            this_sum = _summary_for_compare(run)
            other_sum = _summary_for_compare(compare_run)
            compare_html = (
                '<section class="page-break">'
                '<h1>比較: this run vs compare</h1>'
                '<div class="note">stateless (=毎step prompt 全体を送る) vs chat session thread化 '
                '(= 各 agent が独立 chat session を持つ) の API消費・発話量・silent rate 比較。</div>'
                f'{render_compare_table(this_sum, other_sum)}'
                '</section>'
            )

    title = f"Phase A 単独レポート ({variant}/{condition}) — run #{run_num}"

    # Cover
    cover = (
        '<section class="cover">'
        f'<h1>{html.escape(title)}</h1>'
        '<div class="meta">'
        f'生成: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>'
        f'走査: {len(personas)} agents × {duration} step / seed={seed}<br>'
        f'run dir: {html.escape(str(run))}<br>'
        f'メッセージ総件数: {len(msgs)}'
        '</div>'
        '</section>'
    )

    # Persona table
    persona_html = (
        '<section class="page-break">'
        '<h1>登場人物一覧</h1>'
        f'{render_persona_table(personas)}'
        '</section>'
    )

    # Speech summary
    speech_summary_html = (
        '<section>'
        '<h2>発話統計サマリ</h2>'
        '<div class="note">各 agent の sent (発話数) / silent (空文字応答数) / 主な相手。'
        'silent が 0 の agent は「沈黙が valid」プロンプトを尊重していない可能性がある。</div>'
        f'{render_speech_summary(personas, msgs)}'
        '</section>'
    )

    # Interactive timeline (optional, swallow errors)
    timeline_html = ""
    try:
        ts = sim.get("time_scale", {})
        mps = int(ts.get("step_duration_minutes", 2) or 2)
        start = str(ts.get("start_time", "09:00"))
        try:
            sh, sm = start.split(":")
            sh, sm = int(sh), int(sm)
        except Exception:
            sh, sm = 9, 0
        timeline_html = build_phaseB_interactive_timeline(
            data, personas, int(duration) if str(duration).isdigit() else 30,
            start_hour=sh, mins_per_step=mps,
            phase_id="phaseA", start_minute=sm,
        )
        timeline_html = (
            '<section class="page-break">'
            '<h1>Phase A タイムライン</h1>'
            '<div class="note">'
            '<span style="color:#888">●</span> 思考のみ &nbsp;'
            '<span style="color:#ff9f43">●</span> 発話 &nbsp;'
            '<span style="color:#4ea8ff">青</span>=男 / '
            '<span style="color:#e85a9b">桃</span>=女 / '
            '<span style="color:#f2d23f">黄</span>=その他'
            '</div>'
            f'{timeline_html}'
            '</section>'
        )
    except Exception as e:
        timeline_html = f'<section><p style="color:#888">(タイムライン生成失敗: {html.escape(str(e))})</p></section>'

    # All messages per agent (重複ループ可視化用)
    msgs_per_agent_html = (
        '<section class="page-break">'
        '<h1>各 agent の発話全件 (step 順)</h1>'
        '<div class="note">同じ相手に同じテーマを繰り返してないかを目視チェックする用。silent (空文字) の応答は灰色斜体。</div>'
        f'{render_messages_per_agent(personas, msgs)}'
        '</section>'
    )

    # 全 step ログ (memory+reasoning+発話+受信)
    full_log_html = (
        '<section class="page-break">'
        '<h1>各 agent の全 step ログ (memory + reasoning + 発話 + 受信)</h1>'
        '<div class="note">memory 圧縮の archive 出力や、相手別過去発話の可視化を含む全データ。</div>'
    )
    id_to_name = {a["id"]: a.get("name", f"#{a['id']}") for a in personas}
    parts = []
    for p in personas:
        try:
            rows = build_full_history_rows(data, p["id"], id_to_name=id_to_name)
        except Exception:
            rows = []
        parts.append(
            f'<div class="msg-block"><div class="agent-name">{html.escape(p.get("name","?"))}</div>'
            f'{render_history_details(rows, show_movement=False)}'
            f'</div>'
        )
    full_log_html += "\n".join(parts) + "</section>"

    # API meta
    api_section = (
        '<section class="page-break">'
        '<h1>API消費・所要時間</h1>'
        f'{api_table_html}'
        f'<div class="api-meta" style="margin-top:0.6em">'
        f'<dl><dt>生成日時</dt><dd>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</dd></dl>'
        f'</div>'
        '</section>'
    )

    body = (
        HEAD.format(title=html.escape(title))
        + cover + compare_html + persona_html + speech_summary_html + timeline_html
        + msgs_per_agent_html + full_log_html + api_section
        + FOOT
    )

    out_path = Path(args.out) if args.out else (run / f"v3_report{suffix}.html")
    out_path.write_text(body, encoding="utf-8")
    print(f"[ok] wrote: {out_path} ({len(body)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
