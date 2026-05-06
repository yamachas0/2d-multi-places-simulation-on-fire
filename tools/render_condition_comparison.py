"""【えびねこ評価軸 / 今回限定実装】 同一 variant・同一 town・同一 seed の異なる触媒条件を並列比較するレポート。

Usage:
  python tools/render_condition_comparison.py \
    --label-a "A: 触媒なし"      --run-a   simulations/<...> \
    --label-b1 "B: UMA"         --run-b1  simulations/<...> \
    --label-b2 "B: AIロボ"      --run-b2  simulations/<...> \
    --label-b3 "B: AI神"        --run-b3  simulations/<...> \
    --log-a ... --log-b1 ... --log-b2 ... --log-b3 ... \
    --out simulations/<...>/condition_comparison_v1.html
"""
from __future__ import annotations
import argparse
import collections
import html
import io
import json
import logging
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
    load_run, get_catalyst_info, is_participant,
    parse_sim_log, parse_logbuf, estimate_cost_usd, fmt_duration,
    TEMPERAMENT_LABELS, USD_TO_JPY, _strip_prompt_leak,
)
from render_baseline_overview import (  # noqa: E402
    stats_for_run, sample_messages, render_persona_pills,
    render_field_questions, render_team_visions,
    extract_team_visions, render_sample_messages,
)


AXES = ["A", "R", "E", "I", "P", "B"]
AXIS_FULL = {
    "A": "Awareness/気づき",
    "R": "Reflection/内省",
    "E": "Engagement/関与",
    "I": "Interaction/相互作用",
    "P": "Perspective Shift/視点変容",
    "B": "Behavior Change/行動変容",
}


def call_gemini(client, sys_p, user_p, max_tokens=2500):
    try:
        return client.generate(sys_p, user_p, temperature=0.4, max_tokens=max_tokens)
    except Exception as e:
        return f"(LLM error: {e})"


def load_tags(run_dir: Path) -> list[dict]:
    p = run_dir / "tagged_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def aggregate_axis(tags_list: list[dict]) -> dict:
    """全体 / per-agent / per-step バケット で 6軸タグ数を集計。"""
    overall = collections.Counter()
    per_agent = collections.defaultdict(lambda: collections.Counter())
    per_step_bucket = [collections.Counter() for _ in range(5)]  # 5 分割
    max_step = max((e.get("step", 0) for e in tags_list), default=1) or 1
    for e in tags_list:
        for t in e.get("tags") or []:
            overall[t] += 1
            per_agent[e.get("name", "?")][t] += 1
            bucket = min(4, int((e.get("step", 0) / max_step) * 5))
            per_step_bucket[bucket][t] += 1
    return {
        "overall": overall,
        "per_agent": dict(per_agent),
        "per_step_bucket": per_step_bucket,
        "n_tagged": sum(1 for e in tags_list if e.get("tags")),
        "n_total": len(tags_list),
    }


def render_overall_axis_table(agg_by_label: dict[str, dict]) -> str:
    """4条件 × 6軸 のクロス表。"""
    labels = list(agg_by_label.keys())
    rows = []
    for axis in AXES:
        row = [f"<th>{axis} <span style='color:#888'>{html.escape(AXIS_FULL[axis])}</span></th>"]
        # 各条件のカウント + 同じ axis 全体での割合
        for lbl in labels:
            agg = agg_by_label[lbl]
            n = agg["overall"].get(axis, 0)
            total = sum(agg["overall"].values()) or 1
            pct = n * 100 / total
            row.append(f"<td>{n} <span style='color:#888;font-size:0.85em'>({pct:.0f}%)</span></td>")
        rows.append("<tr>" + "".join(row) + "</tr>")
    head = "<tr><th>軸</th>" + "".join(f"<th>{html.escape(l)}</th>" for l in labels) + "</tr>"
    return f'<table class="stats-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def render_step_bucket_table(agg_by_label: dict[str, dict]) -> str:
    """各条件で step を5分割した時系列タグ推移。"""
    parts = []
    for lbl, agg in agg_by_label.items():
        rows = []
        for i, bucket in enumerate(agg["per_step_bucket"]):
            cells = "".join(f"<td>{bucket.get(t,0)}</td>" for t in AXES)
            seg_label = f"序盤" if i==0 else "中盤" if i==2 else "中-1" if i==1 else "中-3" if i==3 else "終盤"
            rows.append(f"<tr><th>{seg_label} (step bucket {i+1}/5)</th>{cells}</tr>")
        head = "<tr><th></th>" + "".join(f"<th>{a}</th>" for a in AXES) + "</tr>"
        parts.append(
            f'<h4>{html.escape(lbl)}</h4>'
            f'<table class="stats-table" style="font-size:0.88em"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
        )
    return "\n".join(parts)


def render_per_agent_axis(agg_by_label: dict[str, dict], common_names: list[str]) -> str:
    """各人 × 6軸 × 条件 のミニヒートマップ風 (横にラベル並べる)。"""
    labels = list(agg_by_label.keys())
    # まず各人のタグ合計が大きい人を上位に表示
    name_score = collections.Counter()
    for lbl, agg in agg_by_label.items():
        for nm, c in agg["per_agent"].items():
            name_score[nm] += sum(c.values())
    sorted_names = [n for n, _ in name_score.most_common()]
    # 共通の参加者だけ (触媒は除外する想定)
    sorted_names = [n for n in sorted_names if n in common_names]

    head = (
        "<tr><th rowspan='2'>名前</th>"
        + "".join(f"<th colspan='6' style='text-align:center'>{html.escape(l)}</th>" for l in labels)
        + "</tr><tr>"
        + "".join("".join(f"<th style='font-size:0.78em'>{a}</th>" for a in AXES) for _ in labels)
        + "</tr>"
    )
    rows = []
    for nm in sorted_names[:30]:  # 上位30人
        cells = []
        for lbl in labels:
            counts = agg_by_label[lbl]["per_agent"].get(nm, {})
            for a in AXES:
                v = counts.get(a, 0)
                color = "#fff" if v == 0 else f"rgba(74,138,216,{min(1.0, 0.15+v*0.15):.2f})"
                cells.append(f'<td style="background:{color};font-size:0.85em">{v if v else ""}</td>')
        rows.append(f"<tr><th>{html.escape(nm)}</th>" + "".join(cells) + "</tr>")
    return f'<table class="stats-table" style="font-size:0.85em"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def render_stats_overview(stats_by_label: dict[str, dict]) -> str:
    rows = []
    for lbl, s in stats_by_label.items():
        rows.append(
            f'<tr><th>{html.escape(lbl)}</th>'
            f'<td>{s["n_msgs"]}</td>'
            f'<td>{s["n_personas"]}</td>'
            f'<td>{s["n_silent"]}</td>'
            f'<td>{s["min_per_p"]} / {s["median_per_p"]} / {s["max_per_p"]}</td>'
            f'</tr>'
        )
    return f'<table class="stats-table"><thead><tr><th></th>' \
           f'<th>全メッセージ</th><th>参加者数</th><th>silent</th>' \
           f'<th>発話/人 (min/中央値/max)</th></tr></thead>' \
           f'<tbody>{"".join(rows)}</tbody></table>'


SYSTEM_COND_SYNTH = (
    "ある世代 (variant) で、同じ町・同じ問い・同じ参加者・同じ seed の下、**異なる触媒条件** で議論シミュレーションを回した結果を比較する。"
    "物語化や美化を一切しない。失敗していたら失敗と書く。\n\n"
    "## 6軸 (えびねこ評価軸)\n"
    "- A Awareness 気づき / R Reflection 内省 / E Engagement 関与 / I Interaction 相互作用 / P Perspective Shift 視点変容 / B Behavior Change 行動変容\n\n"
    "出力フォーマット (前置き禁止、いきなり最初の見出しから):\n\n"
    "### 共通の問い\n本実験で投げられた中心問いを1文で。\n\n"
    "### 各条件で何が起きたか (4条件 × 各3-4文)\n"
    "条件A (触媒なし) / B-UMA / B-AIロボ / B-AI神 それぞれで何が起きたか。**何を決めたか・未来像はどう描かれたか・誰が動いたか・誰が黙ったか** を率直に。\n\n"
    "### 6軸タグでみる条件間の差 (必須)\n"
    "提供された **6軸タグ集計** を参照して、各軸 (A/R/E/I/P/B) で **どの条件で多かったか・少なかったか** を具体的に書く。"
    "例:「B-AI神 で B(行動変容) タグが○件、A条件は○件 → 触媒の介入で行動変容が促されたとは言える/言えない」のように。"
    "**数字を使って** データに即して書く。\n\n"
    "### 触媒の質的な違い\n"
    "UMA (発話なし、ただ居る) / AIロボ (知識ゼロ、質問しまくる) / AI神 (全知全能、提案しまくる) の3触媒が場に与えた効果の違いを3-5文。"
    "とりわけ **参加者の主体性 (Engagement / Behavior Change) を奪うか引き上げるか** という観点で書く。\n\n"
    "### 印象的な発話 (各条件1件、引用形式)\n"
    "ログから直接引用できる『実際に起きた瞬間』を条件ごとに1件。創作禁止。\n\n"
    "重要: 「研究員として〜以下の通り要約します」のような前置きや締めは絶対書かない。## や ### の見出しから直接本文に入る。"
)


HTML_TPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{variant_label} 4条件比較 (触媒なし / UMA / AIロボ / AI神)</title>
<style>
body {{ font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1.4em; color: #222; line-height: 1.65; }}
h1 {{ font-size: 1.7em; border-bottom: 3px solid #4a8ad8; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.3em; margin-top: 1.6em; border-left: 5px solid #4a8ad8; padding-left: 0.6em; }}
h3 {{ font-size: 1.05em; margin-top: 1em; }}
h4 {{ font-size: 0.96em; margin-top: 0.6em; color: #555; }}
.subtitle {{ color: #777; font-size: 0.85em; font-weight: normal; }}
.synthesis {{ background: #fff7e6; padding: 1.2em 1.6em; border-radius: 10px; border: 1px solid #ddc88c; white-space: pre-wrap; }}
.stats-table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.92em; }}
.stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: right; }}
.stats-table th:first-child, .stats-table tbody th {{ text-align: left; background: #fafafa; }}
.stats-table thead th {{ background: #f0f0f0; }}
.cond-section {{ border: 2px solid #ddd; border-radius: 10px; padding: 1em 1.4em; margin: 1.2em 0; background: #fafafa; }}
.cond-section h2 {{ margin-top: 0; }}
.msg {{ border-left: 3px solid #d8a040; background: #fff8ef; padding: 0.4em 0.8em; margin: 0.3em 0; font-size: 0.88em; border-radius: 4px; }}
.msg-meta {{ color: #888; font-size: 0.78em; margin-bottom: 0.2em; }}
.msg-body {{ color: #333; }}
.vision {{ background: #f4f7fb; border-left: 4px solid #4a8ad8; padding: 0.5em 0.9em; margin: 0.4em 0; border-radius: 5px; font-size: 0.92em; }}
.vision-label {{ font-weight: bold; color: #2a4a7c; font-size: 0.92em; }}
.vision-members {{ font-size: 0.82em; color: #666; margin-bottom: 0.2em; }}
.vision-future {{ color: #222; }}
.vision-evidence {{ font-size: 0.78em; color: #888; font-style: italic; margin-top: 0.3em; padding-top: 0.3em; border-top: 1px dashed #ccc; }}
.fq-table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 0.3em 0 0.8em; }}
.fq-table th, .fq-table td {{ border: 1px solid #ddd; padding: 0.3em 0.6em; text-align: left; vertical-align: top; }}
.fq-table thead th {{ background: #f0f0f0; }}
.fq-table ul {{ margin: 0; padding-left: 1.2em; }}
.api-table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; margin: 0.5em 0; }}
.api-table th, .api-table td {{ border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: right; }}
.api-table th {{ background: #f0f0f0; font-weight: 500; }}
.api-table th:first-child, .api-table tbody th {{ text-align: left; background: #fafafa; }}
.meta {{ color: #aaa; font-size: 0.8em; margin-top: 3em; text-align: center; }}
</style></head><body>
<h1>{variant_label} 4条件比較 <span class="subtitle">— A: 触媒なし / B: UMA / B: AIロボ / B: AI神</span></h1>
<p style="color:#666">同一 variant・同一町 (品川)・同一問い・同一参加者・同一 seed (42) ・100step。
触媒だけ4条件で振り、その差を観察。</p>

<h2>定量比較</h2>
{stats_overview}

<h2>クロス条件 synthesis (Gemini)</h2>
<div class="synthesis">{synth_html}</div>

<h2>えびねこ評価軸 (A/R/E/I/P/B) 6軸タグ集計</h2>
<h3>条件 × 軸 のクロス集計</h3>
{axis_overall_table}
<h3>各条件の時系列推移 (step を5分割した bucket)</h3>
{axis_step_table}
<h3>各人の軸プロファイル (4条件横並び、上位30人)</h3>
<p style="color:#666;font-size:0.88em">同一参加者でも条件によって出る軸が変わる。色が濃いほど多い。</p>
{per_agent_axis_table}

<h2>API消費・コスト・所要時間</h2>
{api_section}

{cond_sections}

<div class="meta">Generated by tools/render_condition_comparison.py</div>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--log-a", default=None)
    ap.add_argument("--label-b1", required=True)
    ap.add_argument("--run-b1", required=True)
    ap.add_argument("--log-b1", default=None)
    ap.add_argument("--label-b2", required=True)
    ap.add_argument("--run-b2", required=True)
    ap.add_argument("--log-b2", default=None)
    ap.add_argument("--label-b3", required=True)
    ap.add_argument("--run-b3", required=True)
    ap.add_argument("--log-b3", default=None)
    ap.add_argument("--variant-label", default="小学生")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [
        (args.label_a,  Path(args.run_a),  args.log_a),
        (args.label_b1, Path(args.run_b1), args.log_b1),
        (args.label_b2, Path(args.run_b2), args.log_b2),
        (args.label_b3, Path(args.run_b3), args.log_b3),
    ]
    data_by_label = {lbl: load_run(rd) for lbl, rd, _ in runs}
    stats_by_label = {lbl: stats_for_run(d) for lbl, d in data_by_label.items()}
    tags_by_label = {lbl: load_tags(rd) for lbl, rd, _ in runs}
    agg_by_label = {lbl: aggregate_axis(tg) for lbl, tg in tags_by_label.items()}

    # 共通参加者名 (触媒以外、A run から取得)
    a_data = data_by_label[args.label_a]
    cat_a = get_catalyst_info(a_data)
    common_names = [p["name"] for p in a_data["personas"] if is_participant(p, cat_a["axis_id"])]

    # 分析側 LLM コール (synthesis + team_visions)
    analysis_buf = io.StringIO()
    h = logging.StreamHandler(analysis_buf); h.setLevel(logging.INFO); h.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("llm_client_factory").addHandler(h)
    logging.getLogger("llm_client_factory").setLevel(logging.INFO)
    t0 = time.time()

    client = GeminiClient(
        model="gemini-3.1-flash-lite-preview", temperature=0.4, max_tokens=2500,
        enable_cache=False, enable_structured_output=False,
    )

    print("[info] extracting team visions per condition ...")
    visions_by_label = {}
    for lbl, d in data_by_label.items():
        visions_by_label[lbl] = extract_team_visions(d, client, lbl)
        print(f"  [{lbl}] visions={len(visions_by_label[lbl])}")

    # synthesis prompt: 6軸集計を含めて渡す
    quant_lines = ["## 共通の問い (各 agent.current_goal より)"]
    p_goal = ""
    for p in a_data["personas"]:
        if is_participant(p, cat_a["axis_id"]):
            p_goal = (p.get("current_goal") or "")
            break
    quant_lines.append(p_goal)
    quant_lines.append("")
    quant_lines.append("## 全条件の数字概要")
    for lbl, s in stats_by_label.items():
        quant_lines.append(f"- {lbl}: msgs={s['n_msgs']}, silent={s['n_silent']}, 発話中央値/人={s['median_per_p']}")
    quant_lines.append("")
    quant_lines.append("## 6軸タグ集計 (条件 × 軸)")
    for lbl, agg in agg_by_label.items():
        cnt = agg["overall"]
        line = f"- {lbl}: " + ", ".join(f"{a}={cnt.get(a,0)}" for a in AXES) + f" (タグ付き entry {agg['n_tagged']}/{agg['n_total']})"
        quant_lines.append(line)
    quant_lines.append("")
    # 印象的な発話用にサンプル
    quant_lines.append("## 各条件の会話サンプル (10件、序盤/中盤/終盤分散)")
    for lbl, d in data_by_label.items():
        quant_lines.append(f"\n### {lbl}")
        nm_by_id = {p["id"]: p["name"] for p in d["personas"]}
        for m in sample_messages(d, k=10):
            fr = nm_by_id.get(m["from"], "?"); to = nm_by_id.get(m.get("to"), "?")
            quant_lines.append(f"  [step{m.get('step'):03d}] {fr}→{to}: {(m.get('message') or '')[:140]}")
    quant_lines.append("")
    quant_lines.append("## チーム未来像 (条件ごと)")
    for lbl, vs in visions_by_label.items():
        quant_lines.append(f"\n### {lbl}")
        for v in vs[:6]:
            quant_lines.append(f"- {v.get('team_label','?')} ({', '.join(v.get('members') or [])}): {v.get('future_vision','')}")

    print("[info] generating cross-condition synthesis ...")
    synth = call_gemini(client, SYSTEM_COND_SYNTH, "\n".join(quant_lines), max_tokens=2800) or "(synthesis失敗)"
    synth = _strip_prompt_leak(synth)

    analysis_dur = int(time.time() - t0)
    analysis_stats = parse_logbuf(analysis_buf.getvalue())

    # API消費
    sim_stats_list = []
    for lbl, _, log in runs:
        st = parse_sim_log(Path(log)) if log else None
        sim_stats_list.append((f"sim {lbl}", st))
    # 簡易 api section
    api_rows = []
    total_cost = 0.0
    for label, st in sim_stats_list:
        c = estimate_cost_usd(st) if st else 0.0
        total_cost += c
        if st:
            api_rows.append(f"<tr><th>{html.escape(label)}</th><td>{st.get('n_calls',0):,}</td><td>{st.get('input_tokens',0):,}</td><td>{st.get('output_tokens',0):,}</td><td>{fmt_duration(st.get('duration_sec',0))}</td><td>${c:.3f}</td></tr>")
    if analysis_stats:
        c = estimate_cost_usd(analysis_stats); total_cost += c
        api_rows.append(f"<tr><th>分析 (このレポート)</th><td>{analysis_stats.get('n_calls',0):,}</td><td>{analysis_stats.get('input_tokens',0):,}</td><td>{analysis_stats.get('output_tokens',0):,}</td><td>{fmt_duration(analysis_dur)}</td><td>${c:.3f}</td></tr>")
    api_rows.append(f"<tr style='border-top:2px solid #888;font-weight:bold;background:#f8f6f0'><th>合計</th><td colspan='4' style='text-align:right'></td><td>${total_cost:.3f} (≒¥{total_cost*USD_TO_JPY:.0f})</td></tr>")
    api_section = (
        f'<table class="api-table"><thead><tr><th></th><th>API呼び出し</th><th>入力token</th><th>出力token</th><th>所要時間</th><th>コスト</th></tr></thead>'
        f'<tbody>{"".join(api_rows)}</tbody></table>'
    )

    # 各条件のセクション (参加者表は省略=共通だから、サンプル発話/未来像/FQ)
    cond_sections = []
    for lbl, rd, _ in runs:
        d = data_by_label[lbl]
        s = stats_by_label[lbl]
        meta = d.get("cfg",{}).get("metadata",{})
        catalyst_name = meta.get("catalyst_name", "—")
        cond_sections.append(
            f'<div class="cond-section">'
            f'<h2>{html.escape(lbl)} (触媒={html.escape(catalyst_name)})</h2>'
            f'<p>msgs={s["n_msgs"]} / silent={s["n_silent"]}人 / 発話中央値={s["median_per_p"]}</p>'
            f'<h3>各チームが描いた未来像</h3>'
            f'{render_team_visions(visions_by_label[lbl], lbl)}'
            f'<h3>各人のフィールドワーク確認事項</h3>'
            f'{render_field_questions(rd)}'
            f'<h3>会話サンプル (10件、序盤/中盤/終盤分散・長文優先)</h3>'
            f'{render_sample_messages(d, k=10)}'
            f'</div>'
        )

    out_html = HTML_TPL.format(
        variant_label=html.escape(args.variant_label),
        stats_overview=render_stats_overview(stats_by_label),
        synth_html=html.escape(synth).replace("\n", "<br>"),
        axis_overall_table=render_overall_axis_table(agg_by_label),
        axis_step_table=render_step_bucket_table(agg_by_label),
        per_agent_axis_table=render_per_agent_axis(agg_by_label, common_names),
        api_section=api_section,
        cond_sections="\n".join(cond_sections),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
