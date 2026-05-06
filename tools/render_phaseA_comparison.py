"""【Phase A 限定】 baseline20_v2 (注入なし) vs Phase A (現場知覚情報を system_prompt に注入) を比較するレポート。

3 variant それぞれで A条件 (触媒なし) を 2 通り走らせた結果を並べる:
- baseline:  config に context_injection=disabled
- phaseA:    config に context_injection=enabled (docs/shinagawa_context.md)

Usage:
  python tools/render_phaseA_comparison.py \
    --base-elem  simulations/<...> --base-elem-log _v2_elem.log \
    --base-high  simulations/<...> --base-high-log _v2_high.log \
    --base-adult simulations/<...> --base-adult-log _v2_adult.log \
    --pa-elem  simulations/<...>  --pa-elem-log  _phaseA_elem.log \
    --pa-high  simulations/<...>  --pa-high-log  _phaseA_high.log \
    --pa-adult simulations/<...>  --pa-adult-log _phaseA_adult.log \
    --out simulations/<dir>/phaseA_comparison_v1.html
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

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from render_ab_comparison import (  # noqa: E402
    load_run, get_catalyst_info, is_participant,
    parse_sim_log, parse_logbuf, estimate_cost_usd, fmt_duration,
    USD_TO_JPY, _strip_prompt_leak,
)
from render_baseline_overview import (  # noqa: E402
    stats_for_run, sample_messages, render_field_questions,
    render_team_visions, extract_team_visions, render_sample_messages,
)


VARIANT_LABELS = {
    "elementary": ("小学生", "🎈"),
    "high":       ("高校生", "🎒"),
    "adult":      ("社会人", "💼"),
}


def call_gemini(client, sys_p, user_p, max_tokens=2500):
    try:
        return client.generate(sys_p, user_p, temperature=0.4, max_tokens=max_tokens)
    except Exception as e:
        return f"(LLM error: {e})"


SYSTEM_PA_SYNTH = (
    "ある世代 (variant) で、同じ問い・同じ参加者・同じ seed で **コンテキスト注入の有無** だけ変えて議論シミュレーションを回した結果を比較する。\n"
    "- baseline (注入なし): 議論対象のまち = 品川 という設定 + 7項目の要約事前情報のみ与えられた状態。"
    "LLM の事前学習知識 (品川=オフィス街/再開発、というステレオタイプ的な像) に基づいて議論する想定。\n"
    "- Phase A (注入あり): baseline に加えて、品川駅前を実際に歩き回って見える光景・動線・雰囲気を事実ベースで記述した詳細ガイド (約4000文字) を system_prompt に注入。"
    "**LLM事前知識を裏切る具体情報** (例: 駅前に屠殺場が残る、ペンシルビル街、京急改札の狭苦しさ、坂のキツさ、隠れた階段) が含まれる。\n\n"
    "観察したいのは、**コンテキスト注入が議論をどう変えたか**:\n"
    "- 議論の **具体性** (固有名詞・物理的特徴・現場の動線への言及が増えたか)\n"
    "- LLM事前知識ベースの **ステレオタイプ的な議論** (「再開発で便利になった」「効率重視のオフィス街」) の比重が下がったか\n"
    "- 注入だけが効いたのか / 全く変わらなかったのか / 想定と違う方向に効いたのか\n\n"
    "出力フォーマット (前置き禁止、いきなり最初の見出しから):\n\n"
    "### baseline と Phase A の決定的な違い (3変種共通の傾向)\n"
    "3-5文。**注入の有無で何が変わったか** をデータに即して書く。変わらなかったらそう書く。\n\n"
    "### 世代ごとの変化\n"
    "小学生 / 高校生 / 社会人 それぞれで baseline → Phase A で何が変わったか 各3-5文。"
    "**注入された具体情報 (屠殺場、ペンシルビル街、坂、隠れた階段、京急改札の狭さ など) を引いた発言が増えたか**、その世代の語彙範囲で取り込まれているかを観察。\n\n"
    "### LLM事前知識を裏切った具体情報の取り込み\n"
    "注入された情報のうち、特に **LLM事前知識では出にくい (品川=オフィス街と思われがち)** 情報がどれだけ拾われたか。"
    "屠殺場・歴史的にエリアが下に見られてきた経緯・港南ペンシルビル街・京急改札の狭さ・柘榴坂の急さ・隠れた階段などのキーワード言及をデータから読み取って書く。\n\n"
    "### 印象的な発話 (注入で変わったもの、各 variant 1-2件)\n"
    "Phase A で出てきた、baseline では出にくいはずの具体的な発話を引用。\n\n"
    "重要: 「研究員として〜」のような前置きや締めは絶対書かない。## や ### の見出しから直接本文に入る。"
)


def build_synth_user(base_runs: dict, pa_runs: dict) -> str:
    blocks = []
    for vkey, label in [("elementary","小学生"), ("high","高校生"), ("adult","社会人")]:
        b = base_runs[vkey]; p = pa_runs[vkey]
        sb = stats_for_run(b); sp = stats_for_run(p)
        blocks.append(f"=== {label} ===")
        blocks.append(f"baseline: msgs={sb['n_msgs']}, silent={sb['n_silent']}, 中央値発話/人={sb['median_per_p']}")
        blocks.append(f"Phase A:  msgs={sp['n_msgs']}, silent={sp['n_silent']}, 中央値発話/人={sp['median_per_p']}")
        # サンプル発話
        nm_b = {pp["id"]: pp["name"] for pp in b["personas"]}
        nm_p = {pp["id"]: pp["name"] for pp in p["personas"]}
        blocks.append("\n[baseline 会話サンプル]")
        for m in sample_messages(b, k=8):
            blocks.append(f"  [step{m.get('step'):03d}] {nm_b.get(m['from'],'?')}→{nm_b.get(m.get('to'),'?')}: {(m.get('message') or '')[:140]}")
        blocks.append("\n[Phase A 会話サンプル]")
        for m in sample_messages(p, k=8):
            blocks.append(f"  [step{m.get('step'):03d}] {nm_p.get(m['from'],'?')}→{nm_p.get(m.get('to'),'?')}: {(m.get('message') or '')[:140]}")
        blocks.append("")
    return "\n".join(blocks) + "\n\n上記のデータから、上記フォーマット通り比較レポートを書け。"


def keyword_count(data: dict, keywords: list[str]) -> dict[str, int]:
    """全 messages.jsonl 内でキーワード出現回数を集計。"""
    counts = collections.Counter()
    text = "\n".join(m.get("message") or "" for m in data["msgs"])
    for k in keywords:
        counts[k] = text.count(k)
    return dict(counts)


CONTEXT_INJECTED_KEYWORDS = [
    "屠殺場", "食肉市場",
    "ペンシルビル",
    "柘榴坂", "二本榎",
    "京急", "改札",
    "アトレ", "グランドコモンズ", "インターシティ", "シーズンテラス", "アレア",
    "ソニー", "コクヨ",
    "芝浦中央公園", "バラ園", "高輪の森",
    "プリンスホテル", "ウィング高輪", "水族館",
    "歴史的", "下に見られ", "閉鎖的",
    "国道15号", "第一京浜", "環状2号",
    "歩道橋", "デッキ",
]


def render_keyword_table(base: dict, pa: dict, label: str) -> str:
    b_counts = keyword_count(base, CONTEXT_INJECTED_KEYWORDS)
    p_counts = keyword_count(pa,   CONTEXT_INJECTED_KEYWORDS)
    rows = []
    for k in CONTEXT_INJECTED_KEYWORDS:
        b = b_counts.get(k, 0); p = p_counts.get(k, 0)
        delta = p - b
        if b == 0 and p == 0:
            continue  # 全く出ないキーワードは省く
        cls = ""
        if delta > 0: cls = "delta-pos"
        elif delta < 0: cls = "delta-neg"
        rows.append(
            f'<tr><th>{html.escape(k)}</th>'
            f'<td>{b}</td><td>{p}</td>'
            f'<td class="{cls}">{("+" if delta>0 else "")}{delta}</td></tr>'
        )
    if not rows:
        return f'<p style="color:#888">({label}: 注入キーワードの言及はどちらも検出されず)</p>'
    return (
        f'<table class="kw-table">'
        f'<thead><tr><th>キーワード ({label})</th><th>baseline</th><th>Phase A</th><th>差</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


HTML_TPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Phase A: コンテキスト注入比較レポート</title>
<style>
body {{ font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1.4em; color: #222; line-height: 1.65; }}
h1 {{ font-size: 1.7em; border-bottom: 3px solid #4a8ad8; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.3em; margin-top: 1.6em; border-left: 5px solid #4a8ad8; padding-left: 0.6em; }}
h3 {{ font-size: 1.05em; margin-top: 1em; }}
.subtitle {{ color: #777; font-size: 0.85em; font-weight: normal; }}
.synthesis {{ background: #fff7e6; padding: 1.2em 1.6em; border-radius: 10px; border: 1px solid #ddc88c; white-space: pre-wrap; }}
.stats-table, .kw-table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.92em; }}
.stats-table th, .stats-table td, .kw-table th, .kw-table td {{ border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: right; }}
.stats-table th:first-child, .stats-table tbody th, .kw-table th:first-child, .kw-table tbody th {{ text-align: left; background: #fafafa; }}
.stats-table thead th, .kw-table thead th {{ background: #f0f0f0; }}
.delta-pos {{ color: #1a6a3a; font-weight: bold; }}
.delta-neg {{ color: #a4344a; }}
.var-section {{ border: 2px solid #ddd; border-radius: 10px; padding: 1em 1.4em; margin: 1.5em 0; background: #fafafa; }}
.var-section h2 {{ margin-top: 0; }}
.cmp-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8em; margin: 0.5em 0; }}
.cmp-col {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.6em 0.9em; }}
.cmp-col.base {{ border-left: 4px solid #4a8ad8; }}
.cmp-col.pa   {{ border-left: 4px solid #d8704a; }}
.cmp-col h4 {{ margin: 0 0 0.5em 0; font-size: 0.96em; }}
.msg {{ border-left: 3px solid #d8a040; background: #fff8ef; padding: 0.4em 0.8em; margin: 0.3em 0; font-size: 0.86em; border-radius: 4px; }}
.msg-meta {{ color: #888; font-size: 0.78em; margin-bottom: 0.2em; }}
.msg-body {{ color: #333; }}
.api-table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; margin: 0.5em 0; }}
.api-table th, .api-table td {{ border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: right; }}
.api-table th {{ background: #f0f0f0; font-weight: 500; }}
.api-table th:first-child, .api-table tbody th {{ text-align: left; background: #fafafa; }}
.meta {{ color: #aaa; font-size: 0.8em; margin-top: 3em; text-align: center; }}
</style></head><body>
<h1>Phase A: コンテキスト注入比較レポート <span class="subtitle">— baseline20_v2 (注入なし) vs Phase A (品川駅前ガイド注入)</span></h1>
<p style="color:#666">同一問い・同一参加者・同一 seed (42)・100step・3 variant×A条件。
唯一の差分は「品川駅前を歩いて見える事実ベース情報 (約4000字) を system_prompt に注入したかどうか」。</p>

<h2>定量比較 (3 variant × 2 条件)</h2>
{stats_table}

<h2>クロス比較 synthesis (Gemini)</h2>
<div class="synthesis">{synth_html}</div>

<h2>API消費・コスト</h2>
{api_section}

{var_sections}

<div class="meta">Generated by tools/render_phaseA_comparison.py</div>
</body></html>
"""


def render_stats_table(base_runs, pa_runs) -> str:
    rows = []
    for vkey, (label, _) in VARIANT_LABELS.items():
        sb = stats_for_run(base_runs[vkey])
        sp = stats_for_run(pa_runs[vkey])
        d_msgs = sp["n_msgs"] - sb["n_msgs"]
        d_silent = sp["n_silent"] - sb["n_silent"]
        rows.append(
            f'<tr><th>{label}</th>'
            f'<td>{sb["n_msgs"]}</td><td>{sp["n_msgs"]}</td>'
            f'<td class="{"delta-pos" if d_msgs>0 else ("delta-neg" if d_msgs<0 else "")}">{("+" if d_msgs>0 else "")}{d_msgs}</td>'
            f'<td>{sb["n_silent"]}</td><td>{sp["n_silent"]}</td>'
            f'<td class="{"delta-neg" if d_silent>0 else ("delta-pos" if d_silent<0 else "")}">{("+" if d_silent>0 else "")}{d_silent}</td>'
            f'<td>{sb["median_per_p"]} → {sp["median_per_p"]}</td>'
            f'</tr>'
        )
    return (
        f'<table class="stats-table"><thead>'
        f'<tr><th rowspan="2">variant</th>'
        f'<th colspan="3">msgs</th>'
        f'<th colspan="3">silent</th>'
        f'<th rowspan="2">発話/人 中央値</th></tr>'
        f'<tr><th>base</th><th>Phase A</th><th>Δ</th>'
        f'<th>base</th><th>Phase A</th><th>Δ</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    for variant in ["elem", "high", "adult"]:
        ap.add_argument(f"--base-{variant}", required=True)
        ap.add_argument(f"--base-{variant}-log", default=None)
        ap.add_argument(f"--pa-{variant}", required=True)
        ap.add_argument(f"--pa-{variant}-log", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_runs = {
        "elementary": load_run(Path(args.base_elem)),
        "high":       load_run(Path(args.base_high)),
        "adult":      load_run(Path(args.base_adult)),
    }
    pa_runs = {
        "elementary": load_run(Path(args.pa_elem)),
        "high":       load_run(Path(args.pa_high)),
        "adult":      load_run(Path(args.pa_adult)),
    }
    base_paths = {
        "elementary": Path(args.base_elem),
        "high":       Path(args.base_high),
        "adult":      Path(args.base_adult),
    }
    pa_paths = {
        "elementary": Path(args.pa_elem),
        "high":       Path(args.pa_high),
        "adult":      Path(args.pa_adult),
    }

    # Synthesis
    analysis_buf = io.StringIO()
    h = logging.StreamHandler(analysis_buf); h.setLevel(logging.INFO); h.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("llm_client_factory").addHandler(h)
    logging.getLogger("llm_client_factory").setLevel(logging.INFO)
    t0 = time.time()
    client = GeminiClient(model="gemini-3.1-flash-lite-preview", temperature=0.4, max_tokens=2800,
                          enable_cache=False, enable_structured_output=False)
    print("[info] generating Phase A vs baseline synthesis ...")
    user_p = build_synth_user(base_runs, pa_runs)
    synth = call_gemini(client, SYSTEM_PA_SYNTH, user_p, max_tokens=2800) or "(synthesis失敗)"
    synth = _strip_prompt_leak(synth)

    analysis_dur = int(time.time() - t0)
    analysis_stats = parse_logbuf(analysis_buf.getvalue())

    # API section
    api_rows = []
    total_cost = 0.0
    for label, log in [
        ("baseline 小学生", args.base_elem_log),
        ("baseline 高校生", args.base_high_log),
        ("baseline 社会人", args.base_adult_log),
        ("Phase A 小学生", args.pa_elem_log),
        ("Phase A 高校生", args.pa_high_log),
        ("Phase A 社会人", args.pa_adult_log),
    ]:
        st = parse_sim_log(Path(log)) if log else None
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

    # Per variant section
    var_section_html = []
    for vkey, (label, icon) in VARIANT_LABELS.items():
        b = base_runs[vkey]; p = pa_runs[vkey]
        kw_table = render_keyword_table(b, p, label)
        var_section_html.append(
            f'<div class="var-section">'
            f'<h2>{icon} {label} — baseline vs Phase A</h2>'
            f'<h3>注入キーワード言及量 (messages.jsonl 内)</h3>'
            f'<p style="color:#666;font-size:0.88em">注入された context.md にしか出てこないはずの具体名・地名 が、Phase A の発話で出現したかをカウント。0/0 のものは省略。</p>'
            f'{kw_table}'
            f'<h3>会話サンプル比較 (各12件、序盤/中盤/終盤分散・長文優先)</h3>'
            f'<div class="cmp-grid">'
            f'<div class="cmp-col base"><h4>baseline (注入なし)</h4>{render_sample_messages(b, k=12)}</div>'
            f'<div class="cmp-col pa"><h4>Phase A (注入あり)</h4>{render_sample_messages(p, k=12)}</div>'
            f'</div>'
            f'<h3>各人のフィールドワーク確認事項</h3>'
            f'<div class="cmp-grid">'
            f'<div class="cmp-col base"><h4>baseline</h4>{render_field_questions(base_paths[vkey])}</div>'
            f'<div class="cmp-col pa"><h4>Phase A</h4>{render_field_questions(pa_paths[vkey])}</div>'
            f'</div>'
            f'</div>'
        )

    out_html = HTML_TPL.format(
        stats_table=render_stats_table(base_runs, pa_runs),
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
