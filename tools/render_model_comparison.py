"""Compare two FW Phase B sim runs (different LLM models) and emit a side-by-side HTML report.

Usage:
  python tools/render_model_comparison.py <run_a> <run_b> --classroom-run <c> \
    --log-a <log_a> --log-b <log_b> --label-a "..." --label-b "..." [--out compare.html]
"""
from __future__ import annotations
import argparse, datetime, html, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from render_phaseB_report import (  # noqa: E402
    load_run, load_handoffs, per_agent_history,
    SYSTEM_ANALYSIS, build_phaseB_prompt, call_gemini_md,
    parse_sim_log, estimate_cost_usd, fmt_duration, _temp_str,
    extract_highlight_candidates, select_highlights, render_md_to_html,
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, PRICE_CACHE_PER_M, USD_TO_JPY,
)
from llm_client_factory import GeminiClient  # noqa: E402

# 価格テーブル (per 1M tokens, USD) - GA 公開単価ベース。preview は注釈で警告。
PRICES = {
    "gemini-3.1-flash-lite-preview": {"input": 0.10, "output": 0.40, "cache": 0.025, "is_preview": True},
    "gemini-2.5-flash-lite":         {"input": 0.10, "output": 0.40, "cache": 0.025, "is_preview": False},
    "gemini-2.5-flash":              {"input": 0.30, "output": 1.20, "cache": 0.075, "is_preview": False},
    "gemini-2.0-flash":              {"input": 0.10, "output": 0.40, "cache": 0.025, "is_preview": False},
    "gemini-1.5-flash":              {"input": 0.075, "output": 0.30, "cache": 0.01875, "is_preview": False},
    "gemini-1.5-flash-8b":           {"input": 0.0375, "output": 0.15, "cache": 0.009375, "is_preview": False},
}


def _model_of_run(run_dir: Path) -> str:
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return (cfg.get("llm", {}) or {}).get("model", "?")


def _cost_with_prices(stats, prices):
    if not stats or not prices:
        return 0.0
    return (
        stats.get("uncached_input", 0) * prices["input"] / 1_000_000
        + stats.get("cache_read_tokens", 0) * prices["cache"] / 1_000_000
        + stats.get("output_tokens", 0) * prices["output"] / 1_000_000
    )


def render_compare_table(runs):
    """runs: list of dict {label, model, stats, prices}."""
    rows = []
    for r in runs:
        s = r["stats"] or {}
        p = r["prices"]
        cost = _cost_with_prices(s, p) if p else 0.0
        cost_jpy = cost * USD_TO_JPY
        cache_rate = (s.get("cache_read_tokens", 0) / s.get("input_tokens", 1) * 100) if s.get("input_tokens") else 0
        preview_note = " (preview)" if (p and p.get("is_preview")) else ""
        rows.append(
            f'<tr><th>{html.escape(r["label"])}<br><span style="font-weight:normal;font-size:0.8em;color:#888">{html.escape(r["model"])}{preview_note}</span></th>'
            f'<td>{s.get("n_calls", 0):,}</td>'
            f'<td>{s.get("input_tokens", 0):,}</td>'
            f'<td>{s.get("output_tokens", 0):,}</td>'
            f'<td>{cache_rate:.1f}%</td>'
            f'<td>{fmt_duration(s.get("duration_sec", 0))}</td>'
            f'<td>${cost:.4f} (≒¥{cost_jpy:.0f})</td></tr>'
        )
    return (
        '<table class="api-table">'
        '<thead><tr><th></th><th>API呼出</th><th>入力token</th><th>出力token</th>'
        '<th>cache hit率</th><th>所要時間</th><th>推定コスト</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


SYSTEM_ANALYSIS_SHORT = (
    "あなたは観察者。フィールドワーク (FW) 記録を読んで、参加者 1 人の変化を1段落と1行で要約する。"
    "前提: 午前に座学で品川の未来像を議論、午後 13時から品川駅前を歩いて未来像をアップデートする。\n\n"
    "**応答ルール:** 日本語のみ・前置き禁止・出力は次のフォーマットで開始：\n\n"
    "段落: <FWでの変化を 2-3 文で>\n"
    "一行: <この人の変化を 1 文で>"
)


def call_short(client, persona, handoff, fq, thoughts, sent, recv) -> tuple[str, str]:
    user = build_phaseB_prompt(persona, handoff, fq, thoughts, sent, recv)
    resp = client.generate(SYSTEM_ANALYSIS_SHORT, user, temperature=0.4, max_tokens=600)
    if not resp:
        return ("", "")
    para_match = re.search(r"段落[::]\s*(.+?)(?=\n+一行|$)", resp, flags=re.DOTALL)
    one_match = re.search(r"一行[::]\s*(.+)$", resp, flags=re.DOTALL)
    para = para_match.group(1).strip() if para_match else ""
    one = one_match.group(1).strip() if one_match else ""
    return para, one


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=str)
    ap.add_argument("run_b", type=str)
    ap.add_argument("--classroom-run", required=True)
    ap.add_argument("--log-a", required=True)
    ap.add_argument("--log-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run_a_dir = Path(args.run_a)
    run_b_dir = Path(args.run_b)
    A = load_run(run_a_dir)
    B = load_run(run_b_dir)
    handoffs = load_handoffs(Path(args.classroom_run))

    model_a = _model_of_run(run_a_dir)
    model_b = _model_of_run(run_b_dir)
    prices_a = PRICES.get(model_a)
    prices_b = PRICES.get(model_b)
    stats_a = parse_sim_log(args.log_a)
    stats_b = parse_sim_log(args.log_b)

    participants_a = [p for p in A["personas"] if p.get("axis_id") and p.get("axis_id") not in ("Sato", "AIRobo", "AIGod", "UMA")]
    participants_b = [p for p in B["personas"] if p.get("axis_id") and p.get("axis_id") not in ("Sato", "AIRobo", "AIGod", "UMA")]
    by_axis_a = {p["axis_id"]: p for p in participants_a}
    by_axis_b = {p["axis_id"]: p for p in participants_b}
    common_axes = [ax for ax in by_axis_a.keys() if ax in by_axis_b]
    print(f"[info] common participants: {len(common_axes)}")

    client = GeminiClient(enable_structured_output=False, enable_cache=False)

    def analyze(p, run_data):
        ax = p.get("axis_id")
        thoughts, sent, recv = per_agent_history(run_data, p["id"])
        return call_short(client, p, handoffs["handoff_by_axis"].get(ax),
                          handoffs["fq_by_axis"].get(ax), thoughts, sent, recv)

    print("[info] analyzing both runs in parallel...")
    t0 = time.time()
    rows: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for ax in common_axes:
            futs[pool.submit(analyze, by_axis_a[ax], A)] = (ax, "a")
            futs[pool.submit(analyze, by_axis_b[ax], B)] = (ax, "b")
        for f in as_completed(futs):
            ax, side = futs[f]
            try:
                para, one = f.result()
            except Exception as e:
                para, one = "", f"(分析失敗: {e})"
            rows.setdefault(ax, {})[side] = {"para": para, "one": one}
    analysis_dur = int(time.time() - t0)
    print(f"[info] analysis done: {analysis_dur}s")

    # ハイライトも両方
    hl_a = select_highlights(client, extract_highlight_candidates(A, max_pre=30)) or []
    hl_b = select_highlights(client, extract_highlight_candidates(B, max_pre=30)) or []

    # Render
    cmp_table = render_compare_table([
        {"label": args.label_a, "model": model_a, "stats": stats_a, "prices": prices_a},
        {"label": args.label_b, "model": model_b, "stats": stats_b, "prices": prices_b},
    ])

    # 各 agent 並列カード
    agent_rows = []
    for ax in sorted(common_axes):
        pa = by_axis_a[ax]
        rb = rows.get(ax, {})
        ra = rb.get("a", {})
        rb_ = rb.get("b", {})
        agent_rows.append(
            f'<div class="agent-row">'
            f'<div class="agent-head">{html.escape(pa.get("name", "?"))}（{pa.get("age", "?")}歳・{pa.get("gender", "")}） / axis: {ax} / 気質: {html.escape(_temp_str(pa))}</div>'
            f'<div class="cmp-grid">'
            f'<div class="cmp-cell a"><div class="m-label">{html.escape(args.label_a)}</div>'
            f'<p>{html.escape(ra.get("para", "") or "(なし)")}</p>'
            f'<p class="oneliner"><strong>{html.escape(ra.get("one", "") or "")}</strong></p></div>'
            f'<div class="cmp-cell b"><div class="m-label">{html.escape(args.label_b)}</div>'
            f'<p>{html.escape(rb_.get("para", "") or "(なし)")}</p>'
            f'<p class="oneliner"><strong>{html.escape(rb_.get("one", "") or "")}</strong></p></div>'
            f'</div></div>'
        )
    agents_html = "\n".join(agent_rows)

    # ハイライト並列
    def hl_block(hls):
        if not hls:
            return '<div class="empty">(なし)</div>'
        return "\n".join(
            f'<div class="hl-item"><div class="hl-step">step {html.escape(str(h.get("step","?")))} / {html.escape(str(h.get("actors","")))}</div>'
            f'<div class="hl-body">{html.escape(str(h.get("summary","")))}</div></div>'
            for h in hls
        )

    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="color-scheme" content="dark">
<title>FW phaseB モデル比較レポート</title>
<style>
:root {{ color-scheme: dark; }}
body {{ font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
        max-width: 1400px; margin: 0 auto; padding: 1.5em 1em 4em; line-height: 1.7;
        color: #e8e8e8; background: #15171a; }}
h1 {{ font-size: 1.7em; border-bottom: 2px solid #4a8ad8; padding-bottom: 0.3em;
      color: #f5f5ef; margin-top: 0; }}
h2 {{ font-size: 1.2em; color: #cfd0c8; margin-top: 2em; border-left: 4px solid #4a8ad8;
      padding-left: 0.6em; }}
.api-table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; }}
.api-table th, .api-table td {{ padding: 0.5em 0.7em; border: 1px solid #2c2c2c; text-align: right; color: #d8d8d8; }}
.api-table th {{ background: #25282d; color: #cfd0c8; text-align: left; }}
.api-table td:first-child, .api-table th:first-child {{ text-align: left; }}
.note {{ font-size: 0.85em; color: #888; margin: 0.6em 0; }}
.agent-row {{ margin-bottom: 1.5em; padding: 0.8em 1em; background: #1d2025; border-radius: 8px;
              border: 1px solid #2c2c2c; }}
.agent-head {{ font-weight: 600; color: #ffd85f; margin-bottom: 0.6em; }}
.cmp-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8em; }}
.cmp-cell {{ padding: 0.7em 0.9em; border-radius: 6px; border: 1px solid #2c2c2c; background: #15171a; }}
.cmp-cell.a {{ border-left: 4px solid #4ea8ff; }}
.cmp-cell.b {{ border-left: 4px solid #66ccee; }}
.m-label {{ font-size: 0.8em; color: #aaa; margin-bottom: 0.3em; letter-spacing: 0.05em; }}
.cmp-cell p {{ margin: 0.3em 0; color: #d8d8d8; }}
.cmp-cell .oneliner {{ margin-top: 0.4em; padding-top: 0.4em; border-top: 1px dotted #444; color: #ffd85f; }}
.hl-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }}
.hl-side {{ background: #1d2025; padding: 0.8em 1em; border-radius: 8px; border: 1px solid #2c2c2c; }}
.hl-item {{ padding: 0.5em 0; border-bottom: 1px dotted #2c2c2c; }}
.hl-item:last-child {{ border-bottom: none; }}
.hl-step {{ font-size: 0.82em; color: #a0a0a0; }}
.hl-body {{ color: #d8d8d8; }}
.empty {{ color: #555; font-style: italic; }}
</style></head><body>

<h1>FW phaseB モデル比較レポート <span style="font-size:0.55em;color:#888;font-weight:normal">{html.escape(args.label_a)} vs {html.escape(args.label_b)}</span></h1>

<h2>API消費・コスト比較</h2>
{cmp_table}
<p class="note">※ preview は表示単価が公開価格と乖離する可能性あり (実請求が見積もりの 3-4倍 になった事例あり)。"GA" 表記の単価は公開固定。</p>

<h2>各参加者の言語化分析（両モデル並列）</h2>
{agents_html}

<h2>特徴的な出来事タイムライン（両モデル並列）</h2>
<div class="hl-grid">
<div class="hl-side"><div class="m-label">{html.escape(args.label_a)}</div>{hl_block(hl_a)}</div>
<div class="hl-side"><div class="m-label">{html.escape(args.label_b)}</div>{hl_block(hl_b)}</div>
</div>

<p class="note" style="margin-top:2em">レポート生成: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} / 共通参加者: {len(common_axes)}人 / 分析所要 {fmt_duration(analysis_dur)}</p>

</body></html>
"""
    out_path = Path(args.out) if args.out else (run_b_dir / "model_comparison.html")
    out_path.write_text(body, encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
