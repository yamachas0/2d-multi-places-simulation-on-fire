"""V3 統合レポート: Phase A 教室シミュ概要 + Phase B FW シミュ概要 + Phase C アンケート結果。

PDF化を見越して @page 指定 + page-break で章ごとの改ページを入れる。

Usage:
  python tools/render_v3_report.py \
    --run-a <phaseA_run> --run-b <phaseB_run> \
    [--survey <survey_responses.jsonl>] [--out <html>]
"""
from __future__ import annotations
import argparse, datetime, html, json, re, statistics, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402

# phaseB_report の関数を流用
from render_phaseB_report import (  # noqa: E402
    load_run, load_handoffs, per_agent_history, build_full_history_rows,
    SYSTEM_ANALYSIS, SYSTEM_ANALYSIS_HOST,
    build_phaseB_prompt, build_phaseB_host_prompt,
    call_gemini_md, render_md_to_html,
    extract_highlight_candidates, select_highlights, render_highlights_section,
    render_history_details, render_persona_card, render_analysis_card,
    build_phaseB_interactive_timeline, build_phaseB_timeline_svg, _temp_str, _gender_jp,
    parse_sim_log, render_api_table, fmt_duration,
)


# ---------- HTML head/foot (印刷PDF対応) ----------

HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="color-scheme" content="dark">
<title>{title}</title>
<style>
/* スライド 16:9 ワイド (1920x1080 比率, 13.33in x 7.5in)。html_to_pdf.py の slide16x9 と整合 */
@page {{ size: 13.33in 7.5in; margin: 10mm; }}
:root {{ color-scheme: dark; }}
body {{ font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
        max-width: 1200px; margin: 0 auto; padding: 1.2em 1em 3em; line-height: 1.6;
        color: #e8e8e8; background: #15171a; font-size: 14px; }}
section {{ margin-top: 2em; }}
section.page-break {{ page-break-before: always; }}
h1, h2, h3, h4 {{ page-break-after: avoid; break-after: avoid; }}
h1 {{ font-size: 1.55em; border-bottom: 2px solid #4a8ad8; padding-bottom: 0.3em; color: #f5f5ef; margin-top: 0; }}
h2 {{ font-size: 1.18em; color: #cfd0c8; border-left: 4px solid #4a8ad8; padding-left: 0.6em; margin-top: 1.4em; }}
h3 {{ font-size: 1.02em; color: #c8dbef; }}
h4 {{ font-size: 0.94em; color: #ffd85f; border-bottom: 1px dotted #444; padding-bottom: 0.2em; }}
p, li {{ orphans: 4; widows: 4; }}
/* 考察ブロックの分け方:
   小見出し単位で同ページに収まるなら繋げる方針 (auto)。
   無理に1ページずつ強制改ページしない (= 自然なフロー)。 */
.reflection-md h3 {{ page-break-before: auto; break-before: auto; }}
#section-ebineko h2,
#section-nextaction h2 {{ page-break-before: auto; break-before: auto; }}
/* Phase C アンケート (Q1〜Q10) は各 Q の前で改ページ */
#section-phase-c h4 {{ page-break-before: always; break-before: page; }}
#section-phase-c h4:first-of-type {{ page-break-before: auto; break-before: auto; }}
.exp-meta {{ background: #1d2025; padding: 0.8em 1.2em; border-radius: 8px; font-size: 0.95em; border: 1px solid #2c2c2c; }}
.exp-meta li {{ margin: 0.25em 0; }}
.intro-question {{ background: #28230f; padding: 1em 1.4em; border-radius: 8px; border: 1px solid #5a4f1c; margin: 1em 0; font-size: 1.05em; color: #f5e9b8; }}
.personas-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.7em; }}
.persona-card {{ border: 1px solid #2c2c2c; border-radius: 6px; padding: 0.6em 0.9em; background: #1d2025; font-size: 0.85em; line-height: 1.5; page-break-inside: avoid; }}
.persona-card.male   {{ border-left: 4px solid #4ea8ff; }}
.persona-card.female {{ border-left: 4px solid #e85a9b; }}
.persona-card.host   {{ border-left: 4px solid #ffd85f; background: #232014; }}
.persona-card .name {{ font-weight: 600; color: #f5f5ef; }}
.persona-card .meta {{ color: #888; font-size: 0.85em; }}
.analysis-card {{ border: 1px solid #2c2c2c; border-radius: 8px; padding: 1em 1.3em; background: #1d2025; margin-bottom: 1em; page-break-inside: avoid; }}
.analysis-card .agent-name {{ font-weight: 600; color: #ffd85f; font-size: 1.05em; margin-bottom: 0.3em; }}
.analysis-card .agent-meta {{ color: #888; font-size: 0.82em; margin-bottom: 0.6em; }}
.analysis-card p {{ margin: 0.4em 0; color: #d8d8d8; }}
.analysis-card.male   {{ border-left: 4px solid #4ea8ff; }}
.analysis-card.female {{ border-left: 4px solid #e85a9b; }}
.analysis-card.host   {{ border-left: 4px solid #ffd85f; }}
.empty {{ color: #555; font-style: italic; }}
.api-meta {{ background: #1d2025; border: 1px solid #2c2c2c; border-radius: 8px; padding: 1em 1.2em; font-size: 0.92em; }}
.api-meta dl {{ margin: 0; display: grid; grid-template-columns: max-content 1fr; column-gap: 1em; row-gap: 0.3em; }}
.api-meta dt {{ color: #888; }}
.api-meta dd {{ margin: 0; color: #e8e8e8; }}
.api-table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; }}
.api-table th, .api-table td {{ padding: 0.5em 0.7em; border: 1px solid #2c2c2c; text-align: right; color: #d8d8d8; }}
.api-table th {{ background: #25282d; color: #cfd0c8; text-align: left; }}
.api-table td:first-child, .api-table th:first-child {{ text-align: left; }}
table.full-history {{ width: 100%; border-collapse: collapse; font-size: 0.78em; }}
table.full-history th {{ background: #25282d; color: #cfd0c8; padding: 0.4em 0.5em; border: 1px solid #2c2c2c; text-align: left; position: sticky; top: 0; }}
table.full-history td {{ padding: 0.4em 0.5em; border: 1px solid #2c2c2c; vertical-align: top; color: #d8d8d8; }}
table.full-history .t-step {{ width: 3em; color: #ffd85f; font-family: monospace; }}
table.full-history .t-time {{ width: 4em; color: #888; font-family: monospace; }}
table.full-history .t-loc  {{ width: 11em; color: #b0c4de; font-size: 0.92em; }}
table.full-history .t-comp {{ width: 7em; color: #cdb8e8; font-size: 0.85em; }}
table.full-history .t-thought {{ font-size: 0.92em; line-height: 1.45; }}
table.full-history .t-thought .th-mem {{ color: #d8d8d8; padding: 0.1em 0; }}
table.full-history .t-thought .th-reason {{ color: #a0a0a0; margin-top: 0.3em; padding: 0.1em 0; }}
table.full-history .t-thought .th-label {{ display: inline-block; font-size: 0.75em; color: #ffd85f; margin-right: 0.4em; padding: 0.05em 0.4em; background: #2a2d33; border-radius: 3px; }}
table.full-history .t-thought .th-reason .th-label {{ color: #66ccee; }}
table.full-history .t-sent .ev-out {{ color: #ff9f43; padding: 0.15em 0; }}
table.full-history .t-recv .ev-in {{ color: #66ccee; padding: 0.15em 0; }}
table.full-history .empty {{ color: #444; }}
details.history {{ margin-top: 0.8em; border: 1px solid #2c2c2c; border-radius: 6px; }}
details.history summary {{ cursor: pointer; padding: 0.5em 0.8em; color: #a0a0a0; background: #25282d; border-radius: 6px 6px 0 0; font-size: 0.88em; }}
details.history[open] summary {{ border-bottom: 1px solid #2c2c2c; }}
details.history .hbox {{ padding: 0.6em 1em; max-height: 360px; overflow-y: auto; background: #15171a; border-radius: 0 0 6px 6px; }}
details.history h5 {{ margin: 0.6em 0 0.3em; color: #c8dbef; font-size: 0.92em; }}
.tlx-wrap {{ background: #1f1f1f; border-radius: 6px; padding: 0.6em; margin: 0.6em 0; }}
.tlx-controls {{ display: flex; align-items: center; gap: 1em; padding: 0.4em 0.6em; color: #ddd; font-size: 0.92em; }}
.tlx-controls input[type=range] {{ flex: 1; }}
.tlx-controls .step-cur {{ color: #ffd85f; font-weight: bold; min-width: 12em; }}
.tlx-canvas {{ position: relative; overflow-x: auto; }}
.tlx-canvas svg {{ display: block; min-width: 1100px; }}
.tlx-bubble-layer {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
.tlx-bubble {{ position: absolute; width: 360px; max-width: 360px; padding: 10px 14px; background: #2a2d33; color: #e8e8e8; border-radius: 8px; font-size: 13px; line-height: 1.5; box-shadow: 0 6px 18px rgba(0,0,0,0.6); pointer-events: auto; border: 1px solid #555; word-wrap: break-word; z-index: 1; }}
.tlx-bubble.front {{ z-index: 100; }}
.tlx-bubble .close {{ position: absolute; top: 4px; right: 8px; cursor: pointer; color: #888; font-size: 14px; user-select: none; }}
.tlx-bubble .close:hover {{ color: #e8e8e8; }}
.tlx-bubble .meta {{ font-size: 10.5px; color: #888; margin-bottom: 4px; padding-right: 16px; }}
.tlx-cursor {{ position: absolute; top: 30px; bottom: 20px; width: 2px; background: #ffd85f; pointer-events: none; opacity: 0.7; }}
.tlx-step-label {{ position: absolute; top: 4px; padding: 2px 6px; background: #ffd85f; color: #1f1f1f; font-size: 10px; font-weight: bold; border-radius: 3px; transform: translateX(-50%); }}
.hl-grid {{ display: grid; gap: 0.6em; }}
.hl-item {{ background: #1d2025; border: 1px solid #2c2c2c; border-left: 4px solid #ffd85f; padding: 0.7em 1em; border-radius: 6px; page-break-inside: avoid; }}
.hl-step {{ font-size: 0.85em; color: #a0a0a0; margin-bottom: 0.3em; }}
.hl-body {{ color: #e8e8e8; }}
/* survey */
.survey-num-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.8em; margin: 1em 0; }}
.survey-num-card {{ background: #1d2025; border: 1px solid #2c2c2c; border-radius: 8px; padding: 0.8em 1em; }}
.survey-num-card .label {{ color: #888; font-size: 0.85em; margin-bottom: 0.3em; }}
.survey-num-card .value {{ color: #ffd85f; font-size: 1.6em; font-weight: bold; }}
.survey-num-card .breakdown {{ color: #aaa; font-size: 0.8em; margin-top: 0.3em; }}
.survey-num-card.bad .value {{ color: #ff6b6b; }}
.survey-num-card.warn .value {{ color: #ffaf3c; }}
.survey-num-card.ok   .value {{ color: #66ddaa; }}
.free-resp {{ background: #1d2025; border: 1px solid #2c2c2c; border-radius: 8px; padding: 0.7em 1em; margin: 0.5em 0; page-break-inside: avoid; }}
.free-resp .who {{ color: #ffd85f; font-weight: 600; font-size: 0.9em; margin-bottom: 0.2em; }}
.free-resp .body {{ color: #d8d8d8; font-size: 0.92em; line-height: 1.55; }}
/* 数値設問の各人の点数+理由テーブル */
.indiv-block {{ background: #16191e; border: 1px solid #2c2c2c; border-radius: 8px; padding: 0.7em 1em; margin: 0.6em 0 1.2em; page-break-inside: auto; }}
.indiv-block .indiv-head {{ color: #ffd85f; font-size: 0.95em; font-weight: 600; margin-bottom: 0.15em; }}
.indiv-block .indiv-q {{ color: #aaa; font-size: 0.82em; margin-bottom: 0.4em; }}
.indiv-table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
.indiv-table th {{ background: #232830; color: #aaa; text-align: left; padding: 0.3em 0.55em; border-bottom: 1px solid #333; font-weight: normal; }}
.indiv-table td {{ padding: 0.35em 0.55em; border-bottom: 1px solid #2a2d33; vertical-align: top; }}
.indiv-table td.score {{ font-weight: bold; text-align: center; width: 2.6em; color: #ffd85f; }}
.indiv-table tr.hi td.score {{ color: #66ddaa; }}
.indiv-table tr.lo td.score {{ color: #ff6b6b; }}
.indiv-table td.who {{ white-space: nowrap; width: 11em; color: #d8d8d8; }}
.indiv-table td.who .ax {{ color: #888; font-size: 0.78em; margin-left: 0.3em; }}
.indiv-table td.reason {{ color: #d8d8d8; line-height: 1.45; }}
.indiv-table td.reason.thin {{ color: #888; font-style: italic; }}

/* 全体考察セクション (reflection) のビジュアライズ */
.reflection-md h2 {{ color: #ffd85f; border-left: 4px solid #ffd85f; padding: 0.3em 0.7em; background: #231f12; border-radius: 0 4px 4px 0; margin-top: 1em; font-size: 1.1em; }}
.reflection-md h3 {{ color: #62e08a; border-left: 3px solid #62e08a; padding: 0.2em 0.6em; background: #18241c; border-radius: 0 3px 3px 0; margin-top: 0.7em; font-size: 1em; }}
.reflection-md ul, .reflection-md ol {{ background: #1a1d22; border: 1px solid #2c2c2c; padding: 0.6em 1em 0.6em 2em; border-radius: 6px; line-height: 1.7; }}
.reflection-md ul li, .reflection-md ol li {{ margin-bottom: 0.25em; color: #d8d8d8; }}
.reflection-md p {{ line-height: 1.7; color: #d8d8d8; }}
.reflection-md strong {{ color: #ffd85f; }}
.reflection-md em {{ color: #7aa9ff; font-style: normal; }}
.reflection-md blockquote {{ border-left: 3px solid #6e8aaf; background: #1a1d22; padding: 0.5em 0.9em; margin: 0.6em 0; color: #c8d4e3; }}

/* PDF 化時は HTML のダークビジュアルをそのまま維持 (ライトモード変換しない)。
   max-width 制限を外し、見出し直後の改ページ防止と、カード等の page-break-inside: avoid を入れる。 */
@media print {{
  /* PDF 用にフォントを 11px に縮小 (画面表示は 14px のまま)。
     playwright の page.pdf(scale=...) は scale<1 でコンテンツ末尾が切れる事故があったため
     font-size 縮小で全体縮小を実現する方式に変更。 */
  body {{ max-width: none; font-size: 11px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  /* 事象カード・分析カード・ペルソナカード等は中で切れないように */
  .analysis-card, .persona-card, .free-resp, .survey-num-card, .hl-item, .indiv-block,
  .api-meta, .exp-meta, .intro-question {{ page-break-inside: avoid; break-inside: avoid; }}
}}
/* 見出しの直後で改ページしないように (PDF 表示時にも有効) */
h1, h2, h3, h4 {{ page-break-after: avoid; break-after: avoid; }}
table tr, table tbody {{ page-break-inside: avoid; break-inside: avoid; }}
</style></head><body>
"""

FOOT = "</body></html>"


# ---------- Phase A セクション ----------

def render_agent_rules(phase: str, *, step_caption: str | None = None) -> str:
    """エージェントの挙動ルール。 system_prompt / user_prompt の中身まで網羅。
    Phase='A' (教室座学) or 'B' (フィールドワーク)。step_caption に動的時間表記を渡せる。"""

    default_step_caption = (
        "1 step = 2 分。Phase A 30 step = 60分 (10:00〜11:00 座学) / "
        "Phase B 60 step = 120分 (13:00〜15:00 フィールドワーク)"
    )
    use_caption = step_caption or default_step_caption

    # ===== セクション 1: 規模と時間 =====
    if phase == "A":
        scale = [
            ("シーン", "教室固定 (座席に座ったまま)、移動の概念なし"),
            ("通信半径", "12 cells (教室全体に届く)"),
            ("ステップ", use_caption),
            ("LLM call/step",
             "発話 phase のみ 1 call/step (行動 phase は skip)。memory は発話 call と一緒に書き込む"),
        ]
    else:
        scale = [
            ("シーン", "品川駅周辺の屋外フィールド (約 160m × 160m)"),
            ("出発地点", "**東西自由通路** (駅東西を結ぶ歩行者デッキ中央)、港南・高輪両側に等距離"),
            ("通信半径", "10 cells (≒ 50m)"),
            ("ステップ", use_caption),
            ("LLM call/step",
             "発話 phase + 行動 phase の 2 call/step。発話と移動は独立決定 (歩きながら会話可)"),
        ]

    # ===== セクション 2: 1 step の処理フロー =====
    if phase == "A":
        flow = [
            ("step 開始",
             "agent 内部の internal_state (内部状態 = energy 元気度 / hunger 空腹度 / social_fatigue 社交疲労) を更新。Phase A は移動なしなので位置は変動しない"),
            ("Phase 1 第1パス: 誰に話しかけるかの相手 (partner) を選ぶ",
             "通信半径内に居る他人 (nearby = 近くに居る人) を集めて、各候補との「話しやすさスコア」を **社交性 × 相手との関係性 × 距離 × 気分ブースト** で計算し、最も高い相手を partner_id (会話相手の ID) として 1 人選ぶ。スコアで外れても nearby に誰か居れば最寄りの相手を仮 partner にして LLM の判断に渡す (= silent (黙る) はシステムでなく LLM が決める)"),
            ("Phase 1 双方向同時発話排除",
             "A→B かつ B→A の双方向ペアを検出。**id 小さい方が話す、大きい方は黙る (聞く側に回って次 step で返事する)**"),
            ("Phase 1 第2パス: 各 agent ごとに発話 LLM 呼び出し",
             "黙る判定でない agent に対し、Gemini を並列呼び出しして発話判定。message (発話) + reasoning (思考) + memory (記憶) を JSON で返す"),
            ("Phase 2 メッセージ配信 (delivery)",
             "発話成立した agent の message を 1対1 で partner に送信。受信者の received_messages (受信箱) に追加"),
            ("Phase 3 行動 phase は skip",
             "Phase A は教室座席に固定で移動が無いため、行動 LLM 呼び出しはスキップ。action_type=stay (留まる) を機械的に置く"),
            ("step 終了", "memory (記憶) ログ書き込み、relationship (関係性) 更新"),
            ("memory 圧縮",
             "step が 5 の倍数のたび、各 agent の **直近 5 件 memory を 1 文に要約** して archived_summaries (圧縮記憶) に push (= 追加。Gemini 呼び出し、並列)。raw (生記憶) は捨てない"),
        ]
    else:
        flow = [
            ("step 開始",
             "agent 内部の internal_state (内部状態 = energy 元気度 / hunger 空腹度 / social_fatigue 社交疲労) を更新。**現在位置 (x,y) も再計算** (前 step の移動結果を反映)。これが下流の「自分は今どこに居るか / どの place (場所) 内か」 の判定根拠になる"),
            ("perceive 注入 (現地感覚の注入)",
             "agent の現在位置がいずれかの place の polygon (多角形領域) 内に入っていたら、その place の **`perceive_pass` (通り過ぎたときに見えるもの) / `perceive_enter` (入ってみて気づくこと)** をその場で memory に rule-based (機械的) で書き込む (各 place 初回のみ)。LLM が「実際に行ってみないと得られない情報」 を獲得する仕組み"),
            ("Phase 1 第1パス: 誰に話しかけるかの相手 (partner) を選ぶ",
             "通信半径内に居る他人 (nearby = 近くに居る人) を集めて、各候補との「話しやすさスコア」を **社交性 × 相手との関係性 × 距離 × 気分ブースト** で計算し、最も高い相手を partner_id (会話相手の ID) として 1 人選ぶ。スコアで外れても nearby に誰か居れば最寄りの相手を仮 partner にして LLM の判断に渡す (= silent (黙る) はシステムでなく LLM が決める)"),
            ("Phase 1 双方向同時発話排除",
             "A→B かつ B→A の双方向ペアを検出。**id 小さい方が話す、大きい方は黙る (聞く側に回って次 step で返事する)**"),
            ("Phase 1 第2パス: 各 agent ごとに発話 LLM 呼び出し",
             "黙る判定でない agent に対し、Gemini を並列呼び出しして発話判定。message (発話) + reasoning (思考) + memory (記憶) を JSON で返す。**host (企業担当者) は近接に学生が居ないとき LLM 呼び出しスキップ** (コスト最適化、学生が来たら通常通り発話)"),
            ("Phase 2 メッセージ配信 (delivery)",
             "発話成立した agent の message を 1対1 で partner に送信。受信者の received_messages (受信箱) に追加"),
            ("Phase 3 行動 LLM 呼び出し",
             "各 agent ごと並列で decide_action (行動判定)。walk_toward (向かって歩く) / walk_along (沿って歩く) / enter (入る) / stay (留まる) / approach (近づく) / wander (うろつく) から LLM が文脈で選ぶ。memory + reasoning も同時に書き込み"),
            ("Phase 4 移動実行",
             "navigation.py が walkable mask (歩行可 cell マップ) 上で 1 step 最大 12 cells (≒60m) 移動。道路幅員 ≥ 5 cell の道は両端 sidewalk (歩道) のみ歩行可、施設は apron-jump (建物の前まで歩いて中に入る挙動) で入る"),
            ("step 終了", "memory ログ、relationship (関係性) 更新、event awareness (イベント認知) ログ"),
            ("memory 圧縮",
             "step が 5 の倍数のたび、各 agent の **直近 5 件 memory を 1 文に要約** して archived_summaries (圧縮記憶) に push (= 追加。Gemini 呼び出し、並列)。raw (生記憶) は捨てない"),
        ]

    # ===== セクション 3: system_prompt の構造 (cache 領域) =====
    sysprompt_a = [
        ("WHO YOU ARE (あなたは誰か)", "(なし。persona は user_prompt 側)"),
        ("MESSAGE RULES (発話ルール)",
         "broadcast 形式 / 200 words 以内 / 沈黙 OK / 重複禁止 / メカニカルな発言禁止"),
        ("CONVERSATION PRINCIPLES (会話原則)",
         "relationship label (stranger / face familiar / acquaintance / friend / close) ごとの tone ガイド"),
        ("RESPONSE FORMAT (出力形式)", "JSON 出力 (message + reasoning [+ memory])、日本語必須"),
        ("BEHAVIORAL GUIDANCE (振る舞いの指針)",
         "「すべての思考を発話するな」「同じ話題の繰り返し回避」「沈黙は valid」"),
        ("発話判定の権限委譲 (システム側で silent 強制しない)",
         "「発話するかしないかは性格・状況・内発的動機に従って自分で決める。話しかけられたら自然に応じる。同じ人に同じ内容を再送するのは禁止」 ─ 確率は partner 選択の補助に留め、silent / 発話の最終決定は LLM の文脈判断に任せる"),
        ("ON-SITE PERCEPTION OF THE TOWN (現地で見えるまちの特徴)",
         "東京都都市整備局 GL2020 に基づく品川の特徴 (地形二面性 / 交通結節点 / 歴史 (旧東海道〜現在) / 業務集積 / 整備方針 (国際ビジネスセンター・MICE等) / 主要再開発 / 生活) を全文注入"),
        ("CENTRAL QUESTION (中心問い)",
         "「このまちのいいところ・悪いところを話し合い、こうなったらいいという未来像を描いてください」を末尾に固定挿入"),
    ]
    sysprompt_b = [
        ("WORLD STRUCTURE (世界構造)",
         "2D グリッド、座標系、通信半径 10、通信ルール (同じ area / 半径内 のみ届く)"),
        ("PLACE LOCATIONS (場所一覧)", "全 place の座標 + capacity を一覧で渡す"),
        ("DATA INTERPRETATION (データの読み方)",
         "定量データ (距離・占有率・火災 intensity 等) は数値のみ提示、解釈は agent 側に委ねる"),
        ("MESSAGE RULES (発話ルール)",
         "broadcast / 200 words / 沈黙 OK / 自分の (x,y) は秘匿 / 重複禁止"),
        ("CONVERSATION PRINCIPLES (会話原則)",
         "日本の都市での会話ノルム: 知らない人同士は短く実用的な発話のみ、深い対話は知り合い以降"),
        ("WHEN TRANSIT IS DISRUPTED (交通障害時の判断)",
         "電車運休等の判断ガイド (信頼性 vs 代替路 vs 目的)"),
        ("RESPONSE FORMAT (出力形式)", "JSON 出力 (message + reasoning)"),
        ("EXTENDED GUIDANCE (詳細ガイド)",
         "10サブセクション: 発話 phase の役割 / 位置秘匿の理由 / 良い発話の例 / 沈黙の合理性 / 思考と発話の差 (発話は listener 向けに編集) / turn-taking / 受信メッセージとの距離 / 火災イベント / memory の使い方 / 日本語スタイル ほか"),
        ("BEHAVIORAL GUIDANCE (振る舞いの指針)",
         "「思考を全部喋らない」「相手のメッセージへの反応は自然に」「過去発話の繰り返し回避」"),
        ("発話判定の権限委譲 (システム側で silent 強制しない)",
         "確率は partner 選択の補助に留め、silent / 発話の最終決定は LLM の文脈判断に任せる。話しかけられたら自然に応じる、内発的動機に従う"),
        ("CONVERSATION STYLE GUIDELINES (会話スタイルガイド)",
         "DO/DONT (実名で呼ぶ・座標を喋らない・最適化しすぎない 等)"),
        ("CENTRAL QUESTION (中心問い)",
         "「このまちのいいところや課題を様々な視点で探し、こうなったらいいと思う未来像を描いてください」"),
        ("今日の必須課題 (フィールドワーク)",
         "「FW中、最低でも **2社以上の企業担当者** (受け入れ拠点) を実際に訪ね、自社の取り組みや品川での役割について話を聞く。常に意識して達成する」 ─ system_prompt 末尾に固定 (rolling buffer から消えないよう)"),
    ]

    # ===== セクション 4: user_prompt の構造 (毎step変動) =====
    user_prompt_a = [
        ("WHO YOU ARE (あなたは誰か)",
         "name / age / gender / occupation / 気質 (社交的・楽天的・好奇心の3軸) / background / 口癖 / speech_style / cognitive_biases (認知バイアス)"),
        ("YOUR INTERNAL STATE (内部状態)",
         "energy (元気度) / hunger (空腹度) / social_fatigue (社交疲労)。**現状 LLM の発話判断には強くは効いていない参考情報**だが、内省 prompt として「疲れている自分を踏まえて話す/黙る」 を選ばせる素材になる"),
        ("ACTIVE GROUP IDENTITIES (今のあなたを支配しているアイデンティティ)",
         "agent が複数のアイデンティティ (部活 / クラス / 趣味タグ など) を持つときの「いま強く効いているもの」 を salience ≥ 0.4 のものだけ渡す。誰として今喋るかの軸"),
        ("CURRENT TIME & CONTEXT (現在時刻と文脈)", "Current time / Neighborhood mood / 時間帯ガード文"),
        ("NEARBY PEOPLE (近くに居る他人)",
         "聞こえる範囲の他人。name (名前) + relationship label (関係性ラベル) のみ (座標は秘匿)"),
        ("PREVIOUS MEMORY (これまでの記憶)",
         "**[要約 #N]** 圧縮記憶 (archived_summaries) + **直近 raw (生記憶) 5件** (rolling buffer = 直近の生記憶バッファ)。Phase A は移動が無いため [訪問履歴] は無し"),
        ("あなたの過去発話 (相手別)",
         "nearby (通信半径内) の各人ごとに「あなたが既にこの人に言ったこと top 3」を整理。**同じ人に同じことを再送しないため**"),
        ("RECENT CONVERSATION (直近の会話)", "受信 + 送信のマージ window (直近窓、最大 message_context_size × 2 件)"),
        ("Step", "現在 step 番号"),
    ]
    user_prompt_b = [
        ("WHO YOU ARE (あなたは誰か)",
         "name / age / gender / occupation / 気質 (社交的・楽天的・好奇心の3軸) / background / 口癖 / speech_style / cognitive_biases (認知バイアス)"),
        ("YOUR INTERNAL STATE (内部状態)",
         "energy (元気度) / hunger (空腹度) / social_fatigue (社交疲労)。Phase B では更に「歩いている」 状態を踏まえて、発話/沈黙の選択や近場の place を選ぶ判断材料になる"),
        ("ACTIVE GROUP IDENTITIES (今のあなたを支配しているアイデンティティ)",
         "salience ≥ 0.4 のもののみ。Phase B では「FW中に何者として喋るか」 (高校生 / 部活 / 興味タグ等) を区別"),
        ("CURRENT TIME & CONTEXT (現在時刻と文脈)", "Current time / Neighborhood mood / 時間帯ガード文"),
        ("YOUR CURRENT STATE (あなたの現在状況)",
         "現 (x, y) 座標、in_place フラグ、滞在中の place 名、behavior_layer"),
        ("CURRENT PLACE / FIRE / EVENTS (場所・火災・イベント)",
         "place 占有率、火災情報、運休イベント等"),
        ("NEARBY PLACES (近場の場所)", "近くの place 一覧 (距離+方向)"),
        ("NEARBY PEOPLE (近くに居る他人)",
         "聞こえる範囲の他人。name (名前) + relationship label (関係性ラベル) のみ (座標は秘匿)"),
        ("PREVIOUS MEMORY (これまでの記憶)",
         "**[訪問履歴]** (rule-based 固定: 「○○に到着」 を入場時に機械挿入) + **[要約 #N]** 圧縮記憶 + **直近 raw (生記憶) 5件** (rolling buffer = 直近の生記憶バッファ)"),
        ("あなたの過去発話 (相手別)",
         "nearby (通信半径内) の各人ごとに「あなたが既にこの人に言ったこと top 3」 を整理"),
        ("CONVERSATION OVERHEARD IN YOUR AREA (周りで聞こえる会話)",
         "通信半径内の受信 + 送信メッセージ (最大 message_context_size × 2 件)"),
        ("Step", "現在 step 番号"),
    ]

    # ===== セクション 5: 発話判定の仕組み =====
    speech = [
        ("確率式 (partner 選択の重み付け用)",
         "話しやすさスコア = talkativeness (社交性) × social_likelihood (会話発生率) × relationship (関係性) × proximity_factor (近さ係数) × 各ブースト (上限 1.0)。proximity_factor (近さ係数) は dist (距離) ≤ 2 cell → 1.0 / ≤ 4 cell → 0.4 / それ以上 → 0.1。**この確率は「どの相手を partner (会話相手) に選ぶか」 の重み付けにのみ使い、silent (黙る) 判定には使わない**"),
        ("ブースト (係数の上乗せ)",
         "近接 (dist ≤ 2 cell) で ×1.5、近接 + relationship (関係性) ≥ 0.5 (同行ペア) で更に ×1.5 (重ね掛け最大 2.25x)、火災等で ×3。social_fatigue (社交疲労) で抑制"),
        ("silent (黙る) / 発話の最終判断は LLM 任せ",
         "確率で外れた場合も nearby (通信半径内) に誰か居れば最寄りを partner (会話相手) として仮選択し LLM 呼び出し。**実際に話すか silent (黙る) かは LLM が文脈で判断** (システム側で silent を強制しない)。これにより「黙りたいときは黙る、話したいときは話す」 が persona に従って起きる"),
        ("双方向同時発話排除",
         "A→B かつ B→A の双方向ペアを検出。**id 小さい方が話す、id 大きい方は silent (黙る)** (聞く側に回り、次 step で返事する)。同じ step で 2 人が同時に同じ相手に話しかける現象を構造的に防ぐ"),
        ("同じ人に同じこと禁止",
         "user_prompt に「あなたが nearby (通信半径内) の各人に既に言ったこと top 3」 を可視化 + system_prompt で禁止明示。LLM が「もう A には〇〇を言った」 を認識して回避"),
        ("host idle silence (企業担当者の待機時無発話、Phase B のみ)",
         "is_host=True かつ近接に学生いない場合は LLM 呼び出しをスキップ (silent = 黙る)。コスト最適化、ただし学生が来たら通常通り発話"),
        ("関係性 (relationship) 更新",
         "発話成立で双方の relationship (関係性) が +Δ (微増)。長時間同席でも +Δ (詳細は relationships_timeline.jsonl)"),
    ]

    # ===== セクション 6: 行動判定 (Phase B のみ) =====
    action_b = [
        ("行動の選択肢",
         "walk_toward (向かって歩く) / walk_along (沿って歩く) / enter (入る) / stay (留まる) / approach (近づく) / wander (うろつく) の 6 種から **LLM が文脈判断で選ぶ**。確率なし"),
        ("移動量", "1 step あたり最大 12 cells (≒ 60m)、徒歩相当"),
        ("歩行制約",
         "navigation.py の walkable mask (歩行可 cell マップ) 上のみ通行可。道路幅員 ≥ 5 cell の道は両端 sidewalk_w (歩道幅) cells のみ歩行可 (車道は不可)、交差点は横断可、施設内は apron-jump (建物前まで歩いてから中に入る挙動) で入る"),
        ("経路探索", "BFS (幅優先探索) による最短路。goal (目的地) 到達不能なら最寄りに snap (吸着)"),
    ]

    # ===== セクション 7: memory 管理 =====
    if phase == "A":
        memory = [
            ("memory_size (raw rolling = 直近の生記憶)",
             "**5 件** (= 直近 10 分の raw 思考 = 圧縮前の生の思考)。新規 step memory が追加されるたび、6 件目から最古を pop (= 押し出して捨てる)"),
            ("memory_limit (内部上限)", "20 件まで agent 内部に保持"),
            ("圧縮記憶 (archived_summaries)",
             "**step が 5 の倍数のたび、直近 5 件 raw (生記憶) を 1 文要約 (Gemini)** して archived_summaries (圧縮記憶) に push (= 追加)。固定保管 (消えない)、長期記憶になる"),
            ("prompt に乗せる順序",
             "[要約 #1, #2, ...] (archived = 圧縮記憶 全件) → 直近 raw (生記憶) 5 件、の順で渡す。LLM は長期 + 短期の両方を見ながら判断"),
            ("内省 (memory + reasoning)",
             "毎 step の LLM 呼び出しで「何が起きた + 自分はどう感じた」 を 1〜2 文で memory (記憶) に書く。reasoning (思考) は意思決定の根拠を別途。次 step 以降の文脈に使う"),
        ]
    else:
        memory = [
            ("memory_size (raw rolling = 直近の生記憶)",
             "**5 件** (= 直近 10 分の raw 思考 = 圧縮前の生の思考)。新規 step memory が追加されるたび、6 件目から最古を pop (= 押し出して捨てる)"),
            ("memory_limit (内部上限)", "20 件まで agent 内部に保持"),
            ("圧縮記憶 (archived_summaries)",
             "**step が 5 の倍数のたび、直近 5 件 raw (生記憶) を 1 文要約 (Gemini)** して archived_summaries (圧縮記憶) に push (= 追加)。固定保管 (消えない)、長期記憶になる"),
            ("prompt に乗せる順序",
             "[訪問履歴] (rule-based 固定) → [要約 #1, #2, ...] (archived = 圧縮記憶 全件) → 直近 raw (生記憶) 5 件、の順で渡す。LLM は長期 + 短期の両方を見ながら判断"),
            ("内省 (memory + reasoning)",
             "毎 step の LLM 呼び出しで「何が起きた + 自分はどう感じた」 を 1〜2 文で memory (記憶) に書く。reasoning (思考) は意思決定の根拠を別途。次 step 以降の文脈に使う"),
            ("初期 memory (Phase B 開始時)",
             "Phase A handoff (引継ぎ) で抽出した future_image (未来像) / intent (意図) / 3 key_memories (重要記憶 3件) / 3 field_questions (現地で確かめたい問い 3件) / one_liner (一行コメント) + FW_PREMISE (フィールドワーク前提) + 「現在地」 を Phase B 開始時に initial_memory (初期記憶) として注入"),
            ("perceive 注入 (Phase B のみ)",
             "agent が place (場所) 内に進入すると、その place の `perceive_pass` (外から見た光景) / `perceive_enter` (中で気づくこと) を memory に rule-based (機械的) で注入 (各 place 初回のみ)"),
        ]

    # ===== セクション 8: 引継ぎ =====
    if phase == "A":
        handoff = [
            ("Phase A → Phase B",
             "Phase A 終了後、各 agent の全 memory (記憶) + 全発話を Gemini に投げて 9 項目を抽出: future_image (未来像、1文) / intent (意図、1文) / key_memories (重要記憶、3件) / field_questions (現地で確かめたい問い、3件) / one_liner (一行コメント、1件)。Phase B 開始時に initial_memory (初期記憶) として注入。Phase A の細かい memory・relationship は引継がれない"),
        ]
    else:
        handoff = [
            ("Phase A → Phase B (前段)",
             "Phase A 終了後、Gemini で抽出した 9 項目 (future_image 未来像 / intent 意図 / key_memories 重要記憶 / field_questions 現地で確かめたい問い / one_liner 一行コメント) + FW_PREMISE (フィールドワーク前提) + FW_TASK (フィールドワーク必須課題) + 現在地 を initial_memory (初期記憶) に注入"),
            ("Phase B → Phase C",
             "Phase C は run_survey.py が **Phase A run_dir + Phase B run_dir 両方を直接読む**。各 agent の Phase A memory (最大 30 件) + Phase B memory (最大 30 件) + 発話 (20 件) + 受信 (20 件) を Gemini に user_prompt として渡し、**観察者として「この人物がアンケートにどう答えるか」 を推測**させる (忖度バイアス回避)。数値設問には evidence (具体的事実) 必須"),
        ]

    # ===== セクション 9: ペルソナ生成 =====
    persona = [
        ("ペルソナ生成",
         "tools/generate_classroom_personas.py で Gemini 2.5 flash-lite が 20 人/variant (条件) を生成。axis_id (識別子: S / M / E + 番号) ごとに `tendency` (傾向 = 優等生/だるそう/反抗的/推し活/ガチ勢/内向的/ふつう 等) と `interest_tag` (興味タグ = tech / creative / mobility / space / story / nature / system) と `school_fit` (学校適応 = 適応/中間/不適応、各 variant に不適応 2 人混入) と `catchphrase_style` (口癖の型、20 人で重複しないよう事前割当) を持つ"),
        ("ジェンダーレス",
         "各 variant に 1 人 gender=other (ジェンダーレス) を混入。viewer は黄色アイコンで描画"),
    ]

    # ===== レンダリング =====
    sections = []

    def _table(rows):
        if not rows:
            return ""
        body = "".join(
            f'<tr style="text-align:left">'
            f'<td style="text-align:left;width:13em;font-weight:600;color:#cce;vertical-align:top;padding:0.3em 0.6em 0.3em 0">{html.escape(name)}</td>'
            f'<td style="text-align:left;font-size:0.88em;color:#bbb;line-height:1.55;padding:0.3em 0">{html.escape(desc)}</td>'
            f'</tr>'
            for name, desc in rows
        )
        return (
            '<table class="api-table" style="width:100%;font-size:0.9em;text-align:left;margin-bottom:1em">'
            '<tbody>' + body + '</tbody></table>'
        )

    sections.append(f'<h2 style="margin-top:0.6em">エージェント挙動ルール (Phase {phase})</h2>')
    sections.append('<p style="color:#888;font-size:0.85em;margin-bottom:0.7em">'
                    'シミュレーション内で agent がどんな情報を渡され、どう発話・行動・思考するかを項目ごとに分けて整理。'
                    'system_prompt (システム指示、cache 領域 = 毎 step 同じ) と user_prompt (ユーザ指示、毎 step 変動) の中身、確率式、双方向排除、memory (記憶) の rolling (時間スライド) 圧縮、引継ぎ機構まで含む。</p>')

    sections.append('<h3>1. 規模と時間</h3>')
    sections.append(_table(scale))

    sections.append('<h3>2. 1 step の処理フロー</h3>')
    sections.append(_table(flow))

    sections.append('<h3>3. system_prompt の構造 (cache 領域・毎step同じ)</h3>')
    sections.append(_table(sysprompt_a if phase == "A" else sysprompt_b))

    sections.append('<h3>4. user_prompt の構造 (毎step変動)</h3>')
    sections.append(_table(user_prompt_a if phase == "A" else user_prompt_b))

    sections.append('<h3>5. 発話判定の仕組み</h3>')
    sections.append(_table(speech))

    if phase == "B":
        sections.append('<h3>6. 行動判定 (Phase B のみ)</h3>')
        sections.append(_table(action_b))

    sections.append(f'<h3>{6 if phase == "A" else 7}. memory 管理</h3>')
    sections.append(_table(memory))

    sections.append(f'<h3>{7 if phase == "A" else 8}. 引継ぎ</h3>')
    sections.append(_table(handoff))

    sections.append(f'<h3>{8 if phase == "A" else 9}. ペルソナ生成</h3>')
    sections.append(_table(persona))

    return "\n".join(sections)


REFLECTION_SYSTEM = (
    "あなたは観察者として、品川駅周辺で行われた「企業協力型 学外教育プログラム」の "
    "シミュレーション結果から、**全体を通しての観察** と **3 つの狙いに対する答え合わせ** を執筆します。\n\n"
    "**重要 (絶対守る) — ファクトベース・忖度禁止**:\n"
    "- 入力データに無いことを書かない。盛らない。「○○な気づきがあったかもしれない」のような曖昧な美化は禁止。\n"
    "- **不発・失敗・スカりは包み隠さず指摘する**。例: 「2社訪問が課された生徒のうちN人は1社にも入れなかった」「memory に表面的なコメントしか残っていない学生が複数いた」「アンケート評価の根拠が抽象的だった」 等は、observed なら必ず書く。\n"
    "- 「うまくいった」と書くのは、入力データに具体的事実 (場所名・発話・memory・10段階の数字) で裏付けが取れる場合のみ。\n"
    "- 教育プログラムとして肯定的に締めるための無根拠な前向きな結語は不要。事実の積み上げのまま「現状ではここまでしか言えない」「次に検証すべきはこれ」で終わってよい。\n"
    "- いいものはいいと書いてよい。だが「いいと書きたい」が先にあって事実を引っ張るのは禁止。事実が先、評価は後。\n\n"
    "出力は Markdown で、以下の構成:\n\n"
    "## 全体を通しての観察\n"
    "(3〜5 行の概観。プロジェクト全体で何が起きたかの俯瞰。シミュ動作と教育プログラムとしての全体感を併せて。"
    "シミュ動作の観察 (turn-take 成立率・後半 stay・host 動きすぎ等) もここに混ぜて構わない、別セクションで切らない)\n\n"
    "### 1. 多様なペルソナの参加スタイルが成立する学習環境になっているか\n"
    "(まず特徴的背景を持つ生徒について個別に観察、その上で 1. の総合考察を書く構成)\n\n"
    "#### 1-1. 特徴的背景がシミュレーション上どう働いたか\n"
    "(**3 属性すべて** に触れる: **(a) 学校不適応** / **(b) ハンディキャップ (車いす利用)** / **(c) 外国籍**。"
    "ジェンダーレスは本シミュ参加者にいないので扱わない。"
    "**それぞれの属性について該当 agent を 1 人以上取り上げ**、観点は **「その特徴的背景がシミュ上の言動・移動・関係づくりにどう働いたか / それとも特に効果が見えなかったか」**。"
    "「その人がどう動いたか」 の人物紹介ではなく、**属性 (背景) → 振る舞いの差** に焦点を当てる。"
    "memory/reasoning と message を引いて具体的に書く。1 属性につき 2〜4 文。"
    "発話の量・話題の選び方・視点の違い・物理的な動きにくさ・遠慮・観察者ポジションの取り方などを観察。"
    "**該当 agent と対応する属性が間違っていないか必ず確認する**: 学校不適応 = 該当 agent の `school_fit=不適応` を持つ生徒、ハンディキャップ = `mobility=wheelchair`、外国籍 = `nationality=western/asian` の生徒、を入力データから正確に拾う。)\n\n"
    "#### 1-2. 1. に対する考察\n"
    "(1-1 の観察を踏まえて、「多様なペルソナの参加スタイルが成立する学習環境になっているか」 を総合的に評価。"
    "特定タイプだけが活躍してしまっていなかったか、属性ごとに参加の質に差があったか、そこから何が言えるか。)\n\n"
    "### 2. 子どもの変化 (視野・視座・行動意欲)\n"
    "(数値+自由記述+memory を引いて、見えたこと・見えなかったこと)\n\n"
    "### 3. 「学習環境としても機能するまち」にするために何が必要か\n"
    "(まち側の設計要素 = 滞在しやすい場所/大人との接点/移動/振り返り拠点 等、"
    "+ まちを学び場にするメリット (関係人口・地域の人材育成 等) と"
    " 懸念点 (安全・受け入れ負荷・孤立・活動格差・責任分界 等))。\n"
    "**特に「物理的なアクセシビリティ・施設の入りやすさ」の観点を必ず1段落入れる。**"
    "学生が訪問必須の拠点に向かおうとしたが、入口がわからない / polygon 境界で詰まる / "
    "結果として一度も中に入れず諦めて駅に戻ってしまった、というような「向かったが接点を持てなかった」事例が"
    "入力データの『FW で訪問必須の2拠点 vs 実際の到達結果』に現れている場合は、"
    "それを **「『入りやすい/接点をもちやすい』まちの設計」が学習環境として不可欠**という気づきに昇華させる。"
    "LLM 挙動の問題と片付けず、**現実の子どもをまちに出したときにも同じことが起きる構造的課題** "
    "(企業ロビーは学生にとって入りにくい / 受付が分からない / 担当者の居場所が見えない 等) として書く。\n\n"
    "**ルール**: 全文日本語。段落と箇条書きを混ぜてOK。"
    "シミュ結果の数字は引用してよいが、誇張せず観察。h1 は使わない (## から)。"
    "**表記統一**: 「街」「町」は **『まち』** と書く (表記ゆれ統一のため)。"
    "**まとめ・締めの段落は書かない** (最終まとめは別セクションで扱うため)。"
    "**「プロジェクトの狙いに対する考察」 のような中見出しは出さない**、いきなり ### 1. ... から始めてよい。"
)


def _build_assigned_vs_arrived_block(participants: list[dict], run_b: Path) -> str:
    """各学生の『FW で訪問必須の2拠点 vs 実際に到達できた拠点』を rule-based で集計し、
    reflection prompt に渡すテキストブロックを返す。アクセシビリティ観点の素材になる。"""
    actions_path = run_b / "actions.jsonl"
    if not actions_path.exists():
        return ""
    arrivals_by_id: dict = {}
    targets_by_id: dict = {}
    try:
        for line in actions_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            i = a.get("id")
            if i is None:
                continue
            tp = (a.get("target_place") or "").strip()
            cp = (a.get("current_place_after") or "").strip()
            if tp and tp != "null":
                targets_by_id.setdefault(i, set()).add(tp)
            if cp:
                arrivals_by_id.setdefault(i, set()).add(cp)
    except Exception:
        return ""
    rows = []
    for p in participants:
        if p.get("is_host"):
            continue
        assigned = [a.get("place") for a in (p.get("assigned_hosts") or []) if a.get("place")]
        if not assigned:
            continue
        arrived_set = arrivals_by_id.get(p.get("id"), set())
        target_set = targets_by_id.get(p.get("id"), set())
        hit_target = sum(1 for a in assigned if a in target_set)
        hit_arrived = sum(1 for a in assigned if a in arrived_set)
        miss = [a for a in assigned if a in target_set and a not in arrived_set]
        rows.append(
            f"- {p.get('name','?')}: 課された={assigned} / 向かった={hit_target}/2 / 到達できた={hit_arrived}/2"
            + (f" / 向かったが入れなかった={miss}" if miss else "")
        )
    if not rows:
        return ""
    return (
        "**FW 訪問必須2拠点に対する到達状況 (rule-based 集計)**\n"
        + "\n".join(rows)
        + "\n（『向かったが入れなかった』は、学生がそこを目的地に設定したのに polygon に入れず諦めた事例。"
        "アクセシビリティの観点で重要な兆候。）\n"
    )


def build_reflection_prompt(participants, hosts, handoffs, fw_data, survey_path, run_b: Path | None = None) -> str:
    """Gemini に渡すプロンプト本文。各種データを要約して詰める。"""
    handoff_by_axis = handoffs.get("handoff_by_axis", {}) or {}

    # 特徴的背景マッピング (LLM に厳守させる — 該当者を取り違える事故を防ぐ)
    sf_lines, mb_lines, nat_lines = [], [], []
    for p in participants:
        nm = p.get("name", "?")
        ax = p.get("axis_id", "?")
        if (p.get("school_fit") or "") == "不適応":
            sf_lines.append(f"{nm} ({ax})")
        if (p.get("mobility") or "") == "wheelchair":
            mb_lines.append(f"{nm} ({ax})")
        if (p.get("nationality") or "") in ("western", "asian"):
            nat_lines.append(f"{nm} ({ax})")
    bg_block = (
        "## 特徴的背景マッピング (厳守 — 観察対象の人物を取り違えないこと)\n"
        f"- **学校不適応** (school_fit=不適応): {', '.join(sf_lines) if sf_lines else 'なし'}\n"
        f"- **ハンディキャップ (車いす利用)** (mobility=wheelchair): {', '.join(mb_lines) if mb_lines else 'なし'}\n"
        f"- **外国籍** (nationality=western/asian): {', '.join(nat_lines) if nat_lines else 'なし'}\n"
        "**この対応表に載っていない生徒に上記属性を当てない。違う属性に書き換えるのも禁止。**\n\n"
    )

    def _attr_tag(p) -> str:
        attrs = []
        if (p.get("school_fit") or "") == "不適応":
            attrs.append("学校不適応")
        if (p.get("mobility") or "") == "wheelchair":
            attrs.append("車いす")
        if (p.get("nationality") or "") in ("western", "asian"):
            attrs.append("外国籍")
        return f" [{'/'.join(attrs)}]" if attrs else ""

    # Phase A intent / future_image
    a_lines = []
    for p in participants[:12]:
        ax = p.get("axis_id", "")
        h = handoff_by_axis.get(ax) or {}
        fi = (h.get("future_image") or "").strip()[:140]
        intent = (h.get("intent") or "").strip()[:140]
        if fi or intent:
            a_lines.append(f"- {p.get('name','?')} ({ax}){_attr_tag(p)}: future_image=「{fi}」/ intent=「{intent}」")
    # Phase B memory: 各人の最後のmemory 1〜2件 + 主な会話相手
    id_to_name = {a["id"]: a.get("name", f"#{a['id']}") for a in fw_data["personas"]}
    mr = fw_data.get("mr") or []
    mem_by_id: dict = {}
    for d in mr:
        i = d.get("id")
        if i is None: continue
        mem_by_id.setdefault(i, []).append(d)
    b_lines = []
    for p in participants[:12]:
        recs = mem_by_id.get(p["id"], [])
        if not recs: continue
        last = recs[-1]
        b_lines.append(f"- {p.get('name','?')}: 末尾memory=「{(last.get('memory') or '')[:140]}」")
    # Phase C survey
    survey_lines = []
    if survey_path.exists():
        for line in survey_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            role = r.get("role")
            res = r.get("responses") or {}
            if role == "student":
                nums = " ".join(f"q{q}={res.get(f'q{q}')}" for q in (1,2,3) if res.get(f"q{q}") is not None)
                free_raw = (res.get("q4","") or "")
                free = free_raw[:200] + ("…" if len(free_raw) > 200 else "")
                survey_lines.append(f"- 生徒 {r.get('name','?')}: {nums} / Q4「{free}」")
            elif role == "host":
                nums = " ".join(f"q{q}={res.get(f'q{q}')}" for q in (7,8) if res.get(f"q{q}") is not None)
                free_raw = (res.get("q9","") or "")
                free = free_raw[:200] + ("…" if len(free_raw) > 200 else "")
                survey_lines.append(f"- 企業担当者 {r.get('name','?')}: {nums} / Q9「{free}」")

    # FW 課題拠点に対する到達状況 (アクセシビリティ観点用)
    arrived_block = ""
    if run_b is not None:
        arrived_block = _build_assigned_vs_arrived_block(participants, run_b)

    return (
        "## 入力データ\n\n"
        f"**Phase A 生徒の出発点 (n={len(participants)})**\n"
        + "\n".join(a_lines) + "\n\n"
        "**Phase B 終端memory**\n"
        + "\n".join(b_lines) + "\n\n"
        + (arrived_block + "\n" if arrived_block else "")
        + "**Phase C アンケート (10段階評価および自由記述)**\n"
        + "\n".join(survey_lines[:24]) + "\n\n"
        "## あなたの仕事\n"
        "上記を踏まえ、システムプロンプト指定の構成 (## 全体観察 → ## 答え合わせ ×3 → ## まとめ) で書いてください。"
    )


def render_overall_reflection(participants, hosts, handoffs, fw_data, survey_path: Path,
                              client: GeminiClient, run_b: Path | None = None) -> str:
    """Gemini に「全体考察 + 3つの狙いの答え合わせ」を書かせて HTML 化。"""
    try:
        user = build_reflection_prompt(participants, hosts, handoffs, fw_data, survey_path, run_b=run_b)
        md = client.generate(REFLECTION_SYSTEM, user, temperature=0.5, max_tokens=2200)
    except Exception as e:
        return f'<div class="empty">(全体考察の生成に失敗: {html.escape(str(e))})</div>'
    if not md:
        return '<div class="empty">(全体考察の生成結果なし)</div>'
    # phaseBレポと同じ md→html
    from tools.render_phaseB_report import render_md_to_html
    return f'<div class="reflection-md">{render_md_to_html(md.strip())}</div>'


def render_town_intro(participants: list[dict]) -> str:
    """座学で生徒に渡された品川の事前情報をレポート用に整形して表示。
    background から TOWN_PROFILES["shinagawa"] が build_town_intro 経由で書き出した
    「──── 今日の話し合い対象のまち: 品川 ────」以降のブロックを抜き出す。
    """
    if not participants:
        return ""
    bg = (participants[0].get("background") or "")
    # build_town_intro が "──── 今日の話し合い対象のまち: " で挟んで挿入する
    marker = "──── 今日の話し合い対象のまち:"
    idx = bg.find(marker)
    if idx < 0:
        return ""
    town_block = bg[idx:].strip()
    # ブロック内を行単位に整形
    lines = [l for l in town_block.splitlines() if l.strip()]
    if not lines:
        return ""
    title_line = lines[0]
    body_lines = lines[1:]

    # まず summary、続いて facts (・始まり) を箇条書き、その後 notes (※) は文章として
    summary = ""
    facts = []
    notes = []
    for l in body_lines:
        s = l.strip()
        if s.startswith("・"):
            facts.append(s.lstrip("・").strip())
        elif s.startswith("※"):
            notes.append(s)
        elif s.startswith("以下は"):
            continue  # 「以下は、このまちを理解するための基本的な情報:」の説明文は飛ばす
        else:
            if not summary:
                summary = s
            else:
                # summary or notes 続行行
                if notes:
                    notes[-1] += " " + s
                else:
                    summary += " " + s

    facts_html = "".join(f'<li>{html.escape(f)}</li>' for f in facts)
    notes_html = "".join(
        f'<div style="font-size:0.85em;color:#bbb;margin-top:0.3em;white-space:pre-wrap">{html.escape(n)}</div>'
        for n in notes
    )

    return (
        '<div class="town-intro" style="background:#1a1d22;border-left:4px solid #ffd85f;'
        'padding:0.8em 1em;margin:0.8em 0 1.2em;border-radius:6px">'
        f'<div style="font-weight:600;color:#ffd85f">{html.escape(title_line)}</div>'
        + (f'<div style="margin-top:0.5em">{html.escape(summary)}</div>' if summary else "")
        + (f'<ul style="margin-top:0.4em;padding-left:1.4em">{facts_html}</ul>' if facts else "")
        + notes_html
        + '<div style="margin-top:0.6em;color:#888;font-size:0.78em">'
        '出典: 東京都都市整備局 GL2020 (品川駅・田町駅周辺まちづくりガイドライン2020)。座学開始時に生徒全員に共通配布。'
        '</div>'
        '</div>'
    )


def _agent_label(p: dict) -> str:
    """smoke22: agent name に所属 (host) / 特徴的背景 (学生) を併記したラベルを作る。
    レポート全体で「誰がどこの人か」「特徴的背景は何か」が一目で分かるように。"""
    name = p.get("name", "?")
    tags = []
    is_host = bool(p.get("is_host")) or str(p.get("axis_id", "")).startswith("Host_")
    if is_host:
        place = p.get("initial_place", "")
        occ = p.get("occupation", "") or ""
        # occupation の最初の一塊 (= 企業名) を取り出す
        company = occ.split()[0] if occ else ""
        if company and place:
            tags.append(f"{company} / {place}")
        elif place:
            tags.append(place)
        elif company:
            tags.append(company)
    else:
        # 学生の特徴的背景タグ
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


def render_phaseA_section(run_a: Path, handoffs: dict, participants: list[dict]) -> str:
    """Phase A 教室シミュ概要。Gemini 分析は省略、handoff の future_image / intent / key_memories を表で出す。"""
    cfg = yaml.safe_load((run_a / "config.yaml").read_text(encoding="utf-8"))
    meta = cfg.get("metadata", {})
    sim = cfg.get("simulation", {})
    duration = sim.get("duration", "?")
    seed = sim.get("seed", "?")
    handoff_by_axis = handoffs.get("handoff_by_axis", {}) or {}
    fq_by_axis = handoffs.get("fq_by_axis", {}) or {}

    rows = []
    for p in participants:
        ax = p.get("axis_id", "")
        h = handoff_by_axis.get(ax) or {}
        q = fq_by_axis.get(ax) or {}
        fi = (h.get("future_image") or "").strip()
        intent = (h.get("intent") or "").strip()
        kms = h.get("key_memories") or []
        questions = q.get("questions") or []
        oneliner = (q.get("one_liner") or "").strip()
        gender_cls = p.get("gender", "")
        # 各人サマリーから冗長なグレー情報 (axis ID, 全項目空の気質など) を削る
        temp = _temp_str(p, compact=True)
        school_fit = p.get("school_fit") or ""
        interest = p.get("interest_tag") or ""
        meta_parts = []
        if school_fit and school_fit != "適応":
            meta_parts.append(f"学校適応: {html.escape(school_fit)}")
        if interest:
            meta_parts.append(f"興味: {html.escape(interest)}")
        if temp:
            meta_parts.append(f"気質: {html.escape(temp)}")
        meta_line = " / ".join(meta_parts)
        rows.append(
            f'<div class="analysis-card {gender_cls}">'
            f'<div class="agent-name">{html.escape(_agent_label(p))}（{p.get("age","?")}歳・{html.escape(_gender_jp(p))}）</div>'
            + (f'<div class="agent-meta">{meta_line}</div>' if meta_line else "")
            + f'<h4>座学で描いた未来像</h4><p>{html.escape(fi) or "<span class=empty>(なし)</span>"}</p>'
            f'<h4>FWでこう過ごしたい</h4><p>{html.escape(intent) or "<span class=empty>(なし)</span>"}</p>'
            + (f'<h4>主な関心事</h4><ul>' + "".join(f"<li>{html.escape(k)}</li>" for k in kms[:3]) + "</ul>" if kms else "")
            + (f'<h4>FWで確かめたいこと</h4><ul>' + "".join(f"<li>{html.escape(qq)}</li>" for qq in questions[:3]) + "</ul>" if questions else "")
            + (f'<h4>FW全体テーマ</h4><p>{html.escape(oneliner)}</p>' if oneliner else "")
            + '</div>'
        )

    # 規模情報は intro セクションで既出のため、Phase A セクションでは重複させない
    town_html = render_town_intro(participants)
    intro_section = ""
    if town_html:
        intro_section = (
            '<h2 style="margin-top:0.4em">座学で生徒に共有した品川の事前情報</h2>'
            + town_html
        )

    # Phase A の interactive timeline (Phase B と同じ slider/bubble UX)。
    phaseA_timeline_html = ""
    try:
        data_a = load_run(run_a)
        a_total = int(sim.get("duration", 100) or 100)
        ts_a = sim.get("time_scale", {})
        a_mps = int(ts_a.get("step_duration_minutes", 2) or 2)
        a_start = str(ts_a.get("start_time", "09:00"))
        try:
            sh_a, sm_a = a_start.split(":")
            sh_a = int(sh_a); sm_a = int(sm_a)
        except Exception:
            sh_a, sm_a = 9, 0
        timeline_a = build_phaseB_interactive_timeline(
            data_a, participants, a_total,
            start_hour=sh_a, mins_per_step=a_mps,
            phase_id="phaseA", start_minute=sm_a,
        )
        phaseA_timeline_html = (
            '<h2 style="margin-top:0.8em">Phase A 活動タイムライン</h2>'
            + timeline_a
        )
    except Exception as e:
        phaseA_timeline_html = f'<p style="color:#888">(Phase A タイムライン生成失敗: {html.escape(str(e))})</p>'

    return intro_section + phaseA_timeline_html + "\n<h2>各生徒の座学アウトプット</h2>\n" + "\n".join(rows)


def _bg_summary(p: dict, max_len: int = 200) -> str:
    """persona の background から「まち情報」「FW前提」「シーン文 (放課後…)」などの
    共通注入ブロックを除き、本人固有の背景部分だけを抽出して短く要約。"""
    bg = (p.get("background") or "").strip()
    # まち情報以降を切り捨て
    for marker in ("\n──── 今日の話し合い対象のまち", "──── 今日の話し合い対象のまち"):
        idx = bg.find(marker)
        if idx > 0:
            bg = bg[:idx].strip()
            break
    # シーン文 (build_classroom_ab_config の scene_text) を除去
    # 例: 「放課後、同じクラスのメンバーで教室に集まっている。先生は不在。」
    scene_markers = (
        "放課後、同じクラスのメンバーで",
        "放課後、同じ進路指導クラスのメンバーで",  # 高校生 variant
        "シンギュラボの",  # 社会人 variant
    )
    for sm in scene_markers:
        idx = bg.find(sm)
        if idx >= 0:
            bg = bg[:idx].rstrip("。 ").strip()
            break
    if len(bg) > max_len:
        bg = bg[:max_len].rstrip() + "…"
    return bg


def _diversity_tags(p: dict) -> list[str]:
    """学生の特徴的背景タグ (不適応 / 外国籍 / ジェンダーレス / 車いす)。"""
    tags = []
    if (p.get("school_fit") or "") == "不適応":
        tags.append("学校不適応")
    if p.get("nationality") in ("western", "asian"):
        tags.append("外国籍")
    if p.get("gender") == "other":
        tags.append("ジェンダーレス")
    if p.get("mobility") == "wheelchair":
        tags.append("ハンディキャップ")
    return tags


def render_student_roster(participants: list[dict]) -> str:
    """生徒のみのテーブル。背景は省略せず全文を左揃えで表示。
    本番20人の場合はスライドが分割されてOK。"""
    rows = []
    rows.append(
        '<thead><tr style="text-align:left">'
        '<th style="width:2.5em;text-align:left">id</th>'
        '<th style="width:6em;text-align:left">名前</th>'
        '<th style="width:2.2em;text-align:left">齢</th>'
        '<th style="width:2.5em;text-align:left">性</th>'
        '<th style="width:5em;text-align:left">興味</th>'
        '<th style="width:5em;text-align:left">傾向</th>'
        '<th style="text-align:left">背景</th>'
        '<th style="width:11em;text-align:left">特徴的背景</th>'
        '<th style="width:16em;text-align:left">口癖</th>'
        '</tr></thead>'
    )
    body_rows = []
    for p in participants:
        interest = p.get("interest_tag") or "—"
        tendency = p.get("tendency") or "—"
        catch = (p.get("catchphrase") or "").strip() or "—"
        bg = _bg_summary(p, max_len=10000) or "—"  # 省略しない
        cls = p.get("gender") or "other"
        diversity = _diversity_tags(p)
        if diversity:
            div_html = " / ".join(
                f'<span style="color:#ffd85f">{html.escape(t)}</span>'
                for t in diversity
            )
        else:
            div_html = '<span style="color:#666">—</span>'
        body_rows.append(
            f'<tr class="roster-{cls}" style="text-align:left">'
            f'<td style="text-align:left">{p.get("id","?")}</td>'
            f'<td style="text-align:left">{html.escape(p.get("name","?"))}</td>'
            f'<td style="text-align:left">{p.get("age","?")}</td>'
            f'<td style="text-align:left">{html.escape(_gender_jp(p))}</td>'
            f'<td style="text-align:left">{html.escape(interest)}</td>'
            f'<td style="text-align:left">{html.escape(tendency)}</td>'
            f'<td style="text-align:left;font-size:0.82em;color:#bbb;line-height:1.45">{html.escape(bg)}</td>'
            f'<td style="text-align:left;font-size:0.85em">{div_html}</td>'
            f'<td style="text-align:left;font-size:0.85em">{html.escape(catch)}</td>'
            f'</tr>'
        )
    return (
        '<table class="api-table" style="width:100%;font-size:0.85em;text-align:left">'
        + "".join(rows)
        + '<tbody>' + "".join(body_rows) + '</tbody>'
        + '</table>'
    )


def render_host_roster(hosts: list[dict]) -> str:
    """企業担当者のみのテーブル。担当施設+役職+背景全文。"""
    rows = []
    rows.append(
        '<thead><tr style="text-align:left">'
        '<th style="width:2.5em;text-align:left">id</th>'
        '<th style="width:7em;text-align:left">名前</th>'
        '<th style="width:2.2em;text-align:left">齢</th>'
        '<th style="width:2.5em;text-align:left">性</th>'
        '<th style="width:11em;text-align:left">担当施設</th>'
        '<th style="width:11em;text-align:left">役職</th>'
        '<th style="text-align:left">背景</th>'
        '</tr></thead>'
    )
    body_rows = []
    for p in hosts:
        place = p.get("initial_place") or "—"
        occ = p.get("occupation") or "企業担当者"
        bg = _bg_summary(p, max_len=10000) or "—"  # 省略しない
        body_rows.append(
            f'<tr class="roster-host-row" style="text-align:left">'
            f'<td style="text-align:left">{p.get("id","?")}</td>'
            f'<td style="text-align:left">{html.escape(p.get("name","?"))}</td>'
            f'<td style="text-align:left">{p.get("age","?")}</td>'
            f'<td style="text-align:left">{html.escape(_gender_jp(p))}</td>'
            f'<td style="text-align:left;font-size:0.88em">{html.escape(place)}</td>'
            f'<td style="text-align:left;font-size:0.85em">{html.escape(occ)}</td>'
            f'<td style="text-align:left;font-size:0.82em;color:#bbb;line-height:1.45">{html.escape(bg)}</td>'
            f'</tr>'
        )
    return (
        '<table class="api-table" style="width:100%;font-size:0.85em;text-align:left">'
        + "".join(rows)
        + '<tbody>' + "".join(body_rows) + '</tbody>'
        + '</table>'
    )


def render_persona_roster(participants: list[dict], hosts: list[dict]) -> str:
    """登場人物一覧。生徒テーブルと企業担当者テーブルを分けて出す。"""
    note = (
        f'<div style="color:#888;font-size:0.9em;margin:0.4em 0">'
        f'生徒 {len(participants)}人 / 企業担当者 {len(hosts)}人。'
        f'</div>'
    )
    sections = [note]
    sections.append('<h2 style="margin-top:0.6em">生徒</h2>')
    sections.append(render_student_roster(participants))
    sections.append('<h2 style="margin-top:1em">企業担当者 <span style="color:#888;font-size:0.65em;font-weight:normal">※架空の設定であり、実在する人物とは一切関係しない</span></h2>')
    sections.append(render_host_roster(hosts))
    return "\n".join(sections)


# ---------- Phase C アンケート集計 ----------

QUESTION_LABELS = {
    # (短ラベル, 設問全文, role, kind=numeric|free, 数値設問なら両端の意味)
    "q1": ("Q1 まちの印象変化",
           "座学時と比べて、品川というまちの印象は変わりましたか？",
           "student", "numeric", "1=印象悪化 / 5=変わらず / 10=大幅に向上"),
    "q2": ("Q2 視野が広がった実感",
           "自分の視野が広がった実感はありますか？",
           "student", "numeric", "1=まったくない / 10=強くある"),
    "q3": ("Q3 自分も関われる感",
           "まちに「自分も関われる」と思えた度合いは？",
           "student", "numeric", "1=まったく思えない / 10=強く思える"),
    "q4": ("Q4 座学とFWの違い",
           "座学で描いた未来像と、FW (フィールドワーク) で歩いて見えた現実は、どこが違いましたか？",
           "student", "free", None),
    "q5": ("Q5 教室とまちの学びの違い",
           "教室での学びとまちでの学びの違いを、自分の言葉で書いてください。",
           "student", "free", None),
    "q6": ("Q6 プログラムの良い点 / 悪い点",
           "この教育プログラムの 良かった点 / 悪かった点・改善希望 を一つずつ書いてください。",
           "student", "free", None),
    "q7": ("Q7 学生との対話の手応え",
           "受け入れた学生たちとの対話の手応えは？",
           "host", "numeric", "1=まったく響かなかった / 10=深く議論できた"),
    "q8": ("Q8 自社にとっての価値",
           "このプログラムは、自社にとって価値ある取り組みでしたか？",
           "host", "numeric", "1=まったく価値なし / 10=ぜひ継続したい"),
    "q9": ("Q9 連携の良かった点",
           "学生を受け入れてみて気づいた、まちと学校の連携の 良かった点 を書いてください。",
           "host", "free", None),
    "q10": ("Q10 改善希望",
            "改善してほしい点・運営上の懸念があれば書いてください。",
            "host", "free", None),
}


def _is_meaningful_evidence(s: str) -> bool:
    """evidence の内容が具体性を持つかの簡易チェック。空 or 抽象語のみは False。"""
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) < 12:  # 短すぎ
        return False
    abstract_patterns = ["いろいろ", "なんとなく", "色々", "全体的", "総合的", "特になし"]
    if any(p in s and len(s) < 25 for p in abstract_patterns):
        return False
    return True


def _render_num_card(k: str, rs: list) -> str:
    short, body, role, kind, scale = QUESTION_LABELS[k]
    vals = []
    excluded = 0
    evidence_key = f"{k}_evidence"
    for r in rs:
        if r.get("role") != role: continue
        responses = r.get("responses") or {}
        v = responses.get(k)
        if isinstance(v, (int, float)) and 1 <= v <= 10:
            ev = responses.get(evidence_key, "")
            if _is_meaningful_evidence(ev):
                vals.append(v)
            else:
                excluded += 1
    full_label = (
        f"{short}<br>"
        f"<span style=\"font-size:0.85em;color:#aaa\">{html.escape(body)}</span>"
        f"<span style=\"font-size:0.75em;color:#888;margin-left:0.4em\">(10段階評価)</span>"
    )
    scale_line = f"<div style=\"font-size:0.75em;color:#888;margin-top:0.2em\">{html.escape(scale)}</div>" if scale else ""
    if not vals:
        return (f'<div class="survey-num-card"><div class="label">{full_label}</div>'
                f'<div class="value">—</div><div class="breakdown">回答なし</div>{scale_line}</div>')
    avg = sum(vals) / len(vals)
    median = statistics.median(vals)
    cls = "ok" if avg >= 7 else ("warn" if avg >= 5 else "bad")
    cnt = Counter(vals)
    breakdown = " / ".join(f"{kk}={cnt[kk]}" for kk in sorted(cnt))
    excl_html = f' / <span style="color:#888">evidence不足で除外: {excluded}</span>' if excluded else ""
    return (
        f'<div class="survey-num-card {cls}">'
        f'<div class="label">{full_label}</div>'
        f'<div class="value">{avg:.2f}</div>'
        f'<div class="breakdown">中央値 {median:.1f} / n={len(vals)} / 内訳: {breakdown}{excl_html}</div>'
        f'{scale_line}'
        f'</div>'
    )


def _load_survey_lookup(survey_path: Path) -> dict:
    """survey_responses.jsonl を読み込んで id / axis_id / name で引ける dict を返す。"""
    if not survey_path or not survey_path.exists():
        return {}
    by_id = {}
    by_axis = {}
    by_name = {}
    for line in survey_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("id") is not None:
            by_id[r["id"]] = r
        if r.get("axis_id"):
            by_axis[r["axis_id"]] = r
        if r.get("name"):
            by_name[r["name"]] = r
    return {"by_id": by_id, "by_axis": by_axis, "by_name": by_name}


def _render_survey_for_agent(p: dict, survey_lookup: dict) -> str:
    """各 agent の Phase C アンケート回答を 1 ブロックで返す (logs HTML 末尾用)。"""
    if not survey_lookup:
        return ""
    rec = (
        survey_lookup.get("by_id", {}).get(p.get("id"))
        or survey_lookup.get("by_axis", {}).get(p.get("axis_id"))
        or survey_lookup.get("by_name", {}).get(p.get("name"))
    )
    if not rec:
        return (
            '<div style="margin-top:0.6em;padding:0.6em;background:#1a1d22;border-left:3px solid #555;color:#888;font-size:0.85em">'
            '本人のアンケート回答: (回答記録なし)'
            '</div>'
        )
    role = rec.get("role")
    responses = rec.get("responses") or {}
    if not responses:
        return (
            '<div style="margin-top:0.6em;padding:0.6em;background:#1a1d22;border-left:3px solid #555;color:#888;font-size:0.85em">'
            '本人のアンケート回答: (空)'
            '</div>'
        )
    if role == "student":
        order = ["q1", "q2", "q3", "q4", "q5", "q6"]
    else:
        order = ["q7", "q8", "q9", "q10"]
    rows = []
    for k in order:
        if k not in responses:
            continue
        meta = QUESTION_LABELS.get(k)
        short = meta[0] if meta else k
        body = meta[1] if meta else ""
        v = responses.get(k)
        ev = responses.get(f"{k}_evidence", "")
        if isinstance(v, (int, float)) and 1 <= v <= 10:
            v_html = (
                f'<span style="display:inline-block;min-width:1.6em;text-align:center;'
                f'background:#2a3340;border-radius:3px;padding:0.05em 0.35em;color:#ffd85f;font-weight:bold">'
                f'{int(v)}</span> <span style="color:#888">/10</span>'
            )
        else:
            v_html = html.escape(str(v))
        ev_html = ""
        if ev:
            ev_html = f'<div style="color:#bbb;margin-top:0.2em">理由: {html.escape(str(ev))}</div>'
        rows.append(
            '<div style="margin-bottom:0.4em">'
            f'<div style="color:#9bdfff;font-weight:600">{html.escape(short)}</div>'
            f'<div style="color:#888;font-size:0.86em;margin-bottom:0.15em">{html.escape(body)}</div>'
            f'<div>{v_html}</div>'
            f'{ev_html}'
            '</div>'
        )
    if not rows:
        return ""
    return (
        '<div style="margin-top:0.8em;padding:0.7em 0.9em;background:#1a1d22;'
        'border-left:3px solid #ffd85f;font-size:0.88em;line-height:1.5">'
        '<div style="color:#ffd85f;font-weight:bold;margin-bottom:0.4em">📝 本人のアンケート回答 (Phase C)</div>'
        + "".join(rows)
        + '</div>'
    )


def _affiliation_for_axis(ax: str, persona_by_axis: dict | None) -> str:
    """axis_id (Host_Keikyu 等) を「京急電鉄 / 京急品川」のような名称に変換。
    persona が見つからなければ axis_id そのまま (フォールバック)。"""
    if not persona_by_axis:
        return ax
    p = persona_by_axis.get(ax)
    if not p:
        return ax
    occ = (p.get("occupation") or "").strip()
    place = (p.get("initial_place") or "").strip()
    company = occ.split()[0] if occ else ""
    if company and place and company != place:
        return f"{company} / {place}"
    if occ:
        return occ
    if place:
        return place
    return ax


def _render_num_individual(k: str, rs: list, persona_by_axis: dict | None = None) -> str:
    """数値設問 k の各エージェントの点数 + 理由を表で返す。隠さない方針。"""
    short, body, role, kind, _scale = QUESTION_LABELS[k]
    items = []
    for r in rs:
        if r.get("role") != role:
            continue
        responses = r.get("responses") or {}
        v = responses.get(k)
        if not isinstance(v, (int, float)) or not (1 <= v <= 10):
            continue
        ev = responses.get(f"{k}_evidence", "") or ""
        items.append((float(v), r.get("name", "?"), r.get("axis_id", ""), ev.strip()))
    if not items:
        return ""
    items.sort(key=lambda x: (-x[0], x[2]))
    rows = []
    for v, nm, ax, ev in items:
        cls = "hi" if v >= 8 else ("lo" if v <= 4 else "")
        meaningful = _is_meaningful_evidence(ev)
        ev_class = "" if meaningful else " thin"
        ev_text = html.escape(ev) if ev else "(理由記載なし)"
        # host は所属名で、学生は axis_id を出さない
        if role == "host":
            aff = _affiliation_for_axis(ax, persona_by_axis)
            who_html = (
                f'{html.escape(nm)}'
                f'<span class="ax" style="font-weight:normal">{html.escape(aff)}</span>'
            )
        else:
            who_html = html.escape(nm)
        rows.append(
            f'<tr class="{cls}">'
            f'<td class="score">{int(v)}</td>'
            f'<td class="who">{who_html}</td>'
            f'<td class="reason{ev_class}">{ev_text}</td>'
            f'</tr>'
        )
    table = (
        '<table class="indiv-table">'
        '<thead><tr><th style="width:2.6em;text-align:center">点</th>'
        '<th style="width:11em">回答者</th><th>理由 (本人発話)</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    return (
        f'<div class="indiv-block">'
        f'<div class="indiv-head">{html.escape(short)} の各人の点数と理由 (n={len(items)})</div>'
        f'<div class="indiv-q">設問: 「{html.escape(body)}」</div>'
        f'{table}'
        f'</div>'
    )


def _render_free_block(k: str, rs: list, persona_by_axis: dict | None = None) -> str:
    short, body, role, kind, _ = QUESTION_LABELS[k]
    items = []
    for r in rs:
        if r.get("role") != role: continue
        v = (r.get("responses") or {}).get(k)
        if isinstance(v, str) and v.strip():
            items.append((r.get("name", "?"), r.get("axis_id", ""), v.strip()))
    if not items:
        return ""
    parts = [
        f'<h4 style="margin-top:1.2em">{html.escape(short)}</h4>',
        f'<div style="color:#aaa;font-size:0.9em;margin-bottom:0.6em">設問: 「{html.escape(body)}」</div>',
    ]
    for name, axis, body_text in items:
        if role == "host":
            aff = _affiliation_for_axis(axis, persona_by_axis)
            who = f'{html.escape(name)} <span style="color:#888">({html.escape(aff)})</span>'
        else:
            who = html.escape(name)
        parts.append(
            f'<div class="free-resp"><div class="who">{who}</div>'
            f'<div class="body">{html.escape(body_text)}</div></div>'
        )
    return "\n".join(parts)


def render_survey_section(survey_path: Path, persona_by_axis: dict | None = None) -> str:
    if not survey_path.exists():
        return '<div class="empty">(survey_responses.jsonl が見つかりません)</div>'
    rs = []
    for line in survey_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rs.append(json.loads(line))

    # 学生の数値カード (Q1-Q3) と 企業の数値カード (Q7-Q8) を別グリッドに
    student_num_cards = "".join(_render_num_card(k, rs) for k in ("q1", "q2", "q3"))
    host_num_cards = "".join(_render_num_card(k, rs) for k in ("q7", "q8"))

    # 各設問の per-agent 「点数 + 理由」表 (集計の下に配置、隠さない方針)
    student_indiv = "\n".join(
        _render_num_individual(k, rs, persona_by_axis)
        for k in ("q1", "q2", "q3")
        if _render_num_individual(k, rs, persona_by_axis)
    )
    host_indiv = "\n".join(
        _render_num_individual(k, rs, persona_by_axis)
        for k in ("q7", "q8")
        if _render_num_individual(k, rs, persona_by_axis)
    )

    # 自由記述: 学生 (Q4-Q6) と 企業 (Q9-Q10) を別ブロックに
    student_free = "\n".join(
        _render_free_block(k, rs, persona_by_axis)
        for k in ("q4", "q5", "q6")
        if _render_free_block(k, rs, persona_by_axis)
    )
    host_free = "\n".join(
        _render_free_block(k, rs, persona_by_axis)
        for k in ("q9", "q10")
        if _render_free_block(k, rs, persona_by_axis)
    )

    n_total = len(rs)
    n_student = sum(1 for r in rs if r.get("role") == "student" and r.get("responses"))
    n_host = sum(1 for r in rs if r.get("role") == "host" and r.get("responses"))
    summary = (
        f'<div class="exp-meta"><ul>'
        f'<li>回答者: {n_total} (生徒 {n_student} / 企業担当者 {n_host})</li>'
        f'<li>設問: 生徒 Q1-Q6 (3 数値+3 自由記述) / 企業担当者 Q7-Q10 (2 数値+2 自由記述)</li>'
        f'<li>数値設問は全て 10 段階評価。両端の意味は各カード下部に記載。</li>'
        f'</ul></div>'
    )

    # 順序: 生徒数値 → 生徒自由 → 企業担当者数値 → 企業担当者自由
    sections = [summary]
    sections.append('<h3>【生徒】数値設問 (10段階評価)</h3>')
    sections.append(f'<div class="survey-num-grid">{student_num_cards}</div>'
                    if student_num_cards else '<div class="empty">(生徒の数値回答なし)</div>')
    if student_indiv:
        sections.append('<h4 style="margin-top:1.2em;color:#aaa">各人の点数と理由</h4>')
        sections.append(student_indiv)
    sections.append('<h3>【生徒】自由記述</h3>')
    sections.append(student_free or '<div class="empty">(生徒の自由記述なし)</div>')
    sections.append('<h3>【企業担当者】数値設問 (10段階評価)</h3>')
    sections.append(f'<div class="survey-num-grid">{host_num_cards}</div>'
                    if host_num_cards else '<div class="empty">(企業担当者の数値回答なし)</div>')
    if host_indiv:
        sections.append('<h4 style="margin-top:1.2em;color:#aaa">各人の点数と理由</h4>')
        sections.append(host_indiv)
    sections.append('<h3>【企業担当者】自由記述</h3>')
    sections.append(host_free or '<div class="empty">(企業担当者の自由記述なし)</div>')

    return "\n".join(sections)


# ---------- Phase B セクション (analysis + timeline + highlights) ----------

def render_phaseB_section(fw_data: dict, handoffs: dict, participants: list[dict],
                           hosts: list[dict], client: GeminiClient, workers: int = 4) -> tuple[str, str, dict]:
    """Phase B FW のセクション HTML 3本 (analysis_html, timeline_html, highlights_html) を返す。"""
    id_to_name = {a["id"]: a.get("name", f"#{a['id']}") for a in fw_data["personas"]}
    cfg = fw_data["cfg"]
    sim = cfg.get("simulation", {})
    duration = sim.get("duration", 30)
    ts = sim.get("time_scale", {})
    mps = ts.get("step_duration_minutes", 2)
    start_time = ts.get("start_time", "13:00")
    try:
        sh = int(str(start_time).split(":")[0])
    except Exception:
        sh = 13

    # 学生のみ Gemini 分析 (host は省略コスト削減)
    def analyze_one(p):
        ax = p.get("axis_id")
        thoughts, sent, recv = per_agent_history(fw_data, p["id"])
        full_rows = build_full_history_rows(fw_data, p["id"], id_to_name=id_to_name)
        prompt = build_phaseB_prompt(
            p, handoffs["handoff_by_axis"].get(ax),
            handoffs["fq_by_axis"].get(ax), thoughts, sent, recv,
        )
        try:
            md = call_gemini_md(client, SYSTEM_ANALYSIS, prompt, max_tokens=1500)
        except Exception as e:
            md = f"(分析失敗: {e})"
        return p, md, full_rows

    print(f"[info] analyzing {len(participants)} students in parallel ...")
    student_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_one, p) for p in participants]
        for f in as_completed(futures):
            p, md, fr = f.result()
            print(f"  done: {p.get('name')}")
            student_results.append((p, md, fr))
    student_results.sort(key=lambda r: r[0].get("axis_id", ""))

    # host も Gemini で「現地で得た気づき / 一行で言うと」の短い分析を並列生成。
    def analyze_host(h):
        thoughts, sent, recv = per_agent_history(fw_data, h["id"])
        full_rows = build_full_history_rows(fw_data, h["id"], id_to_name=id_to_name)
        prompt = build_phaseB_host_prompt(h, sent, recv, thoughts)
        try:
            md = call_gemini_md(client, SYSTEM_ANALYSIS_HOST, prompt, max_tokens=900)
        except Exception as e:
            md = f"(分析失敗: {e})"
        return h, md, full_rows

    print(f"[info] analyzing {len(hosts)} hosts in parallel ...")
    host_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_host, h) for h in hosts]
        for f in as_completed(futures):
            h, md, fr = f.result()
            print(f"  done: {h.get('name')} (host)")
            host_results.append((h, md, fr))
    host_order = {h["id"]: i for i, h in enumerate(hosts)}
    host_results.sort(key=lambda r: host_order.get(r[0]["id"], 999))

    # host 用の id set (会話相手のラベル付与で使う)
    host_id_set = {h["id"] for h in hosts}

    # 本文では全 step ログは出さない (バックデータ別 HTML に出す)。
    # 「現地で得た気づき」セクション内に「移動経路+主な会話相手」+ Gemini 分析 を統合。
    analysis_parts = ["<h3>生徒の言語化分析 (座学 → FW の変化)</h3>"]
    analysis_parts += [
        render_analysis_card(p, md, fr, include_history=False, host_id_set=host_id_set)
        for p, md, fr in student_results
    ]
    if hosts:
        analysis_parts.append("<h3>企業担当者 の現地での動き</h3>")
        for h, md, fr in host_results:
            analysis_parts.append(
                render_analysis_card(h, md, fr, include_history=False, host_id_set=host_id_set)
            )
    analysis_html = "\n".join(analysis_parts)

    # 全 agent (学生+host) でタイムライン (host も dot に出る)
    all_for_timeline = participants + hosts
    timeline_html = build_phaseB_interactive_timeline(
        fw_data, all_for_timeline, int(duration) if str(duration).isdigit() else 30,
        start_hour=sh, mins_per_step=mps,
    )

    cand = extract_highlight_candidates(fw_data, max_pre=40)
    highlights = select_highlights(client, cand) if cand else []
    highlights_html = render_highlights_section(highlights)

    return analysis_html, timeline_html, highlights_html, student_results, host_results


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="Phase A (座学) run dir")
    ap.add_argument("--run-b", required=True, help="Phase B (FW) run dir")
    ap.add_argument("--survey", default=None,
                    help="survey_responses.jsonl path (default: <run-b>/survey_responses.jsonl)")
    ap.add_argument("--out", default=None,
                    help="HTML output path (default: <run-b>/v3_report_<NN>.html where NN is run number)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-a", default=None, help="Phase A sim log (token usage)")
    ap.add_argument("--log-b", default=None, help="Phase B sim log (token usage)")
    ap.add_argument("--suffix", default="", help="出力3点セット (v3_report/v3_logs/viewer_3d_bundled) のファイル名末尾に追加する文字列。例: '_v2' で <NN>_v2 になる。")
    args = ap.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    fw_data = load_run(run_b)
    handoffs = load_handoffs(run_a)

    # 3点セット (本文/ログ/3D) のファイル名末尾に Phase B run の通し番号を付与する。
    # run_dir 形式: <YYYY-MM-DD>_<HHMM>_<NNN>_<name...>
    # → parts[2] が通し番号 (例: 2026-05-04_2306_128_classroom_... → "128")
    def _extract_run_number(p: Path) -> str:
        parts = p.name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            return parts[2]
        return ""
    run_num = _extract_run_number(run_b)
    base_suffix = f"_{run_num}" if run_num else ""
    suffix = base_suffix + (args.suffix or "")

    participants = [
        p for p in fw_data["personas"]
        if p.get("axis_id") and not str(p.get("axis_id")).startswith("Host_")
        and not p.get("is_host") and p.get("axis_id") not in ("Sato", "AIRobo", "AIGod", "UMA")
    ]
    hosts = [p for p in fw_data["personas"] if p.get("is_host") or str(p.get("axis_id", "")).startswith("Host_")]
    print(f"[info] participants: {len(participants)} / hosts: {len(hosts)}")

    client = GeminiClient(enable_structured_output=False, enable_cache=False)
    t0 = time.time()

    # ---------- Sections ----------
    print("[info] rendering Phase A section ...")
    phaseA_html = render_phaseA_section(run_a, handoffs, participants)

    print("[info] rendering Phase B section ...")
    pb_analysis, pb_timeline, pb_highlights, student_results, host_results = render_phaseB_section(
        fw_data, handoffs, participants, hosts, client, workers=args.workers,
    )

    print("[info] rendering Phase C survey section ...")
    survey_path = Path(args.survey) if args.survey else (run_b / "survey_responses.jsonl")
    # axis_id → persona の lookup を hosts から作って所属名表示に使う
    persona_by_axis = {p.get("axis_id"): p for p in fw_data["personas"] if p.get("axis_id")}
    survey_html = render_survey_section(survey_path, persona_by_axis)

    print("[info] rendering overall reflection (Gemini) ...")
    reflection_html = render_overall_reflection(participants, hosts, handoffs, fw_data, survey_path, client, run_b=run_b)

    analysis_dur = int(time.time() - t0)

    # API meta
    log_a_path = Path(args.log_a) if args.log_a else None
    log_b_path = Path(args.log_b) if args.log_b else None
    # Phase A/B の各 run_dir 内に保存された sim.log (main.py が書き出す) を最優先で読む
    if not log_a_path:
        cand = run_a / "sim.log"
        if cand.exists():
            log_a_path = cand
    if not log_b_path:
        cand = run_b / "sim.log"
        if cand.exists():
            log_b_path = cand
    stats_a = parse_sim_log(log_a_path) if log_a_path else None
    stats_b = parse_sim_log(log_b_path) if log_b_path else None
    api_table_a = render_api_table(stats_a, 0, len(participants), sim_label="座学シミュ") if stats_a else '<div class="empty">(Phase A の API ログがありません — log-a を指定すれば集計します)</div>'
    api_table_b = render_api_table(stats_b, analysis_dur, len(participants), sim_label="FW シミュ")
    # Phase C は run_survey.py の Gemini call (生徒+host で20-22回) で誤差レベル → 表示省略
    api_table = (
        '<h3 style="margin-top:0">Phase A — 教室での座学</h3>'
        + api_table_a
        + '<h3 style="margin-top:1em">Phase B — フィールドワーク</h3>'
        + api_table_b
    )

    # ---------- Compose HTML ----------
    cfg_b = fw_data["cfg"]
    meta_b = cfg_b.get("metadata", {})
    sim_b = cfg_b.get("simulation", {})
    ts_b = sim_b.get("time_scale", {})
    duration = sim_b.get("duration", "?")
    mps = ts_b.get("step_duration_minutes", "?")
    start_time = ts_b.get("start_time", "?")
    seed = sim_b.get("seed", "?")
    variant = meta_b.get("variant", "elementary")
    variant_label = {"high": "高校生版", "adult": "社会人版", "elementary": "小学生版", "junior_high": "中学生版"}.get(variant, variant)
    llm_model = (cfg_b.get("llm", {}) or {}).get("model", "?")

    # Phase A の time_scale も config から動的に取り出す (時間表記をハードコードから動的計算に)
    try:
        cfg_a_full = yaml.safe_load((run_a / "config.yaml").read_text(encoding="utf-8"))
        sim_a = (cfg_a_full or {}).get("simulation", {})
        ts_a = sim_a.get("time_scale", {})
        duration_a = sim_a.get("duration")
        mps_a = ts_a.get("step_duration_minutes", 2)
        start_a = str(ts_a.get("start_time", "10:00"))
    except Exception:
        duration_a, mps_a, start_a = None, 2, "10:00"

    def _time_range_str(start_hhmm: str, total_minutes: int) -> str:
        try:
            sh, sm = start_hhmm.split(":")
            sh, sm = int(sh), int(sm)
        except Exception:
            return "?"
        end_total = sh * 60 + sm + total_minutes
        eh = (end_total // 60) % 24
        em = end_total % 60
        return f"{sh:02d}:{sm:02d} 〜 {eh:02d}:{em:02d}"

    if duration_a and mps_a:
        a_total_min = int(duration_a) * int(mps_a)
        a_time_range = _time_range_str(start_a, a_total_min)
        a_phase_str = f"{a_time_range} の {a_total_min} 分 (= {mps_a}分／ステップ × {duration_a} ステップ)"
    else:
        a_phase_str = "(Phase A 時間情報取得失敗)"

    if str(duration).isdigit() and str(mps).isdigit():
        b_total_min = int(duration) * int(mps)
        b_time_range = _time_range_str(str(start_time), b_total_min)
        b_phase_str = f"{b_time_range} の {b_total_min} 分 (= {mps}分／ステップ × {duration} ステップ)"
        b_end_time = b_time_range.split(" 〜 ")[-1] if " 〜 " in b_time_range else "?"
    else:
        b_phase_str = "(Phase B 時間情報取得失敗)"
        b_end_time = "?"
    goal = ""
    if participants:
        goal = (participants[0].get("current_goal") or "").strip()

    title = "まちと企業がつくる学習環境は、子どもに何をもたらすのか"
    sub_caption = f"企業協力型 学外教育プログラム を題材とした LLM マルチエージェント・シミュレーション ({variant_label})"
    sub_title = f"シミュレーションレポート ({variant_label})"
    if str(duration).isdigit() and str(mps).isdigit():
        total_min_str = f"{int(duration)*int(mps)}分"
    else:
        total_min_str = "?"

    intro = f"""
<section id="section-cover">
<h1 style="font-size:1.7em;line-height:1.35;margin-bottom:0.2em">{html.escape(title)}</h1>
<div style="font-size:0.95em;color:#bbb;margin-bottom:1em;line-height:1.5">{html.escape(sub_caption)}</div>
<div style="font-size:0.92em;color:#bbb;margin-bottom:1em">
<strong>提出者（チーム）</strong>: えびやま
<div style="margin-top:0.3em;color:#aaa">
　・やまちゃそ <span style="color:#888">(専門: まちづくり・都市開発)</span><br>
　・えびねこ <span style="color:#888">(専門: 教育)</span>
</div>
</div>

<div class="exp-meta" style="line-height:1.7">
<p><strong>本シミュレーションのテーマ</strong><br>
「<strong>教育 × まち × 企業</strong>」のカケザンで生まれる、<strong>新しい学びの形</strong> の検証・シミュレーション<br>
学校の外にあるまち・企業・インフラ・公共空間を、子どもたちの学びを支える環境として捉え直し、
教育をまちに展開したときに、子どもたちの <em>視野・問い・主体感・参加スタイル</em> にどのような変化が
生まれるのかを検証する。</p>

<p style="margin-top:0.9em"><strong>本シミュレーションの狙い</strong><br>
品川駅周辺での「<strong>企業協力型 学外教育プログラム</strong>」を例題に、
教育をまちに展開したときに起きることや、学習環境として機能するために必要なこと等の気づきを得ることを目的とする。
具体的には以下の3点に着目する。</p>
<ol>
<li><strong>多様な子どもたちの参加スタイルが成立する学習環境になっているか</strong><br>
学校適応度・国籍・身体特性・関心領域などの差を持つ多様な子どもたちが、
それぞれの特性に応じた参加スタイルで学びに加われているかを検証する。
特に、自分らしいやり方で関わる余地があるか、特定タイプの子どもだけが活躍してしまっていないか、
多様な背景を持つ子どもならではの視点が学びの中で活かされているかに着目する。</li>
<li><strong>子どもたちの変化</strong><br>
まち全体を学び場にしたとき、子どもの <em>視野・視座・問い・行動意欲</em> がどう変わるかを検証する。
特に、学校への適応感が低い子や、教室内の学びに参加しづらさを感じる子が、まちを学びの場にしたときに、
どのような参加の仕方を見せるのか、また新たな関心や気づきが生まれる可能性があるのかを確認する。
また、まちの良いところや課題を知るだけでなく、<strong>自分もまちに関われる</strong>という感覚が
生まれるかにも着目する。</li>
<li><strong>企業や地域と連携した学習環境にするために何が必要か</strong><br>
企業担当者や地域の大人との対話が、単なる企業訪問や説明の受け取りに留まらず、
子どもの気づきや違和感を <em>問い</em> へと深める機会になるかを検証する。
また、まちを学び場として機能させるために、どのような設計要素が必要かを考察する。
たとえば、子どもが安心して移動・滞在できるルート、地域の大人や企業担当者と出会える接点、
興味や違和感を記録する仕組み、振り返りの場、企業や地域からのフィードバックの仕組み等に着目する。</li>
</ol>
</div>
</section>

<section class="page-break" id="section-overview">
<h1 style="font-size:1.45em">{html.escape(sub_title)}</h1>
<div class="exp-meta" style="line-height:1.7">
<p><strong>本シミュレーションの構成 (3 フェーズ)</strong>:</p>
<ul>
<li><strong>A. まちの特徴についての座学</strong>: 教室で品川の歴史・地形・整備方針などを学び、生徒は事前情報をもとに「品川の未来像」と「FW で確かめたいこと」を考える。</li>
<li><strong>B. 実際にまちを歩いてまちのプレイヤーと対話しながら学びを深めるフィールドワーク</strong>: 品川駅周辺を歩き、12社の企業担当者と対話する。最低 2社以上の訪問を課題とする。</li>
<li><strong>C. 当プログラムについてのアンケート</strong>: プログラム終了後、生徒・企業担当者それぞれが本プログラムについて 10段階評価および自由記述で回答する。</li>
</ul>

<p><strong>規模</strong>:
生徒 <strong>{len(participants)}人</strong> ({variant_label}) ／
企業担当者 <strong>{len(hosts)}人</strong> (品川エリアの 12社)</p>

<p><strong>各フェーズの時間設計</strong>:</p>
<ul style="margin-top:0.2em">
<li><strong>Phase A (まちの特徴についての座学)</strong>: {a_phase_str}。教室にて品川の事前情報を共有して議論。</li>
<li><strong>Phase B (まちでのフィールドワーク)</strong>: {b_phase_str}。品川駅前 (東西自由通路) からスタートし、まちを歩いて 12 社の企業担当者と対話。最終ステップ ({b_end_time}) には集合場所に戻る。</li>
<li><strong>Phase C (アンケート)</strong>: プログラム終了後、生徒・企業担当者それぞれに 10 段階評価＋自由記述で回答。</li>
</ul>
</div>
<div class="intro-question">
<div class="label" style="font-size:0.8em;color:#a89e80">課題 (生徒に提示する問い)</div>
{html.escape(goal) if goal else "(未設定)"}
</div>
</section>
"""

    roster_html = render_persona_roster(participants, hosts)
    section_roster = f"""
<section class="page-break" id="section-roster">
<h1>登場人物一覧</h1>
{roster_html}
</section>
"""

    # Phase A/B の時間表記 (動的) を「ステップ」列に流す
    step_caption_dyn = (
        f"1 step = {mps_a} 分。"
        f"Phase A {duration_a} step = {(int(duration_a)*int(mps_a)) if duration_a and mps_a else '?'}分 ({a_time_range if duration_a else '?'} 座学) / "
        f"Phase B {duration} step = {(int(duration)*int(mps)) if str(duration).isdigit() and str(mps).isdigit() else '?'}分 ({b_time_range if str(duration).isdigit() else '?'} FW)。"
    )

    section_a = f"""
<section class="page-break" id="section-phase-a">
<h1>Phase A — 教室での座学</h1>
{render_agent_rules('A', step_caption=step_caption_dyn)}
{phaseA_html}
</section>
"""

    section_b = f"""
<section class="page-break" id="section-phase-b">
<h1>Phase B — 品川駅前フィールドワーク</h1>
<h2>全体活動タイムライン</h2>
{render_agent_rules('B', step_caption=step_caption_dyn)}
{pb_timeline}

<h2>各参加者の言語化分析</h2>
{pb_analysis}
</section>
"""

    section_c = f"""
<section class="page-break" id="section-phase-c">
<h1>Phase C — シミュ後アンケート</h1>
{survey_html}
</section>
"""

    section_meta = f"""
<section class="page-break" id="section-api">
<h1>API消費・所要時間 (参考)</h1>
{api_table}
<div class="api-meta" style="margin-top:0.8em">
<dl>
<dt>レポート生成日時</dt><dd>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</dd>
<dt>分析・レポート所要</dt><dd>{fmt_duration(analysis_dur)}</dd>
</dl>
</div>
</section>
"""

    # ---------- 全 step ログ別 HTML (バックデータ) ----------
    logs_path = run_b / f"v3_logs{suffix}.html"
    logs_title = f"全 step ログ (バックデータ) — {variant_label}"
    logs_parts = [
        f'<h1>{html.escape(logs_title)}</h1>',
        '<p style="color:#bbb">レポート本文のバックデータ。Phase A (座学) と Phase B (FW) の'
        '全 step を、思考 (memory + reasoning)、移動、同行者、発話、受信メッセージ単位で展開しています。</p>',
    ]

    # Phase A の全ログ (生徒のみ)
    logs_parts.append('<h1 style="margin-top:1.5em;border-top:2px solid #444;padding-top:0.6em">Phase A — 教室での座学</h1>')
    try:
        data_a = load_run(run_a)
        id_to_name_a = {a["id"]: a.get("name", f"#{a['id']}") for a in data_a["personas"]}
        for p in participants:
            try:
                fr_a = build_full_history_rows(data_a, p["id"], id_to_name=id_to_name_a)
            except Exception:
                fr_a = []
            logs_parts.append(
                f'<div class="analysis-card {p.get("gender","")}">'
                f'<div class="agent-name">{html.escape(_agent_label(p))}</div>'
                f'{render_history_details(fr_a, show_movement=False)}'
                f'</div>'
            )
    except Exception as e:
        logs_parts.append(f'<p style="color:#888">(Phase A ログ取得失敗: {html.escape(str(e))})</p>')

    # Phase B の全ログ (生徒+企業担当者) — アンケートはレポート本文側に既出のため logs には入れない
    logs_parts.append('<h1 style="margin-top:1.5em;border-top:2px solid #444;padding-top:0.6em">Phase B — フィールドワーク</h1>')
    logs_parts.append('<h2>生徒</h2>')
    host_id_set = {h["id"] for h in hosts}
    for p, md, fr in student_results:
        logs_parts.append(render_analysis_card(p, md, fr, include_history=True, host_id_set=host_id_set))
    logs_parts.append('<h2 style="margin-top:1em">企業担当者</h2>')
    for h, md, fr in host_results:
        logs_parts.append(render_analysis_card(h, md, fr, include_history=True, host_id_set=host_id_set))
    logs_body = HEAD.format(title=html.escape(logs_title)) + "\n".join(logs_parts) + FOOT
    logs_path.write_text(logs_body, encoding="utf-8")
    print(f"[ok] wrote logs (back data): {logs_path} ({len(logs_body)/1024:.0f} KB)")

    # 本文末尾にバックデータへのリンクを差し込む
    backdata_link = (
        f'<section id="section-backdata"><h2>全ステップログ (バックデータ)</h2>'
        f'<p>Phase A・Phase B の各 agent 全 step ログ (思考・移動・同行・発話・受信) は、'
        f'分量が大きいため別 HTML に分離しています。</p>'
        f'<p>→ <a href="{logs_path.name}" style="color:#ffd85f">{html.escape(logs_path.name)}</a></p>'
        f'</section>'
    )

    section_reflection = f"""
<section class="page-break" id="section-reflection">
<h1>全体を通しての考察</h1>
{reflection_html}
</section>
"""

    # チームメンバー えびねこ (教育観点) からの補足考察
    section_ebineko = """
<section class="page-break" id="section-ebineko">
<h1>えびねこ考察（教育観点）</h1>

<h2>1. 前提：「社会に開かれた教育」のその先へ</h2>
<p>
今回、私たちは高校生を対象に、品川駅周辺を題材とした学外教育プログラムのシミュレーションを行った。
これは、単なるまち歩きや企業訪問ではなく、学校の外にあるまち・企業・インフラ・公共空間を、
子どもたちの学びを支える環境として捉え直す試みである。
</p>
<p style="margin-top:0.5em">
この考え方は、文部科学省が掲げる<strong>「社会に開かれた教育課程」</strong>の方向性と重なる。
文部科学省は、社会とのつながりの中で学ぶことで、子どもたちが
「自分の力で人生や社会をよりよくできる」 という実感を持つことの重要性を示している。
</p>
<p style="margin-top:0.5em">
私たちはこの方向性を前提にしながら、これからの教育では、学校が社会と連携するだけでなく、
<strong>まちそのものが子どもたちの学びを支える環境として機能していく可能性</strong>があると考えている。
まちは、社会の仕組みを実感し、自分なりの問いを持ち、企業や地域の大人と対話しながら考えを深める場になり得る。
</p>
<p style="margin-top:0.5em">
今回のシミュレーションでは、品川駅周辺を一つの例として、まち全体を学習環境として捉えたときに、
子どもたちの視野や参加の仕方にどのような変化が生まれるのかを検討した。
これは、社会に開かれた教育の方向性を大切にしながら、
<strong>まち・企業・地域が一体となって子どもの学びを支える教育環境</strong>の可能性と課題を探る第一歩である。
</p>

<h2>2. まちは、知識を実感に変える学習環境になり得る</h2>
<p>
生徒には、まちの良いところや課題を様々な視点で探し、未来像を描く問いを提示し、
座学、まち歩き、企業担当者との対話、アンケートまでを一連の学習プロセスとして設計した。
</p>
<p style="margin-top:0.5em">
その結果、<strong>まちは教室で得た知識を、より具体的な実感へとつなげる学習環境になり得る</strong>という手応えを得た。
生徒アンケートでは、まちの印象変化や視野の広がりについて一定の結果が見られた。
座学だけでなく、実際にまちを歩き、企業担当者と対話することで、生徒の見方が広がる可能性が示された。
</p>
<p style="margin-top:0.5em">
ここで重要なのは、まちが単なる活動の背景ではなく、社会の仕組みを具体的に感じられる場所になっていた点である。
交通、通信、税、水、食、観光、働く場、バリアフリーなど、教室で聞いた知識が、
現地の風景や人の流れ、企業担当者の言葉、インフラの存在と結びつくことで、
子どもたちの中で「知っていること」が「自分の目で確かめたこと」へと変化していた。
</p>

<h2>3. 「知る」から「関わる」への転換には、もう一段の設計が必要である</h2>
<p>
一方で、今回の結果から、私たちは大きな課題も感じた。
それは、まちを見る視野は広がったものの、<strong>「自分もまちに関われる」という主体感までは十分に高まりきらなかった</strong>ことである。
</p>
<p style="margin-top:0.5em">
生徒アンケートでは、「まちに自分も関われる感」が、まちの印象変化や視野の広がりに比べて低い結果になっていた。
これは、まちに出て企業の話を聞くことで、生徒は「まちを知る」ことはできる一方で、
それだけではまだ都市の観察者に留まりやすいことを示している。
</p>
<p style="margin-top:0.5em">
文部科学省が示す「自分の力で人生や社会をよりよくできるという実感」につなげるには、
まちを知るだけでなく、<strong>自分の問いや違和感が他者や社会に届く経験</strong>が必要になると考えられる。
そのため次の段階では、生徒が見つけた違和感や発見を、企業や地域の大人に返し、フィードバックを受け、
未来のまちへの提案につなげる仕組みを入れたい。
まちを「見る場所」 から、子どもたちが <strong>「関わることのできる学習環境」</strong> へと変えていくことが、次の検証課題である。
</p>

<h2>4. 多様な子どもたちにとって開かれた学習環境になっているか</h2>
<p>
現在の教育現場では、不登校や多様な支援ニーズへの対応が大きな課題になっている。
こうした状況を踏まえると、これからの教育では、学校の中だけに支援や学習機会を増やすのではなく、
<strong>学校外のまち・企業・地域も含めて、多様な子どもたちが学びにアクセスできる環境</strong>を考える必要がある。
</p>
<p style="margin-top:0.5em">
今回、私たちは、学校不適応・車いす利用 (ハンディキャップ)・外国籍・ジェンダーレスなどの
特徴的な背景を持つ生徒を設定した。これは、多様な背景を持つ子どもたちが、
それぞれの特性や関心に応じた参加スタイルで学びに加われるかを見たいと考えたためである。
</p>
<p style="margin-top:0.5em">
この設計を通じて、私たちは、<strong>多様な背景を持つ生徒は、まちを見るための重要な視点を持ち得る</strong>と感じた。
一方で、今回の結果からは、多様な生徒を配置するだけでは、その視点が自然に深まるわけではないことも分かった。
その子ならではの見え方を深めるには、視点に応じたルート、観察の問い、記録の仕組み、共有の時間、
企業担当者による問い返しが必要である。
</p>
<p style="margin-top:0.5em">
したがって、次に私たちが見たいのは、まちが学習環境になり得るかどうかだけではない。
<em>そのまちは誰にとって学びやすく、誰にとって参加しにくいのか。
どのような支援や設計があれば、多様な子どもたちが自分なりの参加スタイルで学びに加われるのか。</em>
この点をより丁寧に検証していきたい。
</p>

<h2>5. 企業は、訪問先ではなく問いを深める伴走者になり得る</h2>
<p>
今回、私たちは 12 社の企業担当者を配置し、生徒がまちを支える多様なプレイヤーと出会えるようにした。
これは、交通、通信、税、水、食、観光、働く場など、まちの裏側にある社会の仕組みを学ぶうえで有効だった。
</p>
<p style="margin-top:0.5em">
一方で、結果を見ると、<strong>企業訪問が目的化する場面</strong>もあった。
最低 2 社以上を訪問するという課題が強く働いたことで、生徒自身の問いや自由探索よりも、
企業訪問ミッションの達成が優先される場面が生まれていた。
</p>
<p style="margin-top:0.5em">
ここから私たちは、企業を単なる「訪問先」や「説明役」として置くだけでは不十分だと考えた。
企業担当者は、都市の学習資源であると同時に、
<strong>生徒の問いを受け止め、問い返し、まちの未来を一緒に考える伴走者にもなり得る</strong>。
</p>
<p style="margin-top:0.5em">
企業が自社の取り組みを説明するだけでなく、生徒の観察を受け止め、
「なぜそう思ったのか」 「それは誰にとって困るのか」 「その気づきを未来のまちに活かすならどうなるか」
と問い返すことで、企業連携は単なる職業理解に留まらず、社会参画につながる学びへ発展していくと考えられる。
</p>

<h2>6. AI 時代に「まち・企業・AI」 がつながる学習環境</h2>
<p>
私たちは、AI が教育現場に入っていく近未来において重要なのは、AI を単なる情報検索や効率化の道具として使うことではなく、
<strong>子どもの問いを深め、人や社会との接続を強める触媒として設計すること</strong> だと考えている。
</p>
<p style="margin-top:0.6em">
今回のシミュレーションでは、AI パートナーを生徒一人ひとりに付けるところまでは行っていない。
しかし、今回の結果から、次に AI パートナーを検証する意義は明確になった。今回見えた課題は次のとおり:
</p>
<ul>
<li>気づきが問いに変わりきらない</li>
<li>個別の視点が他者に共有されにくい</li>
<li>企業担当者への質問が深まりきらない</li>
<li>「自分もまちに関われる感」が弱い</li>
<li>多様な背景から生まれる視点が十分に回収されない</li>
</ul>
<p style="margin-top:0.6em">
ここに <strong>AI パートナー</strong> を入れることで、
AI が子どもの観察や違和感を問いに変え、企業や他の生徒との対話につなげる触媒になり得るかを検証したい。
</p>
<p style="margin-top:0.6em">
また、AI パートナーについても、情報をすぐに提供する <em>全知全能型</em> の AI だけでなく、
すぐに答えを出さず問いを返す <em>ソクラテス型</em> の AI など、複数のタイプを比較したい。
AI が子どもの思考を先回りしてしまうのか、それとも子どもの問いを深め、人との対話を広げる存在になれるのかを、
次のシミュレーションで見ていきたい。
</p>

<h2>7. 生徒ペルソナを変更したケースについて</h2>
<p>
今回の本案 (高校生 10 人) と並行して、<strong>同じ品川駅周辺のフィールドワークを、生徒ペルソナだけ小学生に差し替えて回す</strong>
試行も行った。問い・場・企業担当者・タイムテーブルは固定し、参加者の発達段階のみを変えたときに、
まちの捉え方がどう変わるのかを見るためである。
</p>
<p style="margin-top:0.5em">
その結果、<strong>小学生にした場合は、まちを社会システムとして捉えるというよりも、
まずは目に見えるもの・体験できるもの・自分の生活に近いものへの反応が中心になっていた</strong>。
高校生では「再開発」「インフラ」「歴史と現代の共存」のように構造を語る言葉が出てきたのに対し、
小学生では「マイクラと結びつけて街を見る」 といった、自分が今もっている世界観に引き寄せた語り方が目立った
（最近の子らしい反応だなという印象を受けた）。
</p>
<p style="margin-top:0.5em">
ここから私たちは、<strong>小学生では、気づきそのものは豊かに出る可能性があるが、
それを社会的な問いに変換するには支援が必要である</strong>と考えている。
身近な体験や好きな世界に紐づいた発見は、本人にとって強い実感を伴う一方で、
そのままでは「まち全体の課題」 や 「他者にとっての困りごと」 に接続しづらい。
気づきを社会的な問いに翻訳する伴走者の存在が、低学年ほど重要になりそうだ。
</p>
<p style="margin-top:0.5em">
そして、<strong>この支援にこそ AI は有効に働き得る</strong> と感じている。
子どもが見つけた「マイクラっぽい」 「ここワクワクする」 といった素朴な気づきを受け止め、
「それは誰にとっての街か」 「もし自分がここに住むならどう変えたいか」 と問い返す相手としての AI は、
大人の伴走者だけでは間に合わない局面を補う触媒になれる可能性がある。
次の検証対象として、<em>生徒ペルソナの発達段階 × AI パートナー有無</em> の組み合わせも見ていきたい。
</p>
</section>
"""

    section_nextaction = """
<section class="page-break" id="section-nextaction">
<h1>やまちゃそ考察（シミュ実装観点 + まちづくり観点）</h1>

<h2>1. LLM の言動コントロール</h2>
<p>
本シミュレーションの構築過程では、LLM (Gemini 2.5 flash-lite) を使った multi-agent 環境特有の
不自然さを潰すために、<strong>170 以上の試行錯誤 (issue 検知 → 修正 → 再 smoke)</strong>
を重ねた。今回の本案 (157) で実際に観察された主な事象と、その後の検証結果を以下にまとめる。
</p>

<div style="margin:0.8em 0;padding:0.8em 1em;border-left:3px solid #ffd85f;background:#241e10;page-break-inside:avoid;break-inside:avoid">
<strong>事象 A — Phase A で 1 度も発話できない生徒が出る (例: 桜井さん)</strong>
<ul style="margin:0.4em 0 0.2em 0">
<li><strong>原因</strong>: Phase 1 第1パスの確率ゲート (社交性 × 関係性 × 距離 × ブースト) で partner (会話相手) が決まらないと silent (黙る) 確定 になる。Phase A は 30 step 短期 + 教室密集で「12 人中の最大 p_speak の相手」 が確率分布で偏り、桜井さんは 30 step 全部で partner 候補上位に来ず、結果として発話 0 件・memory ログ 0 件 で終わった。</li>
<li><strong>対策案</strong>: ① 受信トリガー保持期間を 1 step → 2〜3 step に延長 ② 「nearby (近くに居る人) に誰か居れば必ず誰かを partner として組ませる」 強制割当 ③ 確率ブースト (talkativeness の下限を引き上げ)。</li>
<li><strong>副作用</strong>: ② を入れると不適応 persona (silent でいたい) も毎 step 強制発話してしまい、persona 表現が壊れる。③ は API call 増 + コスト増。</li>
<li><strong>157 以降での検証</strong>: 案 A で「premise の強制ルール削除 + WORKING STATE 終盤メッセージ弱化」 で sim4''' / sim5''' を回し、Phase B では学生発話 12/12 復活を確認。だが Phase A 桜井問題そのものは構造的に未解決 (短期 + 密集だと確率の偏りで「全 step 外す」 が起きる)、これが残課題。</li>
</ul>
</div>

<div style="margin:0.8em 0;padding:0.8em 1em;border-left:3px solid #ffd85f;background:#241e10;page-break-inside:avoid;break-inside:avoid">
<strong>事象 B — 話しかけたのに返ってこない (Phase A 全 36 発話中 29 件 = 80%が片想い)</strong>
<ul style="margin:0.4em 0 0.2em 0">
<li><strong>原因</strong>: ① 双方向同時発話排除で id 大きい方が silent → 翌 step で返事を期待するが LLM が別話題に行く ② opener gate (確率 skip) で B が次 step に partner 選ばれない ③ Jaccard 4-gram ≥ 0.50 後フィルタで類似発話が silent 降格 ④ memory rolling buffer (直近 5 件) からの押し出しで「step N で何を聞かれたか」 を忘れる、の 4 重の構造的要因。</li>
<li><strong>対策案</strong>: ① 受信トリガー保持期間延長 (1→3 step) ② 双方向同時を「両方発言残し、応答 step を 1 ずらす」 に変更 ③ Jaccard 閾値を緩める (0.50→0.65)。</li>
<li><strong>副作用</strong>: ② は同時発話の混雑感が出て turn-take が崩れる。③ は同テーマ連投が増える。</li>
<li><strong>157 以降での検証</strong>: 未検証。turn-take 設計は次の検証対象。</li>
</ul>
</div>

<div style="margin:0.8em 0;padding:0.8em 1em;border-left:3px solid #ffd85f;background:#241e10;page-break-inside:avoid;break-inside:avoid">
<strong>事象 C — Phase B 後半で思考も移動も止まる (例: 中島さん税務署で課題完了後 stay 連発、車いず辻さん、不適応宮原さん)</strong>
<ul style="margin:0.4em 0 0.2em 0">
<li><strong>原因</strong>: ① 必須課題 (2 社訪問) を完了すると次の動機が agent から消える ② 「対話継続 vs 移動開始」 の trade-off で stay (留まる) を選びがち ③ navigation で goal (目的地) との距離が 1 step 移動量内に入ると moved=0 になり walk_along + target=null が出る (実質その場停止)。中島さんは reasoning に「東西自由通路に戻る」と書きながら 13 step 連続 stay。</li>
<li><strong>対策案</strong>: ① WORKING STATE block で残り step / 未訪問先 / 未解決問い を毎 step 表示 (smoke7-③) ② 残り 5 step 以下で rule-based に walk_toward (向かって歩く) 集合場所を強制 ③ premise block (cache 領域) に「終盤は戻りも視野」 を弱く挿入。</li>
<li><strong>副作用</strong>: ②③ を強くすると「対話控えて移動」 と LLM が学習し、序盤から学生発話が激減する regression (大量沈黙化) が出た。</li>
<li><strong>157 以降での検証</strong>: 案 A (premise の強制ルール削除 + WORKING STATE 終盤メッセージを「戻りも視野に」 程度に弱化 + 残り 5 step だけ rule-based 強制) で sim4''' / sim5''' を回し、学生発話 12/12 復活 + 終盤の集合場所への戻り行動も観察できた。</li>
</ul>
</div>

<div style="margin:0.8em 0;padding:0.8em 1em;border-left:3px solid #ffd85f;background:#241e10;page-break-inside:avoid;break-inside:avoid">
<strong>事象 D — 企業担当者が動きすぎて持ち場放棄、訪ねて来た生徒に会えない</strong>
<ul style="margin:0.4em 0 0.2em 0">
<li><strong>事象</strong>: 157 では host (企業担当者) 12 人中ほぼ全員が配置を離れて動いていた。中村 (JR) は 27/50 step 移動し、グランドプリンス新高輪に居座って通る生徒に話しかける状態 (= 配置のアトレ品川に居ない)。三宅さんがアトレに来たのに中村不在で右往左往した。高梨 (水族館) はリーさんと意気投合で同行→離れた瞬間に持ち場に戻らず停止。</li>
<li><strong>原因</strong>: 157 は host 不動修正前。host も decide_action (行動判定) で walk_toward を選び、学生に応答するうち別の場所まで同行する挙動が起きた。</li>
<li><strong>対策案</strong>: ① simulation.py の Phase 3 で <code>is_host=true</code> のとき action_type=stay (留まる) を強制、LLM 呼び出し自体スキップ ② host idle silence (待機時の発話スキップ) と組合せて学生が来るまで黙って待つ。</li>
<li><strong>副作用</strong>: host が完全に黙って待つだけになると、動的な対話パターンが減って静的になる。「学生から声掛けられない host」 は永遠に発話しない。</li>
<li><strong>157 以降での検証</strong>: smoke31 以降で host 不動 (action_type=stay 強制) を入れた。学生が拠点訪問時に host が居る確率が大幅に改善。だが副作用として「不在問題は解消したが入口で固定される host と中の学生で会話が起きない」 という別の課題が出ており、これは host 配置 (拠点入口の歩道側) の見直しで対応中。</li>
</ul>
</div>

<div style="margin:0.8em 0;padding:0.8em 1em;border-left:3px solid #ffd85f;background:#241e10;page-break-inside:avoid;break-inside:avoid">
<strong>事象 E — 双方向同時発話排除の副作用ですれ違いが固定化 (例: 篠田 → アンナ 0 回成立)</strong>
<ul style="margin:0.4em 0 0.2em 0">
<li><strong>事象</strong>: トヨタ篠田さんはアンナさんへ 3 回 partner 設定したが、アンナの memory には「篠田に会わなければ」 系の意識が記録されているのに、双方向で partner 認識が成立せず一度も対話成立せず。西野 ↔ 西田 (ソニー) も同様に 短い接触の後、両者が互いを partner として認識しなくなり stay 連発。</li>
<li><strong>原因</strong>: 「双方向同時発話排除 (A→B かつ B→A の同 step なら id 大 silent)」 と「opener gate」 の組合せで「お互い相手を partner にしたが受信トリガー無しで両方 silent」 状態がロック。最初の opener が来ない限り stay が固定。</li>
<li><strong>対策案</strong>: ① 双方向同時を「両方発言残し、応答 step を 1 ずらす」 に変更 ② nearby に居る + 関係性 ≥ 0.5 の同行ペア を「会話継続中」 とみなし opener 確率を強くブースト。</li>
<li><strong>副作用</strong>: ① は同時発話の混雑感、turn-take 崩れ。② は逆に同行が成立しないペアでブースト効かず効果半減。</li>
<li><strong>157 以降での検証</strong>: 未検証。turn-take 設計は次の検証対象。</li>
</ul>
</div>

<div style="margin:0.8em 0;padding:0.8em 1em;border-left:3px solid #ffd85f;background:#241e10;page-break-inside:avoid;break-inside:avoid">
<strong>事象 F — 同じ人に同じテーマを繰り返す / 移動が片寄る (副次課題)</strong>
<ul style="margin:0.4em 0 0.2em 0">
<li><strong>事象 1 (繰り返し)</strong>: LLM は自分の発話を内省できず、微妙な言い換えで同テーマを連投する。「品川って色々混ざってる感じ」 を 5 step にわたって相手を変えて繰り返す等。</li>
<li><strong>事象 2 (移動偏り)</strong>: agent の reasoning に「高輪行きたい」 と書いているのに実 position は港南側に張り付く。原因は navigation の歩道制約で国道 15 号を渡れず東側歩道で詰まる構造的課題。</li>
<li><strong>対策</strong>: 事象 1 → 4-gram Jaccard 類似度後フィルタで silent 化 + user_prompt に「あなたが nearby 各人に既に言ったこと top 3」 を可視化。事象 2 → 道路全幅を歩行可に変更。</li>
<li><strong>副作用</strong>: Jaccard 後フィルタが強すぎると同テーマ展開 (深掘り) も silent 化されてしまう。</li>
<li><strong>157 以降での検証</strong>: 事象 2 は解消、事象 1 は閾値 0.50 で本案運用中。事象 B と合わせて turn-take 設計の見直しで再検証予定。</li>
</ul>
</div>

<h2>2. LLM エージェント自体の設計</h2>
<p>
1. で挙げた事象 A〜F は、突き詰めると <strong>「各エージェントを 1 つの巨大な LLM プロセスでまとめて管理している」 こと</strong>
に起因する課題群と読み替えることができる。
具体的には、step ごとに 全 agent の発話/行動を 1 つのオーケストレーターから並列に呼び出し、
agent ごとの状態 (記憶・関係性・直近受信) を user_prompt に毎回詰め直す構造になっており、
agent 個々が「自分の脳」 を持って自律的に振る舞っているわけではない。
</p>
<p style="margin-top:0.5em">
この構造の制約として、以下のような副作用が出る:
</p>
<ul>
<li><strong>記憶の押し出し</strong>: rolling buffer (直近 5 件) からこぼれた記憶は、5 step ごとの圧縮要約に頼るしかない。
個別 agent が「自分の文脈は自分で管理する」 形になっていない。</li>
<li><strong>turn-take の不自然さ</strong>: 双方向同時排除や opener gate は、本来「相手が話したから自分も応答する」 という非同期メッセージングで自然に解決するもの。
全員を同 step で同期処理するアーキテクチャに「人間っぽい turn-take」 を後付けしているため、片想い・すれ違いが構造的に残る。</li>
<li><strong>silent 制御の二重化</strong>: 確率ゲートと LLM 任せ判断が二重に効いて挙動が読みづらくなる。
agent ごとに「自分の判断ロジック」 が独立していれば、こういう二重化は不要。</li>
</ul>
<p style="margin-top:0.6em">
<strong>ネクストアクション</strong>: 各エージェントが <em>API 側でコンテキストを保持した状態で参加</em> できる仕組みになれば
(<em>moltbook</em> のように Discord 風チャンネル + 各 agent が独立した会話スレッドを持つ構成、
あるいは A2A プロトコル等による agent 間メッセージング)、
1. の事象 A〜F の相当部分はアーキテクチャレベルで吸収される想像。
具体的には:
</p>
<ul>
<li>各 agent は自分の Thread (= context window) を独占し、記憶の管理を自分で行う → 押し出し問題が消える</li>
<li>発話は「相手の Thread にメッセージを post」 する非同期通信になり、双方向同時排除のような同 step 制約が消える → 片想い・すれ違いが減る</li>
<li>silent 判定は agent 自身が「自分のスレッドを今どう扱うか」 を決めるだけ → 確率ゲートと LLM 任せの二重化が解消する</li>
<li>agent 間で観測できる情報 (誰が今どこに居るか、誰と誰が話しているか) はチャンネル側で共有され、persona は「読みたいときに読む」 構造になる</li>
</ul>
<p style="margin-top:0.5em">
本ハッカソンでは、こうした構造を 1 プロセス内 + prompt 工夫 + ルールベース後フィルタの組合せで擬似再現している。
これでも 170 以上の試行錯誤で大半の不自然さは抑えられたが、
本格運用 (= 教育現場や都市計画現場で意思決定の素材になる粒度) では
agent ごとに自律したアーキテクチャの方向で検証する必要がある。
</p>

<h2>3. まちのコンテキストの与え方</h2>
<p>
本シミュレーションでは、まちのコンテキスト (品川の地形・歴史・進行中の再開発・各施設の利用層など) は、
すべて <strong>テキスト</strong> として agent の system_prompt や memory に与えている。
そのため agent は「実際に歩いて見た光景」ではなく「テキストとして読まれた知識」をベースに思考している。
</p>
<p style="margin-top:0.5em">
<strong>ネクストアクション</strong>: 3D 環境上で agent が様々なパラメータ (歩道の幅・店舗のサイン・
人の密度・段差・音・光) を直接感知できる仕組みになれば、
よりリアルなシミュレーションが可能になる想像。
スペースデータの <em>OpenUSD</em> ベースの空間表現や、
都市デジタルツインと組み合わせると、テキスト記述では拾いきれない空間品質の差
(= 「ここは誰のためのまちか」を agent が場の作りから感じ取る) を再現できる可能性がある。
</p>
<p style="margin-top:0.5em">
さらに <strong>時間帯・天候・季節</strong> といった環境変数もシミュレーションに組み込めると、
よりリアルな振る舞いが期待できる。
例えば <em>雨の日</em> なら濡れたくないので雨宿り可能なルート (アーケード・地下道) を選んだり、
遠い場所の訪問を諦めたりする。<em>夜</em> なら暗い裏道を避け、
<em>夏の暑い日</em> なら日陰や涼しい屋内を経由する、といった判断が emerge する可能性がある。
これらの環境変数は、まちが「誰にとって/いつ/どんな天気で 過ごしやすいか」 という
「まちの設計品質」 を多層で炙り出すための鍵になる。
</p>

<h2>4. まちづくり・都市開発への応用</h2>
<p>
都市開発の現場では、経済合理性を重視するデベロッパーの視点や、まちづくりビジョンや上位計画にもとづく行政の視点、
建築家・都市設計者の作家性のような視点が、構造的にどうしても先行しがちで、
<strong>実際にそのまちを使う住民・来街者・働く人といったユーザーの視点が後回しにされる</strong> 傾向が依然として強い。
住民参加ワークショップやパブリックコメントといった機会はあるものの、
意思決定が動く前段で「広く・多様に・深く」 ユーザーの声を拾う仕組みは、現状でも十分とは言えない。
</p>
<p style="margin-top:0.6em">
本シミュレーションが目指している先には、こうした構造への打ち手としての応用がある。
今回のような LLM agent シミュレーションを、より精度高く、より多様なペルソナとシナリオで回せるようになれば、
<strong>デジタルツイン上で「そのまちが完成する前」 にユーザーの声を拾う</strong> ことができる可能性がある。
</p>
<ul>
<li>属性・年齢・障害特性・国籍・所得などが異なる多様な persona を、再開発計画地・新設駅・商業施設のリニューアル後の姿の上で動かし、
「誰がどこで詰まるか / 誰が来ないか / 誰の動線が交差しないか」 を <strong>計画段階で可視化</strong> する</li>
<li>「広場のベンチが足りない」「車いずでは横断歩道までの段差が辛い」「外国籍の来街者は案内サインの英語密度が低くて駅から出られない」 等の
ユーザー目線の課題を、設計確定前にフィードバックする</li>
<li>計画案 A / B / C を同じ persona セットでシミュレートし、<strong>「どの案が誰にとって過ごしやすいか」 を比較定量化</strong> する</li>
</ul>
<p style="margin-top:0.6em">
これは、上位計画やデベロッパーロジックから降ってくる従来型の都市計画に対し、
<strong>ユーザーのまちでの過ごし方・困りごと・期待を起点にした「ボトムアップ型都市計画」 の素材を提供する</strong> 試みでもある。
今回の品川 FW シミュは「教育プログラム」 の枠で書いているが、
裏側のエンジン (多様 persona × 場所 × 行動 × 対話 のシミュレーション) はそのまま都市計画の意思決定支援に転用できる。
本案 (157) はその第一歩であり、turn-take・記憶・空間表現の精度を上げていくことで、
<strong>「ユーザーの声から先につくるまち」 の設計サイクル</strong> を回せるようになっていきたい。
</p>

</section>
"""

    section_summary = """
<section class="page-break" id="section-summary">
<h1>まとめ — 全体考察・えびねこ考察・やまちゃそ考察を通して</h1>
<p>
本ハッカソンでは、品川駅周辺を題材に「企業協力型 学外教育プログラム」 を LLM agent シミュレーション上で再構成し、
全体考察 (シミュ結果ベースの観察) / えびねこ考察 (教育観点) / やまちゃそ考察 (シミュ実装観点) の 3 つの視点から
評価した。
</p>
<p style="margin-top:0.6em">
シミュ結果からは、<strong>まちは知識を実感に変える学習環境になり得る</strong>という手応えが得られた一方で、
「自分もまちに関われる」 という主体感までは届ききらず、観察者ポジションに留まる場面が残った。
特徴的背景を持つ生徒については、属性ごとに参加スタイルの差は確かに観察できたが、
その視点を回収し他者に届ける仕組みは不足しており、
<strong>多様な生徒を配置するだけでは多様な学びにはならない</strong>ことも分かった。
</p>
<p style="margin-top:0.6em">
シミュ自体としては、170 以上の試行錯誤を経て「人間の観察に耐えるリアリティ」 に近づけたが、
turn-take の片想い・後半の停滞・拠点での接点不全といった残課題は実装上もそのまま残っている。
</p>
<p style="margin-top:0.6em">
次に試したいのは、<strong>「知る」 から「関わる」 への転換</strong> を支える仕組み —
AI パートナー (問いを深める触媒) / 企業担当者の問い返し / 多様な背景に応じたルートと記録 —
を組み合わせた次世代シミュレーションである。
本案 (157) はその基準点 (本案 = 比較対象) として、今後の改善を測る尺度になる。
</p>
</section>
"""

    # 目次 (TOC): 各 section にジャンプできるリンク。
    toc_html = f"""
<section class="page-break" id="section-toc" style="page-break-before:always">
<h1>目次</h1>
<ol style="line-height:2.4;font-size:1.05em">
<li><a href="#section-cover" style="color:#ffd85f;text-decoration:none">表紙 — 本シミュレーションのテーマ・狙い</a></li>
<li><a href="#section-overview" style="color:#ffd85f;text-decoration:none">シミュレーションレポート ({variant_label}) — 構成・規模・課題</a></li>
<li><a href="#section-roster" style="color:#ffd85f;text-decoration:none">登場人物一覧 (生徒 {len(participants)}人 / 企業担当者 {len(hosts)}人)</a></li>
<li><a href="#section-phase-a" style="color:#ffd85f;text-decoration:none">Phase A — 教室での座学 (挙動ルール・タイムライン・各生徒の座学アウトプット)</a></li>
<li><a href="#section-phase-b" style="color:#ffd85f;text-decoration:none">Phase B — 品川駅前フィールドワーク (挙動ルール・タイムライン・各人の現地気づき)</a></li>
<li><a href="#section-phase-c" style="color:#ffd85f;text-decoration:none">Phase C — シミュ後アンケート (生徒/企業担当者 数値・自由記述)</a></li>
<li><a href="#section-reflection" style="color:#ffd85f;text-decoration:none">全体考察 (全体を通しての観察 / プロジェクトの狙いに対する考察 / シミュレーション結果に関する考察)</a></li>
<li><a href="#section-ebineko" style="color:#ffd85f;text-decoration:none">えびねこ考察 (教育観点)</a></li>
<li><a href="#section-nextaction" style="color:#ffd85f;text-decoration:none">やまちゃそ考察 (シミュ実装観点 + ネクストアクション)</a></li>
<li><a href="#section-summary" style="color:#ffd85f;text-decoration:none">まとめ — 全体考察・えびねこ考察・やまちゃそ考察を通して</a></li>
<li><a href="#section-api" style="color:#ffd85f;text-decoration:none">API消費・所要時間 (参考)</a></li>
<li><a href="#section-backdata" style="color:#ffd85f;text-decoration:none">全ステップログ (バックデータ)</a></li>
</ol>
</section>
"""

    body = (
        HEAD.format(title=html.escape(title))
        + intro + toc_html + section_roster + section_a + section_b + section_c
        + section_reflection + section_ebineko + section_nextaction + section_summary + section_meta
        + backdata_link + FOOT
    )
    # 表記ゆれ修正 (2026-05-05): 「街」「町」→「まち」を最終本文で保険置換。
    # 注: 固有名詞 (品川宿・宿場町・北品川・町並み 等) を意図せず壊さないように、
    # 最低限の単独形のみ置換する。慎重を期して、reflection / 自由記述部分でも残ったら
    # 一括置換でカバー。
    # 「街」を「まち」に。「町」は宿場町・北品川・町並み 等の固有/熟語が多いので個別対応。
    body = body.replace("街", "まち")
    # 「町」のうち、「町並み」「街道沿い」などは別途。今は単純に「町」も置換しない (固有名詞リスク)。
    out_path = Path(args.out) if args.out else (run_b / f"v3_report{suffix}.html")
    out_path.write_text(body, encoding="utf-8")
    print(f"[ok] wrote: {out_path} ({len(body)/1024:.0f} KB)")

    # 3D viewer bundled HTML を自動生成 (3点セット共有用)。
    # simulation_data.json に scene_3d が既に入っているので追加引数不要。
    try:
        import subprocess
        bundled_path = run_b / f"viewer_3d_bundled{suffix}.html"
        bundle_cmd = [sys.executable, str(ROOT / "tools" / "bundle_viewer.py"),
                      str(run_b), "--viewer", "3d", "--out", str(bundled_path)]
        result = subprocess.run(bundle_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            if bundled_path.exists():
                print(f"[ok] wrote 3D viewer (bundled): {bundled_path} ({bundled_path.stat().st_size/1024:.0f} KB)")
        else:
            print(f"[warn] bundle_viewer.py failed: {result.stderr[:300]}", file=sys.stderr)
    except Exception as e:
        print(f"[warn] bundle_viewer skip: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
