"""
Generate a single-page HTML report combining animated GIF, conversation log,
and thinking log from simulation output.
"""
import glob
import html
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

GIF_FRAME_DURATION_MS = 500

AGENT_COLORS = [
    "#4a9eff", "#ff7a4a", "#4fdb8b", "#e05ac9",
    "#f0c040", "#9b6dff", "#50d8d0", "#ff5f7e",
]


def _agent_color(agent_id: int) -> str:
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


def build_gif(output_dir: str, gif_path: str) -> bool:
    # New layout: frames live in `{output_dir}/frames/`. Legacy layout: frames
    # at `{output_dir}/` root. Check both so old smoke configs still work.
    frame_paths = sorted(glob.glob(os.path.join(output_dir, "frames", "frame_*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    if not frame_paths:
        logger.warning("No frames found for GIF generation")
        return False

    frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
        disposal=2,
    )
    logger.info(f"Saved GIF ({len(frames)} frames): {gif_path}")
    return True


def _read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _group_by_step(records: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for r in records:
        step = r.get("step", 0)
        grouped[step].append(r)
    return grouped


def _render_messages_html(messages: List[Dict]) -> str:
    grouped = _group_by_step(messages)
    if not grouped:
        return '<p class="empty">会話はまだありません</p>'

    blocks = []
    for step in sorted(grouped.keys()):
        recs = grouped[step]
        # Deduplicate: same (from, message) across multiple recipients -> one card
        seen = set()
        unique = []
        for r in recs:
            key = (r.get("from"), r.get("message"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)

        items = []
        for r in unique:
            sender = r.get("from", "?")
            message = r.get("message", "")
            if not message:
                continue
            color = _agent_color(int(sender)) if isinstance(sender, int) else "#888"
            items.append(
                f'<div class="card" style="border-left-color:{color}">'
                f'<div class="sender" style="color:{color}">Agent {html.escape(str(sender))}</div>'
                f'<div class="body">{html.escape(message)}</div>'
                f'</div>'
            )
        if items:
            blocks.append(
                f'<div class="step-group">'
                f'<div class="step-label">Step {step}</div>'
                + "".join(items)
                + "</div>"
            )

    if not blocks:
        return '<p class="empty">会話はまだありません</p>'
    return "".join(blocks)


def _render_thoughts_html(memory_reasoning: List[Dict]) -> str:
    grouped = _group_by_step(memory_reasoning)
    if not grouped:
        return '<p class="empty">思考ログはまだありません</p>'

    blocks = []
    for step in sorted(grouped.keys()):
        recs = sorted(grouped[step], key=lambda r: r.get("id", 0))
        items = []
        for r in recs:
            agent_id = r.get("id", "?")
            memory = r.get("memory", "")
            reasoning = r.get("reasoning", "")
            if not memory and not reasoning:
                continue
            color = _agent_color(int(agent_id)) if isinstance(agent_id, int) else "#888"
            body = []
            if reasoning:
                body.append(
                    f'<div class="reasoning"><span class="tag">判断</span>'
                    f'{html.escape(reasoning)}</div>'
                )
            if memory:
                body.append(
                    f'<div class="memory"><span class="tag">記憶</span>'
                    f'{html.escape(memory)}</div>'
                )
            items.append(
                f'<div class="card" style="border-left-color:{color}">'
                f'<div class="sender" style="color:{color}">Agent {html.escape(str(agent_id))}</div>'
                + "".join(body)
                + "</div>"
            )
        if items:
            blocks.append(
                f'<div class="step-group">'
                f'<div class="step-label">Step {step}</div>'
                + "".join(items)
                + "</div>"
            )

    if not blocks:
        return '<p class="empty">思考ログはまだありません</p>'
    return "".join(blocks)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Simulation Report</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #1a1a1a; color: #e8e8e8;
    font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", "Meiryo", sans-serif; }}
  .container {{ display: flex; height: 100vh; }}
  .left {{ width: 66.66%; display: flex; flex-direction: column; padding: 12px; box-sizing: border-box;
    align-items: center; justify-content: center; background: #121212; }}
  .left img {{ max-width: 100%; max-height: calc(100vh - 60px); object-fit: contain;
    border: 1px solid #2c2c2c; border-radius: 4px; }}
  .left .title {{ font-size: 13px; color: #888; margin-bottom: 10px; }}
  .right {{ width: 33.34%; display: flex; flex-direction: column; border-left: 1px solid #2c2c2c; }}
  .pane {{ flex: 1; overflow-y: auto; padding: 14px 16px; box-sizing: border-box;
    scrollbar-width: thin; scrollbar-color: #444 #1a1a1a; }}
  .pane-top {{ border-bottom: 1px solid #2c2c2c; }}
  .pane h2 {{ margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #9cc;
    letter-spacing: 0.5px; text-transform: uppercase; border-bottom: 1px solid #2c2c2c;
    padding-bottom: 6px; position: sticky; top: -14px; background: #1a1a1a; z-index: 1; }}
  .step-group {{ margin-bottom: 14px; }}
  .step-label {{ font-size: 11px; color: #666; margin-bottom: 4px;
    font-family: "SF Mono", Consolas, monospace; }}
  .card {{ margin-bottom: 6px; padding: 8px 10px; background: #242424;
    border-left: 3px solid #555; border-radius: 3px; }}
  .card .sender {{ font-size: 11px; font-weight: 600; margin-bottom: 4px; }}
  .card .body {{ font-size: 13px; line-height: 1.5; color: #d8d8d8; }}
  .card .reasoning, .card .memory {{ font-size: 12px; line-height: 1.45;
    color: #c8c8c8; margin-top: 3px; }}
  .tag {{ display: inline-block; font-size: 10px; padding: 1px 6px; margin-right: 6px;
    background: #333; color: #aaa; border-radius: 2px; vertical-align: 1px; }}
  .empty {{ color: #555; font-size: 12px; font-style: italic; }}
  .meta {{ font-size: 11px; color: #777; margin-top: 8px; }}
  .meta span {{ margin-right: 10px; }}
</style>
</head>
<body>
<div class="container">
  <div class="left">
    <div class="title">{title}</div>
    <img src="{gif_name}" alt="simulation animation">
    <div class="meta">
      <span>Steps: {total_steps}</span>
      <span>Model: {model}</span>
    </div>
  </div>
  <div class="right">
    <div class="pane pane-top">
      <h2>会話</h2>
      {messages_html}
    </div>
    <div class="pane pane-bottom">
      <h2>思考</h2>
      {thoughts_html}
    </div>
  </div>
</div>
</body>
</html>
"""


def build_markdown_transcript(output_dir: str, config: Dict, total_steps: int,
                              basename: Optional[str] = None) -> str:
    """Write a chronological Markdown transcript of messages and thoughts for AI consumption."""
    messages = _read_jsonl(os.path.join(output_dir, "messages.jsonl"))
    memory_reasoning = _read_jsonl(os.path.join(output_dir, "memory_reasoning.jsonl"))

    msg_by_step = _group_by_step(messages)
    thought_by_step = _group_by_step(memory_reasoning)
    all_steps = sorted(set(msg_by_step.keys()) | set(thought_by_step.keys()))

    model = config.get("llm", {}).get("model", "unknown")
    num_agents = config.get("agents", {}).get("num_agents", "unknown")
    places = config.get("places", [])
    fires = config.get("fires", [])

    lines: List[str] = []
    lines.append("# Simulation Transcript")
    lines.append("")
    lines.append(f"- Total steps: {total_steps}")
    lines.append(f"- Agents: {num_agents}")
    lines.append(f"- Model: {model}")
    if places:
        lines.append("- Places:")
        for p in places:
            lines.append(
                f"  - `{p.get('name')}` ({p.get('type')}) "
                f"center=({p.get('center_x')}, {p.get('center_y')}), "
                f"capacity={p.get('capacity')}"
            )
    if fires:
        lines.append("- Fires:")
        for f in fires:
            pos = f"({f.get('center_x')}, {f.get('center_y')})" if 'center_x' in f else "random"
            lines.append(
                f"  - `{f.get('name')}` start_step={f.get('start_step')}, "
                f"intensity={f.get('intensity')}, radius={f.get('radius')}, position={pos}"
            )
    lines.append("")

    for step in all_steps:
        lines.append(f"## Step {step}")
        lines.append("")

        # Messages: deduplicate same (from, message) across recipients
        msgs = msg_by_step.get(step, [])
        seen = set()
        unique_msgs = []
        for m in msgs:
            key = (m.get("from"), m.get("message"))
            if key in seen:
                continue
            seen.add(key)
            unique_msgs.append(m)

        if unique_msgs:
            lines.append("### 会話")
            lines.append("")
            for m in unique_msgs:
                sender = m.get("from", "?")
                message = m.get("message", "")
                if not message:
                    continue
                lines.append(f"- **Agent {sender}**: {message}")
                reason = m.get("reasoning", "")
                if reason:
                    lines.append(f"  - _発言理由_: {reason}")
            lines.append("")

        thoughts = sorted(thought_by_step.get(step, []), key=lambda r: r.get("id", 0))
        if thoughts:
            lines.append("### 思考")
            lines.append("")
            for t in thoughts:
                agent_id = t.get("id", "?")
                reasoning = t.get("reasoning", "")
                memory = t.get("memory", "")
                if not reasoning and not memory:
                    continue
                lines.append(f"- **Agent {agent_id}**")
                if reasoning:
                    lines.append(f"  - _判断_: {reasoning}")
                if memory:
                    lines.append(f"  - _記憶_: {memory}")
            lines.append("")

    transcript_name = f"{basename}_transcript.md" if basename else "transcript.md"
    md_path = os.path.join(output_dir, transcript_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved transcript: {md_path}")
    return md_path


def _format_step_raw(step: int, msgs: List[Dict], thoughts: List[Dict]) -> List[str]:
    """Return markdown lines for a single step's raw content."""
    lines: List[str] = []
    seen = set()
    unique_msgs = []
    for m in msgs:
        key = (m.get("from"), m.get("message"))
        if key in seen:
            continue
        seen.add(key)
        unique_msgs.append(m)

    if unique_msgs:
        lines.append("### 会話")
        lines.append("")
        for m in unique_msgs:
            sender = m.get("from", "?")
            message = m.get("message", "")
            if not message:
                continue
            lines.append(f"- **Agent {sender}**: {message}")
            reason = m.get("reasoning", "")
            if reason:
                lines.append(f"  - _発言理由_: {reason}")
        lines.append("")

    thoughts_sorted = sorted(thoughts, key=lambda r: r.get("id", 0))
    if thoughts_sorted:
        lines.append("### 思考")
        lines.append("")
        for t in thoughts_sorted:
            agent_id = t.get("id", "?")
            reasoning = t.get("reasoning", "")
            memory = t.get("memory", "")
            if not reasoning and not memory:
                continue
            lines.append(f"- **Agent {agent_id}**")
            if reasoning:
                lines.append(f"  - _判断_: {reasoning}")
            if memory:
                lines.append(f"  - _記憶_: {memory}")
        lines.append("")
    return lines


def _summarize_step_with_llm(
    client,
    model: str,
    step: int,
    msgs: List[Dict],
    thoughts: List[Dict],
) -> str:
    """Use Claude to summarize one step's activity into 2-3 concise Japanese lines."""
    seen = set()
    unique_msgs = []
    for m in msgs:
        key = (m.get("from"), m.get("message"))
        if key in seen:
            continue
        seen.add(key)
        unique_msgs.append(m)

    msg_block = "\n".join(
        f"Agent {m.get('from')}: {m.get('message', '')}"
        for m in unique_msgs if m.get("message")
    ) or "（発言なし）"

    thought_block = "\n".join(
        f"Agent {t.get('id')}: 判断={t.get('reasoning', '')} / 記憶={t.get('memory', '')}"
        for t in sorted(thoughts, key=lambda r: r.get("id", 0))
    ) or "（思考ログなし）"

    prompt = f"""以下は多エージェントシミュレーションのStep {step}の全ログです。
このステップで何が起きたかを、**日本語で3〜4行の箇条書き**に要約してください。
会話の主題、主な移動・判断、注目すべき個別行動があれば含めてください。

【会話ログ】
{msg_block}

【思考ログ】
{thought_block}

=== 出力フォーマット（厳守）===
- 会話の主題: <1行>
- 主な動き: <1行>
- 特筆: <1行、なければ「特になし」>"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
    except Exception as e:
        logger.error(f"Step {step} summary failed: {e}")
    return "（要約失敗）"


def build_markdown_condensed(
    output_dir: str,
    config: Dict,
    total_steps: int,
    sample_steps: Optional[List[int]] = None,
    basename: Optional[str] = None,
) -> str:
    """Write a condensed Markdown transcript: raw for sample steps, LLM summaries for the rest."""
    from anthropic import Anthropic

    messages = _read_jsonl(os.path.join(output_dir, "messages.jsonl"))
    memory_reasoning = _read_jsonl(os.path.join(output_dir, "memory_reasoning.jsonl"))
    msg_by_step = _group_by_step(messages)
    thought_by_step = _group_by_step(memory_reasoning)
    all_steps = sorted(set(msg_by_step.keys()) | set(thought_by_step.keys()))

    fires = config.get("fires", [])
    if sample_steps is None:
        sample_steps_set = set()
        if all_steps:
            sample_steps_set.add(all_steps[0])
            sample_steps_set.add(all_steps[-1])
        for f in fires:
            start = f.get("start_step")
            if start and start in all_steps:
                sample_steps_set.add(start)
                if start + 1 in all_steps:
                    sample_steps_set.add(start + 1)
        sample_steps = sorted(sample_steps_set)

    model = config.get("llm", {}).get("model", "claude-haiku-4-5-20251001")
    num_agents = config.get("agents", {}).get("num_agents", "unknown")
    places = config.get("places", [])

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else None
    if client is None:
        logger.error("ANTHROPIC_API_KEY not set; cannot generate summaries")
        return ""

    lines: List[str] = []
    lines.append("# Simulation Transcript (Condensed)")
    lines.append("")
    lines.append(f"- Total steps: {total_steps}")
    lines.append(f"- Agents: {num_agents}")
    lines.append(f"- Model: {model}")
    lines.append(f"- Sample (full) steps: {sample_steps}")
    if places:
        lines.append("- Places:")
        for p in places:
            lines.append(
                f"  - `{p.get('name')}` ({p.get('type')}) "
                f"center=({p.get('center_x')}, {p.get('center_y')}), "
                f"capacity={p.get('capacity')}"
            )
    if fires:
        lines.append("- Fires:")
        for f in fires:
            pos = f"({f.get('center_x')}, {f.get('center_y')})" if 'center_x' in f else "random"
            lines.append(
                f"  - `{f.get('name')}` start_step={f.get('start_step')}, "
                f"intensity={f.get('intensity')}, radius={f.get('radius')}, position={pos}"
            )
    lines.append("")

    for step in all_steps:
        msgs = msg_by_step.get(step, [])
        thoughts = thought_by_step.get(step, [])

        if step in sample_steps:
            lines.append(f"## Step {step} (原文サンプル)")
            lines.append("")
            lines.extend(_format_step_raw(step, msgs, thoughts))
        else:
            summary = _summarize_step_with_llm(client, model, step, msgs, thoughts)
            lines.append(f"## Step {step}")
            lines.append("")
            lines.append(summary)
            lines.append("")
            logger.info(f"Step {step} summarized")

    condensed_name = f"{basename}_transcript_condensed.md" if basename else "transcript_condensed.md"
    md_path = os.path.join(output_dir, condensed_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved condensed transcript: {md_path}")
    return md_path


def build_report(
    output_dir: str,
    config: Dict,
    total_steps: int,
    basename: Optional[str] = None,
) -> str:
    gif_name = "animation.gif"
    gif_path = os.path.join(output_dir, gif_name)
    build_gif(output_dir, gif_path)

    messages = _read_jsonl(os.path.join(output_dir, "messages.jsonl"))
    memory_reasoning = _read_jsonl(os.path.join(output_dir, "memory_reasoning.jsonl"))

    messages_html = _render_messages_html(messages)
    thoughts_html = _render_thoughts_html(memory_reasoning)

    model = config.get("llm", {}).get("model", "unknown")
    title = f"2D Multi-Place Simulation"

    rendered = HTML_TEMPLATE.format(
        title=html.escape(title),
        gif_name=gif_name,
        total_steps=total_steps,
        model=html.escape(model),
        messages_html=messages_html,
        thoughts_html=thoughts_html,
    )

    html_name = f"{basename}.html" if basename else "report.html"
    report_path = os.path.join(output_dir, html_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rendered)
    logger.info(f"Saved report: {report_path}")

    build_markdown_transcript(output_dir, config, total_steps, basename=basename)
    return report_path
