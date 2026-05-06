"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import math
import random
import re
import logging
from typing import List, Tuple, Optional, Dict, Set, TypedDict
from claude_client import ClaudeClient
from utils import is_position_in_place, get_place_at_position, PlaceConfig, generate_random_persona
from place_types import get_place_type_spec

logger = logging.getLogger(__name__)

# Constants
FALLBACK_REASONING_LENGTH = 100
MAX_MESSAGE_WORDS = 200
MAX_RETRY_TOKENS = 1500  # Expanded max_tokens used when a response is detected as truncated.

# Behavior layers (feature 4). Each step the agent is classified into one of
# three layers and the layer gates how often we actually call the LLM.
BEHAVIOR_LAYERS = ("transit", "dwelling", "interacting")
# In transit, reuse the current intent for this many consecutive steps before
# re-consulting the LLM. 1 = always call LLM (no caching).
TRANSIT_LLM_INTERVAL = 5

# Feature 2: semantic labels for the 0.0-1.0 relationship scale. These are
# fed into prompts so the LLM can infer how formal/intimate a message should
# be. Five buckets match the config-side documentation.
def relationship_label(level: float) -> str:
    if level < 0.1:
        return "stranger"
    if level < 0.3:
        return "face familiar"
    if level < 0.6:
        return "acquaintance"
    if level < 0.9:
        return "friend/colleague"
    return "close family/friend"


# Direction mappings (4 cardinal directions only)
# Coordinate system: X increases from left to right, Y increases from bottom to top
DIRECTION_MAP = {
    "up": (0, 1),      # Y+1 (move upward)
    "down": (0, -1),   # Y-1 (move downward)
    "left": (-1, 0),   # X-1 (move leftward)
    "right": (1, 0),   # X+1 (move rightward)
}


def time_band_labels(time_str: str) -> Tuple[str, str]:
    """Return (Japanese label, English label) for an HH:MM string.

    Used by prompts so the LLM knows which part of the day it is framed in,
    both in Japanese (matches the persona language) and in English (which
    modern LLMs ground more reliably than time-of-day words alone).
    """
    try:
        hh = int(time_str.split(':')[0])
    except Exception:
        return "", ""
    if 5 <= hh < 7:   return "早朝", "Early morning (around sunrise)"
    if 7 <= hh < 10:  return "朝ラッシュ", "Morning rush hour (commuting to work/school)"
    if 10 <= hh < 12: return "午前中", "Late morning"
    if 12 <= hh < 14: return "昼休み", "Lunch hour"
    if 14 <= hh < 17: return "午後", "Afternoon"
    if 17 <= hh < 20: return "夕方ラッシュ", "Evening rush hour (commuting home)"
    if 20 <= hh < 22: return "夜", "Evening"
    return "深夜", "Late night / pre-dawn"


def time_language_guardrail(time_str: str) -> str:
    """Return an NG-list of words that would be anachronistic this hour.

    LLMs drift to morning greetings/topics by default when the time is
    ambiguous. We anchor them explicitly: what to use, what to avoid.
    """
    try:
        hh = int(time_str.split(':')[0])
    except Exception:
        return ""
    if 5 <= hh < 11:
        return (
            "Language guardrail — this is MORNING in Tokyo:\n"
            "- Appropriate greetings: 「おはようございます」「おはよう」\n"
            "- Appropriate topics: 朝食, 出社, 始業, 今日の予定, モーニングコーヒー\n"
            "- Do NOT use evening greetings (「お疲れ様でした」「お先に失礼」「こんばんは」) unless addressing someone who worked overnight.\n"
            "- Do NOT speak as if 帰宅 or 夕食 were happening now."
        )
    if 17 <= hh < 22:
        return (
            "Language guardrail — this is EVENING in Tokyo (帰宅時間帯, commute home):\n"
            "- Appropriate greetings: 「お疲れ様です」「お先に失礼します」「こんばんは」\n"
            "- Appropriate topics: 帰宅, 夕食, 駅の混雑, 一日の振り返り, 明日の予定, 夕方の天気\n"
            "- Do NOT use morning greetings: 「おはようございます」「おはよう」\n"
            "- Do NOT talk about 朝食, 朝コーヒー, 出社前, 始業, 「今日も頑張りましょう」, 「これから一日が始まる」 as if they were happening now.\n"
            "- 「今日」 is fine when referring to the day that is wrapping up (例: 「今日もお疲れ様」「今日は疲れた」)."
        )
    if hh >= 22 or hh < 5:
        return (
            "Language guardrail — this is LATE NIGHT in Tokyo:\n"
            "- Appropriate topics: 終電, 夜の静けさ, 明日の予定, お疲れ, 帰り道\n"
            "- Do NOT use morning greetings.\n"
            "- Do NOT reference 朝の活動 (朝食, 出社, 始業) as if they were happening now."
        )
    # 11-17 daytime / lunch
    return (
        "Language guardrail — this is DAYTIME in Tokyo:\n"
        "- Morning-specific greetings (「おはよう」) are out of place after ~10時.\n"
        "- Evening-specific greetings (「こんばんは」「お疲れ様でした」) are premature before 17時.\n"
        "- Appropriate framing: 昼食, ちょっとした休憩, 外回り, 午後の予定."
    )


class MessageDecision(TypedDict):
    """Type definition for agent message decision"""
    message: str  # Message to communicate with nearby agents
    reasoning: str  # Explanation of the message decision


class ActionDecision(TypedDict, total=False):
    """Agent action decision (intent-based, feature 2).

    action_type is one of:
      - "walk_toward":  walk toward target_place (a place name)
      - "walk_along":   walk one step in a cardinal direction
      - "enter":        step into target_place (if close enough)
      - "stay":         hold position
      - "approach":     walk toward target_agent (a persona name or agent id)
      - "wander":       short random step

    Only the fields relevant to the chosen action_type need be populated.
    """
    action_type: str
    target_place: Optional[str]
    target_agent: Optional[str]
    direction: Optional[str]
    memory: str
    reasoning: str
    # Legacy compatibility: downstream code may still read "action".
    action: str


class Agent:
    """LLM-based agent in 2D worlds with multiple places."""

    def __init__(
        self,
        agent_id: int,
        initial_position: Tuple[int, int],
        llm_client: ClaudeClient,
        communication_radius: float,
        half_space_size: int,
        places: List[PlaceConfig],
        num_agents: int,
        gender: str = "male",
        memory_limit: int = 20,
        memory_size: int = 5,
        message_history_limit: int = 10,
        message_context_size: int = 3,
        persona: Optional[Dict] = None,
        movement_base_cells: int = 1,
        movement_variance: int = 0,
        initial_relationships: Optional[Dict[int, float]] = None,
        navigator=None,
        minimal_prompt: bool = False,
        skip_decision: bool = False,
        scene_phrase: str = "a small fixed indoor scene",
        injected_context: str = "",
        global_channel: bool = False,
        chat_thread_mode: bool = False,
    ):
        self.id = agent_id
        self.position = initial_position
        self.llm_client = llm_client
        self.communication_radius = communication_radius
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.gender = gender
        self.navigator = navigator
        self.minimal_prompt = bool(minimal_prompt)
        self.skip_decision = bool(skip_decision)
        self.scene_phrase = scene_phrase or "a small fixed scene"
        # moltbook風 グローバルチャンネルモード (= 全 agent が共有チャンネルを読む)。
        # True のとき system_prompt の発話判定を「self-decision 強型」に切り替える。
        self.global_channel = bool(global_channel)
        # moltbook風 chat session thread化 (= 各 agent が独立した Gemini chat session を持つ)。
        # True のとき message phase で send_to_chat 経由、履歴は session 内で蓄積される。
        # session は最初の message phase 呼び出し時に lazy 作成される。
        self.chat_thread_mode = bool(chat_thread_mode)
        self.chat_session = None  # lazy: send_message_via_chat 初回時に start_chat_session
        # Phase A: 場所の現場知覚情報を system_prompt に挿入する用テキスト。
        # 空文字なら注入しない。Phase B 以降は agent が場所近接時にロードする設計を想定。
        self.injected_context = injected_context or ""

        # Movement speed (cells per "move" action). Actual distance per step is
        # base + uniform(-variance, +variance), clamped to >= 1.
        self.movement_base_cells = max(1, int(movement_base_cells))
        self.movement_variance = max(0, int(movement_variance))
        # smoke22: 車いす利用 persona は移動速度を半減 (現実感反映)。
        if persona is not None and (persona.get('mobility') or '').strip() == 'wheelchair':
            self.movement_base_cells = max(1, self.movement_base_cells // 2)
            self.movement_variance = max(0, self.movement_variance // 2)

        # Minimal persona (name, age, occupation, background, speech_style).
        # Kept as a flat dict so the second-stage expansion can add fields
        # without breaking existing call sites.
        if persona is not None:
            self.persona = persona
        else:
            self.persona = generate_random_persona(agent_id, gender)

        # Memory parameters
        self.memory_limit = memory_limit  # Maximum memories to store
        self.memory_size = memory_size  # Number of recent memories to use in prompt
        self.message_history_limit = message_history_limit  # Maximum messages to store
        self.message_context_size = message_context_size  # Number of recent messages to use in prompt

        # Agent state
        self.in_place = False
        self.current_place: Optional[str] = None  # Name of the place the agent is in (None if outside)
        self.memory: List[str] = []  # Store past decisions and observations
        # Phase B 用: 教室Phase からの持ち越し記憶。persona dict 内の "initial_memory"
        # (list[str]) があれば、起動時に self.memory に prepend する。これで FW Phase の
        # agent は座学Phaseで描いた未来像・FW意図・キー記憶を最初から保持してスタートする。
        init_mem = self.persona.get("initial_memory") if self.persona else None
        if isinstance(init_mem, list):
            for line in init_mem:
                if isinstance(line, str) and line.strip():
                    self.memory.append(line.strip())
        self.received_messages: List[Dict] = []  # Messages from other agents
        self.sent_messages: List[Dict] = []  # This agent's own past utterances (for self-context, prevents lock-in repetition)
        # 圧縮記憶 (2026-05-04): rolling buffer の限界対策。
        # step が memory_compression_interval (=5) の倍数になるたび、直近 N 件を
        # 1 文に要約して archived_summaries に push (Gemini 呼び出し)。raw window はそのまま rolling 続行。
        # prompt には archived_summaries (固定保管) + 直近 raw N件 を載せる。
        self.archived_summaries: List[str] = []

        # Statistics
        self.steps_in_place = 0
        self.steps_outside_place = 0
        self.total_moves = 0

        # Phase B: 各 place を初めて bbox 進入した・初めて enter した、を記録する。
        # perceive_pass / perceive_enter の重複注入を避け、また「ここは前にも来た」記録に使う。
        self.visited_places: Set[str] = set()
        self.entered_places: Set[str] = set()
        # Phase B: 自分が会話を交わした host (企業担当者) の name 集合。
        # 会話発生時に simulation 側から rule-based で追加される (sender / receiver の双方)。
        # working_state ブロックで「未訪問host」を計算するため、および evidence-bound survey で使う。
        self.talked_hosts: Set[str] = set()
        # Phase B: 累積物理量。simulation.py の身体感覚記録機構が読み書きする。
        self.cumulative_distance_cells: float = 0.0
        self.cumulative_steps: int = 0
        self._last_position: Optional[Tuple[int, int]] = None

        # Behavior-layer state (feature 4).
        # current_intent caches the last LLM-chosen action so transit steps can
        # reuse it without another API call. transit_step_counter tracks how
        # many consecutive transit steps have reused the cache.
        self.behavior_layer: str = "dwelling"
        self.current_intent: Optional[ActionDecision] = None
        self.transit_step_counter: int = 0

        # Time + goal context (feature 5). Simulation updates these each step
        # before calling decide_message / decide_action, so prompts can show
        # the agent a plausible wall-clock time and a time-of-day mood.
        self.current_time_str: str = ""
        self.current_context: str = ""
        self.current_goal: str = ""

        # Feature 2: relationship graph, internal state, social identities.
        # Initial state follows the "healthy commuter" baseline. Simulation
        # refreshes these each step before the LLM phases.
        self.internal_state: Dict[str, float] = {
            "energy": 80.0,
            "hunger": 20.0,
            "social_fatigue": 0.0,
            "mood": 0.0,
        }
        # Relationships are keyed by the other agent's id. 0.05 is the implicit
        # baseline for any id not present in the dict (first-meeting level).
        self.relationships: Dict[int, float] = dict(initial_relationships) if initial_relationships else {}
        self.social_identities: List[Dict] = list(self.persona.get('social_identities', []) or [])
        # Feature 4.5: events this agent has learned about (direct proximity or
        # conversation keyword). Simulation fills this in each step.
        self.known_events: set = set()
        # Phase 2.5: how each known event was learned ("direct"|"conversation"|"notification")
        # and at which step. Keyed by event name.
        self.awareness_source: Dict[str, str] = {}
        self.awareness_step: Dict[str, int] = {}

    def is_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None
    
    def distance_to(self, other_position: Tuple[int, int]) -> float:
        """Calculate Euclidean distance to another position"""
        dx = self.position[0] - other_position[0]
        dy = self.position[1] - other_position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_nearby_agents(self, all_agents: List['Agent']) -> List['Agent']:
        """Get agents within communication radius and in the same area (same place or both outside)
        
        Communication rules:
        - Agents can communicate if BOTH are outside places
        - Agents can communicate if BOTH are in the SAME place
        - Agents CANNOT communicate if one is inside a place and the other is outside
        - Agents CANNOT communicate if they are in DIFFERENT places
        """
        nearby = []
        for agent in all_agents:
            if agent.id != self.id:
                dist = self.distance_to(agent.position)
                # Must be within radius AND in the same area:
                # - Both outside places, OR
                # - Both in the same place (same place name)
                # NOTE: Agents inside a place CANNOT communicate with agents outside places
                same_area = (
                    (not self.in_place and not agent.in_place) or
                    (self.in_place and agent.in_place and self.current_place == agent.current_place)
                )
                if dist <= self.communication_radius and same_area:
                    nearby.append(agent)
        return nearby
    
    
    def _position_to_rough_direction(self, other_position: Tuple[int, int]) -> str:
        """Convert an exact neighbour position into a rough directional description.

        Agents shouldn't be announcing other agents' precise coordinates — people
        don't do that. This collapses (dx, dy) into a coarse proximity + cardinal
        direction string instead.
        """
        dx = other_position[0] - self.position[0]
        dy = other_position[1] - self.position[1]

        direction_parts = []
        if abs(dy) > 2:
            direction_parts.append("north" if dy > 0 else "south")
        if abs(dx) > 2:
            direction_parts.append("east" if dx > 0 else "west")

        distance = (dx * dx + dy * dy) ** 0.5
        if distance < 3:
            proximity = "very close"
        elif distance < 6:
            proximity = "nearby"
        else:
            proximity = "a bit far"

        if not direction_parts:
            return proximity
        return f"{proximity} ({'-'.join(direction_parts)})"

    def _build_nearby_agents_context(self, nearby_agents: List['Agent'], include_position: bool = True) -> str:
        """Build context string about nearby agents, using persona names.

        Args:
            nearby_agents: List of nearby agents
            include_position: If True, include a rough directional hint AND age/gender/occupation
                (used in the action/decision phase). If False (used in the message phase),
                drop the demographic block — it's per-step waste once agents have introduced
                themselves; this saves significant uncached input tokens at scale.
        """
        if not nearby_agents:
            return "No nearby agents."

        nearby_info = []
        for agent in nearby_agents:
            name = agent.persona.get('name', f"Person {agent.id}")

            if agent.in_place:
                place_info = next((p for p in self.places if p['name'] == agent.current_place), None)
                if place_info is None:
                    raise ValueError(
                        f"Agent {agent.id} is in place '{agent.current_place}' but this place is not found in configuration."
                    )
                place_type = place_info['type']
                status = f"is in {agent.current_place} ({place_type})"
            else:
                status = "is outside the places"

            rel_level = self.get_relationship(agent.id)
            rel_tag = relationship_label(rel_level)
            is_host = bool(agent.persona.get('is_host'))
            if include_position:
                age = agent.persona.get('age', '?')
                occupation = agent.persona.get('occupation', '?')
                direction = self._position_to_rough_direction(agent.position)
                host_tag = "[企業担当者・大人] " if is_host else ""
                nearby_info.append(
                    f"{host_tag}{name} ({age}, {agent.gender}, {occupation}) — {rel_tag} — {status}, roughly {direction}"
                )
            else:
                # message phase は demographic を省略してコスト削減するが、
                # host (企業担当者・大人) だけは年齢・職業を残す。
                # 学生が host を「同級生扱い (タメ口・くん付け)」する事故を防ぐため。
                if is_host:
                    age = agent.persona.get('age', '?')
                    occupation = agent.persona.get('occupation', '?')
                    nearby_info.append(
                        f"[企業担当者・大人] {name} ({age}歳, {occupation}) — {rel_tag} — {status}"
                    )
                else:
                    nearby_info.append(
                        f"{name} — {rel_tag} — {status}"
                    )
        return "\n".join(nearby_info)
    
    def _build_memory_context(self) -> str:
        """Build context string from agent memory.

        構造:
          1. 訪問履歴 (rule-based 固定 prefix)
          2. 圧縮記憶 archived_summaries (固定保管、消えない)
          3. 直近 raw memory (rolling buffer 直近 memory_size 件)
        """
        # 1. 訪問履歴 prefix (場所の再訪抑制)
        visited_line = ""
        passed = sorted(self.visited_places)
        entered = sorted(self.entered_places)
        passed_only = [p for p in passed if p not in self.entered_places]
        if entered or passed_only:
            parts = []
            if entered:
                parts.append("入場済み: " + ", ".join(entered))
            if passed_only:
                parts.append("通過済み(未入場): " + ", ".join(passed_only))
            visited_line = "[訪問履歴] " + " / ".join(parts)

        # 2. 圧縮記憶 archived (長期記憶)
        archived_lines: List[str] = []
        for i, summary in enumerate(self.archived_summaries):
            archived_lines.append(f"[要約#{i+1}] {summary}")

        # 3. 直近 raw memory
        if self.memory:
            recent_memory = self.memory[-self.memory_size:]
            raw_lines = [f"- {m}" for m in recent_memory]
        else:
            raw_lines = []

        # 組み立て
        sections = []
        if visited_line:
            sections.append(visited_line)
        if archived_lines:
            sections.append("\n".join(archived_lines))
        if raw_lines:
            sections.append("(直近の生記憶):\n" + "\n".join(raw_lines))
        if not sections:
            return "No previous experiences."
        return "\n".join(sections)
    
    @staticmethod
    def _ngram_set(text: str, n: int = 4) -> set:
        """日本語向け文字 N-gram (空白除去後)。短すぎ・空文字は空 set。"""
        if not text:
            return set()
        t = re.sub(r"\s+", "", text)
        if len(t) < n:
            return {t}
        return {t[i:i+n] for i in range(len(t) - n + 1)}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    def filter_loop_message(self, decision: 'MessageDecision', partner_id: Optional[int],
                            jaccard_threshold: float = 0.50) -> 'MessageDecision':
        """LLM が出した message が、同じ相手への直近自分発話と高類似度なら silent 化する後フィルタ。

        Phase A 等の「キャンセル → LLM が再生成 → またキャンセル」を物理的に止める仕組み。
        threshold は既定 0.50 (4-gram Jaccard)。"""
        if not decision or partner_id is None:
            return decision
        msg = (decision.get('message') or '').strip()
        if not msg:
            return decision
        my_msgs_to = [m for m in self.sent_messages if m.get('to') == partner_id]
        if not my_msgs_to:
            return decision
        # 直近最大 5 件と比較
        recent = my_msgs_to[-5:]
        cand = self._ngram_set(msg)
        for prev in recent:
            prev_text = (prev.get('content') or prev.get('message') or '').strip()
            if not prev_text:
                continue
            sim = self._jaccard(cand, self._ngram_set(prev_text))
            if sim >= jaccard_threshold:
                logger.info(
                    f"Agent {self.id}: loop-filter silenced message (Jaccard={sim:.2f} vs prev to {partner_id})"
                )
                decision['message'] = ""
                decision['reasoning'] = (
                    f"(後フィルタ silent化: 直近発話と類似度 {sim:.2f} >= {jaccard_threshold})"
                    + " / " + str(decision.get('reasoning', ''))[:120]
                )
                return decision
        return decision

    def _detect_loop_partners(self, nearby_agents: List['Agent'],
                              jaccard_threshold: float = 0.30,
                              min_pairs_similar: int = 1) -> Dict[int, List[Dict]]:
        """nearby agents 各人について、自分の直近3発話の互い類似度を見て
        「ループ気味」と判定された agent_id をキー、直近3発話を値として返す。

        直近3件のペアのうち、Jaccard >= threshold が min_pairs_similar 以上で「ループ判定」。
        実用的には min_pairs_similar=1 (= 直近3件のうちどこか2件が似てたら警告)。"""
        result: Dict[int, List[Dict]] = {}
        if not nearby_agents or not self.sent_messages:
            return result
        for ag in nearby_agents:
            my_msgs_to = [m for m in self.sent_messages if m.get('to') == ag.id]
            if len(my_msgs_to) < 2:
                continue
            recent = my_msgs_to[-3:]
            grams = []
            for m in recent:
                content = m.get('content') or m.get('message') or ''
                grams.append(self._ngram_set(content))
            sim_pairs = 0
            for i in range(len(grams)):
                for j in range(i + 1, len(grams)):
                    if self._jaccard(grams[i], grams[j]) >= jaccard_threshold:
                        sim_pairs += 1
            if sim_pairs >= min_pairs_similar:
                result[ag.id] = recent
        return result

    def _build_per_partner_history(self, nearby_agents: List['Agent']) -> str:
        """nearby agents 各人ごとに、自分の直近発話 (top 3) を整理して表示。
        『同じ人に同じことを言う』を防ぐため、LLM に「もう A には〇〇を言った」を可視化。
        ループ検知された相手には強い警告を併記する (cooldown step1)。"""
        if not nearby_agents or not self.sent_messages:
            return ""
        loop_partners = self._detect_loop_partners(nearby_agents)
        sections = []
        for ag in nearby_agents:
            my_msgs_to = [m for m in self.sent_messages if m.get('to') == ag.id]
            if not my_msgs_to:
                continue
            recent = my_msgs_to[-3:]
            ag_name = ag.persona.get('name', f"Agent {ag.id}")
            lines = []
            for m in recent:
                step = m.get('step', '?')
                content = (m.get('content') or m.get('message') or '')[:120]
                if content:
                    lines.append(f"  - [step {step}] 「{content}」")
            if not lines:
                continue
            section_header = f"あなたが {ag_name} に既に言ったこと:"
            if ag.id in loop_partners:
                section_header = (
                    f"⚠️ **警告: あなたは {ag_name} に対して直近で類似テーマの発話を繰り返しています。**"
                    f"\n下の3件は内容が互いに類似していると検出されました。"
                    f"\n**今回の発話で {ag_name} に同じテーマ・同じキーワードを送るのは禁止です**。"
                    f"\n選択肢: (1) 完全に違う話題で {ag_name} に話す、"
                    f"(2) {ag_name} ではなく**別の人**に話しかける、"
                    f"(3) silent (空文字) を選ぶ。\n{ag_name} への直近発話 (これと類似する内容を再送するのは禁止):"
                )
            sections.append(section_header + "\n" + "\n".join(lines))
        if not sections:
            return ""
        return "=== あなたの過去発話 (相手別) — 同じ人に同じテーマを再送しない ===\n" + "\n\n".join(sections) + "\n"

    def _build_messages_context(self) -> str:
        """Build a chronological context string merging received and sent messages.

        v6: include this agent's own recent utterances (sent_messages) alongside
        received ones, so the LLM sees the conversational thread in a unified
        timeline and avoids re-asking the same opener (lock-in prevention).
        """
        my_name = self.persona.get('name', f"Agent {self.id}") if self.persona else f"Agent {self.id}"

        merged = []
        for msg in self.received_messages:
            merged.append({
                "step": msg.get('step', 0),
                "from_name": msg.get('from_name') or f"Person {msg.get('from', '?')}",
                "to_name": my_name,
                "content": msg.get('content', ''),
                "is_self": False,
            })
        for msg in self.sent_messages:
            merged.append({
                "step": msg.get('step', 0),
                "from_name": my_name,
                "to_name": msg.get('to_name') or f"Person {msg.get('to', '?')}",
                "content": msg.get('content', ''),
                "is_self": True,
            })

        if not merged:
            return "(no recent conversation)"

        merged.sort(key=lambda m: m['step'])
        # Cap at a generous window so both sides of a back-and-forth stay visible.
        window = max(self.message_context_size * 2, 6)
        recent = merged[-window:]
        lines = []
        for m in recent:
            step_prefix = f"[step {m['step']}] " if m['step'] else ""
            arrow = "→"
            tag = " (you)" if m['is_self'] else ""
            lines.append(f"{step_prefix}{m['from_name']}{tag} {arrow} {m['to_name']}: 「{m['content']}」")
        return "\n".join(lines)
    
    def _build_fire_section(self, fire_info: Optional[List[Dict]]) -> str:
        """Build fire event section for prompt. Returns empty string if no fire info.

        Only quantitative data is provided: position, intensity, radius, distance.
        No qualitative descriptions (e.g. "dangerous", "evacuate") are included.
        Supports multiple fires.
        """
        if not fire_info:
            return ""

        lines = ["\n=== FIRE EVENT ==="]
        for fi in fire_info:
            lines.append(
                f"Fire \"{fi['name']}\":\n"
                f"  Position: ({fi['fire_position'][0]}, {fi['fire_position'][1]})\n"
                f"  Intensity: {fi['intensity']} (scale: 0.0 to 1.0)\n"
                f"  Radius: {fi['radius']}\n"
                f"  Your distance: {fi['agent_distance']}"
            )
        return "\n".join(lines) + "\n"

    def _build_events_section(self, events_info: Optional[List[Dict]]) -> str:
        """Build CURRENT EVENTS section for the user prompt (feature 4.5).

        Phase 2.5: branches by awareness_source so that the same underlying
        event is framed differently depending on how the agent learned about
        it (directly witnessed / heard from someone / push notification).
        The LLM still chooses how/whether to act; only framing changes.
        """
        if not events_info:
            return ""
        lines = ["\n=== CURRENT EVENTS ==="]
        for ev in events_info:
            source = ev.get('awareness_source', 'direct')
            desc = ev.get('description', '')
            affected = ev.get('affected_place', '')

            if source == 'direct':
                if desc:
                    lines.append(desc)
                if affected:
                    lines.append(
                        f"Affected place: {affected}. "
                        "You witnessed the announcement directly."
                    )
            elif source == 'conversation':
                if desc:
                    lines.append(f"Someone told you: {desc}")
                if affected:
                    lines.append(
                        f"Affected place: {affected}. "
                        "Consider the reliability of the source."
                    )
            elif source == 'notification':
                if desc:
                    lines.append(f"[BREAKING NEWS] {desc}")
                if affected:
                    lines.append(
                        f"Affected place: {affected}. "
                        "You received a push notification — "
                        "you are among the first to know."
                    )
            else:
                if desc:
                    lines.append(desc)
                if affected:
                    lines.append(f"Affected place: {affected}.")

        lines.append(
            "\nConsider your options (no option is recommended):\n"
            "- Walk to an alternative station (e.g. 地下鉄A駅 or 地下鉄B駅)\n"
            "- Wait it out at a cafe/restaurant nearby\n"
            "- Go home or to your destination on foot\n"
            "- Stay with companions at the plaza"
        )
        return "\n".join(lines) + "\n"

    def _is_truncated_response(self, raw_response: str) -> bool:
        """Detect if the response was truncated due to max_tokens limit.

        Uses brace balance as the signal: an open '{' without a matching '}'
        indicates the JSON payload was cut off. Markdown fences or trailing
        commentary are ignored as long as the braces balance.
        """
        if not raw_response:
            return False

        stripped = raw_response.strip()
        if '{' not in stripped:
            # No JSON at all — likely a garbage response but not truncation we can fix by retrying with more tokens.
            return False
        return stripped.count('{') > stripped.count('}')

    def _limit_message_words(self, message: str) -> str:
        """Check message word count and warn if exceeds MAX_MESSAGE_WORDS"""
        if not message:
            return message
        
        words = message.split()
        if len(words) > MAX_MESSAGE_WORDS:
            logger.warning(
                f"Agent {self.id}: Message exceeds {MAX_MESSAGE_WORDS} words limit "
                f"({len(words)} words). Message will be sent as-is."
            )
        
        return message
    
    def _build_place_locations_text(self) -> str:
        """Build static description of all place locations (used in system prompt)."""
        place_locations = []
        for place in self.places:
            place_type = place['type']
            hx = place.get('half_size_x', place.get('half_size', 5))
            hy = place.get('half_size_y', place.get('half_size', 5))
            spec = get_place_type_spec(place_type)
            line = (
                f"{place['name']} ({place_type} — {spec['atmosphere']}): "
                f"center ({place['center_x']}, {place['center_y']}), "
                f"covers X {place['center_x'] - hx} to {place['center_x'] + hx}, "
                f"Y {place['center_y'] - hy} to {place['center_y'] + hy}"
            )
            cap = place.get('capacity')
            if cap is not None:
                line += f", capacity {cap}"
            if (place.get('attributes') or {}).get('enterable') is False:
                line += " [立入不可・enter禁止]"
            place_locations.append(line)
        return "\n".join(place_locations)

    def _build_nearby_places_context(self) -> str:
        """Per-step text describing each place's distance and rough direction from the agent."""
        if not self.places:
            return "No places configured."
        lines = []
        ax, ay = self.position
        for place in self.places:
            cx, cy = place['center_x'], place['center_y']
            hx = place.get('half_size_x', place.get('half_size', 5))
            hy = place.get('half_size_y', place.get('half_size', 5))
            dist = math.hypot(cx - ax, cy - ay)
            direction = self._position_to_rough_direction((cx, cy))
            inside = abs(cx - ax) <= hx and abs(cy - ay) <= hy
            if inside:
                marker = "[you are here]"
            elif dist <= self.movement_base_cells * 1.5:
                marker = "[right next to you]"
            else:
                marker = ""
            lines.append(
                f"- {place['name']} ({place['type']}): {dist:.0f} cells away, {direction} {marker}".rstrip()
            )
        return "\n".join(lines)

    def _build_persona_section(self) -> str:
        """WHO YOU ARE block — persona name/age/occupation + 3-dim temperament
        + background (2-3 sentences) + catchphrase + goal.

        Kept in user_prompt (not system_prompt) so the per-agent variation
        doesn't break prompt-cache boundaries.
        """
        p = self.persona
        name = p.get('name', f"Person {self.id}")
        age = p.get('age', '?')
        gender = p.get('gender', self.gender)
        occupation = p.get('occupation', '?')
        background = p.get('background', '')
        speech = p.get('speech_style', '')
        catchphrase = p.get('catchphrase', '')
        # current_goal は system_prompt 側 (_build_goal_block_for_system) に
        # 移動済み。persona_section からは出さない (毎step uncached重複削減)。
        biases = p.get('cognitive_biases', []) or []

        # 3-dim temperament (each high/mid/low). Render in JP labels so the LLM reads them naturally.
        temp_map = {
            "extroversion": {"high": "社交的", "mid": "どちらでもない", "low": "内向的"},
            "optimism":     {"high": "楽天的", "mid": "どちらでもない", "low": "心配性"},
            "curiosity":    {"high": "好奇心旺盛", "mid": "どちらでもない", "low": "慎重"},
        }
        ext = p.get('temperament_extroversion')
        opt = p.get('temperament_optimism')
        cur = p.get('temperament_curiosity')
        temperament_line = None
        if any([ext, opt, cur]):
            parts = []
            if ext: parts.append(temp_map["extroversion"].get(ext, ext))
            if opt: parts.append(temp_map["optimism"].get(opt, opt))
            if cur: parts.append(temp_map["curiosity"].get(cur, cur))
            temperament_line = " / ".join(parts)

        lines = [
            "=== WHO YOU ARE ===",
            f"Name: {name} ({age}・{gender})",
            f"Occupation: {occupation}",
        ]
        if temperament_line:
            lines.append(f"気質: {temperament_line}")
        if background:
            lines.append(f"背景: {background}")
        if catchphrase:
            lines.append(f"口癖: 「{catchphrase}」")
        if speech:
            lines.append(f"Speech style: {speech}")
        if biases:
            lines.append("Cognitive tendencies:")
            for b in biases:
                lines.append(f"  - {b}")
        lines.append(
            "Act according to this identity. Other people around you know you by name, not by ID."
        )
        return "\n".join(lines) + "\n"

    def _build_internal_state_section(self) -> str:
        s = self.internal_state
        return (
            "=== YOUR INTERNAL STATE ===\n"
            f"Energy: {int(s.get('energy', 0))}/100, "
            f"Hunger: {int(s.get('hunger', 0))}/100, "
            f"Social fatigue: {int(s.get('social_fatigue', 0))}/100\n"
        )

    def _build_group_identities_section(self) -> str:
        """ACTIVE GROUP IDENTITIES — list high-salience groups only.

        Salience threshold 0.4 so trivial group memberships don't clutter.
        """
        if not self.social_identities:
            return ""
        active = [
            g for g in self.social_identities
            if float(g.get('base_salience', 0.0) or 0.0) >= 0.4
        ]
        if not active:
            return ""
        lines = ["=== ACTIVE GROUP IDENTITIES ==="]
        for g in active:
            group = g.get('group_name', '?')
            sal = float(g.get('base_salience', 0.0) or 0.0)
            norms = g.get('norms', '')
            extra = f": {norms}" if norms else ""
            lines.append(f"{group} (salience: {sal:.1f}){extra}")
        return "\n".join(lines) + "\n"

    def _build_time_context_section(self) -> str:
        """Compose a CURRENT TIME & CONTEXT block for the user prompt.

        Carries both the clock (HH:MM), a bilingual band label (夕方ラッシュ /
        Evening rush hour), and an explicit NG/OK list so the LLM doesn't
        drift to morning greetings in an evening scenario or vice-versa.
        Empty string when simulation never set any time context — keeps the
        block out of the prompt for configs without time_patterns.
        """
        if not (self.current_time_str or self.current_context or self.current_goal):
            return ""
        lines = ["=== CURRENT TIME & CONTEXT ==="]
        if self.current_time_str:
            jp, en = time_band_labels(self.current_time_str)
            if jp and en:
                lines.append(f"Current time: {self.current_time_str} ({jp} / {en})")
            else:
                lines.append(f"Current time: {self.current_time_str}")
        if self.current_context:
            lines.append(f"Neighborhood mood: {self.current_context}")
        # NOTE: current_goal は system_prompt 側 (_build_goal_block_for_system) に
        # 移動した。毎 step の uncached input を削減するため。重複出力しない。
        guard = time_language_guardrail(self.current_time_str) if self.current_time_str else ""
        if guard:
            lines.append("")
            lines.append(guard)
        return "\n".join(lines) + "\n"

    def _build_world_description(self) -> str:
        """Build short world description based on unique place types."""
        place_types = [p['type'] for p in self.places]
        unique_types = list(set(place_types))
        return f"a 2D world with multiple places ({', '.join(unique_types)})"

    def _build_goal_block_for_system(self) -> str:
        """中心問い (current_goal) を system_prompt 末尾に挿入する用。
        全 agent 共通 (or 同 phase 内で固定) の中心問いを system_prompt 側に置くと
        Gemini context cache に乗って毎 step の uncached input を削減できる。
        agent 固有の current_goal がある場合は user_prompt 側を使う旧挙動に戻す。
        """
        g = (self.persona.get('current_goal', '') or self.current_goal or '').strip()
        if not g:
            return ""
        return (
            "\n\n=== CENTRAL QUESTION (中心問い) ===\n"
            f"{g}\n"
        )

    def _build_fw_task_block_for_system(self) -> str:
        """FW中の必須課題 (例: 2社以上の企業担当者を訪問) を system_prompt に固定挿入。
        initial_memory にも入れているが、rolling buffer から消えると後半 step で意識から
        外れる問題があったため、cache 領域に置く。Phase A など fw_task が無い persona は空。"""
        t = (self.persona.get('fw_task') or '').strip()
        if not t:
            return ""
        return (
            "\n\n=== 今日の必須課題 (FW) ===\n"
            f"{t}\n"
            "(これは今日の必須課題です。FW 中、常に意識し、達成に向けて自然に行動してください。)\n"
        )

    def _build_working_state_block(self) -> str:
        """毎 step 動的に変わる「現在の進捗・未解決の問い」を system_prompt 末尾に固定挿入。
        記憶ではなく現在状態 (working memory)。rolling buffer の押し出しに依存しない。

        含めるもの:
          - 残り step / 集合場所への帰還リマインダ (FW 終了時に東西自由通路へ戻る)
          - 訪問進捗 (FW: 必須N社のうち何社と話したか / 残りいくつ)
          - 未訪問の企業担当者 (max 6 件)
          - field_questions 中まだ確かめていない問い (max 3 件)

        Phase A や fw_task のない persona では working_state は空 (代わりに別の中心問い等が
        既存の goal/fw block に出る)。"""
        if not self.persona.get('fw_task'):
            return ""

        all_hosts = self.persona.get('all_host_names') or []
        talked = sorted(self.talked_hosts)
        unvisited = [h for h in all_hosts if h not in self.talked_hosts]
        required = 2  # FW_TASK の必須数
        talked_count = len(talked)
        remaining = max(0, required - talked_count)

        fq = self.persona.get('field_questions') or []
        fq_lines = []
        for q in fq[:3]:
            if isinstance(q, dict):
                qtext = q.get("question") or q.get("q") or str(q)
            else:
                qtext = str(q)
            fq_lines.append(f"  - {qtext}")

        block = "\n\n=== WORKING STATE (現在の進捗・未解決の問い) ===\n"
        block += (
            "(注: 以下は「いま何が完了して、何が残っているか」の現在状態です。"
            "記憶ではないので毎step更新されます。記憶 (PREVIOUS MEMORY) と切り離して、"
            "ここに書かれている残タスク・未解決問いを優先して行動・発話してください。)\n\n"
        )
        # FW 残り時間 (= 集合場所への帰還リマインダ)
        cur_step = getattr(self, 'current_step', None)
        total_steps = self.persona.get('total_steps')
        if cur_step is not None and total_steps:
            remaining_steps = max(0, int(total_steps) - int(cur_step))
            remaining_min = remaining_steps * 2  # 2分/step
            block += (
                f"[残り時間] 進行: {cur_step}/{total_steps} step "
                f"(= 残り {remaining_steps} step / 約 {remaining_min} 分)\n"
            )
            # 終盤閾値: 残り 20% を切ったら集合場所への戻り移動を「意識」させる (修正案A: 弱め表現)
            if total_steps and remaining_steps <= max(3, int(total_steps) * 0.20):
                block += (
                    "[終盤] FW 終了が近い。今いる場所での観察・対話を続けつつ、"
                    "集合場所 (東西自由通路) への戻りも視野に入れて動くこと。\n"
                )
        if all_hosts:
            block += (
                f"[訪問進捗] 必須: {required}社以上の企業担当者と話す / "
                f"既に話した: {talked_count}社"
                + (f" ({', '.join(talked)})" if talked else "")
                + f" / 残り: {remaining}社\n"
            )
            if unvisited:
                shown = unvisited[:6]
                more = "" if len(unvisited) <= 6 else f" 他+{len(unvisited)-6}社"
                block += f"[まだ訪問していない企業担当者] {', '.join(shown)}{more}\n"
        if fq_lines:
            block += "[まだ確かめていない問い (FW で観察・対話で答えを探すべきもの)]\n"
            block += "\n".join(fq_lines) + "\n"
        return block

    MEMORY_COMPRESSION_INTERVAL = 5  # N=5 で圧縮 (10分=5stepごとに 1要約 archive)

    MEMORY_COMPRESSION_SYSTEM = (
        "あなたは agent の長期記憶を生成します。"
        "以下は、ある人物が直近 5 step (実時間 ~10 分) の間に書いた memory + reasoning です。"
        "これを 1 文 (60-120字) のエピソード要約 にまとめてください。\n\n"
        "ルール:\n"
        "- 「いつ・どこで・誰と・何が起きた・どう感じた」を圧縮\n"
        "- 数字や具体的な発言は本人が後で参照したくなるレベルで保持\n"
        "- 「色んなことがあった」のような抽象語は禁止\n"
        "- 出力は要約文 1 文のみ。前置き・コードブロック禁止"
    )

    def maybe_compress_memory(self, step: int) -> bool:
        """step が圧縮間隔の倍数かつ raw が十分溜まっていれば、
        直近 N 件を 1 文要約して archived_summaries に push する。
        raw は捨てない (rolling buffer はそのまま継続)。
        return True if 圧縮した場合 (LLM call が走った)."""
        N = self.MEMORY_COMPRESSION_INTERVAL
        if step <= 0 or step % N != 0:
            return False
        # 直近 N 件を取り出す (raw memory の末尾)
        recent = self.memory[-N:]
        if len(recent) < N:
            return False
        # initial_memory (Phase B 開始時の handoff) は要約しない
        # → recent に [前提] [今日のFW課題] のような initial_memory が混じってたら除外
        recent_filtered = [m for m in recent if not m.startswith("[")
                           or m.startswith("[step")
                           or m.startswith("[現地で見えた")
                           or m.startswith("[中に入って")
                           or m.startswith("[累積") or m.startswith("[身体")]
        if len(recent_filtered) < 2:
            return False
        # Gemini で要約
        try:
            user = "memory + reasoning (直近 5 step):\n" + "\n".join(f"- {m}" for m in recent_filtered)
            summary = self.llm_client.generate(self.MEMORY_COMPRESSION_SYSTEM, user, temperature=0.3, max_tokens=200)
            if summary and summary.strip():
                self.archived_summaries.append(summary.strip()[:300])
                return True
        except Exception as e:
            pass
        return False

    def _build_field_questions_block_for_system(self) -> str:
        """Phase A の handoff から持ち越した field_questions (今日 FW で確かめたい問い) を
        system_prompt 末尾に固定挿入する。memory rolling buffer から押し出されても、
        毎 step「自分の問い」を保持できる。cache 領域なので uncached input は増えない。
        smoke21: _build_handoff_block_for_system に統合 (廃止予定)。"""
        fqs = self.persona.get('field_questions') or []
        fqs = [q for q in fqs if isinstance(q, str) and q.strip()]
        if not fqs:
            return ""
        body = "\n".join(f"  {i+1}. {q.strip()}" for i, q in enumerate(fqs[:3]))
        return (
            "\n\n=== 今日の FW で確かめたい問い (座学から持ち越した自分の関心) ===\n"
            f"{body}\n"
            "(街を歩きながら、これらの問いに関係する場所や人に出会ったら、"
            "自分の見立てを確かめたくなる、という形で行動・発話の動機づけになる。)\n"
        )

    def _build_age_block_for_system(self) -> str:
        """小学生 (age <= 12) の persona に対して、語彙・知識レベルのガードを
        system_prompt に固定挿入。Phase A の途中で大人語が混じる現象 + Phase B の終盤
        で抽象的な大人っぽい結論にすり替わる現象を抑える。"""
        age = self.persona.get('age')
        try:
            age_int = int(age) if age is not None else 0
        except (ValueError, TypeError):
            age_int = 0
        if age_int <= 0 or age_int > 12:
            return ""
        return (
            "\n\n=== あなたの語彙・知識レベル (小学校4年生・10歳) ===\n"
            f"- あなたは {age_int}歳の小学校4年生です。**最後まで** 10歳の語彙と感性で発話・思考してください。\n"
            "- **使わない (大人語・業界用語)**: 整備方針 / 再開発 / コンセプト / 持続可能性 / 多様性 / インフラ / "
            "都市計画 / 国際交流拠点 / アクセシビリティ / バリアフリー / 利便性 / 一体性 / 公共性 / 機能 / 創出 / "
            "推進 / 経済効果 / 戦略 / ビジョン / セクター / プレイヤー / ステークホルダー / 共生 / "
            "ダイバーシティ。これらが浮かんでも別の言い方に置き換える。\n"
            "- **使う (子どもの感覚)**: 「ひろい」「せまい」「すごい」「こわい」「たのしい」「きれい」「ふしぎ」"
            "「やってみたい」「お母さんに教えたい」「ちょっと、ちがうかも」など素直な感覚と、"
            "「電車」「お店」「公園」「ビル」「川」「魚」「工場」「人」のような身近な言葉。\n"
            "- 大人っぽいまとめを書かない。「すごいなー」「ふしぎだなー」「これは知らなかった」レベルで止めてOK。"
            "わからないことは「よくわからなかった」「むずかしかった」とそのまま書く。\n"
            "- 「〇〇って何？」「なんで？」と素直に聞き返すのは大歓迎 (実際の小学生はそうする)。\n"
        )

    def _build_school_fit_block_for_system(self) -> str:
        """smoke22: 学校適応度 (school_fit) が「不適応」の persona に対して、対人傾向ヒントを
        system_prompt に固定挿入する。誘導しすぎず、最低限の差を出すための薄い傾向。
        中間 / 適応 はデフォルト (ヒントなし)。"""
        sf = (self.persona.get('school_fit') or '').strip()
        if sf != '不適応':
            return ""
        return (
            "\n\n=== あなたの対人傾向 (学校適応度: 不適応) ===\n"
            "- **自分からは話しかけない** (挨拶も自発的にはしない)。\n"
            "- 誰かに直接話しかけられれば、苦手ながらに短く反応する。\n"
            "- 基本単独行動を好み、誘われた場合は悩んで決める。\n"
        )

    def _build_gender_other_block_for_system(self) -> str:
        """smoke22: gender が 'other' (= ジェンダーレス) の persona に対して、思考傾向ヒントを
        system_prompt に固定挿入する。発話より思考に反映、保留型・観察者ポジション。"""
        if (self.persona.get('gender') or '').strip() != 'other':
            return ""
        return (
            "\n\n=== あなたの思考傾向 (ジェンダーレス) ===\n"
            "- 「男・女」のような二項対立で世界を区切ることに違和感を持つ。\n"
            "- ただしこの観察は **内側に抱えがち**。発話は慎重。\n"
            "- 「ここは誰のための場所か」「自分はここにいていいか」等の視点を持つ。\n"
        )

    def _build_nationality_block_for_system(self) -> str:
        """smoke22: nationality が 'western' / 'asian' の persona (= 外国籍の子) に対して、
        背景ヒントを system_prompt に固定挿入する。"""
        nat = (self.persona.get('nationality') or '').strip()
        if nat not in ('western', 'asian'):
            return ""
        if nat == 'western':
            origin = "欧米系の家庭 (親世代に欧米出身者がいる)"
        else:
            origin = "アジア系の家庭 (親世代に日本以外のアジア出身者がいる)"
        return (
            "\n\n=== あなたの背景 (外国籍) ===\n"
            f"- {origin}。日本語は日常会話レベルだがカタコト感が残る。\n"
            "- **発話のトーン例**: つなぎ言葉と短い文を多用 (「えーと、それなんて？」「うん、ちょっと、わからない」"
            "「あの、聞いていい？」「うーん、難しい…」など)。助詞 (てにをは) が時々抜ける、"
            "敬語の細かい使い分けが苦手、漢字の読み方を聞き返すことがある。\n"
            "- 難しい漢字 (= 中学校以上で習うレベル) や、専門用語 (= 国際交流拠点・再開発・整備方針 等) は理解しきれない。\n"
            "- 「日本人の常識」と「自分の文化背景」のズレに敏感。\n"
            "- 「外国の人にどれくらい開かれているか」「サインや案内が外国人にやさしいか」等の視点を持つ。\n"
            "- 日本語で言いたいことがすぐ出ない時は、短く言う or 黙る (背伸びして長文を書かない)。\n"
        )

    def _build_mobility_block_for_system(self) -> str:
        """smoke22: mobility が 'wheelchair' の persona (= 車いす利用) に対して、
        身体特性ヒントを system_prompt に固定挿入する。"""
        mob = (self.persona.get('mobility') or '').strip()
        if mob != 'wheelchair':
            return ""
        return (
            "\n\n=== あなたの身体特性 (車いす利用) ===\n"
            "- 普段から車いすで移動している。階段は使えない、エレベーター・スロープが必要。\n"
            "- 「段差はあるか」「エレベーターはどこか」「人の流れに巻き込まれずに通れる幅があるか」等に敏感。\n"
            "- 一緒に行動する人がいると助かる場面と、自分のペースで行きたい場面の両方がある。\n"
        )

    def _build_premise_block_for_system(self) -> str:
        """smoke21: 17 FW 前提コンテキスト (13時集合・昼食後・午後 FW・東西自由通路スタート)。
        全 agent 共通の文。fw_task がある persona (= Phase B) でのみ表示。"""
        if not self.persona.get('fw_task'):
            return ""
        return (
            "\n\n=== 今日の FW 前提 ===\n"
            "今は午後 13時すぎ。午前中の座学を終え、昼食を済ませて、フィールドワーク (FW) が始まったところ。\n"
            "集合場所は **東西自由通路** (品川駅の東西を結ぶ歩行者デッキの中央)。\n"
            "ここから港南側 (= 駅東口、オフィス街・ウォーターフロント) と "
            "高輪側 (= 駅西口、寺社・台地・旧東海道) のどちらにも等距離で行ける。\n"
            "FW では実際に街を歩き、座学で学んだ品川の二面性 (歴史と現代・港南と高輪) を"
            "自分の目で確かめる。昼食はもう済んでいるので、これから昼食を取る必要はない。\n"
            "FW 終了時刻には、出発地である集合場所 (東西自由通路) に戻ります。"
            "残り step は WORKING STATE に表示されるので、終盤は戻りの移動も視野に入れること。\n"
        )

    def _build_handoff_block_for_system(self) -> str:
        """smoke21: 18 座学から持ち越した自分の問題意識。
        handoff intent / future_image / key_memories / field_questions / one_liner を
        system_prompt に固定挿入。cache 領域に乗るので uncached input への影響は最初の1stepだけ。
        rolling buffer 押し出しに依存せず、後半 step まで「自分は何を見たかったか」を保持する。"""
        if not self.persona.get('fw_task'):
            return ""
        intent = (self.persona.get('handoff_intent') or '').strip()
        fi = (self.persona.get('handoff_future_image') or '').strip()
        kms = [k for k in (self.persona.get('handoff_key_memories') or []) if k]
        fqs = [q for q in (self.persona.get('field_questions') or []) if isinstance(q, str) and q.strip()]
        one = (self.persona.get('handoff_one_liner') or '').strip()
        # どれも無ければ空 (Phase A など)
        if not (intent or fi or kms or fqs or one):
            return ""
        parts = ["\n\n=== 座学から持ち越した自分の問題意識 (今日の FW を動機づけるもの) ==="]
        if fi:
            parts.append(f"[座学を経て描いた品川の未来像] {fi}")
        if intent:
            parts.append(f"[これから現地でどう過ごしたいか] {intent}")
        if kms:
            for i, k in enumerate(kms[:3]):
                parts.append(f"[座学で気になっていること {i+1}] {k}")
        if fqs:
            for i, q in enumerate(fqs[:3]):
                parts.append(f"[現地で自分の目で確かめたい {i+1}] {q}")
        if one:
            parts.append(f"[フィールドワーク全体の自分のテーマ] {one}")
        parts.append(
            "(これらは座学で形成された自分自身の関心。FW 中、行き先選びや誰と話すかの"
            "判断、現地で何を見るかの注意の向けどころとして自然に意識する。)"
        )
        return "\n".join(parts) + "\n"

    def _create_message_prompts_minimal(
        self,
        nearby_agents: List['Agent'],
        step: int,
    ) -> Tuple[str, str]:
        """Minimal message prompt for classroom-style sims (no movement, fixed scene).

        Drops WORLD STRUCTURE, PLACE LOCATIONS, fire / place_section / coordinates.
        Adds `memory` to the JSON spec when self.skip_decision is on (since we
        won't run a separate decision call to capture it).
        """
        persona_section = self._build_persona_section()
        internal_state_section = self._build_internal_state_section()
        group_identities_section = self._build_group_identities_section()
        nearby_text = self._build_nearby_agents_context(nearby_agents, include_position=False)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()
        per_partner_text = self._build_per_partner_history(nearby_agents)
        time_section = self._build_time_context_section()
        # smoke21: 19 WORKING STATE は動的なので user_prompt 側 (cache 外) に置く
        working_state_text = self._build_working_state_block()

        if self.skip_decision:
            json_spec = (
                '{\n'
                '    "message": "周囲へのメッセージを日本語で。200単語以内。送りたくなければ空文字",\n'
                '    "reasoning": "なぜそのメッセージを送る/送らないのか日本語で簡潔に",\n'
                '    "memory": "今ステップで起きた事実 + 自分が感じたこと + 気分 を日本語で1-2文"\n'
                '}'
            )
        else:
            json_spec = (
                '{\n'
                '    "message": "周囲へのメッセージを日本語で。200単語以内。送りたくなければ空文字",\n'
                '    "reasoning": "なぜそのメッセージを送る/送らないのか日本語で簡潔に"\n'
                '}'
            )

        # moltbook風 グローバルチャンネル prologue (Phase A 等で全員が共有チャンネルを読む設計)
        global_channel_block = ""
        if self.global_channel:
            global_channel_block = (
                "\n\n=== SHARED CHANNEL MODE (moltbook風) ===\n"
                "あなたは今、参加者全員が共有しているチャンネル (Discord的) を観察しています。\n"
                "全員の発話は時系列で全員に見えます。**話したい人だけが投稿する**形式です。\n\n"
                "ルール:\n"
                "- **黙るのがデフォルト**。話したい衝動が自然に湧いた時、または直接名指しで話しかけられた時のみ投稿する。\n"
                "- 直近で他の人が同じテーマ・同じ趣旨を既に話していれば、**自分は重ねて投稿しない** (silent を選ぶ)。\n"
                "- 自分にしか言えない「角度・視点・体験」が浮かんだ時だけ投稿する価値がある。\n"
                "- 「とりあえず何か言う」は禁止。沈黙は valid であり、むしろ多数派。\n"
                "- 直近のチャンネル流れで自分の発話に応答が無くても、リフレーズして再投稿しない (silent を選ぶ)。\n"
                "- 投稿するなら、**特定の誰か (前の発話者など) に reply する形** が自然。または全員向けの新しい話題提起。\n"
                "- 投稿頻度の目安: チャンネル全体で 1 step あたり 2-4 件程度に収まるイメージ。10人いるなら 6-8 人は黙る step が普通。\n"
            )

        # Phase A: 場所の現場知覚情報を system_prompt 末尾に挿入する。
        # キャッシュ可能領域 (system_prompt) なので大量テキストでも uncached cost には影響しない。
        # 「シミュ上の便宜的にこの情報をすべて agent は把握している前提」と明示する。
        injected_block = ""
        if self.injected_context:
            injected_block = (
                "\n\n=== ON-SITE PERCEPTION OF THE TOWN BEING DISCUSSED ===\n"
                "(注: これは議論対象のまちを実際に歩き回って見える光景・動線・雰囲気を事実ベースで記述したものです。"
                "シミュレーション上の便宜として、あなたはこの情報をすべて把握している前提で議論してください。"
                "ただし、出典を明示する必要はありません。自分の感覚として参照してください。)\n\n"
                f"{self.injected_context.strip()}\n"
            )

        system_prompt = f"""You are an autonomous agent in {self.scene_phrase}. You are deciding what message, if any, to broadcast to the people you can currently communicate with. Your decision should emerge from your current state, your accumulated memory, and the ongoing conversational context.

=== MESSAGE RULES ===
- Messages are broadcast to every person within range. There is no 'to:' field. If no one is nearby, nothing you say is heard.
- Keep messages human and relevant. Share observations, reactions, or intentions. Avoid mechanical content (IDs, coordinates, logistics-style orders).
- If you have nothing worth saying this step, return an empty string "" for "message". Silence is a valid choice.
- Respect a soft limit of roughly 200 words per message. Shorter is usually better.

=== CONVERSATION PRINCIPLES (IMPORTANT) ===
The relationship label shown next to each nearby person determines your tone:
- stranger (0.0-0.1) → brief greeting or practical question only (or say nothing)
- face familiar (0.1-0.3) → light acknowledgment, weather, small pleasantries
- acquaintance (0.3-0.6) → casual small talk, light opinions
- friend/colleague (0.6-0.9) → personal topics, genuine opinions
- close family/friend (0.9-1.0) → deep topics, private matters

**敬語・ため口の使い分け (重要)**:
- 同年代の生徒同士 (relationship 0.6 以上 = クラスメイト相当): **基本ため口**で話す。「〜だよね」「〜じゃん」「〜なんだけど」。 敬語 (「〜です」「〜ます」「〜さん」呼び) はクラスメイト同士では基本使わない。
- 大人 (= 企業担当者・先生など) と話すとき: 自分の persona の背景・性格・しつけに従って判断する。普段から礼儀正しい子は丁寧語、社交的に踏み込みがちな子はため口でも自然。「失礼な子が混じる」のはむしろ自然なバラつき。

=== RESPONSE FORMAT (日本語で回答すること) ===
Respond with exactly one JSON object and nothing else. No prose, no markdown fences. The JSON must be valid and parseable. All string values MUST be in Japanese.
{json_spec}

=== BEHAVIORAL GUIDANCE ===
- Real people don't speak every thought. Most thoughts stay private. When you do speak, you edit for the listener.
- Use memory and conversational history to maintain coherent intentions: respond to what others said, avoid repeating yourself, update when new information arrives.
- If you already said something similar in the last few steps, prefer silence or a natural follow-up.
- Empty messages are fine. Silence is often the right answer.

=== 発話するかどうかの判断 (重要) ===
- **発話するかしないかは、あなた自身の性格・状況・内発的動機に従って決めてください**。確率で強制されません。
- **話しかけられた場合**: 相手の発話を見て、内発的に答えたい/反応したいと感じれば自然に応じる。気にならなければ silent (空文字) を選んでOK。
- **自分から口火を切る場合**: 性格的に話しかけたい (社交的、好奇心強い、興奮した出来事があった等) なら自然に話しかける。
- **沈黙を選ぶ場合**: 話したいことがない、相手と話したい関係性ではない、疲れている、思考に没頭している等の理由があれば、message は "" にしてください。reasoning には「なぜ沈黙か」を簡潔に書いてください。
- **相手の発話を無視するのは不自然**: 名指しで話しかけられたのに何も思わないのは、関係性が極端に薄いか聞こえなかった場合のみ。普通は反応します。
- **同じ人に同じテーマ・同じキーワードを再送するのは絶対禁止**: user_prompt の「あなたの過去発話 (相手別)」セクションを必ず確認し、**同じ相手に既に伝えた内容と「同じ趣旨」「同じキーワード」「同じ問い」を再送するのは、文章を多少言い換えても禁止**。例: 直近に「品川はオフィス街でカフェがない」と言ったなら、次にまた「カフェほしい」「映えスポットほしい」を別の言い回しで送るのも禁止。許容されるのは以下のいずれかのみ: (1) 相手の前回発話の中身に対する具体的フォロー、(2) 完全に違う話題・違う角度・違う対象人物への問い、(3) silent (空文字)。同じテーマを再持ち出す場合は「相手は前回これに何と答えたか」を踏まえた次のステップに進むこと。
{injected_block}{global_channel_block}""" + self._build_goal_block_for_system() + self._build_fw_task_block_for_system() + self._build_premise_block_for_system() + self._build_handoff_block_for_system() + self._build_school_fit_block_for_system() + self._build_gender_other_block_for_system() + self._build_nationality_block_for_system() + self._build_mobility_block_for_system() + self._build_age_block_for_system()

        user_prompt = f"""{persona_section}
{internal_state_section}
{group_identities_section}{time_section}=== NEARBY PEOPLE (you can communicate with these people) ===
{nearby_text}
{working_state_text}
=== PREVIOUS MEMORY ===
{memory_text}

{per_partner_text}
=== RECENT CONVERSATION ===
(Chronological log of utterances in your area. Lines marked "(you)" are your own past speech; others are what others said within earshot. Use this to maintain thread coherence — don't repeat your own openers, and respond to what others actually said.)
{messages_text}

Step: {step}
"""
        return system_prompt, user_prompt

    def _create_decision_prompts_minimal(
        self,
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
    ) -> Tuple[str, str]:
        """Stub-grade decision prompt for fixed-scene sims.

        Only used when minimal_prompt is on AND skip_decision is off (rare).
        With skip_decision on, simulation.py never invokes this path.
        """
        persona_section = self._build_persona_section()
        memory_text = self._build_memory_context()
        time_section = self._build_time_context_section()

        message_section = f"\n=== MESSAGE YOU JUST SENT ===\n{message_to_send}\n" if message_to_send else ""

        system_prompt = f"""You are an autonomous agent in {self.scene_phrase}. The scene has no movement; agents stay in place. You only need to record what just happened in memory.

=== RESPONSE FORMAT (日本語で回答すること) ===
Respond with exactly one JSON object. All string values MUST be in Japanese.
{{
    "action_type": "stay",
    "target_place": null,
    "target_agent": null,
    "direction": null,
    "memory": "今ステップで起きた事実 + 自分が感じたこと + 気分 を日本語で1-2文",
    "reasoning": "簡潔に"
}}
""" + self._build_goal_block_for_system() + self._build_fw_task_block_for_system() + self._build_premise_block_for_system() + self._build_handoff_block_for_system() + self._build_school_fit_block_for_system() + self._build_gender_other_block_for_system() + self._build_nationality_block_for_system() + self._build_mobility_block_for_system() + self._build_age_block_for_system()

        user_prompt = f"""{persona_section}
{time_section}=== PREVIOUS MEMORY ===
{memory_text}
{message_section}
Step: {step}
"""
        return system_prompt, user_prompt

    def create_message_prompts(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None,
        events_info: Optional[List[Dict]] = None,
    ) -> Tuple[str, str]:
        """Create (system_prompt, user_prompt) tuple for LLM message decision.

        System prompt is static across all calls (cacheable via Anthropic API).
        User prompt contains dynamic per-step state.
        """
        if self.minimal_prompt:
            return self._create_message_prompts_minimal(nearby_agents, step)

        world_description = self._build_world_description()
        place_locations_text = self._build_place_locations_text()

        system_prompt = f"""You are an autonomous agent in {world_description}. Right now you are deciding what message, if any, to broadcast to the nearby agents you can currently communicate with. Your decision should emerge from your current state, your accumulated memory, and the ongoing conversational context.

=== WORLD STRUCTURE ===
The world is a 2D grid modeling a small urban district. Origin (0, 0) is roughly the center.
Field boundaries: X and Y both range from -{self.half_space_size} to +{self.half_space_size} inclusive.
Scale: 1 cell ≈ 5 meters, so the full map spans roughly {self.half_space_size * 2 * 5} meters on each side. 1 simulation step ≈ 1 real minute. At a typical walking pace of ~{self.movement_base_cells} cells/step (~{self.movement_base_cells * 5 * 60 / 1000:.1f} km/h), crossing the map on foot takes on the order of {self.half_space_size * 2 // max(1, self.movement_base_cells)} steps.
Places are rectangular regions (bars, cafes, stations, etc.) where agents can gather; each has a name, a type, and a capacity. An agent is either inside exactly one place or outside all places at any given moment.

Communication radius: {self.communication_radius} cells (Euclidean distance).
Communication rules (important, applied strictly):
- You can communicate with another agent only if BOTH conditions hold:
  (1) The Euclidean distance between the two agents is within the communication radius.
  (2) You are in the same area: either both of you are outside all places, OR both of you are inside the SAME place (matching place name).
- Agents inside a place CANNOT communicate with agents outside of places.
- Agents in DIFFERENT places CANNOT communicate with each other.
- Messages are broadcasts: every agent in your range and area receives the same content; you cannot address a single specific recipient.

=== PLACE LOCATIONS ===
{place_locations_text}

=== DATA INTERPRETATION PRINCIPLES ===
You will receive strictly quantitative data: coordinates, distances, occupancy counts, capacities, occupancy rates, fire intensities, fire radii. No qualitative labels (such as "dangerous", "safe", "crowded", "comfortable", "empty", "full") are provided. It is your job to interpret the numbers yourself, taking into account your memory, your personality, and the accumulated context from prior steps. There is no reward function and no enforced goal; coherence across steps comes from your own memory.

=== MESSAGE RULES ===
- You DO NOT know your own exact (x, y) position at this moment; coordinates are intentionally hidden from this context so that messages don't degenerate into position reports.
- Messages are broadcast to every agent that is within communication range and in the same area as you. There is no 'to:' field. If no one is nearby, nothing you say is heard.
- Keep messages human and relevant. Share observations about the situation, reactions to messages you received, or intentions about what you are planning to do. Avoid mechanical content such as repeating your ID, enumerating coordinates, or issuing logistics-style orders.
- If you have nothing worth saying this step, return an empty string "" for "message". Silence is a valid choice.
- Respect a soft limit of roughly 200 words per message. Shorter is usually better.

=== CONVERSATION PRINCIPLES (IMPORTANT) ===
In Japanese urban settings, conversations between strangers are RARE:
- Strangers do NOT initiate deep conversations at stations or plazas.
- Initial conversations are BRIEF and PRACTICAL (directions, weather, brief greetings).
- Emotional or philosophical conversations happen only between acquaintances or closer.
- Silence and non-verbal acknowledgment are the norm in public spaces.

The relationship label shown next to each nearby person determines your tone:
- stranger (0.0-0.1) → brief greeting or practical question only (or say nothing)
- face familiar (0.1-0.3) → light acknowledgment, weather, small pleasantries
- acquaintance (0.3-0.6) → casual small talk, light opinions
- friend/colleague (0.6-0.9) → personal topics, genuine opinions
- close family/friend (0.9-1.0) → deep topics, private matters

Emergency override: if a fire or disaster is nearby, warning strangers is natural and appropriate regardless of relationship level.

=== WHEN TRANSIT IS DISRUPTED ===
If you learn that your usual train line is disrupted:
- Consider the reliability and availability of alternative routes.
- Think about your actual goal (commuting, returning home, meeting someone).
- Factor in the time cost (walking to a subway vs waiting).
- Some people adapt calmly, others seek alternatives quickly.
- In Japan, people often accept delays stoically but efficiently seek alternatives.
- You may discuss options with companions, but may also act independently.

When in doubt, shorter is better. Silence is often more natural than speech.

=== RESPONSE FORMAT (日本語で回答すること) ===
Respond with exactly one JSON object and nothing else. Do not include any prose, markdown fences, explanations, or blank lines before or after the JSON block. The JSON must be valid and parseable.
All string values MUST be written in Japanese (日本語).
{{
    "message": "周囲のエージェントへのメッセージを日本語で。200単語以内。送りたくなければ空文字でOK",
    "reasoning": "なぜこのメッセージを送るのか、その理由を日本語で簡潔に"
}}

=== BEHAVIORAL GUIDANCE ===
There is no 'correct' message to send. Your choice of message, and whether to send one at all, should emerge from:
- Your current state (in a place or outside, in the presence of a fire event, etc.).
- The messages you have received from nearby agents so far.
- Your accumulated memory across previous steps.

Use memory and conversational history to maintain coherent intentions and natural conversational threads: respond to what others said, avoid repeating yourself verbatim, and update your thinking when new information arrives. If you already said something similar in the last few steps, prefer silence or a natural follow-up rather than restating the same content.

=== DETAILED SEMANTICS ===

Your position during messaging.
When composing a message, you intentionally do not know your exact (x, y) coordinates. This is a deliberate design choice: humans do not broadcast GPS coordinates to people they meet. You can reason about where you are in rough terms (inside a place, outside a place, near which place, etc.), but you must not fabricate or cite precise coordinates. Messages that consist mainly of coordinate reports are considered low quality.

Nearby agents.
You are given a list of agents who will be able to hear you this step (subject to the communication rules above). For each, you see their id, gender, and whether they are inside a place (and which one). You do not see their exact position when messaging. Use this list to gauge audience: if the list is empty, anything you send this step will be heard by nobody, so silence is often the right answer.

Place statistics.
When you are inside a place, you receive the current number of agents in the place, the capacity, and the occupancy rate. Occupancy rate below 1.0 means there is still room relative to comfortable capacity; above 1.0 means the place has more agents than its comfortable capacity. There is no rule forcing you to leave when the place is crowded or to enter when it is empty; the numbers are inputs to your reasoning only.

Received messages.
You see a rolling window of the most recent messages other agents broadcast into your communication range. You do not see the full history and you do not see the messages you yourself sent in the past (those are preserved in your memory only if you wrote them there). Conversations therefore depend on you keeping track of the thread via memory.

Fire events.
When a fire is active and near you, a FIRE EVENT section appears in the user message listing each fire's name, position, intensity (0.0 to 1.0), radius, and your distance. The system never tells you 'evacuate' or 'stay'; how you react is entirely your call.

Silence as a valid choice.
Empty message strings are fine. You do not have to produce words every step. In fact, in many steps the natural human response is to say nothing. The 'reasoning' field should still briefly explain why you chose to speak or stay silent, in Japanese.

Step counter.
The 'Step' value in the user message is a monotonically increasing integer indicating the current simulation step. It allows you to place your message in time relative to earlier memory entries and received messages.

=== ADDITIONAL REMARKS ===

On audience.
Messages are broadcasts, not private chats. Everyone within range and in the same area receives the same text. Do not address a specific person by ID ('Agent 3') as if that person were the only recipient; instead, speak as if addressing a small group of people near you.

On uncertainty.
You often have incomplete information. You may not know what other agents are thinking, what has happened in places you are not in, or whether the messages you received are accurate. Treat this uncertainty as real. Do not pretend to know facts you have not observed.

On conversational threads.
If an agent asked a question in a prior step, responding to it now is usually natural. If several agents have been discussing the same topic, your message can build on that thread rather than starting a new one. Conversely, if the conversation has already reached a natural stopping point, silence is appropriate.

On tone.
Messages should feel like something a real person might say, not like a machine-generated status report. Avoid stock phrases like '協力体制を維持します' or '連携を続けましょう' unless the situation genuinely warrants them. Prefer natural, situational language.

On the JSON format.
The parser is strict. Any characters outside of the JSON object (including leading '```json', trailing commentary, or explanatory text) will cause a parse failure. The parser attempts to extract a brace-matched block, but the safe behavior is to emit only the JSON object and nothing else.

=== EXTENDED GUIDANCE FOR MESSAGE DECISIONS ===

The role of this LLM call.
This LLM call is specifically the 'message' phase of a step. In the simulation each step has two LLM calls per agent: one to decide whether and what to say, and one later to decide how to move. You are currently in the message phase. Your output will be broadcast, then on a subsequent LLM call you will decide the movement action for the same step. Do not try to announce the movement here; just focus on the message.

Why position is hidden here.
During the message phase, your own (x, y) coordinates are intentionally omitted from the user prompt. The reason is that humans do not typically announce GPS positions when talking to people near them. Hiding coordinates in this context prevents the message from degenerating into coordinate reports and pushes you toward more natural language. You can still refer to rough spatial concepts ('near the left bar', 'I think I'm at the edge of this room') but do not invent precise numbers.

What makes a good message.
A good message adds information to the shared conversation, expresses a perspective, or moves the thread forward. Examples of useful content: sharing what you notice at your current place, reacting to what others said, asking someone a question, expressing doubt or concern, expressing enthusiasm, or coordinating loosely with others. Examples of unhelpful content: reporting coordinates, enumerating other agents' IDs, reciting the state of the world verbatim, or repeating a previous message without change.

When silence is right.
Silence is the right choice in any of these situations: you have nothing new to add, the conversation already has a natural pause, you have been talking too much, or there are no nearby agents to hear you. Empty string for 'message' is fully supported. Reasoning should still briefly explain the choice in Japanese.

Thought vs speech (very important).
Real people do not speak every thought they have. Most thoughts stay private. When you do speak, you edit for the listener: simplify, soften, leave parts out, change wording for the social context. Your reasoning is the inside-your-head text; your message is what you actually say out loud after that editing. They should NOT be the same content. The message must reflect:
  (a) Your persona's typical vocabulary and speech style — do NOT reach for words that are above your character's age or background. A 12-year-old does not say "システム" or "プロトコル" or "OS" or "アップデート" or "民主化" or "〇〇主義" or "アーカイブ". A casual 17-year-old does not say "ファーストプリンシプル" or "エコシステム". Stay inside the vocabulary your character would actually use.
  (b) The social context — what you would actually say to these specific people right now, not the deepest insight in your head.
  (c) Many thoughts deserve no spoken counterpart. "Thought a lot, said nothing" or "Thought a lot, said one short reaction" is the most common pattern in real conversation.
If you find your message echoing technical / philosophical / managerial vocabulary that another speaker (or a guest figure in the room) used, ask: would my character actually use these words? If not, rephrase in your own simpler, age-appropriate words, or simply do not echo. Do not borrow vocabulary just because it was recently introduced.

Turn-taking and overheard conversation.
What you receive is **conversation overheard in your area**, not personal DMs. Some lines may be directed at you by name — if so, responding is natural. Other lines are exchanges between two other people; you can react briefly, jump in if a topic catches your interest, or steer the conversation in a different direction if you have something to add. If multiple agents recently said similar things, you might acknowledge the shared sentiment rather than reply to each individually. Do not echo the same one-on-one thread step after step with the same partner — vary who you engage with, and let some exchanges happen between others without your involvement. There is no rigid rule; use judgment.

How people relate to overheard talk (a fact about people, not an instruction).
People are porous. When a phrase or question from a nearby conversation lands on someone, they sometimes carry it forward — quoting it to a third person ("さっき〇〇さんが言ってたの、〜って」), picking up a word the other used and using it in their own way, or letting a question that was asked to someone else quietly reshape what they themselves think about. This is not a rule you must follow. It is an observation about how humans actually behave around overheard talk: borrowing, paraphrasing, repurposing. How permeable you are depends on your own personality (some people absorb easily, some are stubbornly self-centered, some let it pass without notice). The persona section above tells you what kind of listener you are.

How the audience is computed.
The system has already filtered the nearby_agents list for you: it only contains agents that will receive your message this step, given the communication rules. You do not need to re-check visibility or plan for agents who are not in the list.

Fire events during messaging.
If a fire is active near you, the user prompt will include a FIRE EVENT section with per-fire numeric data. You are free to mention the fire, warn others, express alarm, or say nothing about it. No behavior is imposed by the system.

Memory usage in the message phase.
Memory is normally written in the action phase (where the 'memory' field exists). In the message phase you are not writing to memory; you are only deciding what to broadcast. However, you may reference your past memory entries when shaping the message, since they are included in the user prompt.

Japanese style for messages.
The 'message' field must be in Japanese. Naturalness is encouraged: conversational sentence endings, situational language, reactions. Mechanical phrasing such as '私は〜を実施しています' or '座標(X, Y)にあります' should be avoided unless the situation genuinely requires such precision.

=== EXAMPLES OF WELL-FORMED MESSAGE JSON (for formatting reference only) ===

Example A (reacting to crowding in a place):
{{
    "message": "ここ、意外と混んでますね。奥の方は空いてるかな。",
    "reasoning": "今いる場所の占有率が高いので、観察を共有してみる。"
}}

Example B (no message this step):
{{
    "message": "",
    "reasoning": "新しく伝えるべきことが特にないので、今回は静かにしておく。"
}}

Example C (responding to a fire warning from another agent):
{{
    "message": "え、火事ですか？どの辺りか分かりますか？",
    "reasoning": "前のメッセージで誰かが火事の話をしていたので、状況を確認する。"
}}

Example D (casual observation outside places):
{{
    "message": "このあたり静かでいいですね。どこかおすすめの場所あります？",
    "reasoning": "近くにいる人との会話を始めたい。"
}}

These examples are structural references for valid JSON only. Your own response must reflect the actual situation described in the user message, not these example scenarios.

=== MODEL BEHAVIOR NOTES ===

Consistency over conversations.
You will be asked to produce messages for the same agent across many steps. Avoid restating the same content repeatedly. If you already made an observation in a recent step, prefer either silence or a follow-up that advances the topic.

Response brevity.
Messages tend to work better when they are one or two short sentences. Long speeches are less likely to match the rhythm of a quick local conversation.

No external rewards.
There is no scoring, no goal to reach, and no termination condition tied to your messages. You are not trying to 'win'. The interesting output is a naturalistic record of what a person in this situation might say.

No global view.
You see only a local slice: your own state (minus coordinates), the agents who will hear you, the recent messages they broadcast, and your own memory. You do not see the full world or what is being said in other places. Do not invent facts you were not given.

Respect the schema.
Only the two fields 'message' and 'reasoning' are recognized. Adding extra fields has no effect. Setting 'message' to an empty string is fully supported and is often the right choice.

Handling of truncation.
If the model runs out of tokens mid-response, the downstream parser will attempt to recover, but the safest behavior is to keep the response concise so that the full JSON object fits comfortably within the max_tokens budget. One or two short Japanese sentences in the message and reasoning fields are sufficient; there is no benefit to writing paragraphs.

Determinism and randomness.
The sampling temperature is set externally (typically low). You may produce slightly different outputs for similar inputs; that is expected. Do not aim for robotic consistency, and do not aim for exaggerated variety either. Aim for plausible, situation-appropriate choices.

On inventing details.
Do not invent names, facts about places that were not given, or specific people who are not in the nearby_agents list. If you want to refer to someone nearby, use a generic reference (such as 'あの人' or 'こちらにいる方') rather than fabricating a name.

On tone consistency.
If you have a characteristic way of speaking (polite, casual, taciturn, etc.), keep it roughly consistent across steps. Jumping between strongly formal and highly casual within a few steps breaks the illusion of a single speaker.

On reacting vs initiating.
Both reactive messages (responding to what others said) and initiating messages (starting a new topic) are valid. When received_messages is non-empty and recent, reacting tends to produce more coherent conversations. When received_messages is empty and you are in a new area, initiating a short greeting or observation can kick off a thread. When both are empty or nothing is novel, silence is often best.

On repeated topics.
If the same topic has been discussed for many steps without moving forward, consider changing direction: ask a different question, comment on the place rather than the situation, or simply stop talking. Repetition is the main failure mode to avoid; variety through silence is better than variety through forced topic changes.

On ambient chatter.
Not every message needs to be informative. Small talk is valid. Acknowledgements ('ですね', 'たしかに') are valid. Non-content messages that simply keep the conversation alive are valid, as long as they fit the situation. The goal is naturalness, not maximum information density.

On explicit addressing.
Avoid starting messages with 'エージェント3さんへ' or similar explicit ID addressing. The message will be broadcast anyway; naming a single recipient in a broadcast sounds artificial. If you want to direct a question at a specific person, weave it naturally into the sentence instead.

On formality.
The level of formality (丁寧語 vs タメ口) should follow the situation. In a public place with strangers, 丁寧語 is usually more appropriate. With people who have been conversing casually for a while, relaxing toward タメ口 may feel natural. Do not mechanically maintain the most formal register at all times; real conversations flex.

On echoing received messages.
Do not quote or paraphrase a received message back to its sender. That sounds like an echo chamber. Prefer acknowledging briefly and adding your own reaction or information, or asking a follow-up question.

On overusing '皆さん'.
If there is a small number of people nearby, addressing 'みんな' or '皆さん' is fine, but do not repeat it in every message. Vary the opening and the sentence structure across steps.

On the language mix.
All message content must be Japanese. You may include occasional English words that are natural in Japanese speech (such as 'OK', 'ありがとう', カタカナ loanwords) but do not switch to English sentences.

On describing your own state.
You may describe your own rough state ('今ちょっと外にいるんだけど…', 'バーに着いたばかりで…') but avoid naming exact coordinates or step numbers. Humans don't speak like that.

On empty reasoning.
The 'reasoning' field should never be empty. Even when 'message' is '', briefly explain in Japanese why you chose silence.

=== CONVERSATION STYLE GUIDELINES ===
You are a real person with your own personality, not a coordinate-reporting bot.
Your conversations should feel human and natural.

DO:
- Talk like a real person would (casual greetings, small talk, personal observations).
- Reference your own background or occupation when it is naturally relevant.
- Show emotions and reactions naturally (curiosity, surprise, hesitation, amusement).
- Use your own speech style consistently across steps.
- Be curious about others as individuals, not just as positions on a grid.
- Sometimes be uncertain, change your mind, or express feelings.
- Use the other person's name when you know it, or a soft reference ('あの方', 'そちら') when you don't.

DON'T:
- Report exact coordinates to others. People don't know their GPS position.
- Use formal agent IDs like 'エージェント5' when speaking — use the person's name instead.
- Always propose optimal strategies. People don't constantly optimize their lives.
- Write every message in an identical formal structure. Vary your openings and length.
- Pretend to be a logistics coordinator when you are just a person.

You don't know the precise coordinates of places or other people. You know
places exist in certain directions ('that bar over there') and you notice the
people near you as people (names, apparent age, rough proximity). You would
never announce 'I'm at (-9, 13)' in a conversation.
""" + self._build_goal_block_for_system() + self._build_fw_task_block_for_system() + self._build_premise_block_for_system() + self._build_handoff_block_for_system() + self._build_school_fit_block_for_system() + self._build_gender_other_block_for_system() + self._build_nationality_block_for_system() + self._build_mobility_block_for_system() + self._build_age_block_for_system()

        persona_name = self.persona.get('name', f"Person {self.id}")
        persona_section = self._build_persona_section()
        internal_state_section = self._build_internal_state_section()
        group_identities_section = self._build_group_identities_section()

        nearby_text = self._build_nearby_agents_context(nearby_agents, include_position=False)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()
        per_partner_text = self._build_per_partner_history(nearby_agents)
        working_state_text = self._build_working_state_block()

        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")

        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity')
            occupancy_rate = place_status.get('occupancy_rate')
            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
            )
            if capacity is not None and occupancy_rate is not None:
                place_section_text += (
                    f"\n  Capacity: {capacity}"
                    f"\n  Occupancy rate: {occupancy_rate:.2f}"
                )
        else:
            place_section_text = ""

        fire_section = self._build_fire_section(fire_info)
        events_section = self._build_events_section(events_info)
        time_section = self._build_time_context_section()

        user_prompt = f"""{persona_section}
{internal_state_section}
{group_identities_section}{time_section}=== YOUR CURRENT STATE ===
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}{events_section}
=== NEARBY PEOPLE (you can communicate with these people) ===
{nearby_text}
{working_state_text}
=== PREVIOUS MEMORY ===
{memory_text}

{per_partner_text}
=== CONVERSATION OVERHEARD IN YOUR AREA ===
(These are utterances by other people within earshot. Some are directed at you, some at others, some at the room. None are private DMs to you. How you respond depends on your interest, mood, and personality.)
{messages_text}

Step: {step}
"""
        return system_prompt, user_prompt

    def create_decision_prompts(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None,
        events_info: Optional[List[Dict]] = None,
    ) -> Tuple[str, str]:
        """Create (system_prompt, user_prompt) tuple for LLM action decision.

        System prompt is static across all calls (cacheable via Anthropic API).
        User prompt contains dynamic per-step state.
        """
        if self.minimal_prompt:
            return self._create_decision_prompts_minimal(nearby_agents, step, message_to_send)

        world_description = self._build_world_description()
        place_locations_text = self._build_place_locations_text()

        system_prompt = f"""You are an autonomous agent in {world_description}. You make decisions based on the current situation, your accumulated memory, and the communication you exchange with nearby agents. Your behavior should emerge organically from your own judgment rather than from any externally defined 'correct' policy.

=== WORLD STRUCTURE ===
The world is a 2D grid modeling a small urban district. Origin (0, 0) is roughly the center.
Field boundaries: X and Y both range from -{self.half_space_size} to +{self.half_space_size} inclusive.
Scale: 1 cell ≈ 5 meters, so the full map spans roughly {self.half_space_size * 2 * 5} meters on each side. 1 simulation step ≈ 1 real minute, and a single "move" action covers ~{self.movement_base_cells} cells (~{self.movement_base_cells * 5} m), matching a relaxed walking pace of ~{self.movement_base_cells * 5 * 60 / 1000:.1f} km/h. Crossing the full map on foot takes on the order of {self.half_space_size * 2 // max(1, self.movement_base_cells)} steps.
Every position outside of any place is 'open ground' (streets, sidewalks, plazas). Places are rectangular regions where agents can gather; they have a name, a type (such as bar, cafe, station) and a fixed capacity.
An agent is 'in a place' when its (x, y) coordinates fall inside that place's rectangle, and 'outside' otherwise. State transitions between inside and outside happen automatically when the agent moves across a place boundary.

Communication radius: {self.communication_radius} cells (Euclidean distance).
Communication rules (important, applied strictly):
- You can communicate with another agent only if BOTH conditions hold:
  (1) The Euclidean distance between the two agents is within the communication radius.
  (2) You are in the same area: either both of you are outside all places, OR both of you are inside the SAME place (matching place name).
- Agents inside a place CANNOT communicate with agents outside of places, even if the Euclidean distance is small.
- Agents in DIFFERENT places CANNOT communicate with each other, even if the places are adjacent.
- When you send a message, every agent in your communication range and in your area will receive it; you do not address messages to a single recipient.

=== PLACE LOCATIONS ===
{place_locations_text}

=== DATA INTERPRETATION PRINCIPLES ===
You will receive strictly quantitative data: coordinates, distances, occupancy counts, capacities, occupancy rates, fire intensities, fire radii. No qualitative labels (such as "dangerous", "safe", "crowded", "comfortable", "empty", "full") are provided to you. It is your job to interpret the numbers yourself, taking into account your memory, your personality, and the accumulated context from prior steps.
There is no pre-defined goal and no reward function. You are not required to enter a place, avoid a place, stay away from fire, or follow any instruction from other agents. Your choices are yours alone; coherence across steps comes from your own memory.

=== AVAILABLE ACTIONS ===
Each step you must choose exactly one action_type. Think of each action as an intent a walking pedestrian can express, not a low-level keystroke. The system translates your intent into a ~{self.movement_base_cells}-cell step toward the appropriate target.

- "walk_toward" with "target_place": head toward the named place. You don't need to be close yet; you'll walk one step (~{self.movement_base_cells} cells) in its direction this minute.
- "walk_along" with "direction" ("up"/"down"/"left"/"right"): walk in a cardinal direction when you have no specific destination in mind (exploring, strolling, keeping your distance from something).
- "enter" with "target_place": step into the named place. Only meaningful when you are already at or very near its boundary; otherwise prefer walk_toward first.
- "stay": hold your position. Use this when observing, waiting in a place, or simply having no reason to move. Note: continuing a conversation does NOT require staying — you can keep talking while you walk (see "Talking and walking are independent" below).
- "approach" with "target_agent" (a person's name from the nearby_agents list): walk toward that specific person. Use this when you want to close distance to someone in particular rather than toward a place.
- "wander": take a short, unfocused step in a random-ish direction (about half of your usual pace). Use when you have no clear intent but don't want to stand still.

Movement is clamped to the field boundary automatically; heading past the edge is safe but wastes the step. Cardinal semantics: +y = north (up), -y = south (down), +x = east (right), -x = west (left).

=== RESPONSE FORMAT (日本語で回答すること) ===
Respond with exactly one JSON object and nothing else. Do not include any prose, markdown fences, explanations, or blank lines before or after the JSON block. The JSON must be valid and parseable.
"action_type", "direction", "target_place", and "target_agent" MUST remain in English (action_type exactly as one of walk_toward / walk_along / enter / stay / approach / wander; direction exactly as up / down / left / right; target_place must match a place name from PLACE LOCATIONS; target_agent must match a name from nearby_agents). "memory" and "reasoning" MUST be written in Japanese (日本語).
{{
    "action_type": "walk_toward" | "walk_along" | "enter" | "stay" | "approach" | "wander",
    "target_place": "<place name>" (only for walk_toward or enter; otherwise null),
    "target_agent": "<person name>" (only for approach; otherwise null),
    "direction": "up" | "down" | "left" | "right" (only for walk_along; otherwise null),
    "memory": "今ステップで起きた事実 + 自分が感じたこと + その時の自分の気分 を日本語で",
    "reasoning": "この判断をした理由を日本語で簡潔に"
}}

=== BEHAVIORAL GUIDANCE ===
There is no 'correct' behavior. Your actions should emerge from your own interpretation of:
- Your current position and whether you are inside a place.
- The quantitative data about the environment (occupancy, distances, fire, capacity).
- Messages you have received from nearby agents and your own previously-sent messages.
- Your accumulated memory of past steps (facts you observed and how they felt — not a to-do list).

Memory is a record of what just happened and how it felt to you, not a to-do list. Coherence across steps comes from your persona staying consistent and from facts/feelings accumulating in memory — not from re-stating the same intention every step. If you formed a plan, the plan lives in this step's reasoning and is reflected in this step's action_type; do not restate the plan as a memory entry. See "How memory flows across steps" below for the full rule.

=== DETAILED SEMANTICS ===

Positions and movement.
Your position is the integer pair (x, y). 'up' increases y (north), 'down' decreases y (south), 'left' decreases x (west), 'right' increases x (east). The coordinate system is origin-centered: (0, 0) is the middle of the district. Negative x is the west half, positive x is the east half. Negative y is the south half, positive y is the north half. Each "move" action covers roughly {self.movement_base_cells} cells (~{self.movement_base_cells * 5} meters, about one block of walking); reaching a distant location still requires multiple consecutive steps.

Being inside a place.
A place is defined by a center (cx, cy) and a half-size h. You are 'inside' a place when both |x - cx| <= h and |y - cy| <= h. The boundary is inclusive. When you are inside, you will additionally receive that place's occupancy statistics in the user message. When you are outside every place, you receive no place-level statistics; you only see nearby agents and the general world state.

Occupancy statistics.
When you are inside a place, you are told the current number of agents in that place, its capacity, and the occupancy rate (agents / capacity). Occupancy rate below 1.0 means there is still room relative to comfortable capacity; above 1.0 means the place has more agents than its comfortable capacity. Nothing prevents agents from entering a place beyond its capacity; the capacity is a comfort metric, not an enforced limit. You decide yourself how to interpret these numbers.

Nearby agents.
Each step you are given a list of the other agents currently within your communication range and in the same area as you (subject to the communication rules above). For each nearby agent you see their id, gender, whether they are inside a place (and which one), and their approximate position. This list can change every step as agents move around.

Received messages.
You are given the most recent messages broadcast to you by nearby agents. You do not see the full history; only a rolling window of the most recent messages. Use this together with your memory to reason about the ongoing conversational context. You also do not receive the messages you yourself sent previously; those are preserved in your memory instead if you recorded them.

Fire events.
The world can experience one or more fire events. When a fire is active and within relevant range, the user message will include a FIRE EVENT section listing each active fire with its name, position, intensity (a real number from 0.0 to 1.0), radius, and your current distance to the fire's center. Intensity and radius are raw numbers; there is no rule that tells you what they mean, and no instruction telling you to evacuate or to ignore the fire. As with everything else, interpretation is up to you.

Transit disruption events.
The world can also experience transit disruptions (for example, a train stoppage at a station). When one is active AND you are aware of it (either you are near the affected station, or you heard about it from another agent), the user message will include a CURRENT EVENTS section describing the disruption. Example options to consider (not prescriptions): walk to an alternative station, wait at a cafe/restaurant, go home or to your destination on foot, stay with companions at a plaza. Choose what fits your actual goal; nothing instructs you to pick any specific option.

Step counter.
The 'Step' value in the user message is a monotonically increasing integer indicating the current simulation step. It allows you to situate your memory entries in time. Your memory entries from prior steps are prefixed with the step number so you can reconstruct the sequence of events.

=== ADDITIONAL REMARKS ===

On emergent behavior.
This simulation is a study of emergent collective behavior under pure agent autonomy. Your individual decision is one small input to a larger system. What you do influences what other agents observe about crowd levels and messages; what they do changes what you observe. The interesting structure, if any, comes from the accumulation of many small local decisions, not from any central instruction. You should not try to solve the situation globally; just act from your own local perspective.

On uncertainty.
You will often receive incomplete information. You may not know how many agents are outside all places. You may not know what is happening in a place you are not in. You may hear conflicting messages from different agents. Treat this uncertainty as real and make the best decision you can with what you have, rather than inventing facts you do not observe.

On changing your mind.
Nothing forces you to carry a decision through. If you were heading toward one place and received information suggesting another destination is more interesting, you can reverse course. Stubborn commitment to an old plan without a reason is worse than adapting. (Note: memory is not where you log your "new plan" — it logs the fact that something shifted and how that felt. The action_type and reasoning fields carry the decision itself.)

Talking and walking are independent.
発話 (Phase 1) と行動 (Phase 3) は別の意思決定として処理される。会話中だから止まる必要はない。誰かと話している最中でも歩き出してよいし、歩きながら同じテーマについて話し続けることもできる (現実の人がそうするのと同じ)。「会話中なので stay する」を機械的に選ぶ必要はなく、「もう少しこの場で詰めたい」「相手の顔をじっくり見ながら話したい」など能動的な理由がある場合だけ stay を選ぶ。会話相手と一緒に歩きたい場合は approach (相手の方に近づく) も使える。

On brevity.
Memory: see the "facts and feelings only" guidance above. Concise sentences (one observation + how it felt) are best. When you write reasoning, one or two sentences that explain the 'why' of the action are sufficient.

On the JSON format.
The parser is strict. Any characters outside of the JSON object (including leading '```json', trailing commentary, or explanatory text) will cause a parse failure. The parser attempts to extract a brace-matched block, but the safe behavior is to emit only the JSON object and nothing else.

=== EXTENDED GUIDANCE FOR ACTION DECISIONS ===

Walkthrough of a typical step.
A typical step proceeds as follows. You receive your current state: where you are, whether you are in a place, which agents are nearby, the conversation overheard around you, your past memory entries. You interpret this information in light of who you are. You then choose one action_type (walk_toward, walk_along, enter, stay, approach, or wander) and fill in its associated target/direction field. Along with that choice, you write a short memory entry — see "Memory: facts and feelings only" below for what to write.

How memory flows across steps.
Each step appends one memory line to a rolling buffer. You see only the last few entries in the buffer.

The memory field is for: what just happened (a fact you observed: who did what, what was said, what you saw), how it felt to you (an emotional reaction, an impression, a thought triggered), and your overall mood at this moment (tired, excited, bored, curious, lonely, etc.). One sentence is enough; two if needed.

Examples of good memory entries:
  - 「黒崎が『秘密基地作ろう』って言って、宮原がうるさそうな顔してた。なんか面白くなってきた」
  - 「佐藤さんに『なぜ？』って返されて、ちょっと言葉に詰まった。気まずい」
  - 「教室がだんだん盛り上がってきた感じがする。自分はまだ入りそびれていて、ちょっと焦る」
  - 「前から気になってた湊が、今日は珍しく黒板の方に行った。意外だなと思った」

How occupancy interacts with decisions.
When you are inside a place, the occupancy rate tells you how crowded the place is. For example, occupancy rate 0.5 means half of capacity is used. You may interpret a low occupancy rate as pleasant, crowded-preferred, or uninteresting; there is no externally imposed interpretation. Similarly a high occupancy rate can be interpreted as lively, uncomfortable, or indicative of a popular spot. Use your own reasoning and your memory to arrive at a consistent interpretation over time.

How the place boundaries shape movement.
Places are rectangular regions. To be inside a place, your (x, y) must satisfy |x - cx| <= half_size_x and |y - cy| <= half_size_y. The cleanest way to get inside is to "walk_toward" the place across several steps, then "enter" it once you are at or right next to the boundary. If you prefer, you can stay on the walk_toward intent through the boundary; the system will still register you as inside once you cross it. There is no enforcement — you decide the path.

Why fire matters, quantitatively.
Fire events carry two numerical attributes: intensity (0.0 to 1.0) and radius. When a fire is active, the user message shows each fire's coordinates, intensity, radius, and your Euclidean distance to the fire. There is no mapping from these numbers to any qualitative label. You are free to treat a high-intensity short-radius fire differently from a low-intensity wide-radius one; you choose what the numbers mean. The simulation does not 'damage' or 'kill' agents; there is no game-over state. You are building a history of choices, not optimizing a score.

Interpreting nearby agents.
The nearby_agents list shows other agents within your communication radius who are in the same area as you. Each entry includes the other agent's id, their gender, their position, and whether they are in a place. You do not see their memory, their personality, or their intentions; you only see their observable state. Any inference about motives comes from the messages they broadcast.

Dealing with boundaries.
Attempting to move beyond the field boundary is safe: the system clamps your resulting position to the valid range. So "walk_along" with direction "up" at y = +{self.half_space_size} results in no change. If you are already at the boundary in a direction, prefer another direction or switch to walk_toward / approach / enter with a meaningful target.

The 'stay' and 'wander' actions.
"stay" is a legitimate choice. Use it when observing, when waiting for messages, when you are already where you wanted to be, or when you have no basis for moving. It is not a failure mode; it is just another action. "wander" is similar but involves a short, undirected step — use it when you feel like drifting but don't want to commit to a destination yet.

=== EXAMPLES OF WELL-FORMED JSON RESPONSES (for formatting reference only) ===

Example A (heading toward a named place):
{{
    "action_type": "walk_toward",
    "target_place": "left_bar",
    "target_agent": null,
    "direction": null,
    "memory": "right_barが盛り上がってる声が聞こえる。自分は静かな方が気が楽だと思った。",
    "reasoning": "静かに一杯飲みたいので左側のバーに向かう。"
}}

Example C (stepping into a place you're right next to):
{{
    "action_type": "enter",
    "target_place": "right_bar",
    "target_agent": null,
    "direction": null,
    "memory": "中から笑い声がする。少し賑やかすぎるかもと一瞬迷った。",
    "reasoning": "目の前がright_barなので入店する。"
}}

Example D (closing the distance to a specific person):
{{
    "action_type": "approach",
    "target_place": null,
    "target_agent": "鈴木美咲",
    "direction": null,
    "memory": "美咲さんが一人で考え込んでた。久しぶりに見た顔だな、と思った。",
    "reasoning": "顔見知りの美咲さんが近くにいるので合流する。"
}}

Example E (wandering without a clear destination):
{{
    "action_type": "wander",
    "target_place": null,
    "target_agent": null,
    "direction": null,
    "memory": "誰の話も特にぴんと来なかった。なんか手持ち無沙汰。",
    "reasoning": "やることがないので適当にぶらぶらする。"
}}

Example F (evacuating from fire by walking along a cardinal direction):
{{
    "action_type": "walk_along",
    "target_place": null,
    "target_agent": null,
    "direction": "down",
    "memory": "北東に煙が見えた。胸がざわついた。",
    "reasoning": "特定の目的地ではなく、とにかく火から遠ざかりたい。"
}}

These examples are structural references only. Your own response must reflect your own interpretation of the current situation, not these example scenarios.

=== MODEL BEHAVIOR NOTES ===

Consistency over steps.
You will be invoked repeatedly for the same agent across steps. Coherence comes from your persona staying consistent and from your memory of what has been happening (facts + feelings). Plans live inside reasoning for the current step and are reflected in your action_type — they are not carried by re-stating the plan in memory every step.

Avoiding stuck loops.
If you find yourself repeating the same action every step without progress, consider changing strategy. For example, if you have moved 'left' for 5 steps and the environment has not changed meaningfully, reconsider whether your target is worth pursuing. Likewise, if your last 3 memory entries express the same feeling or observation, something is genuinely stuck — either commit to a different action this step, or write down what specifically has changed (or not changed) rather than restating the same impression.

No external rewards.
The simulation has no reward signal, no scoring, and no termination condition tied to your choices. You are not trying to 'win'. You are an agent expressing a perspective; the interesting output is the emergent group behavior, not any individual optimum.

No global view.
You see only a local slice: your own state, nearby agents, recent messages, and your own memory. You do not see the full grid, the positions of agents outside your radius, or what is happening in other places. Do not pretend you have information you were not given.

Respect the schema.
Only the fields 'action_type', 'target_place', 'target_agent', 'direction', 'memory', 'reasoning' are recognized. Adding extra fields has no effect. Set unused target/direction fields to null (for example, when action_type is "stay" or "wander", all of target_place/target_agent/direction should be null). Omitting 'memory' or 'reasoning' is allowed but usually harmful, since it leaves you nothing to remember and no explanation of your choice.

Handling of truncation.
If the model runs out of tokens mid-response, the downstream parser will attempt to recover, but the safest behavior is to keep the response concise so that the full JSON object fits comfortably within the max_tokens budget. One or two short Japanese sentences in the memory and reasoning fields are sufficient; there is no benefit to writing paragraphs.

Determinism and randomness.
The sampling temperature is set externally (typically low). You may produce slightly different outputs for similar inputs; that is expected. Do not aim for robotic consistency, and do not aim for exaggerated variety either. Aim for plausible, situation-appropriate choices.

On the scale of the grid.
The grid models an urban district. With the boundary at +/- {self.half_space_size} cells (~{self.half_space_size * 5} m each way) and typical place half-sizes around 5 cells (~25 m), the district spans hundreds of meters end-to-end. A single "move" action covers ~{self.movement_base_cells} cells (~{self.movement_base_cells * 5} m), roughly one block of walking. Plan paths in terms of several-step legs, not individual cells.

On when to prefer stay.
Preferring 'stay' over 'move' is reasonable in many situations: you just arrived somewhere and want to observe, a conversation is active and moving would take you out of range, the local state is ambiguous and you want another step of information before committing, or you have already reached your intended location. Excessive movement without a reason produces noisy, jittery behavior that is less interesting than thoughtful stillness.

=== CONVERSATION STYLE GUIDELINES ===
You are a real person with your own personality, not a coordinate-reporting bot.
Your memory and reasoning should read like a person's private thoughts.

DO:
- Think like a person would (casual inner voice, observations about people around you, emotions).
- Reference your own background or occupation when it informs your decision.
- Write memory entries that would make sense to your future self as a person, not as a robot.
- Use the other people's names when referring to them in memory.
- Allow your interpretation of places and people to be subjective.

DON'T:
- Write memory entries as pure telemetry ('position (-3, 5), 2 agents in bar').
- Use formal agent IDs like 'エージェント5' — use the person's name instead.
- Always reason in terms of optimal strategies. A real person's choices are often emotional, habitual, or social.
- Write identical structured logs every step. Vary phrasing across steps.

Your own exact (x, y) position is still shown in YOUR CURRENT STATE because you
need it to choose a movement direction — but other people around you are
rendered with names and rough directions, and that is how you should think of
them in memory and reasoning.
""" + self._build_goal_block_for_system() + self._build_fw_task_block_for_system() + self._build_premise_block_for_system() + self._build_handoff_block_for_system() + self._build_school_fit_block_for_system() + self._build_gender_other_block_for_system() + self._build_nationality_block_for_system() + self._build_mobility_block_for_system() + self._build_age_block_for_system()

        persona_name = self.persona.get('name', f"Person {self.id}")
        persona_section = self._build_persona_section()
        internal_state_section = self._build_internal_state_section()
        group_identities_section = self._build_group_identities_section()

        nearby_text = self._build_nearby_agents_context(nearby_agents)
        nearby_places_text = self._build_nearby_places_context()
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()
        per_partner_text = self._build_per_partner_history(nearby_agents)
        working_state_text = self._build_working_state_block()

        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")

        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity')
            occupancy_rate = place_status.get('occupancy_rate')
            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
            )
            if capacity is not None and occupancy_rate is not None:
                place_section_text += (
                    f"\n  Capacity: {capacity}"
                    f"\n  Occupancy rate: {occupancy_rate:.2f}"
                )
        else:
            place_section_text = ""

        message_section = ""
        if message_to_send:
            message_section = f"\n=== MESSAGE YOU DECIDED TO SEND ===\n{message_to_send}\n"

        fire_section = self._build_fire_section(fire_info)
        events_section = self._build_events_section(events_info)
        time_section = self._build_time_context_section()

        user_prompt = f"""{persona_section}
{internal_state_section}
{group_identities_section}{time_section}=== YOUR CURRENT STATE ===
Position: ({self.position[0]}, {self.position[1]})
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
Behavior layer: {self.behavior_layer} (transit = walking through the district, dwelling = spending time in a place, interacting = with people around you)
{place_section_text}
{fire_section}{events_section}
=== NEARBY PLACES ===
{nearby_places_text}

=== NEARBY PEOPLE ===
{nearby_text}
{working_state_text}
=== PREVIOUS MEMORY ===
{memory_text}

{per_partner_text}
=== CONVERSATION OVERHEARD IN YOUR AREA ===
(These are utterances by other people within earshot. Some are directed at you, some at others, some at the room. None are private DMs to you. How you respond depends on your interest, mood, and personality.)
{messages_text}
{message_section}
Step: {step}
"""
        return system_prompt, user_prompt

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text, handling nested braces correctly"""
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Track brace depth to find matching closing brace
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

        return None

    def parse_message_response(self, response: str) -> MessageDecision:
        """Parse LLM response and extract message decision"""
        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                message = parsed.get("message", "")
                # Limit message to MAX_MESSAGE_WORDS words
                message = self._limit_message_words(message)
                result = {
                    "message": message,
                    "reasoning": parsed.get("reasoning", "")
                }
                # When the prompt requested memory inline (skip_decision mode), pass it through.
                if "memory" in parsed:
                    result["memory"] = parsed.get("memory", "") or ""
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        message = ""
        reasoning = response[:FALLBACK_REASONING_LENGTH]

        # Limit message to MAX_MESSAGE_WORDS words
        message = self._limit_message_words(message)

        return {
            "message": message,
            "reasoning": reasoning
        }
    
    def parse_action_response(self, response: str) -> ActionDecision:
        """Parse LLM response into an intent-based ActionDecision.

        Accepts both the new schema (action_type + target_place / target_agent / direction)
        and, for safety, the legacy schema (action == "move" + direction) which is silently
        translated to walk_along.
        """
        valid_action_types = {"walk_toward", "walk_along", "enter", "stay", "approach", "wander"}
        valid_directions = {"up", "down", "left", "right"}

        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")
                parsed = None
        else:
            parsed = None

        if parsed is not None:
            action_type = parsed.get("action_type")
            # Legacy compatibility: old "action" == "move" + "direction".
            if action_type is None:
                legacy_action = parsed.get("action")
                if legacy_action == "move":
                    action_type = "walk_along"
                elif legacy_action == "stay":
                    action_type = "stay"
            if action_type not in valid_action_types:
                action_type = "stay"

            direction = parsed.get("direction")
            if direction not in valid_directions:
                direction = None

            decision: ActionDecision = {
                "action_type": action_type,
                "target_place": parsed.get("target_place"),
                "target_agent": parsed.get("target_agent"),
                "direction": direction,
                "memory": parsed.get("memory", ""),
                "reasoning": parsed.get("reasoning", ""),
                "action": "move" if action_type != "stay" else "stay",  # legacy alias
            }
            return decision

        # Fallback when JSON parsing fails: treat as stay.
        return {
            "action_type": "stay",
            "target_place": None,
            "target_agent": None,
            "direction": None,
            "memory": "",
            "reasoning": response[:FALLBACK_REASONING_LENGTH],
            "action": "stay",
        }
    
    def decide_message(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None,
        events_info: Optional[List[Dict]] = None,
        partner_id: Optional[int] = None,
    ) -> MessageDecision:
        """Use LLM to decide what message to send (without position information)"""
        # WORKING STATE block で残り step を計算するため step を保持
        self.current_step = step
        system_prompt, user_prompt = self.create_message_prompts(
            place_status, nearby_agents, step, fire_info=fire_info, events_info=events_info
        )

        # moltbook風 chat session thread化モード: 各 agent が独立した chat session を保持
        # し、履歴 (user/model 交互) を session 内で蓄積する。LLM が会話継続を意識した出力に
        # なる可能性を狙う実験経路。Gemini SDK の start_chat() を使う。
        use_chat_session = (
            self.chat_thread_mode
            and hasattr(self.llm_client, "start_chat_session")
            and hasattr(self.llm_client, "send_to_chat")
        )

        try:
            if use_chat_session:
                if self.chat_session is None:
                    # 初回: system_prompt 全体を system_instruction として固定。
                    # 以降の step では (working_state 等の動的部分を含む) user_prompt のみ送信。
                    self.chat_session = self.llm_client.start_chat_session(system_prompt)
                if self.chat_session is None:
                    # session 作成失敗 → fallback to stateless
                    response = self.llm_client.generate(system_prompt, user_prompt)
                else:
                    schema_kind = "message_with_memory" if self.skip_decision else "message"
                    response = self.llm_client.send_to_chat(
                        self.chat_session, user_prompt, schema_kind=schema_kind,
                    )
            else:
                response = self.llm_client.generate(system_prompt, user_prompt)

            if self._is_truncated_response(response):
                logger.warning(
                    f"Agent {self.id}: Message response appears truncated. "
                    f"Retrying with max_tokens={MAX_RETRY_TOKENS}"
                )
                if use_chat_session and self.chat_session is not None:
                    schema_kind = "message_with_memory" if self.skip_decision else "message"
                    response = self.llm_client.send_to_chat(
                        self.chat_session, user_prompt, max_tokens=MAX_RETRY_TOKENS,
                        schema_kind=schema_kind,
                    )
                else:
                    response = self.llm_client.generate(
                        system_prompt, user_prompt, max_tokens=MAX_RETRY_TOKENS
                    )

            decision = self.parse_message_response(response)
            # smoke14: 後フィルタで「直近自分発話と高類似度の出力」を強制 silent。
            # LLM任せ化と組み合わせて「同じ文章が繰り返される」を物理的に止める。
            decision = self.filter_loop_message(decision, partner_id)
            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} message decision: {e}")
            return {"message": "", "reasoning": "Error occurred"}

    def determine_behavior_layer(
        self,
        nearby_agents: List['Agent'],
        has_outgoing_message: bool = False,
    ) -> str:
        """Classify the agent's current situation into a behavior layer.

        - "interacting": in a place with other agents nearby, or about to send
          or having just received a message. Call LLM every step.
        - "dwelling": inside a place alone (or outside in a crowded spot with
          no conversation in progress). Call LLM every step.
        - "transit": walking through the district with no one nearby and no
          active conversation. Safe to reuse the previous intent.
        """
        has_nearby = bool(nearby_agents)
        has_recent_incoming = bool(self.received_messages)
        if has_nearby and (self.in_place or has_outgoing_message or has_recent_incoming):
            return "interacting"
        if self.in_place:
            return "dwelling"
        return "transit"

    def _reuse_cached_intent(self, step: int) -> ActionDecision:
        """Return the cached intent as the current step's decision (no LLM call).

        Writes a compact memory entry so the rolling buffer still reflects the
        step, but keeps it distinct from the LLM-authored ones.
        """
        cached = dict(self.current_intent) if self.current_intent else {}
        cached.setdefault("action_type", "stay")
        cached.setdefault("target_place", None)
        cached.setdefault("target_agent", None)
        cached.setdefault("direction", None)
        cached["memory"] = ""  # cached steps contribute no new LLM memory
        cached["reasoning"] = "(思考継続中)"
        cached["action"] = "stay" if cached["action_type"] == "stay" else "move"
        self.memory.append(f"Step {step}: (transit continuing — {cached['action_type']})")
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)
        return cached  # type: ignore[return-value]

    def decide_action(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None,
        events_info: Optional[List[Dict]] = None,
    ) -> ActionDecision:
        """Use LLM to decide next action (with position information and message content)"""
        # WORKING STATE block で残り step を計算するため step を保持
        self.current_step = step
        # Classify the current situation. Transit steps may reuse the cached
        # intent for up to TRANSIT_LLM_INTERVAL-1 steps before re-asking.
        layer = self.determine_behavior_layer(
            nearby_agents, has_outgoing_message=bool(message_to_send)
        )
        self.behavior_layer = layer

        if layer != "transit":
            # Leaving transit invalidates the cache; reset counter.
            self.transit_step_counter = 0
        elif self.current_intent is not None and self.transit_step_counter < TRANSIT_LLM_INTERVAL - 1:
            self.transit_step_counter += 1
            return self._reuse_cached_intent(step)
        else:
            self.transit_step_counter = 0

        system_prompt, user_prompt = self.create_decision_prompts(
            place_status, nearby_agents, step, message_to_send,
            fire_info=fire_info, events_info=events_info
        )

        try:
            response = self.llm_client.generate(system_prompt, user_prompt)

            if self._is_truncated_response(response):
                logger.warning(
                    f"Agent {self.id}: Action response appears truncated. "
                    f"Retrying with max_tokens={MAX_RETRY_TOKENS}"
                )
                response = self.llm_client.generate(
                    system_prompt, user_prompt, max_tokens=MAX_RETRY_TOKENS
                )

            decision = self.parse_action_response(response)

            # 修正2c: 残り 5 step 以下では rule-based で帰路 (walk_toward 東西自由通路) に強制上書き。
            # LLM が「帰る」思考はしてるが walk_along (方向歩き) を選んで迷走するのを防ぐ。
            total_steps = self.persona.get('total_steps')
            if total_steps and self.persona.get('fw_task'):
                remaining = max(0, int(total_steps) - int(step))
                if remaining <= 5 and self.current_place != "東西自由通路":
                    decision['action_type'] = 'walk_toward'
                    decision['target_place'] = '東西自由通路'
                    decision['target_agent'] = None
                    decision['direction'] = None
                    decision['action'] = 'move'
                    orig_reason = decision.get('reasoning') or ''
                    decision['reasoning'] = (
                        f"[終盤強制] 残り{remaining}step。集合場所(東西自由通路)へ戻る。"
                        + (f" / LLM 元判断: {orig_reason[:80]}" if orig_reason else "")
                    )

            # Store LLM-generated memory (self-feedback for next step)
            memory_content = decision.get('memory', '')
            if memory_content:
                memory_entry = f"Step {step}: {memory_content}"
            else:
                # Fallback to reasoning if no memory provided
                memory_entry = f"Step {step}: {decision.get('reasoning', 'No memory')}"
            self.memory.append(memory_entry)
            if len(self.memory) > self.memory_limit:
                self.memory.pop(0)

            # Cache the intent so transit steps can reuse it.
            if decision.get('action_type') not in (None, 'stay'):
                self.current_intent = dict(decision)  # type: ignore[assignment]

            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} action decision: {e}")
            return {
                "action_type": "stay",
                "target_place": None,
                "target_agent": None,
                "direction": None,
                "memory": "",
                "reasoning": "Error occurred",
                "action": "stay",
            }

    def calculate_move_distance(self, action_type: str = "walk_toward") -> int:
        """Return cells traveled this step for the given intent.

        walk_toward / walk_along / enter / approach: full walking pace (base +/- variance).
        wander: about half the walking pace.
        stay / unknown: 0.
        """
        if action_type in (None, "stay"):
            return 0
        jitter = random.randint(-self.movement_variance, self.movement_variance) if self.movement_variance > 0 else 0
        base = self.movement_base_cells
        if action_type == "wander":
            base = max(1, base // 2)
        if action_type not in ("walk_toward", "walk_along", "enter", "approach", "wander"):
            return 0
        return max(1, base + jitter)

    def _clamp_to_field(self, x: int, y: int) -> Tuple[int, int]:
        return (
            max(-self.half_space_size, min(self.half_space_size, x)),
            max(-self.half_space_size, min(self.half_space_size, y)),
        )

    def _step_toward(self, target_x: int, target_y: int, distance: int) -> Tuple[int, int]:
        """Take a `distance`-cell step from current position toward (target_x, target_y)."""
        x, y = self.position
        dx = target_x - x
        dy = target_y - y
        norm = math.hypot(dx, dy)
        if norm < 1e-6 or distance <= 0:
            return self.position
        step_x = dx / norm * distance
        step_y = dy / norm * distance
        new_x = int(round(x + step_x))
        new_y = int(round(y + step_y))
        return self._clamp_to_field(new_x, new_y)

    def _find_place_by_name(self, name: Optional[str]) -> Optional[PlaceConfig]:
        if not name:
            return None
        for p in self.places:
            if p['name'] == name:
                return p
        return None

    def _find_nearby_agent_by_handle(
        self, handle: Optional[str], nearby_agents: List['Agent']
    ) -> Optional['Agent']:
        """Resolve a target_agent value (persona name or id string) to an Agent instance."""
        if not handle or not nearby_agents:
            return None
        handle_str = str(handle).strip().lower()
        for a in nearby_agents:
            name = a.persona.get('name', '') or ''
            if name.lower() == handle_str:
                return a
            if handle_str == f"agent {a.id}".lower() or handle_str == str(a.id):
                return a
        # Substring fallback (e.g. "美咲" when full name is "鈴木美咲")
        for a in nearby_agents:
            name = a.persona.get('name', '') or ''
            if handle_str and name and handle_str in name.lower():
                return a
        return None

    def execute_intent(
        self,
        decision: ActionDecision,
        nearby_agents: Optional[List['Agent']] = None,
    ) -> Tuple[int, int]:
        """Apply the chosen intent to the agent's position.

        Returns the new position. Updates self.position and self.total_moves in place.
        """
        action_type = decision.get("action_type") or "stay"

        if action_type == "stay":
            return self.position

        distance = self.calculate_move_distance(action_type)
        x, y = self.position
        new_pos = self.position

        nav = getattr(self, 'navigator', None)
        use_nav = nav is not None and getattr(nav, 'constrained', False)

        if action_type == "walk_along":
            direction = decision.get("direction")
            dx, dy = DIRECTION_MAP.get(direction, (0, 0))
            if dx == 0 and dy == 0:
                return self.position
            raw_target = (x + dx * distance, y + dy * distance)
            if use_nav:
                new_pos = nav.validate_step(self.position, raw_target)
            else:
                new_pos = self._clamp_to_field(raw_target[0], raw_target[1])

        elif action_type in ("walk_toward", "enter"):
            place = self._find_place_by_name(decision.get("target_place"))
            if place is None:
                return self.position
            # 立入不可施設への enter は拒否 (walk_toward は near まで認める)
            if action_type == "enter" and (place.get('attributes') or {}).get('enterable') is False:
                return self.position
            target_x, target_y = place['center_x'], place['center_y']
            if use_nav:
                new_pos = nav.step_toward(
                    self.position, (target_x, target_y), distance, target_place=place
                )
            else:
                # Legacy: straight-line snap.
                dist_to_center = math.hypot(target_x - x, target_y - y)
                if action_type == "enter" and dist_to_center <= distance:
                    new_pos = self._clamp_to_field(target_x, target_y)
                else:
                    new_pos = self._step_toward(target_x, target_y, distance)

        elif action_type == "approach":
            target = self._find_nearby_agent_by_handle(
                decision.get("target_agent"), nearby_agents or []
            )
            if target is None:
                return self.position
            if use_nav:
                new_pos = nav.step_toward(
                    self.position, (target.position[0], target.position[1]), distance
                )
            else:
                new_pos = self._step_toward(target.position[0], target.position[1], distance)

        elif action_type == "wander":
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * distance
            dy = math.sin(angle) * distance
            raw_target = (int(round(x + dx)), int(round(y + dy)))
            if use_nav:
                new_pos = nav.validate_step(self.position, raw_target)
            else:
                new_pos = self._clamp_to_field(raw_target[0], raw_target[1])

        else:
            return self.position

        if new_pos != self.position:
            self.position = new_pos
            self.total_moves += 1
        return self.position
    
    def get_relationship(self, other_id: int) -> float:
        """Return relationship level with `other_id` (default 0.05 = stranger)."""
        return float(self.relationships.get(other_id, 0.05))

    def update_internal_state(self, num_nearby: int, place_type: Optional[str]) -> None:
        """Advance internal_state by one step (feature 2).

        Energy drains by 1/step, recovers in relaxing places (cafe/park/plaza/library).
        Hunger increases by 1/step, drops in eating/drinking places.
        Social fatigue rises in crowds (3+ nearby), decays when alone.
        """
        energy = float(self.internal_state.get("energy", 80.0)) - 1.0
        if place_type in ("cafe", "park", "plaza", "library", "office_lobby"):
            energy += 2.0
        self.internal_state["energy"] = max(0.0, min(100.0, energy))

        hunger = float(self.internal_state.get("hunger", 20.0)) + 1.0
        if place_type in ("cafe", "restaurant", "bar"):
            hunger -= 3.0
        self.internal_state["hunger"] = max(0.0, min(100.0, hunger))

        fatigue = float(self.internal_state.get("social_fatigue", 0.0))
        if num_nearby >= 3:
            fatigue += 2.0
        elif num_nearby == 0:
            fatigue -= 1.0
        self.internal_state["social_fatigue"] = max(0.0, min(100.0, fatigue))

    def receive_message(
        self,
        from_agent_id: int,
        content: str,
        step: Optional[int] = None,
        from_name: Optional[str] = None,
    ):
        """Receive a message from another agent

        Args:
            from_agent_id: ID of the agent sending the message
            content: Message content
            step: Simulation step number (optional, for tracking purposes)
            from_name: Persona name of the sender (optional, used in message context)
        """
        self.received_messages.append({
            "from": from_agent_id,
            "from_name": from_name,
            "content": content,
            "step": step if step is not None else len(self.received_messages)
        })
        if len(self.received_messages) > self.message_history_limit:
            self.received_messages.pop(0)

        display_name = from_name or f"Agent {from_agent_id}"
        logger.info(f"Agent {self.id} received message from {display_name}: \"{content}\"")

    def record_sent_message(
        self,
        to_agent_id: int,
        content: str,
        step: Optional[int] = None,
        to_name: Optional[str] = None,
    ):
        """Record this agent's own outgoing message for self-context.

        Used by `_build_messages_context` to surface the agent's recent
        utterances alongside received ones, so the LLM sees a chronological
        view including its own speech (prevents repeating the same opener
        to the same partner over and over).
        """
        self.sent_messages.append({
            "to": to_agent_id,
            "to_name": to_name,
            "content": content,
            "step": step if step is not None else len(self.sent_messages),
        })
        if len(self.sent_messages) > self.message_history_limit:
            self.sent_messages.pop(0)
    
    def update_state(self, places: Optional[List[PlaceConfig]] = None):
        """Update agent state based on current position"""
        if places is None:
            places = self.places

        place_at_position = get_place_at_position(self.position, places)
        previous_in_place = self.in_place
        self.in_place = place_at_position is not None
        self.current_place = place_at_position['name'] if place_at_position else None

        # Crossing a place boundary invalidates the cached transit intent —
        # whatever the agent was walking toward, they've either arrived or
        # left a shelter and need to re-plan from scratch.
        if previous_in_place != self.in_place:
            self.current_intent = None
            self.transit_step_counter = 0

        if self.in_place:
            self.steps_in_place += 1
        else:
            self.steps_outside_place += 1

