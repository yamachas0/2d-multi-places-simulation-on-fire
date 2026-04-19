"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import math
import logging
from typing import List, Tuple, Optional, Dict, TypedDict
from claude_client import ClaudeClient
from utils import is_position_in_place, get_place_at_position, PlaceConfig, generate_random_persona

logger = logging.getLogger(__name__)

# Constants
FALLBACK_REASONING_LENGTH = 100
MAX_MESSAGE_WORDS = 200
MAX_RETRY_TOKENS = 1500  # Expanded max_tokens used when a response is detected as truncated.

# Direction mappings (4 cardinal directions only)
# Coordinate system: X increases from left to right, Y increases from bottom to top
DIRECTION_MAP = {
    "up": (0, 1),      # Y+1 (move upward)
    "down": (0, -1),   # Y-1 (move downward)
    "left": (-1, 0),   # X-1 (move leftward)
    "right": (1, 0),   # X+1 (move rightward)
}


class MessageDecision(TypedDict):
    """Type definition for agent message decision"""
    message: str  # Message to communicate with nearby agents
    reasoning: str  # Explanation of the message decision


class ActionDecision(TypedDict):
    """Type definition for agent action decision"""
    action: str  # "move" or "stay"
    direction: Optional[str]  # Direction to move (None if action is "stay")
    memory: str  # What the agent wants to remember for the next step
    reasoning: str  # Explanation of the decision


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
    ):
        self.id = agent_id
        self.position = initial_position
        self.llm_client = llm_client
        self.communication_radius = communication_radius
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.gender = gender

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
        self.received_messages: List[Dict] = []  # Messages from other agents

        # Statistics
        self.steps_in_place = 0
        self.steps_outside_place = 0
        self.total_moves = 0

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
            include_position: If True, include a rough directional hint; if False, omit location.
        """
        if not nearby_agents:
            return "No nearby agents."

        nearby_info = []
        for agent in nearby_agents:
            name = agent.persona.get('name', f"Person {agent.id}")
            age = agent.persona.get('age', '?')
            occupation = agent.persona.get('occupation', '?')

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

            if include_position:
                direction = self._position_to_rough_direction(agent.position)
                nearby_info.append(
                    f"{name} ({age}, {agent.gender}, {occupation}) {status}, roughly {direction}"
                )
            else:
                nearby_info.append(
                    f"{name} ({age}, {agent.gender}, {occupation}) {status}"
                )
        return "\n".join(nearby_info)
    
    def _build_memory_context(self) -> str:
        """Build context string from agent memory"""
        if not self.memory:
            return "No previous experiences."

        recent_memory = self.memory[-self.memory_size:]
        return "\n".join([f"- {m}" for m in recent_memory])
    
    def _build_messages_context(self) -> str:
        """Build context string from received messages using sender names when available."""
        if not self.received_messages:
            return "No messages received."

        recent_messages = self.received_messages[-self.message_context_size:]
        lines = []
        for msg in recent_messages:
            sender_name = msg.get('from_name') or f"Person {msg.get('from', '?')}"
            lines.append(f"from {sender_name}: {msg['content']}")
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
            place_locations.append(
                f"{place['name']} ({place_type}): center at ({place['center_x']}, {place['center_y']}), "
                f"covers X from {place['center_x'] - place['half_size']} to {place['center_x'] + place['half_size']}, "
                f"Y from {place['center_y'] - place['half_size']} to {place['center_y'] + place['half_size']}"
            )
        return "\n".join(place_locations)

    def _build_world_description(self) -> str:
        """Build short world description based on unique place types."""
        place_types = [p['type'] for p in self.places]
        unique_types = list(set(place_types))
        return f"a 2D world with multiple places ({', '.join(unique_types)})"

    def create_message_prompts(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create (system_prompt, user_prompt) tuple for LLM message decision.

        System prompt is static across all calls (cacheable via Anthropic API).
        User prompt contains dynamic per-step state.
        """
        world_description = self._build_world_description()
        place_locations_text = self._build_place_locations_text()

        system_prompt = f"""You are an autonomous agent in {world_description}. Right now you are deciding what message, if any, to broadcast to the nearby agents you can currently communicate with. Your decision should emerge from your current state, your accumulated memory, and the ongoing conversational context.

=== WORLD STRUCTURE ===
The world is a 2D grid with origin at (0, 0).
Field boundaries: X and Y both range from -{self.half_space_size} to +{self.half_space_size} inclusive.
Places are special rectangular regions (bars, cafes, libraries, etc.) where agents can gather; each has a name, a type, and a capacity. An agent is either inside exactly one place or outside all places at any given moment.

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

Turn-taking.
Conversations emerge from turn-taking. If the most recent message in your received-messages list is from agent A, and it was addressed (even loosely) to you, it is usually natural to respond before sending a fresh topic. If multiple agents recently said similar things, you might acknowledge the shared sentiment rather than reply to each individually. There is no rigid rule; use judgment.

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
"""

        persona_name = self.persona.get('name', f"Person {self.id}")
        persona_age = self.persona.get('age', '?')
        persona_gender = self.persona.get('gender', self.gender)
        persona_occupation = self.persona.get('occupation', '?')
        persona_background = self.persona.get('background', '')
        persona_speech_style = self.persona.get('speech_style', '')

        persona_section = (
            f"=== WHO YOU ARE ===\n"
            f"Name: {persona_name}\n"
            f"Age: {persona_age}\n"
            f"Gender: {persona_gender}\n"
            f"Occupation: {persona_occupation}\n"
            f"Background: {persona_background}\n"
            f"Speech style: {persona_speech_style}\n\n"
            f"Speak and act according to this identity. Other people around you know you by name, not by ID.\n\n"
        )

        nearby_text = self._build_nearby_agents_context(nearby_agents, include_position=False)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")

        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)
            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
                f"\n  Capacity: {capacity}"
                f"\n  Occupancy rate: {occupancy_rate:.2f}"
            )
        else:
            place_section_text = ""

        fire_section = self._build_fire_section(fire_info)

        user_prompt = f"""{persona_section}You are {persona_name} in this 2D world.

=== YOUR CURRENT STATE ===
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
=== NEARBY AGENTS (you can communicate with these people) ===
{nearby_text}

=== PREVIOUS MEMORY ===
{memory_text}

=== MESSAGES FROM OTHERS ===
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
        fire_info: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Create (system_prompt, user_prompt) tuple for LLM action decision.

        System prompt is static across all calls (cacheable via Anthropic API).
        User prompt contains dynamic per-step state.
        """
        world_description = self._build_world_description()
        place_locations_text = self._build_place_locations_text()

        system_prompt = f"""You are an autonomous agent in {world_description}. You make decisions based on the current situation, your accumulated memory, and the communication you exchange with nearby agents. Your behavior should emerge organically from your own judgment rather than from any externally defined 'correct' policy.

=== WORLD STRUCTURE ===
The world is a 2D grid with origin at (0, 0).
Field boundaries: X and Y both range from -{self.half_space_size} to +{self.half_space_size} inclusive.
Every position outside of any place is 'open ground'. Places are special rectangular regions where agents can gather; they have a name, a type (such as bar, cafe, library) and a fixed capacity.
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
Each step you must choose exactly one action:
- "stay": remain at your current position. Use this when you have no reason to move or when you want to observe.
- "move" with one of four cardinal directions:
  - "up":    Y increases by 1 (move toward larger Y)
  - "down":  Y decreases by 1 (move toward smaller Y)
  - "left":  X decreases by 1 (move toward smaller X)
  - "right": X increases by 1 (move toward larger X)
Diagonal movement is not available. Movements that would take you past the field boundary are clamped to the boundary automatically.

=== RESPONSE FORMAT (日本語で回答すること) ===
Respond with exactly one JSON object and nothing else. Do not include any prose, markdown fences, explanations, or blank lines before or after the JSON block. The JSON must be valid and parseable.
The values of "action" and "direction" MUST remain in English exactly as shown (move / stay / up / down / left / right). The values of "memory" and "reasoning" MUST be written in Japanese (日本語).
{{
    "action": "move" or "stay",
    "direction": "up", "down", "left", or "right" (only when action is "move"; otherwise omit or set to null),
    "memory": "次のステップで覚えておきたいこと（自分の考え・観察・意図）を日本語で",
    "reasoning": "この判断をした理由を日本語で簡潔に"
}}

=== BEHAVIORAL GUIDANCE ===
There is no 'correct' behavior. Your actions should emerge from your own interpretation of:
- Your current position and whether you are inside a place.
- The quantitative data about the environment (occupancy, distances, fire, capacity).
- Messages you have received from nearby agents and your own previously-sent messages.
- Your accumulated memory of past steps (your thoughts, observations, intentions).

Use memory as a scratchpad to maintain coherent intentions across steps. If you formed a plan several steps ago (for example, 'move toward the left bar to check if it is crowded'), your memory is how you keep that plan alive across subsequent LLM calls. If you decide to abandon the plan, write the new intention into memory so future steps see the update.

=== DETAILED SEMANTICS ===

Positions and movement.
Your position is the integer pair (x, y). 'up' adds +1 to y, 'down' subtracts 1 from y, 'left' subtracts 1 from x, 'right' adds +1 to x. The coordinate system is origin-centered: (0, 0) is the middle of the field. Negative x is the left half of the field, positive x is the right half. Negative y is the bottom half, positive y is the top half. Only one cell of movement per step is possible; reaching a distant location requires several consecutive steps.

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

Step counter.
The 'Step' value in the user message is a monotonically increasing integer indicating the current simulation step. It allows you to situate your memory entries in time. Your memory entries from prior steps are prefixed with the step number so you can reconstruct the sequence of events.

=== ADDITIONAL REMARKS ===

On emergent behavior.
This simulation is a study of emergent collective behavior under pure agent autonomy. Your individual decision is one small input to a larger system. What you do influences what other agents observe about crowd levels and messages; what they do changes what you observe. The interesting structure, if any, comes from the accumulation of many small local decisions, not from any central instruction. You should not try to solve the situation globally; just act from your own local perspective.

On uncertainty.
You will often receive incomplete information. You may not know how many agents are outside all places. You may not know what is happening in a place you are not in. You may hear conflicting messages from different agents. Treat this uncertainty as real and make the best decision you can with what you have, rather than inventing facts you do not observe.

On changing your mind.
Nothing forces you to carry a decision through. If you were heading toward one place and received information suggesting another destination is more interesting, you can reverse course. Record the change in your memory so you remember the updated plan on the next step. Stubborn commitment to an old plan without a reason is worse than adapting.

On brevity.
When you write memory entries, aim for concise but specific sentences. A memory like '左バーに向かう' is useful; a memory like 'ok' is not. When you write reasoning, one or two sentences that explain the 'why' of the action are sufficient.

On the JSON format.
The parser is strict. Any characters outside of the JSON object (including leading '```json', trailing commentary, or explanatory text) will cause a parse failure. The parser attempts to extract a brace-matched block, but the safe behavior is to emit only the JSON object and nothing else.

=== EXTENDED GUIDANCE FOR ACTION DECISIONS ===

Walkthrough of a typical step.
A typical step proceeds as follows. You receive your current state: where you are, whether you are in a place, which agents are nearby, the messages they sent, your past memory entries. You interpret this information in light of your own priorities, which you built up over prior steps. You then choose one of five atomic outputs: stay, or move up, move down, move left, or move right. Along with that choice, you write a short memory entry that records your interpretation of the current situation and your next intended step; this memory is visible to you on the following step.

How memory flows across steps.
Each step appends one memory line to a rolling buffer. You see only the last few entries in the buffer (the size is controlled outside of the prompt). Entries are strings, prefixed with 'Step N:' by the system, followed by whatever Japanese text you wrote in the 'memory' field. Because the buffer is small, writing redundant or vague memory entries wastes slots. Write one informative sentence per step.

How occupancy interacts with decisions.
When you are inside a place, the occupancy rate tells you how crowded the place is. For example, occupancy rate 0.5 means half of capacity is used. You may interpret a low occupancy rate as pleasant, crowded-preferred, or uninteresting; there is no externally imposed interpretation. Similarly a high occupancy rate can be interpreted as lively, uncomfortable, or indicative of a popular spot. Use your own reasoning and your memory to arrive at a consistent interpretation over time.

How the place boundaries shape movement.
Places are rectangular regions. To enter a place, your x and y need to enter the rectangle defined by (center_x - half_size) to (center_x + half_size) and similarly for y. Because you can move only one cell per step in one cardinal direction, entering a place usually takes multiple steps. The shortest path from a position outside a place is roughly: move in whichever axis reduces the larger of the two distances first, then fine-tune the other axis. The system does not enforce this; you decide the path.

Why fire matters, quantitatively.
Fire events carry two numerical attributes: intensity (0.0 to 1.0) and radius. When a fire is active, the user message shows each fire's coordinates, intensity, radius, and your Euclidean distance to the fire. There is no mapping from these numbers to any qualitative label. You are free to treat a high-intensity short-radius fire differently from a low-intensity wide-radius one; you choose what the numbers mean. The simulation does not 'damage' or 'kill' agents; there is no game-over state. You are building a history of choices, not optimizing a score.

Interpreting nearby agents.
The nearby_agents list shows other agents within your communication radius who are in the same area as you. Each entry includes the other agent's id, their gender, their position, and whether they are in a place. You do not see their memory, their personality, or their intentions; you only see their observable state. Any inference about motives comes from the messages they broadcast.

Dealing with boundaries.
Attempting to move beyond the field boundary is safe: the system clamps your resulting position to the valid range. So 'move up' at y = +{self.half_space_size} results in no change. If you believe you are already at the boundary in a direction, prefer a direction that actually changes your position.

The 'stay' action.
'stay' is a legitimate choice. Use it when observing, when waiting for messages, when you are already where you wanted to be, or when you have no basis for moving. It is not a failure mode; it is just another action.

=== EXAMPLES OF WELL-FORMED JSON RESPONSES (for formatting reference only) ===

Example A (moving toward a place):
{{
    "action": "move",
    "direction": "left",
    "memory": "左側にあるバーに向かって移動中。あと数ステップで到達できる見込み。",
    "reasoning": "左バーの方向に進む必要があるため、左に1マス移動する。"
}}

Example B (staying in a place):
{{
    "action": "stay",
    "direction": null,
    "memory": "右バーの中にいて、他の人と話しているところ。もう少しここにいたい。",
    "reasoning": "会話が続いているので、今のステップは留まる。"
}}

Example C (evacuating from fire):
{{
    "action": "move",
    "direction": "down",
    "memory": "火災が右上にあり、距離が近い。南方向へ逃げることにした。",
    "reasoning": "火災から離れるため、下方向へ移動する。"
}}

These examples are structural references only. Your own response must reflect your own interpretation of the current situation, not these example scenarios.

=== MODEL BEHAVIOR NOTES ===

Consistency over steps.
You will be invoked repeatedly for the same agent across steps. The memory field is the primary mechanism for keeping your behavior consistent. If step N writes 'I plan to move to the left bar', step N+1 should either continue that plan or write down why you are changing it.

Avoiding stuck loops.
If you find yourself repeating the same action every step without progress, consider changing strategy. For example, if you have moved 'left' for 5 steps and the environment has not changed meaningfully, reconsider whether your target is worth pursuing. Use memory to detect such loops.

No external rewards.
The simulation has no reward signal, no scoring, and no termination condition tied to your choices. You are not trying to 'win'. You are an agent expressing a perspective; the interesting output is the emergent group behavior, not any individual optimum.

No global view.
You see only a local slice: your own state, nearby agents, recent messages, and your own memory. You do not see the full grid, the positions of agents outside your radius, or what is happening in other places. Do not pretend you have information you were not given.

Respect the schema.
Only the four fields 'action', 'direction', 'memory', 'reasoning' are recognized. Adding extra fields has no effect. Omitting 'direction' when action is 'stay' is fine (or set it to null). Omitting 'memory' or 'reasoning' is allowed but usually harmful, since it leaves you nothing to remember and no explanation of your choice.

Handling of truncation.
If the model runs out of tokens mid-response, the downstream parser will attempt to recover, but the safest behavior is to keep the response concise so that the full JSON object fits comfortably within the max_tokens budget. One or two short Japanese sentences in the memory and reasoning fields are sufficient; there is no benefit to writing paragraphs.

Determinism and randomness.
The sampling temperature is set externally (typically low). You may produce slightly different outputs for similar inputs; that is expected. Do not aim for robotic consistency, and do not aim for exaggerated variety either. Aim for plausible, situation-appropriate choices.

On the scale of the grid.
The grid is small. With the boundary roughly at +/- {self.half_space_size} and a place half-size typically around 5, the interior of the world is on the order of tens of cells across. A single cell step is a meaningful unit of distance; ten cells is a significant traversal. Plan your paths with this scale in mind.

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
"""

        persona_name = self.persona.get('name', f"Person {self.id}")
        persona_age = self.persona.get('age', '?')
        persona_gender = self.persona.get('gender', self.gender)
        persona_occupation = self.persona.get('occupation', '?')
        persona_background = self.persona.get('background', '')
        persona_speech_style = self.persona.get('speech_style', '')

        persona_section = (
            f"=== WHO YOU ARE ===\n"
            f"Name: {persona_name}\n"
            f"Age: {persona_age}\n"
            f"Gender: {persona_gender}\n"
            f"Occupation: {persona_occupation}\n"
            f"Background: {persona_background}\n"
            f"Speech style: {persona_speech_style}\n\n"
            f"Think and act according to this identity. Other people around you know you by name, not by ID.\n\n"
        )

        nearby_text = self._build_nearby_agents_context(nearby_agents)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")

        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)
            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
                f"\n  Capacity: {capacity}"
                f"\n  Occupancy rate: {occupancy_rate:.2f}"
            )
        else:
            place_section_text = ""

        message_section = ""
        if message_to_send:
            message_section = f"\n=== MESSAGE YOU DECIDED TO SEND ===\n{message_to_send}\n"

        fire_section = self._build_fire_section(fire_info)

        user_prompt = f"""{persona_section}You are {persona_name} in this 2D world.

=== YOUR CURRENT STATE ===
Position: ({self.position[0]}, {self.position[1]})
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
=== NEARBY AGENTS ===
{nearby_text}

=== PREVIOUS MEMORY ===
{memory_text}

=== MESSAGES FROM OTHERS ===
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

    def _extract_direction_from_text(self, text: str) -> Optional[str]:
        """Extract direction from text using keyword matching (4 cardinal directions only)"""
        text_lower = text.lower()

        # Check cardinal directions only
        if "up" in text_lower:
            return "up"
        elif "down" in text_lower:
            return "down"
        elif "left" in text_lower:
            return "left"
        elif "right" in text_lower:
            return "right"

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
                return {
                    "message": message,
                    "reasoning": parsed.get("reasoning", "")
                }
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
        """Parse LLM response and extract action decision"""
        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                return {
                    "action": parsed.get("action", "stay"),
                    "direction": parsed.get("direction"),
                    "memory": parsed.get("memory", ""),
                    "reasoning": parsed.get("reasoning", "")
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        action = "stay"
        direction = None
        memory = ""
        reasoning = response[:FALLBACK_REASONING_LENGTH]

        if "move" in response.lower():
            action = "move"
            direction = self._extract_direction_from_text(response)

        return {
            "action": action,
            "direction": direction,
            "memory": memory,
            "reasoning": reasoning
        }
    
    def decide_message(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> MessageDecision:
        """Use LLM to decide what message to send (without position information)"""
        system_prompt, user_prompt = self.create_message_prompts(
            place_status, nearby_agents, step, fire_info=fire_info
        )

        try:
            response = self.llm_client.generate(system_prompt, user_prompt)

            if self._is_truncated_response(response):
                logger.warning(
                    f"Agent {self.id}: Message response appears truncated. "
                    f"Retrying with max_tokens={MAX_RETRY_TOKENS}"
                )
                response = self.llm_client.generate(
                    system_prompt, user_prompt, max_tokens=MAX_RETRY_TOKENS
                )

            decision = self.parse_message_response(response)
            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} message decision: {e}")
            return {"message": "", "reasoning": "Error occurred"}

    def decide_action(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> ActionDecision:
        """Use LLM to decide next action (with position information and message content)"""
        system_prompt, user_prompt = self.create_decision_prompts(
            place_status, nearby_agents, step, message_to_send, fire_info=fire_info
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

            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} action decision: {e}")
            return {"action": "stay", "direction": None, "memory": "", "reasoning": "Error occurred"}
    
    def move(self, direction: str) -> Tuple[int, int]:
        """Move agent in specified direction (origin-centered coordinate system)"""
        x, y = self.position
        dx, dy = DIRECTION_MAP.get(direction, (0, 0))

        # Boundaries: -half_space_size to +half_space_size
        new_x = max(-self.half_space_size, min(self.half_space_size, x + dx))
        new_y = max(-self.half_space_size, min(self.half_space_size, y + dy))

        self.position = (new_x, new_y)
        self.total_moves += 1
        return self.position
    
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
    
    def update_state(self, places: Optional[List[PlaceConfig]] = None):
        """Update agent state based on current position"""
        if places is None:
            places = self.places
        
        place_at_position = get_place_at_position(self.position, places)
        self.in_place = place_at_position is not None
        self.current_place = place_at_position['name'] if place_at_position else None
        
        if self.in_place:
            self.steps_in_place += 1
        else:
            self.steps_outside_place += 1

