"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import math
import random
import logging
from typing import List, Tuple, Optional, Dict, TypedDict
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
    ):
        self.id = agent_id
        self.position = initial_position
        self.llm_client = llm_client
        self.communication_radius = communication_radius
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.gender = gender

        # Movement speed (cells per "move" action). Actual distance per step is
        # base + uniform(-variance, +variance), clamped to >= 1.
        self.movement_base_cells = max(1, int(movement_base_cells))
        self.movement_variance = max(0, int(movement_variance))

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

            rel_level = self.get_relationship(agent.id)
            rel_tag = f"{relationship_label(rel_level)} ({rel_level:.2f})"
            if include_position:
                direction = self._position_to_rough_direction(agent.position)
                nearby_info.append(
                    f"{name} ({age}, {agent.gender}, {occupation}) — {rel_tag} — {status}, roughly {direction}"
                )
            else:
                nearby_info.append(
                    f"{name} ({age}, {agent.gender}, {occupation}) — {rel_tag} — {status}"
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
            hx = place.get('half_size_x', place.get('half_size', 5))
            hy = place.get('half_size_y', place.get('half_size', 5))
            spec = get_place_type_spec(place_type)
            place_locations.append(
                f"{place['name']} ({place_type} — {spec['atmosphere']}): "
                f"center ({place['center_x']}, {place['center_y']}), "
                f"covers X {place['center_x'] - hx} to {place['center_x'] + hx}, "
                f"Y {place['center_y'] - hy} to {place['center_y'] + hy}, "
                f"capacity {place.get('capacity', '?')}"
            )
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
        """WHO YOU ARE block — persona name/age/occupation/background/speech + biases + goal.

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
        goal = p.get('current_goal', '') or self.current_goal
        biases = p.get('cognitive_biases', []) or []

        lines = [
            "=== WHO YOU ARE ===",
            f"Name: {name}",
            f"Age: {age}, Gender: {gender}, Occupation: {occupation}",
        ]
        if background:
            lines.append(f"Background: {background}")
        if speech:
            lines.append(f"Speech style: {speech}")
        if goal:
            lines.append(f"Current goal: {goal}")
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
        """Compose a TIME & CONTEXT block for the user prompt (feature 5).

        Empty string when simulation never set any time context — keeps the
        block out of the prompt for configs without time_patterns.
        """
        if not (self.current_time_str or self.current_context or self.current_goal):
            return ""
        lines = ["=== TIME & CONTEXT ==="]
        if self.current_time_str:
            lines.append(f"Current time (approx): {self.current_time_str}")
        if self.current_context:
            lines.append(f"Neighborhood mood: {self.current_context}")
        if self.current_goal:
            lines.append(f"What people around here are typically doing now: {self.current_goal}")
        return "\n".join(lines) + "\n"

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
        persona_section = self._build_persona_section()
        internal_state_section = self._build_internal_state_section()
        group_identities_section = self._build_group_identities_section()

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
        time_section = self._build_time_context_section()

        user_prompt = f"""{persona_section}
{internal_state_section}
{group_identities_section}{time_section}=== YOUR CURRENT STATE ===
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
=== NEARBY PEOPLE (you can communicate with these people) ===
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
- "stay": hold your position. Use this when observing, waiting in a place, continuing a conversation, or simply having no reason to move.
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
A typical step proceeds as follows. You receive your current state: where you are, whether you are in a place, which agents are nearby, the messages they sent, your past memory entries. You interpret this information in light of your own priorities, which you built up over prior steps. You then choose one action_type (walk_toward, walk_along, enter, stay, approach, or wander) and fill in its associated target/direction field. Along with that choice, you write a short memory entry that records your interpretation of the current situation and your next intended step; this memory is visible to you on the following step.

How memory flows across steps.
Each step appends one memory line to a rolling buffer. You see only the last few entries in the buffer (the size is controlled outside of the prompt). Entries are strings, prefixed with 'Step N:' by the system, followed by whatever Japanese text you wrote in the 'memory' field. Because the buffer is small, writing redundant or vague memory entries wastes slots. Write one informative sentence per step.

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
    "memory": "left_barに向かって歩き始めた。数分で到着しそう。",
    "reasoning": "静かに一杯飲みたいので左側のバーに向かう。"
}}

Example B (staying in a place during a conversation):
{{
    "action_type": "stay",
    "target_place": null,
    "target_agent": null,
    "direction": null,
    "memory": "right_barで美咲さんと話している。もう少しここにいる。",
    "reasoning": "会話が続いているので留まる。"
}}

Example C (stepping into a place you're right next to):
{{
    "action_type": "enter",
    "target_place": "right_bar",
    "target_agent": null,
    "direction": null,
    "memory": "right_barの入口に着いた。入って様子を見る。",
    "reasoning": "目の前がright_barなので入店する。"
}}

Example D (closing the distance to a specific person):
{{
    "action_type": "approach",
    "target_place": null,
    "target_agent": "鈴木美咲",
    "direction": null,
    "memory": "美咲さんに近づいて声をかけたい。",
    "reasoning": "顔見知りの美咲さんが近くにいるので合流する。"
}}

Example E (wandering without a clear destination):
{{
    "action_type": "wander",
    "target_place": null,
    "target_agent": null,
    "direction": null,
    "memory": "特に予定なし。少しぶらついて様子を見る。",
    "reasoning": "やることがないので適当にぶらぶらする。"
}}

Example F (evacuating from fire by walking along a cardinal direction):
{{
    "action_type": "walk_along",
    "target_place": null,
    "target_agent": null,
    "direction": "down",
    "memory": "火災が北東にあるので南方向へ離れる。",
    "reasoning": "特定の目的地ではなく、とにかく火から遠ざかりたい。"
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
"""

        persona_name = self.persona.get('name', f"Person {self.id}")
        persona_section = self._build_persona_section()
        internal_state_section = self._build_internal_state_section()
        group_identities_section = self._build_group_identities_section()

        nearby_text = self._build_nearby_agents_context(nearby_agents)
        nearby_places_text = self._build_nearby_places_context()
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
        time_section = self._build_time_context_section()

        user_prompt = f"""{persona_section}
{internal_state_section}
{group_identities_section}{time_section}=== YOUR CURRENT STATE ===
Position: ({self.position[0]}, {self.position[1]})
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
Behavior layer: {self.behavior_layer} (transit = walking through the district, dwelling = spending time in a place, interacting = with people around you)
{place_section_text}
{fire_section}
=== NEARBY PLACES ===
{nearby_places_text}

=== NEARBY PEOPLE ===
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
        cached["reasoning"] = "(continuing previous intent)"
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
        fire_info: Optional[List[Dict]] = None
    ) -> ActionDecision:
        """Use LLM to decide next action (with position information and message content)"""
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

        if action_type == "walk_along":
            direction = decision.get("direction")
            dx, dy = DIRECTION_MAP.get(direction, (0, 0))
            if dx == 0 and dy == 0:
                return self.position
            new_pos = self._clamp_to_field(x + dx * distance, y + dy * distance)

        elif action_type in ("walk_toward", "enter"):
            place = self._find_place_by_name(decision.get("target_place"))
            if place is None:
                return self.position
            target_x, target_y = place['center_x'], place['center_y']
            # "enter" snaps into the place when within one step, so the agent
            # doesn't overshoot the box.
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
            new_pos = self._step_toward(target.position[0], target.position[1], distance)

        elif action_type == "wander":
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * distance
            dy = math.sin(angle) * distance
            new_pos = self._clamp_to_field(int(round(x + dx)), int(round(y + dy)))

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

