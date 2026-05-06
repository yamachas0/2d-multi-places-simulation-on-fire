"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import os
import random
import threading
import yaml
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from agent import Agent
from claude_client import ClaudeClient
from llm_client_factory import create_llm_client
from navigation import Navigator
from place_types import get_place_type_spec
from utils import (
    is_position_in_place,
    get_place_at_position,
    PlaceConfig,
    FireConfig,
    generate_random_persona,
)

logger = logging.getLogger(__name__)

# Constants
MAX_POSITION_ATTEMPTS = 1000
LOG_INTERVAL = 10


class Simulation:
    """Main simulation class for LLM-based agent in 2D worlds with multiple places."""
    
    def __init__(self, config_path: str = "config.yaml", output_dir: Optional[str] = None, seed: Optional[int] = None):
        """Initialize simulation from config file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Handle shinagawa_config.yaml shorthand (`- axis: ".." ; center_x: ..`)
        # the same way tools/bundle_viewer.py does.
        import re as _re
        _SC = _re.compile(r'^(\s*-\s)(.+?\s;\s.+)$', _re.MULTILINE)
        def _norm(m):
            prefix, body = m.group(1), m.group(2).strip()
            fields = [f.strip() for f in body.split(';') if f.strip()]
            return f"{prefix}{{ {', '.join(fields)} }}"
        self.config = yaml.safe_load(_SC.sub(_norm, raw))

        # Output directory for logs
        self.output_dir = output_dir
        # JSONL writers are hit from multiple threads in Phase 1/3, so every
        # append goes through a dedicated lock. One lock per sink is enough —
        # sinks are disjoint files, so contention stays minimal.
        self._log_locks: Dict[str, threading.Lock] = {
            "messages": threading.Lock(),
            "memory_reasoning": threading.Lock(),
            "should_speak": threading.Lock(),
            "relationships_timeline": threading.Lock(),
        }

        # Reproducibility: seed both `random` and numpy. Falls back to
        # simulation.seed in config if not provided on the CLI.
        if seed is None:
            seed = self.config.get('simulation', {}).get('seed')
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"Random seed set: {seed}")
        self.seed = seed

        # Simulation parameters
        sim_config = self.config['simulation']
        self.duration = sim_config['duration']
        self.half_space_size = sim_config['half_space_size']
        self.half_place_size = sim_config.get('half_place_size', 5)

        # Cost-reduction flags for fixed-scene sims (classroom AB etc).
        # minimal_prompt_mode: agent.py uses _create_*_prompts_minimal (drops
        #   WORLD STRUCTURE / PLACE LOCATIONS / fire / coordinates).
        # skip_decision_prompt: Phase 3 (decide_action) is replaced with a
        #   stub; memory is captured in the message JSON instead. Halves LLM
        #   call count. Movement and per-step 2D state stop being meaningful.
        self.minimal_prompt_mode = bool(sim_config.get('minimal_prompt_mode', False))
        self.skip_decision_prompt = bool(sim_config.get('skip_decision_prompt', False))
        # moltbook風 グローバルチャンネルモード: 教室シーン等で全 agent が同じチャンネルを
        # 読み、self-decision で発話するか黙るかを選ぶ。Phase A 限定で実験的に試す。
        self.global_channel_mode = bool(sim_config.get('global_channel', False))
        # moltbook風 chat session thread化モード: 各 agent が独立した Gemini chat session
        # を保持して履歴を session 内で蓄積する experiment 経路。
        self.chat_thread_mode = bool(sim_config.get('chat_thread_mode', False))
        # Scene phrase for the minimal prompt header (e.g. "a small fixed indoor
        # scene (a classroom)" / "a small fixed outdoor scene (a park)"). Built
        # by tools/build_classroom_ab_config.py from SCENE_PROFILES.
        self.scene_phrase = str(sim_config.get('scene_phrase') or "a small fixed scene")

        # Phase A: context injection (場所の現場知覚情報を system_prompt に注入)。
        # Phase B 以降では agent が場所に近接したときのみロードに切り替える設計だが、
        # 現状はシミュ起動時に file を全文読み込み Agent に渡す。
        ctx_cfg = (sim_config.get('context_injection') or {}) if isinstance(sim_config.get('context_injection'), dict) else {}
        self.context_injection_text = ""
        if ctx_cfg.get('enabled'):
            ctx_path = ctx_cfg.get('file')
            if ctx_path:
                try:
                    self.context_injection_text = Path(ctx_path).read_text(encoding='utf-8')
                    logger.info(f"Context injection: loaded {ctx_path} ({len(self.context_injection_text)} chars)")
                except Exception as e:
                    logger.warning(f"Context injection: failed to load {ctx_path}: {e}")

        # Phase B 用フラグ: perceive (Layer 1/2) 注入と身体感覚記録
        self.phase_b_perceive_enabled = bool(sim_config.get('phase_b_perceive_enabled', False))
        self.phase_b_body_log_interval = int(sim_config.get('phase_b_body_log_interval', 10))
        if self.phase_b_perceive_enabled:
            logger.info(f"Phase B perceive injection enabled (body_log_interval={self.phase_b_body_log_interval})")
        
        # Agent parameters
        agent_config = self.config['agents']
        self.num_agents = agent_config['num_agents']
        self.communication_radius = agent_config['communication_radius']
        self.memory_limit = agent_config.get('memory_limit', 20)
        self.memory_size = agent_config.get('memory_size', 5)
        self.message_history_limit = agent_config.get('message_history_limit', 10)
        self.message_context_size = agent_config.get('message_context_size', 3)
        self.skip_probability = agent_config.get('skip_probability', 0.0)
        self.parallel_workers = agent_config.get('parallel_workers', 1)

        # Walking speed (feature 1 / urban scale).
        movement_cfg = agent_config.get('movement_speed', {}) or {}
        self.movement_base_cells = int(movement_cfg.get('base_cells_per_step', 1))
        self.movement_variance = int(movement_cfg.get('variance', 0))

        # Time scale (feature 1 / urban scale, extended in feature 5).
        # `patterns` (optional): list of hour-bucket dicts driving station
        # spawn/despawn probabilities and neighborhood mood strings.
        time_cfg = sim_config.get('time_scale', {}) or {}
        self.step_duration_minutes = int(time_cfg.get('step_duration_minutes', 1))
        self.start_time_str = str(time_cfg.get('start_time', '08:00'))
        self.time_patterns: List[Dict] = list(time_cfg.get('patterns', []) or [])
        self._initialize_time()

        # Dynamic agent pool (feature 5). Spawns respect max_agents; despawns
        # remove agents that are currently inside a station.
        self.max_agents = int(agent_config.get('max_agents', max(self.num_agents * 2, self.num_agents + 10)))
        self.spawn_enabled = bool(agent_config.get('spawn_enabled', bool(self.time_patterns)))
        self.next_agent_id = self.num_agents
        
        # Place parameters - support multiple places
        if 'places' not in self.config:
            raise ValueError("No 'places' configuration found in config file. Please use 'places:' key.")
        
        self.places = self.config['places']
        
        # Validate places configuration
        if not isinstance(self.places, list):
            raise ValueError("'places' must be a list of place configurations.")
        
        if len(self.places) == 0:
            raise ValueError("At least one place must be configured in 'places'.")
        
        # Validate each place configuration. A place must specify its footprint
        # either as `half_size` (square shorthand) or `half_size_x` + `half_size_y`
        # (rectangle, feature 3).
        base_required = ['name', 'type', 'center_x', 'center_y']
        for i, place in enumerate(self.places):
            if not isinstance(place, dict):
                raise ValueError(f"Place at index {i} must be a dictionary.")

            for field in base_required:
                if field not in place:
                    raise ValueError(f"Place at index {i} is missing required field: '{field}'")

            has_square = 'half_size' in place
            has_rect = 'half_size_x' in place and 'half_size_y' in place
            if not (has_square or has_rect):
                raise ValueError(
                    f"Place at index {i} ('{place.get('name')}') must specify either "
                    f"'half_size' or both 'half_size_x' and 'half_size_y'."
                )
            # Normalise: always expose both rectangular half-sizes so downstream
            # code can rely on place['half_size_x'] / place['half_size_y'].
            if not has_rect:
                place['half_size_x'] = place['half_size']
                place['half_size_y'] = place['half_size']
        
        place_names = [place['name'] for place in self.places]
        place_types = [place['type'] for place in self.places]
        logger.info(f"Initialized {len(self.places)} place(s): {place_names} (types: {place_types})")

        # Passability layer. Builds a walkable mask from scene_3d (roads,
        # decks, stairs) + place interiors. When scene_3d is absent, falls
        # back to "everything walkable" so legacy configs still work.
        self.navigator = Navigator(self.config, self.half_space_size)
        
        # Fire parameters (multiple fires supported)
        fires_config = self.config.get('fires', [])
        self.fire_configs: List[Dict] = []
        for i, fc in enumerate(fires_config):
            config_entry = {
                'name': fc.get('name', f'fire_{i}'),
                'start_step': fc['start_step'],
                'intensity': fc['intensity'],
                'radius': fc['radius'],
            }
            if 'center_x' in fc and 'center_y' in fc:
                config_entry['center_x'] = fc['center_x']
                config_entry['center_y'] = fc['center_y']
            self.fire_configs.append(config_entry)
            pos_info = f"({fc['center_x']}, {fc['center_y']})" if 'center_x' in fc else "random"
            logger.info(
                f"Fire '{config_entry['name']}' configured: step={fc['start_step']}, "
                f"intensity={fc['intensity']}, radius={fc['radius']}, position={pos_info}"
            )
        self.fire_states: List[Dict] = []  # Active fires

        # Generic events (feature 4.5). Supports type:transit_disruption (JR運休)
        # and type:last_train (終電後 — same awareness plumbing, different prompt
        # context). Fires stay on the legacy `fires:` key; these live under
        # `events:` so both can run in parallel without touching each other.
        self.event_configs: List[Dict] = []
        _SUPPORTED_EVENT_TYPES = ('transit_disruption', 'last_train')
        for i, ec in enumerate(self.config.get('events', []) or []):
            if ec.get('type') not in _SUPPORTED_EVENT_TYPES:
                logger.warning(
                    f"Event #{i} '{ec.get('name')}' has unsupported type "
                    f"'{ec.get('type')}' — skipping."
                )
                continue
            effects = ec.get('effects', {}) or {}
            self.event_configs.append({
                'name': ec.get('name', f'event_{i}'),
                'type': ec.get('type'),
                'start_step': int(ec['start_step']),
                'end_step': int(ec.get('end_step', 10**9)),
                'affected_place': ec.get('affected_place'),
                'description': ec.get('description', ''),
                'block_spawn_at': list(effects.get('block_spawn_at', []) or []),
                'boost_spawn_at': list(effects.get('boost_spawn_at', []) or []),
                'notify_radius': float(effects.get('notify_agents_within_radius', 0) or 0),
                'salience_boost_targets': list(ec.get('salience_boost_targets', []) or []),
            })
            logger.info(
                f"Event '{self.event_configs[-1]['name']}' configured: "
                f"type={ec.get('type')}, start={ec['start_step']}, "
                f"end={ec.get('end_step','inf')}, affected={ec.get('affected_place')}"
            )
        # Active events (mirrors fire_states). Items have the same keys as
        # event_configs plus 'active' bool and 'place_position' resolved from
        # the affected_place.
        self.event_states: List[Dict] = []

        # Keywords that indicate an event in a received message — used by the
        # 2nd propagation path (conversation-based awareness). Defaults to the
        # JR-disruption set; configs can override via `events_keywords:` at top
        # level to swap in scenario-specific vocab (e.g. 終電/新幹線).
        default_kw = [
            '運休', '止まって', '止まった', '電車', '事故', '山手線',
            '京浜東北', '地下鉄', '歩いて', '振替', '運転見合わせ',
        ]
        self._transit_keywords = list(self.config.get('events_keywords') or default_kw)
        # Extra log sink for event awareness snapshots.
        self._log_locks['event_awareness'] = threading.Lock()
        # Phase 2.5: per-transition log of awareness propagation (direct/conversation/notification).
        self._log_locks['awareness_propagation'] = threading.Lock()

        # LLM parameters — factory selects provider (anthropic/openai/google)
        llm_config = self.config['llm']
        self.llm_client = create_llm_client(llm_config)
        
        # Initialize agents
        self.agents: List[Agent] = []
        self.step = 0
        self.history: List[Dict] = []
        # Feature 6: per-step message edges as (from_id, to_id) tuples,
        # recorded during Phase 2 and consumed by the visualizer.
        self.last_step_messages: List[Tuple[int, int]] = []

        # Feature 5 (Canvas viewer): per-step snapshot timeline, flushed once
        # at simulation end into simulation_data.json. _viewer_step_convs is a
        # rolling buffer for the current step's conversations, reset every
        # step after snapshot capture.
        self._viewer_timeline: List[Dict] = []
        self._viewer_step_convs: List[Dict] = []


        # Statistics - track per place
        self.stats = {
            'place_occupancy': [],  # Overall occupancy (all places combined)
            'agents_in_place': [],  # Total agents in any place
            'agents_outside_place': [],
            'communication_events': [],
            'places': {place['name']: {
                'occupancy': [],
                'agents_in_place': []
            } for place in self.places},
            'agents_in_fire_radius': [],  # Total agents in any fire radius
        }
        
    def _initialize_time(self) -> None:
        """Parse start_time (HH:MM) into a datetime anchor for wall-clock tracking."""
        try:
            hh, mm = self.start_time_str.split(":")
            start_time = datetime(2000, 1, 1, int(hh), int(mm))
        except Exception as e:
            logger.warning(f"Invalid start_time '{self.start_time_str}', defaulting to 08:00. ({e})")
            start_time = datetime(2000, 1, 1, 8, 0)
        self.start_datetime = start_time
        self.current_datetime = start_time

    def _advance_time(self) -> None:
        """Advance wall-clock by step_duration_minutes."""
        self.current_datetime = self.current_datetime + timedelta(minutes=self.step_duration_minutes)

    def _current_time_str(self) -> str:
        return self.current_datetime.strftime("%H:%M")

    def _get_time_pattern(self) -> Optional[Dict]:
        """Return the time_patterns entry whose `hours` list includes current hour, if any."""
        if not self.time_patterns:
            return None
        hour = self.current_datetime.hour
        for pattern in self.time_patterns:
            hours = pattern.get('hours', []) or []
            if hour in hours:
                return pattern
        return None

    def _find_spawn_places(self) -> List[Dict]:
        """Places flagged as is_spawn_point by their place-type (e.g. stations)."""
        spawn_places: List[Dict] = []
        for place in self.places:
            spec = get_place_type_spec(place.get('type', ''))
            if spec.get('is_spawn_point'):
                spawn_places.append(place)
        return spawn_places

    def _spawn_agent_at(self, station: Dict) -> Optional[Agent]:
        """Create a new agent positioned at the station's center."""
        if len(self.agents) >= self.max_agents:
            return None
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        gender = random.choice(["male", "female"])
        persona = generate_random_persona(agent_id, gender)
        position = (int(station['center_x']), int(station['center_y']))
        new_agent = Agent(
            agent_id=agent_id,
            initial_position=position,
            llm_client=self.llm_client,
            communication_radius=self.communication_radius,
            half_space_size=self.half_space_size,
            places=self.places,
            num_agents=len(self.agents) + 1,
            gender=gender,
            memory_limit=self.memory_limit,
            memory_size=self.memory_size,
            message_history_limit=self.message_history_limit,
            message_context_size=self.message_context_size,
            persona=persona,
            movement_base_cells=self.movement_base_cells,
            movement_variance=self.movement_variance,
            navigator=self.navigator,
            minimal_prompt=self.minimal_prompt_mode,
            skip_decision=self.skip_decision_prompt,
            scene_phrase=self.scene_phrase,
            injected_context=self.context_injection_text,
            global_channel=self.global_channel_mode,
            chat_thread_mode=self.chat_thread_mode,
        )
        new_agent.update_state(self.places)
        self.agents.append(new_agent)
        logger.info(
            f"Step {self.step} {self._current_time_str()}: SPAWN agent {agent_id} "
            f"({persona.get('name', '?')}) at {station['name']}"
        )
        return new_agent

    def _despawn_one_at_station(self) -> Optional[Agent]:
        """Pick one agent currently inside any spawn-point place and remove them."""
        spawn_names = {p['name'] for p in self._find_spawn_places()}
        if not spawn_names:
            return None
        candidates = [a for a in self.agents if a.current_place in spawn_names]
        if not candidates:
            return None
        victim = random.choice(candidates)
        self.agents.remove(victim)
        logger.info(
            f"Step {self.step} {self._current_time_str()}: DESPAWN agent {victim.id} "
            f"({victim.persona.get('name', '?')}) from {victim.current_place}"
        )
        return victim

    def _handle_agent_spawning(self) -> None:
        """Apply time-pattern spawn/despawn probabilities for this step.
        Active transit-disruption events can block specific spawn places and
        boost others via a per-place multiplier (feature 4.5)."""
        if not self.spawn_enabled:
            return
        pattern = self._get_time_pattern()
        if pattern is None:
            return
        spawn_places = self._find_spawn_places()
        if not spawn_places:
            return

        # Apply transit-event spawn effects.
        blocked: set = set()
        multipliers: Dict[str, float] = {}
        for ev in self._active_transit_events():
            for name in ev.get('block_spawn_at', []):
                blocked.add(name)
            for entry in ev.get('boost_spawn_at', []):
                name = entry.get('place') if isinstance(entry, dict) else None
                mult = float(entry.get('multiplier', 1.0)) if isinstance(entry, dict) else 1.0
                if name:
                    multipliers[name] = multipliers.get(name, 1.0) * mult

        allowed_places = [p for p in spawn_places if p['name'] not in blocked]
        if not allowed_places:
            # Everything is blocked — skip entering but still allow exits.
            allowed_places = []

        enter_p = float(pattern.get('enter_per_step', 0.0))
        exit_p = float(pattern.get('exit_per_step', 0.0))

        if allowed_places and enter_p > 0 and len(self.agents) < self.max_agents:
            weights = [multipliers.get(p['name'], 1.0) for p in allowed_places]
            effective_enter_p = min(1.0, enter_p * (sum(weights) / len(weights)))
            if random.random() < effective_enter_p:
                station = random.choices(allowed_places, weights=weights, k=1)[0]
                self._spawn_agent_at(station)

        if exit_p > 0 and random.random() < exit_p:
            self._despawn_one_at_station()

    def _apply_time_context_to_agents(self) -> None:
        """Push current time string and neighborhood mood into each agent before the LLM phase."""
        time_str = self._current_time_str()
        pattern = self._get_time_pattern()
        mood = pattern.get('neighborhood_mood', '') if pattern else ''
        default_goal = pattern.get('default_goal', '') if pattern else ''
        for agent in self.agents:
            agent.current_time_str = time_str
            agent.current_context = mood
            # Only override current_goal when pattern provides one; persona-level
            # goals stay untouched otherwise.
            if default_goal:
                agent.current_goal = default_goal

    def _is_position_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None

    def _log_message(
        self,
        from_agent: Agent,
        to_agent: Agent,
        message: str,
        reasoning: str = ""
    ) -> None:
        """Log a message to messages.jsonl file (feature 4)."""
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        messages_file = os.path.join(self.output_dir, "messages.jsonl")
        relationship = round(from_agent.get_relationship(to_agent.id), 2)
        record = {
            "step": self.step,
            "time": self._current_time_str(),
            "from": from_agent.id,
            "from_name": from_agent.persona.get('name', f"Agent {from_agent.id}"),
            "to": to_agent.id,
            "to_name": to_agent.persona.get('name', f"Agent {to_agent.id}"),
            "relationship": relationship,
            "message": message,
            "reasoning": reasoning,
        }

        with self._log_locks["messages"]:
            with open(messages_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Feature 5: also buffer for the per-step viewer snapshot.
        self._viewer_step_convs.append(record)

    def _log_memory_reasoning_batch(
        self,
        records: List[Dict]
    ) -> None:
        """Log memory and reasoning records in batch to memory_reasoning.jsonl file
        
        This is more efficient than writing one record at a time, especially
        when logging for all agents in each step.
        """
        if not self.output_dir or not records:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        memory_reasoning_file = os.path.join(self.output_dir, "memory_reasoning.jsonl")

        # Write all records at once (buffered I/O)
        with self._log_locks["memory_reasoning"]:
            with open(memory_reasoning_file, 'a', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _generate_random_position(self) -> Tuple[int, int]:
        """Generate a random position within the space (origin-centered coordinate system)"""
        return (
            random.randint(-self.half_space_size, self.half_space_size),
            random.randint(-self.half_space_size, self.half_space_size)
        )
    
    def _generate_initial_positions(
        self,
        avoid_places: bool = True,
        personas_by_id: Optional[Dict[int, Dict]] = None,
    ) -> List[Tuple[int, int]]:
        """Generate initial positions for agents.

        In constrained-passability mode, every agent spawns at a random
        walkable cell (road/deck/place interior). The `initial_place`
        persona hint and `avoid_places` flag are honoured only when
        the navigator is unconstrained (legacy mode).
        """
        personas_by_id = personas_by_id or {}
        place_by_name = {p['name']: p for p in self.places}

        positions: List[Optional[Tuple[int, int]]] = [None] * self.num_agents
        used_positions: Set[Tuple[int, int]] = set()

        constrained = getattr(self, 'navigator', None) and self.navigator.constrained

        if constrained:
            # In constrained mode: honour persona.initial_place when it points
            # to a walkable place (pedestrian_street / plaza / etc). This lets
            # scenarios pin the starting location (e.g. 全員東西自由通路). When
            # no initial_place is set, fall back to a random walkable cell.
            rng = random.Random(random.random())
            for i in range(self.num_agents):
                persona = personas_by_id.get(i)
                # 優先: persona.initial_position が指定されていればそれを使う
                # (host を最寄り道路の歩道側に固定配置するなどの用途)。
                explicit_pos = (persona or {}).get('initial_position')
                if explicit_pos and isinstance(explicit_pos, (list, tuple)) and len(explicit_pos) == 2:
                    try:
                        ep = (int(round(explicit_pos[0])), int(round(explicit_pos[1])))
                        if ep not in used_positions:
                            positions[i] = ep
                            used_positions.add(ep)
                            continue
                    except (TypeError, ValueError):
                        pass
                place_name = (persona or {}).get('initial_place')
                pos = None
                if place_name:
                    place = place_by_name.get(place_name)
                    if place is None:
                        logger.warning(
                            f"Agent {i}: initial_place '{place_name}' not found — "
                            "falling back to random walkable cell."
                        )
                    else:
                        pos = self._sample_walkable_cell_in_place(
                            place, used_positions
                        )
                        if pos is None:
                            logger.warning(
                                f"Agent {i}: '{place_name}' has no free walkable "
                                "cell — falling back to random walkable cell."
                            )
                if pos is None:
                    pos = self.navigator.sample_walkable_cell(rng, used_positions)
                if pos is None:
                    logger.warning(
                        f"Agent {i}: no walkable cell available — using origin."
                    )
                    pos = (0, 0)
                positions[i] = pos
                used_positions.add(pos)
            return [p for p in positions if p is not None]

        # Legacy (unconstrained) mode: original behaviour.
        for i in range(self.num_agents):
            persona = personas_by_id.get(i)
            if not persona:
                continue
            # 優先: persona.initial_position
            explicit_pos = persona.get('initial_position')
            if explicit_pos and isinstance(explicit_pos, (list, tuple)) and len(explicit_pos) == 2:
                try:
                    ep = (int(round(explicit_pos[0])), int(round(explicit_pos[1])))
                    if ep not in used_positions:
                        positions[i] = ep
                        used_positions.add(ep)
                        continue
                except (TypeError, ValueError):
                    pass
            place_name = persona.get('initial_place')
            if not place_name:
                continue
            place = place_by_name.get(place_name)
            if not place:
                logger.warning(
                    f"Agent {i}: initial_place '{place_name}' not found — "
                    "falling back to random spawn"
                )
                continue
            pos = self._sample_position_in_place(place, used_positions)
            if pos is None:
                logger.warning(
                    f"Agent {i}: could not spawn inside '{place_name}' — "
                    "falling back to random spawn"
                )
                continue
            positions[i] = pos
            used_positions.add(pos)

        remaining = [i for i, p in enumerate(positions) if p is None]
        attempts = 0
        while remaining and attempts < MAX_POSITION_ATTEMPTS:
            position = self._generate_random_position()
            if position in used_positions:
                attempts += 1
                continue
            if avoid_places and self._is_position_in_place(position):
                attempts += 1
                continue
            positions[remaining.pop(0)] = position
            used_positions.add(position)
            attempts += 1

        if remaining:
            logger.warning(
                f"Could only generate positions for "
                f"{self.num_agents - len(remaining)} / {self.num_agents} agents "
                "while avoiding places. Filling the rest anywhere."
            )
            while remaining:
                position = self._generate_random_position()
                if position in used_positions:
                    continue
                positions[remaining.pop(0)] = position
                used_positions.add(position)

        return [p for p in positions if p is not None]

    def _sample_position_in_place(
        self,
        place: Dict,
        used_positions: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Return a random integer cell strictly inside the given place, or None."""
        cx = int(place.get('center_x', 0))
        cy = int(place.get('center_y', 0))
        hx = int(place.get('half_size_x', place.get('half_size', self.half_place_size)))
        hy = int(place.get('half_size_y', place.get('half_size', self.half_place_size)))
        for _ in range(200):
            x = random.randint(cx - hx, cx + hx)
            y = random.randint(cy - hy, cy + hy)
            pos = (x, y)
            if pos not in used_positions:
                return pos
        return None

    def _sample_walkable_cell_in_place(
        self,
        place: Dict,
        used_positions: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Constrained-mode variant: sample a cell inside the place bbox that
        is also walkable per the navigator mask. Used when personas pin
        initial_place and we still must respect road/passage geometry."""
        nav = getattr(self, 'navigator', None)
        cx = int(place.get('center_x', 0))
        cy = int(place.get('center_y', 0))
        hx = int(place.get('half_size_x', place.get('half_size', self.half_place_size)))
        hy = int(place.get('half_size_y', place.get('half_size', self.half_place_size)))
        candidates = []
        for x in range(cx - hx, cx + hx + 1):
            for y in range(cy - hy, cy + hy + 1):
                pos = (x, y)
                if pos in used_positions:
                    continue
                if nav is not None and not nav.is_walkable(x, y):
                    continue
                candidates.append(pos)
        if not candidates:
            return None
        return random.choice(candidates)
    
    def _initialize_relationships(
        self, personas_by_id: Dict[int, Dict]
    ) -> Dict[int, Dict[int, float]]:
        """Build per-agent relationship dicts from config (feature 2).

        For each persona:
        - start with its declared `initial_relationships`
        - boost anyone listed under a `social_identities.member_ids` to at least 0.5
          (same-group coworkers/family), so configs don't have to repeat
          both the identity and the numeric relationship.
        Agents without a persona entry (or without these fields) get an empty
        dict → everyone defaults to the 0.05 stranger baseline.
        """
        out: Dict[int, Dict[int, float]] = {}
        for pid, persona in personas_by_id.items():
            rel: Dict[int, float] = {}
            for k, v in (persona.get('initial_relationships') or {}).items():
                rel[int(k)] = float(v)
            for identity in (persona.get('social_identities') or []):
                for member_id in (identity.get('member_ids') or []):
                    mid = int(member_id)
                    rel[mid] = max(0.5, rel.get(mid, 0.0))
            rel.pop(pid, None)  # no self-relationship
            out[pid] = rel
        return out

    def initialize_agents(self):
        """Initialize agents at random positions, attaching a persona to each."""
        logger.info(f"Initializing {self.num_agents} agents...")

        personas_config = self.config.get('agents', {}).get('personas', []) or []
        personas_by_id = {p['id']: p for p in personas_config if 'id' in p}

        positions = self._generate_initial_positions(
            avoid_places=True, personas_by_id=personas_by_id,
        )

        relationships_by_id = self._initialize_relationships(personas_by_id)

        for i in range(self.num_agents):
            if i in personas_by_id:
                persona = dict(personas_by_id[i])
                persona.setdefault('id', i)
                gender = persona.get('gender') or random.choice(["male", "female"])
                persona['gender'] = gender
            else:
                gender = random.choice(["male", "female"])
                persona = generate_random_persona(i, gender)

            agent = Agent(
                agent_id=i,
                initial_position=positions[i],
                llm_client=self.llm_client,
                communication_radius=self.communication_radius,
                half_space_size=self.half_space_size,
                places=self.places,
                num_agents=self.num_agents,
                gender=gender,
                memory_limit=self.memory_limit,
                memory_size=self.memory_size,
                message_history_limit=self.message_history_limit,
                message_context_size=self.message_context_size,
                persona=persona,
                movement_base_cells=self.movement_base_cells,
                movement_variance=self.movement_variance,
                initial_relationships=relationships_by_id.get(i, {}),
                navigator=self.navigator,
                minimal_prompt=self.minimal_prompt_mode,
                skip_decision=self.skip_decision_prompt,
                scene_phrase=self.scene_phrase,
                injected_context=self.context_injection_text,
                global_channel=self.global_channel_mode,
                chat_thread_mode=self.chat_thread_mode,
            )
            agent.update_state()
            self.agents.append(agent)
            logger.info(
                f"Agent {i}: {persona.get('name', '?')} "
                f"({persona.get('age', '?')}, {persona.get('occupation', '?')})"
            )

        logger.info("Agents initialized successfully")
    
    def get_agents_in_place(self, place_name: Optional[str] = None) -> List[Agent]:
        """Get list of agents currently in a specific place or any place"""
        if place_name:
            return [agent for agent in self.agents if agent.current_place == place_name]
        return [agent for agent in self.agents if agent.in_place]
    
    def get_place_status(self, place_name: Optional[str] = None) -> Dict:
        """Get current place status for a specific place or overall status"""
        if place_name:
            # Get status for a specific place
            place_config = next((p for p in self.places if p['name'] == place_name), None)
            if not place_config:
                raise ValueError(f"Place '{place_name}' not found")
            
            agents_in_place = len(self.get_agents_in_place(place_name))
            capacity = place_config.get('capacity')
            occupancy_rate = (agents_in_place / capacity) if capacity else None

            return {
                "place_name": place_name,
                "agents_in_place": agents_in_place,
                "capacity": capacity,
                "occupancy_rate": occupancy_rate,
            }
        else:
            # Get overall status (all places combined). Denominator uses the
            # current roster so spawn/despawn doesn't break the ratio.
            agents_in_place = len(self.get_agents_in_place())
            total_agents = max(1, len(self.agents))
            occupancy_rate = agents_in_place / total_agents
            
            # Get per-place status (optimized: calculate directly instead of recursive calls)
            place_statuses = {}
            for place in self.places:
                place_agents = len(self.get_agents_in_place(place['name']))
                place_capacity = place.get('capacity')
                place_occupancy_rate = (
                    place_agents / place_capacity if place_capacity else None
                )

                place_statuses[place['name']] = {
                    "place_name": place['name'],
                    "agents_in_place": place_agents,
                    "capacity": place_capacity,
                    "occupancy_rate": place_occupancy_rate,
                }
            
            return {
                "agents_in_place": agents_in_place,
                "occupancy_rate": occupancy_rate,
                "places": place_statuses
            }
    
    def get_fire_info_for_agent(self, agent: Agent) -> Optional[List[Dict]]:
        """Return list of perceived fire info dicts, or None if no fires perceived.

        Implements Model B: only agents within each fire's radius get that fire's data.
        Agents outside all radii must learn about fires through messages.
        """
        if not self.fire_states:
            return None

        perceived = []
        for fire in self.fire_states:
            if not fire.get('active'):
                continue
            fire_pos = fire['position']
            distance = agent.distance_to(fire_pos)
            if distance <= fire['radius']:
                perceived.append({
                    'name': fire['name'],
                    'fire_position': fire_pos,
                    'intensity': fire['intensity'],
                    'radius': fire['radius'],
                    'agent_distance': round(distance, 2),
                })
        return perceived if perceived else None

    def _update_relationships(self) -> None:
        """Feature 2 dynamic update: conversation +, proximity +, slow decay.

        Parameters are intentionally small so the graph evolves over tens of
        steps, not within a single conversation. Floor is 0.05 (stranger).
        """
        PER_CONVERSATION = 0.02
        PER_PROXIMITY = 0.005
        DECAY = 0.001
        FLOOR = 0.05

        # Build a name→agent lookup once for the symmetric bumps below.
        agents_by_id = {a.id: a for a in self.agents}

        # Bidirectional conversation bump: for every sender→receiver edge this
        # step, both sides of the pair move closer. Using last_step_messages
        # keeps this aligned with Phase 2 broadcast edges.
        for from_id, to_id in self.last_step_messages:
            for left_id, right_id in ((from_id, to_id), (to_id, from_id)):
                left = agents_by_id.get(left_id)
                if left is None or left_id == right_id:
                    continue
                current = left.relationships.get(right_id, FLOOR)
                left.relationships[right_id] = min(1.0, current + PER_CONVERSATION)

        for agent in self.agents:

            # Anyone co-located in the same place — micro-bump.
            if agent.in_place and agent.current_place:
                for other in self.agents:
                    if other.id == agent.id:
                        continue
                    if other.in_place and other.current_place == agent.current_place:
                        current = agent.relationships.get(other.id, FLOOR)
                        agent.relationships[other.id] = min(1.0, current + PER_PROXIMITY)

            # Decay anyone already in the graph.
            for other_id in list(agent.relationships.keys()):
                decayed = agent.relationships[other_id] - DECAY
                agent.relationships[other_id] = max(FLOOR, decayed)

    def _update_internal_states(self) -> None:
        """Advance each agent's internal_state (energy/hunger/social_fatigue)."""
        for agent in self.agents:
            nearby_count = len(agent.get_nearby_agents(self.agents))
            place_type: Optional[str] = None
            if agent.in_place and agent.current_place:
                p = next((pl for pl in self.places if pl['name'] == agent.current_place), None)
                if p:
                    place_type = p.get('type')
            agent.update_internal_state(nearby_count, place_type)

    def _infer_talkativeness(self, agent: Agent) -> float:
        """Fallback talkativeness from speech_style when persona lacks the field."""
        t = agent.persona.get('talkativeness')
        if t is not None:
            return float(t)
        style = str(agent.persona.get('speech_style', ''))
        if '陽気' in style or 'カジュアル' in style:
            return 0.4
        if '人見知り' in style or '無愛想' in style:
            return 0.1
        if '丁寧' in style or '堅い' in style:
            return 0.2
        return 0.25

    def _fires_near(self, agent: Agent, multiplier: float = 2.0) -> bool:
        """True if any active fire is within `multiplier × radius` of the agent."""
        for fire in self.fire_states:
            if not fire.get('active'):
                continue
            dist = agent.distance_to(fire['position'])
            if dist <= fire['radius'] * multiplier:
                return True
        return False

    # ---- Feature 4.5: transit disruption events ------------------------------

    def _place_center(self, place_name: str) -> Optional[Tuple[float, float]]:
        """Return (x, y) center of a place by name, or None if not found."""
        for p in self.places:
            if p['name'] == place_name:
                return (float(p.get('center_x', 0.0)), float(p.get('center_y', 0.0)))
        return None

    def _update_event_states(self) -> None:
        """Activate events whose start_step has arrived, deactivate past end_step."""
        active_names = {e['name'] for e in self.event_states if e.get('active')}
        for ec in self.event_configs:
            if ec['name'] in active_names:
                continue
            if not (ec['start_step'] <= self.step <= ec['end_step']):
                continue
            pos = self._place_center(ec['affected_place']) if ec['affected_place'] else None
            if pos is None:
                logger.warning(
                    f"Event '{ec['name']}' affected_place '{ec['affected_place']}' "
                    f"not found — event will still fire but without a position anchor."
                )
            state = dict(ec)
            state['place_position'] = pos
            state['active'] = True
            state['activated_at_step'] = self.step
            self.event_states.append(state)
            logger.info(
                f"EVENT '{ec['name']}' activated at step {self.step} "
                f"(type={ec['type']}, affected={ec['affected_place']}, pos={pos})"
            )
            # Path 3: push-notification propagation. Broadcast model — fires
            # once at activation, not per-step. Controlled by event's
            # `broadcast_once` flag (default True). Set to False to opt out,
            # in which case this event won't use notification-based awareness.
            if state['type'] in ('transit_disruption', 'last_train') and state.get('broadcast_once', True):
                self._propagate_event_via_notification(state)
        for st in self.event_states:
            if st.get('active') and self.step > st['end_step']:
                st['active'] = False
                logger.info(f"EVENT '{st['name']}' deactivated at step {self.step}")

    def _active_transit_events(self) -> List[Dict]:
        return [e for e in self.event_states
                if e.get('active') and e.get('type') in ('transit_disruption', 'last_train')]

    def _mark_aware(self, agent: Agent, event: Dict, source: str,
                    source_agent_id: Optional[int] = None) -> bool:
        """Record that `agent` just became aware of `event` via `source`.
        Returns True if this was a new transition (first-time awareness).
        Idempotent — repeat calls are no-ops."""
        name = event['name']
        if name in agent.known_events:
            return False
        agent.known_events.add(name)
        agent.awareness_source[name] = source
        agent.awareness_step[name] = self.step
        self._apply_salience_boost(agent, event)
        self._log_awareness_transition(agent, event, source, source_agent_id)
        return True

    def _propagate_direct_event_awareness(self) -> None:
        """Path 1: agents inside notify_radius of an active event auto-learn it."""
        events = self._active_transit_events()
        if not events:
            return
        for ev in events:
            pos = ev.get('place_position')
            if pos is None or ev.get('notify_radius', 0) <= 0:
                continue
            for agent in self.agents:
                if agent.distance_to(pos) <= ev['notify_radius']:
                    self._mark_aware(agent, ev, source='direct')

    def _propagate_event_via_message(self, receiver: Agent, text: str,
                                     sender_id: Optional[int] = None) -> None:
        """Path 2: a received message containing transit keywords marks the
        receiver as aware of every currently-active transit event.

        Deliberately coarse — we don't try to parse which event the message is
        about. With one transit event per run (the hackathon scenario) this is
        accurate enough and keeps the keyword list cheap.
        """
        if not text:
            return
        events = self._active_transit_events()
        if not events:
            return
        if any(kw in text for kw in self._transit_keywords):
            for ev in events:
                self._mark_aware(receiver, ev, source='conversation',
                                 source_agent_id=sender_id)

    def _propagate_event_via_notification(self, event: Dict) -> None:
        """Path 3: broadcast-model push notification. At the moment an event
        activates, each agent rolls ONCE against their per-persona
        phone_check_rate. Winners learn the event; losers never get another
        notification roll for the same event (they can still become aware
        via path-1 direct or path-2 conversation).

        Design choice (vs. per-step polling): per-step would make cumulative
        probability → 1 over a long observation window, collapsing the rate
        difference between personas. Broadcast-once preserves rate as the
        actual success probability, so 0.4 means ~40% of those personas
        learn it, 0.7 means ~70%, etc. This matches real-world transit
        alerts where apps push at event time, not on a polling loop.

        Rationale: captures the Jacobs-style asymmetry where some people opt
        into transit alerts and some don't, independent of proximity.
        """
        for agent in self.agents:
            if event['name'] in agent.known_events:
                continue
            rate = float(agent.persona.get('phone_check_rate', 0.5) or 0.0)
            if rate <= 0.0:
                continue
            if random.random() < rate:
                self._mark_aware(agent, event, source='notification')

    def _apply_salience_boost(self, agent: Agent, event: Dict) -> None:
        """Bump base_salience by +0.2 for identities listed in salience_boost_targets.
        Bounded at 1.0. Idempotent (safe to call on repeat awareness)."""
        targets = set(event.get('salience_boost_targets') or [])
        if not targets or not agent.social_identities:
            return
        boosted_key = f"_salience_boosted::{event['name']}"
        if getattr(agent, '_salience_marks', None) is None:
            agent._salience_marks = set()
        if boosted_key in agent._salience_marks:
            return
        for ident in agent.social_identities:
            if ident.get('group_name') in targets:
                ident['base_salience'] = min(
                    1.0, float(ident.get('base_salience', 0.0)) + 0.2
                )
        agent._salience_marks.add(boosted_key)

    # ===== Phase B helpers =====
    def _place_contains(self, place: Dict, x: int, y: int) -> bool:
        cx = place.get('center_x', 0)
        cy = place.get('center_y', 0)
        hx = place.get('half_size_x', 0)
        hy = place.get('half_size_y', 0)
        return (cx - hx) <= x <= (cx + hx) and (cy - hy) <= y <= (cy + hy)

    def _phase_b_inject_perceive(self, action_decisions: List[Tuple]) -> None:
        """Phase B: 各 agent が今 step で進入した place の perceive_pass を memory に注入。
        さらに action_type=='enter' で対象 place に成功進入していたら perceive_enter も注入。
        """
        # 各 agent の最後の action_decision を id でマップ
        decision_by_id = {}
        for agent, dec, _nb in action_decisions:
            if isinstance(dec, dict):
                decision_by_id[agent.id] = dec

        n_pass_injected = 0
        n_enter_injected = 0
        for agent in self.agents:
            x, y = agent.position
            # 1) perceive_pass: agent.position が含まれる全 place について、初回なら注入
            for place in self.places:
                nm = place.get('name', '')
                if not nm or nm in agent.visited_places:
                    continue
                if not self._place_contains(place, x, y):
                    continue
                # 初回進入
                agent.visited_places.add(nm)
                pp = place.get('perceive_pass')
                if pp:
                    line = f"[現地で見えた - {nm}] {pp}"
                    agent.memory.append(line)
                    if len(agent.memory) > agent.memory_limit:
                        agent.memory.pop(0)
                    n_pass_injected += 1
                # 同時に place_type 記録 (近隣環境の判定で使う)

            # 2) perceive_enter: action_type=='enter' で current_place に居て、初回なら注入
            dec = decision_by_id.get(agent.id) or {}
            if dec.get('action_type') == 'enter':
                target_name = dec.get('target_place') or agent.current_place
                if target_name and target_name not in agent.entered_places:
                    target_place = next((p for p in self.places if p.get('name') == target_name), None)
                    if target_place:
                        # 進入は実際に bbox 内であることが条件
                        if self._place_contains(target_place, x, y):
                            agent.entered_places.add(target_name)
                            pe = target_place.get('perceive_enter')
                            if pe:
                                line = f"[中に入って気づいた - {target_name}] {pe}"
                                agent.memory.append(line)
                                if len(agent.memory) > agent.memory_limit:
                                    agent.memory.pop(0)
                                n_enter_injected += 1
        if n_pass_injected or n_enter_injected:
            logger.info(f"Phase B perceive: step={self.step}, pass_inj={n_pass_injected}, enter_inj={n_enter_injected}")

    def _phase_b_log_body_sense(self) -> None:
        """Phase B: 累積距離・時間・近隣環境を agent.memory に N step 毎に挿入する。
        判定はあくまで「事実」のみで、疲労・気分はLLMに任せる。
        """
        # 食肉市場と工事現場の位置を予め取得
        meat_market = next((p for p in self.places
                            if (p.get('attributes') or {}).get('environment', {}).get('is_meat_market')
                               or p.get('name') == '食肉市場'), None)
        construction_places = [p for p in self.places
                               if (p.get('attributes') or {}).get('environment', {}).get('near_construction')]

        n_body_logged = 0
        for agent in self.agents:
            # 距離累積
            x, y = agent.position
            if agent._last_position is not None:
                dx = x - agent._last_position[0]
                dy = y - agent._last_position[1]
                # 整数座標距離 (cell 単位)
                step_dist = (dx * dx + dy * dy) ** 0.5
                agent.cumulative_distance_cells += step_dist
            agent._last_position = (x, y)
            agent.cumulative_steps += 1

            # 一定 step 毎に身体事実を memory に注入
            if agent.cumulative_steps % max(1, self.phase_b_body_log_interval) != 0:
                continue

            meters_per_cell = 5  # shinagawa_field の規約
            meters = int(agent.cumulative_distance_cells * meters_per_cell)
            cur_place = agent.current_place or "(屋外)"

            # 近隣環境タグ
            near_tags = []
            if meat_market is not None:
                mx = meat_market.get('center_x', 0); my = meat_market.get('center_y', 0)
                d_m = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5 * meters_per_cell
                if d_m < 100:
                    near_tags.append(f"食肉市場まで{int(d_m)}m")
            for cp in construction_places:
                cx2 = cp.get('center_x', 0); cy2 = cp.get('center_y', 0)
                d_c = ((x - cx2) ** 2 + (y - cy2) ** 2) ** 0.5 * meters_per_cell
                if d_c < 80:
                    near_tags.append(f"{cp.get('name')}の工事現場が近い")
                    break  # 1個で十分

            # 屋外/屋内
            cur_place_obj = next((p for p in self.places if p.get('name') == cur_place), None)
            env = ((cur_place_obj or {}).get('attributes') or {}).get('environment', {}) if cur_place_obj else {}
            indoor = env.get('indoor')
            roof = env.get('roof')
            grade = env.get('grade')
            env_str_parts = []
            if indoor is True: env_str_parts.append("屋内")
            elif indoor is False:
                env_str_parts.append("屋外")
                if roof: env_str_parts.append("屋根あり")
                else: env_str_parts.append("屋根なし")
            if grade and grade != 'flat':
                env_str_parts.append(f"傾斜:{grade}")
            env_str = "/".join(env_str_parts) if env_str_parts else ""

            line = (
                f"[身体記録 step{self.step}] これまで{agent.cumulative_steps}step歩き、"
                f"累積{meters}m移動。今いるのは「{cur_place}」"
                + (f" ({env_str})" if env_str else "")
                + (" / " + " / ".join(near_tags) if near_tags else "")
                + "。"
            )
            agent.memory.append(line)
            if len(agent.memory) > agent.memory_limit:
                agent.memory.pop(0)
            n_body_logged += 1
        if n_body_logged:
            logger.info(f"Phase B body log: step={self.step}, agents_logged={n_body_logged}")

    def _compute_emergency_boost(self, agent: Agent) -> float:
        """Phase 2.5: route-specific emergency_boost for should_speak.

        Precedence / rule (max across all active stressors):
        - fire near agent                          → 3.0 (hard override)
        - transit event learned via notification
          * within NOTIFICATION_FRESH_STEPS of event start → 3.0 ("breaking news"
            adrenaline while it's still novel)
          * after the freshness window             → 1.5 (background awareness)
        - transit event learned directly           → 1.5
        - transit event heard in conversation      → 1.0 (default, no boost)
        """
        if self._fires_near(agent, multiplier=2.0):
            return 3.0

        NOTIFICATION_FRESH_STEPS = 5
        boost = 1.0
        for ev in self._active_transit_events():
            name = ev['name']
            if name not in agent.known_events:
                continue
            src = agent.awareness_source.get(name, 'direct')
            if src == 'notification':
                activated = ev.get('activated_at_step')
                if activated is not None and (self.step - activated) < NOTIFICATION_FRESH_STEPS:
                    candidate = 3.0
                else:
                    candidate = 1.5
            elif src == 'direct':
                candidate = 1.5
            else:  # conversation
                candidate = 1.0
            if candidate > boost:
                boost = candidate
        return boost

    def _log_awareness_transition(self, agent: Agent, event: Dict,
                                  source: str,
                                  source_agent_id: Optional[int]) -> None:
        """Append one line per awareness transition to
        output/awareness_propagation_log.jsonl. Transition-triggered — not a
        per-step snapshot. Complements event_awareness_log.jsonl (which IS a
        snapshot)."""
        if not self.output_dir:
            return
        payload = {
            'step': self.step,
            'time': self._current_time_str(),
            'event_name': event['name'],
            'agent_id': agent.id,
            'agent_name': agent.persona.get('name', f'Agent {agent.id}'),
            'source': source,
            'source_agent_id': source_agent_id,
        }
        path = os.path.join(self.output_dir, 'awareness_propagation_log.jsonl')
        with self._log_locks['awareness_propagation']:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def get_active_events_for_agent(self, agent: Agent) -> Optional[List[Dict]]:
        """Return prompt-ready info for events the agent is aware of.
        None if the agent has nothing to be told about (no events or not aware)."""
        if not self.event_states:
            return None
        perceived = []
        for ev in self.event_states:
            if not ev.get('active'):
                continue
            if ev['name'] not in agent.known_events:
                continue
            perceived.append({
                'name': ev['name'],
                'type': ev['type'],
                'affected_place': ev.get('affected_place'),
                'description': ev.get('description', ''),
                'awareness_source': agent.awareness_source.get(ev['name'], 'direct'),
                'awareness_step': agent.awareness_step.get(ev['name']),
            })
        return perceived if perceived else None

    def _log_event_awareness(self) -> None:
        """Append one snapshot of per-agent known_events to
        output/event_awareness_log.jsonl (1 line = 1 step). Only writes while
        at least one event config exists to avoid noise on legacy fire runs."""
        if not self.output_dir or not self.event_configs:
            return
        active_names = [e['name'] for e in self._active_transit_events()]
        payload = {
            'step': self.step,
            'time': self._current_time_str(),
            'active_events': active_names,
            'agents': [
                {
                    'id': a.id,
                    'name': a.persona.get('name', f'Agent {a.id}'),
                    'known_events': sorted(a.known_events),
                }
                for a in self.agents
            ],
        }
        path = os.path.join(self.output_dir, 'event_awareness_log.jsonl')
        with self._log_locks['event_awareness']:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def should_speak(
        self,
        agent: Agent,
        nearby_agents: List[Agent],
        place_config: Optional[Dict],
        *,
        apply_opener_gate: bool = False,
    ) -> Optional[int]:
        """Decide (before any LLM call) whether the agent speaks this step.

        Returns the target agent's id if yes, None if silent.
        Formula: p_speak = talkativeness × social_likelihood × relationship × proximity_factor,
        attenuated by social_fatigue, 3× boosted when a fire is within view.
        Also emits one record per call to output/should_speak_log.jsonl so the
        gate is auditable independent of whether it fired.
        """
        if not nearby_agents:
            return None

        # コスト削減 #5: 企業 host は学生が近接していないステップでは喋らない
        # (受付役なので、客がいないのに同業 host 同士で勝手にお喋りしても
        # シナリオ価値は低い。LLM call を burn しないため即 silent を返す)。
        if agent.persona.get('is_host'):
            has_student_nearby = any(
                not (other.persona.get('is_host')
                     or str(other.persona.get('axis_id', '')).startswith('Host_'))
                for other in nearby_agents
            )
            if not has_student_nearby:
                return None

        talkativeness = self._infer_talkativeness(agent)

        if place_config is not None:
            social_likelihood = float(place_config.get('social_likelihood', 0.05) or 0.05)
        else:
            # Outside any place → open street; use a low default rather than 0
            # so a chance greeting between friends still fires.
            social_likelihood = 0.1

        fatigue = float(agent.internal_state.get('social_fatigue', 0.0))
        fatigue_factor = max(0.2, 1.0 - fatigue / 200.0)
        effective_talk = talkativeness * fatigue_factor

        emergency_boost = self._compute_emergency_boost(agent)

        # Special facilitator role (axis_id ∈ {"Sato", "AIRobo", "AIGod"}): treat all in-room
        # nearby agents as equally close (proximity_factor=1.0) and sample partner
        # weighted by p_speak, so the facilitator distributes attention rather
        # than always picking the geometrically nearest participant.
        is_facilitator = agent.persona.get('axis_id') in ('Sato', 'AIRobo', 'AIGod')

        candidates: List[Dict] = []
        best_prob = 0.0
        best_partner_id: Optional[int] = None
        for other in nearby_agents:
            relationship = agent.get_relationship(other.id)
            dist = agent.distance_to(other.position)
            if is_facilitator:
                proximity_factor = 1.0
            elif dist <= 2:
                proximity_factor = 1.0
            elif dist <= 4:
                proximity_factor = 0.4
            else:
                proximity_factor = 0.1

            # 近接 (dist<=2) のときは 1.5 倍ブースト —「近くにいるのに何も話さない」緩和。
            proximity_close_boost = 1.5 if (not is_facilitator and dist <= 2) else 1.0
            # #27改: 「同行ペア」(relationship≥0.5 かつ dist≤2) はさらに 1.5x ブースト
            # = 一緒に行動中の友達と歩きながら会話している状態を再現。
            companion_boost = 1.5 if (not is_facilitator and dist <= 2 and relationship >= 0.5) else 1.0
            # smoke17: opener モード確率を 2x にブースト (silent 90% → 80% 想定)。
            # 「最初は様子見で黙るが、もう少し口火切る人がいてもいい」というユーザ体感への調整。
            global_speak_multiplier = 2.0
            p_raw = effective_talk * social_likelihood * relationship * proximity_factor * proximity_close_boost * companion_boost * global_speak_multiplier
            p = min(1.0, p_raw * emergency_boost)
            candidates.append({
                "partner_id": other.id,
                "partner_name": other.persona.get('name', f"Agent {other.id}"),
                "relationship": round(relationship, 4),
                "talkativeness": round(effective_talk, 4),
                "social_likelihood": round(social_likelihood, 4),
                "proximity_factor": proximity_factor,
                "companion_boost": companion_boost,
                "emergency_boost": emergency_boost,
                "p_speak": round(p, 6),
            })
            if p > best_prob:
                best_prob = p
                best_partner_id = other.id

        # smoke15: 「確率による silent コントロール」は廃止。p_speak は partner 選択の
        # 重み付け関数として残す (relationship × proximity が高い相手が優先的に選ばれる) が、
        # 「random.random() < best_prob で silent」のサンプリングは削除する。
        # silent 判定は LLM 自身の "" 出力 + Jaccard 後フィルタが担う。
        # nearby 0 / host idle は早期 return で silent 確定 (既存ロジック)。
        selected_partner_id: Optional[int] = None
        if is_facilitator and candidates:
            # facilitator: 確率を相手の重み付けにそのまま使う (確率最大ではなく、weighted pick)。
            weights = [c['p_speak'] for c in candidates]
            wsum = sum(weights)
            if wsum > 0:
                pick = random.uniform(0, wsum)
                acc = 0.0
                for c, w in zip(candidates, weights):
                    acc += w
                    if pick <= acc:
                        selected_partner_id = c['partner_id']
                        break
                if selected_partner_id is None:
                    selected_partner_id = candidates[-1]['partner_id']
        elif best_partner_id is not None:
            # 通常: relationship × proximity が最大の相手を選ぶ。silent 判定は走らせない。
            selected_partner_id = best_partner_id

        # smoke16: opener モード時のみ確率ゲートを適用 (= 受信トリガーなし & イベントなしの step)。
        # 「自分から話しかける動機しかない」step は best_prob 確率で skip → LLM call 節約。
        # 受信あり / イベントあり時 (apply_opener_gate=False) はこのゲートを通さない。
        if apply_opener_gate and selected_partner_id is not None:
            if random.random() >= best_prob:
                selected_partner_id = None

        self._log_should_speak(agent, nearby_agents, candidates, selected_partner_id)
        return selected_partner_id

    def _log_relationships_snapshot(self) -> None:
        """Append one snapshot of every stored relationship edge to
        output/relationships_timeline.jsonl (1 line = 1 step).
        Edges are directional (`from_id → to_id`) because the relationships
        dict is per-agent; downstream analysis can symmetrize if needed."""
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        agents_by_id = {a.id: a for a in self.agents}
        edges: List[Dict] = []
        for agent in self.agents:
            from_name = agent.persona.get('name', f"Agent {agent.id}")
            for other_id, value in agent.relationships.items():
                other = agents_by_id.get(other_id)
                to_name = (
                    other.persona.get('name', f"Agent {other_id}") if other is not None
                    else f"Agent {other_id}"
                )
                edges.append({
                    "from_id": agent.id,
                    "from_name": from_name,
                    "to_id": other_id,
                    "to_name": to_name,
                    "value": round(float(value), 4),
                })

        record = {
            "step": self.step,
            "time": self._current_time_str(),
            "relationships": edges,
        }
        path = os.path.join(self.output_dir, "relationships_timeline.jsonl")
        with self._log_locks["relationships_timeline"]:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _log_should_speak(
        self,
        agent: Agent,
        nearby_agents: List[Agent],
        candidates: List[Dict],
        selected_partner_id: Optional[int],
    ) -> None:
        """Append one should_speak invocation to output/should_speak_log.jsonl."""
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        record = {
            "step": self.step,
            "time": self._current_time_str(),
            "agent_id": agent.id,
            "agent_name": agent.persona.get('name', f"Agent {agent.id}"),
            "nearby_count": len(nearby_agents),
            "candidates": candidates,
            "decision": "speak" if selected_partner_id is not None else "skip",
            "selected_partner_id": selected_partner_id,
        }
        path = os.path.join(self.output_dir, "should_speak_log.jsonl")
        with self._log_locks["should_speak"]:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _get_agent_layer_jp(self, agent: Agent) -> str:
        """Map behavior_layer string to Japanese label used in logs."""
        return {
            "transit": "通過中",
            "dwelling": "滞在中",
            "interacting": "交流中",
        }.get(agent.behavior_layer, agent.behavior_layer or "?")

    def _append_viewer_snapshot(self, memory_reasoning_records: List[Dict]) -> None:
        """Feature 5: record one timeline entry for the Canvas viewer.

        Called at the end of each step_simulation. Consumes the current
        _viewer_step_convs buffer and resets it for the next step.
        """
        mr_by_id = {r.get("id"): r for r in (memory_reasoning_records or []) if r}
        agents_snap: List[Dict] = []
        for a in self.agents:
            mr = mr_by_id.get(a.id, {})
            pos = getattr(a, "position", None) or (0, 0)
            agents_snap.append({
                "id": a.id,
                "x": int(pos[0]),
                "y": int(pos[1]),
                "in_place": bool(a.in_place),
                "current_place": a.current_place,
                "layer": getattr(a, "behavior_layer", None),
                "memory": mr.get("memory", ""),
                "reasoning": mr.get("reasoning", ""),
                "known_events": sorted(list(a.known_events)) if a.known_events else [],
                "awareness_source": dict(a.awareness_source) if a.awareness_source else {},
            })
        events_snap: List[Dict] = []
        for e in self.event_states:
            if not e.get("active"):
                continue
            events_snap.append({
                "name": e.get("name"),
                "type": e.get("type"),
                "affected_place": e.get("affected_place"),
                "remaining_steps": max(0, int(e.get("end_step", 0)) - self.step),
            })
        self._viewer_timeline.append({
            "step": self.step,
            "time": self._current_time_str(),
            "agents": agents_snap,
            "conversations": list(self._viewer_step_convs),
            "events": events_snap,
        })
        self._viewer_step_convs = []

    def export_simulation_data(self) -> Optional[str]:
        """Feature 5: write simulation_data.json for the Canvas viewer.

        One-shot export at simulation end. Bundles metadata, places,
        personas, and the full per-step timeline into a single file.
        Returns the output path on success, None on failure or when
        output_dir is not set.
        """
        if not self.output_dir:
            return None
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            # 既定は都市スケール (1 cell = 5m)。config に metadata.meters_per_cell が
            # あればそれを優先 (建築スケールでは 1m 等に切り替え)。
            cfg_meta = self.config.get("metadata") or {}
            cell_meters = float(cfg_meta.get("meters_per_cell", 5))
            field_side_cells = 2 * self.half_space_size + 1
            metadata = {
                "total_steps": self.step,
                "minutes_per_step": int(getattr(self, "step_duration_minutes", 1)),
                "start_time": self.start_time_str,
                "start_datetime_iso": self.start_datetime.isoformat() if self.start_datetime else None,
                "meters_per_cell": cell_meters,
                "half_space_size": self.half_space_size,
                "field_size_cells": field_side_cells,
                "field_size_meters": field_side_cells * cell_meters,
                "llm_model": getattr(self.llm_client, "model", None),
            }
            # config の metadata から viewer 向けの補助情報を pass-through
            # (name, grid_step_m, hide_place_labels, scenario_kind 等)
            for k in ("name", "description", "scenario_kind",
                      "grid_step_m", "hide_place_labels",
                      "wall_color", "ceiling_height_m"):
                if k in cfg_meta:
                    metadata[k] = cfg_meta[k]
            places = [{
                "name": p.get("name"),
                "type": p.get("type"),
                "center_x": p.get("center_x"),
                "center_y": p.get("center_y"),
                "half_size_x": p.get("half_size_x"),
                "half_size_y": p.get("half_size_y"),
                "capacity": p.get("capacity"),
                "social_likelihood": p.get("social_likelihood"),
                "is_spawn_point": bool(p.get("is_spawn_point", False)),
                "attributes": p.get("attributes") or {},
                "perceive_pass": p.get("perceive_pass"),
                "perceive_enter": p.get("perceive_enter"),
            } for p in self.places]
            personas = [{
                "id": a.id,
                "name": a.persona.get("name"),
                "age": a.persona.get("age"),
                "gender": a.persona.get("gender"),
                "occupation": a.persona.get("occupation"),
                "speech_style": a.persona.get("speech_style"),
                "phone_check_rate": a.persona.get("phone_check_rate"),
                # viewer 側で host判定するために必要 (緑色描画 / 凡例)。
                "is_host": bool(a.persona.get("is_host", False)),
                "axis_id": a.persona.get("axis_id"),
                "variant": a.persona.get("variant"),
                "school_fit": a.persona.get("school_fit"),
                "interest_tag": a.persona.get("interest_tag"),
                "tendency": a.persona.get("tendency"),
                "social_identities": list(getattr(a, "social_identities", []) or []),
                # 3D viewer の追従吹き出しが特徴的背景を表示するために必要
                "mobility": a.persona.get("mobility"),
                "nationality": a.persona.get("nationality"),
                "assigned_hosts": a.persona.get("assigned_hosts"),
                "initial_place": a.persona.get("initial_place"),
            } for a in self.agents]
            payload = {
                "metadata": metadata,
                "places": places,
                "personas": personas,
                "timeline": self._viewer_timeline,
                "focus_agent_id": None,
                "scene_3d": self.config.get("scene_3d") or {},
            }
            out_path = os.path.join(self.output_dir, "simulation_data.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            logger.info(
                f"Exported simulation_data.json ({len(self._viewer_timeline)} steps, "
                f"{len(personas)} personas, {len(places)} places) to {out_path}"
            )
            return out_path
        except Exception as e:
            logger.error(f"Failed to export simulation_data.json: {e}", exc_info=True)
            return None

    def step_simulation(self):
        """Execute one simulation step

        New order:
        1. All agents decide messages (without position information)
        2. Messages are sent to nearby agents (using decision-time positions)
        3. All agents decide actions (with position information and message content)
        4. Agents move to new positions
        """
        self.step += 1
        # Advance wall-clock and (optionally) spawn/despawn agents at stations
        # before any agent reasoning occurs, so the new arrivals see the same
        # step the rest of the population sees.
        self._advance_time()
        self._handle_agent_spawning()
        self._apply_time_context_to_agents()

        # Fire activation check (multiple fires)
        active_names = {f['name'] for f in self.fire_states}
        for fc in self.fire_configs:
            if fc['name'] not in active_names and self.step >= fc['start_step']:
                if 'center_x' in fc and 'center_y' in fc:
                    fire_pos = (fc['center_x'], fc['center_y'])
                else:
                    fire_pos = self._generate_random_position()
                fire_state = {
                    'name': fc['name'],
                    'position': fire_pos,
                    'intensity': fc['intensity'],
                    'radius': fc['radius'],
                    'start_step': fc['start_step'],
                    'active': True,
                }
                self.fire_states.append(fire_state)
                logger.info(
                    f"FIRE '{fc['name']}' started at position {fire_pos} with intensity "
                    f"{fc['intensity']}, radius {fc['radius']}"
                )

        # Transit-disruption event activation (feature 4.5). Same shape as the
        # fire activation block so the two event families stay independent.
        self._update_event_states()
        # Propagate direct-proximity awareness right after events become active,
        # but BEFORE the LLM phases, so prompts built this step see the
        # refreshed known_events.
        self._propagate_direct_event_awareness()

        # Update agent states
        for agent in self.agents:
            agent.update_state(self.places)
        # Feature 2: refresh internal_state before LLM phases so prompts see
        # the current energy/hunger/social_fatigue values.
        self._update_internal_states()

        # Phase 1: Collect message decisions from all agents (without position information).
        # LLM calls are executed in parallel across agents (when parallel_workers > 1).
        # Feature 3: system-level should_speak gate runs BEFORE any LLM call,
        # so most agents skip the message LLM entirely. We keep BOTH the full
        # nearby list (for Phase 3 action + Phase 2 broadcast) and the
        # targeted_nearby (single partner) that is passed to the message LLM.
        skipped_p1 = 0
        gated_p1 = 0
        # phase1_tasks carry the partner_id from should_speak so Phase 2 can
        # deliver the message to that one recipient only (policy D: message is
        # 1-to-1 per the prompt audience, while overheard-agents still get the
        # event-awareness path via _propagate_event_via_message).
        phase1_tasks: List[Tuple[Agent, List[Agent], List[Agent], Optional[Dict], Optional[List[Dict]], Optional[List[Dict]], Optional[int]]] = []
        phase1_results: List[Optional[Tuple[Agent, Dict, List[Agent], Optional[int]]]] = [None] * len(self.agents)

        # ===== 第1パス: 全 agent の partner_id を集計 (LLM call はまだしない) =====
        # 双方向同時発話 (A→B かつ B→A) を検出するため、まず全員の発話相手候補を出す。
        partner_map: Dict[int, Optional[int]] = {}
        nearby_map: Dict[int, List[Agent]] = {}
        skip_set: set = set()
        for agent in self.agents:
            nearby_agents = agent.get_nearby_agents(self.agents)
            nearby_map[agent.id] = nearby_agents
            if self.skip_probability > 0 and random.random() < self.skip_probability:
                skip_set.add(agent.id)
                continue
            current_place_config: Optional[Dict] = None
            if agent.in_place and agent.current_place:
                current_place_config = next(
                    (p for p in self.places if p['name'] == agent.current_place), None
                )
            # smoke16: opener モード判定 — 直前 step に自分宛ての受信があるか、
            # 火災 / イベント等のトリガーがあるか。なければ「自分から話しかけるしか
            # 動機がない step」 = opener モード。confirm_only confirm: ある = 必ず LLM call、
            # なし = 確率 gate (best_prob) で skip 判定して LLM call 節約。
            recent_recv = any(
                m.get('step', 0) >= self.step - 1
                for m in agent.received_messages
            )
            has_trigger = (
                recent_recv
                or bool(self.get_fire_info_for_agent(agent))
                or bool(self.get_active_events_for_agent(agent))
            )
            # smoke23: 不適応の子は opener モード (= 受信トリガーなし) では発話しない (rule-based)。
            # 受信トリガーがあれば普通に応答するが、自分から話しかけることは確実にゼロ。
            if not has_trigger and (agent.persona.get('school_fit') or '').strip() == '不適応':
                partner_map[agent.id] = None
                continue
            # smoke23: host (企業担当者) は学生が nearby にいる場合、opener gate を強制 ON
            # (= 必ず LLM call) する。host が動かない設計のため、学生が来た時に自分から
            # 声をかけるのが host 側の役割。受信トリガー有無に関わらず opener する。
            is_host_agent = bool(agent.persona.get('is_host'))
            student_nearby_for_host = is_host_agent and any(
                not (other.persona.get('is_host')
                     or str(other.persona.get('axis_id', '')).startswith('Host_'))
                for other in nearby_agents
            )
            if student_nearby_for_host:
                # host で学生が近接 → 必ず LLM call (opener gate 適用しない)
                effective_apply_opener_gate = False
            else:
                effective_apply_opener_gate = not has_trigger
            partner_id = self.should_speak(
                agent, nearby_agents, current_place_config,
                apply_opener_gate=effective_apply_opener_gate,
            )
            # should_speak の None 戻り値の意味:
            #   ① opener モードで確率 gate に外れた → LLM call スキップ確定 (silent)
            #   ② nearby 0 / host で学生不在 → silent 確定
            #   ③ それ以外 (= has_trigger だが best_partner_id 0 件) → 最近接にフォールバック
            if partner_id is None:
                if not has_trigger:
                    # ①: opener モードで gate 外れ → silent 確定 (LLM call せず)
                    partner_map[agent.id] = None
                    continue
                # has_trigger=True なのに None → ② or ③ の判定
                is_host = bool(agent.persona.get('is_host'))
                has_student_nearby = any(
                    not (other.persona.get('is_host')
                         or str(other.persona.get('axis_id', '')).startswith('Host_'))
                    for other in nearby_agents
                )
                if not nearby_agents or (is_host and not has_student_nearby):
                    partner_map[agent.id] = None
                    continue
                nearest = min(nearby_agents, key=lambda a: agent.distance_to(a.position))
                partner_id = nearest.id
            partner_map[agent.id] = partner_id

        # ===== 双方向同時発話排除 =====
        # A の partner==B かつ B の partner==A → 大きい id を silent (聞く側)、
        # 小さい id が話す。これで「同時に話しかけ合う」現象を消す。turn-taking 自然化。
        silent_set: set = set()
        for a, b in partner_map.items():
            if b is None: continue
            if partner_map.get(b) == a and a < b:
                silent_set.add(b)

        # ===== 第2パス: 各 agent についてタスク化 / silent =====
        for idx, agent in enumerate(self.agents):
            nearby_agents = nearby_map[agent.id]
            if agent.id in skip_set:
                phase1_results[idx] = (agent, {"message": "", "reasoning": "(思考スキップ)"}, nearby_agents, None)
                skipped_p1 += 1
                continue
            partner_id = partner_map.get(agent.id)
            if partner_id is None:
                phase1_results[idx] = (agent, {"message": "", "reasoning": "(発話相手なし)"}, nearby_agents, None)
                gated_p1 += 1
                continue
            if agent.id in silent_set:
                phase1_results[idx] = (agent, {"message": "", "reasoning": "(同時発話排除: 相手が話すので聞く側に回る)"}, nearby_agents, None)
                gated_p1 += 1
                continue

            partner = next((a for a in nearby_agents if a.id == partner_id), None)
            targeted_nearby = [partner] if partner is not None else nearby_agents

            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            fire_info = self.get_fire_info_for_agent(agent)
            events_info = self.get_active_events_for_agent(agent)
            phase1_tasks.append((agent, targeted_nearby, nearby_agents, agent_place_status, fire_info, events_info, partner_id))
            # Reserve slot — filled in after the executor returns.
            phase1_results[idx] = None

        if phase1_tasks:
            workers = max(1, min(self.parallel_workers, len(phase1_tasks)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        ag.decide_message, ps, targ_nb, self.step,
                        fire_info=fi, events_info=ei, partner_id=pid,
                    )
                    for ag, targ_nb, _full_nb, ps, fi, ei, pid in phase1_tasks
                ]
                task_results = []
                for (ag, _targ_nb, full_nb, _, _, _, pid), fut in zip(phase1_tasks, futures):
                    try:
                        decision = fut.result()
                    except Exception as e:
                        logger.error(f"Agent {ag.id} Phase 1 parallel execution failed: {e}")
                        decision = {"message": "", "reasoning": "Parallel execution error"}
                    # Keep full_nb (for Phase 2 event propagation + Phase 3),
                    # and partner_id (who the LLM's message was written for).
                    task_results.append((ag, decision, full_nb, pid))

            # Merge task_results back into phase1_results in original agent order.
            result_iter = iter(task_results)
            for idx, slot in enumerate(phase1_results):
                if slot is None:
                    phase1_results[idx] = next(result_iter)

        message_decisions: List[Tuple[Agent, Dict, List[Agent], Optional[int]]] = [r for r in phase1_results if r is not None]

        # Phase 2: Deliver messages.
        # Policy D (2026-04-20): the message content goes to the single partner
        # the LLM was prompted with (1-to-1 matching prompt audience). Other
        # agents in `nearby_agents` still receive the event-awareness side-
        # effect via _propagate_event_via_message — they "overhear" the gist
        # of transit news without being logged as individual recipients.
        self.last_step_messages = []
        for agent, message_decision, nearby_agents, partner_id in message_decisions:
            message_content = message_decision.get('message', '')
            if not (message_content and nearby_agents):
                continue
            sender_name = agent.persona.get('name') or f"Agent {agent.id}"
            partner = None
            if partner_id is not None:
                partner = next((a for a in nearby_agents if a.id == partner_id), None)

            if partner is not None:
                partner_name = partner.persona.get('name', f'Agent {partner.id}')
                logger.info(
                    f"Step {self.step}: {sender_name} → {partner_name}: "
                    f"\"{message_content}\""
                )
                partner.receive_message(
                    agent.id,
                    message_content,
                    step=self.step,
                    from_name=sender_name,
                )
                # Record the sender's own utterance so they see their recent
                # speech in next step's prompt (prevents repeating themselves).
                agent.record_sent_message(
                    partner.id,
                    message_content,
                    step=self.step,
                    to_name=partner_name,
                )
                # working_state 用: 会話相手が host (企業担当者) なら「話した」とカウント。
                # 双方向に rule-based で記録 (sender が学生・partner が host のとき、
                # sender.talked_hosts に partner.name を加える、逆も同様)。
                if partner.persona.get('is_host'):
                    agent.talked_hosts.add(partner_name)
                if agent.persona.get('is_host'):
                    partner.talked_hosts.add(sender_name)
                self._propagate_event_via_message(
                    partner, message_content, sender_id=agent.id
                )
                self.last_step_messages.append((agent.id, partner.id))
                self._log_message(
                    from_agent=agent,
                    to_agent=partner,
                    message=message_content,
                    reasoning=message_decision.get('reasoning', '')
                )

            # Overheard path: every OTHER nearby agent still has a chance to
            # pick up the event keyword (transit disruption). No message log,
            # no memory push — only the awareness side-effect.
            for other_agent in nearby_agents:
                if partner is not None and other_agent.id == partner.id:
                    continue
                self._propagate_event_via_message(
                    other_agent, message_content, sender_id=agent.id
                )

            # moltbook風: グローバルチャンネルモードでは、近接の有無に関係なく全 agent が
            # この発話を「チャンネル投稿」として読める状態にする。各 agent の received_messages
            # に追加して、次 step の prompt で全員が共有チャンネルを読む形にする。
            # primary partner と nearby_agents 既出の overheard は重複させない。
            if self.global_channel_mode and message_content:
                handled = set()
                if partner is not None:
                    handled.add(partner.id)
                for oa in nearby_agents:
                    handled.add(oa.id)
                for ag_other in self.agents:
                    if ag_other.id == agent.id or ag_other.id in handled:
                        continue
                    ag_other.receive_message(
                        agent.id,
                        message_content,
                        step=self.step,
                        from_name=sender_name,
                    )

        # Phase 3: Collect action decisions from all agents (with position information and message content).
        # LLM calls are executed in parallel across agents (when parallel_workers > 1).
        action_decisions: List[Tuple[Agent, Dict, List[Agent]]] = [None] * len(message_decisions)
        memory_reasoning_records: List[Optional[Dict]] = [None] * len(message_decisions)
        phase3_tasks: List[Tuple[int, Agent, List[Agent], Optional[Dict], str, Optional[List[Dict]], Optional[List[Dict]]]] = []
        skipped_p3 = 0

        for idx, (agent, message_decision, nearby_agents, _partner_id) in enumerate(message_decisions):
            # 修正1: host (企業受け入れ担当) は不動。Phase 3 を skip して action_type=stay 強制。
            # 学生が来ない時に host が wander/approach で歩き回って配置から離れ、結果として
            # 「学生が来た頃には host が居ない」現象を防ぐ。
            if bool(agent.persona.get('is_host')):
                memory = (message_decision.get('memory') or '').strip()
                reasoning = (message_decision.get('reasoning') or '').strip()
                action_decision = {
                    "action_type": "stay",
                    "target_place": None,
                    "target_agent": None,
                    "direction": None,
                    "memory": memory,
                    "reasoning": reasoning or "(host 不動: 自分の配置で学生を待機)",
                    "action": "stay",
                }
                action_decisions[idx] = (agent, action_decision, nearby_agents)
                memory_reasoning_records[idx] = {
                    "step": self.step,
                    "time": self._current_time_str(),
                    "id": agent.id,
                    "name": agent.persona.get('name', f"Agent {agent.id}"),
                    "layer": self._get_agent_layer_jp(agent),
                    "memory": memory,
                    "reasoning": reasoning or "(host 不動: 自分の配置で学生を待機)",
                }
                continue
            if self.skip_probability > 0 and random.random() < self.skip_probability:
                action_decision = {
                    "action_type": "stay",
                    "target_place": None,
                    "target_agent": None,
                    "direction": None,
                    "memory": "",
                    "reasoning": "(思考スキップ)",
                    "action": "stay",
                }
                action_decisions[idx] = (agent, action_decision, nearby_agents)
                memory_reasoning_records[idx] = {
                    "step": self.step,
                    "time": self._current_time_str(),
                    "id": agent.id,
                    "name": agent.persona.get('name', f"Agent {agent.id}"),
                    "layer": self._get_agent_layer_jp(agent),
                    "memory": "",
                    "reasoning": "(思考スキップ)",
                }
                skipped_p3 += 1
                continue
            if self.skip_decision_prompt:
                # Cost-reduction path: no separate decision LLM call. The
                # memory/reasoning were captured in the message JSON (see
                # _create_message_prompts_minimal with skip_decision=True).
                memory = message_decision.get('memory', '') or ''
                reasoning = message_decision.get('reasoning', '') or ''
                stub_decision = {
                    "action_type": "stay",
                    "target_place": None,
                    "target_agent": None,
                    "direction": None,
                    "memory": memory,
                    "reasoning": reasoning,
                    "action": "stay",
                }
                action_decisions[idx] = (agent, stub_decision, nearby_agents)
                memory_reasoning_records[idx] = {
                    "step": self.step,
                    "time": self._current_time_str(),
                    "id": agent.id,
                    "name": agent.persona.get('name', f"Agent {agent.id}"),
                    "layer": self._get_agent_layer_jp(agent),
                    "memory": memory,
                    "reasoning": reasoning,
                }
                continue
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            message_content = message_decision.get('message', '')
            fire_info = self.get_fire_info_for_agent(agent)
            events_info = self.get_active_events_for_agent(agent)
            phase3_tasks.append((idx, agent, nearby_agents, agent_place_status, message_content, fire_info, events_info))

        if phase3_tasks:
            workers = max(1, min(self.parallel_workers, len(phase3_tasks)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        ag.decide_action, ps, nb, self.step, mc, fire_info=fi, events_info=ei
                    )
                    for _, ag, nb, ps, mc, fi, ei in phase3_tasks
                ]
                for (idx, agent, nb, _, _, _, _), fut in zip(phase3_tasks, futures):
                    try:
                        decision = fut.result()
                    except Exception as e:
                        logger.error(f"Agent {agent.id} Phase 3 parallel execution failed: {e}")
                        decision = {
                            "action_type": "stay",
                            "target_place": None,
                            "target_agent": None,
                            "direction": None,
                            "memory": "",
                            "reasoning": "Parallel execution error",
                            "action": "stay",
                        }
                    action_decisions[idx] = (agent, decision, nb)
                    memory_reasoning_records[idx] = {
                        "step": self.step,
                        "time": self._current_time_str(),
                        "id": agent.id,
                        "name": agent.persona.get('name', f"Agent {agent.id}"),
                        "layer": self._get_agent_layer_jp(agent),
                        "memory": decision.get('memory', ''),
                        "reasoning": decision.get('reasoning', ''),
                    }

        # Drop None entries defensively (should not happen if everything ran).
        action_decisions = [a for a in action_decisions if a is not None]
        memory_reasoning_records = [r for r in memory_reasoning_records if r is not None]

        if self.skip_probability > 0:
            logger.debug(
                f"Step {self.step}: Skipped Phase1={skipped_p1}/{self.num_agents}, "
                f"Phase3={skipped_p3}/{self.num_agents}"
            )
        
        # Write all memory/reasoning records in batch (more efficient than individual writes)
        self._log_memory_reasoning_batch(memory_reasoning_records)

        # smoke20: 行動ログ強化 — Phase 4 (movement) 直前に position_before を集め、
        # 直後に position_after を加えて action.jsonl に出力。
        # 「LLM が target_place を A に指定したのに position は B 方向に動いた」という
        # 不整合をピンポイント診断するため。Phase B のみ phase_b_perceive_enabled で gating。
        action_log_records = []
        if getattr(self, 'phase_b_perceive_enabled', False):
            for agent, action_decision, _nb in action_decisions:
                pb = list(agent.position) if agent.position is not None else [None, None]
                action_log_records.append({
                    "step": self.step,
                    "time": self._current_time_str(),
                    "id": agent.id,
                    "name": agent.persona.get('name', f'#{agent.id}'),
                    "action_type": action_decision.get('action_type'),
                    "target_place": action_decision.get('target_place'),
                    "target_agent": action_decision.get('target_agent'),
                    "direction": action_decision.get('direction'),
                    "position_before": [round(pb[0], 2) if pb[0] is not None else None,
                                        round(pb[1], 2) if pb[1] is not None else None],
                    "current_place_before": agent.current_place,
                })

        # Phase 4: Execute movement (after messages are sent and actions are decided)
        for agent, action_decision, nearby_agents in action_decisions:
            agent.execute_intent(action_decision, nearby_agents=nearby_agents)

        # Update states after movement
        for agent in self.agents:
            agent.update_state(self.places)

        # smoke20: position_after / current_place_after を埋めて action.jsonl に書き出す
        if action_log_records:
            for rec, (agent, _, _) in zip(action_log_records, action_decisions):
                pa = list(agent.position) if agent.position is not None else [None, None]
                rec["position_after"] = [round(pa[0], 2) if pa[0] is not None else None,
                                          round(pa[1], 2) if pa[1] is not None else None]
                rec["current_place_after"] = agent.current_place
                pb_x, pb_y = rec["position_before"]
                pa_x, pa_y = rec["position_after"]
                if pb_x is not None and pa_x is not None:
                    dx = pa_x - pb_x
                    dy = pa_y - pb_y
                    rec["moved_dx"] = round(dx, 2)
                    rec["moved_dy"] = round(dy, 2)
                    rec["moved_dist"] = round((dx * dx + dy * dy) ** 0.5, 2)
            try:
                action_log_path = os.path.join(self.output_dir, "actions.jsonl")
                with open(action_log_path, "a", encoding="utf-8") as f:
                    for rec in action_log_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"actions.jsonl write failed: {e}")

        # Phase B: perceive 注入機構 (place 進入時に perceive_pass / enter 行為で perceive_enter を agent.memory に注入)
        if getattr(self, 'phase_b_perceive_enabled', False):
            self._phase_b_inject_perceive(action_decisions)
            self._phase_b_log_body_sense()

        # Feature 2: evolve the relationship graph once per step, after all
        # messages for this step have been delivered (last_step_messages is
        # populated in Phase 2).
        self._update_relationships()
        self._log_relationships_snapshot()
        self._log_event_awareness()

        # 圧縮記憶 (2026-05-04): step が 5 の倍数のたびに、各 agent の直近 5 件 memory を
        # 1 文要約して archived_summaries に push (Gemini 呼び出し)。並列で。
        if self.step > 0 and self.step % 5 == 0:
            try:
                workers = max(1, min(self.parallel_workers, len(self.agents)))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(lambda a: a.maybe_compress_memory(self.step), self.agents))
                n_arch = sum(1 for a in self.agents if a.archived_summaries)
                logger.info(f"Memory compression at step {self.step}: agents with archive: {n_arch}/{len(self.agents)}")
            except Exception as e:
                logger.warning(f"Memory compression failed at step {self.step}: {e}")

        # Record statistics
        agents_in_place = len(self.get_agents_in_place())
        overall_status = self.get_place_status()
        self.stats['place_occupancy'].append(overall_status['occupancy_rate'])
        self.stats['agents_in_place'].append(agents_in_place)
        self.stats['agents_outside_place'].append(len(self.agents) - agents_in_place)
        
        # Record per-place statistics
        for place in self.places:
            place_status = self.get_place_status(place['name'])
            self.stats['places'][place['name']]['occupancy'].append(place_status['occupancy_rate'])
            self.stats['places'][place['name']]['agents_in_place'].append(place_status['agents_in_place'])
        
        # Record fire statistics (count agents in any active fire radius)
        if self.fire_states:
            agents_in_any_fire = set()
            for fire in self.fire_states:
                if fire.get('active'):
                    for agent in self.agents:
                        if agent.distance_to(fire['position']) <= fire['radius']:
                            agents_in_any_fire.add(agent.id)
            self.stats['agents_in_fire_radius'].append(len(agents_in_any_fire))
        else:
            self.stats['agents_in_fire_radius'].append(0)

        # Store history
        self.history.append({
            'step': self.step,
            'place_status': overall_status,
            'agent_positions': [agent.position for agent in self.agents],
            'agents_in_place': [agent.id for agent in self.get_agents_in_place()],
            'fire_states': list(self.fire_states),
        })

        # Feature 5: capture per-step snapshot for the Canvas viewer.
        self._append_viewer_snapshot(memory_reasoning_records)
        
        if self.step % LOG_INTERVAL == 0:
            place_info = ", ".join([
                f"{place['name']}: {self.get_place_status(place['name'])['agents_in_place']}"
                for place in self.places
            ])
            logger.info(
                f"Step {self.step}/{self.duration}: "
                f"{agents_in_place} agents in places ({place_info}), "
                f"{overall_status['occupancy_rate']:.1%} overall occupancy"
            )
    
    def run(self):
        """Run the full simulation"""
        logger.info("Starting simulation...")
        
        # Check LLM API connection
        if not self.llm_client.check_connection():
            logger.error("Cannot connect to LLM API. Check ANTHROPIC_API_KEY and network.")
            return
        
        # Initialize agents
        self.initialize_agents()
        
        # Run simulation
        try:
            while self.step < self.duration:
                self.step_simulation()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error during simulation: {e}", exc_info=True)

        # Feature 5: export the unified Canvas-viewer dataset at the end.
        self.export_simulation_data()

        logger.info("Simulation completed")
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        if not self.stats['place_occupancy']:
            return {}
        
        place_occupancy = np.array(self.stats['place_occupancy'])
        agents_in_place = np.array(self.stats['agents_in_place'])
        
        return {
            'mean_occupancy': float(np.mean(place_occupancy)),
            'std_occupancy': float(np.std(place_occupancy)),
            'mean_agents_in_place': float(np.mean(agents_in_place)),
            'max_agents_in_place': int(np.max(agents_in_place)),
            'min_agents_in_place': int(np.min(agents_in_place)),
            'total_steps': self.step
        }

