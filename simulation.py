"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import os
import random
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from agent import Agent
from claude_client import ClaudeClient
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
    
    def __init__(self, config_path: str = "config.yaml", output_dir: Optional[str] = None):
        """Initialize simulation from config file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Output directory for logs
        self.output_dir = output_dir
        
        # Simulation parameters
        sim_config = self.config['simulation']
        self.duration = sim_config['duration']
        self.half_space_size = sim_config['half_space_size']
        self.half_place_size = sim_config.get('half_place_size', 5)
        
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
        base_required = ['name', 'type', 'center_x', 'center_y', 'capacity']
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

        # LLM parameters
        llm_config = self.config['llm']
        self.llm_client = ClaudeClient(
            base_url=llm_config['base_url'],
            model=llm_config['model'],
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 200),
        )
        
        # Initialize agents
        self.agents: List[Agent] = []
        self.step = 0
        self.history: List[Dict] = []
        
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
        """Apply time-pattern spawn/despawn probabilities for this step."""
        if not self.spawn_enabled:
            return
        pattern = self._get_time_pattern()
        if pattern is None:
            return
        spawn_places = self._find_spawn_places()
        if not spawn_places:
            return

        enter_p = float(pattern.get('enter_per_step', 0.0))
        exit_p = float(pattern.get('exit_per_step', 0.0))

        if enter_p > 0 and random.random() < enter_p and len(self.agents) < self.max_agents:
            station = random.choice(spawn_places)
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
        from_agent_id: int,
        to_agent_id: int,
        message: str,
        reasoning: str = ""
    ) -> None:
        """Log a message to messages.jsonl file"""
        if not self.output_dir:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        messages_file = os.path.join(self.output_dir, "messages.jsonl")
        record = {
            "step": self.step,
            "from": from_agent_id,
            "to": to_agent_id,
            "message": message,
            "reasoning": reasoning
        }

        with open(messages_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

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
        with open(memory_reasoning_file, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _generate_random_position(self) -> Tuple[int, int]:
        """Generate a random position within the space (origin-centered coordinate system)"""
        return (
            random.randint(-self.half_space_size, self.half_space_size),
            random.randint(-self.half_space_size, self.half_space_size)
        )
    
    def _generate_initial_positions(self, avoid_places: bool = True) -> List[Tuple[int, int]]:
        """Generate initial positions for agents"""
        positions: List[Tuple[int, int]] = []
        used_positions: Set[Tuple[int, int]] = set()
        attempts = 0
        
        while len(positions) < self.num_agents and attempts < MAX_POSITION_ATTEMPTS:
            position = self._generate_random_position()
            
            # Skip if position is already used
            if position in used_positions:
                attempts += 1
                continue
            
            # Skip if position is in any place and we want to avoid it
            if avoid_places and self._is_position_in_place(position):
                attempts += 1
                continue
            
            positions.append(position)
            used_positions.add(position)
            attempts += 1
        
        # If we couldn't generate enough positions avoiding places, fill remaining
        if len(positions) < self.num_agents:
            logger.warning(
                f"Could only generate {len(positions)} unique positions avoiding places. "
                "Using all available space."
            )
            while len(positions) < self.num_agents:
                position = self._generate_random_position()
                if position not in used_positions:
                    positions.append(position)
                    used_positions.add(position)
        
        return positions
    
    def initialize_agents(self):
        """Initialize agents at random positions, attaching a persona to each."""
        logger.info(f"Initializing {self.num_agents} agents...")

        positions = self._generate_initial_positions(avoid_places=True)

        personas_config = self.config.get('agents', {}).get('personas', []) or []
        personas_by_id = {p['id']: p for p in personas_config if 'id' in p}

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
            capacity = place_config['capacity']
            occupancy_rate = agents_in_place / capacity

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
                place_capacity = place['capacity']
                place_occupancy_rate = place_agents / place_capacity

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

        # Update agent states
        for agent in self.agents:
            agent.update_state(self.places)

        # Phase 1: Collect message decisions from all agents (without position information).
        # LLM calls are executed in parallel across agents (when parallel_workers > 1).
        skipped_p1 = 0
        phase1_tasks: List[Tuple[Agent, List[Agent], Optional[Dict], Optional[List[Dict]]]] = []
        phase1_results: List[Optional[Tuple[Agent, Dict, List[Agent]]]] = [None] * len(self.agents)
        for idx, agent in enumerate(self.agents):
            nearby_agents = agent.get_nearby_agents(self.agents)
            if self.skip_probability > 0 and random.random() < self.skip_probability:
                phase1_results[idx] = (agent, {"message": "", "reasoning": "Skipped (random)"}, nearby_agents)
                skipped_p1 += 1
                continue
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            fire_info = self.get_fire_info_for_agent(agent)
            phase1_tasks.append((agent, nearby_agents, agent_place_status, fire_info))
            # Reserve slot — filled in after the executor returns.
            phase1_results[idx] = None

        if phase1_tasks:
            workers = max(1, min(self.parallel_workers, len(phase1_tasks)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        ag.decide_message, ps, nb, self.step, fire_info=fi
                    )
                    for ag, nb, ps, fi in phase1_tasks
                ]
                task_results = []
                for (ag, nb, _, _), fut in zip(phase1_tasks, futures):
                    try:
                        decision = fut.result()
                    except Exception as e:
                        logger.error(f"Agent {ag.id} Phase 1 parallel execution failed: {e}")
                        decision = {"message": "", "reasoning": "Parallel execution error"}
                    task_results.append((ag, decision, nb))

            # Merge task_results back into phase1_results in original agent order.
            result_iter = iter(task_results)
            for idx, slot in enumerate(phase1_results):
                if slot is None:
                    phase1_results[idx] = next(result_iter)

        message_decisions: List[Tuple[Agent, Dict, List[Agent]]] = [r for r in phase1_results if r is not None]

        # Phase 2: Send messages (using decision-time nearby agents, before movement)
        for agent, message_decision, nearby_agents in message_decisions:
            message_content = message_decision.get('message', '')
            if message_content and nearby_agents:
                sender_name = agent.persona.get('name') or f"Agent {agent.id}"
                logger.info(
                    f"Step {self.step}: {sender_name} sends message to {len(nearby_agents)} nearby agent(s): "
                    f"\"{message_content}\""
                )
                for other_agent in nearby_agents:
                    other_agent.receive_message(
                        agent.id,
                        message_content,
                        step=self.step,
                        from_name=sender_name,
                    )
                    # Log message to jsonl file
                    self._log_message(
                        from_agent_id=agent.id,
                        to_agent_id=other_agent.id,
                        message=message_content,
                        reasoning=message_decision.get('reasoning', '')
                    )

        # Phase 3: Collect action decisions from all agents (with position information and message content).
        # LLM calls are executed in parallel across agents (when parallel_workers > 1).
        action_decisions: List[Tuple[Agent, Dict, List[Agent]]] = [None] * len(message_decisions)
        memory_reasoning_records: List[Optional[Dict]] = [None] * len(message_decisions)
        phase3_tasks: List[Tuple[int, Agent, List[Agent], Optional[Dict], str, Optional[List[Dict]]]] = []
        skipped_p3 = 0

        for idx, (agent, message_decision, nearby_agents) in enumerate(message_decisions):
            if self.skip_probability > 0 and random.random() < self.skip_probability:
                action_decision = {
                    "action_type": "stay",
                    "target_place": None,
                    "target_agent": None,
                    "direction": None,
                    "memory": "",
                    "reasoning": "Skipped (random)",
                    "action": "stay",
                }
                action_decisions[idx] = (agent, action_decision, nearby_agents)
                memory_reasoning_records[idx] = {
                    "step": self.step,
                    "id": agent.id,
                    "memory": "",
                    "reasoning": "Skipped (random)",
                }
                skipped_p3 += 1
                continue
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            message_content = message_decision.get('message', '')
            fire_info = self.get_fire_info_for_agent(agent)
            phase3_tasks.append((idx, agent, nearby_agents, agent_place_status, message_content, fire_info))

        if phase3_tasks:
            workers = max(1, min(self.parallel_workers, len(phase3_tasks)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        ag.decide_action, ps, nb, self.step, mc, fire_info=fi
                    )
                    for _, ag, nb, ps, mc, fi in phase3_tasks
                ]
                for (idx, agent, nb, _, _, _), fut in zip(phase3_tasks, futures):
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
                        "id": agent.id,
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

        # Phase 4: Execute movement (after messages are sent and actions are decided)
        for agent, action_decision, nearby_agents in action_decisions:
            agent.execute_intent(action_decision, nearby_agents=nearby_agents)

        # Update states after movement
        for agent in self.agents:
            agent.update_state(self.places)
        
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

