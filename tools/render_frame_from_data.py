"""Regenerate a single matplotlib frame from simulation_data.json
without rerunning the LLM simulation. Used to quickly iterate on
visualization layout without paying Gemini costs."""
import argparse
import json
import math
import os
import sys
from pathlib import Path


class MockAgent:
    __slots__ = ('id', 'gender', 'in_place', 'current_place', 'position', 'persona')

    def __init__(self, id_, gender, in_place, current_place, pos, persona):
        self.id = id_
        self.gender = gender
        self.in_place = in_place
        self.current_place = current_place
        self.position = pos
        self.persona = persona or {}

    def distance_to(self, other_pos):
        return math.hypot(self.position[0] - other_pos[0], self.position[1] - other_pos[1])


def compute_place_status(agents, places):
    total_cap = sum(p.get('capacity', 0) for p in places)
    in_place_count = sum(1 for a in agents if a.in_place)
    per_place = {}
    for p in places:
        name = p['name']
        cap = p.get('capacity', 0)
        count = sum(1 for a in agents if a.in_place and a.current_place == name)
        per_place[name] = {
            'agents_in_place': count,
            'capacity': cap,
            'occupancy_rate': (count / cap) if cap else 0.0,
        }
    return {
        'agents_in_place': in_place_count,
        'occupancy_rate': (in_place_count / total_cap) if total_cap else 0.0,
        'places': per_place,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to simulation_data.json')
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--out', required=True, help='Output PNG path')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from visualization import Visualizer

    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    meta = data['metadata']
    places = data['places']
    personas = {p['id']: p for p in data['personas']}

    step_record = next((s for s in data['timeline'] if s['step'] == args.step), None)
    if step_record is None:
        raise SystemExit(f"Step {args.step} not found in timeline")

    agents = [
        MockAgent(
            id_=a['id'],
            gender=personas.get(a['id'], {}).get('gender', 'male'),
            in_place=a.get('in_place', False),
            current_place=a.get('current_place'),
            pos=(a['x'], a['y']),
            persona=personas.get(a['id']),
        )
        for a in step_record['agents']
    ]

    place_status = compute_place_status(agents, places)

    step_messages = []
    for c in step_record.get('conversations', []):
        fid = c.get('from_id') if 'from_id' in c else c.get('from')
        tid = c.get('to_id') if 'to_id' in c else c.get('to')
        if fid is not None and tid is not None:
            step_messages.append((fid, tid))

    viz = Visualizer(
        half_space_size=meta['half_space_size'],
        places=places,
        num_agents=len(personas),
    )
    viz.visualize_step(
        agents=agents,
        place_status=place_status,
        step=args.step,
        communication_radius=3.0,
        save_path=args.out,
        fire_states=[],
        time_str=step_record.get('time'),
        step_messages=step_messages,
    )
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
