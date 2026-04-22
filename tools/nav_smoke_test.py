"""Navigation-only smoke test for shinagawa_nav_test.yaml.

No Simulation instance, no LLM calls. Just loads the config, constructs a
Navigator, and checks:
  - How many cells of the grid are walkable
  - Whether each place has at least one walkable cell nearby (the apron)
  - For N random start positions, whether BFS can reach each place
  - Which places are UNREACHABLE from most starting points (dead zones)
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navigation import Navigator  # noqa: E402

CONFIG = ROOT / "shinagawa_nav_test.yaml"
N_STARTS = 40
N_GOALS_PER_START = 48  # every place from every start


def load_config(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    sh = re.compile(r"^(\s*-\s)(.+?\s;\s.+)$", re.MULTILINE)
    def norm(m):
        prefix, body = m.group(1), m.group(2).strip()
        fields = [f.strip() for f in body.split(";") if f.strip()]
        return f"{prefix}{{ {', '.join(fields)} }}"
    return yaml.safe_load(sh.sub(norm, raw))


def main() -> int:
    cfg = load_config(CONFIG)
    half = cfg["simulation"]["half_space_size"]
    nav = Navigator(cfg, half)

    if not nav.constrained:
        print("[err] navigator is NOT constrained — scene_3d is empty")
        return 2

    # Count walkable cells
    total = 0
    walk = 0
    for y in range(-half, half):
        for x in range(-half, half):
            total += 1
            if nav.is_walkable(x, y):
                walk += 1
    print(f"grid          : {total} cells total, {walk} walkable ({walk/total*100:.1f}%)")

    places = cfg.get("places", [])
    # Check each place has a walkable entry hint
    place_entries = []
    missing_entry = []
    for p in places:
        name = p.get("name") or "(no-name)"
        hint = nav._place_entry.get(name) if hasattr(nav, "_place_entry") else None
        if hint is None:
            missing_entry.append(name)
        else:
            place_entries.append((name, hint))
    print(f"places        : {len(places)} total, "
          f"{len(place_entries)} with walkable entry, {len(missing_entry)} without")
    if missing_entry:
        print("  unreachable place entries (no walkable cell near border):")
        for n in missing_entry[:12]:
            print(f"    - {n}")
        if len(missing_entry) > 12:
            print(f"    ... (+{len(missing_entry)-12} more)")

    # Sample N_STARTS random walkable cells as agent start positions
    rng = random.Random(42)
    used = set()
    starts = []
    for _ in range(N_STARTS):
        pos = nav.sample_walkable_cell(rng, used)
        if pos is None:
            break
        starts.append(pos)
        used.add(pos)
    print(f"starts        : sampled {len(starts)} walkable start cells")

    # For each start, try to BFS-reach a random subset of place centers
    unreach_counter: Counter = Counter()
    total_tries = 0
    total_fail = 0
    # Also record which starts had ANY failure (= isolated starts)
    isolated_starts = []
    for i, s in enumerate(starts):
        goals = places if N_GOALS_PER_START >= len(places) else rng.sample(places, N_GOALS_PER_START)
        fails_here = 0
        for p in goals:
            gx, gy = int(round(p["center_x"])), int(round(p["center_y"]))
            path = nav.find_path(s, (gx, gy), max_nodes=60000)
            total_tries += 1
            # In actual sim, step_toward with target_place does apron-jump when
            # the path endpoint is within 6 cells of the place bounding box.
            # Mirror that logic here for a fair reachability verdict.
            reached = False
            if path:
                ex, ey = path[-1]
                hx = float(p.get("half_size_x", p.get("half_size", 2)))
                hy = float(p.get("half_size_y", p.get("half_size", 2)))
                dx = abs(ex - gx) - hx
                dy = abs(ey - gy) - hy
                apron = max(0.0, dx) + max(0.0, dy)
                if apron <= 6:
                    reached = True
            if not reached:
                total_fail += 1
                fails_here += 1
                unreach_counter[p.get("name") or "(no-name)"] += 1
        if fails_here > 0:
            isolated_starts.append((s, fails_here))
    fail_rate = total_fail / total_tries * 100 if total_tries else 0
    print(f"BFS tries     : {total_tries} attempted, {total_fail} failed "
          f"({fail_rate:.1f}% unreachable)")

    if unreach_counter:
        print("  most-unreachable places (from sampled starts):")
        for name, cnt in unreach_counter.most_common(12):
            print(f"    - {cnt:3d}x  {name}")
    if isolated_starts:
        print(f"  isolated starts (some goals unreachable): {len(isolated_starts)}/{len(starts)}")
        for s, f in isolated_starts[:8]:
            print(f"    - start={s}  unreachable_goals={f}")

    # Also: identify connected components of the walkable mask (rough)
    print("\nconnectivity check:")
    seen = set()
    components = []
    for s in starts:
        if s in seen:
            continue
        # BFS flood from this start within walkable mask
        q = [s]
        comp = set()
        while q:
            cur = q.pop()
            if cur in comp:
                continue
            comp.add(cur)
            x, y = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in comp:
                    continue
                if abs(nx) > half or abs(ny) > half:
                    continue
                if nav.is_walkable(nx, ny):
                    q.append((nx, ny))
        components.append((s, len(comp)))
        seen |= comp

    components.sort(key=lambda t: -t[1])
    print(f"  distinct components touched by starts: {len(components)}")
    for s, sz in components[:5]:
        print(f"    start={s}  component size={sz} cells")

    return 0


if __name__ == "__main__":
    sys.exit(main())
