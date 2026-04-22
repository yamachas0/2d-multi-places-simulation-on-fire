"""Passability / pathfinding layer.

Builds a 2D walkable mask from `config.scene_3d` (roads / decks / stairs) plus
`config.places` (facility interiors). Agents can only transit through walkable
cells; "black" (non-walkable) areas outside facilities are blocked.

Places get a built-in last-mile exception: when a target lies inside a place,
movement is routed to the nearest walkable cell, then allowed to "jump" into
the place even if the apron between road and building is black.

If no `scene_3d` is present in config, the navigator falls back to
"everything walkable" — preserves legacy behaviour.
"""

import logging
import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Cell = Tuple[int, int]


class Navigator:
    """Grid-based pathfinder over a precomputed walkable mask."""

    # Place types whose INTERIOR counts as walkable transit (open-air spaces
    # that agents literally walk across). Buildings are NOT walkable transit —
    # they're destinations entered via apron-jump in step_toward().
    WALKABLE_PLACE_TYPES = frozenset({
        'park', 'plaza', 'pedestrian_street', 'wide_street', 'narrow_street',
    })

    def __init__(self, config: Dict, half_space_size: int):
        self.H = int(half_space_size)
        self.size = 2 * self.H + 1
        # mask[gy, gx] = 1 walkable, 0 blocked. Grid origin (gx=H, gy=H) ↔ world (0, 0).
        self.mask = np.zeros((self.size, self.size), dtype=np.uint8)

        scene = (config or {}).get('scene_3d') or {}
        n_marked = 0
        for r in scene.get('roads') or []:
            n_marked += self._fill_rect(r)
        for d in scene.get('decks') or []:
            for seg in (d.get('segments') or []):
                n_marked += self._fill_deck_segment(seg)
        for s in scene.get('stairs') or []:
            n_marked += self._fill_stair(s)
        # Only open-air place types (parks/plazas/streets) are walkable transit.
        # Building interiors are reached via apron-jump, not by pathing through.
        for p in (config or {}).get('places') or []:
            if (p.get('type') or '') in self.WALKABLE_PLACE_TYPES:
                n_marked += self._fill_place(p)

        scene_has_content = bool(
            scene.get('roads') or scene.get('decks') or scene.get('stairs')
        )
        if not scene_has_content:
            # Legacy mode: no scene_3d → entire field walkable.
            self.mask.fill(1)
            self.constrained = False
            logger.info("Navigator: no scene_3d → unconstrained (legacy mode)")
        else:
            self.constrained = True
            walkable_pct = 100.0 * float(self.mask.sum()) / float(self.size * self.size)
            logger.info(
                f"Navigator: constrained mode | walkable {self.mask.sum()} / "
                f"{self.size*self.size} cells ({walkable_pct:.1f}%)"
            )

        # Per-place entry hint: a walkable cell adjacent-ish to each place, used
        # by step_toward() so agents can jump the black apron into the building.
        self._place_entry: Dict[str, Cell] = {}
        for p in (config or {}).get('places') or []:
            name = p.get('name')
            if not name:
                continue
            entry = self._find_place_entry(p)
            if entry is not None:
                self._place_entry[name] = entry

    # ---- mask building ----------------------------------------------------

    def _xy_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(x)) + self.H, int(round(y)) + self.H

    def _mark(self, x: float, y: float) -> int:
        gx, gy = self._xy_to_grid(x, y)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            if self.mask[gy, gx] == 0:
                self.mask[gy, gx] = 1
                return 1
        return 0

    def _fill_rect(self, road: Dict) -> int:
        axis = (road.get('axis') or 'ew').lower()
        cx = float(road.get('center_x', 0))
        cy = float(road.get('center_y', 0))
        w = float(road.get('width_cells', 2))
        L = float(road.get('length_cells', 10))
        hw = max(0.5, w / 2.0)
        hL = max(0.5, L / 2.0)
        if axis == 'ew':
            x0, x1 = cx - hL, cx + hL
            y0, y1 = cy - hw, cy + hw
        else:
            x0, x1 = cx - hw, cx + hw
            y0, y1 = cy - hL, cy + hL
        n = 0
        for y in range(int(math.floor(y0)), int(math.floor(y1)) + 1):
            for x in range(int(math.floor(x0)), int(math.floor(x1)) + 1):
                n += self._mark(x, y)
        return n

    def _fill_deck_segment(self, seg: Dict) -> int:
        f = seg.get('from') or [0, 0]
        t = seg.get('to') or [0, 0]
        w = float(seg.get('width', 3))
        return self._draw_thick_line(
            float(f[0]), float(f[1]), float(t[0]), float(t[1]), w
        )

    def _fill_stair(self, s: Dict) -> int:
        g = s.get('ground_position') or [0, 0]
        d = s.get('deck_position') or [0, 0]
        w = float(s.get('width_cells', 3))
        return self._draw_thick_line(
            float(g[0]), float(g[1]), float(d[0]), float(d[1]), w
        )

    def _fill_place(self, p: Dict) -> int:
        cx = float(p.get('center_x', 0))
        cy = float(p.get('center_y', 0))
        hx = float(p.get('half_size_x', p.get('half_size', 2)))
        hy = float(p.get('half_size_y', p.get('half_size', 2)))
        n = 0
        for y in range(int(math.floor(cy - hy)), int(math.floor(cy + hy)) + 1):
            for x in range(int(math.floor(cx - hx)), int(math.floor(cx + hx)) + 1):
                n += self._mark(x, y)
        return n

    def _draw_thick_line(
        self, x1: float, y1: float, x2: float, y2: float, width: float
    ) -> int:
        length = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(1, int(math.ceil(length * 2)))
        hw = max(0.5, width / 2.0)
        hw_i = int(math.floor(hw))
        n = 0
        for i in range(steps + 1):
            t = i / steps
            cx = x1 + (x2 - x1) * t
            cy = y1 + (y2 - y1) * t
            for dy in range(-hw_i, hw_i + 1):
                for dx in range(-hw_i, hw_i + 1):
                    n += self._mark(cx + dx, cy + dy)
        return n

    # ---- queries ----------------------------------------------------------

    def is_walkable(self, x: float, y: float) -> bool:
        gx, gy = self._xy_to_grid(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        return bool(self.mask[gy, gx])

    def nearest_walkable(self, x: float, y: float, max_radius: int = 30) -> Optional[Cell]:
        gx, gy = self._xy_to_grid(x, y)
        if 0 <= gx < self.size and 0 <= gy < self.size and self.mask[gy, gx]:
            return (int(round(x)), int(round(y)))
        best = self._nearest_walkable_grid(gx, gy, max_radius)
        if best is None:
            return None
        return (best[0] - self.H, best[1] - self.H)

    def _nearest_walkable_grid(
        self, gx: int, gy: int, max_radius: int = 30
    ) -> Optional[Tuple[int, int]]:
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.mask[ny, nx]:
                        return (nx, ny)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.mask[ny, nx]:
                        return (nx, ny)
        return None

    def _find_place_entry(self, place: Dict) -> Optional[Cell]:
        """Return a walkable cell just outside the place bbox (nearest road/deck)."""
        cx = int(round(float(place.get('center_x', 0))))
        cy = int(round(float(place.get('center_y', 0))))
        hx = int(math.ceil(float(place.get('half_size_x', place.get('half_size', 2)))))
        hy = int(math.ceil(float(place.get('half_size_y', place.get('half_size', 2)))))
        # Spiral outward starting just beyond the bbox.
        for extra in range(1, 15):
            for y in range(cy - hy - extra, cy + hy + extra + 1):
                for x in (cx - hx - extra, cx + hx + extra):
                    if self.is_walkable(x, y):
                        return (x, y)
            for x in range(cx - hx - extra + 1, cx + hx + extra):
                for y in (cy - hy - extra, cy + hy + extra):
                    if self.is_walkable(x, y):
                        return (x, y)
        return None

    # ---- pathfinding ------------------------------------------------------

    def find_path(
        self, start: Cell, goal: Cell, max_nodes: int = 40000
    ) -> List[Cell]:
        """BFS path from `start` to `goal`. If goal is blocked/unreachable,
        returns a path to the closest reachable walkable cell instead.
        Path is a list of (x, y) world-cells including start and ending cell."""
        sx, sy = int(round(start[0])), int(round(start[1]))
        gx_s, gy_s = self._xy_to_grid(sx, sy)
        gx_g, gy_g = self._xy_to_grid(goal[0], goal[1])

        # If start is blocked, snap to the nearest walkable cell. Keeps agents
        # from getting stuck after a config change made their current cell dead.
        if not (0 <= gx_s < self.size and 0 <= gy_s < self.size):
            return [start]
        if not self.mask[gy_s, gx_s]:
            snap = self._nearest_walkable_grid(gx_s, gy_s)
            if snap is None:
                return [start]
            gx_s, gy_s = snap
            sx, sy = gx_s - self.H, gy_s - self.H

        # If goal cell is blocked, BFS still runs — we'll track the closest cell
        # to goal that we reach, and return a path to that.
        visited = np.zeros_like(self.mask, dtype=np.uint8)
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        q = deque()
        q.append((gx_s, gy_s))
        visited[gy_s, gx_s] = 1

        best = (gx_s, gy_s)
        best_d = abs(gx_s - gx_g) + abs(gy_s - gy_g)
        expanded = 0

        while q and expanded < max_nodes:
            x, y = q.popleft()
            expanded += 1
            if x == gx_g and y == gy_g:
                best = (x, y)
                break
            d = abs(x - gx_g) + abs(y - gy_g)
            if d < best_d:
                best_d = d
                best = (x, y)
                if d == 0:
                    break
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.size and 0 <= ny < self.size
                    and not visited[ny, nx] and self.mask[ny, nx]
                ):
                    visited[ny, nx] = 1
                    parent[(nx, ny)] = (x, y)
                    q.append((nx, ny))

        # Reconstruct.
        path_grid: List[Tuple[int, int]] = []
        cur = best
        while cur != (gx_s, gy_s):
            path_grid.append(cur)
            cur = parent.get(cur)
            if cur is None:
                break
        path_grid.append((gx_s, gy_s))
        path_grid.reverse()
        return [(gx - self.H, gy - self.H) for gx, gy in path_grid]

    # ---- movement primitives ---------------------------------------------

    def step_toward(
        self,
        start: Cell,
        goal: Cell,
        distance: int,
        target_place: Optional[Dict] = None,
    ) -> Cell:
        """Advance up to `distance` cells along the walkable path from start
        toward goal. If the path can't reach goal but target_place is set and
        the closest reachable cell is in its entry apron, allow a direct
        step INTO the place (last-mile facility entry)."""
        distance = max(1, int(distance))

        # Fast-path: straight-line if already inside the target place.
        if target_place and self._inside_place(start, target_place):
            return self._clamp(start)

        path = self.find_path(start, goal)
        if len(path) <= 1:
            # No route — but if target_place is given and our current cell is
            # already the place's registered entry, jump in.
            if target_place:
                name = target_place.get('name')
                entry = self._place_entry.get(name) if name else None
                if entry and abs(start[0] - entry[0]) <= 1 and abs(start[1] - entry[1]) <= 1:
                    return self._clamp(self._place_center(target_place))
            return self._clamp(start)

        idx = min(distance, len(path) - 1)
        next_pos = path[idx]

        # Last-mile facility entry: we've walked to the closest walkable
        # cell near the target place but path can't enter (because the apron
        # between road and building is blocked). Allow the jump.
        if target_place and idx == len(path) - 1 and not self._inside_place(next_pos, target_place):
            cx = int(round(float(target_place.get('center_x', 0))))
            cy = int(round(float(target_place.get('center_y', 0))))
            hx = float(target_place.get('half_size_x', target_place.get('half_size', 2)))
            hy = float(target_place.get('half_size_y', target_place.get('half_size', 2)))
            dx = abs(next_pos[0] - cx) - hx
            dy = abs(next_pos[1] - cy) - hy
            apron_dist = max(0.0, dx) + max(0.0, dy)
            # Jump if we're within a few cells of the building bbox.
            if apron_dist <= 6:
                return self._clamp(self._place_center(target_place))

        return self._clamp(next_pos)

    def _inside_place(self, pos: Cell, place: Dict) -> bool:
        cx = float(place.get('center_x', 0))
        cy = float(place.get('center_y', 0))
        hx = float(place.get('half_size_x', place.get('half_size', 2)))
        hy = float(place.get('half_size_y', place.get('half_size', 2)))
        return abs(pos[0] - cx) <= hx and abs(pos[1] - cy) <= hy

    def _place_center(self, place: Dict) -> Cell:
        return (
            int(round(float(place.get('center_x', 0)))),
            int(round(float(place.get('center_y', 0)))),
        )

    def _clamp(self, pos: Cell) -> Cell:
        x = max(-self.H, min(self.H, int(round(pos[0]))))
        y = max(-self.H, min(self.H, int(round(pos[1]))))
        return (x, y)

    # ---- initial placement -----------------------------------------------

    def sample_walkable_cell(
        self,
        rng: random.Random,
        used: Optional[set] = None,
        max_attempts: int = 500,
    ) -> Optional[Cell]:
        used = used or set()
        for _ in range(max_attempts):
            gx = rng.randint(0, self.size - 1)
            gy = rng.randint(0, self.size - 1)
            if not self.mask[gy, gx]:
                continue
            pos = (gx - self.H, gy - self.H)
            if pos in used:
                continue
            return pos
        # Fallback: scan deterministically.
        walkable_cells = np.argwhere(self.mask == 1)
        for gy, gx in walkable_cells:
            pos = (int(gx) - self.H, int(gy) - self.H)
            if pos not in used:
                return pos
        return None

    def validate_step(self, from_pos: Cell, to_pos: Cell) -> Cell:
        """Snap a proposed destination onto the walkable mask by routing
        through it. Used for `walk_along` where the LLM picked a raw direction
        that may point into the black."""
        if self.is_walkable(to_pos[0], to_pos[1]):
            return self._clamp(to_pos)
        # Walk as far as possible along the proposed vector.
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return self._clamp(from_pos)
        cur = from_pos
        for i in range(1, steps + 1):
            nx = from_pos[0] + int(round(dx * i / steps))
            ny = from_pos[1] + int(round(dy * i / steps))
            if self.is_walkable(nx, ny):
                cur = (nx, ny)
            else:
                break
        return self._clamp(cur)
