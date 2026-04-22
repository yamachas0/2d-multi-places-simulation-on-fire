"""Merge scene_export.yaml (viewer output) into shinagawa_config.yaml.

Strategy:
- places       : export (user's canonical edits)
- scene_3d     : export as-is EXCEPT:
                 - 自由通路 deck -> demoted to an EW road (1F ground level)
                 - decks / stairs dropped entirely
- metadata / proposed_type_extensions / scenario : preserved from original
- personas     : not included here (pull from config_jr_disruption.yaml at sim time)

Output: shinagawa_config.merged.yaml (original is left untouched).
"""
import re
import sys
from pathlib import Path
import yaml

ORIG = Path(r"D:\ユーザー\ダウンロード\shinagawa_config.yaml")
EXPORT = Path(r"D:\ユーザー\ダウンロード\scene_export.yaml")
OUT = Path(r"D:\ユーザー\ダウンロード\shinagawa_config.merged.yaml")


def load_with_shorthand(path: Path) -> dict:
    """Load yaml, normalizing the non-standard `- axis: ns ; center_x: 1 ; ...` shorthand."""
    raw = path.read_text(encoding="utf-8")
    sh = re.compile(r"^(\s*-\s)(.+?\s;\s.+)$", re.MULTILINE)
    def norm(m):
        prefix, body = m.group(1), m.group(2).strip()
        fields = [f.strip() for f in body.split(";") if f.strip()]
        return f"{prefix}{{ {', '.join(fields)} }}"
    return yaml.safe_load(sh.sub(norm, raw))


DECK_ROAD_EXTEND_CELLS = 8  # pad each deck-road by 4 cells per end so it overlaps
                            # adjacent NS/EW roads after cell-grid discretization
                            # (prevents a 1-cell black gap breaking BFS connectivity)

# Places that the viewer-edit session left behind by accident (confirmed with
# user). These are dropped from the merged places list.
PLACES_TO_DROP = frozenset({
    "駅前プロムナード",
})

# Place-position overrides applied after export. Used to shift buildings that
# the viewer left outside the apron-jump threshold (6 cells) from the nearest
# road, so agents can actually reach them.
#   新高輪プリンスホテル : (-60, 11.3) → (-36, 11.3)  (reach via EW@y=20, apron ~0.2)
#   プリンスさくらタワー : (-41, 30.8) → (-32, 30.8)  (reach 国道15号 west edge, apron ~0.6)
#   高輪プリンスホテル   : (-43, 42.5) → (-38, 42.5)  (reach 国道15号 west edge, apron ~5.0)
PLACE_MOVES = {
    "新高輪プリンスホテル": {"center_x": -36.0},
    "プリンスさくらタワー": {"center_x": -32.0},
    "高輪プリンスホテル":   {"center_x": -38.0},
}

# Small facilities added post-export (user-requested landmarks not in the
# viewer-exported places).
PLACES_TO_ADD = [
    # JR改札 — at intersection of 自由通路 and JR本線 (rails center x=2.57, y≈0.29)
    {
        "name": "JR改札",
        "type": "subway_station",
        "center_x": 2.57,
        "center_y": 0.29,
        "half_size_x": 1.5,
        "half_size_y": 1.0,
        "capacity": 40,
        "attributes": {"height_m": 3},
        "social_likelihood": 0.2,
        "is_spawn_point": False,
    },
    # 京急改札 — at intersection of 自由通路 and 京急本線 (rails center x=-5.74, y≈0.29)
    {
        "name": "京急改札",
        "type": "subway_station",
        "center_x": -5.74,
        "center_y": 0.29,
        "half_size_x": 1.0,
        "half_size_y": 1.0,
        "capacity": 25,
        "attributes": {"height_m": 3},
        "social_likelihood": 0.2,
        "is_spawn_point": False,
    },
    # タクシー乗り場 — west edge of 港南口広場 (plaza cx=30.4, west edge ~22)
    # Walkable plaza so agents can actually board/approach.
    {
        "name": "タクシー乗り場",
        "type": "plaza",
        "center_x": 22.5,
        "center_y": 5.0,
        "half_size_x": 1.5,
        "half_size_y": 1.5,
        "capacity": 20,
        "social_likelihood": 0.2,
        "is_spawn_point": False,
    },
    # バス乗り場 — west edge of 港南口広場, south of taxi
    {
        "name": "バス乗り場",
        "type": "plaza",
        "center_x": 22.5,
        "center_y": -1.0,
        "half_size_x": 1.5,
        "half_size_y": 1.5,
        "capacity": 25,
        "social_likelihood": 0.2,
        "is_spawn_point": False,
    },
    # 品川駅 東西自由通路 — the spawn point for v3 last-shinkansen scenario.
    # Duplicates the demoted deck-road's footprint as a walkable place so
    # persona initial_place can target it. center/size mirrors the road at
    # axis=ew, center=(0.33, 0.29), width=6, length≈35.
    {
        "name": "品川駅 東西自由通路",
        "type": "pedestrian_street",
        "center_x": 0.33,
        "center_y": 0.29,
        "half_size_x": 17.0,
        "half_size_y": 3.0,
        "capacity": 60,
        "attributes": {"height_m": 4},
        "social_likelihood": 0.3,
        "is_spawn_point": True,
    },
]

# EW connecting road injected to unify the north-20 corridor. Touches:
#   - 二本榎通り (x=-70.5), 国道15号 (-19.5), NS@17.6, NS@50.4, NS@69.6
#   - 高輪の森公園 (y=19±5), プリンスさくらタワー (y=30.8), 柘榴坂住宅群
# Single road that bridges ~6 previously-isolated places in one go.
INJECTED_EW_ROADS = [
    {
        "axis": "ew",
        "road_class": "local",
        "width_cells": 3,
        "length_cells": 150,
        "center_x": 0,
        "center_y": 20,
        "name": "(auto) EW連絡道 y=20",
    },
]


def deck_to_roads(deck: dict) -> list:
    """Convert a deck (with segments) into a list of 1F road dicts."""
    roads = []
    name = deck.get("name") or ""
    for i, seg in enumerate(deck.get("segments", [])):
        if not seg.get("from") or not seg.get("to"):
            continue
        fx, fy = seg["from"]
        tx, ty = seg["to"]
        cx = round((fx + tx) / 2, 3)
        cy = round((fy + ty) / 2, 3)
        dx = abs(tx - fx)
        dy = abs(ty - fy)
        axis = "ew" if dx >= dy else "ns"
        raw_length = max(dx, dy)
        length = round(raw_length + DECK_ROAD_EXTEND_CELLS, 3)
        width = seg.get("width", 6)
        road = {
            "axis": axis,
            "road_class": "local",
            "width_cells": width,
            "length_cells": length if length > 0 else 6,
            "center_x": cx,
            "center_y": cy,
        }
        if len(deck.get("segments", [])) == 1:
            road["name"] = name
        else:
            road["name"] = f"{name} (seg{i+1})"
        roads.append(road)
    return roads


def main() -> int:
    if not ORIG.exists():
        print(f"[err] original not found: {ORIG}", file=sys.stderr)
        return 1
    if not EXPORT.exists():
        print(f"[err] export not found: {EXPORT}", file=sys.stderr)
        return 1

    orig = load_with_shorthand(ORIG)
    export = load_with_shorthand(EXPORT)

    export_scene = export.get("scene_3d", {}) or {}
    orig_scene = orig.get("scene_3d", {}) or {}

    # Start roads from export, then append any deck-derived roads, then inject
    # the auto EW connectors that were missing from the viewer edit.
    roads = list(export_scene.get("roads", []) or [])
    decks = export_scene.get("decks", []) or []
    for deck in decks:
        roads.extend(deck_to_roads(deck))
    roads.extend(INJECTED_EW_ROADS)

    # Drop places that user flagged as accidental leftovers from viewer editing.
    export_places = export.get("places", []) or []
    places_kept = [p for p in export_places if (p.get("name") or "") not in PLACES_TO_DROP]
    dropped = [p.get("name") for p in export_places if (p.get("name") or "") in PLACES_TO_DROP]

    # Apply position overrides for places that were outside apron-jump reach.
    moves_applied = []
    for p in places_kept:
        name = p.get("name") or ""
        if name in PLACE_MOVES:
            before = (p.get("center_x"), p.get("center_y"))
            for k, v in PLACE_MOVES[name].items():
                p[k] = v
            after = (p.get("center_x"), p.get("center_y"))
            moves_applied.append((name, before, after))

    # Append user-requested facilities (改札 x2, 乗り場 x2) not in the viewer export.
    import copy as _copy
    added_names = []
    for add in PLACES_TO_ADD:
        places_kept.append(_copy.deepcopy(add))
        added_names.append(add.get("name"))

    merged_scene = {}
    if export_scene.get("rails"):
        merged_scene["rails"] = export_scene["rails"]
    if roads:
        merged_scene["roads"] = roads
    if export_scene.get("landmarks"):
        merged_scene["landmarks"] = export_scene["landmarks"]
    # decks / stairs: intentionally dropped

    merged = {
        "metadata": orig.get("metadata", {}),
    }
    if orig.get("proposed_type_extensions"):
        merged["proposed_type_extensions"] = orig["proposed_type_extensions"]

    merged["places"] = places_kept
    if merged_scene:
        merged["scene_3d"] = merged_scene
    if orig.get("scenario"):
        merged["scenario"] = orig["scenario"]

    header = (
        "# ================================================================\n"
        "# 品川駅周辺 400m圏 シーン設定 (merged)\n"
        "# ================================================================\n"
        "# Produced by tools/merge_scene_export.py\n"
        "#\n"
        "# Sources:\n"
        f"#   orig   : {ORIG.name}\n"
        f"#   export : {EXPORT.name}\n"
        "#\n"
        "# Merge rules:\n"
        "#   - places    : export (user's canonical edits)\n"
        "#   - scene_3d  : export; 東西自由通路 was demoted from deck to 1F EW road\n"
        "#   - decks/stairs: intentionally dropped (simplification)\n"
        "#   - metadata / proposed_type_extensions / scenario: from orig\n"
        "#   - personas  : NOT in this file; pull from config_jr_disruption.yaml at sim time\n"
        "# ================================================================\n\n"
    )

    OUT.write_text(
        header + yaml.safe_dump(
            merged,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=4096,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {OUT}")
    print(f"     places : {len(merged.get('places', []))}")
    s = merged.get("scene_3d", {})
    print(f"     rails  : {len(s.get('rails', []))}")
    print(f"     roads  : {len(s.get('roads', []))}")
    print(f"     decks  : {len(s.get('decks', []))} (dropped)")
    print(f"     stairs : {len(s.get('stairs', []))} (dropped)")
    print(f"     lmks   : {len(s.get('landmarks', []))}")
    # Summary of what was demoted:
    promoted = [r for r in s.get("roads", []) if "自由通路" in (r.get("name") or "")]
    for r in promoted:
        print(f"     promoted road: {r.get('name')} axis={r.get('axis')} "
              f"center=({r.get('center_x')},{r.get('center_y')}) "
              f"w={r.get('width_cells')} l={r.get('length_cells')}")
    injected = [r for r in s.get("roads", []) if (r.get("name") or "").startswith("(auto)")]
    for r in injected:
        print(f"     injected road: {r.get('name')} axis={r.get('axis')} "
              f"center=({r.get('center_x')},{r.get('center_y')}) "
              f"w={r.get('width_cells')} l={r.get('length_cells')}")
    if dropped:
        print(f"     dropped places : {dropped}")
    for name, before, after in moves_applied:
        print(f"     moved place    : {name}  {before} -> {after}")
    if added_names:
        print(f"     added places   : {added_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
