"""Build a runnable shinagawa test config by combining:
  - scene/places from shinagawa_config.merged.yaml  (viewer-edited field)
  - simulation/agents/personas/llm/visualization from config_jr_disruption.yaml

Output: shinagawa_nav_test.yaml (at project root).

This is intended for a quick navigation smoke test with NO events/fires,
so agents just wander/chat under the passability constraints added by
navigation.py. Run with a small num_agents + short duration first.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MERGED = Path(r"D:\ユーザー\ダウンロード\shinagawa_config.merged.yaml")
SIMBASE = ROOT / "config_jr_disruption.yaml"
OUT = ROOT / "shinagawa_nav_test.yaml"


def load_yaml_with_shorthand(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    sh = re.compile(r"^(\s*-\s)(.+?\s;\s.+)$", re.MULTILINE)
    def norm(m):
        prefix, body = m.group(1), m.group(2).strip()
        fields = [f.strip() for f in body.split(";") if f.strip()]
        return f"{prefix}{{ {', '.join(fields)} }}"
    return yaml.safe_load(sh.sub(norm, raw))


def main() -> int:
    if not MERGED.exists():
        print(f"[err] missing: {MERGED}", file=sys.stderr)
        return 1
    if not SIMBASE.exists():
        print(f"[err] missing: {SIMBASE}", file=sys.stderr)
        return 1

    scene = load_yaml_with_shorthand(MERGED)
    base = load_yaml_with_shorthand(SIMBASE)

    # ----- simulation -----
    sim = copy.deepcopy(base.get("simulation", {}))
    # Match shinagawa scale (400m radius = 80 cells at 5m/cell).
    sim["half_space_size"] = scene.get("metadata", {}).get("half_space_size", 80)

    # ----- agents: keep structure, shrink for smoke test -----
    agents = copy.deepcopy(base.get("agents", {}))
    # Full personas retained so social_identities still work. Count capped via
    # num_agents; persona pool beyond num_agents is simply unused.
    agents["num_agents"] = 20
    agents["max_agents"] = 25

    out = {
        "metadata": scene.get("metadata", {}),
        "proposed_type_extensions": scene.get("proposed_type_extensions", []),
        "simulation": sim,
        "agents": agents,
        "places": scene.get("places", []),
        "scene_3d": scene.get("scene_3d", {}),
        "llm": copy.deepcopy(base.get("llm", {})),
        # No hazards for the smoke test; isolate navigation from event effects.
        "fires": [],
        "events": [],
        "visualization": copy.deepcopy(base.get("visualization", {})),
    }

    header = (
        "# ================================================================\n"
        "# shinagawa navigation smoke test config\n"
        "# ================================================================\n"
        "# Built by tools/build_shinagawa_run_config.py\n"
        "#\n"
        "# Purpose:\n"
        "#   Validate that the Navigator (road mask + BFS + apron-jump) lets\n"
        "#   agents move across the simplified shinagawa field without\n"
        "#   getting stuck. NO fires / NO events so behavior is isolated.\n"
        "#\n"
        "# Sources:\n"
        f"#   scene : {MERGED.name}  (places + scene_3d)\n"
        f"#   sim   : {SIMBASE.name} (simulation + agents.personas + llm)\n"
        "#\n"
        "# Overrides:\n"
        "#   - half_space_size forced to 80 (shinagawa scale)\n"
        "#   - num_agents capped to 20 / max_agents 25 (smoke test size)\n"
        "#   - fires / events set to [] for clean nav observation\n"
        "# ================================================================\n\n"
    )

    OUT.write_text(
        header + yaml.safe_dump(
            out,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=4096,
        ),
        encoding="utf-8",
    )

    s = out.get("scene_3d", {}) or {}
    print(f"[ok] wrote: {OUT}")
    print(f"     places           : {len(out['places'])}")
    print(f"     scene_3d roads   : {len(s.get('roads', []))}")
    print(f"     scene_3d rails   : {len(s.get('rails', []))}")
    print(f"     num_agents       : {out['agents'].get('num_agents')}")
    print(f"     half_space_size  : {out['simulation'].get('half_space_size')}")
    print(f"     personas in pool : {len(out['agents'].get('personas', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
