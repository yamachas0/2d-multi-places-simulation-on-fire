"""Debug one specific BFS to see what endpoint actually is."""
from __future__ import annotations
import re
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navigation import Navigator

CONFIG = ROOT / "shinagawa_nav_test.yaml"

def load(path):
    raw = path.read_text(encoding="utf-8")
    sh = re.compile(r"^(\s*-\s)(.+?\s;\s.+)$", re.MULTILINE)
    def norm(m):
        prefix, body = m.group(1), m.group(2).strip()
        fields = [f.strip() for f in body.split(";") if f.strip()]
        return f"{prefix}{{ {', '.join(fields)} }}"
    return yaml.safe_load(sh.sub(norm, raw))

cfg = load(CONFIG)
nav = Navigator(cfg, cfg["simulation"]["half_space_size"])

targets = ["新高輪プリンスホテル", "プリンスさくらタワー"]
for p in cfg["places"]:
    if p.get("name") in targets:
        gx = int(round(p["center_x"]))
        gy = int(round(p["center_y"]))
        hx = float(p.get("half_size_x", 2))
        hy = float(p.get("half_size_y", 2))
        print(f"\n=== {p['name']} ===")
        print(f"  center=({gx},{gy}) hs=({hx:.2f},{hy:.2f})")
        # Check specific cells that should be walkable
        for (cx, cy), label in [
            ((-36, 20), "EW road @ (-36,20)"),
            ((-25, 11), "国道15号 @ (-25,11)"),
            ((-25, 30), "国道15号 @ (-25,30)"),
            ((gx, gy), "place center"),
            ((gx+int(hx)+1, gy), "just east of place"),
            ((gx, gy-int(hy)-1), "just south of place"),
        ]:
            mark = "W" if nav.is_walkable(cx, cy) else "-"
            print(f"  [{mark}] {label}  ({cx},{cy})")
        # BFS from a safe main-comp start
        start = (-52, -74)
        path = nav.find_path(start, (gx, gy), max_nodes=60000)
        ex, ey = path[-1]
        dx = abs(ex - gx) - hx
        dy = abs(ey - gy) - hy
        apron = max(0.0, dx) + max(0.0, dy)
        print(f"  BFS start={start} goal=({gx},{gy}) endpoint=({ex},{ey})")
        print(f"  apron_dx={dx:.2f} apron_dy={dy:.2f} apron={apron:.2f}  "
              f"(<=6: {'YES' if apron<=6 else 'NO'})")
        # Show last 5 steps of path
        if len(path) > 5:
            print(f"  last 5 path: {path[-5:]}")
        else:
            print(f"  full path: {path}")
