"""Bundle viewer HTML + simulation_data.json (and Three.js when needed)
into a single self-contained HTML that runs from file:// without a local
web server.

Usage:
  python tools/bundle_viewer.py <run_dir>                 # 2D (default)
  python tools/bundle_viewer.py <run_dir> --viewer 3d     # 3D
  python tools/bundle_viewer.py <run_dir> --viewer 3d \\
      --scene-3d config_shinagawa.yaml                    # + scene_3d

Outputs:
  <run_dir>/viewer_v2_bundled.html   (2D)
  <run_dir>/viewer_3d_bundled.html   (3D)

For 3D, Three.js r128 (three.min.js + OrbitControls.js) is fetched once
from unpkg.com and cached under visualization/vendor/three-0.128/.
The cached files are inlined into the output HTML so the bundle works
fully offline.

With --scene-3d, the `scene_3d:` subtree of the given YAML (shinagawa
config style) is parsed and injected into the viewer's scene-3d script
tag for rendering of rails / roads / decks / stairs / landmarks.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / 'visualization'
VENDOR_DIR = VIEWER_DIR / 'vendor' / 'three-0.128'

TEMPLATE_2D = VIEWER_DIR / 'viewer_v2.html'
TEMPLATE_3D = VIEWER_DIR / 'viewer_3d.html'

DATA_MARKER = '<script id="sim-data" type="application/json"></script>'
SCENE3D_MARKER = '<script id="scene-3d" type="application/json"></script>'
THREE_MARKER = '<script id="three-lib" src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>'
ORBIT_MARKER = '<script id="orbit-lib" src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>'

THREE_URL = 'https://unpkg.com/three@0.128.0/build/three.min.js'
ORBIT_URL = 'https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js'


def _escape_script(body: str) -> str:
    """Prevent the inlined body from terminating the host <script> tag."""
    return body.replace('</script>', '<\\/script>')


def _fetch_vendor(url: str, dest: Path) -> str:
    """Return vendor JS contents, downloading+caching on first use."""
    if dest.exists():
        return dest.read_text(encoding='utf-8')
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'singulabo-bundler/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode('utf-8')
    dest.write_text(body, encoding='utf-8')
    return body


def _inject_data(html: str, data_raw: str) -> str:
    if DATA_MARKER not in html:
        raise RuntimeError(f"Data inject marker missing: {DATA_MARKER}")
    # Guard against string payloads containing "</script>" which would
    # terminate the host script tag.
    data_safe = data_raw.replace('</', '<\\/')
    inline = (
        f'<script id="sim-data" type="application/json">'
        f'{data_safe}</script>'
    )
    return html.replace(DATA_MARKER, inline)


# shinagawa_config.yaml uses a non-standard inline shorthand for road
# entries: `- axis: "ns" ; center_x: -28 ; width_cells: 1.5 ; ...`
# PyYAML rejects the semicolon separators, so normalize these lines to
# flow mapping syntax before parsing.
_ROAD_SHORTHAND = re.compile(r'^(\s*-\s)(.+?\s;\s.+)$', re.MULTILINE)


def _normalize_semicolon_shorthand(text: str) -> str:
    def repl(m):
        prefix, body = m.group(1), m.group(2).strip()
        fields = [f.strip() for f in body.split(';') if f.strip()]
        return f"{prefix}{{ {', '.join(fields)} }}"
    return _ROAD_SHORTHAND.sub(repl, text)


def _extract_scene_3d(yaml_path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "PyYAML required for --scene-3d. Install with: pip install pyyaml"
        ) from e
    raw = yaml_path.read_text(encoding='utf-8')
    normalized = _normalize_semicolon_shorthand(raw)
    doc = yaml.safe_load(normalized)
    if not isinstance(doc, dict):
        raise RuntimeError(f"{yaml_path}: expected a top-level mapping")
    if 'scene_3d' not in doc:
        raise RuntimeError(f"{yaml_path}: no `scene_3d:` section found")
    return doc['scene_3d']


def _inject_scene3d(html: str, s3: dict) -> str:
    if SCENE3D_MARKER not in html:
        # Older template without the marker — just no-op.
        print("  (template lacks scene-3d marker, skipping injection)")
        return html
    body = json.dumps(s3, ensure_ascii=False).replace('</', '<\\/')
    inline = f'<script id="scene-3d" type="application/json">{body}</script>'
    return html.replace(SCENE3D_MARKER, inline)


def _inject_three(html: str) -> str:
    if THREE_MARKER not in html:
        raise RuntimeError(f"Three.js inject marker missing: {THREE_MARKER}")
    if ORBIT_MARKER not in html:
        raise RuntimeError(f"OrbitControls inject marker missing: {ORBIT_MARKER}")

    three_js = _fetch_vendor(THREE_URL, VENDOR_DIR / 'three.min.js')
    orbit_js = _fetch_vendor(ORBIT_URL, VENDOR_DIR / 'OrbitControls.js')

    three_inline = f'<script id="three-lib">{_escape_script(three_js)}</script>'
    orbit_inline = f'<script id="orbit-lib">{_escape_script(orbit_js)}</script>'
    html = html.replace(THREE_MARKER, three_inline)
    html = html.replace(ORBIT_MARKER, orbit_inline)
    return html


def bundle_viewer(
    run_dir: str,
    viewer: str = '2d',
    template_path: str = None,
    scene_3d_yaml: str = None,
) -> str:
    run = Path(run_dir)
    data_path = run / 'simulation_data.json'
    if not data_path.exists():
        raise FileNotFoundError(f"Not found: {data_path}")

    if template_path:
        tpl = Path(template_path)
        out_name = f"viewer_{viewer}_bundled.html"
    elif viewer == '3d':
        tpl = TEMPLATE_3D
        out_name = 'viewer_3d_bundled.html'
    else:
        tpl = TEMPLATE_2D
        out_name = 'viewer_v2_bundled.html'
    if not tpl.exists():
        raise FileNotFoundError(f"Viewer template not found: {tpl}")

    html = tpl.read_text(encoding='utf-8')
    if viewer == '3d':
        html = _inject_three(html)
        if scene_3d_yaml:
            s3 = _extract_scene_3d(Path(scene_3d_yaml))
            html = _inject_scene3d(html, s3)
            counts = {k: (len(v) if isinstance(v, list) else '?') for k, v in s3.items()}
            print(f"  scene_3d injected: {counts}")
    elif scene_3d_yaml:
        print("  --scene-3d ignored for 2D viewer")
    html = _inject_data(html, data_path.read_text(encoding='utf-8'))

    out = run / out_name
    out.write_text(html, encoding='utf-8')
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='Run directory containing simulation_data.json')
    ap.add_argument('--viewer', choices=['2d', '3d'], default='2d',
                    help='Viewer flavor (default: 2d)')
    ap.add_argument('--scene-3d', dest='scene_3d', default=None,
                    help='Path to a YAML file containing a scene_3d: section '
                         '(3D viewer only)')
    args = ap.parse_args()
    out = bundle_viewer(
        args.run_dir,
        viewer=args.viewer,
        scene_3d_yaml=args.scene_3d,
    )
    size_kb = Path(out).stat().st_size / 1024
    print(f"Wrote: {out} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
