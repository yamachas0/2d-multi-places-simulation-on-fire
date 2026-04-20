"""Bundle viewer_v2.html + simulation_data.json into a single
self-contained HTML (viewer_v2_bundled.html) that runs from file://
without a local web server.

Usage:
  python tools/bundle_viewer.py <run_dir>
    expects <run_dir>/simulation_data.json
    writes   <run_dir>/viewer_v2_bundled.html

Or imported: bundle_viewer(run_dir).
"""
import argparse
import json
import os
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / 'visualization' / 'viewer_v2.html'
INJECT_MARKER = '<script id="sim-data" type="application/json"></script>'


def bundle_viewer(run_dir: str, template_path: str = None) -> str:
    run = Path(run_dir)
    data_path = run / 'simulation_data.json'
    if not data_path.exists():
        raise FileNotFoundError(f"Not found: {data_path}")
    tpl = Path(template_path) if template_path else TEMPLATE
    if not tpl.exists():
        raise FileNotFoundError(f"Viewer template not found: {tpl}")

    html = tpl.read_text(encoding='utf-8')
    if INJECT_MARKER not in html:
        raise RuntimeError(
            f"Inject marker missing in {tpl}. Expected literal: {INJECT_MARKER}"
        )

    data_raw = data_path.read_text(encoding='utf-8')
    # Guard against string payloads containing "</script>" which would
    # terminate the host script tag. JSON allows "<\/script>" via the
    # forward-slash escape.
    data_safe = data_raw.replace('</', '<\\/')
    inline = (
        f'<script id="sim-data" type="application/json">'
        f'{data_safe}</script>'
    )
    bundled = html.replace(INJECT_MARKER, inline)

    out = run / 'viewer_v2_bundled.html'
    out.write_text(bundled, encoding='utf-8')
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='Run directory containing simulation_data.json')
    args = ap.parse_args()
    out = bundle_viewer(args.run_dir)
    print(f"Wrote: {out}")


if __name__ == '__main__':
    main()
