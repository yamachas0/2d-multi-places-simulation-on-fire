"""Retrofit existing output directories into the `simulations/` structure.

NOTE: one-shot migration — the directories it used to move are gone.
Kept in-tree as a historical record of how the `simulations/` layout
was seeded.

New layout per run:
  outputs/{YYYY-MM-DD_HHMM}_{NN}_{name}/
    {basename}.html                  (ex-report.html)
    {basename}.md                    (copy of validation report, if any)
    {basename}_transcript.md         (ex-transcript.md)
    animation.gif
    *.jsonl                          (raw sinks — left at root)
    frames/
      frame_NNNN.png
      statistics.png

Only retrofits runs that contain the full deliverable trio
(report.html + transcript.md + animation.gif). Fragmentary or
partial runs are left untouched.
"""
import os
import shutil
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "simulations")

# (source_dir, new_name, validation_report_src_path-or-None)
MAPPING = [
    ("output_feat6_test",     "feat6_test",            None),
    ("output_stage_a_precheck","stage_a_precheck",     None),
    ("output_compare_claude",  "compare_claude",       None),
    ("output_compare_openai",  "compare_openai",       None),
    ("output_compare_gemini",  "compare_gemini",       None),
    ("output",                 "default",              None),
    ("_urban_v2_out",          "phase1_urban_v2",      "V2_PHASE1_URBAN_STRUCTURE.md"),
    ("_jr_disruption_out",     "phase2_jr_disruption", "V2_PHASE2_JR_DISRUPTION.md"),
]


def format_dt(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d_%H%M")


def retrofit_one(src_dir: str, new_name: str, report_md_src: str | None, seq: int) -> str:
    src_path = os.path.join(ROOT, src_dir)
    if not os.path.isdir(src_path):
        print(f"  skip (not a dir): {src_dir}")
        return ""
    # Must have the trio to qualify
    for req in ("report.html", "transcript.md", "animation.gif"):
        if not os.path.isfile(os.path.join(src_path, req)):
            print(f"  skip (missing {req}): {src_dir}")
            return ""

    mtime = os.path.getmtime(src_path)
    dt_str = format_dt(mtime)
    folder = f"{dt_str}_{seq:02d}_{new_name}"
    dest = os.path.join(OUTPUTS, folder)
    if os.path.exists(dest):
        print(f"  skip (dest exists): {dest}")
        return ""
    os.makedirs(dest)

    # Move everything from source to dest
    for name in os.listdir(src_path):
        shutil.move(os.path.join(src_path, name), os.path.join(dest, name))

    # Move frames + statistics.png into frames/ subfolder
    frames_dir = os.path.join(dest, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for name in list(os.listdir(dest)):
        if name.startswith("frame_") and name.endswith(".png"):
            shutil.move(os.path.join(dest, name), os.path.join(frames_dir, name))
        elif name == "statistics.png":
            shutil.move(os.path.join(dest, name), os.path.join(frames_dir, name))

    # Rename report.html, transcript.md to match folder basename
    basename = folder
    renames = [
        ("report.html", f"{basename}.html"),
        ("transcript.md", f"{basename}_transcript.md"),
        ("transcript_condensed.md", f"{basename}_transcript_condensed.md"),
    ]
    for old, new in renames:
        src = os.path.join(dest, old)
        if os.path.isfile(src):
            os.rename(src, os.path.join(dest, new))

    # Copy the validation report (if any)
    if report_md_src:
        full_src = os.path.join(ROOT, report_md_src)
        if os.path.isfile(full_src):
            shutil.copy2(full_src, os.path.join(dest, f"{basename}.md"))

    # Remove the now-empty source directory
    try:
        os.rmdir(src_path)
    except OSError:
        print(f"  warning: source dir not empty: {src_path}")

    return folder


def main():
    os.makedirs(OUTPUTS, exist_ok=True)
    # Sort mapping by mtime so sequence numbers are stable
    mapping_with_mtime = []
    for src_dir, new_name, md in MAPPING:
        p = os.path.join(ROOT, src_dir)
        if os.path.isdir(p):
            mapping_with_mtime.append((os.path.getmtime(p), src_dir, new_name, md))
    mapping_with_mtime.sort()

    for seq, (_mtime, src, name, md) in enumerate(mapping_with_mtime, start=1):
        print(f"retrofit [{seq:02d}] {src} -> {name}")
        folder = retrofit_one(src, name, md, seq)
        if folder:
            print(f"  -> outputs/{folder}")


if __name__ == "__main__":
    main()
