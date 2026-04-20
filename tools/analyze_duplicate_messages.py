"""Aggregate duplicate message bursts in messages.jsonl.

A "duplicate burst" is (step, from_id) where the same message body goes to
>=2 distinct to_ids. Prints the counts and the first few examples.

Usage:
  python tools/analyze_duplicate_messages.py <run_dir>
"""
import argparse
import collections
import json
from pathlib import Path


def analyze(path: Path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    groups = collections.defaultdict(list)
    for r in rows:
        key = (r['step'], r['from'], r['message'])
        groups[key].append(r)

    bursts = {k: v for k, v in groups.items() if len(v) >= 2}

    total_msg = len(rows)
    total_burst_rows = sum(len(v) for v in bursts.values())
    unique_speak_events = len(groups)
    burst_speak_events = len(bursts)

    print(f"Total message rows: {total_msg}")
    print(f"Unique (step,from,body) groups: {unique_speak_events}")
    print(f"Burst groups (>=2 recipients, same body): {burst_speak_events}")
    print(f"Message rows inside bursts: {total_burst_rows}")
    print(f"Duplicated rows (=rows - groups): {total_burst_rows - burst_speak_events}")
    if total_msg:
        pct = 100 * (total_burst_rows - burst_speak_events) / total_msg
        print(f"Duplicate-row ratio: {pct:.1f}%")

    size_hist = collections.Counter(len(v) for v in bursts.values())
    print(f"\nBurst size histogram (#recipients -> #bursts):")
    for size in sorted(size_hist):
        print(f"  {size}: {size_hist[size]}")

    print("\nFirst 5 bursts:")
    for (step, frm, body), v in list(bursts.items())[:5]:
        names = [x['to_name'] for x in v]
        print(f"  step={step} from={v[0]['from_name']} (id={frm})")
        print(f"    recipients ({len(v)}): {names}")
        print(f"    body: {body[:70]}...")

    # flag cases where body mentions a recipient name but goes to others
    name_mismatch = 0
    mismatch_examples = []
    for (step, frm, body), v in bursts.items():
        for x in v:
            for y in v:
                if y['to_name'] and y['to_name'] in body and y['to'] != x['to']:
                    name_mismatch += 1
                    if len(mismatch_examples) < 3:
                        mismatch_examples.append(
                            (step, v[0]['from_name'], y['to_name'], x['to_name'], body[:60])
                        )
                    break
    print(f"\nName-mismatch rows (body mentions one recipient's name, delivered to others too): {name_mismatch}")
    for ex in mismatch_examples:
        print(f"  step={ex[0]} from={ex[1]} body says '{ex[2]}' but delivered to {ex[3]}: {ex[4]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    args = ap.parse_args()
    analyze(Path(args.run_dir) / 'messages.jsonl')


if __name__ == '__main__':
    main()
