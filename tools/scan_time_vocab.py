"""Scan messages.jsonl for morning/evening vocabulary to audit time consistency.

Usage:
  python tools/scan_time_vocab.py <run_dir>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

MORNING = ['おはよう', '朝食', '出社', '今日も頑張', '始ま', 'スタート',
           '朝のコーヒー', '朝の', '今朝', '目覚め', '朝ごはん']
EVENING = ['お疲れ', 'お先に', 'こんばんは', '帰り', '帰宅', '夕食',
           '明日', '夜', '退勤', '夕方']


def scan(path: Path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    m_cnt = Counter()
    e_cnt = Counter()
    m_hits = []
    e_hits = []
    for r in rows:
        msg = r.get('message', '') or ''
        for w in MORNING:
            if w in msg:
                m_cnt[w] += 1
                if len(m_hits) < 12:
                    m_hits.append((r['step'], r.get('time', ''),
                                   r.get('from_name', ''), w, msg[:60]))
                break
        for w in EVENING:
            if w in msg:
                e_cnt[w] += 1
                if len(e_hits) < 6:
                    e_hits.append((r['step'], r.get('time', ''),
                                   r.get('from_name', ''), w, msg[:60]))
                break

    print(f"Total message rows: {len(rows)}")
    print(f"\nMORNING hits: {sum(m_cnt.values())} ({len([r for r in rows if any(w in (r.get('message') or '') for w in MORNING)])} rows)")
    for w, c in m_cnt.most_common():
        print(f"  {w}: {c}")
    print(f"\nMorning examples (first {len(m_hits)}):")
    for step, t, name, w, msg in m_hits:
        print(f"  step={step} {t} {name} [{w}]: {msg}...")

    print(f"\nEVENING hits: {sum(e_cnt.values())}")
    for w, c in e_cnt.most_common():
        print(f"  {w}: {c}")
    print(f"\nEvening examples (first {len(e_hits)}):")
    for step, t, name, w, msg in e_hits:
        print(f"  step={step} {t} {name} [{w}]: {msg}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    args = ap.parse_args()
    scan(Path(args.run_dir) / 'messages.jsonl')


if __name__ == '__main__':
    main()
