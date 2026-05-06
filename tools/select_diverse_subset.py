"""smoke22: 各 variant の 20人 persona yaml から、10人 smoke 用の subset を生成する。
先頭 6人 + 末尾 4人 (= 車いす + 欧米系 + アジア系 + ジェンダーレス) = 10 人。
これにより 10人 smoke でも多様性メンバーが含まれる。
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


SOURCE_TARGETS = [
    ("classroom_personas.yaml",            "classroom_personas_10diverse.yaml"),
    ("classroom_personas_elementary.yaml", "classroom_personas_elementary_10diverse.yaml"),
    ("classroom_personas_junior_high.yaml","classroom_personas_junior_high_10diverse.yaml"),
]


def make_subset(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[skip] {src} not found")
        return
    personas = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(personas, list) or len(personas) < 20:
        print(f"[err] {src.name} has only {len(personas) if isinstance(personas, list) else '?'} personas")
        return
    # 先頭 6 (idx 0-5) + 末尾 4 (idx 16-19)
    subset = personas[:6] + personas[16:20]
    dst.write_text(
        yaml.dump(subset, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    names = [p.get("name", "?") for p in subset]
    print(f"[ok] {dst.name}: 10 personas: {' / '.join(names)}")


def main():
    for src_n, dst_n in SOURCE_TARGETS:
        make_subset(ROOT / src_n, ROOT / dst_n)


if __name__ == "__main__":
    main()
