"""smoke22: 各 variant persona yaml の catchphrase 被りを潰す。
LLM 生成では「あー、それ、なんかこう、ピンとこない」「フォトナで出てくる」等の表現が
複数 persona に同質化される。手動で固有のフレーズに置き換える。
"""
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


# 高校生 (classroom_personas.yaml)
# 元の被り: "ピンとこない" 3人 (渡辺・小林・加藤), "どうでもいいかも" 2人 (鈴木・伊藤),
#          "想像しちゃう" 2人 (山田・高橋)
HIGH_PATCHES = {
    "渡辺": "「あー、それは典型的なやつだね」",
    "小林": "「は？それ意味ある？」",
    "加藤": "「天才すぎてしんどい」",
    "鈴木": "「ま、好きにすれば」",
    "伊藤": "「ふぁ〜、めんど」",
    "山田": "「ねぇ、あの曲聴いた？」",
    "高橋": "「……描いておこ」",
}

# 小学生 (classroom_personas_elementary.yaml)
# 元の被り: "フォトナで出てくる" 4人 (倉本・三沢・沼田・篠田)
ELEM_PATCHES = {
    "倉本": "「マイクラで作った城のほうがすごいよ！」",
    "三沢": "「これ、スプラのギアと一緒じゃん」",
    "沼田": "「これマリオカートで言うと…」",
    "篠田": "「クラフトしてみたい」",
}

# 中学生 (classroom_personas_junior_high.yaml) — まだ顕著な被りないが念のためチェック
JR_PATCHES: dict = {}


def patch_yaml(path: Path, patches: dict) -> None:
    if not path.exists() or not patches:
        return
    personas = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(personas, list):
        print(f"[skip] {path}: not a list", file=sys.stderr)
        return
    n = 0
    for p in personas:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "")
        # 苗字マッチ (= 名前の頭が key で始まれば該当)
        for key, new_phrase in patches.items():
            if name.startswith(key):
                old = p.get("catchphrase", "")
                p["catchphrase"] = new_phrase
                print(f"  [{path.name}] {name}: {old}  →  {new_phrase}")
                n += 1
                break
    path.write_text(
        yaml.dump(personas, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[ok] {path.name}: patched {n} personas")


def main():
    print("Catchphrase dedupe (smoke22):")
    patch_yaml(ROOT / "classroom_personas.yaml", HIGH_PATCHES)
    patch_yaml(ROOT / "classroom_personas_elementary.yaml", ELEM_PATCHES)
    patch_yaml(ROOT / "classroom_personas_junior_high.yaml", JR_PATCHES)
    print("Done.")


if __name__ == "__main__":
    main()
