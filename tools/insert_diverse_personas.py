"""smoke22: 既存の各 variant persona yaml に対し、最後の 3 人 (17/18/19 番目) を
車いす利用 / 欧米系外国籍 / アジア系外国籍 に上書きする。

agent.py の _build_nationality_block / _build_mobility_block が
persona.nationality / persona.mobility を読んで system_prompt 固定挿入する。

上書き対象 (各 variant):
  - 17番目 (index 16): mobility=wheelchair (車いす利用、日本人)
  - 18番目 (index 17): nationality=western (欧米系)
  - 19番目 (index 18): nationality=asian (アジア系)

20番目 (gender=other) は既存維持。
"""
import sys, copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


# variant ごとの上書きデータ
DIVERSITY_DATA = {
    "high": {
        16: {
            "name": "辻 真琴",
            "gender": "female",
            "mobility": "wheelchair",
            "family_struct": "両親と弟の4人暮らし。先天的な事情で車いす利用、学校では一部介助あり。",
            "catchphrase": "「ここ、段差ある？」",
        },
        17: {
            "name": "アンナ・ベイリー",
            "gender": "female",
            "nationality": "western",
            "family_struct": "父親はイギリス出身、母親は日本人。家では英語と日本語が混じる。日本生まれ・育ち。",
            "catchphrase": "「えーと、その漢字どう読むの？」",
        },
        18: {
            "name": "リー・ハナ",
            "gender": "female",
            "nationality": "asian",
            "family_struct": "両親とも韓国出身、自分は日本生まれ。日本語は読み書きできるが微妙な敬語は苦手。",
            "catchphrase": "「ちょっと、その言い方わからない…」",
        },
    },
    "elementary": {
        16: {
            "name": "辻 さくら",
            "gender": "female",
            "mobility": "wheelchair",
            "family_struct": "お父さん・お母さん・お兄ちゃんの4人。生まれつき車いすで、学校では先生がときどき手伝ってくれる。",
            "catchphrase": "「ここ、段差ある？」",
        },
        17: {
            "name": "リアム・パーカー",
            "gender": "male",
            "nationality": "western",
            "family_struct": "お父さんはアメリカ人、お母さんは日本人。家では英語が多い。日本生まれ。",
            "catchphrase": "「えーと、それなんて読むの？」",
        },
        18: {
            "name": "チェン ミミ",
            "gender": "female",
            "nationality": "asian",
            "family_struct": "両親とも中国出身。日本生まれ・育ち。日本語は話せるけど、漢字は中国の漢字と違うのでときどきまちがえる。",
            "catchphrase": "「ちょっと、それわからない…」",
        },
    },
    "junior_high": {
        16: {
            "name": "辻 真琴",
            "gender": "female",
            "mobility": "wheelchair",
            "family_struct": "両親と妹の4人暮らし。先天的な事情で車いす利用、学校では一部介助あり。",
            "catchphrase": "「ここ、段差ある？」",
        },
        17: {
            "name": "ジェームズ・モリス",
            "gender": "male",
            "nationality": "western",
            "family_struct": "父親はカナダ出身、母親は日本人。家では英語と日本語が混じる。日本生まれ・育ち。",
            "catchphrase": "「あ、その漢字どう読むの？」",
        },
        18: {
            "name": "キム・ジウ",
            "gender": "female",
            "nationality": "asian",
            "family_struct": "両親とも韓国出身、自分は日本生まれ。日本語は読み書きできるが微妙な言い回しは苦手。",
            "catchphrase": "「ちょっと、その言い方わからない…」",
        },
    },
}


def patch_yaml(variant: str, yaml_path: Path) -> None:
    if not yaml_path.exists():
        print(f"[skip] {yaml_path} not found")
        return
    text = yaml_path.read_text(encoding="utf-8")
    personas = yaml.safe_load(text)
    if not isinstance(personas, list):
        print(f"[err] {yaml_path}: expected list, got {type(personas)}")
        return
    patches = DIVERSITY_DATA.get(variant, {})
    for idx, patch in patches.items():
        if idx >= len(personas):
            print(f"[warn] {variant} idx={idx} out of range (n={len(personas)})")
            continue
        p = personas[idx]
        old_name = p.get("name", "?")
        # 上書きフィールド適用
        for k, v in patch.items():
            p[k] = v
        # background があれば family_struct で上書きされる旨を反映 (= background は build script 側で再生成)
        new_label = ""
        if patch.get("nationality") == "western":
            new_label = " [外国籍: 欧米系]"
        elif patch.get("nationality") == "asian":
            new_label = " [外国籍: アジア系]"
        elif patch.get("mobility") == "wheelchair":
            new_label = " [車いす利用]"
        print(f"  [{variant}] idx={idx}  {old_name}  →  {patch.get('name')}{new_label}")
    yaml_path.write_text(
        yaml.dump(personas, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[ok] wrote {yaml_path}")


def main():
    print("Diversity persona overwrite (smoke22):")
    patch_yaml("high", ROOT / "classroom_personas.yaml")
    patch_yaml("elementary", ROOT / "classroom_personas_elementary.yaml")
    patch_yaml("junior_high", ROOT / "classroom_personas_junior_high.yaml")
    print("Done.")


if __name__ == "__main__":
    main()
