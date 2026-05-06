"""smoke22: docs/shinagawa_field_places.yaml の各 place の perceive_pass 末尾に
「[利用層メモ] 男女比率 / 外国語対応 / バリアフリー」 を追記する。

ジェンダーレスの子・外国籍の子・車いすの子が street/施設レベルで「ここは誰のための場所か」
を観察できるようにするための設定。

ユーザ確定の分類表に基づく (2026-05-05)。
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLACES_YAML = ROOT / "docs" / "shinagawa_field_places.yaml"

# ---- 男女比率 ----
GENDER_MALE = {
    "アレア品川", "品川インターシティ", "品川グランドコモンズ", "品川シーズンテラス",
    "品川セントラルガーデン", "NTT", "コクヨ", "ソニー", "国際会議場",
    "食肉市場", "浄水場", "パチンコ屋",
    "工事中（A地区）", "工事中（D地区）", "工事中（駅ビル）", "工事中（北口広場）",
}
GENDER_MALE_SLIGHT = {
    "新幹線改札", "港南口広場", "税務署",
}
GENDER_FEMALE = {
    "アトレ品川",
}
# それ以外は「半々」


def gender_label(name: str) -> str:
    if name in GENDER_MALE:
        return "男性多め"
    if name in GENDER_MALE_SLIGHT:
        return "男性やや多め"
    if name in GENDER_FEMALE:
        return "女性多め"
    return "半々"


# ---- 外国語対応 (ユーザ修正反映: 公園 / 東西自由通路 / 港南口広場 を A に移動) ----
LANG_MULTI = {  # A: 多言語 (英中韓)
    "JR改札", "京急改札", "新幹線改札",
    "東西自由通路", "港南口広場",
    "アトレ品川", "品川水族館", "国際会議場",
    "品川プリンスホテル", "グランドプリンス新高輪", "グランドプリンス高輪", "プリンスさくらタワー",
    "高輪の森公園", "高輪公園", "芝浦中央公園",
}
LANG_EN_ONLY = {  # B: 英語のみ
    "エキナカ", "ウィング高輪", "ave高輪",
    "品川インターシティ", "品川グランドコモンズ", "品川シーズンテラス",
    "アレア品川", "品川セントラルガーデン",
    "NTT", "コクヨ", "ソニー",
}
# それ以外は C: 日本語のみ


def lang_label(name: str) -> str:
    if name in LANG_MULTI:
        return "多言語対応 (英・中・韓)"
    if name in LANG_EN_ONLY:
        return "英語のみ部分対応"
    return "日本語のみ"


# ---- バリアフリー (ユーザ修正反映) ----
BF_FULL = {  # A: 完全バリアフリー
    "JR改札", "京急改札", "新幹線改札",
    "港南口広場", "芝浦中央公園", "品川セントラルガーデン",
    "アトレ品川", "品川水族館",
    "品川プリンスホテル", "グランドプリンス新高輪", "グランドプリンス高輪", "プリンスさくらタワー",
    "国際会議場", "アレア品川", "品川インターシティ", "品川グランドコモンズ",
}
BF_PARTIAL = {  # B: 部分対応 (= 東西自由通路 工事中)
    "東西自由通路",
}
BF_HARD = {  # C: 段差・階段中心
    "食肉市場", "浄水場", "パチンコ屋",
    "工事中（A地区）", "工事中（D地区）", "工事中（駅ビル）", "工事中（北口広場）",
    "高輪の森公園", "高輪公園",
}
# それ以外は デフォルト B 部分対応扱い (オフィス・商業は基本対応してるが完全認定外)


def bf_label(name: str) -> str:
    if name in BF_FULL:
        return "完全バリアフリー (エレベーター・スロープ・幅員すべて整備)"
    if name in BF_PARTIAL:
        return "部分対応 (※工事中で動線が限定的)"
    if name in BF_HARD:
        return "段差・階段中心 (車いすには厳しい)"
    return "部分対応 (一部スロープ・エレベーターあり)"


def annotate(name: str) -> str:
    g = gender_label(name)
    L = lang_label(name)
    b = bf_label(name)
    return f"\n[利用層メモ] 男女比率: {g} / 外国語対応: {L} / バリアフリー: {b}"


def main():
    text = PLACES_YAML.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, list):
        # 場合により dict 構造のことも → places キー
        if isinstance(data, dict) and "places" in data:
            places = data["places"]
        else:
            print(f"[err] yaml structure unexpected: {type(data)}", file=sys.stderr)
            return 1
    else:
        places = data

    # 既存の[利用層メモ]を全削除してから追記 (= idempotent)
    pat = re.compile(r"\n*\[利用層メモ\][^\n]*", re.M)

    n_named = 0
    n_internal = 0
    for p in places:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "")
        if not name:
            continue
        # アンダースコア始まりの内部 place (例 _オフィス_南東_01) はスキップ
        if name.startswith("_"):
            n_internal += 1
            continue
        n_named += 1
        suffix = annotate(name)
        for key in ("perceive_pass", "perceive_enter"):
            cur = p.get(key) or ""
            if not cur:
                continue
            cur = pat.sub("", cur).rstrip()
            p[key] = cur + suffix

    PLACES_YAML.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[ok] annotated {n_named} named places (skipped {n_internal} internal places).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
