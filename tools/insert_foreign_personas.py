"""ベースラインv2 用に外国人ペルソナ 11人を YAML に挿入 (既存 axis_id を置換)。

high (4): S11/S13/S15/S20  — 台湾系/日系ブラジル/在日中国/在日コリアン
adult (4): A06/A11/A18/A20 — 観光客メキシコ/IT韓国/建設ベトナム/介護パキスタン
elementary (3): E13/E18/E20 — 中国系/フィリピン系/米日ハーフ

run: ./venv/Scripts/python.exe tools/insert_foreign_personas.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


# 共通フィールド: name, gender, age (or grade), family_struct, ...
# axis_id, axis_career/friends/family は既存のキーを引き継ぐ (後方互換)
# variant, tendency, raw_response, temperament 3軸 を含む

HIGH_FOREIGN = {
    "S11": {
        "name": "陳 美佳",
        "gender": "female",
        "age": "17歳",
        "family_struct": "父は日本人、母は台湾出身、家では中国語と日本語が混ざる、姉と4人家族",
        "club": "写真部",
        "subjects": "得意は英語と理科、苦手は古文",
        "future_interest": "留学にちょっと興味あるけど親はあまり乗り気じゃない",
        "catchphrase": "うーん、なんかね",
        "phone_habit": "台湾のドラマと友達のSNS",
        "axis_id": "S11", "axis_career": "明確", "axis_friends": "周辺", "axis_family": "高",
        "variant": "high", "tendency": "内省的",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "mid", "temperament_optimism": "mid", "temperament_curiosity": "high",
    },
    "S13": {
        "name": "リカルド・タナカ",
        "gender": "male",
        "age": "17歳",
        "family_struct": "両親はブラジル出身の日系3世、家ではポルトガル語、弟と4人家族",
        "club": "サッカー部",
        "subjects": "得意は数学と体育、苦手は社会",
        "future_interest": "兄がブラジルで働いていて、自分もどっちで暮らすか考えてる",
        "catchphrase": "マジ？",
        "phone_habit": "サッカーのインスタとブラジルの兄とのチャット",
        "axis_id": "S13", "axis_career": "迷い", "axis_friends": "コア", "axis_family": "低",
        "variant": "high", "tendency": "ふつう",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "high", "temperament_optimism": "mid", "temperament_curiosity": "mid",
    },
    "S15": {
        "name": "高 リン",
        "gender": "female",
        "age": "17歳",
        "family_struct": "祖父母の代から日本、家では日本語、両親が中華料理屋を経営、妹と5人家族",
        "club": "帰宅部",
        "subjects": "得意は数学、苦手は古文と漢文",
        "future_interest": "将来どこで仕事するかまだ決めてない、海外も日本も両方アリ",
        "catchphrase": "べつに、どっちでも",
        "phone_habit": "Bilibili (中国の動画) と日本のSNSを行ったり来たり",
        "axis_id": "S15", "axis_career": "迷い", "axis_friends": "孤立", "axis_family": "高",
        "variant": "high", "tendency": "内省的",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "low", "temperament_optimism": "mid", "temperament_curiosity": "high",
    },
    "S20": {
        "name": "朴 ジホン",
        "gender": "male",
        "age": "17歳",
        "family_struct": "祖父母の代から日本に住む在日コリアン3世、家では日本語、両親と弟の4人家族",
        "club": "バスケ部",
        "subjects": "得意は英語と数学、苦手は古文",
        "future_interest": "韓国留学か日本就職か、まだ決めてない",
        "catchphrase": "うーん、それはまた今度",
        "phone_habit": "K-POPの音楽動画と韓国のスポーツ番組",
        "axis_id": "S20", "axis_career": "未決", "axis_friends": "孤立", "axis_family": "高",
        "variant": "high", "tendency": "内省的",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "low", "temperament_optimism": "mid", "temperament_curiosity": "mid",
    },
}


ADULT_FOREIGN = {
    "A06": {
        "name": "Maria Lopez",
        "gender": "female",
        "age": "32歳",
        "occupation": "メキシコ・グアダラハラのラジオ番組プロデューサー、観光で日本に来日中、便宜上日本語OK",
        "family_struct": "メキシコに夫と娘、品川のホテルに2泊予定で滞在中",
        "join_reason": "友人の友人に「面白い集まりがある」と誘われ、たまたま観光中に参加",
        "recent_concern": "日本での観光経験を母国の番組ネタにしたい、品川の魅力をどう伝えるか",
        "catchphrase": "Oh, interesting...",
        "axis_id": "A06", "axis_career": "観光業/ラジオプロデューサー", "axis_friends": "新参", "axis_family": "中程度",
        "variant": "adult", "tendency": "主張ある",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "high", "temperament_optimism": "high", "temperament_curiosity": "high",
    },
    "A11": {
        "name": "David Kim",
        "gender": "male",
        "age": "35歳",
        "occupation": "韓国ソウル出身、東京のスタートアップで AI エンジニア、日本在住5年",
        "family_struct": "独身、品川区在住、韓国の家族とは月1回ビデオ通話",
        "join_reason": "日本のテック界隈の仲間と知り合いたくて参加",
        "recent_concern": "日本のスタートアップ環境のスローさと韓国のスピード感のギャップを感じてる",
        "catchphrase": "結局、〜じゃないですか",
        "axis_id": "A11", "axis_career": "AIエンジニア(外国人)", "axis_friends": "周辺", "axis_family": "豊富",
        "variant": "adult", "tendency": "主張ある",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "mid", "temperament_optimism": "mid", "temperament_curiosity": "high",
    },
    "A18": {
        "name": "Phan Van Long",
        "gender": "male",
        "age": "28歳",
        "occupation": "ベトナム・ハイフォン出身、品川の建設会社で現場監督、技能実習5年→特定技能で在住7年",
        "family_struct": "独身、神奈川のシェアハウス住、ベトナムに両親と妹",
        "join_reason": "日本語の勉強会で知り合った人に誘われ、たまたま参加",
        "recent_concern": "永住権を取りたい、お金を貯めて将来は日本かベトナムか決めたい",
        "catchphrase": "そうですね、わたしはまだ勉強中です",
        "axis_id": "A18", "axis_career": "建設業(外国人)", "axis_friends": "新参", "axis_family": "中程度",
        "variant": "adult", "tendency": "空気読む",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "mid", "temperament_optimism": "high", "temperament_curiosity": "mid",
    },
    "A20": {
        "name": "Aisha Tariq",
        "gender": "female",
        "age": "45歳",
        "occupation": "パキスタン・カラチ出身、品川区の介護施設で介護福祉士、来日20年で日本国籍取得済",
        "family_struct": "日本人の夫と中学生の息子、家では日本語と英語が混ざる、宗教はムスリム",
        "join_reason": "地域の集まりがあると聞いて、近所のことを知るために参加",
        "recent_concern": "高齢化していく現場で人手が足りない、外国人介護士をもっと増やしたい",
        "catchphrase": "インシャアッラー、なるようになる",
        "axis_id": "A20", "axis_career": "介護福祉(外国人)", "axis_friends": "周辺", "axis_family": "中程度",
        "variant": "adult", "tendency": "主張ある",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "mid", "temperament_optimism": "mid", "temperament_curiosity": "mid",
    },
}


ELEM_FOREIGN = {
    "E13": {
        "name": "ライアン・ウォン",
        "gender": "male",
        "age": "11歳",
        "grade": "小5",
        "family_struct": "上海から3年前に来日、お父さんとお母さんと妹の4人、家では中国語と日本語、品川の小学校に転入",
        "hobby_like": "マイクラと動画編集、中国のアニメも見る",
        "hobby_dislike": "漢字の書き取り (日本の漢字と中国の漢字が違うので混乱する)",
        "catchphrase": "中国だとこうなんだけど…",
        "axis_id": "E13", "axis_career": "マイペース", "axis_friends": "周辺", "axis_family": "ふつう",
        "variant": "elementary", "tendency": "ふつう",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "mid", "temperament_optimism": "mid", "temperament_curiosity": "high",
    },
    "E18": {
        "name": "アンナ・サントス",
        "gender": "female",
        "age": "11歳",
        "grade": "小5",
        "family_struct": "お父さんは日本人、お母さんはフィリピン人、品川生まれ、弟と4人家族、お弁当にフィリピン料理混じる",
        "hobby_like": "ダンスとTikTok、お母さんが作るフィリピン料理 (アドボ)",
        "hobby_dislike": "算数の文章題、走るのが遅い",
        "catchphrase": "マジで〜？",
        "axis_id": "E18", "axis_career": "苦手意識", "axis_friends": "周辺", "axis_family": "ふつう",
        "variant": "elementary", "tendency": "ふつう",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "high", "temperament_optimism": "mid", "temperament_curiosity": "mid",
    },
    "E20": {
        "name": "エミリー・ヤナギモト",
        "gender": "female",
        "age": "11歳",
        "grade": "小5",
        "family_struct": "お父さんはアメリカ人、お母さんは日本人、品川生まれ、姉と4人家族、家では英語と日本語混在",
        "hobby_like": "英語の本を読むこと、ピアノ、アメリカの祖父母とビデオ通話",
        "hobby_dislike": "漢字の書き取り、算数",
        "catchphrase": "Wait, ちょっと待って",
        "axis_id": "E20", "axis_career": "マイペース", "axis_friends": "ひとり", "axis_family": "やや薄い",
        "variant": "elementary", "tendency": "ふつう",
        "raw_response": "(insert_foreign_personas.py で手書き挿入)",
        "temperament_extroversion": "mid", "temperament_optimism": "high", "temperament_curiosity": "high",
    },
}


def patch(yaml_path: Path, replacements: dict[str, dict]):
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    by_id = {p["axis_id"]: i for i, p in enumerate(data)}
    for axis_id, new in replacements.items():
        idx = by_id.get(axis_id)
        if idx is None:
            print(f"[warn] {axis_id} not found in {yaml_path.name}")
            continue
        data[idx] = new
        print(f"  [{axis_id}] {new['name']} に置き換え")
    yaml_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote {yaml_path}\n")


def main():
    print("=== high (S11/S13/S15/S20) ===")
    patch(ROOT / "classroom_personas.yaml", HIGH_FOREIGN)
    print("=== adult (A06/A11/A18/A20) ===")
    patch(ROOT / "classroom_personas_adult.yaml", ADULT_FOREIGN)
    print("=== elementary (E13/E18/E20) ===")
    patch(ROOT / "classroom_personas_elementary.yaml", ELEM_FOREIGN)


if __name__ == "__main__":
    main()
