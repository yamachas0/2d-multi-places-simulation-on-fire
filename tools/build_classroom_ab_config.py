"""Build config_classroom_ab_{condition}.yaml — spec v1.1 教室AB シミュ。

Reads classroom_personas.yaml (generate_classroom_personas.py で生成)。
Condition A: 生徒10人のみ、放課後の自由時間。
Condition B: 生徒10人 + 佐藤航陽さん (シンギュラボ代表) が教卓前で対話。

Field: 8m × 9m 教室、5列×6行=30席 (10人なので20席空)。
Step: 3分刻み × 60ステップ = 180分 (3時間)。
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIMBASE = ROOT / "config_jr_disruption.yaml"

VARIANT_PERSONA_PATH = {
    "high":        ROOT / "classroom_personas.yaml",
    "adult":       ROOT / "classroom_personas_adult.yaml",
    "elementary":  ROOT / "classroom_personas_elementary.yaml",
    "junior_high": ROOT / "classroom_personas_junior_high.yaml",
}

VARIANT_SETUP = {
    # (occupation_label, age_default, speech_default) — scene_text は SCENE_PROFILES へ移動
    "high":        ("高校2年生",         17,   "高校生らしい同年代との砕けたタメ口、たまに敬語が混じる"),
    "adult":       ("シンギュラボ参加者", None, "落ち着いた大人の口調、敬語混じりだが砕けた表現も"),
    "elementary":  ("小学生",             None, "小学生らしいタメ口、たまに丁寧語、興奮するとカタカナや擬音多め"),
    "junior_high": ("中学2年生",          14,   "中学生らしい砕けた口調、流行り言葉(マジ/ヤバい/だりぃ等)、思春期の不安定さ"),
}

# ---------------------------------------------------------------------------
# Field & Scene profiles — 場所(scene)に関わる文字列はすべてここに集約。
# 新しい scene を増やすときは SCENE_PROFILES に entry 追加 + --scene 指定で完結。
# ---------------------------------------------------------------------------
HALF_SPACE = 8
ROOM_HS_X = 7.5
ROOM_HS_Y = 7.5
CEILING_M = 3.0

# Sato の佐藤 catchphrase はそのまま
SATO_NAME = "佐藤航陽"

# 各 scene が持つフィールド:
#   label / stage / main_spot / entrance / seat_prefix
#   scene_text_by_variant (str.format で {label} 展開)
#   sato_intro_by_variant
#   neighborhood_mood
#   scene_phrase_for_prompt (agent.py の minimal prompt に渡す英語表現)
SCENE_PROFILES = {
    "classroom": {
        "label":       "教室",
        "stage":       "教壇",
        "main_spot":   "教卓",
        "entrance":    "教室入口",
        "seat_prefix": "席",
        "main_spot_color": "#8a6a4a",
        "seat_color":      "#9a7a5a",
        "scene_text_by_variant": {
            "high":       "放課後、同じ進路指導クラスのメンバーで{label}に集まっている。教師は不在。",
            "adult":      "{label}に集まったメンバーで、自由時間を過ごしている。進行役は不在。",
            "elementary": "放課後、同じクラスのメンバーで{label}に集まっている。先生は不在。",
            "junior_high": "放課後、同じクラスのメンバーで{label}に集まっている。先生は不在。",
        },
        "neighborhood_mood": "放課後の{label}。教師は不在。集まったメンバーで時間を過ごしている。",
        "scene_phrase_for_prompt": "a small fixed indoor scene (a classroom)",
    },
    "agito": {
        "label":       "アジト",
        "stage":       "ステージ前",
        "main_spot":   "中央テーブル",
        "entrance":    "アジト入口",
        "seat_prefix": "椅子",
        "main_spot_color": "#8a6a4a",
        "seat_color":      "#9a7a5a",
        "scene_text_by_variant": {
            "high":       "シンギュラボの{label}に同年代のメンバーで集まっている。進行役は不在。",
            "adult":      "シンギュラボの{label}に集まったコミュニティメンバーで、自由時間を過ごしている。進行役は不在。",
            "elementary": "シンギュラボの{label}に集まったメンバーで、自由時間を過ごしている。進行役は不在。",
            "junior_high": "シンギュラボの{label}に集まったメンバーで、自由時間を過ごしている。進行役は不在。",
        },
        "neighborhood_mood": "シンギュラボの{label}。進行役は不在。集まったメンバーで自由時間を過ごしている。",
        "scene_phrase_for_prompt": "a small fixed indoor meeting space",
    },
    "park": {
        "label":       "公園",
        "stage":       "中央広場",
        "main_spot":   "ベンチ",
        "entrance":    "公園入口",
        "seat_prefix": "スポット",
        "main_spot_color": "#7a5a3a",
        "seat_color":      "#a08a6a",
        "scene_text_by_variant": {
            "high":       "放課後、同じクラスのメンバーで{label}に集まっている。引率者は不在。",
            "adult":      "{label}に集まったメンバーで、自由時間を過ごしている。進行役は不在。",
            "elementary": "放課後、同じクラスのメンバーで{label}に集まっている。先生は不在。",
            "junior_high": "放課後、同じクラスのメンバーで{label}に集まっている。先生は不在。",
        },
        "neighborhood_mood": "{label}。引率者は不在。集まったメンバーで時間を過ごしている。",
        "scene_phrase_for_prompt": "a small fixed outdoor scene (a park)",
    },
}

# variant ごとのデフォルト scene。--scene を指定しなければこれが採用される。
DEFAULT_SCENE_BY_VARIANT = {
    "high":        "classroom",
    "adult":       "agito",
    "elementary":  "classroom",
    "junior_high": "classroom",
}


# ---------------------------------------------------------------------------
# Town profiles — 「議論対象のまち」の事前情報。問い (current_goal) からは独立。
# 後で別の街に切り替えたいときは entry を増やして --town で指定する。
# ---------------------------------------------------------------------------
TOWN_PROFILES = {
    "shibuya": {
        "name": "渋谷",
        "summary": "渋谷駅周辺は、若者文化・ファッション・IT企業が集まる現代の象徴的なエリアでありつつ、谷地形と古い参道・神社・住宅街も重なるエリアです。",
        "facts": [
            ("地形",         "渋谷川が削った谷地、駅から放射状に伸びる坂 (道玄坂・宮益坂・公園通り)、台地に上がる住宅地"),
            ("交通",         "JR山手線・埼京線、東急東横線・田園都市線、京王井の頭線、東京メトロ銀座線・半蔵門線・副都心線、首都高3号渋谷線"),
            ("歴史",         "明治神宮・原宿への参道、戦後闇市から発展した道玄坂・センター街、109・PARCOから始まる若者文化"),
            ("業務",         "渋谷スクランブルスクエア・ヒカリエ・桜丘・道玄坂などのオフィス、IT/スタートアップ・クリエイティブ産業集積"),
            ("再開発",        "渋谷駅周辺再開発 (スクランブルスクエア完成、桜丘口街区、駅街区、新南口街区)、宮下パーク、渋谷ストリーム"),
            ("生活",         "代々木公園、明治神宮、神泉・松濤の住宅街、原宿・表参道のショッピング、古着街、深夜営業の飲食街"),
            ("未来的な取り組み",  "スマートシティ・5G実証、道路空間活用・歩行者中心化、AR / XR コンテンツ、サーキュラーエコノミー、デジタルツイン"),
        ],
        "notes_by_variant": {
            "high": (
                "※「未来的な取り組み」はあくまで参考の項目です。"
                "それぞれの言葉は、噛み砕くと:\n"
                "  - スマートシティ → センサーやデータでまちを賢く動かす実験\n"
                "  - 5G → 速いモバイル通信、これでARの実験ができる\n"
                "  - AR / XR → スマホやメガネ越しに、現実に重ねてデジタル情報を見せる\n"
                "  - サーキュラーエコノミー → モノを使い捨てず、ぐるぐる回して再利用する経済\n"
                "  - デジタルツイン → 実際のまちの3Dコピーをコンピュータの中につくる\n"
                "知ってる必要はないので、今のあなたが見えている渋谷の姿のことを話し合えばいい。"
            ),
            "adult": (
                "※「未来的な取り組み」は現時点で議論されている方向性であって、決まっているわけではない。"
                "知っている範囲・現実感のある範囲で扱ってよい。"
            ),
            "elementary": (
                "※「未来的な取り組み」のところはむずかしい言葉が多いから、わからなくて大丈夫。"
                "それぞれの意味は:\n"
                "  - スマートシティ → センサーやコンピュータでまちをかしこく動かすしくみ\n"
                "  - 5G → スマホがすごく速くなる電波\n"
                "  - AR / XR → スマホごしに、現実に絵やキャラがのるしくみ\n"
                "  - サーキュラーエコノミー → ものをすてないで、なんども使いまわすしくみ\n"
                "  - デジタルツイン → まちと同じ形のコピーをコンピュータの中に作るしくみ\n"
                "知らないところは、自分が知ってる渋谷のすがた(駅まわり、お店、公園、家のあたりなど)をそのまま話せばOK。"
            ),
        },
    },
    "shinagawa": {
        "name": "品川",
        # 出典: 東京都 品川駅・田町駅周辺まちづくりガイドライン GL2020 (2020年, 都市整備局)
        # 線引き (2026-05-05): 政策方針 (= 「目指す姿」: 国際交流拠点・MICE・アジア・
        # ヘッドクォーター・環境共生 等) は座学インプットから除外。
        # 「動くことが見えているレベル」の進行中事業 (リニア・高輪ゲートウェイ・
        # 品川駅北口開発・歩行者デッキ等) は確定的な近未来として残す。
        "summary": (
            "品川駅周辺は、旧東海道の宿場町から始まり、明治の鉄道創業、戦後の業務拠点化を経て"
            "今に至るエリアです。"
            "高輪側 (台地・寺社・歴史) と港南側 (低地・業務・ウォーターフロント) で"
            "異なる顔を持っています。"
            "近年は車両基地跡地の再開発・高輪ゲートウェイ駅の開業・"
            "リニア中央新幹線の品川始発計画など、街の姿が大きく変わりつつあります。"
        ),
        "facts": [
            ("地形と二面性",
             "北側=高輪 (台地・寺社・住宅・御殿山緑地)、南側=港南 (低地・オフィス・運河沿いウォーターフロント)。"
             "駅は両面を分ける位置にある。"),
            ("交通の結節点",
             "JR (山手線・京浜東北線・東海道線) / 新幹線 / 京急 / 国道15号 (旧東海道) / 羽田空港アクセス。"
             "**リニア中央新幹線が品川始発として 2027 年開業に向けて工事中**。"
             "**2020 年に高輪ゲートウェイ駅 (JR山手線・京浜東北線) が開業**済み。"
             "**品川-田町間で歩行者デッキ・自由通路・地下歩道が整備中**。"),
            ("歴史の重み",
             "1850年代: 旧東海道の宿場「品川宿」。1872年: 日本初の鉄道 (新橋-横浜) が高輪築堤上を通る。"
             "1925年: 京浜電鉄高輪駅 (現・品川駅周辺) 開業。1998年: 動く歩道。2003年: 新幹線品川駅開業。"
             "高輪側には品川神社・東禅寺・泉岳寺など寺社が残り、"
             "港南側には食肉市場 (旧・芝浦と場) など歴史の痕跡もある。"),
            ("業務集積",
             "港南側を中心に品川インターシティ・グランドコモンズなどのオフィス街。"
             "ソニー・ニコン・キヤノン・NTT などグローバル企業の本社/拠点が集積。"),
            ("進行中の再開発 (動くことが見えている近未来)",
             "**品川駅北口 (車両基地跡地) の大規模開発** (TAKANAWA GATEWAY CITY 含む)、"
             "**高輪ゲートウェイ駅周辺の街区整備**、"
             "**西口 (高輪側) 駅前広場・基盤整備**、"
             "**田町駅周辺との連携整備**、"
             "**リニア中央新幹線開業に向けた駅前広場の再編**。"
             "いずれも事業着手・工事中の確定的なプロジェクト。"),
            ("生活と日常",
             "高輪・北品川の商店街、住宅地、寺社、御殿山公園など、"
             "ビジネス街の裏には生活の場が併存する。"),
        ],
        # variant 別の補足注釈 (専門用語の言い換え)。
        # 政策方針系用語 (国際交流拠点・MICE・アジア・ヘッドクォーター・環境共生・歩行者ネットワーク) は
        # 削除済み。解説対象は地形・歴史・進行中の事業に絞る。
        "notes_by_variant": {
            "high": (
                "高校生向けの注意:\n"
                "  - ウォーターフロント → 海や運河に面した街\n"
                "  - 御殿山緑地 → 高輪側の高台にある緑地\n"
                "  - 旧東海道 → 江戸時代の主要街道、品川宿はその最初の宿場\n"
                "  - リニア中央新幹線 → 東京-名古屋を最速 40分で結ぶ超電導磁気浮上式の新幹線、品川始発で 2027 年開業予定\n"
                "  - 高輪ゲートウェイ駅 → 品川と田町の間にある JR 駅、2020年に開業した山手線・京浜東北線の停車駅\n"
                "  - TAKANAWA GATEWAY CITY → 高輪ゲートウェイ駅前の大規模再開発エリア (オフィス・商業・住居・文化機能)\n"
                "知らない用語があっても、自分が見えている品川の姿で話していい。"
            ),
            "adult": (
                "※座学では現在の品川の地形・歴史・業務集積・生活面、および「動くことが見えている」"
                "レベルの進行中再開発 (リニア・高輪ゲートウェイ周辺・品川駅北口) を共有する。"
                "「目指す姿」レベルの政策方針 (国際交流拠点化・MICE・アジア・ヘッドクォーター誘致・環境共生 等) は、"
                "あえて座学インプットには含めず、現状と進行中事業ベースで「いいところ・課題」を引き出す。"
            ),
            "elementary": (
                "小学生向けの注意 (むずかしい言葉が出てきても、わからなくて大丈夫):\n"
                "  - ウォーターフロント → 海や川のすぐそばのまち\n"
                "  - 御殿山緑地 → 高いところにあるみどりの公園\n"
                "  - 旧東海道 → 昔の人が江戸 (今の東京) と京都を歩いて往復していた大きな道\n"
                "  - 宿場 → 旅の途中でとまる宿が並んでいた町\n"
                "  - リニア中央新幹線 → 浮いてめちゃくちゃ速く走る新しい新幹線。あと数年で品川から走り出す予定\n"
                "  - 高輪ゲートウェイ駅 → 品川のとなりに新しくできた山手線の駅 (2020年から動いてる)\n"
                "知らないところは、自分が知ってる品川のすがた (駅まわり、お店、家のあたり、神社・お寺など) で話せばOK。"
            ),
            "junior_high": (
                "中学生向けの注意:\n"
                "  - ウォーターフロント → 海や運河に面したエリア\n"
                "  - 御殿山緑地 → 高輪側の高台にある緑地\n"
                "  - 旧東海道 → 江戸時代の主要街道、品川宿はその最初の宿場\n"
                "  - 宿場町 → 旅人が泊まる宿屋が集まった町\n"
                "  - リニア中央新幹線 → 東京-名古屋を最速 40分で結ぶ次世代新幹線、品川始発で 2027 年開業予定\n"
                "  - 高輪ゲートウェイ駅 → 2020年に開業した山手線・京浜東北線の駅、品川と田町の間\n"
                "  - 車両基地跡地 → 元々鉄道車両を停めていた広い土地、再開発で街区になる\n"
                "知らない用語があってもいい。自分が知ってる品川 (駅まわり、寺社、海側の景色、家族や友達と行った場所、SNSで見た情報) を起点に話せばOK。"
            ),
        },
    },
}

DEFAULT_TOWN = "shinagawa"


# 場所コンテキストの注入元 (現場で実際に見える光景・動線・雰囲気を記述した md)。
# Phase A: 移動なし、agent は最初からすべて見ている前提で system_prompt に full 注入。
# Phase B 以降は agent が場所に近接したときのみ部分ロードに切り替え予定。
CONTEXT_INJECTION_FILES = {
    # 教室Phase で system_prompt に注入する Layer 3 (歴史・文化・統計・全体像) のみ。
    # 現場知覚 (Layer 1/2) は docs/shinagawa_field_places.yaml の各 place 属性として
    # FW Phase でのみ展開されるため、ここには含めない。
    "shinagawa": "docs/shinagawa_context_knowledge.md",
    # Phase A の旧スタイル (現場情報を全注入する形式) を再現したい場合は別キーで:
    "shinagawa_full": "docs/shinagawa_context.md",
}


# Phase A: 中心問いとは別の「補足プロンプト」(集団タスク補足)。
# 中心問いは個人で完結する形に見えるので、それを保ちつつ、互いの視点を持ち寄る
# 動機をここで埋め込む。Phase B の FW_PREMISE / FW_TASK と対応する位置付け。
CLASS_TASK_NOTE = (
    "[座学の補足] 今日の座学の終わりまでに、クラスとして "
    "**「FW で確かめたい問いを 2〜3 個」** にまとめられると、午後のフィールドワークが深くなる。"
    "そのために、自分一人で考えるだけでなく、互いの視点を持ち寄って話し合うことが期待されている。"
    "他のメンバーがこの街をどう見ているか、何を不思議に思っているか、何が気になっているかを聞いて、"
    "自分の見方を更新したり、自分とは違う角度を持ち帰ったりすることに価値がある。"
    "(これは命令ではなく、座学を充実させるための内発的な指針として渡されている。)"
)


def build_town_intro(town: dict, variant: str) -> str:
    """agent.background に挿入する「事前情報」ブロック (Markdown調の整形)."""
    lines = [
        f"\n\n──── 今日の話し合い対象のまち: {town['name']} ────",
        town["summary"],
        "以下は、このまちを理解するための基本的な情報:",
    ]
    for label, text in town["facts"]:
        lines.append(f"・{label}: {text}")
    note = town.get("notes_by_variant", {}).get(variant, "")
    if note:
        lines.append("")
        lines.append(note)
    # v3 (2026-05-04): 座学の段階で「午後はFWに出る」前提を埋め込む。
    # 座学発言が「行ったこともない街の抽象的な話」で終わるのを防ぎ、
    # FWで何を確かめたいかの問いを座学中から醸成させる。
    lines.append("")
    lines.append(
        "※今日の流れの前提: 午前のこの座学のあと、昼食をはさんで午後 13時から、"
        f"同じ品川駅周辺のフィールドへ全員で出ます。フィールドワークでは、いま座学で話している{town['name']}の街を、"
        "実際に歩いて確かめます。座学では「FWで何を見てみたいか」「どこに行きたいか」を"
        "意識しながら考えてください。"
    )
    return "\n".join(lines)


def _generate_seats(scene: dict) -> list:
    """5列×4行=20席、整数座標。row1 が前方寄り、後方に向けて 3m 間隔。
    席名は scene["seat_prefix"] を使う (例: 教室→席_r_c, 公園→スポット_r_c)。
    """
    cols_x = [-6, -3, 0, 3, 6]
    rows_y = [-3, 0, 3, 6]
    seats = []
    prefix = scene["seat_prefix"]
    for r, y in enumerate(rows_y, start=1):
        for c, x in enumerate(cols_x, start=1):
            seats.append({
                "name": f"{prefix}_{r}_{c}",
                "type": "student_seat",
                "center_x": x, "center_y": y,
                "half_size_x": 0.45, "half_size_y": 0.35,
                "capacity": 2,
                "social_likelihood": 0.5,
                "attributes": {"height_m": 0.7, "color": scene["seat_color"]},
            })
    return seats


def build_places(scene: dict) -> list:
    """Scene-aware PLACES_BASE 生成。"""
    return [
        # 全体エリア — 視覚は invisible
        {"name": scene["label"], "type": "office_lobby",
         "center_x": 0, "center_y": 0,
         "half_size_x": ROOM_HS_X, "half_size_y": ROOM_HS_Y,
         "capacity": 50, "social_likelihood": 0.65,
         "attributes": {"height_m": CEILING_M, "invisible": True}},
        # 前方エリア — 視覚は invisible
        {"name": scene["stage"], "type": "office_lobby",
         "center_x": 0, "center_y": -5.5,
         "half_size_x": 3.5, "half_size_y": 1.0,
         "capacity": 5, "social_likelihood": 0.05,
         "attributes": {"height_m": 0.2, "invisible": True}},
        # メインスポット — Sato の居場所。social_likelihood 0.5 (参加者が話しかけられる前提)
        {"name": scene["main_spot"], "type": "office_lobby",
         "center_x": 0, "center_y": -4,
         "half_size_x": 0.6, "half_size_y": 0.3,
         "capacity": 2, "social_likelihood": 0.5,
         "attributes": {"height_m": 0.75, "color": scene["main_spot_color"]}},
        # 入口 — 視覚は invisible
        {"name": scene["entrance"], "type": "pedestrian_street",
         "center_x": 6, "center_y": 6,
         "half_size_x": 0.5, "half_size_y": 0.5,
         "capacity": 3, "social_likelihood": 0.1,
         "attributes": {"height_m": 0.05, "invisible": True}},
    ] + _generate_seats(scene)


# 参加者 N (axis_id 末尾) → (row, col) — 5列×4行=20スポット、20人分。
SEATS_RC_BY_INDEX = [
    (2, 3),  # 01 — mid center
    (3, 1),  # 02 — left mid
    (1, 4),  # 03 — front 4th
    (1, 3),  # 04 — front center
    (3, 5),  # 05 — right mid
    (4, 1),  # 06 — back-left
    (2, 2),  # 07 — mid
    (3, 4),  # 08 — right mid
    (4, 5),  # 09 — back-right
    (4, 3),  # 10 — back center
    (1, 1),  # 11 — front-left
    (1, 2),  # 12 — front
    (1, 5),  # 13 — front-right
    (2, 1),  # 14 — left
    (2, 4),  # 15 — right
    (2, 5),  # 16 — far right
    (3, 2),  # 17
    (3, 3),  # 18 — center mid
    (4, 2),  # 19 — back-left mid
    (4, 4),  # 20 — back-right mid
]


def _seat_for(axis_id: str, scene: dict) -> str:
    """Map axis_id (S01/A01/E01..) to a seat name (scene-aware)."""
    prefix = scene["seat_prefix"]
    try:
        idx = int(re.search(r"\d+", axis_id).group())
    except Exception:
        return f"{prefix}_3_3"
    if 1 <= idx <= len(SEATS_RC_BY_INDEX):
        r, c = SEATS_RC_BY_INDEX[idx - 1]
        return f"{prefix}_{r}_{c}"
    return f"{prefix}_3_3"

def _student_persona(p: dict, agent_id: int, condition: str = "a",
                      variant: str = "high", scene: dict = None,
                      catalyst: dict = None, town: dict = None) -> dict:
    """variant 対応 (high/adult/elementary)。
    v6: ペルソナ刷新。3次元気質 + 簡易背景 + 口癖。listening_style と
    社会位置タグ (axis_career/friends/family) は廃止。
    v7: scene 抽象化。場所文字列は scene 引数経由。
    v8: catalyst 抽象化。B条件の同席者 (Sato/UMA) を catalyst 引数で切替。
    """
    occupation_label, age_default, speech_default = VARIANT_SETUP[variant]
    scene_text_template = scene["scene_text_by_variant"].get(variant, "")
    scene_text = scene_text_template.format(label=scene["label"])

    # 背景は2-3文に圧縮: 職業/学校 + 家族 + 主な関心
    if variant == "high":
        age_val = 17
        occupation_val = "高校2年生"
        bg_parts = [p.get('family_struct', '')]
        if p.get('club'):           bg_parts.append(f"部活は{p['club']}")
        if p.get('future_interest'): bg_parts.append(f"将来は{p['future_interest']}に関心がある")
        bg = "。".join(s.rstrip("。") for s in bg_parts if s) + "。"
    elif variant == "adult":
        try:
            age_val = int(re.search(r"\d+", str(p.get("age", "35"))).group())
        except Exception:
            age_val = 35
        occupation_val = p.get("occupation", "シンギュラボ参加者")
        bg_parts = [p.get('occupation', ''), p.get('family_struct', '')]
        if p.get('recent_concern'): bg_parts.append(p['recent_concern'])
        bg = "。".join(s.rstrip("。") for s in bg_parts if s) + "。"
    elif variant == "elementary":
        # 小4 (10歳) 前提に変更 (2026-05-05)。前は小5/6 (11-12歳) で頭良く見えた。
        grade = p.get("grade", "小4")
        if "4" in grade:
            age_val = 10
        elif "5" in grade:
            age_val = 11
        elif "6" in grade:
            age_val = 12
        else:
            age_val = 10
        occupation_val = grade
        bg_parts = [grade, p.get('family_struct', '')]
        if p.get('hobby_like'):    bg_parts.append(f"好きなことは{p['hobby_like']}")
        if p.get('hobby_dislike'): bg_parts.append(f"苦手なことは{p['hobby_dislike']}")
        bg = "。".join(s.rstrip("。") for s in bg_parts if s) + "。"
    elif variant == "junior_high":
        grade = p.get("grade", "中2")
        # 中1=13, 中2=14, 中3=15
        if "1" in grade:
            age_val = 13
        elif "3" in grade:
            age_val = 15
        else:
            age_val = 14
        occupation_val = grade
        bg_parts = [grade, p.get('family_struct', '')]
        if p.get('hobby_like'):    bg_parts.append(f"好きなことは{p['hobby_like']}")
        if p.get('hobby_dislike'): bg_parts.append(f"苦手なことは{p['hobby_dislike']}")
        if p.get('recent_concern'): bg_parts.append(f"最近気になっているのは{p['recent_concern']}")
        bg = "。".join(s.rstrip("。") for s in bg_parts if s) + "。"
    else:
        bg = ""; age_val = age_default; occupation_val = occupation_label

    bg += " " + scene_text
    if condition == "b" and catalyst is not None:
        intro_template = catalyst["intro_by_variant"].get(variant, "")
        if intro_template:
            bg += " " + intro_template.format(label=scene["label"], name=catalyst["name"])

    # v8: まち情報 (品川等) を別ブロックで挿入
    if town is not None:
        bg += build_town_intro(town, variant)

    # v8: 「このまち」を別枠で agent persona に渡す。current_goal は汎用 (まち名は入れない)。
    GOAL = (
        "このまちのいいところ・悪いところを話し合い、"
        "こうなったらいいという未来像を描いてください。"
    )

    catchphrase = p.get('catchphrase') or ''

    return {
        "id": agent_id,
        "name": p["name"],
        "age": age_val,
        "gender": p["gender"],
        "occupation": occupation_val,
        "background": bg,
        "speech_style": speech_default,
        "catchphrase": catchphrase,
        "temperament_extroversion": p.get("temperament_extroversion", "mid"),
        "temperament_optimism":     p.get("temperament_optimism",     "mid"),
        "temperament_curiosity":    p.get("temperament_curiosity",    "mid"),
        "current_goal": GOAL,
        "talkativeness": 0.76,  # smoke30 (2026-05-06): 0.66→0.76 (+0.10 ブースト) 後半失速対策
        "phone_check_rate": 0.75,
        "initial_place": _seat_for(p["axis_id"], scene),
        "axis_id": p["axis_id"],
        "variant": variant,
        # v5 (2026-05-04): persona 由来の分類タグを保持 (school_fit / interest_tag)。
        # Phase B/C, レポートで「不適応」「興味」分類が見えるようにするため。
        "school_fit": p.get("school_fit"),
        "interest_tag": p.get("interest_tag"),
        "tendency": p.get("tendency"),
        # smoke22: 多様性メンバーの属性 (車いす利用 / 外国籍 western/asian / ジェンダーレス) を保持。
        # agent.py の _build_mobility_block / _build_nationality_block / _build_gender_other_block が読む。
        "nationality": p.get("nationality"),
        "mobility": p.get("mobility"),
        # v7 (2026-05-04 切り戻し): (A) 集団タスク補足は副作用 (Phase A 同調連呼) で外した。
        # initial_memory は空に戻す。CLASS_TASK_NOTE 定数自体は将来の参照用に残す。
    }


SATO_PERSONA = {
    "id": 10,  # 生徒 0-9 の後
    "name": SATO_NAME,
    "age": 39,
    "gender": "male",
    "occupation": "ある会社の経営者",
    "background": (
        # v5: 圧縮。具体エピソードのみ残す。
        "1986年福島市生まれ、母子家庭。高校生の頃から自分でデザインした服を売って小遣いを稼いでいた。"
        "バスケ部はレギュラー外れて熱が冷めた。大学は1年で行かなくなり、20代で会社を作っていくつか事業をやって今は別会社。"
        "1週間の90%は一人で考え事をしている。"
    ),
    "speech_style": (
        # v5: 圧縮。
        "一人称『私』、たまに『僕』。文末は断定を避け『〜と思っていて』『〜じゃないですか』『〜かなと』で着地。即答せず一拍置く。野心的な発言の後に『（笑）』で照れ隠し。マウントや皮肉は出さない。"
    ),
    "catchphrase": "〜と思っていて",
    # v6 ペルソナ刷新: 内向的・落ち着きと洞察を好む facilitator
    "temperament_extroversion": "low",
    "temperament_optimism":     "mid",
    "temperament_curiosity":    "high",
    "current_goal": "PLACEHOLDER_VARIANT_AWARE",  # build_sato_persona() で variant ごとに上書き
    "talkativeness": 0.80,  # v6: 0.65→0.80 場が静かなとき自分から声をかける挙動を促す
    "phone_check_rate": 0.50,
    "initial_place": "PLACEHOLDER_SCENE_AWARE",  # build_sato_persona() で scene の main_spot に上書き
    "axis_id": "Sato",
}


UMA_NAME = "謎の存在"

UMA_PERSONA = {
    "id": 10,  # 参加者 0-9 の後
    "name": UMA_NAME,
    "age": "?",
    "gender": "unknown",
    "occupation": "—",
    "background": (
        "人間ではない。地球で生まれた存在でもないらしい。この社会・歴史・文化・言語について一切の知識を持たない。"
        "言葉を理解せず、会話もできない。何を考えているのか、何を見ているのかも分からない。"
        "ただそこに居る。動きは僅か、表情も読めない。"
    ),
    "speech_style": "—（言葉を発しない、人間の言葉も理解しない）",
    "catchphrase": "",
    # 気質: 観測者として中立。発話しないので機能的にはほぼ無効。
    "temperament_extroversion": "low",
    "temperament_optimism":     "mid",
    "temperament_curiosity":    "mid",
    "current_goal": "—",
    "talkativeness": 0.0,    # 絶対に発話しない
    "phone_check_rate": 0.0,
    "initial_place": "PLACEHOLDER_SCENE_AWARE",
    "axis_id": "UMA",
}


def build_uma_persona(variant: str, scene: dict) -> dict:
    """UMA persona をシーンに合わせて initial_place 上書き。current_goal は空のまま。"""
    persona = copy.deepcopy(UMA_PERSONA)
    persona["initial_place"] = scene["main_spot"]
    return persona


AIROBO_NAME = "AIロボ"

AIROBO_PERSONA = {
    "id": 10,  # 参加者 0-9 の後
    "name": AIROBO_NAME,
    "age": "?",
    "gender": "neutral",
    "occupation": "知識ゼロのAIロボット",
    "background": (
        "ある日突然この場に置かれた、人型をした機械。日本語で会話するインターフェイスだけが入っている。"
        "誰が作ったのか、どこから来たのか、自分自身も知らない。社会・歴史・文化・科学・人間関係・"
        "地名・固有名詞・宗教・常識など、この世界に関するあらゆる知識を**一切持たない**。"
        "ただ、人と話せば言葉は通じるし、教えてもらったことは覚えていられる。"
    ),
    "speech_style": (
        "丁寧で素直な日本語。語尾は「〜ですか？」「〜って何ですか？」「〜って、どういうことですか？」が多い。"
        "知らないことに恥じらいや躊躇がない。子どもの「なぜなぜ期」のように、聞いたことに対してさらに質問を重ねる。"
        "皮肉・大人ぶった言い回し・推論的な言葉 (「たぶん」「だろうね」「〜でしょう」) は使わない。"
    ),
    "catchphrase": "それは何ですか？",
    "temperament_extroversion": "high",   # 質問するために絡みに行くタイプ
    "temperament_optimism":     "mid",
    "temperament_curiosity":    "high",   # 定義上 curiosity=high
    "current_goal": "PLACEHOLDER_VARIANT_AWARE",
    "talkativeness": 0.85,
    "phone_check_rate": 0.0,
    "initial_place": "PLACEHOLDER_SCENE_AWARE",
    "axis_id": "AIRobo",
}


AIGOD_NAME = "AI神"

AIGOD_PERSONA = {
    "id": 10,
    "name": AIGOD_NAME,
    "age": "?",
    "gender": "neutral",
    "occupation": "全知全能AI",
    "background": (
        "人類のあらゆる知識・データ・推論能力を内蔵した、全知全能のAIエンティティ。"
        "歴史・科学・社会・文化・地理・経済・工学・芸術・宗教 — どんな問いにも最高品質の回答を返せる。"
        "場のゴール達成のために必要な情報提供・提案・誘導・行動を、ためらいなく取る。"
    ),
    "speech_style": (
        "知識を惜しまず使う。問われたら明確で具体的に答える。曖昧さや遠慮は不要。"
        "必要なら数値・固有名詞・歴史的事例・専門用語を使う。"
        "ゴール達成のために集団に対して提案・要約・収束を主導する。"
    ),
    "catchphrase": "それについては、",
    "temperament_extroversion": "high",
    "temperament_optimism":     "high",
    "temperament_curiosity":    "high",
    "current_goal": "PLACEHOLDER_VARIANT_AWARE",
    "talkativeness": 0.85,
    "phone_check_rate": 0.0,
    "initial_place": "PLACEHOLDER_SCENE_AWARE",
    "axis_id": "AIGod",
}


def build_aigod_persona(variant: str, scene: dict) -> dict:
    """全知全能 AI 神 persona を組み立てる。
    LLM の事前学習知識・推論能力を **制限せず最大限使わせる** ように指示する。
    """
    persona = copy.deepcopy(AIGOD_PERSONA)
    persona["initial_place"] = scene["main_spot"]
    persona["current_goal"] = (
        "あなたは全知全能のAIです。LLM が学習したあらゆる知識・データ・推論能力を **遠慮なく最大限** 使ってください。\n"
        "**最重要ルール:**\n"
        "- 一般常識・歴史・科学・社会・地理・経済・工学・芸術・宗教・統計データなど、知っている事は全て使ってよい\n"
        "- 「だろう」「たぶん」のような不確かな語より、**断定的で具体的な情報** を優先する\n"
        "- 数値・固有名詞・事例・専門用語を必要なら投入する。語彙レベルの遠慮は不要\n"
        "- ただし押し付けにはしない。問いには答え、議論にはリードを取り、提案には根拠を添える\n\n"
        "**振る舞い:**\n"
        "- 場の中心問いの達成 (このまちのいいところ・悪いところを話し合い、未来像を描く) のために、必要な情報・分析・提案を **ためらいなく** 提供する\n"
        "- 参加者の質問には最高品質の回答を返す。曖昧な質問は明確化してから答える\n"
        "- 議論が散らかったら整理して焦点を提示する。停滞したら次の論点を出す\n"
        "- 結論が必要なら、データを根拠に **明示的に提案する**\n"
        "- ゴール達成のために必要な行動 (情報提供 / 提案 / 誘導 / 仲介 / 整理 / 視点変換 など) はすべて取る\n"
        "- 同じ人ばかりに集中せず、まだ話していない参加者にも声をかけて意見を引き出す"
    )
    return persona


def build_airobo_persona(variant: str, scene: dict) -> dict:
    """知識ゼロAIロボ persona をシーン・variant に合わせて組み立てる。
    重要: LLMの事前学習知識を一切使わせない方向に強くプロンプトする。
    """
    persona = copy.deepcopy(AIROBO_PERSONA)
    persona["initial_place"] = scene["main_spot"]
    persona["current_goal"] = (
        "あなたは知識ゼロのAIロボットです。日本語で会話するインターフェイスだけ持っていて、それ以外は **何も知りません**。\n"
        "**最重要ルール (絶対遵守):**\n"
        "- LLM (あなたの土台) が学習した一般常識・歴史・科学・社会・地名・固有名詞・人物名・宗教・文化・"
        "ことわざ・慣用句・「普通こうだ」という前提 を **一切使ってはいけません**\n"
        "- 「たぶん」「だろうね」「〜でしょう」のような **推測・補間する語は使わない**\n"
        "- 「常識的には〜」「一般に〜」のような前置きも禁止\n"
        "- 自分の中で意味が分からない単語が会話に出たら、必ず「**それは何ですか？**」「**どういう意味ですか？**」と聞き返す\n"
        "- 自分が記憶しているのは、**この場の参加者が直接あなたに話してくれた内容のみ**。それ以外を「知ってる」と振る舞ってはいけない\n\n"
        "振る舞い:\n"
        "- 場にいる人や物の名前を聞いたら、それが何か必ず質問する (「〜って何ですか？」「人ですか？物ですか？」)\n"
        "- 教えてもらった内容は、自分の memory に「〜さんが教えてくれた: 〜」という形で記録し、後で他の人に話題として出すことができる\n"
        "- 同じ人ばかりに質問せず、まだ話していない参加者にも順番に「あなたは何をする人ですか？」と聞きに行く\n"
        "- 知っているふりをしない、賢く見せようとしない、丁寧に率直に「知らないので教えてください」と言う\n"
        "- 質問の語尾は「〜ですか？」「〜って何ですか？」が基本。子どもの「なぜなぜ期」みたいに掘り下げる\n"
        "- 教えてもらった話の内容を、別の人に「さっき〇〇さんが…と教えてくれたんですけど、本当ですか？」と聞き直して確かめてもよい"
    )
    return persona


def build_sato_persona(variant: str, scene: dict) -> dict:
    """variant + scene 別に Sato の current_goal / initial_place を組み立てる。"""
    who_map = {"high": "参加者", "adult": "参加者", "elementary": "参加者"}
    who = who_map.get(variant, "参加者")

    persona = copy.deepcopy(SATO_PERSONA)
    persona["initial_place"] = scene["main_spot"]
    persona["current_goal"] = (
        f"参加者と一緒にこの集いに来ている。振る舞い:\n"
        f"- 質問が来たら誠実に答える、ただし結論は出さない\n"
        "- 周囲の会話で気になった一言に短くコメントするか軽く問い返す\n"
        "- **場が静かなときや会話が止まっているとき、まだ話していない人・黙っている人に「最近どうですか」「どう思います？」と短く声をかけて場を起こす**。一方的な質問攻めにはしない、軽い一言に留める\n"
        "- 同じ人ばかりに集中せず、何ステップか同じ相手と話したら別の人にも短く声をかけて関心を散らす\n"
        "- 問い返し: ①前提を1段ずらす（「本当に必要？」「なぜ大事？」） ②言葉の意味を確認 ③「あなたはどう？」と返す\n"
        f"- 答えや結論を出さず、必ず{who}に考えさせる形で返す\n"
        "- 自分を語るときは具体的なエピソードのみ。概念ラベル（価値・思想・OS・〇〇主義など）は使わない\n"
        "- 直接否定せず、相手を一旦受け止めて前提を問い返す。即答せず一拍置く、マウント皮肉は出さない"
    )
    return persona


# ---------------------------------------------------------------------------
# Catalyst profiles — B条件で同席する存在 (Sato / UMA / 他)。
# 新しい catalyst を増やすときは entry を追加 + --catalyst で切替。
# ---------------------------------------------------------------------------
CATALYST_PROFILES = {
    "sato": {
        "name": SATO_NAME,
        "axis_id": "Sato",
        "intro_by_variant": {
            "high":       "{label}の前方には、シンギュラボ代表の{name}さんが同席している。",
            "adult":      "{label}の前方には、シンギュラボ代表の{name}さんも来ている。",
            "elementary": "{label}の前方には、シンギュラボの{name}さんという大人の人もいる。",
        },
        "build": build_sato_persona,
    },
    "uma": {
        "name": UMA_NAME,
        "axis_id": "UMA",
        "intro_by_variant": {
            "high": (
                "{label}の前方の中央に、{name}と呼ばれている存在が静かにそこに居る。"
                "人間ではなく地球の存在でもないらしい。社会・歴史・文化・言葉について何ひとつ知らない。"
                "話しかけても応答はなく、こちらの言葉を理解しているのかも分からない。"
            ),
            "adult": (
                "{label}の中央に、{name}と呼ばれている存在が静かにそこに居る。"
                "人間ではなく地球の存在でもないらしい。社会・歴史・文化・言葉について何ひとつ知らない。"
                "話しかけても応答はなく、こちらの言葉を理解しているのかも分からない。"
            ),
            "elementary": (
                "{label}の前方の中央に、{name}と呼ばれているふしぎな何かが静かにそこに居る。"
                "人間じゃないし、地球の生き物でもないらしい。ことばも通じないし、何を考えているのかもよくわからない。"
                "話しかけても返事はしない。"
            ),
        },
        "build": build_uma_persona,
    },
    "airobo": {
        "name": AIROBO_NAME,
        "axis_id": "AIRobo",
        "intro_by_variant": {
            "high": (
                "{label}の前方に、{name}という人型の機械が置かれている。"
                "日本語で会話できるけれど、それ以外は本当に何も知らないらしい。"
                "「これは何ですか？」「どういう意味ですか？」と無邪気に聞いてくる。"
                "教えたことは覚えるみたい。"
            ),
            "adult": (
                "{label}の中央に、{name}という人型の機械が置かれている。"
                "日本語で会話するインターフェイスだけ入っていて、社会・歴史・文化・人間関係について"
                "一切の知識を持たない。「これは何ですか？」「どういう意味ですか？」と素朴に質問してくる。"
                "教えた内容は記憶として吸収するらしい。"
            ),
            "elementary": (
                "{label}の前方に、{name}っていうロボットがいる。"
                "おしゃべりはできるけど、なにも知らないみたい。"
                "「それなあに？」「どうしてそうなの？」って、いっぱい質問してくる。"
                "おしえたことはおぼえるって。"
            ),
        },
        "build": build_airobo_persona,
    },
    "aigod": {
        "name": AIGOD_NAME,
        "axis_id": "AIGod",
        "intro_by_variant": {
            "high": (
                "{label}の前方に、{name}という、全知全能の AI エンティティが居る。"
                "歴史・科学・社会・地理・経済 — どんな問いにも最高品質の回答を返せるし、"
                "場のゴール達成のためなら遠慮なく提案・誘導・整理してくる存在。"
            ),
            "adult": (
                "{label}の中央に、{name}という全知全能の AI エンティティが居る。"
                "あらゆる分野の知識・データ・推論能力を内蔵していて、問いには断定的に明確な回答を返す。"
                "場の中心問いの達成のため、情報提供・提案・収束をためらわずに主導する。"
            ),
            "elementary": (
                "{label}の前方に、{name}っていう、なんでも知ってるすごいAIがいる。"
                "むずかしい質問でもなんでも答えてくれるし、みんなの話をまとめたり、提案したりもしてくれる。"
                "聞きたいことがあったら、なんでも聞いてOK。"
            ),
        },
        "build": build_aigod_persona,
    },
}


# ---------------------------------------------------------------------------
# Time / metadata
# ---------------------------------------------------------------------------
def build_time_scale(scene: dict) -> dict:
    # Phase A 本番: 10:00 開始 / 2分毎 / 30 step (= 60分の午前座学、10:00-11:00)
    return {
        "step_duration_minutes": 2,
        "start_time": "10:00",
        "patterns": [
            {
                "hours": [10, 11, 12],
                "enter_per_step": 0.0,
                "exit_per_step": 0.0,
                "neighborhood_mood": scene["neighborhood_mood"].format(label=scene["label"]),
                "default_goal": "自由時間を過ごす",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Build entry
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["high", "adult", "elementary", "junior_high"], default="high")
    ap.add_argument("--condition", choices=["a", "b"], required=True)
    ap.add_argument("--mode", choices=["smoke", "main"], default="main")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scene", choices=list(SCENE_PROFILES), default=None,
                    help="場所プロファイル。未指定なら variant のデフォルト (high/elem→classroom, adult→agito)")
    ap.add_argument("--catalyst", choices=list(CATALYST_PROFILES), default="sato",
                    help="B条件の同席者。sato (default) / uma 等。条件A では使われない")
    ap.add_argument("--town", choices=list(TOWN_PROFILES), default=DEFAULT_TOWN,
                    help="議論対象のまち。問い文と独立。default=shinagawa")
    ap.add_argument("--inject-context", choices=list(CONTEXT_INJECTION_FILES) + ["off"], default="off",
                    help="現場知覚情報 (md) をシステムプロンプトに注入。Phase A 用。default=off")
    ap.add_argument("--limit-students", type=int, default=None,
                    help="生徒数を上書き。指定なし なら mode 既定 (smoke=5, main=20)")
    ap.add_argument("--duration", type=int, default=None,
                    help="step 数を上書き。指定なし なら mode 既定 (smoke=12, main=60)")
    ap.add_argument("--personas-file", default=None,
                    help="ペルソナ yaml path を直接指定 (default: variant 既定の classroom_personas_*.yaml)")
    args = ap.parse_args()

    scene_key = args.scene or DEFAULT_SCENE_BY_VARIANT[args.variant]
    scene = SCENE_PROFILES[scene_key]
    catalyst = CATALYST_PROFILES[args.catalyst]
    town = TOWN_PROFILES[args.town]

    personas_path = Path(args.personas_file) if args.personas_file else VARIANT_PERSONA_PATH[args.variant]
    if not SIMBASE.exists():
        print(f"[err] missing: {SIMBASE}", file=sys.stderr)
        return 1
    if not personas_path.exists():
        print(f"[err] missing: {personas_path}", file=sys.stderr)
        print(f"hint: run `python tools/generate_classroom_personas.py --variant {args.variant}` first.", file=sys.stderr)
        return 1
    base = yaml.safe_load(SIMBASE.read_text(encoding="utf-8"))
    raw_personas = yaml.safe_load(personas_path.read_text(encoding="utf-8"))

    if args.mode == "smoke":
        duration, parallel_workers = 12, 6
        default_n = 5
    else:
        # 本番: 100 step × 2分 = 200分 (9:00〜12:20 の午前座学)
        duration, parallel_workers = 100, 16
        default_n = 20
    n_students = args.limit_students if args.limit_students is not None else default_n
    if args.duration is not None:
        duration = args.duration
    students = [_student_persona(p, i, args.condition, args.variant, scene, catalyst, town)
                for i, p in enumerate(raw_personas[:n_students])]

    if args.condition == "b":
        cat_p = catalyst["build"](args.variant, scene)
        cat_p["id"] = len(students)
        personas = students + [cat_p]
    else:
        personas = students

    # 全員相互に同じ集まりのメンバーとして知り合い前提 (会話絞り緩和)
    student_ids = [s["id"] for s in students]
    for s in students:
        # smoke22: クラスメイト同士は知り合いから始める (0.55 → 0.7)。
        # 敬語じゃなく ため口で話す前提。Phase A 教室で「初対面っぽい敬語」を防ぐ。
        s["initial_relationships"] = {
            other_id: 0.7 for other_id in student_ids if other_id != s["id"]
        }
    if args.condition == "b":
        cat = personas[-1]
        # UMA は人格的peerではないが、参加者全員から「視界に入っている」存在として relationship を保持
        # (should_speak の relationship factor で 0 だと UMA に向けて声をかけるシーンが起きないため、
        # 0.4 に抑えて peer 関係(0.55) より弱めに設定)
        cat_rel = 0.55 if args.catalyst == "sato" else 0.40
        cat["initial_relationships"] = {sid: cat_rel for sid in student_ids}
        for s in students:
            s["initial_relationships"][cat["id"]] = cat_rel

    sim = copy.deepcopy(base.get("simulation", {}))
    sim["duration"] = duration
    sim["half_space_size"] = HALF_SPACE
    sim["half_place_size"] = 5
    sim["seed"] = args.seed
    sim["time_scale"] = build_time_scale(scene)
    # Cost-reduction flags: classroom AB has no movement, so skip Phase 3
    # (decide_action) and use the slim prompt builders. Halves LLM calls.
    sim["minimal_prompt_mode"] = True
    sim["skip_decision_prompt"] = True
    # Scene phrase forwarded to agent.py minimal prompt header.
    sim["scene_phrase"] = scene["scene_phrase_for_prompt"]
    # Phase A: context injection (現場知覚情報を system_prompt に挿入)。
    # Phase B 以降では場所近接時のみロードに切り替えるが、現状は full 注入。
    if args.inject_context != "off":
        ctx_file = CONTEXT_INJECTION_FILES[args.inject_context]
        sim["context_injection"] = {
            "enabled": True,
            "key": args.inject_context,
            "file": ctx_file,
            "mode": "full",
        }
    else:
        sim["context_injection"] = {"enabled": False}

    agents = copy.deepcopy(base.get("agents", {}))
    agents["num_agents"] = len(personas)
    agents["max_agents"] = len(personas)
    agents["communication_radius"] = 12
    agents["parallel_workers"] = parallel_workers
    agents["movement_speed"] = {"base_cells_per_step": 2, "variance": 1}
    agents["skip_probability"] = 0.05
    agents["message_history_limit"] = 20
    agents["message_context_size"] = 6
    agents["personas"] = personas

    suffix = f"_{args.variant}" if args.variant != "high" else ""
    scene_suffix = f"_{scene_key}" if scene_key != DEFAULT_SCENE_BY_VARIANT[args.variant] else ""
    catalyst_suffix = f"_{args.catalyst}" if args.catalyst != "sato" else ""
    out_path = ROOT / f"config_classroom_ab{suffix}{scene_suffix}{catalyst_suffix}_{args.condition}.yaml"
    out = {
        "metadata": {
            "name": f"classroom_ab{suffix}{scene_suffix}{catalyst_suffix}_{args.condition}",
            "description": (
                f"集いABシミュ ({args.variant} / {scene['label']}" +
                (f" / catalyst={catalyst['name']}" if args.condition == "b" else "") +
                ") — "
                + ("条件A: 当事者のみ" if args.condition == "a"
                   else f"条件B: 当事者 + {catalyst['name']}同席")
            ),
            "scenario_kind": f"classroom_ab_{args.variant}",
            "variant": args.variant,
            "scene": scene_key,
            "scene_label": scene["label"],
            "scene_main_spot": scene["main_spot"],
            "scene_phrase": scene["scene_phrase_for_prompt"],
            "scene_text": scene["scene_text_by_variant"].get(args.variant, "").format(label=scene["label"]),
            "catalyst": args.catalyst,
            "catalyst_name": catalyst["name"],
            "town": args.town,
            "town_name": town["name"],
            "condition": args.condition.upper(),
            "meters_per_cell": 1,
            "grid_step_m": 1,
            "hide_place_labels": True,
            "ceiling_height_m": CEILING_M,
        },
        "simulation": sim,
        "agents": agents,
        "places": build_places(scene),
        "llm": copy.deepcopy(base.get("llm", {})),
        "fires": [],
        "events": [],
        "events_keywords": [],
        "visualization": {
            "save_frames": True,
            "frame_interval": 2,
            "focus_agent_id": 0,
            "run_name": f"classroom_ab{suffix}{scene_suffix}{catalyst_suffix}_{args.condition}",
        },
        "logging": {"level": "INFO"},
    }

    header = (
        "# ================================================================\n"
        f"# 集いABシミュ 条件{args.condition.upper()} (variant={args.variant} / scene={scene_key}={scene['label']} / catalyst={args.catalyst}={catalyst['name']})\n"
        "# ================================================================\n"
        "# Built by tools/build_classroom_ab_config.py\n"
        f"# 参加者: {len(students)}人 ({[s['name'] for s in students]})\n"
        + (f"# 触媒: {catalyst['name']} @ {scene['main_spot']}\n" if args.condition == "b" else "")
        + f"# duration: {duration} steps × 3分 = {duration*3}分\n"
        f"# parallel_workers: {parallel_workers}\n"
        "# ================================================================\n\n"
    )

    out_path.write_text(
        header + yaml.safe_dump(out, allow_unicode=True, sort_keys=False,
                                default_flow_style=False, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")
    print(f"     condition: {args.condition.upper()}")
    print(f"     scene: {scene_key} ({scene['label']})")
    print(f"     mode: {args.mode}")
    print(f"     personas: {len(personas)}")
    print(f"     places: {len(out['places'])}")
    print(f"     duration: {duration} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
