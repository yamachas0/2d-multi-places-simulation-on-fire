"""Phase B: 品川FWフィールドの整備。
merged yaml (D:/ユーザー/ダウンロード/shinagawa_config.merged.yaml) を読み込み、
- 重複名 (居酒屋 (複製) ×7) を 港南ペンシルビル_1〜8 にリネーム
- 空 name を埋める (wide_street → 国道15号、residential → 高輪3丁目住宅地)
- 新規 place 「新幹線改札前」を spawn_point として追加
- 各 place に perceive_pass / perceive_enter / environment 属性を付与

出力: docs/shinagawa_field_places.yaml (Phase B build_shinagawa_field_config.py で読む)

Run:
  python tools/build_shinagawa_field.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"D:\ユーザー\ダウンロード\shinagawa_config.merged.yaml")
OUT = ROOT / "docs" / "shinagawa_field_places.yaml"

# --- 名前マッピング (重複・空名の置換) ---
# (旧名 or idx, 新名) — name 一致で置換、空名は idx で
RENAME_BY_NAME = {
    "居酒屋":                              "港南ペンシルビル_南1",
    "居酒屋 (複製)":                       None,  # 複数あるので idx ベースで処理
    "居酒屋 (複製) (複製)":                None,
    "居酒屋 (複製) (複製) (複製)":         None,
    "居酒屋 (複製) (複製) (複製) (複製)":  None,
    "京急1ビル (複製)":                    "京急1ビル",
}

# (idx, 新名) 居酒屋 (複製)系 (idx=10 以外) を coord でマップ
RENAME_BY_IDX = {
    29: "港南ペンシルビル_北1",   # (42.0,-2.1)
    32: "港南ペンシルビル_南2",   # (47.8, 4.8)
    33: "港南ペンシルビル_北2",   # (48.0,-2.1)
    30: "港南ペンシルビル_南3",   # (53.8, 4.8)
    34: "港南ペンシルビル_北3",   # (54.0,-2.1)
    31: "港南ペンシルビル_南4",   # (59.8, 4.8)
    35: "港南ペンシルビル_北4",   # (60.0,-2.1)
}

# 空 name の置換 (idx → 新名)
RENAME_EMPTY = {
    4:  "国道15号(第一京浜)",
    41: "高輪台住宅地",
    16: "再開発工事中A地区",  # （建設中）の1つ目
    39: "再開発工事中D地区",  # 2つ目
}

# --- 新規 place ---
NEW_PLACES = [
    {
        "name": "新幹線改札前",
        "type": "subway_station",
        "center_x": 8.8,
        "center_y": 0.3,
        "half_size_x": 2.0,
        "half_size_y": 1.5,
        "capacity": 100,
        "social_likelihood": 0.3,
        "is_spawn_point": True,
        "attributes": {"height_m": 4},
    },
]

# --- 各 place の perceive_pass / perceive_enter / environment マッピング ---
# Layer 1 通過知覚 = 通り過ぎたときに見える物理・視覚・人流・雰囲気
# Layer 2 探索知覚 = 中に入る/立ち止まって見たときに気づく内部構造・隠れた要素
# environment = 物理タグ (indoor/roof/grade/near_construction/near_park ...)

PERCEIVE = {
    # === 駅構内 ===
    "新幹線改札前": {
        "perceive_pass": "大きなアトリウム、出張族と観光客が行き交い、案内放送が常に流れる。2階にスタバ、東側アトレ3階にブルーボトルが見える。",
        "perceive_enter": "改札を入ると新幹線ホームへ続く長い通路、改札外も天井が高く明るい。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "JR改札": {
        "perceive_pass": "ガラスのアトリウムで明るい。小さいが目印になる時計がある。待ち合わせや行き交う人であふれている。",
        "perceive_enter": "改札内に入るとエキナカ(飲食・物販)、各ホームへ階段とエスカレーター。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "京急改札": {
        "perceive_pass": "青いサインなし、狭くてごちゃごちゃしている。JR改札から高輪口側に降りる工事中の無機質な階段、バリアフリーELVもある。",
        "perceive_enter": "改札内ホームは狭く、京急本線の電車が頻繁に出入りする。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "品川駅 東西自由通路": {
        "perceive_pass": "天井高い長い通路。中央と端で人の流れが逆方向に走るが通行は秩序立っている。港南口に向かう動線、行き交う人がうじゃうじゃ。",
        "perceive_enter": None,  # 通路なので「中に入る」概念なし
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },

    # === 港南口エリア ===
    "アトレ品川": {
        "perceive_pass": "東西自由通路レベル(2F)がメイン入口。飲食・物販店舗が5階まで広がる。平面の面積は小さいが上に積層している。",
        "perceive_enter": "中に入ると各フロアにブランドショップとレストラン、エスカレーターで階を移動。3階にブルーボトルコーヒー。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南口広場": {
        "perceive_pass": "アトレを抜けて港南口に出ると、北はアレア品川、南はグランドコモンズとインターシティに向かうデッキが伸びる。屋根付きでアンブレラフリー。正面に居酒屋やバー含む飲食店街がまっすぐ伸びる。",
        "perceive_enter": "エスカレーターと階段で広場に降りられる。階段の裏側にタクシー乗り場、アレア2F脇からタクシー乗り場に直接降りれる隠れた階段がある。",
        "environment": {"indoor": False, "roof": True, "grade": "flat"},
    },
    "アレア品川": {
        "perceive_pass": "デッキレベル(2F)にドトールとオフィスロビー、吹き抜けエスカレーターでつながる1Fに飲食店がいくつか。",
        "perceive_enter": "1Fから外に出てまっすぐ進むとNTTビルに至る。シーズンテラス側にはうまく抜けられない(東の道路側に抜ける必要)。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "NTT": {
        "perceive_pass": "古い大きなビル。1Fがオフィスロビーのみ、飲食店等はなく閉鎖的な雰囲気。",
        "perceive_enter": "ロビーは入れるが用事がないと長居できる場所はない。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "イーストワンタワー": {
        "perceive_pass": "グランドコモンズ最北棟、デッキレベルがセントラルガーデンに沿って南北に貫く歩行者メイン動線。",
        "perceive_enter": "オフィスロビーがデッキレベルに面する。1Fは線路側道路から接続する車寄せや駐車場出入口がメインで車中心、用事のないエリア。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "太陽生命": {
        "perceive_pass": "グラコモ群の中の1棟、デッキで隣接ビルとつながる。",
        "perceive_enter": "オフィスロビーは静かで人の出入りが少ない。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "マイクロソフト": {
        "perceive_pass": "グラコモ群の中の1棟、青いサインが目立つ。",
        "perceive_enter": "ロビーが開放的、来訪者用受付がある。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "キャノン": {
        "perceive_pass": "グラコモ群の中の1棟、本社オフィス。",
        "perceive_enter": "1Fロビーに製品ディスプレイ、訪問者向けスペース。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "Vタワーレジデンス": {
        "perceive_pass": "グラコモ南端、レジデンス棟。住人のみ立ち入れる雰囲気。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "インターシティA棟": {
        "perceive_pass": "グラコモ同様デッキレベルが歩行者動線。セントラルガーデンに沿って湾曲しながら南へアクセス。",
        "perceive_enter": "オフィスロビー、ビジネス層中心。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "インターシティ": {
        "perceive_pass": "ビル内に南北に通り抜ける通路、店舗が並ぶ(駅側=タリーズ、中央=スタバ・サンマルク、奥=マック)。オフィス・カンファレンスが主、2F以外はイベント等の目的がある人で日によって濃淡。",
        "perceive_enter": "通路を歩くと店舗が連続、奥に進むほど駅から離れた静けさ。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "品川セントラルガーデン": {
        "perceive_pass": "南北に長い広場、グラコモとインターシティに挟まれた谷間のような場所。比較的開放的、デッキからも眺められる、憩える場所も点在。",
        "perceive_enter": "ベンチや植栽がある。上空2Fレベルでグラコモとインターシティを繋ぐデッキが2本走る。南端に近づくと「品川駅前ここまで」の境界感。",
        "environment": {"indoor": False, "roof": False, "grade": "flat", "near_construction": False, "near_park": True},
    },
    "品川シーズンテラス": {
        "perceive_pass": "1Fはほぼ駐車場で閉鎖的、コンビニ程度。エスカレーター/階段で2Fに誘導される。",
        "perceive_enter": "2Fが商業店舗(カフェからレストランまで)、北側が公園に面してテラス席など雰囲気のいい店。3Fがオフィスロビーとカンファレンス。",
        "environment": {"indoor": True, "roof": True, "grade": "flat", "near_park": True},
    },
    "ソニー": {
        "perceive_pass": "1Fオフィスロビーに過去製品が並び、椅子テーブルあり。開放的で明るい雰囲気。",
        "perceive_enter": "誰でも入れる雰囲気で、過去のソニー製品を眺められる。タリーズあり。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "コクヨ": {
        "perceive_pass": "北館1Fがショールーム、入口は狭いが中はインテリアメーカーらしいおしゃれな空間。",
        "perceive_enter": "南館は本社オフィスでライブオフィス、社員が実際に働いているのが見える、開放的でおしゃれ。コクヨより東側は大小の雑居ビル(特徴薄)。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_南1": {
        "perceive_pass": "10階建前後のペンシルビル、1F〜2Fに居酒屋・バー。昼時や夕方以降は利用客で賑わう、品川でいわゆる雑多な飲み屋街。",
        "perceive_enter": "狭い階段で各階に上がると小さな飲食店、風俗店も数件混じる。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_南2": {
        "perceive_pass": "10階建前後のペンシルビル、1F〜2Fに様々な飲食店。",
        "perceive_enter": "縦長のビル、上階ほど小さなスナックやバー。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_南3": {
        "perceive_pass": "ペンシルビル群の中央、灯りの色が混ざる。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_南4": {
        "perceive_pass": "ペンシルビル群の東端付近、駅から少し離れる。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_北1": {
        "perceive_pass": "デッキ階下、駅前広場の裏側のペンシルビル。やや雑多、車の通りも近い。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_北2": {
        "perceive_pass": "ペンシルビル北側、2F〜にビジネスホテルの看板も混じる。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_北3": {
        "perceive_pass": "ペンシルビル群の北側中央、夜は派手なネオンも。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "港南ペンシルビル_北4": {
        "perceive_pass": "ペンシルビル群東端の北側、駅から離れる分静かな雑居ビル。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "食肉市場": {
        "perceive_pass": "もともと屠殺場、市場関係者のみ立ち入るエリア。町としては閉鎖的な場所だが、駅前にこの規模が残る。風向きによって独特の匂いが流れる。",
        "perceive_enter": None,  # 一般人は入れない
        "environment": {"indoor": False, "roof": False, "grade": "flat", "is_meat_market": True},
    },
    "芝浦中央公園": {
        "perceive_pass": "花々と芝生広場で構成された広いデッキ公園。遊具などは少なく、犬の散歩道のイメージ。",
        "perceive_enter": "一番奥にバラ園。近い将来、JR線路を越えるデッキで高輪ゲートウェイ駅と直結する予定。",
        "environment": {"indoor": False, "roof": False, "grade": "flat", "near_park": True},
    },
    "浄水場": {
        "perceive_pass": "一般人は入れない、中の様子も外からはあまり見えない。広い敷地。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": False, "grade": "flat"},
    },

    # === 高輪口エリア (西側) ===
    "ウィング高輪": {
        "perceive_pass": "京急1ビルの低層商業、地下から3階まで飲食・物販店舗が並ぶ。坂道状のカスケードを登っていくと、その先に水族館と映画館がある特徴的な構造。",
        "perceive_enter": "中はカスケード状で歩いていて楽しい構造、観光客と地元の家族連れが混ざる。",
        "environment": {"indoor": True, "roof": True, "grade": "gentle"},
    },
    "京急1ビル": {
        "perceive_pass": "ウィング高輪の上階のオフィス棟、京急本社系の入居。",
        "perceive_enter": "オフィスロビーは比較的静か。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "京急EXホテル": {
        "perceive_pass": "柘榴坂南側、ビジネスホテル系。足元に飲食店舗が数件。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "品川プリンスホテル": {
        "perceive_pass": "そびえる高層ホテル、大きな荷物を持った観光客が多数行き交う。横断歩道を渡ったところからすぐ目の前。",
        "perceive_enter": "ロビー入ると広いカフェ・レストラン、観光客で賑わう。水族館・映画館への動線もここから。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "品川水族館": {
        "perceive_pass": "プリンスホテル敷地の水族館、ファミリー層と観光客のターゲット。",
        "perceive_enter": "中は薄暗い水槽の連続、子ども連れ多い。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "新高輪プリンスホテル": {
        "perceive_pass": "柘榴坂上側、急こう配の坂道に沿って店舗が数件顔を出している特徴的な雰囲気。",
        "perceive_enter": "ロビーは格式高い雰囲気、宴会場や会議に使われる。",
        "environment": {"indoor": True, "roof": True, "grade": "steep"},
    },
    "高輪プリンスホテル": {
        "perceive_pass": "プリンスホテル群の北側、桜並木のさくら坂沿い。",
        "perceive_enter": "庭園(日本庭園)があり重要文化財の観音堂や鐘楼が見える。",
        "environment": {"indoor": True, "roof": True, "grade": "gentle"},
    },
    "プリンスさくらタワー": {
        "perceive_pass": "さくら坂沿い、観光客が結構行き来。車では先の通り抜けしづらいが、ホテル用のタクシーもよく通る。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "gentle"},
    },
    "国際会議場": {
        "perceive_pass": "プリンスホテル群の西端、国際会議用施設。",
        "perceive_enter": "イベント時のみ賑わう、それ以外は静か。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "税務署": {
        "perceive_pass": "柘榴坂上側、行政機関の硬い建物。",
        "perceive_enter": "中は典型的な行政施設、来客は限られる。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "高輪の森公園": {
        "perceive_pass": "工事現場、グランドプリンス高輪・新高輪に囲まれる場所。柘榴坂から細い道を進んでしか入れない、入口に気持ちばかりの広場。",
        "perceive_enter": "大半は勾配のある森。プレーパークなどで地域の子供たちに使われる。",
        "environment": {"indoor": False, "roof": False, "grade": "steep", "near_park": True},
    },
    "再開発工事中A地区": {
        "perceive_pass": "柘榴坂北側の大規模建築工事、仮囲いで広域が囲われ、工事用車両も頻繁に行き交う。品川駅前A地区の再開発計画。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": False, "grade": "flat", "near_construction": True},
    },
    "再開発工事中D地区": {
        "perceive_pass": "高輪口西側の中規模工事現場。品川駅前D地区の再開発計画。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": False, "grade": "flat", "near_construction": True},
    },
    "三菱関東閣": {
        "perceive_pass": "高輪台の旧岩崎邸、国の重要文化財。緑に包まれて静か。",
        "perceive_enter": "敷地内は迎賓館的な空間、一般立入は制限的。",
        "environment": {"indoor": True, "roof": True, "grade": "gentle"},
    },
    "高輪台住宅地": {
        "perceive_pass": "二本榎通りより西の閑静な住宅街。古くからの一軒家とマンションが混じる。坂が多く車通りはまばら。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": False, "grade": "gentle"},
    },
    "ガーデングレイス御殿山": {
        "perceive_pass": "御殿山の高級マンション・オフィス、緑が豊か。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "gentle"},
    },

    # === 駅東(港南)/北エリア ===
    "ave高輪": {
        "perceive_pass": "屋外階段が特徴的な新しいオフィスビル、国道15号沿い東側、駅ビル工事現場より北側。",
        "perceive_enter": "オフィスロビー、屋外階段でフロア間を移動できる開放感。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "セブンイレブン": {
        "perceive_pass": "駅前のチェーンコンビニ、いつも誰か入っている。",
        "perceive_enter": None,
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "プロント": {
        "perceive_pass": "国道15号沿いの雑居ビル足元のチェーンカフェ。",
        "perceive_enter": "コーヒーと簡単な食事、夜はバー営業もある。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "スタバ": {
        "perceive_pass": "東西自由通路のスタバ、新幹線改札前の2F。",
        "perceive_enter": "席数は多くないが回転は早い、Wi-Fi完備。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },
    "タクシー乗り場": {
        "perceive_pass": "港南口広場の階段裏側、列ができている時間帯多い。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": True, "grade": "flat"},
    },
    "バス乗り場": {
        "perceive_pass": "港南口、都心方面行きのバス。乗客は出張族・勤め人が中心。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": True, "grade": "flat"},
    },
    "アレア品川": {
        "perceive_pass": "デッキレベル(2F)にドトールとオフィスロビー、1Fと吹き抜けエスカレーターでつながり、1Fには飲食店がいくつか。",
        "perceive_enter": "1Fから外に出てまっすぐ進むとNTTビルに至るが、シーズンテラス側にはうまく抜けられない(東の道路側に抜ける必要)。",
        "environment": {"indoor": True, "roof": True, "grade": "flat"},
    },

    # === 国道・道路 ===
    "国道15号(第一京浜)": {
        "perceive_pass": "広幅員の幹線道路。京急鉄道工事と駅ビル工事が進行中、東側に店舗等はない無機質な感じ。横断歩道は信号待ちでごった返す。",
        "perceive_enter": None,
        "environment": {"indoor": False, "roof": False, "grade": "flat", "near_construction": True},
    },
}


def main() -> int:
    if not SRC.exists():
        print(f"[err] missing source: {SRC}", file=sys.stderr)
        return 1
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    places = data["places"]

    # Step 1: rename empty / duplicates
    for i, p in enumerate(places):
        # 居酒屋複製群
        if i in RENAME_BY_IDX:
            p["name"] = RENAME_BY_IDX[i]
        elif p.get("name") == "居酒屋":
            p["name"] = "港南ペンシルビル_南1"
        elif p.get("name") == "京急1ビル (複製)":
            p["name"] = "京急1ビル"
        elif not p.get("name"):
            if i in RENAME_EMPTY:
                p["name"] = RENAME_EMPTY[i]
        elif p.get("name") == "（建設中）":
            if i in RENAME_EMPTY:
                p["name"] = RENAME_EMPTY[i]

    # Step 2: 全 place の name 一意性チェック
    name_counts = {}
    for p in places:
        name_counts[p["name"]] = name_counts.get(p["name"], 0) + 1
    dups = [n for n, c in name_counts.items() if c > 1]
    if dups:
        print(f"[warn] duplicate names remain: {dups}", file=sys.stderr)

    # Step 3: 既存 spawn_point を解除 (新幹線改札前を新 spawn にする)
    for p in places:
        p["is_spawn_point"] = False

    # Step 4: 新規 places 追加
    for np in NEW_PLACES:
        places.append(np)

    # Step 5: perceive_pass / perceive_enter / environment 注入
    n_with_perceive = 0
    n_missing_perceive = 0
    for p in places:
        nm = p.get("name", "")
        per = PERCEIVE.get(nm)
        if per:
            p["perceive_pass"] = per.get("perceive_pass")
            p["perceive_enter"] = per.get("perceive_enter")
            p.setdefault("attributes", {})
            env = per.get("environment", {})
            if env:
                p["attributes"]["environment"] = env
            n_with_perceive += 1
        else:
            n_missing_perceive += 1
            print(f"[info] no perceive mapping for: {nm}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "metadata": {
            "name": "shinagawa_field_phaseB",
            "description": "Phase B: 品川駅周辺 400m圏 (FW シミュ用、perceive 属性付き)",
            "half_space_size": 80,
            "meters_per_cell": 5,
            "generated_from": "build_shinagawa_field.py + shinagawa_config.merged.yaml",
        },
        "places": places,
        "scene_3d": data.get("scene_3d") or {},
    }
    OUT.write_text(yaml.safe_dump(out_dict, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096), encoding="utf-8")
    print(f"[ok] wrote {OUT}")
    print(f"     places: {len(places)} ({n_with_perceive} with perceive, {n_missing_perceive} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
