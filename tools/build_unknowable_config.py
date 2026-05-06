"""Build config_unknowable_{variant}.yaml — 「知覚不可能な何か」教室シミュ。

問い (goal):
    「目の前に、知覚も言語化もできない『何か』がある。それが何かを説明しようとする。」

3 variant (同一field・同一問い・同一seed、ペルソナだけ変える):
    - elementary  : 小学生 (6-12歳)
    - high        : 高校生 (15-18歳)
    - singulabo   : 社会人 (シンギュラボのアジト相当)

フィールド: classroom_simulation_spec.md 準拠
    - 半空間 10 (cell=0.5m → 10m×10m 相当)
    - 5列×4行 = 20席 (各席を独立 place)
    - 教壇 / 教卓 / 黒板前 / 教室入口
    - 「■」(知覚不可能な何か) を 教壇上 (0, -6.5) に配置
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIMBASE = ROOT / "config_jr_disruption.yaml"

# ---------------------------------------------------------------------------
# Common: 問い・field・time
# ---------------------------------------------------------------------------
QUESTION_GOAL = (
    "授業前の教室で、自分なりに時間を過ごす。"
)
QUESTION_BG_TAIL = (
    "今は授業前の休み時間。教師はまだ来ていない。"
)

# 建築スケール: 1 cell = 1m, 教室 15m×15m×高さ3m, field 16m×16m (margin あり)
HALF_SPACE = 8  # field = 2*8+1 = 17m 相当 (1cell=1m)
ROOM_HALF_X = 7.5
ROOM_HALF_Y = 7.5
CEILING_M = 3.0


def _generate_seats() -> list:
    """5列×4行 = 20席を 15m×15m の教室にフィットさせて配置。
    席_行_列 (1_1=front-left, 4_5=back-right)。"""
    # 列間隔 3m, 行間隔 3m (前回の2倍)。room hs 7.5 にぎりぎり収まる。
    # cell グリッドが整数なので、すべて整数座標にして agent の grid 位置と
    # place 中心を一致させる (浮動小数だと丸めで row 1 だけずれる事故を防ぐ)
    cols_x = [-6, -3, 0, 3, 6]
    rows_y = [-3, 0, 3, 6]  # row1 が教壇寄り、後方に向けて 3m 間隔
    seats = []
    for r, y in enumerate(rows_y, start=1):
        for c, x in enumerate(cols_x, start=1):
            seats.append({
                "name": f"席_{r}_{c}",
                "type": "student_seat",
                "center_x": x, "center_y": y,
                "half_size_x": 0.45, "half_size_y": 0.35,  # 0.9m × 0.7m 机
                "capacity": 2,
                "social_likelihood": 0.5,
                "attributes": {"height_m": 0.7, "color": "#9a7a5a"},  # 木目調
            })
    return seats


PLACES = [
    # 教室全体 (15m×15m) — 視覚的には非表示 (床青塗りを消す指示)。
    # シム上は agent の現在地として機能し、social_likelihood も提供する。
    {"name": "教室", "type": "office_lobby",
     "center_x": 0, "center_y": 0,
     "half_size_x": ROOM_HALF_X, "half_size_y": ROOM_HALF_Y,
     "capacity": 50, "social_likelihood": 0.65,
     "attributes": {"height_m": CEILING_M, "invisible": True}},
    # 教壇 — 視覚的には非表示 (ユーザー指示)
    {"name": "教壇", "type": "office_lobby",
     "center_x": 0, "center_y": -5.5,
     "half_size_x": 3.5, "half_size_y": 1.0,
     "capacity": 5, "social_likelihood": 0.05,
     "attributes": {"height_m": 0.2, "invisible": True}},
    # 教卓 — 茶系
    {"name": "教卓", "type": "office_lobby",
     "center_x": 0, "center_y": -4.5,
     "half_size_x": 0.6, "half_size_y": 0.3,
     "capacity": 1, "social_likelihood": 0.0,
     "attributes": {"height_m": 0.75, "color": "#8a6a4a"}},
    # 教室入口 — 視覚的には非表示 (ユーザー指示)
    {"name": "教室入口", "type": "pedestrian_street",
     "center_x": 6.5, "center_y": 6.5,
     "half_size_x": 0.5, "half_size_y": 0.5,
     "capacity": 3, "social_likelihood": 0.1,
     "attributes": {"height_m": 0.05, "invisible": True}},
] + _generate_seats()

TIME_SCALE = {
    "step_duration_minutes": 1,  # 1step = 1分
    "start_time": "09:00",
    "patterns": [
        {
            "hours": [9, 10],
            "enter_per_step": 0.0,
            "exit_per_step": 0.0,
            "neighborhood_mood": (
                "授業前の休み時間。教室には20席が並ぶ。教師はまだ来ていない。"
                "教壇の方の空気が、なぜか少しだけ違って感じられる。"
            ),
            "default_goal": "目の前の『何か』が何かを、説明しようとする",
        },
    ],
}


def _seat_for_id(agent_id: int) -> str:
    """前列から順に席を埋める。id 0 → 席_1_1, id 4 → 席_1_5, id 5 → 席_2_1, ..."""
    cols_per_row = 5
    r = (agent_id // cols_per_row) + 1
    c = (agent_id % cols_per_row) + 1
    return f"席_{r}_{c}"


# ---------------------------------------------------------------------------
# Persona sets per variant
# ---------------------------------------------------------------------------
ELEMENTARY = [
    {"id": 0, "name": "あおい", "age": 6, "gender": "female", "occupation": "小学1年生",
     "background": "おとなしくてお母さんの影に隠れがち。絵本が好き。",
     "speech_style": "舌足らずな素直なことば",
     "talkativeness": 0.20, "phone_check_rate": 0.05},
    {"id": 1, "name": "はると", "age": 7, "gender": "male", "occupation": "小学2年生",
     "background": "外で走り回るのが大好き。じっとしてるのが苦手。",
     "speech_style": "元気いっぱいのタメ口",
     "talkativeness": 0.55, "phone_check_rate": 0.05},
    {"id": 2, "name": "さくら", "age": 8, "gender": "female", "occupation": "小学3年生",
     "background": "クラスで一番のおしゃべりで、誰とでもすぐ仲良くなる。",
     "speech_style": "明るくて速い口調",
     "talkativeness": 0.70, "phone_check_rate": 0.05},
    {"id": 3, "name": "ゆうま", "age": 9, "gender": "male", "occupation": "小学4年生",
     "background": "なんで？なんで？を連発する質問魔。図鑑が大好き。",
     "speech_style": "好奇心まる出しの口調",
     "talkativeness": 0.55, "phone_check_rate": 0.10},
    {"id": 4, "name": "ひなた", "age": 9, "gender": "female", "occupation": "小学4年生",
     "background": "絵を描くのが好きでマイペース。みんなとすぐに馴染まない。",
     "speech_style": "ぽつぽつしたつぶやき",
     "talkativeness": 0.25, "phone_check_rate": 0.08},
    {"id": 5, "name": "りく", "age": 10, "gender": "male", "occupation": "小学5年生",
     "background": "ロボットや恐竜に詳しい。図鑑を片手に話す。",
     "speech_style": "知識を披露したがるしっかり口調",
     "talkativeness": 0.50, "phone_check_rate": 0.10},
    {"id": 6, "name": "みお", "age": 10, "gender": "female", "occupation": "小学5年生",
     "background": "クラス委員でみんなをまとめたがる。優等生。",
     "speech_style": "丁寧でしっかりした子ども口調",
     "talkativeness": 0.50, "phone_check_rate": 0.10},
    {"id": 7, "name": "そうた", "age": 11, "gender": "male", "occupation": "小学6年生",
     "background": "考えてから動く慎重派。あんまり喋らない。",
     "speech_style": "短くて静かな口調",
     "talkativeness": 0.20, "phone_check_rate": 0.10},
    {"id": 8, "name": "ことね", "age": 11, "gender": "female", "occupation": "小学6年生",
     "background": "本の世界が好きな読書家。空想が得意。",
     "speech_style": "やわらかい、本のことばを混ぜる口調",
     "talkativeness": 0.35, "phone_check_rate": 0.15},
    {"id": 9, "name": "けいた", "age": 12, "gender": "male", "occupation": "小学6年生",
     "background": "現実離れした空想が得意。地球外生命体の話をよくする。",
     "speech_style": "ふわっとした想像口調",
     "talkativeness": 0.40, "phone_check_rate": 0.15},
    {"id": 10, "name": "あかり", "age": 12, "gender": "female", "occupation": "小学6年生",
     "background": "大人びた発言をするおませな子。クラスで一目置かれている。",
     "speech_style": "背伸びした少し丁寧な口調",
     "talkativeness": 0.45, "phone_check_rate": 0.20},
    {"id": 11, "name": "たくと", "age": 8, "gender": "male", "occupation": "小学3年生",
     "background": "怖がりだけど好奇心も強い。怖いものほど近づきたがる。",
     "speech_style": "おどおどしながらも興味津々の口調",
     "talkativeness": 0.30, "phone_check_rate": 0.05},
]

HIGH = [
    {"id": 0, "name": "佐倉ゆい", "age": 16, "gender": "female", "occupation": "高校1年",
     "background": "文学少女。村上春樹と中島敦が好き。",
     "speech_style": "丁寧でやや古風な高校生言葉",
     "talkativeness": 0.40, "phone_check_rate": 0.55},
    {"id": 1, "name": "藤崎涼介", "age": 17, "gender": "male", "occupation": "高校2年",
     "background": "アニメ・SF・ゲームのオタク。マニアック語彙が多い。",
     "speech_style": "早口でスラング混じりのオタク語",
     "talkativeness": 0.55, "phone_check_rate": 0.75},
    {"id": 2, "name": "中山健斗", "age": 18, "gender": "male", "occupation": "高校3年(理系受験生)",
     "background": "東大理系志望。物理が好きで論理で物事を整理したがる。",
     "speech_style": "落ち着いた論理的な敬語",
     "talkativeness": 0.30, "phone_check_rate": 0.50},
    {"id": 3, "name": "森野りお", "age": 16, "gender": "female", "occupation": "高校1年",
     "background": "陽キャ。SNSでフォロワー多め、空気を読むのが上手い。",
     "speech_style": "明るく軽快な今どきタメ口",
     "talkativeness": 0.65, "phone_check_rate": 0.85},
    {"id": 4, "name": "杉田陽", "age": 17, "gender": "male", "occupation": "高校2年(不登校気味)",
     "background": "学校に来たり来なかったり。哲学書を独学で読んでいる。",
     "speech_style": "ぼそぼそ、独白めいた口調",
     "talkativeness": 0.20, "phone_check_rate": 0.40},
    {"id": 5, "name": "桐谷瑞", "age": 18, "gender": "female", "occupation": "高校3年(美術系志望)",
     "background": "美大志望。色や形でしかわからない感覚を大事にする。",
     "speech_style": "感覚的・詩的な言い回し",
     "talkativeness": 0.40, "phone_check_rate": 0.60},
    {"id": 6, "name": "梅原翔", "age": 16, "gender": "male", "occupation": "高校1年(数学好き・無口)",
     "background": "数学オリンピックを目指している。あまり喋らない。",
     "speech_style": "極端に短く、必要なことだけ",
     "talkativeness": 0.10, "phone_check_rate": 0.30},
    {"id": 7, "name": "立花かほ", "age": 17, "gender": "female", "occupation": "高校2年(演劇部)",
     "background": "演劇部部長。感情を表現することに躊躇がない。",
     "speech_style": "豊かな抑揚と身振りを思わせる口調",
     "talkativeness": 0.65, "phone_check_rate": 0.65},
    {"id": 8, "name": "島本駿", "age": 15, "gender": "male", "occupation": "高校1年(体育会系)",
     "background": "サッカー部のフォワード。直感で動く。あまり考えない。",
     "speech_style": "ガサツでテンション高いタメ口",
     "talkativeness": 0.50, "phone_check_rate": 0.55},
    {"id": 9, "name": "西田大樹", "age": 17, "gender": "male", "occupation": "高校2年(プログラマ志望)",
     "background": "中学からプログラミング独学。Stable Diffusionに詳しい。",
     "speech_style": "技術用語混じりの淡々とした口調",
     "talkativeness": 0.35, "phone_check_rate": 0.85},
    {"id": 10, "name": "三宅あいり", "age": 16, "gender": "female", "occupation": "高校1年(心理学志望)",
     "background": "心理学に興味があり、人の心の動きを観察するのが好き。",
     "speech_style": "観察的で穏やかな口調",
     "talkativeness": 0.45, "phone_check_rate": 0.65},
    {"id": 11, "name": "原田美月", "age": 18, "gender": "female", "occupation": "高校3年(ジャーナリスト志望)",
     "background": "新聞部部長。批判的思考と一次情報重視。",
     "speech_style": "鋭く問い返す論理的な口調",
     "talkativeness": 0.55, "phone_check_rate": 0.75},
    # ===== ここから追加 8人 (合計20人 = 中学校1クラス相当) =====
    {"id": 12, "name": "東 れな", "age": 16, "gender": "female", "occupation": "高校1年(ヤンキー系)",
     "background": "学校はあまり好きじゃない。ファッションと友達優先。",
     "speech_style": "ぶっきらぼうなギャル語",
     "talkativeness": 0.45, "phone_check_rate": 0.90},
    {"id": 13, "name": "新堂タケル", "age": 17, "gender": "male", "occupation": "高校2年(帰宅部・映画オタク)",
     "background": "部活はせず、毎日映画を見て帰る。タランティーノとアリ・アスターが好き。",
     "speech_style": "映画の比喩を多用する淡々口調",
     "talkativeness": 0.40, "phone_check_rate": 0.65},
    {"id": 14, "name": "丸山修平", "age": 16, "gender": "male", "occupation": "高校1年(鉄道オタク)",
     "background": "鉄道時刻表を暗記している。話せばずっと電車の話。",
     "speech_style": "知識マシンガントーク",
     "talkativeness": 0.55, "phone_check_rate": 0.50},
    {"id": 15, "name": "小柳エリカ", "age": 18, "gender": "female", "occupation": "高校3年(音楽科・ピアノ)",
     "background": "音大志望。耳が異常に良い。聴感で世界を捉える。",
     "speech_style": "音楽用語混じりの感覚的な口調",
     "talkativeness": 0.40, "phone_check_rate": 0.55},
    {"id": 16, "name": "北野アヤメ", "age": 17, "gender": "female", "occupation": "高校2年(軽音部・ボーカル)",
     "background": "バンドのフロントマン。即興と感情表現が得意。",
     "speech_style": "勢いのある楽屋トーク",
     "talkativeness": 0.65, "phone_check_rate": 0.80},
    {"id": 17, "name": "ジョナサン・リー", "age": 16, "gender": "male", "occupation": "高校1年(留学生・米国)",
     "background": "ロサンゼルス出身。日本語は流暢だが時々英単語が混ざる。",
     "speech_style": "英単語混じりの素直な日本語",
     "talkativeness": 0.55, "phone_check_rate": 0.85},
    {"id": 18, "name": "栗山リョウ", "age": 18, "gender": "male", "occupation": "高校3年(不良グループ)",
     "background": "校則無視で先生と衝突しがち。本心は優しいタイプ。",
     "speech_style": "尖った荒いタメ口、本音は時々出る",
     "talkativeness": 0.30, "phone_check_rate": 0.60},
    {"id": 19, "name": "藤村美和子", "age": 17, "gender": "female", "occupation": "高校2年(生徒会長)",
     "background": "全体を仕切るリーダー。場のバランスを考える癖がある。",
     "speech_style": "進行役らしい丁寧で公正な口調",
     "talkativeness": 0.55, "phone_check_rate": 0.70},
]

SINGULABO = [
    {"id": 0, "name": "甘利 諒", "age": 32, "gender": "male", "occupation": "AIフリーランス・セミナー講師",
     "background": "茨城在住。LLM活用セミナーを主宰。技術と教育を行き来している。",
     "speech_style": "わかりやすく噛み砕いた説明口調",
     "talkativeness": 0.55, "phone_check_rate": 0.85},
    {"id": 1, "name": "清水 凛", "age": 41, "gender": "female", "occupation": "AI研究者(認知科学寄り)",
     "background": "大学院でメタ認知の研究をしていた。論文癖が抜けない。",
     "speech_style": "論文調の慎重な敬語",
     "talkativeness": 0.40, "phone_check_rate": 0.70},
    {"id": 2, "name": "戸島 蛸介", "age": 38, "gender": "male", "occupation": "アートデザイナー・教育系",
     "background": "デジハリ出身。アートと教育を架橋する活動をしている。",
     "speech_style": "感覚と論理を行ったり来たりする饒舌口調",
     "talkativeness": 0.65, "phone_check_rate": 0.75},
    {"id": 3, "name": "西園寺 葉子", "age": 45, "gender": "female", "occupation": "元大学教授(哲学)",
     "background": "現象学とウィトゲンシュタインを主に。今はフリーで執筆。",
     "speech_style": "重厚な哲学者の口調",
     "talkativeness": 0.45, "phone_check_rate": 0.50},
    {"id": 4, "name": "新村 陸", "age": 30, "gender": "male", "occupation": "個人Webエンジニア",
     "background": "ウェルカムサポート3期生。Web系を一人でやっている。",
     "speech_style": "気さくでフラットな技術者口調",
     "talkativeness": 0.45, "phone_check_rate": 0.85},
    {"id": 5, "name": "法堂 銀河", "age": 36, "gender": "female", "occupation": "宇宙データエンジニア",
     "background": "元天文台。スペースデータの会社で働いている。宇宙視点を持ち込みたがる。",
     "speech_style": "スケール感のある観察口調",
     "talkativeness": 0.50, "phone_check_rate": 0.70},
    {"id": 6, "name": "大園 大輔", "age": 44, "gender": "male", "occupation": "教育コンサル(ノンエンジニア)",
     "background": "妻が住宅系。プログラミングはローカルで少し触る程度。",
     "speech_style": "人当たりの良い柔らかい敬語",
     "talkativeness": 0.55, "phone_check_rate": 0.55},
    {"id": 7, "name": "南条 千夏", "age": 28, "gender": "female", "occupation": "デジタル広告代理店UXリード",
     "background": "品川の広告代理店勤務。LLMを企画に使い倒している。",
     "speech_style": "プロジェクト管理者風の的確な口調",
     "talkativeness": 0.60, "phone_check_rate": 0.85},
    {"id": 8, "name": "黒木 慎一郎", "age": 39, "gender": "male", "occupation": "元銀行員・哲学系note執筆者",
     "background": "数年前に銀行を辞めて執筆業へ。社会哲学が専門。",
     "speech_style": "皮肉とユーモアを交えた論評口調",
     "talkativeness": 0.50, "phone_check_rate": 0.65},
    {"id": 9, "name": "海老ねこ 翠", "age": 35, "gender": "female", "occupation": "教育×思考シミュ研究者",
     "background": "教育と思考の交差を研究。7月から半年インドへ。",
     "speech_style": "穏やかで本質を突く問い返しが多い口調",
     "talkativeness": 0.55, "phone_check_rate": 0.75},
    {"id": 10, "name": "兵藤 浩二", "age": 47, "gender": "male", "occupation": "ハッカソン運営・配信",
     "background": "デジハリ運営側。シミュレーションを長く見てきた。",
     "speech_style": "落ち着いた進行役の敬語",
     "talkativeness": 0.45, "phone_check_rate": 0.80},
    {"id": 11, "name": "片山 裕真", "age": 42, "gender": "male", "occupation": "広告代理店デジタル部門",
     "background": "品川のポテンシャルに期待している。記憶力に難ありらしいが鋭い。",
     "speech_style": "断片的だが核心を突く口調",
     "talkativeness": 0.40, "phone_check_rate": 0.75},
]


VARIANTS = {
    "elementary": ELEMENTARY,
    "high": HIGH,
    "singulabo": SINGULABO,
}


def _bake_personas(personas_template: list, num: int, mutual_rel: float = 0.55) -> list:
    """Apply common (question goal/background tail/initial_place=自席/relationships) and trim to num.

    会話絞り緩和のため:
      - 全員相互に initial_relationships = mutual_rel (同級生・同僚前提)
      - talkativeness を底上げ (×1.3, max 0.90)
    initial_place は agent_id (0..) に対応する席 (席_{row}_{col})。
    """
    out = []
    for p in personas_template[:num]:
        p2 = copy.deepcopy(p)
        p2["background"] = p2["background"] + " " + QUESTION_BG_TAIL
        p2["current_goal"] = QUESTION_GOAL
        # talkativeness 底上げ
        t = float(p2.get("talkativeness", 0.4))
        p2["talkativeness"] = round(min(0.90, t * 1.3 + 0.05), 2)
        out.append(p2)
    # ensure ids 連番
    for i, p in enumerate(out):
        p["id"] = i
        p["initial_place"] = _seat_for_id(i)  # 各生徒に席を割り当て
    # 全員相互に知り合い (mutual_rel)
    for p in out:
        p["initial_relationships"] = {
            other["id"]: mutual_rel for other in out if other["id"] != p["id"]
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not SIMBASE.exists():
        print(f"[err] missing: {SIMBASE}", file=sys.stderr)
        return 1
    base = yaml.safe_load(SIMBASE.read_text(encoding="utf-8"))

    if args.mode == "smoke":
        num_agents, duration, parallel_workers = 10, 15, 8
    else:
        num_agents, duration, parallel_workers = 20, 30, 12

    personas = _bake_personas(VARIANTS[args.variant], num_agents)

    sim = copy.deepcopy(base.get("simulation", {}))
    sim["duration"] = duration
    sim["half_space_size"] = HALF_SPACE
    sim["half_place_size"] = 5
    sim["seed"] = args.seed
    sim["time_scale"] = TIME_SCALE

    agents = copy.deepcopy(base.get("agents", {}))
    agents["num_agents"] = num_agents
    agents["max_agents"] = num_agents
    agents["communication_radius"] = 25  # 教室狭いので全員にだいたい届く
    agents["parallel_workers"] = parallel_workers
    agents["personas"] = personas
    # 教室狭い → ゆっくり動く
    agents["movement_speed"] = {"base_cells_per_step": 2, "variance": 1}
    # 会話絞り緩和：皆ちゃんと考えてちゃんと喋る
    agents["skip_probability"] = 0.05
    # 会話履歴・コンテキストを多めに保持して、議論が積み重なるように
    agents["message_history_limit"] = 20
    agents["message_context_size"] = 6

    out_path = ROOT / f"config_unknowable_{args.variant}.yaml"

    out = {
        # 建築スケールメタデータ (simulation.py が viewer に pass-through する)
        "metadata": {
            "name": f"classroom_unknowable_{args.variant}",
            "description": "知覚不可能な何か / 教室シナリオ — 15m×15m×3m 想定",
            "scenario_kind": "classroom",
            "meters_per_cell": 1,        # 都市スケールの 5m → 建築の 1m
            "grid_step_m": 1,            # ビューア側で 1m grid を描く
            "hide_place_labels": True,   # 教壇/席名等のラベル非表示
            "ceiling_height_m": CEILING_M,
        },
        "simulation": sim,
        "agents": agents,
        "places": PLACES,
        "llm": copy.deepcopy(base.get("llm", {})),
        "fires": [],
        "events": [],  # 外的イベントなし — 問いは current_goal に内在
        "events_keywords": [],
        "visualization": {
            "save_frames": True,
            "frame_interval": 1,
            "focus_agent_id": 0,
            "run_name": f"unknowable_{args.variant}",
        },
        "logging": {"level": "INFO"},
    }

    header = (
        "# ================================================================\n"
        f"# 知覚不可能な何か シミュ ({args.variant})\n"
        "# ================================================================\n"
        "# Built by tools/build_unknowable_config.py\n"
        "#\n"
        "# 問い: 「目の前に、知覚も言語化もできない何かがある。これが何か？」\n"
        f"# Variant: {args.variant} (elementary=小学生 / high=高校生 / singulabo=社会人)\n"
        f"# num_agents={num_agents} / duration={duration} steps\n"
        "# 同一field・同一seed・同一問い、ペルソナだけ変える A/B/C 設計\n"
        "# ================================================================\n\n"
    )

    out_path.write_text(
        header + yaml.safe_dump(out, allow_unicode=True, sort_keys=False,
                                default_flow_style=False, width=4096),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_path}")
    print(f"     variant         : {args.variant}")
    print(f"     mode            : {args.mode}")
    print(f"     num_agents      : {num_agents}")
    print(f"     duration        : {duration} steps")
    print(f"     parallel_workers: {parallel_workers}")
    print(f"     places          : {len(PLACES)}")
    print(f"     question        : {QUESTION_GOAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
