"""Build config_shinagawa_last_shinkansen.yaml — v3 指示書 (新幹線終電後シナリオ) 用。

Combines:
  - scene/places/scene_3d from shinagawa_config.merged.yaml
  - 30 personas + time_scale + events + keywords embedded below
  - llm/visualization from config_jr_disruption.yaml (gemini preset etc.)

Output: project_root/config_shinagawa_last_shinkansen.yaml
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MERGED = Path(r"D:\ユーザー\ダウンロード\shinagawa_config.merged.yaml")
SIMBASE = ROOT / "configs" / "config_jr_disruption.yaml"
OUT = ROOT / "configs" / "config_shinagawa_last_shinkansen.yaml"

SPAWN_PLACE = "品川駅 東西自由通路"

# ---------------------------------------------------------------------------
# personas: straight transcription of v3 指示書
# ---------------------------------------------------------------------------
PERSONAS = [
    # ===== 大阪・名古屋方面 出張者 (7人) =====
    {
        "id": 0, "name": "中村達也", "age": 42, "gender": "male",
        "occupation": "メーカー営業部長", "address": "大阪府堺市",
        "background": "明日9時に大阪本社でプレゼン。部下の田中さやかと出張。冷静だが内心焦っている。",
        "speech_style": "落ち着いた丁寧語。部下には兄貴分",
        "current_goal": "明日の朝イチ打ち合わせに絶対間に合わせたい。部下のことも気になる。",
        "talkativeness": 0.30, "phone_check_rate": 0.88,
        "initial_relationships": {1: 0.65},
        "social_identities": [{
            "group_name": "大阪出張メーカー営業チーム", "scale": "micro",
            "base_salience": 0.85, "norms": "仕事仲間・責任感",
            "member_ids": [1], "in_group_bias": 0.7,
        }],
    },
    {
        "id": 1, "name": "田中さやか", "age": 28, "gender": "female",
        "occupation": "メーカー営業", "address": "大阪市北区",
        "background": "部長と一緒に出張。明日の資料がまだ完成していない。",
        "speech_style": "ハキハキ敬語、焦ると早口",
        "current_goal": "今夜のうちに資料を仕上げたい。宿のことも考えないといけない。",
        "talkativeness": 0.55, "phone_check_rate": 0.95,
        "initial_relationships": {0: 0.65},
        "social_identities": [{
            "group_name": "大阪出張メーカー営業チーム", "scale": "micro",
            "base_salience": 0.85, "norms": "仕事仲間・責任感",
            "member_ids": [0], "in_group_bias": 0.7,
        }],
    },
    {
        "id": 2, "name": "松田浩二", "age": 55, "gender": "male",
        "occupation": "商社 上席部長", "address": "名古屋市",
        "background": "単独出張。定年まで数年、リスクを取らない慎重派。家族が心配している。",
        "speech_style": "重厚な敬語、ゆっくり",
        "current_goal": "家族を不安にさせたくない。自分の身は自分で守る。",
        "talkativeness": 0.25, "phone_check_rate": 0.70,
    },
    {
        "id": 3, "name": "佐藤美穂", "age": 34, "gender": "female",
        "occupation": "IT企業 PM", "address": "京都市",
        "background": "単独出張。明日午前の会議はリモートでも代替できる。どちらかというと適応型。",
        "speech_style": "淡々とした丁寧語",
        "current_goal": "焦っても仕方ない。一番賢い選択をしたい。",
        "talkativeness": 0.50, "phone_check_rate": 0.92,
    },
    {
        "id": 4, "name": "林雄介", "age": 29, "gender": "male",
        "occupation": "スタートアップ エンジニア", "address": "大阪市福島区",
        "background": "出費を最小限にしたいが明日の午後に大阪で打ち合わせがある。",
        "speech_style": "ため口混じりのカジュアル",
        "current_goal": "できるだけお金を使いたくない。でも打ち合わせには間に合わせたい。",
        "talkativeness": 0.40, "phone_check_rate": 0.88,
    },
    {
        "id": 5, "name": "山岡りえ", "age": 38, "gender": "female",
        "occupation": "コンサルタント", "address": "神戸市",
        "background": "出張トラブルには慣れている。体力よりも快適さを優先するタイプ。",
        "speech_style": "さばさばした丁寧語",
        "current_goal": "余計なストレスをかけずに今夜を乗り越えたい。",
        "talkativeness": 0.45, "phone_check_rate": 0.95,
    },
    {
        "id": 6, "name": "渡辺健一", "age": 47, "gender": "male",
        "occupation": "建設会社 課長", "address": "名古屋市守山区",
        "background": "翌日は有給を取ってあるので実は急ぎではない。",
        "speech_style": "明るい世間話モード",
        "current_goal": "せっかく東京に居る。楽しめるならそれでいい。",
        "talkativeness": 0.55, "phone_check_rate": 0.65,
    },
    # ===== 地方から来た観光・旅行者 (5人) =====
    {
        "id": 7, "name": "大西幸子", "age": 62, "gender": "female",
        "occupation": "パート", "address": "富山市",
        "background": "娘と二人旅。スマホ操作が苦手で娘に頼りっきり。",
        "speech_style": "素朴な方言混じり",
        "current_goal": "娘のそばにいれば大丈夫、でも早く休みたい。",
        "talkativeness": 0.35, "phone_check_rate": 0.15,
        "initial_relationships": {8: 0.98},
        "social_identities": [{
            "group_name": "大西母娘", "scale": "micro",
            "base_salience": 0.95, "norms": "家族・寄り添う",
            "member_ids": [8], "in_group_bias": 0.98,
        }],
    },
    {
        "id": 8, "name": "大西美鈴", "age": 35, "gender": "female",
        "occupation": "会社員", "address": "富山市",
        "background": "母親と二人旅。母を安心させることが最優先。",
        "speech_style": "母には優しく、外には丁寧",
        "current_goal": "お母さんを心配させたくない。自分がなんとかしないといけない。",
        "talkativeness": 0.55, "phone_check_rate": 0.90,
        "initial_relationships": {7: 0.98},
        "social_identities": [{
            "group_name": "大西母娘", "scale": "micro",
            "base_salience": 0.95, "norms": "家族・守る",
            "member_ids": [7], "in_group_bias": 0.98,
        }],
    },
    {
        "id": 9, "name": "川口翔太", "age": 23, "gender": "male",
        "occupation": "大学院生", "address": "福岡市",
        "background": "東京観光の帰り。お金はあまりない。飛行機の選択肢が頭にある。",
        "speech_style": "若者らしいカジュアル",
        "current_goal": "なるべく安く今夜をしのいで明日福岡に帰りたい。",
        "talkativeness": 0.45, "phone_check_rate": 0.88,
    },
    {
        "id": 10, "name": "斎藤英雄", "age": 70, "gender": "male",
        "occupation": "退職者", "address": "仙台市",
        "background": "孫の顔を見に東京へ来た帰り。スマホを使いこなせない。",
        "speech_style": "ゆっくりした丁寧語",
        "current_goal": "仙台への電車があると思っている。早く帰りたい。",
        "talkativeness": 0.40, "phone_check_rate": 0.12,
    },
    {
        "id": 11, "name": "北村彩乃", "age": 27, "gender": "female",
        "occupation": "アパレル販売員", "address": "札幌市",
        "background": "東京出張の帰り。飛行機なので新幹線は関係ないが、この混雑の意味がわからない。",
        "speech_style": "明るくフレンドリー",
        "current_goal": "羽田行きの京急に乗らないといけない。何が起きているのかわからない。",
        "talkativeness": 0.60, "phone_check_rate": 0.92,
    },
    # ===== 東京在住の帰宅組 (8人) =====
    {
        "id": 12, "name": "石川慶太", "age": 31, "gender": "male",
        "occupation": "広告代理店", "address": "目黒区",
        "background": "残業後の帰宅。疲れているのでさっさと帰りたい。",
        "speech_style": "疲れた淡々",
        "current_goal": "早く家に帰って風呂に入りたい。",
        "talkativeness": 0.35, "phone_check_rate": 0.20,
    },
    {
        "id": 13, "name": "加藤真理", "age": 44, "gender": "female",
        "occupation": "中学校教師", "address": "大田区",
        "background": "研修帰り。明日も早い。",
        "speech_style": "落ち着いた敬語",
        "current_goal": "明日も朝から仕事。できるだけ早く帰りたい。",
        "talkativeness": 0.30, "phone_check_rate": 0.30,
    },
    {
        "id": 14, "name": "村田隆司", "age": 38, "gender": "male",
        "occupation": "外資系銀行員", "address": "港区",
        "background": "会食帰り。お金は気にしない。",
        "speech_style": "スマートな敬語",
        "current_goal": "楽に帰れればそれでいい。",
        "talkativeness": 0.20, "phone_check_rate": 0.50,
    },
    {
        "id": 15, "name": "木村奈緒", "age": 26, "gender": "female",
        "occupation": "看護師(夜勤明け)", "address": "品川区",
        "background": "夜勤明け。かなり疲れている。",
        "speech_style": "ぼそぼそ",
        "current_goal": "とにかく早くベッドに横になりたい。",
        "talkativeness": 0.25, "phone_check_rate": 0.55,
    },
    {
        "id": 16, "name": "藤井淳", "age": 52, "gender": "male",
        "occupation": "不動産会社社長", "address": "横浜市",
        "background": "取引先との会食帰り。気分がいい。",
        "speech_style": "貫禄のある余裕口調",
        "current_goal": "今夜は楽しかった。ゆっくり横浜に帰ろう。",
        "talkativeness": 0.50, "phone_check_rate": 0.65,
    },
    {
        "id": 17, "name": "長谷川桜", "age": 21, "gender": "female",
        "occupation": "大学生", "address": "世田谷区",
        "background": "友人と夕食の帰り。まだ話し足りない気分。",
        "speech_style": "楽しげなタメ口",
        "current_goal": "拓ともう少し話したいけど、終電も気になる。",
        "talkativeness": 0.65, "phone_check_rate": 0.90,
        "initial_relationships": {18: 0.88},
        "social_identities": [{
            "group_name": "大学生ペア(桜・拓)", "scale": "micro",
            "base_salience": 0.8, "norms": "親しい友人",
            "member_ids": [18], "in_group_bias": 0.88,
        }],
    },
    {
        "id": 18, "name": "橋本拓", "age": 22, "gender": "male",
        "occupation": "大学生", "address": "世田谷区",
        "background": "長谷川と一緒。明日の課題が気になっている。",
        "speech_style": "優しいタメ口",
        "current_goal": "課題が終わってない。早めに帰りたいが、桜とのこの時間も大事にしたい。",
        "talkativeness": 0.60, "phone_check_rate": 0.88,
        "initial_relationships": {17: 0.88},
        "social_identities": [{
            "group_name": "大学生ペア(桜・拓)", "scale": "micro",
            "base_salience": 0.8, "norms": "親しい友人",
            "member_ids": [17], "in_group_bias": 0.88,
        }],
    },
    {
        "id": 19, "name": "岡田仁", "age": 67, "gender": "male",
        "occupation": "定年退職者", "address": "川崎市",
        "background": "息子夫婦と夕食の帰り。急ぐ理由がない。",
        "speech_style": "ゆったりした丁寧語",
        "current_goal": "夜の散歩でも楽しみながらゆっくり帰ろう。",
        "talkativeness": 0.55, "phone_check_rate": 0.20,
    },
    # ===== ビジネスホテル宿泊者 (4人) =====
    {
        "id": 20, "name": "荒木誠", "age": 45, "gender": "male",
        "occupation": "商社 部長", "address": "福岡市博多区",
        "background": "既に港南口のホテルにチェックイン済み。明日の会議の準備が気になる。",
        "speech_style": "落ち着いた丁寧語",
        "current_goal": "明日の朝一に備えて今夜は早めに休みたい。少し外に出たかった。",
        "talkativeness": 0.30, "phone_check_rate": 0.75,
    },
    {
        "id": 21, "name": "池田麻子", "age": 33, "gender": "female",
        "occupation": "製薬会社MR", "address": "広島市",
        "background": "既にチェックイン済み。夕食を食べ損ねた。",
        "speech_style": "ハキハキ",
        "current_goal": "何かお腹に入れてから部屋に戻りたい。",
        "talkativeness": 0.45, "phone_check_rate": 0.85,
    },
    {
        "id": 22, "name": "田辺修", "age": 39, "gender": "male",
        "occupation": "製造業 係長", "address": "高松市",
        "background": "既にチェックイン済み。散歩に出てきた。",
        "speech_style": "静かな敬語",
        "current_goal": "ぶらっと外の空気を吸いたかっただけ。",
        "talkativeness": 0.35, "phone_check_rate": 0.60,
    },
    {
        "id": 23, "name": "村上千恵", "age": 50, "gender": "female",
        "occupation": "市役所職員", "address": "新潟市",
        "background": "既にチェックイン済み。夜の品川を見てみたかった。",
        "speech_style": "穏やかな丁寧語",
        "current_goal": "少し外を見たら部屋に戻るつもりだった。",
        "talkativeness": 0.50, "phone_check_rate": 0.40,
    },
    # ===== 飲み会帰り・終電気にしてる組 (4人) =====
    {
        "id": 24, "name": "吉田健", "age": 35, "gender": "male",
        "occupation": "出版社", "address": "渋谷区",
        "background": "同僚と飲み会の帰り。まだ飲み足りない。",
        "speech_style": "酔っぱらいのフランク",
        "current_goal": "楽しい夜を続けたいが、終電は逃したくない。",
        "talkativeness": 0.70, "phone_check_rate": 0.70,
        "initial_relationships": {25: 0.72},
        "social_identities": [{
            "group_name": "出版社飲み会ペア", "scale": "micro",
            "base_salience": 0.7, "norms": "同僚ノリ",
            "member_ids": [25], "in_group_bias": 0.72,
        }],
    },
    {
        "id": 25, "name": "福田明美", "age": 32, "gender": "female",
        "occupation": "出版社", "address": "新宿区",
        "background": "同僚と一緒。明日は普通に仕事がある。",
        "speech_style": "赤ら顔のタメ口混じり",
        "current_goal": "もう少し飲みたい気持ちと、ちゃんと帰らなきゃという気持ちが半々。",
        "talkativeness": 0.65, "phone_check_rate": 0.85,
        "initial_relationships": {24: 0.72},
        "social_identities": [{
            "group_name": "出版社飲み会ペア", "scale": "micro",
            "base_salience": 0.7, "norms": "同僚ノリ",
            "member_ids": [24], "in_group_bias": 0.72,
        }],
    },
    {
        "id": 26, "name": "三浦浩", "age": 41, "gender": "male",
        "occupation": "SE", "address": "江東区",
        "background": "一人で飲み会帰り。この混雑が何事かと思っている。",
        "speech_style": "ぼそっと",
        "current_goal": "終電には十分間に合う。何かあったのか気になる。",
        "talkativeness": 0.25, "phone_check_rate": 0.75,
    },
    {
        "id": 27, "name": "平田亜紀", "age": 29, "gender": "female",
        "occupation": "事務職", "address": "墨田区",
        "background": "終電まで2時間はある。夜はまだ長い。",
        "speech_style": "明るいタメ口",
        "current_goal": "せっかく品川まで来たし、もう一軒くらいいいかなという気分。",
        "talkativeness": 0.45, "phone_check_rate": 0.88,
    },
    # ===== 深夜に品川にいる理由ある人 (2人) =====
    {
        "id": 28, "name": "原田哲也", "age": 36, "gender": "male",
        "occupation": "コンビニ夜勤", "address": "品川区",
        "background": "仕事に向かう途中。この混雑には慣れている。",
        "speech_style": "素っ気ない敬語",
        "current_goal": "遅刻せずに出勤する。",
        "talkativeness": 0.20, "phone_check_rate": 0.30,
    },
    {
        "id": 29, "name": "杉山洋子", "age": 48, "gender": "female",
        "occupation": "深夜清掃スタッフ", "address": "港区",
        "background": "仕事の行き帰り。自分のペースを崩さない。",
        "speech_style": "ひかえめ",
        "current_goal": "現場に時間通り着けばいい。",
        "talkativeness": 0.15, "phone_check_rate": 0.25,
    },
]

# All personas spawn at the same walkable place (v3 design principle).
for p in PERSONAS:
    p["initial_place"] = SPAWN_PLACE


EVENTS = [
    {
        "name": "shinkansen_last_departure",
        "type": "last_train",
        "start_step": 0,
        "end_step": 90,
        "broadcast_once": True,
        "affected_place": SPAWN_PLACE,
        "description": (
            "東海道新幹線 のぞみ 最終 新大阪行きが 21:33 に発車した。"
            "以降、今日中に新大阪・京都・名古屋方面へは新幹線では行けない。"
            "次の始発は翌 6:00 台。品川駅構内アナウンスで案内中。"
        ),
        "effects": {
            "notify_agents_within_radius": 999,
        },
        "salience_boost_targets": [
            "大阪出張メーカー営業チーム",
        ],
    },
]


EVENTS_KEYWORDS = [
    '新幹線', '終電', '最終', 'のぞみ', '新大阪', '大阪', '名古屋', '京都',
    '乗れなかった', '乗り遅れ', '逃した', '始発', 'ホテル', '泊まる', '一泊',
    '深夜バス', 'バス', 'タクシー', '羽田', '飛行機',
    '宿', '予約', '明日の朝', 'チェックイン',
    '帰れない', '帰れなくなった', 'どうしよう',
]


TIME_SCALE = {
    "step_duration_minutes": 1,
    "start_time": "21:30",
    "patterns": [
        {
            "hours": [21, 22],
            "enter_per_step": 0.08,
            "exit_per_step": 0.04,
            "neighborhood_mood": (
                "新幹線の終電が出たばかりの品川駅東西自由通路。"
                "帰れなくなった出張者・無関係な帰宅客・夜勤者が入り混じる。"
            ),
            "default_goal": "終電後の自分の状況を整理し、今夜どう動くか決める",
        },
        {
            "hours": [22, 23, 0, 1],
            "enter_per_step": 0.03,
            "exit_per_step": 0.06,
            "neighborhood_mood": (
                "在来線の終電も徐々に消え、タクシー待ちや深夜バス乗り場が目立ち始める。"
            ),
            "default_goal": "今夜の宿泊・移動手段を確定させる",
        },
    ],
}


def load_yaml_with_shorthand(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    sh = re.compile(r"^(\s*-\s)(.+?\s;\s.+)$", re.MULTILINE)
    def norm(m):
        prefix, body = m.group(1), m.group(2).strip()
        fields = [f.strip() for f in body.split(";") if f.strip()]
        return f"{prefix}{{ {', '.join(fields)} }}"
    return yaml.safe_load(sh.sub(norm, raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main"], default="smoke",
                    help="smoke=20x15, main=30x90")
    args = ap.parse_args()

    if not MERGED.exists():
        print(f"[err] missing: {MERGED}", file=sys.stderr)
        return 1
    if not SIMBASE.exists():
        print(f"[err] missing: {SIMBASE}", file=sys.stderr)
        return 1

    scene = load_yaml_with_shorthand(MERGED)
    base = load_yaml_with_shorthand(SIMBASE)

    num_agents, duration = (20, 15) if args.mode == "smoke" else (30, 90)

    sim = copy.deepcopy(base.get("simulation", {}))
    sim["duration"] = duration
    sim["half_space_size"] = scene.get("metadata", {}).get("half_space_size", 80)
    sim["time_scale"] = TIME_SCALE

    agents = copy.deepcopy(base.get("agents", {}))
    agents["num_agents"] = num_agents
    agents["max_agents"] = 30
    agents["personas"] = PERSONAS

    out = {
        "metadata": scene.get("metadata", {}),
        "proposed_type_extensions": scene.get("proposed_type_extensions", []),
        "simulation": sim,
        "agents": agents,
        "places": scene.get("places", []),
        "scene_3d": scene.get("scene_3d", {}),
        "llm": copy.deepcopy(base.get("llm", {})),
        "fires": [],
        "events": EVENTS,
        "events_keywords": EVENTS_KEYWORDS,
        "visualization": {
            "save_frames": True,
            "frame_interval": 1,
            "focus_agent_id": 0,
        },
        "logging": {"level": "INFO"},
    }

    header = (
        "# ================================================================\n"
        "# 品川 新幹線終電後シナリオ v3 (last_shinkansen)\n"
        "# ================================================================\n"
        "# Built by tools/build_last_shinkansen_config.py\n"
        "# Source指示書: 2026-04-20_shinagawa_last_shinkansen_v3.md\n"
        "#\n"
        "# 主な設定:\n"
        "#   - 全員 initial_place = 品川駅 東西自由通路\n"
        "#   - start_time 21:30 / step=1min / duration 15(smoke) or 90(本線)\n"
        "#   - num_agents 20(smoke) / 30(本線)\n"
        "#   - event: last_train (transit_disruption と同じ awareness 配線)\n"
        "#   - events_keywords: 終電シナリオ仕様 (新幹線/ホテル/深夜バス等)\n"
        "# ================================================================\n\n"
    )

    OUT.write_text(
        header + yaml.safe_dump(
            out,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=4096,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {OUT}")
    print(f"     mode             : {args.mode}")
    print(f"     personas in pool : {len(PERSONAS)}")
    print(f"     num_agents       : {agents['num_agents']}")
    print(f"     duration         : {sim['duration']} steps")
    print(f"     places           : {len(out['places'])}")
    print(f"     events           : {[e['name'] for e in EVENTS]}")
    print(f"     event keywords   : {len(EVENTS_KEYWORDS)} terms")
    print(f"     spawn place      : {SPAWN_PLACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
