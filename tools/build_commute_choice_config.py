"""Build config_commute_choice.yaml — 帰宅移動手段の選択シミュ。

概念フィールド：
    [オフィス]──┬── 鉄道(JR)  ──┬──[自宅]
                ├── 路線バス     ──┤
                ├── タクシー     ──┤
                ├── 自転車      ──┤
                └── 徒歩        ──┘
    オフィス南側に 周辺施設 (カフェ/コンビニ/居酒屋) — 電車復旧待ちの避難先

100人の会社員ペルソナを自動生成 (電車80/バス10/自転車5/徒歩5)。
タクシーは平常時は誰も使わず、JR運休時の代替候補としてだけ存在。
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIMBASE = ROOT / "config_jr_disruption.yaml"
OUT = ROOT / "config_commute_choice.yaml"

# ---------------------------------------------------------------------------
# Field layout (half_space_size = 60)
# ---------------------------------------------------------------------------
HALF_SPACE = 60

PLACES = [
    # ===== オフィス & 自宅 (左右の端) =====
    {"name": "オフィス", "type": "office_lobby",
     "center_x": -50, "center_y": 0, "half_size_x": 8, "half_size_y": 35,
     "capacity": 200, "social_likelihood": 0.20},
    {"name": "自宅エリア", "type": "office_lobby",
     "center_x": 50, "center_y": 0, "half_size_x": 8, "half_size_y": 35,
     "capacity": 200, "social_likelihood": 0.05},

    # ===== 5本の通勤レーン (中央を東西に走る) =====
    {"name": "JR線（電車）", "type": "wide_street",
     "center_x": 0, "center_y": 40, "half_size_x": 38, "half_size_y": 4,
     "capacity": 100, "social_likelihood": 0.10,
     "road_direction": "east_west", "lanes": 4},
    {"name": "路線バス", "type": "wide_street",
     "center_x": 0, "center_y": 20, "half_size_x": 38, "half_size_y": 4,
     "capacity": 60, "social_likelihood": 0.15,
     "road_direction": "east_west", "lanes": 2},
    {"name": "タクシー", "type": "wide_street",
     "center_x": 0, "center_y": 0, "half_size_x": 38, "half_size_y": 4,
     "capacity": 40, "social_likelihood": 0.10,
     "road_direction": "east_west", "lanes": 2},
    {"name": "自転車レーン", "type": "pedestrian_street",
     "center_x": 0, "center_y": -20, "half_size_x": 38, "half_size_y": 4,
     "capacity": 40, "social_likelihood": 0.08,
     "road_direction": "east_west", "lanes": 0},
    {"name": "徒歩道", "type": "pedestrian_street",
     "center_x": 0, "center_y": -40, "half_size_x": 38, "half_size_y": 4,
     "capacity": 50, "social_likelihood": 0.20,
     "road_direction": "east_west", "lanes": 0},

    # ===== オフィス南側の周辺施設 (運休待ち避難先) =====
    {"name": "オフィス南カフェ", "type": "cafe",
     "center_x": -55, "center_y": -55, "half_size_x": 4, "half_size_y": 3,
     "capacity": 20, "social_likelihood": 0.5},
    {"name": "オフィス南コンビニ", "type": "convenience_store",
     "center_x": -45, "center_y": -55, "half_size_x": 4, "half_size_y": 3,
     "capacity": 15, "social_likelihood": 0.2},
    {"name": "オフィス南居酒屋", "type": "izakaya",
     "center_x": -35, "center_y": -55, "half_size_x": 4, "half_size_y": 3,
     "capacity": 18, "social_likelihood": 0.7},
]

# ---------------------------------------------------------------------------
# Persona generation
# ---------------------------------------------------------------------------
SURNAMES = [
    "山田", "佐藤", "鈴木", "田中", "伊藤", "渡辺", "中村", "小林", "加藤", "吉田",
    "山口", "松本", "井上", "木村", "林", "清水", "山崎", "池田", "橋本", "阿部",
    "石川", "山下", "中島", "前田", "藤田", "後藤", "岡田", "長谷川", "村上", "近藤",
    "石井", "斎藤", "坂本", "遠藤", "青木", "藤井", "福田", "太田", "西村", "藤原",
    "岡本", "金子", "中川", "中野", "原田", "小野", "田村", "竹内", "安藤", "宮崎",
]
GIVEN_M = [
    "太郎", "健一", "明", "誠", "進", "勇", "博", "清", "隆", "修",
    "哲也", "雅彦", "和夫", "和也", "俊樹", "慎吾", "翔太", "大輔", "優", "一郎",
    "雄一", "健太", "拓海", "悠斗", "陸", "海斗", "駿", "蓮", "樹", "翼",
    "亮", "達也", "智", "拓也", "光", "宏", "聡", "学",
]
GIVEN_F = [
    "花子", "美香", "由美", "恵子", "洋子", "和子", "智子", "典子", "京子", "節子",
    "優子", "香織", "亜紀", "千恵", "麻衣", "美咲", "彩", "結衣", "咲", "莉子",
    "桜", "葵", "詩織", "舞", "理沙", "真理", "由香", "彩花", "あかね", "実",
    "陽子", "千夏", "茜", "瞳", "百合", "奈々",
]
OCCUPATIONS = [
    "営業", "経理", "人事", "マーケター", "SE", "プロダクトマネージャー",
    "コンサルタント", "事務", "開発エンジニア", "企画", "広報", "総務",
    "法務", "購買", "経営企画", "カスタマーサポート", "データアナリスト",
    "デザイナー", "広告営業", "営業企画", "係長", "課長", "部長", "主任",
]
SPEECH_STYLES = [
    "ハキハキした敬語",
    "落ち着いた丁寧語",
    "明るくフレンドリー",
    "カジュアルでタメ口混じり",
    "ぼそぼそした静かな口調",
    "テキパキした口調",
    "穏やかな丁寧語",
    "気さくな砕けた口調",
    "簡潔でビジネスライク",
    "おっとりした柔らかい口調",
]

# 通勤手段ごとの仕様
COMMUTE_MODES = {
    "電車": {
        "count": 80,
        "initial_lane": "JR線（電車）",
        "background_tag": "通勤手段はJR電車。普段は会社→駅→電車→自宅の流れで帰る。",
        "goal_tag": "JR電車で帰宅する予定",
    },
    "バス": {
        "count": 10,
        "initial_lane": "路線バス",
        "background_tag": "通勤手段は路線バス。電車は使わずバス1本で会社と自宅を往復している。",
        "goal_tag": "バスで帰宅する予定",
    },
    "自転車": {
        "count": 5,
        "initial_lane": "自転車レーン",
        "background_tag": "通勤手段は自転車。会社の駐輪場に毎朝停めている。",
        "goal_tag": "自転車で帰宅する予定",
    },
    "徒歩": {
        "count": 5,
        "initial_lane": "徒歩道",
        "background_tag": "通勤手段は徒歩。会社まで歩いて通っている。",
        "goal_tag": "歩いて帰宅する予定",
    },
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _assign_modes() -> list:
    """Returns a list of 100 commute modes in agent-id order, deterministic."""
    out = []
    for mode, spec in COMMUTE_MODES.items():
        out.extend([mode] * spec["count"])
    assert len(out) == 100, f"COMMUTE_MODES total must be 100, got {len(out)}"
    return out


def _phone_check_rate(rng: random.Random, age: int) -> float:
    # 大ざっぱに年齢で減衰、20代で 0.7 平均、50代で 0.45 平均
    base = max(0.30, 0.85 - 0.008 * (age - 22))
    return round(min(0.95, max(0.10, base + rng.uniform(-0.10, 0.10))), 2)


def _talkativeness(rng: random.Random) -> float:
    return round(rng.uniform(0.15, 0.65), 2)


def generate_personas(seed: int) -> list:
    rng = _rng(seed)
    modes = _assign_modes()
    personas = []

    for i, mode in enumerate(modes):
        gender = "male" if rng.random() < 0.55 else "female"
        surname = rng.choice(SURNAMES)
        given = rng.choice(GIVEN_M if gender == "male" else GIVEN_F)
        name = surname + given
        age = rng.randint(22, 58)
        occ = rng.choice(OCCUPATIONS)
        speech = rng.choice(SPEECH_STYLES)
        spec = COMMUTE_MODES[mode]

        background = (
            f"{occ}として勤続{max(1, age - 22)}年の会社員。"
            f"{spec['background_tag']}"
            "タクシーは普段使わず、雨や緊急時のみの選択肢として頭にある。"
        )
        current_goal = (
            f"18:00定時で退勤、{spec['goal_tag']}。早く家に帰りたい。"
        )

        persona = {
            "id": i,
            "name": name,
            "age": age,
            "gender": gender,
            "occupation": occ,
            "background": background,
            "speech_style": speech,
            "current_goal": current_goal,
            "talkativeness": _talkativeness(rng),
            "phone_check_rate": _phone_check_rate(rng, age),
            "initial_place": "オフィス",
            "preferred_commute": mode,  # メタデータ用 (prompt には埋め込まない)
        }
        personas.append(persona)

    return personas


# ---------------------------------------------------------------------------
# Time / Events / Keywords
# ---------------------------------------------------------------------------
TIME_SCALE = {
    "step_duration_minutes": 1,
    "start_time": "18:00",
    "patterns": [
        {
            "hours": [18, 19],
            "enter_per_step": 0.0,  # 全員固定 — 新規 spawn なし
            "exit_per_step": 0.0,   # despawn もなし
            "neighborhood_mood": (
                "退勤時間。オフィスから帰宅手段を選んで自宅へ向かう人々。"
                "JR線・バス・タクシー・自転車・徒歩の5本のレーンが並走している。"
            ),
            "default_goal": "退勤し、適切な手段で帰宅する",
        },
    ],
}


EVENTS = [
    {
        "name": "jr_homebound_outage",
        "type": "transit_disruption",
        # 18:05 (start_step=5) に発火 — 5分間は普通に帰宅、その後JR運休
        "start_step": 5,
        "end_step": 30,
        "broadcast_once": True,
        "affected_place": "JR線（電車）",
        "description": (
            "JR山手線・京浜東北線で人身事故発生、運転見合わせ。"
            "復旧見通し立たず。振替輸送案内中。"
        ),
        "effects": {
            # block_spawn_at は使わない (spawn 自体ゼロ運用) — 通知範囲のみ
            "notify_agents_within_radius": 999,
        },
        # social_identities を持たない sim のため salience_boost は空
        "salience_boost_targets": [],
    },
]


EVENTS_KEYWORDS = [
    # 鉄道・運休関連
    "電車", "JR", "山手線", "京浜東北線", "運休", "運転見合わせ", "人身事故", "遅延", "復旧",
    # 代替手段
    "バス", "路線バス", "タクシー", "自転車", "徒歩", "歩いて", "歩く",
    # 振替・代替
    "振替", "振替輸送", "代替", "迂回",
    # 帰宅判断
    "帰宅", "帰り道", "帰る", "帰れない", "どうする", "どうしよう", "様子見",
    # 待避先
    "カフェ", "コンビニ", "居酒屋", "待つ", "復旧まで",
]


# ---------------------------------------------------------------------------
# Build entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main"], default="smoke",
                    help="smoke=20agents x15step, main=100agents x30step")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not SIMBASE.exists():
        print(f"[err] missing: {SIMBASE}", file=sys.stderr)
        return 1
    base = yaml.safe_load(SIMBASE.read_text(encoding="utf-8"))

    if args.mode == "smoke":
        num_agents, duration, parallel_workers = 20, 15, 8
    else:
        num_agents, duration, parallel_workers = 100, 30, 25

    all_personas = generate_personas(args.seed)
    # smoke のときは比率を保ったまま先頭 num_agents 人を取る:
    # 80:10:5:5 → 20人なら 16:2:1:1 (id 0-15 電車, 16-17 バス, 18 自転車, 19 徒歩)
    if args.mode == "smoke":
        smoke_ids = list(range(16)) + [80, 81, 90, 95]
        personas = [p for p in all_personas if p["id"] in smoke_ids]
        # id 振り直し (連番にしないと色々厄介)
        for new_id, p in enumerate(personas):
            p["id"] = new_id
    else:
        personas = all_personas

    sim = copy.deepcopy(base.get("simulation", {}))
    sim["duration"] = duration
    sim["half_space_size"] = HALF_SPACE
    sim["half_place_size"] = 5
    sim["seed"] = args.seed
    sim["time_scale"] = TIME_SCALE

    agents = copy.deepcopy(base.get("agents", {}))
    agents["num_agents"] = num_agents
    agents["max_agents"] = num_agents
    agents["communication_radius"] = 12
    agents["parallel_workers"] = parallel_workers
    agents["personas"] = personas
    # movement_speed をオフィス→自宅の距離(~100cell)に合わせて少し速く
    agents["movement_speed"] = {"base_cells_per_step": 8, "variance": 2}

    # focus agent: 電車通勤者の id 0 (smoke でも main でも先頭は電車組)
    out = {
        "simulation": sim,
        "agents": agents,
        "places": PLACES,
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
        "# 帰宅移動手段の選択シミュ (commute_choice)\n"
        "# ================================================================\n"
        "# Built by tools/build_commute_choice_config.py\n"
        "#\n"
        "# フィールド：[オフィス]→ 5本のレーン (鉄道/バス/タクシー/自転車/徒歩) →[自宅]\n"
        "# シナリオ：18:00定時退勤 → 18:05にJR運休発生 → 各エージェントが代替手段を選ぶ\n"
        "# 通勤手段比率 (本線): 電車80 / バス10 / 自転車5 / 徒歩5\n"
        "#                     タクシーは平常時0、運休時の代替候補\n"
        f"#   - num_agents       : {num_agents}\n"
        f"#   - duration (steps) : {duration}\n"
        f"#   - parallel_workers : {parallel_workers}\n"
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

    # 簡易レポート
    by_mode = {}
    for p in personas:
        by_mode[p["preferred_commute"]] = by_mode.get(p["preferred_commute"], 0) + 1

    print(f"[ok] wrote: {OUT}")
    print(f"     mode             : {args.mode}")
    print(f"     num_agents       : {num_agents}")
    print(f"     duration         : {duration} steps")
    print(f"     parallel_workers : {parallel_workers}")
    print(f"     places           : {len(PLACES)}")
    print(f"     events           : {[e['name'] for e in EVENTS]}")
    print(f"     event keywords   : {len(EVENTS_KEYWORDS)} terms")
    print(f"     commute mix      : {by_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
