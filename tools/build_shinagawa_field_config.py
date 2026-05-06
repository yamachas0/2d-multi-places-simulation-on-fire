"""【Phase B】 品川FW (フィールドワーク) シミュ用 config を生成する。

教室Phase の出力 (run_dir/fw_handoff.jsonl + run_dir/field_questions.jsonl + run_dir/config.yaml の personas) を持ち越し、
shinagawa_field_places.yaml の物理フィールドで agents を動かす config を出力する。

Usage:
  python tools/build_shinagawa_field_config.py \
    --classroom-run simulations/<classroom_v2_run_dir> \
    [--variant elementary] [--seed 42] [--duration 100]

  → config_shinagawa_field_<variant>.yaml が出力される
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIMBASE = ROOT / "configs" / "config_jr_disruption.yaml"
FIELD_YAML = ROOT / "docs" / "shinagawa_field_places.yaml"
SPAWN_PLACE_NAME = "東西自由通路"  # 駅の東西を結ぶ歩行者デッキの中央 (港南/高輪 両側へ等距離)


# v3: 中心問いを 1文に純化。前提と行動指針は別フィールド (initial_memory) で持たせる。
# ルールベースな「同じ場所避ける」「一緒に行こう同期」は削除し、LLM の自然判断に委ねる。
FW_GOAL = (
    "このまちのいいところや課題を様々な視点で探し、こうなったらいいと思う未来像を描いてください。"
)

# FW 課題 (2026-05-04 v3.1): 中心問いとは別に、運営から渡された具体タスク。
# Phase B smoke で host との対話が成立しなかった (q7 median=1) ため、
# 「最低 2社の企業担当者を訪ねる」ことを明示課題化する。アンケートでこの設定自体に
# コメントが返ることも期待値。
FW_TASK = (
    "[今日のFW課題] FW中、最低でも 2社以上の企業担当者 (受け入れ拠点) のところを"
    "実際に訪ね、自社の取り組みや品川での役割について話を聞いてください。"
    "どの企業を選ぶかは自分の興味で決めて構いませんが、必ず 2社以上に立ち寄ること。"
)

# 中心問いと別に、agent.initial_memory に注入するための前提コンテキスト。
# プロンプト的にも軽量で、ルール文ではなく「状況設定」として与える。
FW_PREMISE = (
    "[前提] これは企業協力型の学外教育プログラムです。午前中 (10:00-11:00) に教室で座学を行い、"
    "昼食をはさんで、午後 13時から 15時までの 2時間、品川駅前に出てフィールドワークをします。"
    "FWではいくつかの企業 (水族館・プリンスホテル・京急・トヨタ・税務署・JR・NTT・ソニー・コクヨ・"
    "食肉市場・浄水場・日鉄興和不動産) が学生の受け入れ担当を配置していて、訪れれば自社の取り組みと"
    "品川の未来について語ってくれます。フィールドは品川駅周辺。15:00 までに出発地である東西自由通路 (駅の東西を結ぶ歩行者デッキの中央、港南側と高輪側のどちらにも等距離) に戻ります。"
    "途中の動きはあなたの判断です。一人で動くか誰かと一緒に動くかも自分で決めて構いません。"
)


def _road_bbox(road: dict) -> tuple[float, float, float, float]:
    """scene_3d.roads の 1 entry から (x_min, x_max, y_min, y_max) を返す。
    axis='ns' (南北) は width が X 方向、length が Y 方向。
    axis='ew' (東西) は length が X 方向、width が Y 方向。
    """
    cx = float(road.get('center_x', 0.0))
    cy = float(road.get('center_y', 0.0))
    w = float(road.get('width_cells', 1.0))
    L = float(road.get('length_cells', 1.0))
    axis = (road.get('axis') or 'ns').lower()
    if axis == 'ew':
        hx, hy = L / 2.0, w / 2.0
    else:
        hx, hy = w / 2.0, L / 2.0
    return cx - hx, cx + hx, cy - hy, cy + hy


def find_pedestrian_position_for_facility(facility_center: tuple[float, float],
                                          roads: list[dict],
                                          offset_cells: float = 0.8) -> tuple[float, float] | None:
    """施設の中心 (fx, fy) から最寄り道路の歩道側 1点を返す。
    歩道 = 道路 bbox 境界の施設寄り側、施設方向に offset_cells 内側。
    fx,fy が ある road の bbox 内 (= 道路の上) なら、その road は最寄り候補から除外。
    """
    fx, fy = facility_center
    best_dist = float('inf')
    best_pos: tuple[float, float] | None = None
    for r in roads:
        x0, x1, y0, y1 = _road_bbox(r)
        # facility が road bbox 内にいるなら歩道は無く、ここはスキップ (別道路の歩道を探す)
        if x0 <= fx <= x1 and y0 <= fy <= y1:
            continue
        # facility 点を road bbox 上にクランプ → 最寄り境界点
        nx = max(x0, min(fx, x1))
        ny = max(y0, min(fy, y1))
        dx = fx - nx
        dy = fy - ny
        dist = (dx * dx + dy * dy) ** 0.5
        if dist >= best_dist:
            continue
        # facility 方向に offset_cells 寄せる (= 歩道 = 道路境界の facility 側 1cell 弱内)
        if dist < 1e-6:
            ped_x, ped_y = nx, ny
        else:
            ped_x = nx + (dx / dist) * offset_cells
            ped_y = ny + (dy / dist) * offset_cells
        best_dist = dist
        best_pos = (ped_x, ped_y)
    return best_pos


def build_initial_memory(handoff: dict, fq: dict) -> list[str]:
    """fw_handoff + field_questions を agent.memory の初期 list に整形する。"""
    lines = []
    fi = (handoff or {}).get("future_image", "").strip()
    intent = (handoff or {}).get("intent", "").strip()
    kms = (handoff or {}).get("key_memories", []) or []
    fq_qs = (fq or {}).get("questions", []) or []
    fq_one = (fq or {}).get("one_liner", "").strip()

    if fi:
        lines.append(f"[座学を経て描いた品川の未来像] {fi}")
    if intent:
        lines.append(f"[これから現地でどう過ごしたいか] {intent}")
    for i, km in enumerate(kms[:3]):
        if km:
            lines.append(f"[座学で気になっていること {i+1}] {km}")
    for i, q in enumerate(fq_qs[:3]):
        if q:
            lines.append(f"[現地で自分の目で確かめたい {i+1}] {q}")
    if fq_one:
        lines.append(f"[フィールドワーク全体の自分のテーマ] {fq_one}")
    # 前提コンテキスト (v3): 中心問いと別に、状況設定として注入する
    lines.append(FW_PREMISE)
    # FW 課題 (v3.1): 最低 2社の企業担当者を訪ねること
    lines.append(FW_TASK)
    # 開始時の状況
    lines.append(
        "[現在地] 東西自由通路 (駅の東西を結ぶ歩行者デッキの中央)。13時に集合してフィールドワークを始めたばかり。港南側にも高輪側にも等距離で行ける。"
        "座学で議論した知識は頭に入っているが、実際に歩いてどんな光景・動線・人の流れに出会うかは現地次第。"
    )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classroom-run", required=True, help="座学シミュの run dir (fw_handoff.jsonl 必須)")
    ap.add_argument("--variant", default="elementary", choices=["high", "adult", "elementary", "junior_high"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--out", default=None, help="出力 yaml path (default: config_shinagawa_field_<variant>.yaml)")
    ap.add_argument("--include-hosts", default=None,
                    help="企業 host persona yaml (12人) を読み込んで personas に append。指定しなければ host なし")
    args = ap.parse_args()

    classroom_run = Path(args.classroom_run)
    if not classroom_run.exists():
        print(f"[err] classroom run dir not found: {classroom_run}", file=sys.stderr)
        return 1
    if not FIELD_YAML.exists():
        print(f"[err] field places yaml not found: {FIELD_YAML}. Run tools/build_shinagawa_field.py first.", file=sys.stderr)
        return 1
    if not SIMBASE.exists():
        print(f"[err] simulation base config not found: {SIMBASE}", file=sys.stderr)
        return 1

    # Load classroom personas + handoff
    classroom_cfg = yaml.safe_load((classroom_run / "config.yaml").read_text(encoding="utf-8"))
    classroom_personas = classroom_cfg["agents"]["personas"]

    handoff_path = classroom_run / "fw_handoff.jsonl"
    fq_path = classroom_run / "field_questions.jsonl"
    if not handoff_path.exists():
        print(f"[err] fw_handoff.jsonl not found: {handoff_path}. Run tools/extract_fw_handoff.py first.", file=sys.stderr)
        return 1

    handoff_by_axis = {}
    for line in handoff_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            handoff_by_axis[r["axis_id"]] = r

    fq_by_axis = {}
    if fq_path.exists():
        for line in fq_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                fq_by_axis[r["axis_id"]] = r

    # Load field places + 3D
    field = yaml.safe_load(FIELD_YAML.read_text(encoding="utf-8"))
    places = field["places"]

    # FW 特例: 食肉市場・浄水場 (industrial) は通常 enterable=False (一般立入禁止) だが、
    # FW では「外の入口で host が出迎えて説明する」想定で、polygon 内に学生が入ること自体は許可する
    # (host の current_goal で「中まで通さず外で説明」と動作レベルで誘導する)。
    for p in places:
        if p.get("name") in ("食肉市場", "浄水場"):
            attrs = p.setdefault("attributes", {})
            attrs["enterable"] = True

    spawn = next((p for p in places if p.get("name") == SPAWN_PLACE_NAME), None)
    if not spawn:
        print(f"[err] spawn place not found in field yaml: {SPAWN_PLACE_NAME}", file=sys.stderr)
        return 1

    # Build personas: copy from classroom, override initial_place + current_goal + initial_memory
    fw_personas = []
    for cp in classroom_personas:
        # 触媒 (Sato/UMA/etc) は持ち越さない (FW phase は当事者のみ)
        axis = cp.get("axis_id", "")
        if axis in ("Sato", "UMA", "AIRobo", "AIGod"):
            continue
        np = copy.deepcopy(cp)
        np["initial_place"] = SPAWN_PLACE_NAME
        np["current_goal"] = FW_GOAL
        # FW_TASK を persona に持ち越し、agent.py の system_prompt 固定挿入で
        # rolling buffer から消えても常に意識下に置く (smoke4 で 2社訪問達成 0/10 だった対策)。
        np["fw_task"] = FW_TASK
        # smoke21: Phase B では persona.background から「まちコンテキスト」(品川 summary +
        # facts + notes_by_variant + 午後FW前提) を除去。Phase A で得た知識は
        # handoff (intent/future_image/key_memories/field_questions) に圧縮済みで、
        # 生 knowledge を Phase B で毎step 1000-1500t 持ち回るのは無駄。
        bg = np.get("background") or ""
        for marker in ("\n\n──── 今日の話し合い対象のまち", "──── 今日の話し合い対象のまち"):
            idx = bg.find(marker)
            if idx >= 0:
                bg = bg[:idx].rstrip()
                break
        np["background"] = bg
        # Phase A の field_questions を persona に持ち越し (smoke21: 18 handoff_block の構成要素)
        fq_rec = fq_by_axis.get(axis) or {}
        fqs = [q for q in (fq_rec.get("questions") or []) if q]
        if fqs:
            np["field_questions"] = fqs[:3]
        # smoke21: handoff intent/future_image/key_memories/one_liner を persona に焼き込む。
        # agent.py の _build_handoff_block_for_system で system_prompt 固定挿入される。
        h = handoff_by_axis.get(axis) or {}
        hi = (h.get("intent") or "").strip()
        hfi = (h.get("future_image") or "").strip()
        hkm = [k for k in (h.get("key_memories") or []) if k]
        fq_one = (fq_rec.get("one_liner") or "").strip()
        if hi:
            np["handoff_intent"] = hi
        if hfi:
            np["handoff_future_image"] = hfi
        if hkm:
            np["handoff_key_memories"] = hkm[:3]
        if fq_one:
            np["handoff_one_liner"] = fq_one
        # 教室シミュでは initial_relationships が他生徒+触媒を含む。FW では当事者のみに絞る
        new_rels = {}
        for other in classroom_personas:
            oa = other.get("axis_id", "")
            if oa in ("Sato", "UMA", "AIRobo", "AIGod"):
                continue
            if other.get("id") == cp.get("id"):
                continue
            old = (cp.get("initial_relationships") or {}).get(other.get("id"))
            if old is not None:
                new_rels[other.get("id")] = old
        np["initial_relationships"] = new_rels
        # 持ち越し memory
        np["initial_memory"] = build_initial_memory(handoff_by_axis.get(axis), fq_by_axis.get(axis))
        # talkativeness を少し上げる (FW中は積極的に動く・話す前提)
        np["talkativeness"] = min(0.85, (np.get("talkativeness") or 0.55) + 0.10)
        # WORKING STATE block で「残り step / 帰還リマインダ」を計算するため total_steps を焼く
        np["total_steps"] = int(args.duration)
        fw_personas.append(np)

    # 学生のスポーン地点 (= 集合場所 = 東西自由通路) を polygon 内に密集配置。
    # 12人想定で 2行 × 6列、spacing 1cell、polygon 中心に置く。
    # 学生数が変わっても自動で grid を切る (cols=ceil(sqrt(N)), rows=ceil(N/cols))。
    students_for_spawn = [p for p in fw_personas if not p.get("is_host")]
    if students_for_spawn and spawn is not None:
        sp_cx = float(spawn.get("center_x", 0.0))
        sp_cy = float(spawn.get("center_y", 0.0))
        sp_hx = float(spawn.get("half_size_x", 5.0))
        sp_hy = float(spawn.get("half_size_y", 2.0))
        n = len(students_for_spawn)
        # 横長 polygon (sp_hx >> sp_hy) なので横並び優先で grid を組む
        if sp_hx >= sp_hy * 2:
            cols = min(n, max(2, int(min(2 * sp_hx, n))))  # 横方向多めに
            rows = math.ceil(n / cols)
        else:
            rows = max(1, int(math.sqrt(n)))
            cols = math.ceil(n / rows)
        spacing_x = min(1.5, (2 * sp_hx - 1.0) / max(1, cols - 1) if cols > 1 else 1.0)
        spacing_y = min(1.0, (2 * sp_hy - 0.5) / max(1, rows - 1) if rows > 1 else 1.0)
        total_w = (cols - 1) * spacing_x
        total_h = (rows - 1) * spacing_y
        for i, sp_persona in enumerate(students_for_spawn):
            col = i % cols
            row = i // cols
            x = sp_cx - total_w / 2 + col * spacing_x
            y = sp_cy - total_h / 2 + row * spacing_y
            sp_persona["initial_position"] = [round(x, 2), round(y, 2)]
        print(f"[info] 学生スポーン: {n}人を {cols}×{rows} grid で東西自由通路内に密集配置")

    print(f"[info] FW personas (students): {len(fw_personas)} (handoff covered: {sum(1 for p in fw_personas if handoff_by_axis.get(p['axis_id']))})")

    # 企業 host persona を append (任意)
    if args.include_hosts:
        host_yaml_path = Path(args.include_hosts)
        if not host_yaml_path.exists():
            print(f"[warn] hosts yaml not found: {host_yaml_path} — skipping", file=sys.stderr)
        else:
            host_doc = yaml.safe_load(host_yaml_path.read_text(encoding="utf-8")) or {}
            hosts = host_doc.get("hosts") or []
            # field yaml の scene_3d.roads を歩道計算に使う
            roads = (field.get("scene_3d") or {}).get("roads") or []
            place_by_name = {p["name"]: p for p in places}
            base_id = max((p.get("id", -1) for p in fw_personas), default=-1) + 1
            host_axes = []
            for i, h in enumerate(hosts):
                hp = copy.deepcopy(h)
                hp["id"] = base_id + i
                hp["variant"] = "host"
                hp["is_host"] = True
                # smoke23: 配置を「東西自由通路 (= 学生スポーン地点) に最も近い施設角」に。
                # 学生がスタート地点から各 host に近づきやすくする (host は固定なので接触確率を上げる)。
                ip_name = hp.get("initial_place")
                facility = place_by_name.get(ip_name)
                if facility is not None:
                    fcx = float(facility.get("center_x", 0.0))
                    fcy = float(facility.get("center_y", 0.0))
                    fhx = float(facility.get("half_size_x", facility.get("half_size", 2.0)) or 2.0)
                    fhy = float(facility.get("half_size_y", facility.get("half_size", 2.0)) or 2.0)
                    # 東西自由通路 中心 (おおよそ x=4.36, y=2.7) との距離が最小の corner
                    central = (4.36, 2.7)
                    corners = [
                        (fcx - fhx, fcy - fhy),
                        (fcx - fhx, fcy + fhy),
                        (fcx + fhx, fcy - fhy),
                        (fcx + fhx, fcy + fhy),
                    ]
                    best_corner = min(corners,
                                      key=lambda c: (c[0]-central[0])**2 + (c[1]-central[1])**2)
                    if roads:
                        ped = find_pedestrian_position_for_facility(best_corner, roads, offset_cells=0.8)
                        if ped is not None:
                            hp["initial_position"] = [round(ped[0], 2), round(ped[1], 2)]
                        else:
                            hp["initial_position"] = [round(best_corner[0], 2), round(best_corner[1], 2)]
                    else:
                        hp["initial_position"] = [round(best_corner[0], 2), round(best_corner[1], 2)]
                # current_goal: 説明員ではなく「街の当事者」型の動機を埋め込む。
                # 行動ルール (3文以内/逆質問必須 等) は入れず、内発的動機として持たせる。
                hp["current_goal"] = (
                    f"あなたは {hp.get('occupation', '企業受け入れ担当')} として、"
                    f"自社拠点 ({hp.get('initial_place', '?')}) で待機しています。"
                    "あなたは単なる施設の説明員ではなく、このまちの運営に関わる一人の当事者です。"
                    "**あなた自身、自社とまちが、若い人や未来の利用者からどう見られているかを知りたい立場にあります**。"
                    "**学生が近くに来たら、必ず自分から声をかけてください** (学生は遠慮しがちなので、"
                    "あなたから「こんにちは、〇〇社の{担当}です。今日は何見てる？」のような短い opener で会話を始める)。"
                    "学生が立ち寄ったら、自社の取り組みやまちへの関わりを語ることも大事ですが、"
                    "それと同じくらい、彼らがこの場所・このまち・自分たちの仕事をどう見ているか、"
                    "何が分かりにくく、何が遠く感じられているかを引き出すことに関心があります。"
                    "正解を一方的に教えるよりも、彼らの観察や違和感を聞いて、"
                    "自分自身のまちの理解を更新する場として、今日の機会を捉えてください。"
                    "学生がいないときは、まちの様子を観察したり、別 host とまちの話をしながら、自分の場で過ごす。"
                )
                # FW 中は host 同士・全 student に対して中立的好意 (0.5)
                rels = {}
                for other in fw_personas:
                    rels[other.get("id")] = 0.5
                hp["initial_relationships"] = rels
                # host の initial_memory: 自分の役割と今日の文脈
                hp["initial_memory"] = [
                    "[役割] 企業受け入れ担当として、自社拠点で学生の訪問を待っている。",
                    "[今日のコンテキスト] 午後 13時から、品川駅周辺で学生たちのフィールドワークが始まった。各拠点の受け入れ担当 12社 (水族館・プリンスホテル・京急・トヨタ・税務署・JR・NTT・ソニー・コクヨ・食肉市場・浄水場・日鉄興和不動産) と連携している。",
                    "[使命] 学生に自社の事業・品川での役割・自分の想いを誠実に伝える。質問に答える。",
                ]
                hp["total_steps"] = int(args.duration)
                fw_personas.append(hp)
                host_axes.append(hp.get("axis_id"))
            # 学生側の relationships に host も追加 (中立 0.45)
            host_id_list = [p.get("id") for p in fw_personas if p.get("is_host")]
            for sp in fw_personas:
                if sp.get("is_host"):
                    continue
                rels = sp.get("initial_relationships") or {}
                for hid in host_id_list:
                    rels.setdefault(hid, 0.45)
                sp["initial_relationships"] = rels
            # working_state 用: 全 host name の一覧を学生 persona に焼き込む。
            # 「未訪問の企業担当者」を毎step prompt 末尾で動的に出すための参照リスト。
            host_name_list = [p.get("name") for p in fw_personas if p.get("is_host") and p.get("name")]
            sp["all_host_names"] = host_name_list  # 既に上のloopで設定済みでもよい
            # smoke23: 各学生に必ず訪問する host を 2 か所 round-robin で割り振り。
            # 12 host を学生 (n人) で round-robin。host あたり訪問者数が均等化される。
            host_full_list = [(p.get("name"), p.get("initial_place", "?"))
                              for p in fw_personas if p.get("is_host")]
            n_h = len(host_full_list)
            students_only = [p for p in fw_personas if not p.get("is_host")]
            for i, sp in enumerate(students_only):
                a1 = host_full_list[(2 * i) % n_h]
                a2 = host_full_list[(2 * i + 1) % n_h]
                sp["all_host_names"] = host_name_list  # 念のため
                sp["assigned_hosts"] = [{"name": a1[0], "place": a1[1]},
                                         {"name": a2[0], "place": a2[1]}]
                # FW_TASK を per-student に custom (assigned 2 host を必ず訪ねる)
                sp["fw_task"] = (
                    "[今日のFW課題] FW中、必ず以下の 2 か所の企業担当者を訪ね、"
                    "自社の取り組みや品川での役割について話を聞いてください。\n"
                    f"  ① {a1[0]} さん (拠点: {a1[1]})\n"
                    f"  ② {a2[0]} さん (拠点: {a2[1]})\n"
                    "自分の関心と違う場所が指定されているように見えても、まずは行ってみる。"
                    "行ってみたからこそ気づけることがある。残りの企業担当者にも余裕があれば立ち寄ってよい。"
                )
            print(f"[info] FW personas (+hosts): {len(fw_personas)} (host axes: {host_axes})")
            print(f"[info] assigned_hosts (round-robin) を {len(students_only)} 学生に割り振り完了")

    # Build sim config
    base = yaml.safe_load(SIMBASE.read_text(encoding="utf-8"))
    sim = copy.deepcopy(base.get("simulation", {}))
    sim["duration"] = args.duration
    sim["half_space_size"] = field["metadata"].get("half_space_size", 80)
    sim["half_place_size"] = 5
    sim["seed"] = args.seed
    # Phase B: 移動あり、Phase 3 復活。
    # minimal_prompt_mode=true だと decision prompt が stub (action_type=stay 強制) になるため、
    # FW では minimal_prompt_mode=false にして full decision prompt (walk_toward/approach 等) を使う。
    sim["minimal_prompt_mode"] = False
    sim["skip_decision_prompt"] = False
    sim["scene_phrase"] = "an outdoor urban scene around Shinagawa Station"
    # context_injection は OFF (knowledge は memory carryover で持ち越し済み)
    sim["context_injection"] = {"enabled": False}
    # Phase B 専用フラグ — perceive 注入機構が読む
    sim["phase"] = "B"
    sim["phase_b_perceive_enabled"] = True
    sim["phase_b_body_log_interval"] = 10  # 何 step 毎に身体感覚 (距離・滞在時間・環境近接) を memory に挿入するか

    # Time scale: 5min/step (歩きシーンとして妥当な粒度)
    sim["time_scale"] = {
        "step_duration_minutes": 2,
        "start_time": "13:00",
        "patterns": [
            {
                "hours": [13, 14, 15, 16, 17, 18],
                "enter_per_step": 0.0,
                "exit_per_step": 0.0,
                "neighborhood_mood": (
                    "品川駅前の午後。午前中の座学と昼食を終えた参加者たちが、ここから街を歩いて確かめ始めている。"
                    "出張族・観光客・地元の人が混在する平日の人流。"
                ),
                "default_goal": "歩きながら、座学で描いた未来像を現地で確かめる",
            },
        ],
    }

    agents = copy.deepcopy(base.get("agents", {}))
    agents["num_agents"] = len(fw_personas)
    agents["max_agents"] = len(fw_personas)
    agents["communication_radius"] = 20  # smoke23: 100m に拡大 (host が place 角に居ても学生 enter 時に nearby 入る)
    agents["parallel_workers"] = 16
    agents["movement_speed"] = {"base_cells_per_step": 12, "variance": 3}  # 60m/step (歩きペースアップ)
    agents["skip_probability"] = 0.05
    agents["message_history_limit"] = 25
    agents["message_context_size"] = 8
    agents["personas"] = fw_personas

    out_path = Path(args.out) if args.out else (ROOT / "configs" / f"config_shinagawa_field_{args.variant}.yaml")

    out = {
        "metadata": {
            "name": f"shinagawa_field_phaseB_{args.variant}",
            "description": (
                f"Phase B: 品川FW シミュ ({args.variant}) — "
                f"座学Phase ({classroom_run.name}) からの memory 持ち越し、"
                "現地で perceive 注入、移動と知覚で未来像をアップデート"
            ),
            "scenario_kind": "shinagawa_field_phaseB",
            "phase": "B",
            "variant": args.variant,
            "town": "shinagawa",
            "town_name": "品川",
            "scene": "shinagawa_field",
            "scene_label": "品川駅前",
            "scene_phrase": sim["scene_phrase"],
            "classroom_run": str(classroom_run),
            "meters_per_cell": field["metadata"].get("meters_per_cell", 5),
            "ceiling_height_m": 5,
            "hide_place_labels": False,
        },
        "simulation": sim,
        "agents": agents,
        "places": places,
        "scene_3d": field.get("scene_3d") or {},
        "llm": copy.deepcopy(base.get("llm", {})),
        "fires": [],
        "events": [],
        "events_keywords": [],
        "visualization": {
            "save_frames": True,
            "frame_interval": 2,
            "focus_agent_id": 0,
            "run_name": f"shinagawa_field_phaseB_{args.variant}",
        },
        "logging": {"level": "INFO"},
    }

    header = (
        "# ================================================================\n"
        f"# Phase B: 品川FW シミュ (variant={args.variant})\n"
        "# ================================================================\n"
        f"# 座学run: {classroom_run.name}\n"
        f"# 参加者: {len(fw_personas)}人 (触媒は除外)\n"
        f"# duration: {args.duration} steps × {sim['time_scale']['step_duration_minutes']}分 = {args.duration*sim['time_scale']['step_duration_minutes']}分\n"
        f"# field: {len(places)} places, half_space={sim['half_space_size']}, scene_3d={'rails+roads+landmarks' if field.get('scene_3d') else 'なし'}\n"
        "# ================================================================\n\n"
    )

    out_path.write_text(
        header + yaml.safe_dump(out, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")
    print(f"     phase: B / variant: {args.variant} / classroom: {classroom_run.name}")
    print(f"     personas: {len(fw_personas)} / places: {len(places)}")
    print(f"     duration: {args.duration} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
