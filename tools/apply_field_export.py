"""scene_export_<n>.yaml (viewer 編集結果) を docs/shinagawa_field_places.yaml に反映する。

- places の座標/サイズ/属性は export を canonical とする
- perceive_pass / perceive_enter / 既存の environment は old yaml から name キーで持ち越す
- capacity フィールドは全削除
- attributes.enterable=false を指定 place 群に付与
- scene_3d (rails/roads/landmarks) は export をそのまま使う
- old yaml の landmarks に余分な内容がある場合のみ復元

Usage:
  python tools/apply_field_export.py --export <scene_export.yaml> [--out <field_places.yaml>]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs" / "shinagawa_field_places.yaml"

# 個別 name による enterable=false 指定 (旧仕様、後方互換用)
NON_ENTERABLE_NAMES = frozenset({
    "再開発工事中A地区", "再開発工事中D地区", "食肉市場", "浄水場",
    "三菱関東閣", "高輪台住宅地", "Vタワーレジデンス",
    "（関東閣）", "関東閣",
})

# type ベースの enterable=false (型として「中に入れない」もの)
NON_ENTERABLE_TYPES = frozenset({
    "construction",  # 工事中ビル
    "industrial",    # 食肉市場・浄水場
    "residential",   # 住宅
})

# 日本語タイプ表記 (auto-name 用)
TYPE_JP = {
    "residential": "住宅",
    "restaurant": "飲食店",
    "cafe": "カフェ",
    "office_lobby": "オフィス",
    "convenience_store": "コンビニ",
    "hotel": "ホテル",
    "construction": "工事中",
    "industrial": "工場",
    "park": "公園",
    "plaza": "広場",
    "wide_street": "通り",
    "narrow_street": "通り",
    "pedestrian_street": "通路",
    "subway_station": "駅",
    "jr_station": "駅",
    "utility": "施設",
    "department_store": "百貨店",
    "izakaya": "居酒屋",
    "museum": "博物館",
    "library": "図書館",
}


def quadrant(cx: float, cy: float) -> str:
    # cy: +が北 / 西=-x / 東=+x
    ns = "北" if cy >= 15 else ("南" if cy <= -15 else "中央")
    ew = "東" if cx >= 15 else ("西" if cx <= -15 else "中央")
    if ns == "中央" and ew == "中央":
        return "中央"
    if ns == "中央":
        return ew
    if ew == "中央":
        return ns
    return f"{ns}{ew}"


def autoname_places(places: list[dict]) -> list[dict]:
    """name が空 / None の place に内部ID (`_<type>_<象限>_NN`) を付与する。

    シミュレーションは name をユニーク識別子として使うので必須。
    ただし「3Dで吹き出しを出さない」用途なので `attributes.hide_label: true`
    を立てる。viewer 側はこの flag を見てラベル非表示にする。
    """
    counters: dict[tuple, int] = {}
    for p in places:
        if p.get("name"):
            continue
        t = p.get("type", "施設")
        tj = TYPE_JP.get(t, t)
        q = quadrant(float(p.get("center_x", 0)), float(p.get("center_y", 0)))
        key = (tj, q)
        counters[key] = counters.get(key, 0) + 1
        p["name"] = f"_{tj}_{q}_{counters[key]:02d}"
        # ラベル非表示 flag (viewer 側で参照)
        attrs = dict(p.get("attributes") or {})
        attrs["hide_label"] = True
        p["attributes"] = attrs
    return places


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(OLD))
    args = ap.parse_args()

    export_path = Path(args.export)
    if not export_path.exists():
        print(f"[err] export not found: {export_path}", file=sys.stderr)
        return 1

    export = yaml.safe_load(export_path.read_text(encoding="utf-8"))
    old = yaml.safe_load(OLD.read_text(encoding="utf-8"))

    # 1) name 無し place に内部ID + hide_label=true を付与 (sim はユニーク識別子が必要)
    autoname_places(export.get("places") or [])
    # 2) 道路 axis 正規化: width > length なら axis を反転して swap
    roads = (export.get("scene_3d") or {}).get("roads") or []
    n_normalized = 0
    for r in roads:
        w = float(r.get("width_cells", 0))
        L = float(r.get("length_cells", 0))
        if w > L and L > 0:
            axis = (r.get("axis") or "ew").lower()
            r["axis"] = "ew" if axis == "ns" else "ns"
            r["width_cells"], r["length_cells"] = L, w
            n_normalized += 1
    if n_normalized:
        print(f"[info] normalized {n_normalized}/{len(roads)} roads (axis swap + dim swap)")

    # name -> old place dict (perceive_*, environment)
    old_by_name = {p["name"]: p for p in (old.get("places") or []) if p.get("name")}
    old_landmarks = ((old.get("scene_3d") or {}).get("landmarks") or [])
    old_by_landmark = {l.get("name"): l for l in old_landmarks if l.get("name")}

    new_places = []
    for p in export.get("places") or []:
        name = p.get("name")
        if not name:
            continue  # autoname 後でも null ならスキップ (異常)
        np = dict(p)
        # capacity 削除
        np.pop("capacity", None)
        # perceive_* / 既存 environment を old から merge
        op = old_by_name.get(name) or {}
        if "perceive_pass" not in np and "perceive_pass" in op:
            np["perceive_pass"] = op["perceive_pass"]
        if "perceive_enter" not in np and "perceive_enter" in op:
            np["perceive_enter"] = op["perceive_enter"]
        # attributes merge
        attrs = dict(np.get("attributes") or {})
        old_attrs = (op.get("attributes") or {})
        old_env = old_attrs.get("environment") or {}
        cur_env = attrs.get("environment") or {}
        merged_env = {**old_env, **cur_env}
        if merged_env:
            attrs["environment"] = merged_env
        # enterable=false: name 個別指定 OR type ベース
        ptype = p.get("type")
        if name in NON_ENTERABLE_NAMES or ptype in NON_ENTERABLE_TYPES:
            attrs["enterable"] = False
        if attrs:
            np["attributes"] = attrs
        new_places.append(np)

    # scene_3d
    new_scene = dict(export.get("scene_3d") or {})
    # landmarks: export 側が truncated っぽい (1個しかない) なら old から補完
    if "landmarks" in new_scene and len(new_scene["landmarks"]) <= 1:
        if old_landmarks and len(old_landmarks) > len(new_scene["landmarks"]):
            new_scene["landmarks"] = old_landmarks

    # metadata
    metadata = dict(old.get("metadata") or {})
    metadata.setdefault("name", "shinagawa_field_phaseB")
    metadata.setdefault(
        "description", "Phase B: 品川駅周辺 400m圏 (FW シミュ用、perceive 属性付き)"
    )
    metadata.setdefault("half_space_size", 80)
    metadata.setdefault("meters_per_cell", 5)
    metadata["generated_from"] = (
        "tools/apply_field_export.py + scene_export (viewer 編集結果)"
    )

    out = {
        "metadata": metadata,
        "places": new_places,
        "scene_3d": new_scene,
    }

    out_path = Path(args.out)
    out_path.write_text(
        yaml.safe_dump(out, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")
    print(f"     places: {len(new_places)}")
    n_blocked = sum(1 for p in new_places if (p.get("attributes") or {}).get("enterable") is False)
    print(f"     enterable=false: {n_blocked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
