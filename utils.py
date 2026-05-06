"""
Shared utility functions for LLM-based agent in 2D worlds with multiple places.
"""
import random
from typing import Tuple, Optional, List, Dict, TypedDict


_PERSONA_POOL = {
    "male_names": ["健一", "太郎", "翔太", "大輔", "拓也", "雅人", "慎二", "浩二", "隆司", "正樹"],
    "female_names": ["美咲", "結衣", "彩香", "由紀", "麻衣", "千尋", "里奈", "沙織", "陽子", "恵美"],
    "family_names": ["田中", "鈴木", "佐藤", "山田", "高橋", "渡辺", "伊藤", "中村", "小林", "加藤"],
    "occupations": [
        "IT会社員", "公務員", "デザイナー", "大学生", "美容師", "料理人",
        "営業職", "看護師", "教師", "フリーター", "自営業", "定年退職", "主婦/主夫",
    ],
    "backgrounds": [
        "最近この街に引っ越してきた",
        "この街で生まれ育った",
        "仕事の合間に散歩しているだけ",
        "友達に会いに来た",
        "常連のバーに向かっている途中",
        "特に目的はなくぶらぶらしている",
        "この街に住んで5年ほど",
        "今日はたまたまこの辺にいる",
    ],
    "speech_styles": [
        "カジュアルでフレンドリー。タメ口混じり",
        "丁寧だが堅すぎない。敬語ベース",
        "のんびりした話し方。間が多い",
        "早口でテンポが良い。ノリが軽い",
        "落ち着いた話し方。聞き上手",
        "少し無愛想。必要最低限の会話",
        "陽気で声が大きいタイプ",
        "人見知り気味で最初はおとなしい",
    ],
    "ages": {
        "young": (18, 30),
        "middle": (31, 50),
        "senior": (51, 75),
    },
}


def generate_random_persona(agent_id: int, gender: Optional[str] = None) -> Dict:
    """Generate a minimal random persona for an agent.

    The returned dict is kept flat and JSON-serialisable on purpose — it will
    later be extended with richer fields (narrative self, biases, etc.) in the
    second-stage agent expansion, and a flat dict is the easiest shape to
    migrate.
    """
    if gender is None:
        gender = random.choice(["male", "female"])

    age_group = random.choice(["young", "middle", "senior"])
    age_range = _PERSONA_POOL["ages"][age_group]
    age = random.randint(*age_range)

    name_pool = _PERSONA_POOL["male_names"] if gender == "male" else _PERSONA_POOL["female_names"]
    family_name = random.choice(_PERSONA_POOL["family_names"])
    first_name = random.choice(name_pool)

    return {
        "id": agent_id,
        "name": f"{family_name}{first_name}",
        "age": age,
        "gender": gender,
        "occupation": random.choice(_PERSONA_POOL["occupations"]),
        "background": random.choice(_PERSONA_POOL["backgrounds"]),
        "speech_style": random.choice(_PERSONA_POOL["speech_styles"]),
    }


class FireConfig(TypedDict, total=False):
    """Type definition for fire event configuration"""
    name: str  # Fire name (required)
    start_step: int  # Step at which fire appears (required)
    intensity: float  # Fire intensity (0.0 to 1.0) (required)
    radius: int  # Perception radius (required)
    center_x: int  # Fire position X (optional, random if omitted)
    center_y: int  # Fire position Y (optional, random if omitted)


class PlaceConfig(TypedDict, total=False):
    """Type definition for place configuration.

    A place is an axis-aligned rectangle. Its X extent is ±half_size_x from
    center_x; its Y extent is ±half_size_y from center_y. For square places,
    half_size can be supplied as a shorthand that populates both axes.
    """
    name: str  # required
    type: str  # required (bar/cafe/station/...)
    center_x: int  # required
    center_y: int  # required
    half_size: int      # square shorthand (populates half_size_x / half_size_y)
    half_size_x: int    # rectangular extent along X
    half_size_y: int    # rectangular extent along Y
    capacity: int  # optional (legacy)


def _resolve_half_sizes(place: Dict) -> Tuple[int, int]:
    """Return (half_size_x, half_size_y) for a place, supporting legacy half_size."""
    if 'half_size_x' in place or 'half_size_y' in place:
        hx = place.get('half_size_x', place.get('half_size', 5))
        hy = place.get('half_size_y', place.get('half_size', 5))
        return int(hx), int(hy)
    hs = place.get('half_size', 5)
    return int(hs), int(hs)


def is_position_in_place(
    position: Tuple[int, int],
    half_size=None,
    center_x: int = 0,
    center_y: int = 0,
    half_size_x: Optional[int] = None,
    half_size_y: Optional[int] = None,
) -> bool:
    """Check if a position is inside a rectangular place centered at (center_x, center_y).

    Back-compat: if only `half_size` is supplied, the place is treated as a
    square of that half-extent on both axes.
    """
    x, y = position
    if half_size_x is None:
        half_size_x = half_size if half_size is not None else 5
    if half_size_y is None:
        half_size_y = half_size if half_size is not None else 5
    return (center_x - half_size_x <= x <= center_x + half_size_x and
            center_y - half_size_y <= y <= center_y + half_size_y)


def get_place_at_position(
    position: Tuple[int, int],
    places: List[PlaceConfig]
) -> Optional[PlaceConfig]:
    """Return the place containing the given position, or None.

    enterable=false の place は素通り扱い (道路/通路と bbox が重なるケースで
    「中にいる」判定にしない)。
    """
    for place in places:
        if (place.get('attributes') or {}).get('enterable') is False:
            continue
        hx, hy = _resolve_half_sizes(place)
        if is_position_in_place(
            position,
            center_x=place['center_x'],
            center_y=place['center_y'],
            half_size_x=hx,
            half_size_y=hy,
        ):
            return place
    return None
