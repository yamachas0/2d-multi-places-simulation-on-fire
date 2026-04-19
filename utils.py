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


class PlaceConfig(TypedDict):
    """Type definition for place configuration"""
    name: str  # Place name (required)
    type: str  # Place type: bar, cafe, library, etc. (required)
    center_x: int  # X coordinate of place center (required)
    center_y: int  # Y coordinate of place center (required)
    half_size: int  # Half size of the place (required)
    capacity: int  # Maximum comfortable capacity of the place (required)


def is_position_in_place(
    position: Tuple[int, int],
    half_size: int,
    center_x: int = 0,
    center_y: int = 0
) -> bool:
    """
    Check if a position is inside a place area.
    Place is centered at (center_x, center_y).

    Args:
        position: (x, y) coordinates to check
        half_size: Half size of the place (place covers -half_size to +half_size from center)
        center_x: X coordinate of place center (default: 0)
        center_y: Y coordinate of place center (default: 0)

    Returns:
        True if position is inside the place, False otherwise
    """
    x, y = position
    # Place covers -half_size to +half_size (inclusive on both ends) from center
    return (center_x - half_size <= x <= center_x + half_size and
            center_y - half_size <= y <= center_y + half_size)


def get_place_at_position(
    position: Tuple[int, int],
    places: List[PlaceConfig]
) -> Optional[PlaceConfig]:
    """
    Get the place that contains the given position.

    Args:
        position: (x, y) coordinates to check
        places: List of place configurations, each with 'center_x', 'center_y', 'half_size'

    Returns:
        Place dictionary if position is in a place, None otherwise
    """
    for place in places:
        if is_position_in_place(
            position,
            place['half_size'],
            place['center_x'],
            place['center_y']
        ):
            return place
    return None
