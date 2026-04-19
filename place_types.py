"""
Place-type definitions for the urban-scale simulation.

Each entry describes qualitative attributes of a place type that agents can
reason about (via the prompt) and that later features can use for behavior
selection (e.g. station as a spawn point, library as a dwelling target).
"""
from typing import Dict, TypedDict


class PlaceTypeSpec(TypedDict):
    atmosphere: str            # Short human-readable label for the feel of the place
    social_likelihood: float   # 0.0-1.0 — how likely people are to talk to strangers here
    typical_stay_duration: int # Typical stay in minutes
    is_road: bool              # True for streets/corridors that people only pass through
    is_spawn_point: bool       # True for places agents can enter/exit the district through


# Order matters only for display; lookup is by key.
PLACE_TYPE_DEFINITIONS: Dict[str, PlaceTypeSpec] = {
    "station": {
        "atmosphere": "busy, transient — commuters coming and going",
        "social_likelihood": 0.25,
        "typical_stay_duration": 3,
        "is_road": False,
        "is_spawn_point": True,
    },
    "cafe": {
        "atmosphere": "warm, relaxed — easy to linger",
        "social_likelihood": 0.6,
        "typical_stay_duration": 25,
        "is_road": False,
        "is_spawn_point": False,
    },
    "bar": {
        "atmosphere": "cozy, social — conversations start naturally",
        "social_likelihood": 0.8,
        "typical_stay_duration": 40,
        "is_road": False,
        "is_spawn_point": False,
    },
    "library": {
        "atmosphere": "quiet, focused — people mostly keep to themselves",
        "social_likelihood": 0.1,
        "typical_stay_duration": 60,
        "is_road": False,
        "is_spawn_point": False,
    },
    "park": {
        "atmosphere": "open, casual — walking, sitting, watching people",
        "social_likelihood": 0.4,
        "typical_stay_duration": 15,
        "is_road": False,
        "is_spawn_point": False,
    },
    "plaza": {
        "atmosphere": "lively, public — meeting point for crossing paths",
        "social_likelihood": 0.5,
        "typical_stay_duration": 10,
        "is_road": False,
        "is_spawn_point": False,
    },
    "road": {
        "atmosphere": "transient — people walk through, rarely stop",
        "social_likelihood": 0.15,
        "typical_stay_duration": 2,
        "is_road": True,
        "is_spawn_point": False,
    },
}


UNKNOWN_PLACE_TYPE_DEFAULT: PlaceTypeSpec = {
    "atmosphere": "generic urban space",
    "social_likelihood": 0.3,
    "typical_stay_duration": 10,
    "is_road": False,
    "is_spawn_point": False,
}


def get_place_type_spec(place_type: str) -> PlaceTypeSpec:
    """Return the spec for a place type, falling back to a generic default."""
    return PLACE_TYPE_DEFINITIONS.get(place_type, UNKNOWN_PLACE_TYPE_DEFAULT)
