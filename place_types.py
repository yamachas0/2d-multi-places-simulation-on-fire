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
    "jr_station": {
        "atmosphere": "busy JR terminal — dense commuter flow, loudspeaker announcements",
        "social_likelihood": 0.05,
        "typical_stay_duration": 3,
        "is_road": False,
        "is_spawn_point": True,
    },
    "subway_station": {
        "atmosphere": "underground transit hub — narrow corridors, purposeful walking",
        "social_likelihood": 0.08,
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
    "izakaya": {
        "atmosphere": "japanese pub — lively tables, easy to overhear neighbours",
        "social_likelihood": 0.75,
        "typical_stay_duration": 50,
        "is_road": False,
        "is_spawn_point": False,
    },
    "restaurant": {
        "atmosphere": "table service — conversation by party, polite to others",
        "social_likelihood": 0.3,
        "typical_stay_duration": 45,
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
    "convenience_store": {
        "atmosphere": "brief, functional — quick stops for coffee, snacks, ATM",
        "social_likelihood": 0.15,
        "typical_stay_duration": 4,
        "is_road": False,
        "is_spawn_point": False,
    },
    "department_store": {
        "atmosphere": "large retail — browsing, quiet announcements, window-shoppers",
        "social_likelihood": 0.2,
        "typical_stay_duration": 30,
        "is_road": False,
        "is_spawn_point": False,
    },
    "office_lobby": {
        "atmosphere": "formal — colleagues nod in passing, reception counter",
        "social_likelihood": 0.2,
        "typical_stay_duration": 5,
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
    "wide_street": {
        "atmosphere": "main avenue — wide sidewalks, steady foot traffic",
        "social_likelihood": 0.1,
        "typical_stay_duration": 2,
        "is_road": True,
        "is_spawn_point": False,
    },
    "narrow_street": {
        "atmosphere": "backstreet alley — quieter, residential feel",
        "social_likelihood": 0.08,
        "typical_stay_duration": 2,
        "is_road": True,
        "is_spawn_point": False,
    },
    "pedestrian_street": {
        "atmosphere": "pedestrian-only promenade — relaxed pace, benches",
        "social_likelihood": 0.2,
        "typical_stay_duration": 4,
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
