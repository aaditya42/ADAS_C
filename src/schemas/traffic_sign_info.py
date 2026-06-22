from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.schemas.bbox import BoundingBox


class SignType(Enum):
    SPEED_LIMIT = "speed_limit"
    STOP = "stop"
    YIELD = "yield"
    CROSSWALK = "crosswalk"
    NO_ENTRY = "no_entry"
    ONE_WAY = "one_way"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    TRAFFIC_LIGHT_RED = "traffic_light_red"
    TRAFFIC_LIGHT_GREEN = "traffic_light_green"
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrafficSignInfo:
    """A detected traffic sign.

    sign_type categorises the sign. bbox is its location in the image.
    value carries an optional numeric payload (e.g. the speed limit in km/h).
    """

    sign_type: SignType
    bbox: BoundingBox
    confidence: float
    value: Optional[float] = None
