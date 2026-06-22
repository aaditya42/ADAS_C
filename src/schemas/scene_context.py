from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class SceneContext:
    """High-level scene understanding synthesised from objects, lanes, signs.

    This is the brain of ADAS — a single snapshot of what the ego vehicle
    should know about the current driving scene.
    """

    active_speed_limit: Optional[float]  # km/h, latched from SPEED_LIMIT signs
    stop_required: bool                  # latched STOP sign in last N frames
    yield_required: bool                 # latched YIELD sign in last N frames
    traffic_light_state: Literal["red", "yellow", "green", "none"]
    ego_lane_object_count: int           # objects w/ bbox-center between lanes
    nearest_in_lane_distance_m: Optional[float]
    nearest_in_lane_track_id: Optional[int]
    crosswalk_pedestrian: bool           # latched crosswalk sign + in-lane person
    timestamp: float
