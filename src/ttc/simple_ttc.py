from __future__ import annotations

import math

from src.schemas import PerceivedObject
from src.schemas.motion import RelativeMotion
from src.ttc.base import BaseTTCEstimator


class SimpleTTCEstimator(BaseTTCEstimator):
    """Time-to-collision from distance and closing speed.

    TTC = distance / speed_mps, only computed for approaching objects
    (speed_mps > min_speed_mps).

    Args:
        max_ttc_seconds: Clip finite TTC values to this maximum.
        min_speed_mps: Minimum closing speed to consider meaningful approach.
    """

    def __init__(
        self,
        max_ttc_seconds: float = 30.0,
        min_speed_mps: float = 0.2,
    ) -> None:
        self._max_ttc = max_ttc_seconds
        self._min_speed = min_speed_mps

    def estimate(
        self,
        perceived: list[PerceivedObject],
        motion: dict[int, RelativeMotion],
    ) -> dict[int, float | None]:
        result: dict[int, float | None] = {}

        for obj in perceived:
            tid = obj.track_id
            mot = motion.get(tid)

            if mot is None or mot.speed_mps is None:
                result[tid] = None
                continue

            dist = obj.distance
            if dist <= 0:
                result[tid] = None
                continue

            speed = mot.speed_mps

            if speed <= self._min_speed:
                result[tid] = math.inf
            else:
                ttc = dist / speed
                result[tid] = min(ttc, self._max_ttc)

        return result
