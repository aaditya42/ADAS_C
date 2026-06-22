from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Direction = Literal["approaching", "receding", "stationary", "unknown"]


@dataclass(frozen=True)
class RelativeMotion:
    """Motion estimate for a single tracked object.

    speed_mps:  closing speed (positive = approaching the ego vehicle).
    acceleration_mps2:  change in speed (positive = accelerating toward ego).
    direction:  semantic label derived from speed and stationary threshold.
    confidence:  quality of the estimate in [0, 1].
    """

    track_id: int
    speed_mps: float | None
    acceleration_mps2: float | None
    direction: Direction
    confidence: float
