from __future__ import annotations

from dataclasses import dataclass

from src.schemas.bbox import BoundingBox


@dataclass(frozen=True)
class LaneLine:
    """A single lane line modelled as a set of polynomial coefficients.

    coeffs are ordered from highest degree to constant term
    (e.g. [a, b, c] for x = a*y^2 + b*y + c in image space).
    """

    coeffs: tuple[float, ...]
    confidence: float


@dataclass(frozen=True)
class LaneInfo:
    """Lane information for the current frame.

    left_lane and right_lane represent the left and right lane boundaries
    respectively. lane_center is an optional suggested centre-line. All
    coordinates are in image pixel space.
    """

    left_lane: LaneLine | None
    right_lane: LaneLine | None
    lane_center: LaneLine | None
    confidence: float
