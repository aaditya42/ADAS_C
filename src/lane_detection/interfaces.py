from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.schemas.lane_info import LaneLine


def _fit_quadratic(
    points: tuple[tuple[int, int], ...],
) -> tuple[float, float, float] | None:
    """Fit x = a*y² + b*y + c to points, return (a, b, c) or None."""
    if len(points) < 3:
        return None
    ys = np.array([p[1] for p in points], dtype=float)
    xs = np.array([p[0] for p in points], dtype=float)
    coeffs = np.polyfit(ys, xs, 2)
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))


@dataclass(frozen=True)
class LaneDetectionResult:
    left_lane_points: tuple[tuple[int, int], ...]
    right_lane_points: tuple[tuple[int, int], ...]
    center_lane_points: tuple[tuple[int, int], ...]
    lane_count: int
    ego_lane_center_x: float | None
    confidence: float
    timestamp: float | None = None

    @property
    def left_lane(self) -> LaneLine | None:
        coeffs = _fit_quadratic(self.left_lane_points)
        if coeffs is None:
            return None
        return LaneLine(coeffs=coeffs, confidence=self.confidence)

    @property
    def right_lane(self) -> LaneLine | None:
        coeffs = _fit_quadratic(self.right_lane_points)
        if coeffs is None:
            return None
        return LaneLine(coeffs=coeffs, confidence=self.confidence)

    @staticmethod
    def empty() -> LaneDetectionResult:
        return LaneDetectionResult(
            left_lane_points=(),
            right_lane_points=(),
            center_lane_points=(),
            lane_count=0,
            ego_lane_center_x=None,
            confidence=0.0,
        )
