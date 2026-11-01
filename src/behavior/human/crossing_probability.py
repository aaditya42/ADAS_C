from __future__ import annotations

import math
from typing import Optional

from src.behavior.base import LaneContext, MotionEstimate
from src.behavior.config import CrossingProbabilityConfig
from src.utils.logger import get_logger


class GeometricCrossingProbabilityEstimator:
    """Estimates the probability that a tracked entity is crossing the road.

    Uses lane geometry, entity position, heading, and speed to compute a
    crossing probability ∈ [0, 1].

    Args:
        config: Crossing probability configuration.
    """

    def __init__(self, config: CrossingProbabilityConfig) -> None:
        self._cfg = config
        self._logger = get_logger(
            f"{__name__}.GeometricCrossingProbabilityEstimator"
        )

    def estimate(
        self,
        foot_x: float,
        foot_y: float,
        lane_context: Optional[LaneContext],
        heading_deg: Optional[float],
        speed_mps: Optional[float],
    ) -> float:
        """Compute crossing probability.

        Args:
            foot_x: Image x-coordinate of the entity's foot (bottom-center).
            foot_y: Image y-coordinate of the entity's foot.
            lane_context: Lane geometry for the current frame.
            heading_deg: Entity heading in degrees (0 = toward ego camera).
            speed_mps: Entity speed in m/s.

        Returns:
            Crossing probability in [0, 1].
        """
        if lane_context is None or (
            lane_context.left_boundary is None
            and lane_context.right_boundary is None
        ):
            self._logger.debug(
                "No lane context available — returning 0.0"
            )
            return 0.0

        # Check if entity is already inside both lane boundaries
        if (
            lane_context.left_boundary is not None
            and lane_context.right_boundary is not None
        ):
            left_x = lane_context.lane_x_at_y(
                lane_context.left_boundary, foot_y
            )
            right_x = lane_context.lane_x_at_y(
                lane_context.right_boundary, foot_y
            )
            if (
                left_x is not None
                and right_x is not None
                and left_x <= foot_x <= right_x
            ):
                self._logger.debug(
                    "Entity at (%.0f, %.0f) inside lane — probability=1.0",
                    foot_x,
                    foot_y,
                )
                return 1.0

        # Distance to nearest lane boundary
        dist_px = lane_context.distance_to_nearest_boundary(foot_x, foot_y)
        if dist_px is None:
            return 0.0

        dist_factor = math.exp(-dist_px / self._cfg.decay_distance_px)

        # Heading factor: how much the entity faces toward the road
        heading_factor = 0.0
        if heading_deg is not None:
            nearest_boundary_x = self._nearest_boundary_x(
                lane_context, foot_x, foot_y
            )
            if nearest_boundary_x is not None:
                dir_to_road = 1.0 if nearest_boundary_x > foot_x else -1.0
                heading_rad = math.radians(heading_deg)
                # Heading has a horizontal component: cos(heading_rad)
                #   heading=0° (toward camera) → h-component = 0
                #   heading=90° (right) → h-component ≈ 1
                horiz_component = math.sin(heading_rad)
                heading_factor = max(
                    0.0, dir_to_road * horiz_component
                )
                # Clip and weight
                heading_factor = min(1.0, max(0.0, heading_factor))

        # Speed factor: moving toward road
        speed_factor = 0.0
        if speed_mps is not None and speed_mps > 0.0:
            speed_factor = min(1.0, speed_mps / 3.0)

        raw = dist_factor * (1.0 - self._cfg.speed_weight - self._cfg.heading_weight)
        raw += heading_factor * self._cfg.heading_weight
        raw += speed_factor * self._cfg.speed_weight

        prob = max(0.0, min(1.0, raw))

        self._logger.debug(
            "Crossing probability: prob=%.3f  dist_factor=%.3f  "
            "heading_factor=%.3f  speed_factor=%.3f  dist_px=%.0f  "
            "heading=%.0f° speed=%.2f",
            prob,
            dist_factor,
            heading_factor,
            speed_factor,
            dist_px,
            heading_deg or 0.0,
            speed_mps or 0.0,
        )
        return prob

    @staticmethod
    def _nearest_boundary_x(
        lane_context: LaneContext,
        foot_x: float,
        foot_y: float,
    ) -> Optional[float]:
        """Return the x-coordinate of the nearest lane boundary at foot_y."""
        dists: list[tuple[float, float]] = []

        if lane_context.left_boundary is not None:
            left_x = lane_context.lane_x_at_y(
                lane_context.left_boundary, foot_y
            )
            if left_x is not None:
                dists.append((abs(foot_x - left_x), left_x))

        if lane_context.right_boundary is not None:
            right_x = lane_context.lane_x_at_y(
                lane_context.right_boundary, foot_y
            )
            if right_x is not None:
                dists.append((abs(foot_x - right_x), right_x))

        if not dists:
            return None
        dists.sort(key=lambda p: p[0])
        return dists[0][1]
