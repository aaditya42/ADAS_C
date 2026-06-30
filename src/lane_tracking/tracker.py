from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.lane_detection.interfaces import LaneDetectionResult

logger = logging.getLogger(__name__)


def _get_bottom_x(points: list[tuple[int, int]]) -> float | None:
    """Return the x-coordinate at the lowest (largest y) point."""
    if not points:
        return None
    max_y = max(p[1] for p in points)
    for x, y in points:
        if y == max_y:
            return float(x)
    return None


def _get_top_x(points: list[tuple[int, int]]) -> float | None:
    """Return the x-coordinate at the highest (smallest y) point."""
    if not points:
        return None
    min_y = min(p[1] for p in points)
    for x, y in points:
        if y == min_y:
            return float(x)
    return None


def _smooth_points(
    new_points: list[tuple[int, int]],
    old_points: list[tuple[int, int]],
    alpha: float,
) -> list[tuple[int, int]]:
    """Apply EMA smoothing per y-level.

    For each y in *new_points*, smooth its x with the previous value
    at the same y from *old_points* (if it exists).  y-values not
    present in *new_points* are dropped from the output so that the
    lane y-range always reflects the current detection.
    """
    old_dict = {y: x for x, y in old_points}
    result: list[tuple[int, int]] = []
    for x, y in new_points:
        if y in old_dict:
            sx = alpha * x + (1.0 - alpha) * old_dict[y]
        else:
            sx = float(x)
        result.append((int(round(sx)), y))
    return result


def _extend_lane(
    points: list[tuple[int, int]],
    img_w: int,
    extend_above: int = 40,
    min_allowed_y: int = 350,
    n_slope_points: int = 4,
) -> list[tuple[int, int]]:
    """Extend lane upward using local linear slope (not quadratic).

    Uses the top *n_slope_points* to fit a line ``x = m*y + b``, then
    extends upward by *extend_above* pixels (clamped to
    *min_allowed_y*).

    This is far more stable than polynomial extrapolation because the
    linear term dominates near the horizon.
    """
    if len(points) < n_slope_points:
        return list(points)

    sorted_pts = sorted(points, key=lambda p: p[1], reverse=True)
    top_segment = sorted_pts[-n_slope_points:]
    xs = np.array([p[0] for p in top_segment], dtype=np.float64)
    ys = np.array([p[1] for p in top_segment], dtype=np.float64)
    m, b = np.polyfit(ys, xs, 1)

    top_y = sorted_pts[-1][1]
    new_y = top_y - extend_above
    new_y = max(new_y, min_allowed_y)

    if new_y >= top_y:
        return list(points)

    new_x = int(round(np.clip(m * new_y + b, 0, img_w - 1)))
    extended = list(points) + [(new_x, new_y)]
    extended.sort(key=lambda p: p[1], reverse=True)
    return extended


class PointLaneTracker:
    """Point-based lane tracker without polynomial coefficients.

    Tracks left/right lane polylines directly.
    Uses EMA smoothing per y-level, lane memory on disappearance,
    bottom-x association, and several geometry validation gates.

    All public API contracts (``LaneDetectionResult``) are preserved.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._alpha = float(config.get("smoothing_alpha", 0.25))
        self._max_missed = int(config.get("max_missing_frames", 5))
        self._conf_threshold = float(config.get("confidence_threshold", 0.5))
        self._min_points = int(config.get("min_points", 6))
        self._max_bottom_jump = float(config.get("max_bottom_jump", 80.0))
        self._max_geometry_delta = float(
            config.get("max_geometry_delta", 600.0)
        )
        self._extend_above = int(config.get("extend_above_pixels", 40))
        self._min_allowed_y = int(config.get("min_allowed_y", 350))

        # Per-track state: smoothed points, missed counter
        self._left_points: list[tuple[int, int]] = []
        self._right_points: list[tuple[int, int]] = []
        self._left_missed: int = 0
        self._right_missed: int = 0
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        result: LaneDetectionResult | None,
        img_h: int,
        img_w: int,
    ) -> LaneDetectionResult | None:
        """Process a detector result and return a stabilised result.

        Args:
            result: Raw ``LaneDetectionResult`` from the detector.
            img_h: Image height (used for geometry constraints).
            img_w: Image width (for coordinate clipping).

        Returns:
            Stabilised ``LaneDetectionResult``, or ``None`` if tracking
            is not active (input was ``None``).
        """
        self._frame_count += 1

        if result is None:
            return None

        raw_left = list(result.left_lane_points)
        raw_right = list(result.right_lane_points)
        self._det_conf = result.confidence
        det_conf = self._det_conf

        # --- Step A: confidence gate ---
        if det_conf < self._conf_threshold:
            logger.debug(
                "Frame %d: confidence %.3f below threshold %.3f — "
                "rejecting all lanes",
                self._frame_count,
                det_conf,
                self._conf_threshold,
            )
            raw_left = []
            raw_right = []

        # --- Step B: validate each raw lane independently ---
        raw_left = self._validate_lane(raw_left, "left")
        raw_right = self._validate_lane(raw_right, "right")

        # --- Step C: check for crossing ---
        if raw_left and raw_right:
            lb = _get_bottom_x(raw_left)
            rb = _get_bottom_x(raw_right)
            if lb is not None and rb is not None and lb >= rb:
                logger.warning(
                    "Frame %d: left/right crossing "
                    "(left_bottom_x=%.1f >= right_bottom_x=%.1f) — rejecting both",
                    self._frame_count,
                    lb,
                    rb,
                )
                raw_left = []
                raw_right = []

        # --- Step D: sudden-jump check (reject if bottom_x changed too much) ---
        if raw_left and self._left_points:
            new_bx = _get_bottom_x(raw_left)
            old_bx = _get_bottom_x(self._left_points)
            if (
                new_bx is not None
                and old_bx is not None
                and abs(new_bx - old_bx) > self._max_bottom_jump
            ):
                logger.warning(
                    "Frame %d: left lane sudden jump "
                    "(bottom_x %.1f -> %.1f) — rejecting detection",
                    self._frame_count,
                    old_bx,
                    new_bx,
                )
                raw_left = []

        if raw_right and self._right_points:
            new_bx = _get_bottom_x(raw_right)
            old_bx = _get_bottom_x(self._right_points)
            if (
                new_bx is not None
                and old_bx is not None
                and abs(new_bx - old_bx) > self._max_bottom_jump
            ):
                logger.warning(
                    "Frame %d: right lane sudden jump "
                    "(bottom_x %.1f -> %.1f) — rejecting detection",
                    self._frame_count,
                    old_bx,
                    new_bx,
                )
                raw_right = []

        # --- Track left lane ---
        if raw_left:
            self._left_points = _smooth_points(raw_left, self._left_points, self._alpha)
            self._left_missed = 0
        else:
            self._left_missed += 1
            if self._left_missed > self._max_missed:
                logger.debug(
                    "Frame %d: left lane expired after %d missed frames",
                    self._frame_count,
                    self._max_missed,
                )
                self._left_points = []

        # --- Track right lane ---
        if raw_right:
            self._right_points = _smooth_points(
                raw_right, self._right_points, self._alpha,
            )
            self._right_missed = 0
        else:
            self._right_missed += 1
            if self._right_missed > self._max_missed:
                logger.debug(
                    "Frame %d: right lane expired after %d missed frames",
                    self._frame_count,
                    self._max_missed,
                )
                self._right_points = []

        # --- Step F: extend lane vertical extent for output ---
        left_out = _extend_lane(
            self._left_points, img_w, self._extend_above, self._min_allowed_y,
        )
        right_out = _extend_lane(
            self._right_points, img_w, self._extend_above, self._min_allowed_y,
        )

        return self._build_result(left_out, right_out, img_w)

    def reset(self) -> None:
        """Clear all tracked lanes (e.g. on scene change)."""
        self._left_points.clear()
        self._right_points.clear()
        self._left_missed = 0
        self._right_missed = 0
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_lane(
        self,
        points: list[tuple[int, int]],
        side: str,
    ) -> list[tuple[int, int]]:
        """Return ``points`` if they pass geometric validation, else ``[]``.

        Checks performed:
          * minimum point count
          * ``|bottom_x - top_x|`` within ``_max_geometry_delta``
        """
        if len(points) < self._min_points:
            logger.debug(
                "Frame %d: %s lane rejected — %d points < %d required",
                self._frame_count,
                side,
                len(points),
                self._min_points,
            )
            return []

        bx = _get_bottom_x(points)
        tx = _get_top_x(points)
        if bx is not None and tx is not None:
            delta = abs(bx - tx)
            if delta > self._max_geometry_delta:
                logger.debug(
                    "Frame %d: %s lane rejected — geometry delta %.0f > %.0f",
                    self._frame_count,
                    side,
                    delta,
                    self._max_geometry_delta,
                )
                return []

        return points

    def _build_result(
        self,
        left_pts: list[tuple[int, int]],
        right_pts: list[tuple[int, int]],
        img_w: int,
    ) -> LaneDetectionResult:
        """Build ``LaneDetectionResult`` from the two tracked point sets."""
        left_t = tuple(left_pts)
        right_t = tuple(right_pts)

        # Centre lane
        center_lane: tuple[tuple[int, int], ...] = ()
        if left_t and right_t:
            left_sorted = sorted(left_t, key=lambda p: p[1])
            right_sorted = sorted(right_t, key=lambda p: p[1])
            min_len = min(len(left_sorted), len(right_sorted))
            if min_len >= 2:
                center_lane = tuple(
                    (
                        (left_sorted[i][0] + right_sorted[i][0]) // 2,
                        (left_sorted[i][1] + right_sorted[i][1]) // 2,
                    )
                    for i in range(min_len)
                )

        # Ego lane centre
        ego_x: float | None = None
        if left_t and right_t:
            max_y = max(p[1] for p in left_t + right_t)
            lx_candidates = [p[0] for p in left_t if p[1] == max_y]
            rx_candidates = [p[0] for p in right_t if p[1] == max_y]
            if lx_candidates and rx_candidates:
                ego_x = (float(lx_candidates[0]) + float(rx_candidates[0])) / 2.0

        lane_count = (1 if left_t else 0) + (1 if right_t else 0)
        conf = self._det_conf if lane_count > 0 else 0.0

        return LaneDetectionResult(
            left_lane_points=left_t,
            right_lane_points=right_t,
            center_lane_points=center_lane,
            lane_count=lane_count,
            ego_lane_center_x=ego_x,
            confidence=conf,
        )
