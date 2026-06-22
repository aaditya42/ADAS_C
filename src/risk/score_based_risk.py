from __future__ import annotations

import math
from typing import Any

from src.schemas import PerceivedObject
from src.schemas.risk_object import RiskLevel, RiskObject, WarningLevel
from src.risk.base import BaseRiskEngine


def _interpolate_x_at_y(
    points: tuple[tuple[int, int], ...],
    target_y: float,
) -> float | None:
    """Interpolate the x-coordinate at the given y from a list of (x, y) points.

    Uses linear interpolation between the two nearest points spanning ``target_y``.
    Returns ``None`` if no spanning pair exists.
    """
    if len(points) < 2:
        return None

    # Sort by y ascending
    sorted_pts = sorted(points, key=lambda p: p[1])

    # Clamp to min/max y
    if target_y <= sorted_pts[0][1]:
        return float(sorted_pts[0][0])
    if target_y >= sorted_pts[-1][1]:
        return float(sorted_pts[-1][0])

    for i in range(len(sorted_pts) - 1):
        y0, y1 = float(sorted_pts[i][1]), float(sorted_pts[i + 1][1])
        if y0 <= target_y <= y1:
            x0, x1 = float(sorted_pts[i][0]), float(sorted_pts[i + 1][0])
            t = (target_y - y0) / (y1 - y0) if y1 != y0 else 0.0
            return x0 + t * (x1 - x0)

    return None


class ScoreBasedRiskEngine(BaseRiskEngine):
    """Score-based risk engine (0–100) with per-contributor weights.

    Each contributor (distance, TTC, speed, lane, scene) produces a
    sub-score in [0, 100]. The final score is a weighted sum renormalised
    by the sum of active weights.

    Args:
        weights: Per-contributor weights (must sum to 1.0 collectively
            but lane/scene default to 0).
        dist_critical_m: Distance at/below which score = 100.
        dist_safe_m: Distance at/above which score = 0.
        ttc_critical_s: TTC at/below which score = 100.
        ttc_safe_s: TTC at/above which score = 0.
        speed_critical_mps: Approaching speed at/above which score = 100.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        dist_critical_m: float = 5.0,
        dist_safe_m: float = 50.0,
        ttc_critical_s: float = 1.0,
        ttc_safe_s: float = 6.0,
        speed_critical_mps: float = 15.0,
    ) -> None:
        self._w = weights or {
            "distance": 0.4,
            "ttc": 0.5,
            "speed": 0.1,
            "lane": 0.0,
            "scene": 0.0,
        }
        self._dist_crit = dist_critical_m
        self._dist_safe = dist_safe_m
        self._ttc_crit = ttc_critical_s
        self._ttc_safe = ttc_safe_s
        self._speed_crit = speed_critical_mps

    # ------------------------------------------------------------------ #
    #  Per-contributor scoring (each returns 0..100)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_distance(d: float | None, crit: float, safe: float) -> float:
        if d is None:
            return 0.0
        if d <= crit:
            return 100.0
        if d >= safe:
            return 0.0
        return (1.0 - (d - crit) / (safe - crit)) * 100.0

    @staticmethod
    def _score_ttc(t: float | None, crit: float, safe: float) -> float:
        if t is None or not math.isfinite(t):
            return 0.0
        if t <= crit:
            return 100.0
        if t >= safe:
            return 0.0
        return (1.0 - (t - crit) / (safe - crit)) * 100.0

    @staticmethod
    def _score_speed(s: float | None, crit: float) -> float:
        if s is None or s <= 0:
            return 0.0
        if s >= crit:
            return 100.0
        return (s / crit) * 100.0

    @staticmethod
    def _score_lane(lane_info: Any, obj: PerceivedObject) -> float:
        """Lane-departure risk based on bbox horizontal offset from lane center.

        Supports both:
          - New ``LaneDetectionResult`` (with ``left_lane_points`` / ``right_lane_points``)
          - Legacy ``LaneInfo`` (with ``left_lane`` / ``right_lane`` + ``coeffs``)

        Returns 0 if no lane information or both lane boundaries missing.
        """
        if lane_info is None:
            return 0.0

        bbox_cx = (obj.bbox.x1 + obj.bbox.x2) / 2.0
        by = float(obj.bbox.y2)

        # New schema: point-based lanes
        if hasattr(lane_info, "left_lane_points"):
            left_pts = lane_info.left_lane_points
            right_pts = lane_info.right_lane_points
            if not left_pts or not right_pts:
                return 0.0

            lx = _interpolate_x_at_y(left_pts, by)
            rx = _interpolate_x_at_y(right_pts, by)
            if lx is None or rx is None:
                return 0.0

            lane_center_x = (lx + rx) / 2.0
            lane_half_w = abs(rx - lx) / 2.0
            if lane_half_w < 1.0:
                return 0.0

            offset = abs(bbox_cx - lane_center_x)
            return min(100.0, (offset / lane_half_w) * 50.0)

        # Legacy schema: polynomial-based lanes
        left = lane_info.left_lane
        right = lane_info.right_lane
        if left is None or right is None:
            return 0.0

        lx = left.coeffs[0] * by * by + left.coeffs[1] * by + left.coeffs[2]
        rx = right.coeffs[0] * by * by + right.coeffs[1] * by + right.coeffs[2]

        lane_center_x = (lx + rx) / 2.0
        lane_half_w = abs(rx - lx) / 2.0
        if lane_half_w < 1.0:
            return 0.0

        offset = abs(bbox_cx - lane_center_x)
        return min(100.0, (offset / lane_half_w) * 50.0)

    @staticmethod
    def _score_scene(scene_context: Any, obj: PerceivedObject) -> float:
        """Phase 14 scene-aware risk scoring.

        - +20 if this object is the nearest in-lane vehicle ahead.
        - +25 if a stop is required and this object is in-lane within 15 m.
        - +30 if traffic light is red and this object is the ego-lane vehicle ahead.
        - Capped at 100.
        """
        if scene_context is None:
            return 0.0

        score = 0.0

        if scene_context.nearest_in_lane_track_id == obj.track_id:
            score += 20.0

        if (
            scene_context.stop_required
            and scene_context.nearest_in_lane_track_id == obj.track_id
            and obj.distance is not None
            and obj.depth_confidence > 0
            and obj.distance < 15.0
        ):
            score += 25.0

        if (
            scene_context.traffic_light_state == "red"
            and scene_context.nearest_in_lane_track_id == obj.track_id
        ):
            score += 30.0

        return min(score, 100.0)

    # ------------------------------------------------------------------ #
    #  Level mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _level_from_score(score: float) -> RiskLevel:
        if score < 25:
            return RiskLevel.SAFE
        if score < 50:
            return RiskLevel.LOW
        if score < 75:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    @staticmethod
    def _warning_from_risk(level: RiskLevel) -> WarningLevel:
        mapping = {
            RiskLevel.SAFE: WarningLevel.NONE,
            RiskLevel.LOW: WarningLevel.ADVISORY,
            RiskLevel.MEDIUM: WarningLevel.CAUTION,
            RiskLevel.HIGH: WarningLevel.CRITICAL,
        }
        return mapping[level]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        perceived: list[PerceivedObject],
        motion: dict[int, Any],
        ttc: dict[int, float | None],
        lane_info: Any = None,
        scene_context: Any = None,
    ) -> list[RiskObject]:
        active_contributors = [
            k for k, v in self._w.items() if k in ("distance", "ttc", "speed") or v > 0
        ]
        denom = sum(self._w[k] for k in active_contributors)
        if denom == 0:
            denom = 1.0

        results: list[RiskObject] = []

        for obj in perceived:
            tid = obj.track_id
            mot = motion.get(tid)

            d = obj.distance if obj.depth_confidence > 0 else None
            t = ttc.get(tid)
            s = mot.speed_mps if mot is not None else None

            score = 0.0
            if "distance" in active_contributors:
                score += self._w["distance"] * self._score_distance(
                    d, self._dist_crit, self._dist_safe
                )
            if "ttc" in active_contributors:
                score += self._w["ttc"] * self._score_ttc(
                    t, self._ttc_crit, self._ttc_safe
                )
            if "speed" in active_contributors:
                score += self._w["speed"] * self._score_speed(s, self._speed_crit)
            if "lane" in active_contributors:
                score += self._w["lane"] * self._score_lane(lane_info, obj)
            if "scene" in active_contributors:
                score += self._w["scene"] * self._score_scene(scene_context, obj)

            score = max(0.0, min(100.0, score / denom))

            level = self._level_from_score(score)
            warning = self._warning_from_risk(level)

            results.append(RiskObject(
                track_id=tid,
                risk_level=level,
                risk_score=round(score, 1),
                warning_level=warning,
            ))

        return results
