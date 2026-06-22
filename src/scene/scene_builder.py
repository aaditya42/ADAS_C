from __future__ import annotations

import math
from typing import Any, Optional

from src.lane_detection.interfaces import LaneDetectionResult
from src.schemas import Frame, PerceivedObject
from src.schemas.scene_context import SceneContext
from src.schemas.traffic_sign_info import SignType, TrafficSignInfo


def _interpolate_x_at_y(
    points: tuple[tuple[int, int], ...],
    target_y: float,
) -> float | None:
    """Interpolate the x-coordinate at the given y from a list of (x, y) points."""
    if len(points) < 2:
        return None
    sorted_pts = sorted(points, key=lambda p: p[1])
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


class SceneBuilder:
    """Synthesises a ``SceneContext`` from perceived objects, lanes, and signs.

    Sign latching: once a sign is detected it remains active for
    ``sign_latch_frames`` frames even if not currently visible. This
    provides temporal persistence for the ADAS brain.

    Args:
        sign_latch_frames: Number of frames a sign persists after last seen.
    """

    def __init__(self, sign_latch_frames: int = 30) -> None:
        self._latch_frames = sign_latch_frames
        self._latches: dict[str, Any] = {}  # sign_type -> {"frame": int, "value": float | None}

    def build(
        self,
        perceived: list[PerceivedObject],
        lane_info: LaneDetectionResult | None,
        signs: list[TrafficSignInfo],
        frame: Frame,
    ) -> SceneContext:
        self._update_latches(signs, frame)
        self._purge_stale(frame.frame_id)

        active_speed_limit = self._get_latched_value(SignType.SPEED_LIMIT)
        stop_required = self._is_latched(SignType.STOP)
        yield_required = self._is_latched(SignType.YIELD)
        traffic_light_state = self._resolve_traffic_light()

        in_lane_objects = self._filter_ego_lane(perceived, lane_info, frame)
        ego_lane_object_count = len(in_lane_objects)

        nearest_dist = None
        nearest_tid = None
        if in_lane_objects:
            nearest = min(in_lane_objects, key=lambda o: o.distance if o.depth_confidence > 0 else math.inf)
            if nearest.depth_confidence > 0:
                nearest_dist = nearest.distance
                nearest_tid = nearest.track_id

        crosswalk_active = self._is_latched(SignType.PEDESTRIAN_CROSSING)
        crosswalk_ped = crosswalk_active and any(
            o.class_name == "person"
            for o in in_lane_objects
        )

        return SceneContext(
            active_speed_limit=active_speed_limit,
            stop_required=stop_required,
            yield_required=yield_required,
            traffic_light_state=traffic_light_state,
            ego_lane_object_count=ego_lane_object_count,
            nearest_in_lane_distance_m=nearest_dist,
            nearest_in_lane_track_id=nearest_tid,
            crosswalk_pedestrian=crosswalk_ped,
            timestamp=frame.timestamp,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _update_latches(self, signs: list[TrafficSignInfo], frame: Frame) -> None:
        """Record every detected sign in the current frame."""
        for sign in signs:
            key = sign.sign_type.value
            self._latches[key] = {
                "frame": frame.frame_id,
                "value": sign.value,
            }

    def _purge_stale(self, current_frame: int) -> None:
        """Remove latches older than ``sign_latch_frames``."""
        stale = [
            k for k, v in self._latches.items()
            if (current_frame - v["frame"]) >= self._latch_frames
        ]
        for k in stale:
            del self._latches[k]

    def _is_latched(self, sign_type: SignType) -> bool:
        return sign_type.value in self._latches

    def _get_latched_value(self, sign_type: SignType) -> Optional[float]:
        entry = self._latches.get(sign_type.value)
        if entry is None:
            return None
        return entry["value"]

    def _resolve_traffic_light(self) -> str:
        """Return the most recent traffic light colour, or ``"none"``."""
        order = (SignType.TRAFFIC_LIGHT_RED, SignType.TRAFFIC_LIGHT_YELLOW, SignType.TRAFFIC_LIGHT_GREEN)
        for st in order:
            if st.value in self._latches:
                return st.value.split("_")[-1]
        return "none"

    @staticmethod
    def _filter_ego_lane(
        perceived: list[PerceivedObject],
        lane_info: LaneDetectionResult | None,
        frame: Frame,
    ) -> list[PerceivedObject]:
        """Return objects whose bbox-bottom-centre falls in the ego lane.

        Supports both the new point-based ``LaneDetectionResult`` and legacy
        polynomial-based ``LaneInfo``.
        """
        if not perceived:
            return []

        h, w = frame.image.shape[:2]

        if lane_info is not None:
            # New schema: point-based lanes
            if hasattr(lane_info, "left_lane_points") and hasattr(lane_info, "right_lane_points"):
                left_pts = lane_info.left_lane_points
                right_pts = lane_info.right_lane_points
                if left_pts and right_pts:
                    in_lane: list[PerceivedObject] = []
                    for obj in perceived:
                        by = float(obj.bbox.y2)
                        lx = _interpolate_x_at_y(left_pts, by)
                        rx = _interpolate_x_at_y(right_pts, by)
                        if lx is None or rx is None:
                            continue
                        cx = (obj.bbox.x1 + obj.bbox.x2) / 2.0
                        if min(lx, rx) <= cx <= max(lx, rx):
                            in_lane.append(obj)
                    return in_lane

            # Legacy schema: polynomial-based lanes
            if lane_info.left_lane is not None and lane_info.right_lane is not None:
                left_line = lane_info.left_lane
                right_line = lane_info.right_lane

                def _lane_x(line: Any, y: float) -> float:
                    c = line.coeffs
                    return c[0] * y * y + c[1] * y + c[2]

                in_lane_legacy: list[PerceivedObject] = []
                for obj in perceived:
                    by = float(obj.bbox.y2)
                    lx = _lane_x(left_line, by)
                    rx = _lane_x(right_line, by)
                    cx = (obj.bbox.x1 + obj.bbox.x2) / 2.0
                    if min(lx, rx) <= cx <= max(lx, rx):
                        in_lane_legacy.append(obj)
                return in_lane_legacy

        # Fallback: central 50 % of image width
        cw = w * 0.25
        cx_img = w / 2.0
        in_lane_fb: list[PerceivedObject] = []
        for obj in perceived:
            cx = (obj.bbox.x1 + obj.bbox.x2) / 2.0
            if abs(cx - cx_img) <= cw:
                in_lane_fb.append(obj)
        return in_lane_fb
