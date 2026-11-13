"""Direction estimation component for forward-facing monocular ADAS.

Determines camera-relative spatial direction of threats using bounding box
footprints, polynomial lane boundaries, and lateral motion vectors with
temporal stability filtering to prevent boundary jitter.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

from src.schemas.bbox import BoundingBox
from src.threat.schemas import CameraDirection, LaneRelationship
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _interpolate_x_at_y(
    points: Sequence[tuple[int | float, int | float]],
    target_y: float,
) -> float | None:
    """Interpolate x-coordinate at target_y from sequence of (x, y) points."""
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


def _get_lane_boundaries(
    lane_info: Any,
    by: float,
) -> tuple[float | None, float | None]:
    """Retrieve left and right lane boundaries at vertical coordinate by."""
    if lane_info is None:
        return None, None

    if hasattr(lane_info, "left_lane_points") and hasattr(lane_info, "right_lane_points"):
        left_pts = lane_info.left_lane_points
        right_pts = lane_info.right_lane_points
        if not left_pts or not right_pts:
            return None, None
        lx = _interpolate_x_at_y(left_pts, by)
        rx = _interpolate_x_at_y(right_pts, by)
        return lx, rx

    left = getattr(lane_info, "left_lane", None)
    right = getattr(lane_info, "right_lane", None)
    if left is not None and right is not None and hasattr(left, "coeffs") and hasattr(right, "coeffs"):
        lx = float(left.coeffs[0] * by * by + left.coeffs[1] * by + left.coeffs[2])
        rx = float(right.coeffs[0] * by * by + right.coeffs[1] * by + right.coeffs[2])
        return lx, rx

    return None, None


class DirectionEstimator:
    """Estimates camera-relative spatial direction for road obstacles.

    Zones:
    - FRONT: Within or directly leading the ego travel path.
    - FRONT_LEFT: Forward left sector (left adjacent lane / verge).
    - FRONT_RIGHT: Forward right sector (right adjacent lane / verge).
    - LEFT: Lateral optical periphery alongside ego vehicle on the left.
    - RIGHT: Lateral optical periphery alongside ego vehicle on the right.
    - UNKNOWN: Insufficient geometric evidence.

    Forward-Camera Limitations:
    - Zero visibility into blind spots or rear quadrants. REAR_LEFT, REAR_RIGHT,
      and Blind Spot Monitoring (BSM) are physically unsupported.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        threat_cfg = self.config.get("threat_assessment", {})
        dir_cfg = threat_cfg.get("direction", {})

        self.enabled: bool = bool(dir_cfg.get("enabled", True))
        self.min_confirmation_frames: int = int(dir_cfg.get("min_confirmation_frames", 2))
        self.stale_timeout_seconds: float = float(threat_cfg.get("stale_timeout_seconds", 2.0))

        # Temporal history of candidate directions: track_id -> deque[CameraDirection]
        self._history: dict[int, deque[CameraDirection]] = {}
        self._confirmed: dict[int, CameraDirection] = {}
        self._last_seen: dict[int, float] = {}

    def reset(self) -> None:
        """Clear internal direction history."""
        self._history.clear()
        self._confirmed.clear()
        self._last_seen.clear()

    def evict_stale_tracks(self, current_time: float, stale_timeout_s: float | None = None) -> None:
        """Evict history for tracks not updated within stale_timeout_seconds."""
        timeout = stale_timeout_s if stale_timeout_s is not None else self.stale_timeout_seconds
        stale_ids = [
            tid
            for tid, t in self._last_seen.items()
            if (current_time - t) > timeout or (current_time < t)
        ]
        for tid in stale_ids:
            self._history.pop(tid, None)
            self._confirmed.pop(tid, None)
            self._last_seen.pop(tid, None)

    def purge_stale_tracks(self, current_timestamp: float, stale_timeout_s: float | None = None) -> None:
        """Alias for evict_stale_tracks for interface convenience."""
        self.evict_stale_tracks(current_timestamp, stale_timeout_s=stale_timeout_s)

    def get_direction(self, track_id: int) -> CameraDirection:
        """Return the confirmed direction for track_id, or UNKNOWN."""
        return self._confirmed.get(track_id, CameraDirection.UNKNOWN)

    def estimate(
        self,
        track_id: int,
        bbox: BoundingBox,
        lane_info: Any = None,
        lane_relationship: LaneRelationship = LaneRelationship.UNKNOWN,
        img_width: float = 1280.0,
        img_height: float = 720.0,
        timestamp: float = 0.0,
    ) -> CameraDirection:
        """Estimate temporally stabilized camera-relative direction for an obstacle.

        Args:
            track_id: Persistent track identity.
            bbox: Obstacle bounding box.
            lane_info: Lane geometry result or polynomial coefficients.
            lane_relationship: Extracted spatial relationship from Stage 2.
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            timestamp: Video presentation timestamp.

        Returns:
            Stabilized CameraDirection enum.
        """
        if not self.enabled:
            return CameraDirection.UNKNOWN

        self.evict_stale_tracks(timestamp)
        self._last_seen[track_id] = timestamp

        # Compute candidate direction for current frame
        candidate = self._compute_instantaneous_direction(
            bbox=bbox,
            lane_info=lane_info,
            lane_relationship=lane_relationship,
            img_width=img_width,
            img_height=img_height,
        )

        # Update candidate buffer
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=5)
        self._history[track_id].append(candidate)

        # Stabilize direction against frame-to-frame boundary jitter
        prev = self._confirmed.get(track_id)
        if prev is None:
            self._confirmed[track_id] = candidate
            return candidate

        # Count occurrences of candidate in recent history
        matches = sum(1 for d in self._history[track_id] if d == candidate)
        if candidate == prev or matches >= self.min_confirmation_frames:
            self._confirmed[track_id] = candidate
            return candidate
        else:
            # Retain previous direction during transient 1-frame boundary fluctuation
            return prev

    def _compute_instantaneous_direction(
        self,
        bbox: BoundingBox | None,
        lane_info: Any,
        lane_relationship: LaneRelationship,
        img_width: float,
        img_height: float,
    ) -> CameraDirection:
        """Compute raw single-frame direction before temporal filtering."""
        if bbox is None or not hasattr(bbox, "width") or not hasattr(bbox, "height"):
            return CameraDirection.UNKNOWN

        if bbox.width <= 0 or bbox.height <= 0:
            return CameraDirection.UNKNOWN

        cx = (bbox.x1 + bbox.x2) / 2.0
        by = float(bbox.y2)

        # 1. Check lateral optical periphery (alongside ego vehicle)
        if cx <= 0.15 * img_width and by > 0.50 * img_height:
            return CameraDirection.LEFT
        elif cx >= 0.85 * img_width and by > 0.50 * img_height:
            return CameraDirection.RIGHT

        # 2. Check lane-aware geometry if available
        lx, rx = _get_lane_boundaries(lane_info, by)
        lane_conf = getattr(lane_info, "confidence", 1.0) if lane_info else 0.0

        if lx is not None and rx is not None and rx > lx and lane_conf >= 0.20:
            lane_width = rx - lx
            # Corridor check with slight margin (10% of lane width)
            corridor_left = lx - 0.10 * lane_width
            corridor_right = rx + 0.10 * lane_width

            if corridor_left <= cx <= corridor_right or lane_relationship == LaneRelationship.EGO_LANE:
                return CameraDirection.FRONT
            elif cx < corridor_left:
                return CameraDirection.FRONT_LEFT
            else:
                return CameraDirection.FRONT_RIGHT

        # 3. Geometric fallback when lane geometry is missing or unconfident
        # Optical FOV center corridor: [40%, 60%] of image width
        center_left = 0.40 * img_width
        center_right = 0.60 * img_width

        if lane_relationship == LaneRelationship.EGO_LANE or (center_left <= cx <= center_right):
            return CameraDirection.FRONT
        elif cx < center_left:
            return CameraDirection.FRONT_LEFT
        else:
            return CameraDirection.FRONT_RIGHT
