"""Per-object threat feature extraction engine.

Extracts kinematic, spatial, behavioral, and conflict indicators for each
tracked road user from existing perception outputs, assigning physical
validity metadata to every indicator.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

from src.schemas.bbox import BoundingBox
from src.schemas.perceived_object import PerceivedObject
from src.schemas.frame import Frame
from src.schemas.motion import RelativeMotion
from src.schemas.trajectory import TrajectoryPrediction
from src.trajectory.drac import calculate_drac
from src.threat.schemas import (
    CameraDirection,
    IndicatorValidity,
    LaneRelationship,
    ThreatFeatures,
    TrajectoryConflictState,
    ValidityStatus,
    VulnerabilityLevel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _interpolate_x_at_y(
    points: Sequence[tuple[int | float, int | float]],
    target_y: float,
) -> float | None:
    """Interpolate x coordinate at target_y from sequence of (x, y) points."""
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


def _get_lane_boundaries_at_y(
    lane_info: Any,
    y: float,
) -> tuple[float | None, float | None]:
    """Retrieve left and right lane x-coordinates at target y."""
    if lane_info is None:
        return None, None

    # Point-based lane detection result
    if hasattr(lane_info, "left_lane_points") and hasattr(lane_info, "right_lane_points"):
        left_pts = lane_info.left_lane_points
        right_pts = lane_info.right_lane_points
        if not left_pts or not right_pts:
            return None, None
        lx = _interpolate_x_at_y(left_pts, y)
        rx = _interpolate_x_at_y(right_pts, y)
        return lx, rx

    # Legacy polynomial-based lane info
    left = getattr(lane_info, "left_lane", None)
    right = getattr(lane_info, "right_lane", None)
    if left is not None and right is not None and hasattr(left, "coeffs") and hasattr(right, "coeffs"):
        lx = float(left.coeffs[0] * y * y + left.coeffs[1] * y + left.coeffs[2])
        rx = float(right.coeffs[0] * y * y + right.coeffs[1] * y + right.coeffs[2])
        return lx, rx

    return None, None


class ThreatFeatureExtractor:
    """Extracts comprehensive threat features from perception and motion outputs.

    Ensures that every risk indicator carries explicit physical validity metadata,
    distinguishing metric units from optical proxies, and preventing silent conversion
    of unavailable measurements into zeros.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        threat_cfg = self.config.get("threat_assessment", {})
        feat_cfg = threat_cfg.get("features", {})

        # Configuration parameters
        self.enabled: bool = bool(threat_cfg.get("enabled", True))
        self.stale_timeout_seconds: float = float(
            threat_cfg.get("stale_timeout_seconds", 2.0)
        )
        self.id_reuse_gap_seconds: float = float(
            threat_cfg.get("id_reuse_gap_seconds", 2.0)
        )
        self.min_history_points: int = int(feat_cfg.get("min_history_points", 3))
        self.history_window_seconds: float = float(
            feat_cfg.get("history_window_seconds", 1.5)
        )

        # Check metric calibration status
        dist_cfg = self.config.get("distance", {})
        self.is_metric: bool = dist_cfg.get("type", "relative") == "calibrated"

        # Per-track internal history
        self._track_first_seen: dict[int, float] = {}
        self._track_last_seen: dict[int, float] = {}
        self._track_frames: dict[int, int] = {}
        self._track_classes: dict[int, str] = {}
        self._prev_lane_rel: dict[int, LaneRelationship] = {}

        # Kinematic time series: (timestamp, value)
        self._bbox_history: dict[int, deque[tuple[float, BoundingBox]]] = {}
        self._speed_history: dict[int, deque[tuple[float, float]]] = {}
        self._accel_history: dict[int, deque[tuple[float, float]]] = {}

    def reset(self) -> None:
        """Clear all internal tracking histories."""
        self._track_first_seen.clear()
        self._track_last_seen.clear()
        self._track_frames.clear()
        self._track_classes.clear()
        self._prev_lane_rel.clear()
        self._bbox_history.clear()
        self._speed_history.clear()
        self._accel_history.clear()

    def evict_stale_tracks(self, current_timestamp: float) -> None:
        """Purge tracks not seen within stale_timeout_seconds."""
        stale_ids = [
            tid
            for tid, last_t in self._track_last_seen.items()
            if (current_timestamp - last_t) > self.stale_timeout_seconds
            or (current_timestamp < last_t)  # Video rewound
        ]
        for tid in stale_ids:
            self._track_first_seen.pop(tid, None)
            self._track_last_seen.pop(tid, None)
            self._track_frames.pop(tid, None)
            self._track_classes.pop(tid, None)
            self._prev_lane_rel.pop(tid, None)
            self._bbox_history.pop(tid, None)
            self._speed_history.pop(tid, None)
            self._accel_history.pop(tid, None)

    def extract(
        self,
        perceived_objects: list[PerceivedObject],
        motion_map: dict[int, RelativeMotion] | None = None,
        ttc_map: dict[int, float | None] | None = None,
        lane_info: Any = None,
        scene_context: Any = None,
        behavior_frame: Any = None,
        trajectories: list[TrajectoryPrediction] | dict[int, TrajectoryPrediction] | None = None,
        frame: Frame | None = None,
    ) -> dict[int, ThreatFeatures]:
        """Extract threat features for all tracked objects in current frame.

        Args:
            perceived_objects: Tracked objects with depth and distance.
            motion_map: Relative motion estimates per track ID.
            ttc_map: Time-to-collision estimates per track ID.
            lane_info: Lane geometry and boundary curves.
            scene_context: Environmental scene context (traffic light, stop signs).
            behavior_frame: VRU behavior states with crossing probabilities.
            trajectories: Trajectory predictions per track.
            frame: Current video frame with image and presentation timestamp.

        Returns:
            Dictionary mapping track_id to ThreatFeatures record.
        """
        if not self.enabled:
            return {}

        current_time = float(frame.timestamp) if frame is not None else 0.0
        if frame is not None and hasattr(frame, "image") and frame.image is not None and len(frame.image.shape) >= 2:
            img_h, img_w = float(frame.image.shape[0]), float(frame.image.shape[1])
        else:
            img_w, img_h = 1280.0, 720.0

        # Evict stale tracks
        self.evict_stale_tracks(current_time)

        # Index trajectories by track_id
        traj_map: dict[int, TrajectoryPrediction] = {}
        if trajectories is not None:
            if isinstance(trajectories, dict):
                traj_map = trajectories
            elif isinstance(trajectories, (list, tuple)):
                for t in trajectories:
                    if hasattr(t, "track_id"):
                        traj_map[t.track_id] = t

        # Index behavior states by track_id
        behavior_map: dict[int, Any] = {}
        if behavior_frame is not None and hasattr(behavior_frame, "states"):
            for s in behavior_frame.states:
                if hasattr(s, "track_id"):
                    behavior_map[s.track_id] = s

        motion_dict = motion_map or {}
        ttc_dict = ttc_map or {}

        features_by_track: dict[int, ThreatFeatures] = {}

        for obj in perceived_objects:
            tid = obj.track_id
            class_name = str(obj.class_name).lower()
            bbox = obj.bbox

            # Tracker ID reuse check: if track ID returns after long absence or class changed
            last_seen = self._track_last_seen.get(tid)
            prev_class = self._track_classes.get(tid)
            if last_seen is not None and (
                (current_time - last_seen) > self.id_reuse_gap_seconds
                or (prev_class is not None and prev_class != class_name)
            ):
                # Reset history for this reused ID
                self._track_first_seen[tid] = current_time
                self._track_frames[tid] = 0
                self._bbox_history.pop(tid, None)
                self._speed_history.pop(tid, None)
                self._accel_history.pop(tid, None)
                self._prev_lane_rel.pop(tid, None)

            # Update observation tracking
            if tid not in self._track_first_seen:
                self._track_first_seen[tid] = current_time
                self._track_frames[tid] = 1
            else:
                self._track_frames[tid] += 1
            self._track_last_seen[tid] = current_time
            self._track_classes[tid] = class_name

            obs_duration = max(0.0, current_time - self._track_first_seen[tid])
            obs_count = self._track_frames[tid]

            # Update histories
            if tid not in self._bbox_history:
                self._bbox_history[tid] = deque(maxlen=60)
                self._speed_history[tid] = deque(maxlen=60)
                self._accel_history[tid] = deque(maxlen=60)

            self._bbox_history[tid].append((current_time, bbox))

            # Maintain sliding window
            self._prune_history(self._bbox_history[tid], current_time)
            self._prune_history(self._speed_history[tid], current_time)
            self._prune_history(self._accel_history[tid], current_time)

            # Extract indicators
            ind_validity: dict[str, IndicatorValidity] = {}

            # 1. Motion & Kinematics
            rel_motion = motion_dict.get(tid)
            speed_mps, accel_mps2, motion_dir, speed_var, accel_noise = (
                self._extract_motion_features(
                    tid, rel_motion, current_time, ind_validity
                )
            )

            # 2. Looming Rate (Optical area expansion)
            looming_rate = self._compute_looming_rate(tid, ind_validity)

            # 3. Spatial & Lane Relationship
            (
                direction,
                lane_rel,
                lane_overlap,
                lat_dev,
            ) = self._extract_spatial_relationship(
                tid, bbox, lane_info, img_w, img_h, ind_validity
            )

            # 4. TTC (Time to Collision)
            raw_ttc = ttc_dict.get(tid)
            ttc_val = self._extract_ttc(raw_ttc, speed_mps, ind_validity)

            # 5. PET (Post-Encroachment Time)
            pet_val = self._extract_pet(ind_validity)

            # 6. DRAC (Deceleration Rate to Avoid Collision)
            drac_val = self._extract_drac(
                obj.distance, speed_mps, ind_validity
            )

            # 7. Trajectory Conflict
            traj_conflict, traj_conflict_time = self._extract_trajectory_conflict(
                tid, bbox, traj_map.get(tid), lane_info, ind_validity
            )

            # 8. Vulnerability & Behavior
            vulnerability = self._map_vulnerability(class_name)
            cross_prob, behavior_label = self._extract_behavior(
                tid, behavior_map.get(tid), ind_validity
            )

            # Build ThreatFeatures instance
            feat = ThreatFeatures(
                track_id=tid,
                tracker_id=tid,
                object_class=class_name,
                bbox=bbox,
                timestamp=current_time,
                direction=direction,
                lane_relationship=lane_rel,
                lane_overlap=lane_overlap,
                lateral_deviation=lat_dev,
                ttc=ttc_val,
                pet=pet_val,
                drac=drac_val,
                relative_speed=speed_mps,
                relative_acceleration=accel_mps2,
                motion_direction=motion_dir,
                speed_variation=speed_var,
                acceleration_noise=accel_noise,
                looming_rate=looming_rate,
                trajectory_conflict=traj_conflict,
                trajectory_conflict_time=traj_conflict_time,
                vulnerability=vulnerability,
                crossing_probability=cross_prob,
                behavior_label=behavior_label,
                observation_duration=obs_duration,
                observation_count=obs_count,
                is_metric=self.is_metric,
                indicator_validity=ind_validity,
            )

            features_by_track[tid] = feat

        return features_by_track

    def _prune_history(
        self, history: deque[tuple[float, Any]], current_time: float
    ) -> None:
        """Remove observations older than history_window_seconds."""
        cutoff = current_time - self.history_window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _extract_motion_features(
        self,
        track_id: int,
        motion: RelativeMotion | None,
        current_time: float,
        validity: dict[str, IndicatorValidity],
    ) -> tuple[float | None, float | None, str, float | None, float | None]:
        """Extract relative speed, acceleration, direction, and turbulence metrics."""
        if motion is None or motion.speed_mps is None:
            validity["relative_motion"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Insufficient track history snapshots for motion regression.",
            )
            return None, None, "unknown", None, None

        speed = float(motion.speed_mps)
        accel = float(motion.acceleration_mps2) if motion.acceleration_mps2 is not None else None
        m_dir = str(motion.direction)

        # Append to time-series history
        self._speed_history[track_id].append((current_time, speed))
        if accel is not None:
            self._accel_history[track_id].append((current_time, accel))

        # Compute speed variation (sample std dev)
        speeds = [s[1] for s in self._speed_history[track_id]]
        if len(speeds) >= self.min_history_points:
            mean_s = sum(speeds) / len(speeds)
            var_s = sum((x - mean_s) ** 2 for x in speeds) / (len(speeds) - 1)
            speed_var = math.sqrt(max(0.0, var_s))
        else:
            speed_var = None

        # Compute acceleration noise (RMS)
        accels = [a[1] for a in self._accel_history[track_id]]
        if len(accels) >= self.min_history_points:
            accel_noise = math.sqrt(sum(a ** 2 for a in accels) / len(accels))
        else:
            accel_noise = None

        status = ValidityStatus.VALID_METRIC if self.is_metric else ValidityStatus.VALID_PROXY
        reason = (
            "Calibrated metric closing speed and acceleration."
            if self.is_metric
            else "Proxy closing speed derived from relative depth rate."
        )
        validity["relative_motion"] = IndicatorValidity(
            status=status,
            value={"speed": speed, "accel": accel, "direction": m_dir},
            reason=reason,
        )

        return speed, accel, m_dir, speed_var, accel_noise

    def _compute_looming_rate(
        self, track_id: int, validity: dict[str, IndicatorValidity]
    ) -> float | None:
        """Calculate optical looming rate (1/A * dA/dt) from bbox expansion."""
        history = self._bbox_history.get(track_id)
        if not history or len(history) < 2:
            validity["looming_rate"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Insufficient bounding box observations for looming rate.",
            )
            return None

        t0, b0 = history[0]
        t1, b1 = history[-1]
        dt = t1 - t0
        if dt <= 0.05:
            validity["looming_rate"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Temporal delta too short for reliable looming computation.",
            )
            return None

        area0 = float(b0.width * b0.height)
        area1 = float(b1.width * b1.height)
        if area0 <= 1.0:
            validity["looming_rate"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Initial bounding box area too small.",
            )
            return None

        looming = (area1 - area0) / (area0 * dt)
        validity["looming_rate"] = IndicatorValidity(
            status=ValidityStatus.VALID_PROXY,
            value=looming,
            reason="Optical scale expansion rate (1/A * dA/dt).",
        )
        return looming

    def _extract_spatial_relationship(
        self,
        track_id: int,
        bbox: BoundingBox,
        lane_info: Any,
        img_w: float,
        img_h: float,
        validity: dict[str, IndicatorValidity],
    ) -> tuple[CameraDirection, LaneRelationship, float, float | None]:
        """Compute camera-relative sector, lane relationship, overlap, and lateral deviation."""
        cx = (bbox.x1 + bbox.x2) / 2.0
        by = float(bbox.y2)
        bw = max(1.0, float(bbox.width))

        # Camera direction estimation
        # Periphery check: lateral margins alongside vehicle
        if cx <= 0.15 * img_w and by > 0.50 * img_h:
            direction = CameraDirection.LEFT
        elif cx >= 0.85 * img_w and by > 0.50 * img_h:
            direction = CameraDirection.RIGHT
        else:
            # Check against lane or image center
            lx, rx = _get_lane_boundaries_at_y(lane_info, by)
            if lx is not None and rx is not None and rx > lx:
                if cx < lx:
                    direction = CameraDirection.FRONT_LEFT
                elif cx > rx:
                    direction = CameraDirection.FRONT_RIGHT
                else:
                    direction = CameraDirection.FRONT
            else:
                if cx < 0.40 * img_w:
                    direction = CameraDirection.FRONT_LEFT
                elif cx > 0.60 * img_w:
                    direction = CameraDirection.FRONT_RIGHT
                else:
                    direction = CameraDirection.FRONT

        # Lane relationship and overlap
        lx, rx = _get_lane_boundaries_at_y(lane_info, by)
        prev_rel = self._prev_lane_rel.get(track_id)

        if lx is None or rx is None or rx <= lx:
            validity["lane_overlap"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=0.0,
                reason="Lane boundary detection unavailable or unconfident.",
            )
            self._prev_lane_rel[track_id] = LaneRelationship.UNKNOWN
            return direction, LaneRelationship.UNKNOWN, 0.0, None

        lane_w = rx - lx
        lane_center = (lx + rx) / 2.0
        lat_dev = abs(cx - lane_center) / (lane_w / 2.0)

        # Overlap fraction
        overlap_w = max(0.0, min(float(bbox.x2), rx) - max(float(bbox.x1), lx))
        overlap_frac = min(1.0, max(0.0, overlap_w / bw))

        # Lane relationship assignment with transition awareness
        if overlap_frac >= 0.50 or (lx <= cx <= rx):
            current_rel = LaneRelationship.EGO_LANE
        elif float(bbox.x2) <= lx:
            current_rel = LaneRelationship.LEFT_ADJACENT_LANE
        elif float(bbox.x1) >= rx:
            current_rel = LaneRelationship.RIGHT_ADJACENT_LANE
        else:
            # Overlapping boundary
            if cx < lane_center:
                current_rel = (
                    LaneRelationship.ENTERING_EGO_LANE
                    if prev_rel == LaneRelationship.LEFT_ADJACENT_LANE
                    else LaneRelationship.LEFT_ADJACENT_LANE
                )
            else:
                current_rel = (
                    LaneRelationship.ENTERING_EGO_LANE
                    if prev_rel == LaneRelationship.RIGHT_ADJACENT_LANE
                    else LaneRelationship.RIGHT_ADJACENT_LANE
                )

        # Dynamic transition: leaving ego lane
        if prev_rel == LaneRelationship.EGO_LANE and overlap_frac < 0.40:
            current_rel = LaneRelationship.LEAVING_EGO_LANE

        self._prev_lane_rel[track_id] = current_rel

        lane_conf = getattr(lane_info, "confidence", 1.0)
        v_status = (
            ValidityStatus.VALID_METRIC
            if (lane_conf >= 0.4 and self.is_metric)
            else (ValidityStatus.VALID_PROXY if lane_conf >= 0.4 else ValidityStatus.UNCERTAIN)
        )
        validity["lane_overlap"] = IndicatorValidity(
            status=v_status,
            value=overlap_frac,
            reason=f"Lane overlap {overlap_frac:.2f} with lane confidence {lane_conf:.2f}.",
        )

        return direction, current_rel, overlap_frac, lat_dev

    def _extract_ttc(
        self,
        raw_ttc: float | None,
        speed_mps: float | None,
        validity: dict[str, IndicatorValidity],
    ) -> float | None:
        """Validate and wrap Time to Collision (TTC) with physical validity."""
        if raw_ttc is None or not math.isfinite(raw_ttc) or raw_ttc <= 0.0:
            validity["ttc"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Object is receding, stationary, or non-finite closing velocity.",
            )
            return None

        # Sanity check: must be closing
        if speed_mps is not None and speed_mps <= 0.0:
            validity["ttc"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Closing speed is non-positive; no collision closure path.",
            )
            return None

        if self.is_metric:
            validity["ttc"] = IndicatorValidity(
                status=ValidityStatus.VALID_METRIC,
                value=raw_ttc,
                reason="Metric TTC computed from calibrated distance and velocity.",
            )
        else:
            validity["ttc"] = IndicatorValidity(
                status=ValidityStatus.VALID_PROXY,
                value=raw_ttc,
                reason="Proxy TTC derived from relative depth closure rate.",
            )

        return raw_ttc

    def _extract_pet(
        self, validity: dict[str, IndicatorValidity]
    ) -> float | None:
        """Post-Encroachment Time (PET) extraction."""
        # Single forward monocular camera cannot establish 2D ground conflict polygon
        validity["pet"] = IndicatorValidity(
            status=ValidityStatus.UNAVAILABLE,
            value=None,
            reason="PET requires ground-plane conflict polygon and temporal occupancy tracking; unavailable in monocular camera.",
        )
        return None

    def _extract_drac(
        self,
        distance: float | None,
        closing_speed_mps: float | None,
        validity: dict[str, IndicatorValidity],
    ) -> float | None:
        """Calculate Deceleration Rate to Avoid Collision (DRAC)."""
        if not self.is_metric:
            validity["drac"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="DRAC requires calibrated metric distance (m) and speed (m/s); relative depth cannot compute m/s^2.",
            )
            return None

        if distance is None or closing_speed_mps is None:
            validity["drac"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Missing metric distance or closing speed.",
            )
            return None

        # DRAC = (vf - vl)^2 / (2d). Here closing_speed = vf - vl.
        drac_val, drac_status = calculate_drac(
            following_speed_mps=closing_speed_mps,
            leading_speed_mps=0.0,
            gap_m=distance,
            is_metric=True,
            require_metric=True,
        )

        if drac_status == "VALID" and drac_val is not None:
            validity["drac"] = IndicatorValidity(
                status=ValidityStatus.VALID_METRIC,
                value=drac_val,
                reason="Calibrated metric DRAC calculation.",
            )
            return drac_val
        elif drac_status == "ZERO_CLOSING_SPEED":
            validity["drac"] = IndicatorValidity(
                status=ValidityStatus.VALID_METRIC,
                value=0.0,
                reason="Leading vehicle moving faster or equal to follower; zero avoidance deceleration required.",
            )
            return 0.0
        else:
            validity["drac"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason=f"DRAC calculation returned {drac_status}.",
            )
            return None

    def _extract_trajectory_conflict(
        self,
        track_id: int,
        bbox: BoundingBox,
        prediction: TrajectoryPrediction | None,
        lane_info: Any,
        validity: dict[str, IndicatorValidity],
    ) -> tuple[TrajectoryConflictState, float | None]:
        """Assess trajectory conflict between forecast waypoints and ego travel path."""
        if prediction is None or not prediction.predicted_positions:
            validity["trajectory_conflict"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="No active trajectory forecast for this track.",
            )
            return TrajectoryConflictState.UNAVAILABLE, None

        waypoints = prediction.predicted_positions
        if len(waypoints) < 3:
            validity["trajectory_conflict"] = IndicatorValidity(
                status=ValidityStatus.UNCERTAIN,
                value=None,
                reason="Trajectory forecast has insufficient points (<3).",
            )
            return TrajectoryConflictState.UNCERTAIN, None

        first_conflict_time: float | None = None
        has_conflict = False

        for wp in waypoints:
            px, py = wp.position
            lx, rx = _get_lane_boundaries_at_y(lane_info, py)
            if lx is not None and rx is not None and rx > lx:
                # Check if waypoint falls inside ego lane corridor
                if lx <= px <= rx:
                    has_conflict = True
                    if first_conflict_time is None:
                        first_conflict_time = wp.time_offset
            else:
                # Fallback: check if prediction enters central image corridor
                if 0.35 * 1280.0 <= px <= 0.65 * 1280.0:
                    has_conflict = True
                    if first_conflict_time is None:
                        first_conflict_time = wp.time_offset

        state = (
            TrajectoryConflictState.CONFLICT
            if has_conflict
            else TrajectoryConflictState.NO_CONFLICT
        )
        validity["trajectory_conflict"] = IndicatorValidity(
            status=ValidityStatus.VALID_PROXY,
            value=state.value,
            reason="Image-space 2D Constant Acceleration Kalman trajectory projection against ego path corridor.",
        )
        return state, first_conflict_time

    def _map_vulnerability(self, class_name: str) -> VulnerabilityLevel:
        """Map object class name to physical vulnerability category."""
        c = class_name.lower()
        if c in ("person", "pedestrian"):
            return VulnerabilityLevel.CRITICAL
        elif c in ("bicycle", "cyclist"):
            return VulnerabilityLevel.CRITICAL
        elif c in ("motorcycle", "motorbike"):
            return VulnerabilityLevel.HIGH
        elif c in ("car",):
            return VulnerabilityLevel.STANDARD
        elif c in ("truck", "bus"):
            return VulnerabilityLevel.LOW_HEAVY
        return VulnerabilityLevel.UNKNOWN

    def _extract_behavior(
        self,
        track_id: int,
        behavior_state: Any,
        validity: dict[str, IndicatorValidity],
    ) -> tuple[float | None, str | None]:
        """Extract VRU behavior classification and road-crossing probability."""
        if behavior_state is None:
            validity["crossing_probability"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="No behavior subsystem output for this track.",
            )
            return None, None

        primary_lbl = getattr(behavior_state, "primary_label", None)
        lbl_str = primary_lbl.value if hasattr(primary_lbl, "value") else str(primary_lbl)
        cross_prob = getattr(behavior_state, "crossing_probability", None)

        if cross_prob is not None and math.isfinite(cross_prob):
            validity["crossing_probability"] = IndicatorValidity(
                status=ValidityStatus.VALID_PROXY,
                value=cross_prob,
                reason="Pose/motion heuristic crossing probability.",
            )
        else:
            validity["crossing_probability"] = IndicatorValidity(
                status=ValidityStatus.UNAVAILABLE,
                value=None,
                reason="Crossing probability not computed for this entity.",
            )

        return cross_prob, lbl_str
