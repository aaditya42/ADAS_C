from __future__ import annotations

import math
from typing import Optional, Tuple

from src.behavior.base import (
    BaseHumanBehaviorClassifier,
    LaneContext,
    MotionEstimate,
    TrackHistory,
)
from src.behavior.config import BehaviorConfig
from src.behavior.human.crossing_probability import (
    GeometricCrossingProbabilityEstimator,
)
from src.schemas.behavior import BehaviorLabel, BehaviorState
from src.schemas.pose import HumanKeypointName, HumanPose
from src.utils.logger import get_logger


class RuleBasedHumanBehaviorClassifier(BaseHumanBehaviorClassifier):
    """Rule-based human behavior classifier.

    Uses configurable thresholds for speed, keypoint geometry,
    head orientation, and lane context to produce a single
    :class:`BehaviorState` per call.

    Args:
        config: Full behavior configuration.
    """

    def __init__(self, config: BehaviorConfig) -> None:
        self._cfg = config
        self._logger = get_logger(
            f"{__name__}.RuleBasedHumanBehaviorClassifier"
        )
        self._crossing_estimator = GeometricCrossingProbabilityEstimator(
            config.crossing_probability
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        pose_history: Tuple[HumanPose, ...],
        track_history: Optional[TrackHistory],
        motion: Optional[MotionEstimate],
        lane_context: Optional[LaneContext],
        timestamp: float,
    ) -> BehaviorState:
        if not pose_history:
            return self._unknown(track_history, timestamp)

        latest = pose_history[-1]
        track_id = latest.track_id if latest.track_id is not None else 0
        if track_history is not None:
            track_id = track_history.track_id

        speed_mps = motion.speed_mps if motion is not None else None
        heading_deg = motion.heading_deg if motion is not None else None

        # ------ Evaluate all labels and their confidences ------ #
        motion_label, motion_conf = self._classify_motion(speed_mps)
        fallen_label, fallen_conf = self._classify_fallen(latest)
        distracted_label, distracted_conf = self._classify_distracted(
            latest, heading_deg, motion_conf
        )
        road_label, road_conf, crossing_prob = self._classify_road_state(
            latest, lane_context, heading_deg, speed_mps
        )

        feature_log = {
            "speed_mps": speed_mps,
            "heading_deg": heading_deg,
            "torso_angle": self._torso_angle(latest),
            "wider_than_tall": self._bbox_wider_than_tall(latest),
            "head_vs_motion_angle": (
                self._head_motion_angle_diff(latest, heading_deg)
                if heading_deg is not None and latest.head_orientation_deg is not None
                else None
            ),
            "crossing_prob": crossing_prob,
        }

        # ------ Priority-based label selection ------ #
        if fallen_label is not None and fallen_conf > 0.0:
            primary = fallen_label
            confidence = fallen_conf
        elif road_label is not None and road_conf > 0.0:
            primary = road_label
            confidence = road_conf
        elif distracted_label is not None and distracted_conf > 0.0:
            primary = distracted_label
            confidence = distracted_conf
        elif motion_label is not None:
            primary = motion_label
            confidence = motion_conf
        else:
            primary = BehaviorLabel.UNKNOWN
            confidence = 0.0

        self._logger.debug(
            "Human classify  |  track=%d label=%s confidence=%.3f "
            "features=%s",
            track_id,
            primary.value,
            confidence,
            feature_log,
        )

        return BehaviorState(
            track_id=track_id,
            entity_type="human",
            primary_label=primary,
            secondary_labels=self._secondary_labels(
                primary, motion_label, road_label, fallen_label, distracted_label
            ),
            confidence=confidence,
            crossing_probability=crossing_prob,
            body_orientation_deg=latest.body_orientation_deg,
            head_orientation_deg=latest.head_orientation_deg,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers — per-label classifiers
    # ------------------------------------------------------------------ #

    def _classify_motion(
        self,
        speed_mps: Optional[float],
    ) -> Tuple[Optional[BehaviorLabel], float]:
        """Speed-based motion classification."""
        if speed_mps is None:
            return (None, 0.0)

        cfg = self._cfg.human.speed
        if speed_mps >= cfg.running_min_mps:
            margin = (speed_mps - cfg.running_min_mps) / cfg.running_min_mps
            conf = self._compute_confidence(margin)
            return (BehaviorLabel.HUMAN_RUNNING, conf)
        if speed_mps >= cfg.walking_min_mps:
            margin = (speed_mps - cfg.walking_min_mps) / cfg.walking_min_mps
            conf = self._compute_confidence(margin)
            return (BehaviorLabel.HUMAN_WALKING, conf)
        return (BehaviorLabel.HUMAN_STANDING, 0.7)

    def _classify_fallen(
        self,
        pose: HumanPose,
    ) -> Tuple[Optional[BehaviorLabel], float]:
        """Keypoint geometry-based fall detection.

        A person is classified as fallen when:
        * torso is significantly tilted from vertical, **and**
        * bounding-box is wider than tall (aspect_ratio > 1.0).
        """
        torso_angle = self._torso_angle(pose)
        is_wider_than_tall = self._bbox_wider_than_tall(pose)

        cfg = self._cfg.human.fallen
        if (
            torso_angle is not None
            and torso_angle > cfg.torso_angle_threshold_deg
            and is_wider_than_tall
        ):
            angle_margin = (
                torso_angle - cfg.torso_angle_threshold_deg
            ) / 90.0
            conf = self._compute_confidence(angle_margin)
            return (BehaviorLabel.HUMAN_FALLEN, conf)
        return (None, 0.0)

    def _classify_distracted(
        self,
        pose: HumanPose,
        heading_deg: Optional[float],
        motion_confidence: float,
    ) -> Tuple[Optional[BehaviorLabel], float]:
        """Head orientation vs motion vector distraction check."""
        if (
            heading_deg is None
            or pose.head_orientation_deg is None
        ):
            return (None, 0.0)

        angle_diff = self._head_motion_angle_diff(pose, heading_deg)
        if angle_diff is None:
            return (None, 0.0)

        cfg = self._cfg.human.distracted
        if abs(angle_diff) > cfg.head_motion_angle_threshold_deg:
            margin = (
                abs(angle_diff) - cfg.head_motion_angle_threshold_deg
            ) / 90.0
            conf = self._compute_confidence(margin) * motion_confidence
            return (BehaviorLabel.HUMAN_DISTRACTED, conf)
        return (None, 0.0)

    def _classify_road_state(
        self,
        pose: HumanPose,
        lane_context: Optional[LaneContext],
        heading_deg: Optional[float],
        speed_mps: Optional[float],
    ) -> Tuple[Optional[BehaviorLabel], float, Optional[float]]:
        """Lane-proximity road state classification."""
        if lane_context is None:
            return (None, 0.0, None)

        bbox = pose.bbox
        foot_x = (bbox[0] + bbox[2]) / 2.0
        foot_y = bbox[3]

        crossing_prob = self._crossing_estimator.estimate(
            foot_x=foot_x,
            foot_y=foot_y,
            lane_context=lane_context,
            heading_deg=heading_deg,
            speed_mps=speed_mps,
        )

        cfg = self._cfg.human.road

        dist_px = lane_context.distance_to_nearest_boundary(foot_x, foot_y)
        if dist_px is None:
            return (None, 0.0, crossing_prob)

        is_inside = lane_context.is_inside(foot_x, foot_y)

        # --- Entity is ON the road ----------------------------------------- #
        if is_inside:
            # Heading toward the nearest edge → LEAVING_ROAD
            if (
                heading_deg is not None
                and dist_px < cfg.leaving_distance_px
                and self._heading_toward_road(
                    lane_context, foot_x, foot_y, heading_deg
                )
            ):
                margin = (cfg.leaving_distance_px - dist_px) / cfg.leaving_distance_px
                conf = self._compute_confidence(margin)
                return (BehaviorLabel.HUMAN_LEAVING_ROAD, conf, crossing_prob)

            # Heading across the road → CROSSING_ROAD
            if crossing_prob >= cfg.crossing_probability_high:
                return (BehaviorLabel.HUMAN_CROSSING_ROAD, crossing_prob, crossing_prob)

            return (None, 0.0, crossing_prob)

        # --- Entity is OFF the road ---------------------------------------- #
        # Very close to the road (crossing prob already high from distance)
        if crossing_prob >= cfg.crossing_probability_high:
            return (BehaviorLabel.HUMAN_CROSSING_ROAD, crossing_prob, crossing_prob)

        # Approaching the road
        if (
            heading_deg is not None
            and dist_px < cfg.approaching_distance_px
            and self._heading_toward_road(
                lane_context, foot_x, foot_y, heading_deg
            )
        ):
            margin = (cfg.approaching_distance_px - dist_px) / cfg.approaching_distance_px
            conf = self._compute_confidence(margin)
            return (BehaviorLabel.HUMAN_APPROACHING_ROAD, conf, crossing_prob)

        return (None, 0.0, crossing_prob)

    # ------------------------------------------------------------------ #
    #  Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _torso_angle(pose: HumanPose) -> Optional[float]:
        """Angle of torso from vertical in degrees.

        Uses left_hip → left_shoulder vector in image coordinates.
        0 = perfectly upright, 90 = horizontal.
        """
        kp_map = {kp.name: kp for kp in pose.keypoints}
        try:
            hip = kp_map[HumanKeypointName.LEFT_HIP.value]
            shoulder = kp_map[HumanKeypointName.LEFT_SHOULDER.value]
        except KeyError:
            return None

        if not hip.visible or not shoulder.visible:
            return None

        dx = shoulder.x - hip.x
        dy = shoulder.y - hip.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return None
        # Angle from vertical via dot-product with (0, -1)
        cos_theta = min(1.0, abs(dy) / length)
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def _bbox_wider_than_tall(pose: HumanPose) -> bool:
        """Whether the bounding box is wider than tall."""
        bbox = pose.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return h > 0 and w > h

    @staticmethod
    def _head_motion_angle_diff(
        pose: HumanPose,
        heading_deg: float,
    ) -> Optional[float]:
        """Angular difference between head orientation and motion heading."""
        if pose.head_orientation_deg is None:
            return None
        diff = pose.head_orientation_deg - heading_deg
        diff = (diff + 180) % 360 - 180
        return diff

    @staticmethod
    def _heading_toward_road(
        lane_context: LaneContext,
        foot_x: float,
        foot_y: float,
        heading_deg: float,
    ) -> bool:
        """Whether the entity's heading points toward the nearest lane."""
        nearest_x = None
        if lane_context.left_boundary is not None:
            lx = lane_context.lane_x_at_y(lane_context.left_boundary, foot_y)
            if lx is not None:
                nearest_x = lx
        if lane_context.right_boundary is not None:
            rx = lane_context.lane_x_at_y(lane_context.right_boundary, foot_y)
            if rx is not None:
                if nearest_x is None or abs(foot_x - rx) < abs(foot_x - nearest_x):
                    nearest_x = rx

        if nearest_x is None:
            return False

        dir_to_road = 1.0 if nearest_x > foot_x else -1.0
        heading_rad = math.radians(heading_deg)
        horiz_component = math.sin(heading_rad)

        return dir_to_road * horiz_component > 0.0

    @staticmethod
    def _compute_confidence(margin: float) -> float:
        """Map a margin into a confidence in [0.5, 0.99]."""
        raw = 0.5 + min(0.49, abs(margin) * 0.5)
        return max(0.0, min(0.99, raw))

    @staticmethod
    def _secondary_labels(
        primary: BehaviorLabel,
        motion_label: Optional[BehaviorLabel],
        road_label: Optional[BehaviorLabel],
        fallen_label: Optional[BehaviorLabel],
        distracted_label: Optional[BehaviorLabel],
    ) -> Tuple[BehaviorLabel, ...]:
        """Build secondary labels excluding the primary."""
        candidates = [
            lbl
            for lbl in (fallen_label, road_label, motion_label, distracted_label)
            if lbl is not None and lbl != primary
        ]
        return tuple(candidates)

    @staticmethod
    def _unknown(
        track_history: Optional[TrackHistory],
        timestamp: float,
    ) -> BehaviorState:
        tid = track_history.track_id if track_history is not None else 0
        return BehaviorState(
            track_id=tid,
            entity_type="human",
            primary_label=BehaviorLabel.UNKNOWN,
            secondary_labels=(),
            confidence=0.0,
            timestamp=timestamp,
        )
