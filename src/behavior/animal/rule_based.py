from __future__ import annotations

import math
from typing import Optional, Tuple

from src.behavior.base import (
    BaseAnimalBehaviorClassifier,
    LaneContext,
    MotionEstimate,
    TrackHistory,
)
from src.behavior.config import BehaviorConfig
from src.behavior.human.crossing_probability import (
    GeometricCrossingProbabilityEstimator,
)
from src.schemas.behavior import BehaviorLabel, BehaviorState
from src.schemas.pose import AnimalKeypointName, AnimalPose
from src.utils.logger import get_logger


class RuleBasedAnimalBehaviorClassifier(BaseAnimalBehaviorClassifier):
    """Rule-based animal behavior classifier.

    Uses configurable thresholds for speed, keypoint geometry,
    and lane proximity.

    Args:
        config: Full behavior configuration.
    """

    def __init__(self, config: BehaviorConfig) -> None:
        self._cfg = config
        self._logger = get_logger(
            f"{__name__}.RuleBasedAnimalBehaviorClassifier"
        )
        self._crossing_estimator = GeometricCrossingProbabilityEstimator(
            config.crossing_probability
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        pose_history: Tuple[AnimalPose, ...],
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

        # ------ Evaluate all labels ------ #
        motion_label, motion_conf = self._classify_motion(speed_mps)
        lying_label, lying_conf = self._classify_lying(latest)
        grazing_label, grazing_conf = self._classify_grazing(
            latest, speed_mps
        )
        road_label, road_conf, crossing_prob = self._classify_road_state(
            latest, lane_context, heading_deg, speed_mps
        )

        feature_log = {
            "speed_mps": speed_mps,
            "heading_deg": heading_deg,
            "torso_height": self._torso_height(latest),
            "spine_angle": self._spine_angle(latest),
            "head_down_angle": self._head_down_angle(latest),
            "crossing_prob": crossing_prob,
        }

        # ------ Priority-based label selection ------ #
        if lying_label is not None and lying_conf > 0.0:
            primary = lying_label
            confidence = lying_conf
        elif grazing_label is not None and grazing_conf > 0.0:
            primary = grazing_label
            confidence = grazing_conf
        elif road_label is not None and road_conf > 0.0:
            primary = road_label
            confidence = road_conf
        elif motion_label is not None:
            primary = motion_label
            confidence = motion_conf
        else:
            primary = BehaviorLabel.UNKNOWN
            confidence = 0.0

        self._logger.debug(
            "Animal classify  |  track=%d label=%s confidence=%.3f "
            "features=%s",
            track_id,
            primary.value,
            confidence,
            feature_log,
        )

        return BehaviorState(
            track_id=track_id,
            entity_type="animal",
            primary_label=primary,
            secondary_labels=self._secondary_labels(
                primary, motion_label, road_label, lying_label, grazing_label
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
        if speed_mps is None:
            return (None, 0.0)

        cfg = self._cfg.animal.speed
        if speed_mps >= cfg.running_min_mps:
            margin = (speed_mps - cfg.running_min_mps) / cfg.running_min_mps
            conf = self._compute_confidence(margin)
            return (BehaviorLabel.ANIMAL_RUNNING, conf)
        if speed_mps >= cfg.walking_min_mps:
            margin = (speed_mps - cfg.walking_min_mps) / cfg.walking_min_mps
            conf = self._compute_confidence(margin)
            return (BehaviorLabel.ANIMAL_WALKING, conf)
        return (BehaviorLabel.ANIMAL_STANDING, 0.7)

    def _classify_lying(
        self,
        pose: AnimalPose,
    ) -> Tuple[Optional[BehaviorLabel], float]:
        """Keypoint geometry: low torso + horizontal spine → LYING."""
        torso_height = self._torso_height(pose)
        spine_angle = self._spine_angle(pose)

        cfg = self._cfg.animal.lying
        if torso_height is not None and spine_angle is not None:
            if (
                torso_height < cfg.torso_height_threshold
                and abs(spine_angle) < cfg.spine_angle_threshold_deg
            ):
                margin = (
                    cfg.torso_height_threshold - torso_height
                ) / cfg.torso_height_threshold
                conf = self._compute_confidence(margin)
                return (BehaviorLabel.ANIMAL_LYING, conf)
        return (None, 0.0)

    def _classify_grazing(
        self,
        pose: AnimalPose,
        speed_mps: Optional[float],
    ) -> Tuple[Optional[BehaviorLabel], float]:
        """Head down + low speed → GRAZING (scene context optional)."""
        head_down = self._head_down_angle(pose)
        if head_down is None:
            return (None, 0.0)

        cfg = self._cfg.animal.grazing
        speed_ok = (
            speed_mps is None or speed_mps <= cfg.max_speed_mps
        )
        if abs(head_down) > cfg.head_down_angle_threshold_deg and speed_ok:
            margin = (
                abs(head_down) - cfg.head_down_angle_threshold_deg
            ) / 90.0
            conf = self._compute_confidence(margin)
            return (BehaviorLabel.ANIMAL_GRAZING, conf)
        return (None, 0.0)

    def _classify_road_state(
        self,
        pose: AnimalPose,
        lane_context: Optional[LaneContext],
        heading_deg: Optional[float],
        speed_mps: Optional[float],
    ) -> Tuple[Optional[BehaviorLabel], float, Optional[float]]:
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

        from src.behavior.human.rule_based import RuleBasedHumanBehaviorClassifier

        cfg = self._cfg.animal.road

        dist_px = lane_context.distance_to_nearest_boundary(foot_x, foot_y)
        if dist_px is None:
            return (None, 0.0, crossing_prob)

        is_inside = lane_context.is_inside(foot_x, foot_y)

        # --- Entity is ON the road ----------------------------------------- #
        if is_inside:
            if heading_deg is not None and dist_px < cfg.approaching_distance_px:
                dir_toward_edge = RuleBasedHumanBehaviorClassifier._heading_toward_road(
                    lane_context, foot_x, foot_y, heading_deg
                )
                if dir_toward_edge:
                    margin = (cfg.approaching_distance_px - dist_px) / cfg.approaching_distance_px
                    conf = self._compute_confidence(margin)
                    return (BehaviorLabel.ANIMAL_APPROACHING_ROAD, conf, crossing_prob)

            if crossing_prob >= cfg.crossing_probability_high:
                return (BehaviorLabel.ANIMAL_CROSSING_ROAD, crossing_prob, crossing_prob)

            return (None, 0.0, crossing_prob)

        # --- Entity is OFF the road ---------------------------------------- #
        if crossing_prob >= cfg.crossing_probability_high:
            return (BehaviorLabel.ANIMAL_CROSSING_ROAD, crossing_prob, crossing_prob)

        if heading_deg is not None and dist_px < cfg.approaching_distance_px:
            dir_toward_road = RuleBasedHumanBehaviorClassifier._heading_toward_road(
                lane_context, foot_x, foot_y, heading_deg
            )
            if dir_toward_road:
                margin = (cfg.approaching_distance_px - dist_px) / cfg.approaching_distance_px
                conf = self._compute_confidence(margin)
                return (
                    BehaviorLabel.ANIMAL_APPROACHING_ROAD,
                    conf,
                    crossing_prob,
                )

        return (None, 0.0, crossing_prob)

    # ------------------------------------------------------------------ #
    #  Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _torso_height(pose: AnimalPose) -> Optional[float]:
        """Normalised vertical distance between mid_back and front paw.

        Returns fraction of bbox height (0=on ground, 1=full height).
        """
        kp_map = {kp.name: kp for kp in pose.keypoints}
        # Use mid_back as top of torso, nearest front paw as bottom
        try:
            back = kp_map[AnimalKeypointName.MID_BACK.value]
        except KeyError:
            return None

        # Find lowest visible front paw as proxy for ground contact
        front_paw_names = [
            AnimalKeypointName.LEFT_FRONT_PAW.value,
            AnimalKeypointName.RIGHT_FRONT_PAW.value,
        ]
        bottom_y: Optional[float] = None
        for name in front_paw_names:
            kp = kp_map.get(name)
            if kp is not None and kp.visible:
                if bottom_y is None or kp.y > bottom_y:
                    bottom_y = kp.y

        if not back.visible or bottom_y is None:
            return None

        torso_height = abs(back.y - bottom_y)
        bbox_h = pose.bbox[3] - pose.bbox[1]
        if bbox_h <= 1e-6:
            return None
        return torso_height / bbox_h

    @staticmethod
    def _spine_angle(pose: AnimalPose) -> Optional[float]:
        """Angle of spine from horizontal in degrees.

        Uses neck → mid_back vector.
        Returns 0 = perfectly horizontal (either direction), 90 = vertical.
        """
        kp_map = {kp.name: kp for kp in pose.keypoints}
        try:
            neck = kp_map[AnimalKeypointName.NECK.value]
            mid_back = kp_map[AnimalKeypointName.MID_BACK.value]
        except KeyError:
            return None

        if not neck.visible or not mid_back.visible:
            return None

        dx = mid_back.x - neck.x
        dy = mid_back.y - neck.y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        raw = abs(math.degrees(math.atan2(dy, dx)))
        # Normalise to [0, 90]: 0 = horizontal, 90 = vertical
        return min(raw, 180.0 - raw)

    @staticmethod
    def _head_down_angle(pose: AnimalPose) -> Optional[float]:
        """Angle of nose→neck vector below the horizontal plane.

        Positive when nose is below neck (head down), zero when level.
        Max 90° when nose is directly below neck.
        """
        kp_map = {kp.name: kp for kp in pose.keypoints}
        try:
            nose = kp_map[AnimalKeypointName.NOSE.value]
            neck = kp_map[AnimalKeypointName.NECK.value]
        except KeyError:
            return None

        if not nose.visible or not neck.visible:
            return None

        dx = nose.x - neck.x
        dy = nose.y - neck.y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return math.degrees(math.atan2(dy, abs(dx)))

    @staticmethod
    def _compute_confidence(margin: float) -> float:
        raw = 0.5 + min(0.49, abs(margin) * 0.5)
        return max(0.0, min(0.99, raw))

    @staticmethod
    def _secondary_labels(
        primary: BehaviorLabel,
        motion_label: Optional[BehaviorLabel],
        road_label: Optional[BehaviorLabel],
        lying_label: Optional[BehaviorLabel],
        grazing_label: Optional[BehaviorLabel],
    ) -> Tuple[BehaviorLabel, ...]:
        candidates = [
            lbl
            for lbl in (
                lying_label,
                road_label,
                motion_label,
                grazing_label,
            )
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
            entity_type="animal",
            primary_label=BehaviorLabel.UNKNOWN,
            secondary_labels=(),
            confidence=0.0,
            timestamp=timestamp,
        )
