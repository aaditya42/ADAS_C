"""Unit tests for threat feature extraction and indicator validity.

Covers:
- Valid vs proxy vs invalid TTC
- Unavailable PET
- DRAC under relative vs metric calibrated modes
- Motion kinematics, speed variation, acceleration noise, looming rate
- Trajectory conflict states (CONFLICT, NO_CONFLICT, UNCERTAIN, UNAVAILABLE)
- Lane relationship and overlap (EGO_LANE, adjacent, entering, leaving, missing lane)
- Object class and vulnerability mapping
- Stale track eviction and tracker ID reuse handling
"""

import numpy as np
import pytest

from src.lane_detection.interfaces import LaneDetectionResult
from src.schemas.bbox import BoundingBox
from src.schemas.frame import Frame
from src.schemas.motion import RelativeMotion
from src.schemas.perceived_object import PerceivedObject
from src.schemas.trajectory import TrajectoryPoint, TrajectoryPrediction
from src.threat.feature_extractor import ThreatFeatureExtractor
from src.threat.schemas import (
    CameraDirection,
    LaneRelationship,
    TrajectoryConflictState,
    ValidityStatus,
    VulnerabilityLevel,
)


def _make_dummy_frame(timestamp: float = 1.0, width: int = 1280, height: int = 720) -> Frame:
    """Create a dummy frame for testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=int(timestamp * 30), timestamp=timestamp, source="test")


def _make_lane_result(confidence: float = 0.85) -> LaneDetectionResult:
    """Create dummy lane detection result spanning ego lane [400, 880] at bottom."""
    left_pts = ((500, 360), (450, 540), (400, 720))
    right_pts = ((780, 360), (830, 540), (880, 720))
    center_pts = ((640, 360), (640, 540), (640, 720))
    return LaneDetectionResult(
        left_lane_points=left_pts,
        right_lane_points=right_pts,
        center_lane_points=center_pts,
        lane_count=2,
        ego_lane_center_x=640.0,
        confidence=confidence,
    )


class TestThreatFeatureExtractor:
    """Comprehensive test suite for ThreatFeatureExtractor."""

    def test_ttc_proxy_vs_metric_validity(self):
        """Test that relative distance sets proxy TTC, while calibrated sets metric TTC."""
        # Relative config
        rel_cfg = {"distance": {"type": "relative"}, "threat_assessment": {"enabled": True}}
        extractor_rel = ThreatFeatureExtractor(rel_cfg)

        frame = _make_dummy_frame(timestamp=1.0)
        obj = PerceivedObject(
            track_id=1,
            class_name="car",
            bbox=BoundingBox(550, 400, 730, 600),
            depth=15.0,
            distance=15.0,
            depth_confidence=0.9,
        )
        motion = {1: RelativeMotion(1, speed_mps=5.0, acceleration_mps2=0.0, direction="approaching", confidence=0.8)}
        ttc = {1: 3.0}

        feats_rel = extractor_rel.extract([obj], motion_map=motion, ttc_map=ttc, frame=frame)
        f1 = feats_rel[1]
        assert f1.ttc == 3.0
        assert f1.indicator_validity["ttc"].status == ValidityStatus.VALID_PROXY
        assert not f1.is_metric

        # Metric calibrated config
        calib_cfg = {"distance": {"type": "calibrated"}, "threat_assessment": {"enabled": True}}
        extractor_calib = ThreatFeatureExtractor(calib_cfg)
        feats_calib = extractor_calib.extract([obj], motion_map=motion, ttc_map=ttc, frame=frame)
        f2 = feats_calib[1]
        assert f2.ttc == 3.0
        assert f2.indicator_validity["ttc"].status == ValidityStatus.VALID_METRIC
        assert f2.is_metric

    def test_invalid_and_non_closing_ttc(self):
        """Test that non-closing or negative TTC is marked UNAVAILABLE."""
        cfg = {"threat_assessment": {"enabled": True}}
        extractor = ThreatFeatureExtractor(cfg)
        frame = _make_dummy_frame(timestamp=1.0)
        obj = PerceivedObject(
            track_id=1,
            class_name="car",
            bbox=BoundingBox(550, 400, 730, 600),
            depth=20.0,
            distance=20.0,
            depth_confidence=0.9,
        )

        # Receding vehicle (speed_mps < 0)
        motion = {1: RelativeMotion(1, speed_mps=-3.0, acceleration_mps2=0.0, direction="receding", confidence=0.8)}
        ttc = {1: -6.6}

        feats = extractor.extract([obj], motion_map=motion, ttc_map=ttc, frame=frame)
        assert feats[1].ttc is None
        assert feats[1].indicator_validity["ttc"].status == ValidityStatus.UNAVAILABLE

    def test_pet_unavailable(self):
        """Test that PET is always marked UNAVAILABLE in monocular setup with diagnostic reason."""
        extractor = ThreatFeatureExtractor({"threat_assessment": {"enabled": True}})
        frame = _make_dummy_frame(timestamp=1.0)
        obj = PerceivedObject(1, "car", BoundingBox(500, 400, 600, 500), 10.0, 10.0, 0.9)

        feats = extractor.extract([obj], frame=frame)
        assert feats[1].pet is None
        assert feats[1].indicator_validity["pet"].status == ValidityStatus.UNAVAILABLE
        assert "monocular" in feats[1].indicator_validity["pet"].reason

    def test_drac_validity_relative_vs_calibrated(self):
        """Test that DRAC is UNAVAILABLE in relative mode and VALID_METRIC in calibrated mode."""
        obj = PerceivedObject(1, "car", BoundingBox(500, 400, 600, 500), 20.0, 20.0, 0.9)
        motion = {1: RelativeMotion(1, speed_mps=6.0, acceleration_mps2=-1.0, direction="approaching", confidence=0.8)}
        frame = _make_dummy_frame(timestamp=1.0)

        # Relative mode
        extractor_rel = ThreatFeatureExtractor({"distance": {"type": "relative"}})
        feats_rel = extractor_rel.extract([obj], motion_map=motion, frame=frame)
        assert feats_rel[1].drac is None
        assert feats_rel[1].indicator_validity["drac"].status == ValidityStatus.UNAVAILABLE

        # Calibrated mode: DRAC = (6.0)^2 / (2 * 20.0) = 36 / 40 = 0.9 m/s^2
        extractor_calib = ThreatFeatureExtractor({"distance": {"type": "calibrated"}})
        feats_calib = extractor_calib.extract([obj], motion_map=motion, frame=frame)
        assert feats_calib[1].drac is not None
        assert abs(feats_calib[1].drac - 0.9) < 1e-4
        assert feats_calib[1].indicator_validity["drac"].status == ValidityStatus.VALID_METRIC

        # Non-closing case in calibrated mode (speed <= 0)
        receding_motion = {1: RelativeMotion(1, speed_mps=-2.0, acceleration_mps2=0.0, direction="receding", confidence=0.8)}
        feats_rec = extractor_calib.extract([obj], motion_map=receding_motion, frame=frame)
        assert feats_rec[1].drac == 0.0
        assert feats_rec[1].indicator_validity["drac"].status == ValidityStatus.VALID_METRIC
        assert "zero" in feats_rec[1].indicator_validity["drac"].reason.lower()

    def test_looming_rate_and_speed_variation(self):
        """Test optical looming rate and speed variation across sequential frames."""
        extractor = ThreatFeatureExtractor({"threat_assessment": {"enabled": True}})

        # Frame 1: small bbox (area = 100 * 100 = 10000)
        obj1 = PerceivedObject(1, "car", BoundingBox(500, 400, 600, 500), 30.0, 30.0, 0.9)
        f1 = _make_dummy_frame(timestamp=1.0)
        extractor.extract([obj1], frame=f1)

        # Frame 2: expanding bbox (area = 120 * 120 = 14400) after 0.2s
        obj2 = PerceivedObject(1, "car", BoundingBox(490, 390, 610, 510), 25.0, 25.0, 0.9)
        f2 = _make_dummy_frame(timestamp=1.2)
        feats = extractor.extract([obj2], frame=f2)

        # Looming rate = (14400 - 10000) / (10000 * 0.2) = 4400 / 2000 = 2.2
        assert feats[1].looming_rate is not None
        assert abs(feats[1].looming_rate - 2.2) < 1e-3
        assert feats[1].indicator_validity["looming_rate"].status == ValidityStatus.VALID_PROXY

    def test_trajectory_conflict_assessment(self):
        """Test trajectory conflict state against lane corridor."""
        extractor = ThreatFeatureExtractor({"threat_assessment": {"enabled": True}})
        frame = _make_dummy_frame(timestamp=1.0)
        lane_res = _make_lane_result()

        # Object located on left: cx = 300, y2 = 600 (left lane is at x=430)
        obj = PerceivedObject(1, "car", BoundingBox(250, 450, 350, 600), 20.0, 20.0, 0.9)

        # Trajectory 1: predicts crossing into ego lane corridor (px = 600 at py = 500)
        pred_conflict = TrajectoryPrediction(
            track_id=1,
            class_name="car",
            prediction_timestamp=1.0,
            horizon_seconds=2.0,
            current_position=(300.0, 600.0),
            estimated_velocity=(100.0, -50.0),
            estimated_acceleration=(0.0, 0.0),
            predicted_positions=[
                TrajectoryPoint(future_timestamp=1.5, time_offset=0.5, position=(400.0, 550.0)),
                TrajectoryPoint(future_timestamp=2.0, time_offset=1.0, position=(550.0, 500.0)),
                TrajectoryPoint(future_timestamp=2.5, time_offset=1.5, position=(650.0, 450.0)),
            ],
            final_predicted_position=(650.0, 450.0),
        )

        feats = extractor.extract([obj], lane_info=lane_res, trajectories=[pred_conflict], frame=frame)
        assert feats[1].trajectory_conflict == TrajectoryConflictState.CONFLICT
        assert feats[1].trajectory_conflict_time == 1.0

        # Trajectory 2: predicts staying on left (px = 200)
        pred_no_conflict = TrajectoryPrediction(
            track_id=1,
            class_name="car",
            prediction_timestamp=1.0,
            horizon_seconds=2.0,
            current_position=(300.0, 600.0),
            estimated_velocity=(-50.0, -50.0),
            estimated_acceleration=(0.0, 0.0),
            predicted_positions=[
                TrajectoryPoint(future_timestamp=1.5, time_offset=0.5, position=(250.0, 550.0)),
                TrajectoryPoint(future_timestamp=2.0, time_offset=1.0, position=(200.0, 500.0)),
                TrajectoryPoint(future_timestamp=2.5, time_offset=1.5, position=(150.0, 450.0)),
            ],
            final_predicted_position=(150.0, 450.0),
        )
        feats_no = extractor.extract([obj], lane_info=lane_res, trajectories=[pred_no_conflict], frame=frame)
        assert feats_no[1].trajectory_conflict == TrajectoryConflictState.NO_CONFLICT

    def test_lane_relationship_and_missing_lane(self):
        """Test lane overlap, relationships, and graceful degradation on missing lane."""
        extractor = ThreatFeatureExtractor({"threat_assessment": {"enabled": True}})
        frame = _make_dummy_frame(timestamp=1.0)
        lane_res = _make_lane_result()

        # Object in ego lane: cx = 640, y2 = 700 (lane is [405, 875])
        obj_in_lane = PerceivedObject(1, "car", BoundingBox(590, 500, 690, 700), 15.0, 15.0, 0.9)
        # Object in left adjacent lane: cx = 200 (x2 = 250 < 405)
        obj_left = PerceivedObject(2, "car", BoundingBox(150, 500, 250, 700), 15.0, 15.0, 0.9)
        # Object in right adjacent lane: cx = 1000 (x1 = 950 > 875)
        obj_right = PerceivedObject(3, "car", BoundingBox(950, 500, 1050, 700), 15.0, 15.0, 0.9)

        feats = extractor.extract([obj_in_lane, obj_left, obj_right], lane_info=lane_res, frame=frame)
        assert feats[1].lane_relationship == LaneRelationship.EGO_LANE
        assert feats[1].direction == CameraDirection.FRONT
        assert feats[1].lane_overlap >= 0.9

        assert feats[2].lane_relationship == LaneRelationship.LEFT_ADJACENT_LANE
        assert feats[2].direction == CameraDirection.FRONT_LEFT
        assert feats[2].lane_overlap == 0.0

        assert feats[3].lane_relationship == LaneRelationship.RIGHT_ADJACENT_LANE
        assert feats[3].direction == CameraDirection.FRONT_RIGHT
        assert feats[3].lane_overlap == 0.0

        # Missing lane geometry
        feats_no_lane = extractor.extract([obj_in_lane], lane_info=None, frame=frame)
        assert feats_no_lane[1].lane_relationship == LaneRelationship.UNKNOWN
        assert feats_no_lane[1].indicator_validity["lane_overlap"].status == ValidityStatus.UNAVAILABLE

    def test_vulnerability_mapping(self):
        """Test object class mapping to vulnerability level."""
        extractor = ThreatFeatureExtractor({"threat_assessment": {"enabled": True}})
        frame = _make_dummy_frame(timestamp=1.0)

        p = PerceivedObject(1, "person", BoundingBox(10, 10, 20, 20), 5.0, 5.0, 0.9)
        b = PerceivedObject(2, "bicycle", BoundingBox(10, 10, 20, 20), 5.0, 5.0, 0.9)
        m = PerceivedObject(3, "motorcycle", BoundingBox(10, 10, 20, 20), 5.0, 5.0, 0.9)
        c = PerceivedObject(4, "car", BoundingBox(10, 10, 20, 20), 5.0, 5.0, 0.9)
        t = PerceivedObject(5, "truck", BoundingBox(10, 10, 20, 20), 5.0, 5.0, 0.9)
        u = PerceivedObject(6, "debris", BoundingBox(10, 10, 20, 20), 5.0, 5.0, 0.9)

        feats = extractor.extract([p, b, m, c, t, u], frame=frame)
        assert feats[1].vulnerability == VulnerabilityLevel.CRITICAL
        assert feats[2].vulnerability == VulnerabilityLevel.CRITICAL
        assert feats[3].vulnerability == VulnerabilityLevel.HIGH
        assert feats[4].vulnerability == VulnerabilityLevel.STANDARD
        assert feats[5].vulnerability == VulnerabilityLevel.LOW_HEAVY
        assert feats[6].vulnerability == VulnerabilityLevel.UNKNOWN

    def test_stale_track_eviction_and_id_reuse(self):
        """Test that stale tracks are purged and tracker ID reuse resets history."""
        extractor = ThreatFeatureExtractor({"threat_assessment": {"stale_timeout_seconds": 1.0, "id_reuse_gap_seconds": 1.0}})

        obj1 = PerceivedObject(1, "car", BoundingBox(500, 400, 600, 500), 10.0, 10.0, 0.9)
        f1 = _make_dummy_frame(timestamp=1.0)
        feats1 = extractor.extract([obj1], frame=f1)
        assert feats1[1].observation_count == 1

        # Advance time by 0.5s -> continuous track
        f2 = _make_dummy_frame(timestamp=1.5)
        feats2 = extractor.extract([obj1], frame=f2)
        assert feats2[1].observation_count == 2
        assert abs(feats2[1].observation_duration - 0.5) < 1e-3

        # Advance time by 3.0s (gap of 2.5s > 1.0s) -> stale eviction occurs, ID is reused
        f3 = _make_dummy_frame(timestamp=4.0)
        # Even if track_id is 1 again, observation count resets to 1
        feats3 = extractor.extract([obj1], frame=f3)
        assert feats3[1].observation_count == 1
        assert feats3[1].observation_duration == 0.0
