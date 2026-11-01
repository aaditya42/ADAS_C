from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from src.behavior.base import (
    BaseAnimalBehaviorClassifier,
    BaseHumanBehaviorClassifier,
    LaneContext,
    MotionEstimate,
    TrackHistory,
)
from src.behavior.subsystem import BehaviorSubsystem
from src.memory.track_history_manager import TrackHistoryManager
from src.pose.history import PoseHistoryManager
from src.schemas import BoundingBox
from src.schemas.behavior import BehaviorFrame, BehaviorLabel, BehaviorState
from src.schemas.lane_info import LaneInfo, LaneLine
from src.schemas.pose import AnimalPose, HumanPose, Keypoint, PoseFrame


class TestBehaviorSubsystemConstruction:

    def test_disabled_config_returns_empty_frame_on_process(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": False}})
        assert not subsystem.enabled

    def test_enabled_config(self) -> None:
        subsystem = BehaviorSubsystem({
            "behavior": {"enabled": True},
        })
        assert subsystem.enabled


class TestBehaviorSubsystemProcess:

    @pytest.fixture
    def mock_human_classifier(self) -> MagicMock:
        cls = MagicMock(spec=BaseHumanBehaviorClassifier)
        cls.classify.return_value = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_WALKING,
            secondary_labels=(),
            confidence=0.9,
            timestamp=0.0,
        )
        return cls

    @pytest.fixture
    def mock_animal_classifier(self) -> MagicMock:
        cls = MagicMock(spec=BaseAnimalBehaviorClassifier)
        cls.classify.return_value = BehaviorState(
            track_id=5,
            entity_type="animal",
            primary_label=BehaviorLabel.ANIMAL_STANDING,
            secondary_labels=(),
            confidence=0.8,
            timestamp=0.0,
        )
        return cls

    @pytest.fixture
    def pose_frame(self) -> PoseFrame:
        return PoseFrame(
            frame_id=10,
            timestamp=1.0,
            humans=(
                HumanPose(
                    track_id=1,
                    detection_id=100,
                    keypoints=(),
                    bbox=(100, 200, 150, 400),
                    overall_confidence=0.95,
                    body_orientation_deg=None,
                    head_orientation_deg=None,
                    timestamp=1.0,
                ),
                HumanPose(
                    track_id=2,
                    detection_id=101,
                    keypoints=(),
                    bbox=(200, 250, 260, 450),
                    overall_confidence=0.9,
                    body_orientation_deg=None,
                    head_orientation_deg=None,
                    timestamp=1.0,
                ),
            ),
            animals=(),
        )

    @pytest.fixture
    def animal_pose_frame(self) -> PoseFrame:
        return PoseFrame(
            frame_id=10,
            timestamp=1.0,
            humans=(),
            animals=(
                AnimalPose(
                    track_id=5,
                    detection_id=200,
                    keypoints=(),
                    bbox=(300, 300, 400, 500),
                    overall_confidence=0.85,
                    body_orientation_deg=None,
                    head_orientation_deg=None,
                    species="dog",
                    timestamp=1.0,
                ),
            ),
        )

    @pytest.fixture
    def track_history_mgr(self) -> TrackHistoryManager:
        mgr = TrackHistoryManager()
        # Manually inject a snapshot for track 1
        from src.memory.track_history_manager import TrackSnapshot
        mgr._history[1] = __import__("collections").deque([
            TrackSnapshot(
                frame_id=9, timestamp=0.9,
                bbox=BoundingBox(100, 200, 150, 400),
                distance=10.0, depth_confidence=0.9,
            ),
            TrackSnapshot(
                frame_id=10, timestamp=1.0,
                bbox=BoundingBox(105, 205, 155, 405),
                distance=10.0, depth_confidence=0.9,
            ),
        ], maxlen=60)
        return mgr

    def test_empty_pose_returns_empty_frame(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": True}})
        result = subsystem.process(
            pose_frame=None,
            pose_history_mgr=MagicMock(spec=PoseHistoryManager),
            track_history_mgr=MagicMock(spec=TrackHistoryManager),
            motion_frame={},
            lane_info=None,
            frame_id=10,
            timestamp=1.0,
        )
        assert isinstance(result, BehaviorFrame)
        assert result.frame_id == 10
        assert result.timestamp == 1.0
        assert len(result.states) == 0

    def test_returns_state_for_each_human(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": True}})
        # Replace classifiers with mocks
        subsystem._human_classifier = MagicMock(spec=BaseHumanBehaviorClassifier)
        subsystem._human_classifier.classify.return_value = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_WALKING,
            secondary_labels=(),
            confidence=0.9,
            timestamp=1.0,
        )
        pose_frame = PoseFrame(
            frame_id=10, timestamp=1.0,
            humans=(
                HumanPose(
                    track_id=1, detection_id=100, keypoints=(),
                    bbox=(0, 0, 10, 20),
                    overall_confidence=0.9, body_orientation_deg=None,
                    head_orientation_deg=None, timestamp=1.0,
                ),
            ),
            animals=(),
        )
        result = subsystem.process(
            pose_frame=pose_frame,
            pose_history_mgr=MagicMock(spec=PoseHistoryManager),
            track_history_mgr=MagicMock(spec=TrackHistoryManager),
            motion_frame={},
            lane_info=None,
            frame_id=10,
            timestamp=1.0,
        )
        assert len(result.states) == 1
        assert result.states[0].track_id == 1
        assert result.states[0].primary_label == BehaviorLabel.HUMAN_WALKING

    def test_skips_track_without_track_id(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": True}})
        subsystem._human_classifier = MagicMock(spec=BaseHumanBehaviorClassifier)
        pose_frame = PoseFrame(
            frame_id=10, timestamp=1.0,
            humans=(
                HumanPose(
                    track_id=None, detection_id=100, keypoints=(),
                    bbox=(0, 0, 10, 20),
                    overall_confidence=0.9, body_orientation_deg=None,
                    head_orientation_deg=None, timestamp=1.0,
                ),
            ),
            animals=(),
        )
        result = subsystem.process(
            pose_frame=pose_frame,
            pose_history_mgr=MagicMock(spec=PoseHistoryManager),
            track_history_mgr=MagicMock(spec=TrackHistoryManager),
            motion_frame={},
            lane_info=None,
            frame_id=10,
            timestamp=1.0,
        )
        assert len(result.states) == 0
        subsystem._human_classifier.classify.assert_not_called()

    def test_lane_context_built_from_lane_info(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": True}})
        subsystem._human_classifier = MagicMock(spec=BaseHumanBehaviorClassifier)
        subsystem._human_classifier.classify.return_value = BehaviorState(
            track_id=1, entity_type="human",
            primary_label=BehaviorLabel.HUMAN_STANDING,
            secondary_labels=(), confidence=0.9, timestamp=1.0,
        )
        lane_info = LaneInfo(
            left_lane=LaneLine(coeffs=(0.0, 0.0, 100.0), confidence=0.9),
            right_lane=LaneLine(coeffs=(0.0, 0.0, 500.0), confidence=0.9),
            lane_center=None,
            confidence=0.9,
        )
        pose_frame = PoseFrame(
            frame_id=10, timestamp=1.0,
            humans=(
                HumanPose(
                    track_id=1, detection_id=100, keypoints=(),
                    bbox=(200, 300, 220, 400),
                    overall_confidence=0.9, body_orientation_deg=None,
                    head_orientation_deg=None, timestamp=1.0,
                ),
            ),
            animals=(),
        )
        result = subsystem.process(
            pose_frame=pose_frame,
            pose_history_mgr=MagicMock(spec=PoseHistoryManager),
            track_history_mgr=MagicMock(spec=TrackHistoryManager),
            motion_frame={},
            lane_info=lane_info,
            frame_id=10,
            timestamp=1.0,
        )
        call_kwargs = subsystem._human_classifier.classify.call_args[1]
        lc = call_kwargs["lane_context"]
        assert isinstance(lc, LaneContext)
        assert lc.left_boundary is lane_info.left_lane
        assert lc.right_boundary is lane_info.right_lane
        assert lc.confidence == 0.9

    def test_motion_estimate_built(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": True}})
        subsystem._human_classifier = MagicMock(spec=BaseHumanBehaviorClassifier)
        subsystem._human_classifier.classify.return_value = BehaviorState(
            track_id=1, entity_type="human",
            primary_label=BehaviorLabel.HUMAN_STANDING,
            secondary_labels=(), confidence=0.9, timestamp=1.0,
        )
        from src.schemas.motion import RelativeMotion
        motion_frame = {
            1: RelativeMotion(
                track_id=1, speed_mps=1.5, acceleration_mps2=0.0,
                direction="approaching", confidence=0.8,
            ),
        }
        pose_frame = PoseFrame(
            frame_id=10, timestamp=1.0,
            humans=(
                HumanPose(
                    track_id=1, detection_id=100, keypoints=(),
                    bbox=(0, 0, 10, 20),
                    overall_confidence=0.9, body_orientation_deg=None,
                    head_orientation_deg=None, timestamp=1.0,
                ),
            ),
            animals=(),
        )
        result = subsystem.process(
            pose_frame=pose_frame,
            pose_history_mgr=MagicMock(spec=PoseHistoryManager),
            track_history_mgr=MagicMock(spec=TrackHistoryManager),
            motion_frame=motion_frame,
            lane_info=None,
            frame_id=10,
            timestamp=1.0,
        )
        call_kwargs = subsystem._human_classifier.classify.call_args[1]
        me = call_kwargs["motion"]
        assert isinstance(me, MotionEstimate)
        assert me.speed_mps == 1.5
        assert me.track_id == 1

    def test_null_classifiers_with_pose_returns_empty_states(self) -> None:
        subsystem = BehaviorSubsystem({"behavior": {"enabled": True}})
        # Null classifiers means both are None
        subsystem._human_classifier = None
        subsystem._animal_classifier = None
        pose_frame = PoseFrame(
            frame_id=10, timestamp=1.0,
            humans=(
                HumanPose(
                    track_id=1, detection_id=100, keypoints=(),
                    bbox=(0, 0, 10, 20),
                    overall_confidence=0.9, body_orientation_deg=None,
                    head_orientation_deg=None, timestamp=1.0,
                ),
            ),
            animals=(),
        )
        result = subsystem.process(
            pose_frame=pose_frame,
            pose_history_mgr=MagicMock(spec=PoseHistoryManager),
            track_history_mgr=MagicMock(spec=TrackHistoryManager),
            motion_frame={},
            lane_info=None,
            frame_id=10,
            timestamp=1.0,
        )
        assert len(result.states) == 0
