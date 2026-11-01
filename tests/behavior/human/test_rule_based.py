from __future__ import annotations

from typing import Optional, Tuple

import pytest

from src.behavior.base import LaneContext, MotionEstimate, TrackHistory
from src.behavior.config import BehaviorConfig
from src.behavior.human.rule_based import RuleBasedHumanBehaviorClassifier
from src.schemas.behavior import BehaviorLabel
from src.schemas.lane_info import LaneLine
from src.schemas.pose import HumanKeypointName, HumanPose, Keypoint


# --------------------------------------------------------------------------- #
#  Shared fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def classifier() -> RuleBasedHumanBehaviorClassifier:
    return RuleBasedHumanBehaviorClassifier(BehaviorConfig())


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _kp(name: str, x: float, y: float, visible: bool = True) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, confidence=0.9, visible=visible)


def _make_human_pose(
    track_id: int = 1,
    kps: Optional[dict[str, Keypoint]] = None,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 50.0, 200.0),
    body_orientation_deg: Optional[float] = None,
    head_orientation_deg: Optional[float] = None,
) -> HumanPose:
    """Create a HumanPose. If kps is None, default standing keypoints are used."""
    if kps is None:
        kps = _default_standing_kps()
    return HumanPose(
        track_id=track_id,
        detection_id=0,
        keypoints=_ordered_kps(kps),
        bbox=bbox,
        overall_confidence=0.9,
        body_orientation_deg=body_orientation_deg,
        head_orientation_deg=head_orientation_deg,
        timestamp=0.0,
    )


def _ordered_kps(kps: dict[str, Keypoint]) -> Tuple[Keypoint, ...]:
    return tuple(
        kps.get(
            k.value,
            Keypoint(name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False),
        )
        for k in HumanKeypointName
    )


def _default_standing_kps() -> dict[str, Keypoint]:
    return {
        HumanKeypointName.NOSE.value: _kp("nose", 25, 30),
        HumanKeypointName.LEFT_EYE.value: _kp("left_eye", 22, 28),
        HumanKeypointName.RIGHT_EYE.value: _kp("right_eye", 28, 28),
        HumanKeypointName.LEFT_EAR.value: _kp("left_ear", 18, 30),
        HumanKeypointName.RIGHT_EAR.value: _kp("right_ear", 32, 30),
        HumanKeypointName.LEFT_SHOULDER.value: _kp("left_shoulder", 20, 50),
        HumanKeypointName.RIGHT_SHOULDER.value: _kp("right_shoulder", 30, 50),
        HumanKeypointName.LEFT_ELBOW.value: _kp("left_elbow", 18, 80),
        HumanKeypointName.RIGHT_ELBOW.value: _kp("right_elbow", 32, 80),
        HumanKeypointName.LEFT_WRIST.value: _kp("left_wrist", 18, 110),
        HumanKeypointName.RIGHT_WRIST.value: _kp("right_wrist", 32, 110),
        HumanKeypointName.LEFT_HIP.value: _kp("left_hip", 22, 150),
        HumanKeypointName.RIGHT_HIP.value: _kp("right_hip", 28, 150),
        HumanKeypointName.LEFT_KNEE.value: _kp("left_knee", 22, 180),
        HumanKeypointName.RIGHT_KNEE.value: _kp("right_knee", 28, 180),
        HumanKeypointName.LEFT_ANKLE.value: _kp("left_ankle", 22, 200),
        HumanKeypointName.RIGHT_ANKLE.value: _kp("right_ankle", 28, 200),
    }


def _make_pose_history(pose: HumanPose) -> Tuple[HumanPose, ...]:
    return (pose,)


def _make_motion(
    speed_mps: Optional[float] = None,
    heading_deg: Optional[float] = None,
) -> Optional[MotionEstimate]:
    if speed_mps is None and heading_deg is None:
        return None
    return MotionEstimate(
        track_id=1,
        speed_mps=speed_mps,
        heading_deg=heading_deg,
        acceleration_mps2=None,
        confidence=0.8,
    )


def _make_lane_context(
    left_x: float = 500.0,
    right_x: float = 800.0,
) -> LaneContext:
    return LaneContext(
        left_boundary=LaneLine(coeffs=(0.0, 0.0, left_x), confidence=0.9),
        right_boundary=LaneLine(coeffs=(0.0, 0.0, right_x), confidence=0.9),
        confidence=0.9,
    )


def _make_track_history(track_id: int = 1) -> TrackHistory:
    return TrackHistory(
        track_id=track_id,
        positions=((100.0, 200.0),),
        timestamps=(0.0,),
    )


# ====================================================================== #
#  Motion-based labels
# ====================================================================== #


class TestMotionLabels:
    """STANDING / WALKING / RUNNING based on speed."""

    def test_standing(self, classifier: RuleBasedHumanBehaviorClassifier) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_STANDING
        assert state.confidence > 0.0

    def test_walking(self, classifier: RuleBasedHumanBehaviorClassifier) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_WALKING
        assert state.confidence > 0.0

    def test_running(self, classifier: RuleBasedHumanBehaviorClassifier) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=2.5),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_RUNNING
        assert state.confidence > 0.0

    def test_no_motion_returns_unknown(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """No motion data → UNKNOWN (speed-based classification not possible)."""
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=None,
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.UNKNOWN


# ====================================================================== #
#  Fallen label
# ====================================================================== #


class TestFallenLabel:
    """FALLEN based on torso angle + wider-than-tall bbox."""

    def test_fallen(self, classifier: RuleBasedHumanBehaviorClassifier) -> None:
        """Person with horizontal torso and bbox wider than tall → FALLEN."""
        pose = _make_human_pose(
            kps=_make_fallen_kps(),
            bbox=(0.0, 50.0, 250.0, 150.0),  # w=250, h=100 → wider than tall
        )
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_FALLEN
        assert state.confidence > 0.0

    def test_standing_not_fallen(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Upright person → NOT FALLEN."""
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.HUMAN_FALLEN

    def test_no_keypoints_does_not_crash(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Missing keypoints → no fallen detection, no crash."""
        kps = {
            k.value: Keypoint(
                name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False
            )
            for k in HumanKeypointName
        }
        pose = _make_human_pose(kps=kps)
        classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )


def _make_fallen_kps() -> dict[str, Keypoint]:
    """Keypoints for a person lying horizontally."""
    kps = {}
    for k in HumanKeypointName:
        kps[k.value] = Keypoint(
            name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False
        )
    # Torso (hip→shoulder) is horizontal:
    # hip at left, shoulder further left → "proper" lying down
    kps[HumanKeypointName.LEFT_HIP.value] = _kp("left_hip", 100, 100)
    kps[HumanKeypointName.RIGHT_HIP.value] = _kp("right_hip", 105, 100)
    kps[HumanKeypointName.LEFT_SHOULDER.value] = _kp("left_shoulder", 200, 100)
    kps[HumanKeypointName.RIGHT_SHOULDER.value] = _kp("right_shoulder", 205, 100)
    return kps


# ====================================================================== #
#  Distracted label
# ====================================================================== #


class TestDistractedLabel:
    """DISTRACTED based on head vs motion direction."""

    def test_distracted(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Head facing 90°, motion heading 0° → large diff → DISTRACTED."""
        pose = _make_human_pose(head_orientation_deg=90.0)
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_DISTRACTED
        assert state.confidence > 0.0

    def test_not_distracted_when_aligned(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Head and motion both pointing up → no distraction."""
        pose = _make_human_pose(head_orientation_deg=0.0)
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.HUMAN_DISTRACTED

    def test_no_head_orientation_not_distracted(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """No head orientation data → no distracted."""
        pose = _make_human_pose(head_orientation_deg=None)
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.HUMAN_DISTRACTED

    def test_no_motion_not_distracted(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """No motion → no distracted classification."""
        pose = _make_human_pose(head_orientation_deg=90.0)
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=None,
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.HUMAN_DISTRACTED


# ====================================================================== #
#  Road-state labels
# ====================================================================== #


class TestRoadStateLabels:
    """CROSSING_ROAD / APPROACHING_ROAD / LEAVING_ROAD."""

    def test_crossing_road(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Entity deep inside lane boundaries → crossing road."""
        # Foot at x=650, inside lane [500, 800]
        pose = _make_human_pose(bbox=(625.0, 0.0, 675.0, 200.0))
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=90.0),
            lane_context=_make_lane_context(left_x=500.0, right_x=800.0),
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_CROSSING_ROAD
        assert state.confidence > 0.0
        assert state.crossing_probability is not None
        assert state.crossing_probability >= 0.7

    def test_approaching_road(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Off-road, near lane, heading toward it → APPROACHING_ROAD."""
        # Foot at x=450, left lane at x=500 → 50px away (< 100px threshold)
        # Heading 90° (right) → toward road
        pose = _make_human_pose(bbox=(425.0, 0.0, 475.0, 200.0))
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=90.0),
            lane_context=_make_lane_context(left_x=500.0, right_x=800.0),
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_APPROACHING_ROAD
        assert state.confidence > 0.0

    def test_leaving_road(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        """Inside lane, near edge, heading toward edge → LEAVING_ROAD."""
        # Foot at x=510, inside lane [500, 800], 10px from left boundary
        # Heading -90° (left) → toward left boundary
        pose = _make_human_pose(bbox=(500.0, 0.0, 520.0, 200.0))  # foot x=510
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.5, heading_deg=-90.0),
            lane_context=_make_lane_context(left_x=500.0, right_x=800.0),
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_LEAVING_ROAD
        assert state.confidence > 0.0

    def test_no_lane_context_does_not_crash(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert isinstance(state.primary_label, BehaviorLabel)
        assert state.crossing_probability is None


# ====================================================================== #
#  UNKNOWN label & edge cases
# ====================================================================== #


class TestUnknownAndEdgeCases:
    def test_empty_pose_history_returns_unknown(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=(),
            track_history=_make_track_history(),
            motion=None,
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.UNKNOWN
        assert state.confidence == 0.0

    def test_confidence_never_exceeds_range(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        for speed in [-1.0, 0.0, 1.0, 5.0, 100.0]:
            state = classifier.classify(
                pose_history=_make_pose_history(_make_human_pose()),
                track_history=_make_track_history(),
                motion=_make_motion(speed_mps=speed),
                lane_context=None,
                timestamp=0.0,
            )
            assert 0.0 <= state.confidence <= 1.0

    def test_secondary_labels_exclude_primary(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.HUMAN_WALKING
        assert BehaviorLabel.HUMAN_WALKING not in state.secondary_labels

    def test_entity_type_is_human(
        self, classifier: RuleBasedHumanBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_human_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.entity_type == "human"


# ====================================================================== #
#  Animal __init__ test placeholder
# ====================================================================== #


class TestAnimalInit:
    """Placeholder — animal tests in tests/behavior/animal/"""
