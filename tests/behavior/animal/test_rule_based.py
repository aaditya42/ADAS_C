from __future__ import annotations

from typing import Optional, Tuple

import pytest

from src.behavior.animal.rule_based import RuleBasedAnimalBehaviorClassifier
from src.behavior.base import LaneContext, MotionEstimate, TrackHistory
from src.behavior.config import BehaviorConfig
from src.schemas.behavior import BehaviorLabel
from src.schemas.lane_info import LaneLine
from src.schemas.pose import AnimalKeypointName, AnimalPose, Keypoint


# --------------------------------------------------------------------------- #
#  Shared fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def classifier() -> RuleBasedAnimalBehaviorClassifier:
    return RuleBasedAnimalBehaviorClassifier(BehaviorConfig())


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _kp(name: str, x: float, y: float, visible: bool = True) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, confidence=0.9, visible=visible)


def _make_animal_pose(
    track_id: int = 1,
    kps: Optional[dict[str, Keypoint]] = None,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 150.0),
    body_orientation_deg: Optional[float] = None,
    head_orientation_deg: Optional[float] = None,
    species: str = "dog",
) -> AnimalPose:
    """Create an AnimalPose. If kps is None, default standing keypoints used."""
    if kps is None:
        kps = _default_standing_animal_kps()
    return AnimalPose(
        track_id=track_id,
        detection_id=0,
        keypoints=_ordered_kps(kps),
        bbox=bbox,
        overall_confidence=0.9,
        body_orientation_deg=body_orientation_deg,
        head_orientation_deg=head_orientation_deg,
        species=species,
        timestamp=0.0,
    )


def _ordered_kps(kps: dict[str, Keypoint]) -> Tuple[Keypoint, ...]:
    return tuple(
        kps.get(
            k.value,
            Keypoint(name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False),
        )
        for k in AnimalKeypointName
    )


def _default_standing_animal_kps() -> dict[str, Keypoint]:
    """Standing dog: horizontal spine (neck→mid_back), head level."""
    def e(name: AnimalKeypointName) -> str:
        return name.value
    return {
        e(AnimalKeypointName.LEFT_EYE): _kp(e(AnimalKeypointName.LEFT_EYE), 30, 60),
        e(AnimalKeypointName.RIGHT_EYE): _kp(e(AnimalKeypointName.RIGHT_EYE), 35, 60),
        e(AnimalKeypointName.NOSE): _kp(e(AnimalKeypointName.NOSE), 20, 70),
        e(AnimalKeypointName.NECK): _kp(e(AnimalKeypointName.NECK), 50, 70),
        e(AnimalKeypointName.LEFT_FRONT_SHOULDER): _kp(e(AnimalKeypointName.LEFT_FRONT_SHOULDER), 60, 80),
        e(AnimalKeypointName.RIGHT_FRONT_SHOULDER): _kp(e(AnimalKeypointName.RIGHT_FRONT_SHOULDER), 65, 80),
        e(AnimalKeypointName.LEFT_FRONT_PAW): _kp(e(AnimalKeypointName.LEFT_FRONT_PAW), 60, 140),
        e(AnimalKeypointName.RIGHT_FRONT_PAW): _kp(e(AnimalKeypointName.RIGHT_FRONT_PAW), 65, 140),
        e(AnimalKeypointName.LEFT_BACK_SHOULDER): _kp(e(AnimalKeypointName.LEFT_BACK_SHOULDER), 110, 75),
        e(AnimalKeypointName.RIGHT_BACK_SHOULDER): _kp(e(AnimalKeypointName.RIGHT_BACK_SHOULDER), 115, 75),
        e(AnimalKeypointName.LEFT_BACK_PAW): _kp(e(AnimalKeypointName.LEFT_BACK_PAW), 110, 140),
        e(AnimalKeypointName.RIGHT_BACK_PAW): _kp(e(AnimalKeypointName.RIGHT_BACK_PAW), 115, 140),
        e(AnimalKeypointName.TAIL): _kp(e(AnimalKeypointName.TAIL), 130, 60),
        e(AnimalKeypointName.LEFT_EAR): _kp(e(AnimalKeypointName.LEFT_EAR), 40, 55),
        e(AnimalKeypointName.RIGHT_EAR): _kp(e(AnimalKeypointName.RIGHT_EAR), 45, 55),
        e(AnimalKeypointName.HEAD_TOP): _kp(e(AnimalKeypointName.HEAD_TOP), 35, 55),
        e(AnimalKeypointName.MID_BACK): _kp(e(AnimalKeypointName.MID_BACK), 85, 78),
    }


def _make_pose_history(pose: AnimalPose) -> Tuple[AnimalPose, ...]:
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
    """ANIMAL_STANDING / ANIMAL_WALKING / ANIMAL_RUNNING."""

    def test_standing(self, classifier: RuleBasedAnimalBehaviorClassifier) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_STANDING
        assert state.confidence > 0.0

    def test_walking(self, classifier: RuleBasedAnimalBehaviorClassifier) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.8),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_WALKING
        assert state.confidence > 0.0

    def test_running(self, classifier: RuleBasedAnimalBehaviorClassifier) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=2.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_RUNNING
        assert state.confidence > 0.0

    def test_no_motion_returns_unknown(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """No motion data → UNKNOWN (speed-based classification not possible)."""
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=None,
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.UNKNOWN


# ====================================================================== #
#  Lying label
# ====================================================================== #


class TestLyingLabel:
    """ANIMAL_LYING based on low torso height + horizontal spine."""

    def test_lying(self, classifier: RuleBasedAnimalBehaviorClassifier) -> None:
        """Animal lying down → torso_height small + spine horizontal."""
        # Lying: mid_back near ground, front paws near mid_back
        # mid_back at same y as paws → torso_height ≈ 0
        pose = _make_animal_pose(
            kps=_make_lying_animal_kps(),
            bbox=(0.0, 120.0, 150.0, 160.0),  # bbox_h=40
        )
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_LYING
        assert state.confidence > 0.0

    def test_standing_not_lying(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """Standing animal → NOT LYING."""
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.ANIMAL_LYING

    def test_no_keypoints_does_not_crash(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        kps = {
            k.value: Keypoint(
                name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False
            )
            for k in AnimalKeypointName
        }
        pose = _make_animal_pose(kps=kps)
        classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )


def _make_lying_animal_kps() -> dict[str, Keypoint]:
    """Keypoints for an animal lying down."""
    kps = {}
    for k in AnimalKeypointName:
        kps[k.value] = Keypoint(
            name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False
        )
    # All on same y-level (on ground)
    y_ground = 150.0
    kps[AnimalKeypointName.MID_BACK.value] = _kp(AnimalKeypointName.MID_BACK.value, 85, y_ground)
    kps[AnimalKeypointName.NECK.value] = _kp(AnimalKeypointName.NECK.value, 50, y_ground + 2)
    kps[AnimalKeypointName.NOSE.value] = _kp(AnimalKeypointName.NOSE.value, 20, y_ground + 5)
    kps[AnimalKeypointName.LEFT_FRONT_PAW.value] = _kp(AnimalKeypointName.LEFT_FRONT_PAW.value, 60, y_ground + 3)
    kps[AnimalKeypointName.RIGHT_FRONT_PAW.value] = _kp(AnimalKeypointName.RIGHT_FRONT_PAW.value, 65, y_ground + 3)
    kps[AnimalKeypointName.LEFT_EYE.value] = _kp(AnimalKeypointName.LEFT_EYE.value, 30, y_ground)
    kps[AnimalKeypointName.RIGHT_EYE.value] = _kp(AnimalKeypointName.RIGHT_EYE.value, 35, y_ground)
    kps[AnimalKeypointName.HEAD_TOP.value] = _kp(AnimalKeypointName.HEAD_TOP.value, 35, y_ground - 2)
    kps[AnimalKeypointName.TAIL.value] = _kp(AnimalKeypointName.TAIL.value, 130, y_ground)
    return kps


# ====================================================================== #
#  Grazing label
# ====================================================================== #


class TestGrazingLabel:
    """ANIMAL_GRAZING: head down + low speed."""

    def test_grazing(self, classifier: RuleBasedAnimalBehaviorClassifier) -> None:
        """Animal with head pointed down + stationary → GRAZING."""
        pose = _make_animal_pose(kps=_make_grazing_animal_kps())
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_GRAZING
        assert state.confidence > 0.0

    def test_grazing_with_slow_movement(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """Grazing but very slow movement is still GRAZING."""
        pose = _make_animal_pose(kps=_make_grazing_animal_kps())
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.2),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_GRAZING

    def test_fast_movement_not_grazing(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """Moving fast with head down → not GRAZING (speed > threshold)."""
        pose = _make_animal_pose(kps=_make_grazing_animal_kps())
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.ANIMAL_GRAZING

    def test_head_up_not_grazing(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """Head pointing forward → not grazing."""
        pose = _make_animal_pose()  # default standing
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label != BehaviorLabel.ANIMAL_GRAZING


def _make_grazing_animal_kps() -> dict[str, Keypoint]:
    """Keypoints for an animal grazing with head down."""
    kps = {}
    for k in AnimalKeypointName:
        kps[k.value] = Keypoint(
            name=k.value, x=0.0, y=0.0, confidence=0.0, visible=False
        )
    # Standing body with nose well below neck
    y_neck = 80.0
    kps[AnimalKeypointName.MID_BACK.value] = _kp(AnimalKeypointName.MID_BACK.value, 85, y_neck + 5)
    kps[AnimalKeypointName.NECK.value] = _kp(AnimalKeypointName.NECK.value, 50, y_neck)
    # Nose is far below neck → head down
    kps[AnimalKeypointName.NOSE.value] = _kp(AnimalKeypointName.NOSE.value, 30, y_neck + 60)
    kps[AnimalKeypointName.LEFT_EYE.value] = _kp(AnimalKeypointName.LEFT_EYE.value, 35, y_neck - 5)
    kps[AnimalKeypointName.RIGHT_EYE.value] = _kp(AnimalKeypointName.RIGHT_EYE.value, 40, y_neck - 5)
    kps[AnimalKeypointName.HEAD_TOP.value] = _kp(AnimalKeypointName.HEAD_TOP.value, 40, y_neck - 10)
    kps[AnimalKeypointName.LEFT_EAR.value] = _kp(AnimalKeypointName.LEFT_EAR.value, 45, y_neck - 8)
    kps[AnimalKeypointName.RIGHT_EAR.value] = _kp(AnimalKeypointName.RIGHT_EAR.value, 48, y_neck - 8)
    kps[AnimalKeypointName.LEFT_FRONT_SHOULDER.value] = _kp(AnimalKeypointName.LEFT_FRONT_SHOULDER.value, 60, y_neck + 10)
    kps[AnimalKeypointName.RIGHT_FRONT_SHOULDER.value] = _kp(AnimalKeypointName.RIGHT_FRONT_SHOULDER.value, 65, y_neck + 10)
    kps[AnimalKeypointName.LEFT_FRONT_PAW.value] = _kp(AnimalKeypointName.LEFT_FRONT_PAW.value, 60, y_neck + 70)
    kps[AnimalKeypointName.RIGHT_FRONT_PAW.value] = _kp(AnimalKeypointName.RIGHT_FRONT_PAW.value, 65, y_neck + 70)
    kps[AnimalKeypointName.LEFT_BACK_SHOULDER.value] = _kp(AnimalKeypointName.LEFT_BACK_SHOULDER.value, 110, y_neck + 8)
    kps[AnimalKeypointName.RIGHT_BACK_SHOULDER.value] = _kp(AnimalKeypointName.RIGHT_BACK_SHOULDER.value, 115, y_neck + 8)
    kps[AnimalKeypointName.LEFT_BACK_PAW.value] = _kp(AnimalKeypointName.LEFT_BACK_PAW.value, 110, y_neck + 70)
    kps[AnimalKeypointName.RIGHT_BACK_PAW.value] = _kp(AnimalKeypointName.RIGHT_BACK_PAW.value, 115, y_neck + 70)
    kps[AnimalKeypointName.TAIL.value] = _kp(AnimalKeypointName.TAIL.value, 130, y_neck)
    return kps


# ====================================================================== #
#  Road-state labels
# ====================================================================== #


class TestRoadStateLabels:
    """ANIMAL_CROSSING_ROAD / ANIMAL_APPROACHING_ROAD."""

    def test_crossing_road(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """Animal inside lane boundaries → crossing road."""
        pose = _make_animal_pose(bbox=(600.0, 0.0, 650.0, 100.0))
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=90.0),
            lane_context=_make_lane_context(left_x=500.0, right_x=800.0),
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_CROSSING_ROAD
        assert state.confidence > 0.0
        assert state.crossing_probability is not None
        assert state.crossing_probability >= 0.7

    def test_approaching_road(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        """Off-road, near boundary, heading toward it → APPROACHING_ROAD."""
        pose = _make_animal_pose(bbox=(400.0, 0.0, 450.0, 100.0))  # foot x=425
        state = classifier.classify(
            pose_history=_make_pose_history(pose),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=1.0, heading_deg=90.0),
            lane_context=_make_lane_context(left_x=500.0, right_x=800.0),
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_APPROACHING_ROAD
        assert state.confidence > 0.0

    def test_no_lane_context_does_not_crash(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
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
        self, classifier: RuleBasedAnimalBehaviorClassifier
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
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        for speed in [-1.0, 0.0, 0.5, 2.0, 10.0]:
            state = classifier.classify(
                pose_history=_make_pose_history(_make_animal_pose()),
                track_history=_make_track_history(),
                motion=_make_motion(speed_mps=speed),
                lane_context=None,
                timestamp=0.0,
            )
            assert 0.0 <= state.confidence <= 1.0

    def test_secondary_labels_exclude_primary(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.8),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.primary_label == BehaviorLabel.ANIMAL_WALKING
        assert BehaviorLabel.ANIMAL_WALKING not in state.secondary_labels

    def test_entity_type_is_animal(
        self, classifier: RuleBasedAnimalBehaviorClassifier
    ) -> None:
        state = classifier.classify(
            pose_history=_make_pose_history(_make_animal_pose()),
            track_history=_make_track_history(),
            motion=_make_motion(speed_mps=0.0),
            lane_context=None,
            timestamp=0.0,
        )
        assert state.entity_type == "animal"
