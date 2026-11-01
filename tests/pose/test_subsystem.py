from __future__ import annotations

from typing import Any, List, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.pose.subsystem import PoseSubsystem
from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox
from src.schemas.pose import AnimalPose, HumanPose, PoseFrame


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def disabled_config() -> dict[str, Any]:
    return {"pose": {"enabled": False}}


@pytest.fixture
def null_config() -> dict[str, Any]:
    return {
        "pose": {
            "enabled": True,
            "human": {"estimator": "null"},
            "animal": {"estimator": "null"},
        }
    }


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def empty_detections() -> list[Detection]:
    return []


@pytest.fixture
def empty_tracks() -> list[TrackedObject]:
    return []


@pytest.fixture
def person_detections() -> list[Detection]:
    return [
        Detection(
            class_id=0, class_name="person", confidence=0.95,
            bbox=BoundingBox(100, 50, 300, 400),
        ),
    ]


@pytest.fixture
def mock_tracks() -> list[TrackedObject]:
    return [
        TrackedObject(
            track_id=5, class_name="person", confidence=0.95,
            bbox=BoundingBox(100, 50, 300, 400),
        ),
    ]


# --------------------------------------------------------------------------- #
#  PoseSubsystem tests
# --------------------------------------------------------------------------- #


class TestPoseSubsystemConstruction:
    """Tests for PoseSubsystem construction."""

    def test_construct_with_disabled_config(
        self, disabled_config: dict[str, Any]
    ) -> None:
        """Even with disabled config, subsystem constructs with null estimators."""
        sub = PoseSubsystem(disabled_config)
        assert sub._human_estimator is not None
        assert sub._animal_estimator is not None

    def test_construct_with_null_config(
        self, null_config: dict[str, Any]
    ) -> None:
        sub = PoseSubsystem(null_config)
        assert sub._human_estimator is not None
        assert sub._animal_estimator is not None

    def test_history_enabled_by_default(self, null_config: dict[str, Any]) -> None:
        sub = PoseSubsystem(null_config)
        assert sub.history is not None

    def test_history_disabled(self) -> None:
        sub = PoseSubsystem({"pose": {"history": {"enabled": False}}})
        assert sub.history is None


class TestPoseSubsystemProcess:
    """Tests for the ``process()`` method."""

    def test_returns_poseframe(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        empty_detections: list[Detection],
        empty_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        result = sub.process(
            frame=synthetic_frame,
            detections=empty_detections,
            tracks=empty_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert isinstance(result, PoseFrame)

    def test_frame_id_and_timestamp(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        empty_detections: list[Detection],
        empty_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        result = sub.process(
            frame=synthetic_frame,
            detections=empty_detections,
            tracks=empty_tracks,
            frame_id=42,
            timestamp=1.5,
        )
        assert result.frame_id == 42
        assert result.timestamp == 1.5

    def test_empty_input_returns_empty_poseframe(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        empty_detections: list[Detection],
        empty_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        result = sub.process(
            frame=synthetic_frame,
            detections=empty_detections,
            tracks=empty_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert len(result.humans) == 0
        assert len(result.animals) == 0

    def test_null_estimators_with_person_detections_returns_empty_humans(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        person_detections: list[Detection],
        empty_tracks: list[TrackedObject],
    ) -> None:
        """Null estimators ignore person detections and return empty."""
        sub = PoseSubsystem(null_config)
        result = sub.process(
            frame=synthetic_frame,
            detections=person_detections,
            tracks=empty_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert len(result.humans) == 0
        assert len(result.animals) == 0


class TestPoseSubsystemErrorHandling:
    """Tests for error handling inside the subsystem."""

    def test_human_estimator_exception_returns_empty_humans(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        person_detections: list[Detection],
        mock_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        # Replace the human estimator with a failing one
        failing_mock = MagicMock()
        failing_mock.estimate.side_effect = RuntimeError("Human model crashed")
        sub._human_estimator = failing_mock

        result = sub.process(
            frame=synthetic_frame,
            detections=person_detections,
            tracks=mock_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        # Humans should be empty, animals unaffected
        assert len(result.humans) == 0
        assert isinstance(result, PoseFrame)
        # The error was logged, not raised
        failing_mock.estimate.assert_called_once()

    def test_animal_estimator_exception_returns_empty_animals(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        person_detections: list[Detection],
        mock_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        failing_mock = MagicMock()
        failing_mock.estimate.side_effect = RuntimeError("Animal model crashed")
        sub._animal_estimator = failing_mock

        result = sub.process(
            frame=synthetic_frame,
            detections=person_detections,
            tracks=mock_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert len(result.animals) == 0
        assert isinstance(result, PoseFrame)

    def test_both_estimators_fail_gracefully(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        person_detections: list[Detection],
        mock_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        sub._human_estimator = MagicMock()
        sub._human_estimator.estimate.side_effect = RuntimeError("Human fail")
        sub._animal_estimator = MagicMock()
        sub._animal_estimator.estimate.side_effect = RuntimeError("Animal fail")

        result = sub.process(
            frame=synthetic_frame,
            detections=person_detections,
            tracks=mock_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert len(result.humans) == 0
        assert len(result.animals) == 0
        assert isinstance(result, PoseFrame)

    def test_history_is_updated_after_process(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        empty_detections: list[Detection],
        empty_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)
        from src.schemas.pose import (AnimalKeypointName, HumanKeypointName,
                                      Keypoint)

        human_kps = tuple(
            Keypoint(name=kp.value, x=0.0, y=0.0, confidence=0.0, visible=False)
            for kp in HumanKeypointName
        )
        animal_kps = tuple(
            Keypoint(name=kp.value, x=0.0, y=0.0, confidence=0.0, visible=False)
            for kp in AnimalKeypointName
        )

        human_pose = HumanPose(
            track_id=1, detection_id=0, keypoints=human_kps,
            bbox=(0.0, 0.0, 10.0, 20.0), overall_confidence=0.9,
            body_orientation_deg=None, head_orientation_deg=None,
            timestamp=0.0,
        )
        animal_pose = AnimalPose(
            track_id=10, detection_id=0, keypoints=animal_kps,
            bbox=(0.0, 0.0, 10.0, 20.0), overall_confidence=0.85,
            body_orientation_deg=None, head_orientation_deg=None,
            species="dog", timestamp=0.0,
        )

        human_mock = MagicMock()
        human_mock.estimate.return_value = [human_pose]
        animal_mock = MagicMock()
        animal_mock.estimate.return_value = [animal_pose]

        sub._human_estimator = human_mock
        sub._animal_estimator = animal_mock

        sub.process(
            frame=synthetic_frame,
            detections=empty_detections,
            tracks=empty_tracks,
            frame_id=0,
            timestamp=1.0,
        )

        assert sub.history is not None
        assert 1 in sub.history.active_track_ids()
        assert 10 in sub.history.active_track_ids()
        assert len(sub.history.get_human_history(1)) == 1
        assert len(sub.history.get_animal_history(10)) == 1


class TestPoseSubsystemWithMockedEstimator:
    """Tests with a mock estimator that returns synthetic poses."""

    def test_unified_poseframe_combines_human_and_animal(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        person_detections: list[Detection],
        mock_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)

        # Mock human estimator to return one synthetic HumanPose
        human_mock = MagicMock()
        human_kp = _make_dummy_human_keypoints()
        human_pose = HumanPose(
            track_id=5, detection_id=0, keypoints=human_kp,
            bbox=(100, 50, 300, 400), overall_confidence=0.92,
            body_orientation_deg=None, head_orientation_deg=None,
            timestamp=0.0,
        )
        human_mock.estimate.return_value = [human_pose]

        # Mock animal estimator to return one synthetic AnimalPose
        animal_mock = MagicMock()
        animal_kp = _make_dummy_animal_keypoints()
        animal_pose = AnimalPose(
            track_id=10, detection_id=1, keypoints=animal_kp,
            bbox=(200, 100, 500, 400), overall_confidence=0.88,
            body_orientation_deg=None, head_orientation_deg=None,
            species="dog", timestamp=0.0,
        )
        animal_mock.estimate.return_value = [animal_pose]

        sub._human_estimator = human_mock
        sub._animal_estimator = animal_mock

        result = sub.process(
            frame=synthetic_frame,
            detections=person_detections,
            tracks=mock_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert len(result.humans) == 1
        assert len(result.animals) == 1
        assert result.humans[0].track_id == 5
        assert result.animals[0].species == "dog"

    def test_error_in_one_estimator_does_not_affect_other(
        self, null_config: dict[str, Any],
        synthetic_frame: np.ndarray,
        person_detections: list[Detection],
        mock_tracks: list[TrackedObject],
    ) -> None:
        sub = PoseSubsystem(null_config)

        # Human fails, animal works
        human_mock = MagicMock()
        human_mock.estimate.side_effect = ValueError("Human OOM")

        animal_mock = MagicMock()
        animal_kp = _make_dummy_animal_keypoints()
        animal_pose = AnimalPose(
            track_id=10, detection_id=1, keypoints=animal_kp,
            bbox=(200, 100, 500, 400), overall_confidence=0.88,
            body_orientation_deg=None, head_orientation_deg=None,
            species="dog", timestamp=0.0,
        )
        animal_mock.estimate.return_value = [animal_pose]

        sub._human_estimator = human_mock
        sub._animal_estimator = animal_mock

        result = sub.process(
            frame=synthetic_frame,
            detections=person_detections,
            tracks=mock_tracks,
            frame_id=0,
            timestamp=0.0,
        )
        assert len(result.humans) == 0  # failed
        assert len(result.animals) == 1  # still works
        assert result.animals[0].species == "dog"


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _make_dummy_human_keypoints() -> tuple[Any, ...]:
    """Create 17 valid COCO keypoints for testing."""
    from src.schemas.pose import HumanKeypointName, Keypoint

    names = list(HumanKeypointName)
    return tuple(
        Keypoint(name=kp.value, x=0.0, y=0.0, confidence=0.0, visible=False)
        for kp in names
    )


def _make_dummy_animal_keypoints() -> tuple[Any, ...]:
    """Create 17 valid AP-10K keypoints for testing."""
    from src.schemas.pose import AnimalKeypointName, Keypoint

    names = list(AnimalKeypointName)
    return tuple(
        Keypoint(name=kp.value, x=0.0, y=0.0, confidence=0.0, visible=False)
        for kp in names
    )
