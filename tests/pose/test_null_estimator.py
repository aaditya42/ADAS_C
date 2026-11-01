from __future__ import annotations

import numpy as np
import pytest

from src.pose.base import BaseAnimalPoseEstimator, BaseHumanPoseEstimator
from src.pose.null_estimator import NullAnimalPoseEstimator, NullHumanPoseEstimator
from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox
from src.schemas.pose import AnimalPose, HumanPose


class TestNullHumanPoseEstimator:
    """Tests for the NullHumanPoseEstimator class."""

    def test_is_instance_of_base(self) -> None:
        estimator = NullHumanPoseEstimator()
        assert isinstance(estimator, BaseHumanPoseEstimator)

    def test_estimate_returns_empty_sequence(self) -> None:
        estimator = NullHumanPoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame, [], [])
        assert len(result) == 0

    def test_returns_tuple(self) -> None:
        estimator = NullHumanPoseEstimator()
        result = estimator.estimate(
            np.zeros((100, 100, 3), dtype=np.uint8), [], []
        )
        assert isinstance(result, tuple)

    def test_estimate_with_detections(self) -> None:
        estimator = NullHumanPoseEstimator()
        bbox = BoundingBox(0.0, 0.0, 10.0, 10.0)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=bbox),
        ]
        tracks = [
            TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=bbox),
        ]
        result = estimator.estimate(
            np.zeros((480, 640, 3), dtype=np.uint8), detections, tracks
        )
        assert len(result) == 0

    def test_multiple_calls_return_empty(self) -> None:
        estimator = NullHumanPoseEstimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(5):
            result = estimator.estimate(frame, [], [])
            assert len(result) == 0

    def test_mutability_of_input_not_affected(self) -> None:
        """Verify the null estimator does not mutate input arguments."""
        estimator = NullHumanPoseEstimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dets: list[Detection] = []
        trks: list[TrackedObject] = []
        estimator.estimate(frame, dets, trks)
        # Should still be valid and unchanged
        assert frame.shape == (100, 100, 3)
        assert dets == []
        assert trks == []


class TestNullAnimalPoseEstimator:
    """Tests for the NullAnimalPoseEstimator class."""

    def test_is_instance_of_base(self) -> None:
        estimator = NullAnimalPoseEstimator()
        assert isinstance(estimator, BaseAnimalPoseEstimator)

    def test_estimate_returns_empty_sequence(self) -> None:
        estimator = NullAnimalPoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame, [], [])
        assert len(result) == 0

    def test_returns_tuple(self) -> None:
        estimator = NullAnimalPoseEstimator()
        result = estimator.estimate(
            np.zeros((100, 100, 3), dtype=np.uint8), [], []
        )
        assert isinstance(result, tuple)

    def test_estimate_with_tracks(self) -> None:
        estimator = NullAnimalPoseEstimator()
        bbox = BoundingBox(0.0, 0.0, 10.0, 10.0)
        detections = [
            Detection(class_id=1, class_name="dog", confidence=0.8, bbox=bbox),
        ]
        tracks = [
            TrackedObject(track_id=2, class_name="dog", confidence=0.8, bbox=bbox),
        ]
        result = estimator.estimate(
            np.zeros((480, 640, 3), dtype=np.uint8), detections, tracks
        )
        assert len(result) == 0

    def test_multiple_calls_return_empty(self) -> None:
        estimator = NullAnimalPoseEstimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(5):
            result = estimator.estimate(frame, [], [])
            assert len(result) == 0

    def test_mutability_of_input_not_affected(self) -> None:
        estimator = NullAnimalPoseEstimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dets: list[Detection] = []
        trks: list[TrackedObject] = []
        estimator.estimate(frame, dets, trks)
        assert frame.shape == (100, 100, 3)
        assert dets == []
        assert trks == []


class TestAbstractBaseClasses:
    """Verify the ABCs enforce the contract."""

    def test_human_pose_estimator_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseHumanPoseEstimator()  # type: ignore[abstract]

    def test_animal_pose_estimator_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseAnimalPoseEstimator()  # type: ignore[abstract]
