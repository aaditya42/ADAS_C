from __future__ import annotations

import numpy as np
import pytest

from src.pose.base import BaseAnimalPoseEstimator, BaseHumanPoseEstimator
from src.pose.factory import PoseEstimatorBundle, build_pose_estimators
from src.pose.null_estimator import NullAnimalPoseEstimator, NullHumanPoseEstimator
from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox


class TestBuildPoseEstimators:
    """Tests for the ``build_pose_estimators`` factory function."""

    def test_missing_pose_section_returns_null(self) -> None:
        bundle = build_pose_estimators({})
        assert isinstance(bundle, PoseEstimatorBundle)
        assert isinstance(bundle.human, NullHumanPoseEstimator)
        assert isinstance(bundle.animal, NullAnimalPoseEstimator)

    def test_disabled_returns_null(self) -> None:
        bundle = build_pose_estimators({"pose": {"enabled": False}})
        assert isinstance(bundle.human, NullHumanPoseEstimator)
        assert isinstance(bundle.animal, NullAnimalPoseEstimator)

    def test_null_config_returns_null(self) -> None:
        bundle = build_pose_estimators({
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            }
        })
        assert isinstance(bundle.human, NullHumanPoseEstimator)
        assert isinstance(bundle.animal, NullAnimalPoseEstimator)

    def test_null_estimators_return_empty_results(self) -> None:
        bundle = build_pose_estimators({})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections: list[Detection] = []
        tracks: list[TrackedObject] = []

        humans = bundle.human.estimate(frame, detections, tracks)
        animals = bundle.animal.estimate(frame, detections, tracks)

        assert len(humans) == 0
        assert len(animals) == 0

    def test_null_estimators_with_real_inputs(self) -> None:
        bundle = build_pose_estimators({})
        frame = np.ones((240, 320, 3), dtype=np.uint8)
        bbox = BoundingBox(10.0, 20.0, 100.0, 200.0)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=bbox),
            Detection(class_id=1, class_name="dog", confidence=0.8, bbox=bbox),
        ]
        tracks = [
            TrackedObject(track_id=1, class_name="person", confidence=0.9, bbox=bbox),
            TrackedObject(track_id=2, class_name="dog", confidence=0.8, bbox=bbox),
        ]

        humans = bundle.human.estimate(frame, detections, tracks)
        animals = bundle.animal.estimate(frame, detections, tracks)

        assert isinstance(humans, (list, tuple))
        assert isinstance(animals, (list, tuple))
        assert len(humans) == 0
        assert len(animals) == 0

    def test_unknown_human_estimator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown human pose estimator"):
            build_pose_estimators({
                "pose": {
                    "enabled": True,
                    "human": {"estimator": "nonexistent_model"},
                    "animal": {"estimator": "null"},
                }
            })

    def test_unknown_animal_estimator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown animal pose estimator"):
            build_pose_estimators({
                "pose": {
                    "enabled": True,
                    "human": {"estimator": "null"},
                    "animal": {"estimator": "bogus_model"},
                }
            })

    def test_not_implemented_human_estimator_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="not implemented"):
            build_pose_estimators({
                "pose": {
                    "enabled": True,
                    "human": {"estimator": "rtmpose"},
                    "animal": {"estimator": "null"},
                }
            })

    def test_ap10k_hrnet_estimator_builds(self) -> None:
        from src.pose.animal import AP10KHRNetAnimalPoseEstimator as AnimalImpl

        bundle = build_pose_estimators({
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {
                    "estimator": "ap10k_hrnet",
                    "model_path": "dummy.onnx",
                },
            }
        })
        assert isinstance(bundle.animal, AnimalImpl)

    def test_bundle_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        bundle = build_pose_estimators({})
        with pytest.raises(FrozenInstanceError):
            bundle.human = None  # type: ignore


class TestPoseEstimatorBundle:
    """Tests for the PoseEstimatorBundle dataclass."""

    def test_construct_with_explicit_estimators(self) -> None:
        human = NullHumanPoseEstimator()
        animal = NullAnimalPoseEstimator()
        bundle = PoseEstimatorBundle(human=human, animal=animal)
        assert bundle.human is human
        assert bundle.animal is animal

    def test_types_are_abstract_base_classes(self) -> None:
        bundle = build_pose_estimators({})
        assert isinstance(bundle.human, BaseHumanPoseEstimator)
        assert isinstance(bundle.animal, BaseAnimalPoseEstimator)
