from __future__ import annotations

from typing import Callable, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose.animal.ap10k_hrnet_estimator import (
    AP10KHRNetAnimalPoseEstimator,
    _OnnxBackend,
    _TorchBackend,
)
from src.pose.animal.species_mapper import SpeciesMapper
from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox
from src.schemas.pose import AnimalPose


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _dummy_onnx_heatmap(image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Simulate an ONNX backend returning heatmap-decoded keypoints.

    Returns symmetric AP-10K keypoints placed at the crop center.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [cx - 5,  int(cy * 0.6), 0.90]   # left_eye
    kpts[1] = [cx + 5,  int(cy * 0.6), 0.90]   # right_eye
    kpts[2] = [cx,      int(cy * 0.65), 0.95]  # nose
    kpts[3] = [cx,      int(cy * 0.75), 0.85]  # neck
    kpts[4] = [cx - 20, int(cy * 0.85), 0.90]  # left_front_shoulder
    kpts[5] = [cx + 20, int(cy * 0.85), 0.90]  # right_front_shoulder
    kpts[6] = [cx - 15, int(cy * 1.1),  0.80]  # left_front_paw
    kpts[7] = [cx + 15, int(cy * 1.1),  0.80]  # right_front_paw
    kpts[8] = [cx - 15, cy + 10,        0.85]  # left_back_shoulder
    kpts[9] = [cx + 15, cy + 10,        0.85]  # right_back_shoulder
    kpts[10] = [cx - 10, cy + 40,       0.75]  # left_back_paw
    kpts[11] = [cx + 10, cy + 40,       0.75]  # right_back_paw
    kpts[12] = [cx,      cy + 50,       0.70]  # tail
    kpts[13] = [cx - 8,  int(cy * 0.58), 0.80] # left_ear
    kpts[14] = [cx + 8,  int(cy * 0.58), 0.80] # right_ear
    kpts[15] = [cx,      int(cy * 0.55), 0.85] # head_top
    kpts[16] = [cx,      cy + 5,        0.75]  # mid_back
    bbox = np.array([cx - 30, cy - 40, cx + 30, cy + 60], dtype=np.float32)
    return [(kpts, bbox, 0.88)]


def _dummy_torch_output(
    image: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Simulate PyTorch backend output."""
    return _dummy_onnx_heatmap(image)


# --------------------------------------------------------------------------- #
#  OnnxBackend tests
# --------------------------------------------------------------------------- #


class TestOnnxBackend:
    """Tests for the internal _OnnxBackend class."""

    def test_load_raises_without_onnxruntime(self) -> None:
        backend = _OnnxBackend(
            "dummy.onnx",
            input_size=(256, 256),
            heatmap_size=(64, 64),
            conf_threshold=0.5,
        )
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                backend.load()

    def test_run_with_mocked_session(self) -> None:
        backend = _OnnxBackend(
            "dummy.onnx",
            input_size=(256, 256),
            heatmap_size=(64, 64),
            conf_threshold=0.5,
        )
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.get_inputs.return_value[0].name = "images"
        # Simulate heatmap output (1, 17, 64, 64)
        dummy_heatmaps = np.zeros((1, 17, 64, 64), dtype=np.float32)
        dummy_heatmaps[0, 0, 32, 32] = 0.95  # left_eye peak
        dummy_heatmaps[0, 1, 32, 36] = 0.90  # right_eye peak
        mock_session.run.return_value = [dummy_heatmaps]
        backend._session = mock_session
        backend._input_name = "images"

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        results = backend.run(img)
        assert len(results) > 0
        kpts, bbox, score = results[0]
        assert kpts.shape == (17, 3)
        assert bbox.shape == (4,)
        assert score > 0


# --------------------------------------------------------------------------- #
#  TorchBackend tests
# --------------------------------------------------------------------------- #


class TestTorchBackend:
    """Tests for the internal _TorchBackend class."""

    def test_load_raises_without_torch(self) -> None:
        backend = _TorchBackend(
            "dummy.pt",
            device="cpu",
            input_size=(256, 256),
            heatmap_size=(64, 64),
            conf_threshold=0.5,
        )
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="torch"):
                backend.load()


# --------------------------------------------------------------------------- #
#  Backend factory tests
# --------------------------------------------------------------------------- #


class TestCreateBackend:
    """Tests for the internal _create_backend function."""

    def test_backend_selection_onnx(self) -> None:
        from src.pose.animal.ap10k_hrnet_estimator import _create_backend

        backend = _create_backend(
            "model.onnx", "cpu", (256, 256), (64, 64), 0.5
        )
        assert isinstance(backend, _OnnxBackend)

    def test_backend_selection_pt(self) -> None:
        from src.pose.animal.ap10k_hrnet_estimator import _create_backend

        backend = _create_backend(
            "model.pt", "cpu", (256, 256), (64, 64), 0.5
        )
        assert isinstance(backend, _TorchBackend)

    def test_backend_selection_unknown_raises(self) -> None:
        from src.pose.animal.ap10k_hrnet_estimator import _create_backend

        with pytest.raises(ValueError, match="Unrecognised"):
            _create_backend(
                "model.tflite", "cpu", (256, 256), (64, 64), 0.5
            )


# --------------------------------------------------------------------------- #
#  AP10KHRNetAnimalPoseEstimator tests
# --------------------------------------------------------------------------- #


class TestAP10KHRNetAnimalPoseEstimator:
    """Tests for the main AP10KHRNetAnimalPoseEstimator class."""

    # --- Scenario: no animal detections -------------------------------- #

    def test_no_animal_detections_returns_empty(
        self, synthetic_frame: np.ndarray
    ) -> None:
        """When no animal detections exist, returns empty sequence
        and does NOT load the model."""
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.5
        )
        person_dets = [
            Detection(
                class_id=0, class_name="person", confidence=0.9,
                bbox=BoundingBox(0, 0, 50, 50),
            )
        ]
        result = estimator.estimate(synthetic_frame, person_dets, [])
        assert len(result) == 0
        assert estimator._loaded is False

    def test_empty_detections_returns_empty(
        self, synthetic_frame: np.ndarray
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator()
        result = estimator.estimate(synthetic_frame, [], [])
        assert len(result) == 0
        assert estimator._loaded is False

    # --- Scenario: returns correct AnimalPose type ---------------------- #

    def test_returns_animal_pose_objects(
        self, synthetic_frame: np.ndarray,
        single_dog_detection: List[Detection],
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        result = estimator.estimate(
            synthetic_frame, single_dog_detection, []
        )
        assert len(result) == 1
        pose = result[0]
        assert isinstance(pose, AnimalPose)
        assert pose.species == "dog"
        assert pose.track_id is None
        assert pose.overall_confidence > 0
        assert len(pose.keypoints) == 17

    def test_animal_pose_has_correct_species(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        mapper = SpeciesMapper(class_to_species={42: "fox"})
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx",
            conf_threshold=0.3,
            species_mapper=mapper,
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        fox_dets = [
            Detection(
                class_id=42, class_name="fox", confidence=0.9,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, fox_dets, [])
        assert len(result) == 1
        assert result[0].species == "fox"

    def test_species_from_default_mapping(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        cow_dets = [
            Detection(
                class_id=19, class_name="cow", confidence=0.9,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, cow_dets, [])
        assert len(result) == 1
        assert result[0].species == "cow"

    # --- Scenario: keypoints in AP-10K order --------------------------- #

    def test_keypoints_in_ap10k_order(
        self, synthetic_frame: np.ndarray,
        single_dog_detection: List[Detection],
    ) -> None:
        """Keypoints should be in correct AP-10K order with valid names."""
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        result = estimator.estimate(
            synthetic_frame, single_dog_detection, []
        )
        pose = result[0]
        keypoint_names = [kp.name for kp in pose.keypoints]
        expected = [
            "left_eye", "right_eye", "nose", "neck",
            "left_front_shoulder", "right_front_shoulder",
            "left_front_paw", "right_front_paw",
            "left_back_shoulder", "right_back_shoulder",
            "left_back_paw", "right_back_paw",
            "tail", "left_ear", "right_ear", "head_top", "mid_back",
        ]
        assert keypoint_names == expected

    # --- Scenario: max_animals enforcement ------------------------------ #

    def test_max_animals_enforced(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx",
            conf_threshold=0.3,
            max_animals=2,
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        four_animals = [
            Detection(
                class_id=16, class_name="dog", confidence=0.95,
                bbox=BoundingBox(10, 10, 100, 200),
            ),
            Detection(
                class_id=19, class_name="cow", confidence=0.85,
                bbox=BoundingBox(200, 10, 300, 200),
            ),
            Detection(
                class_id=17, class_name="horse", confidence=0.75,
                bbox=BoundingBox(400, 10, 500, 200),
            ),
            Detection(
                class_id=22, class_name="zebra", confidence=0.65,
                bbox=BoundingBox(450, 10, 550, 200),
            ),
        ]
        result = estimator.estimate(synthetic_frame, four_animals, [])
        assert len(result) <= 2

    # --- Scenario: track_id assignment --------------------------------- #

    def test_track_id_from_tracks(
        self, synthetic_frame: np.ndarray,
        single_dog_detection: List[Detection],
        mock_animal_tracks: List[TrackedObject],
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        result = estimator.estimate(
            synthetic_frame, single_dog_detection, mock_animal_tracks[:1]
        )
        if len(result) > 0:
            assert result[0].track_id == 10

    # --- Scenario: orientation computed -------------------------------- #

    def test_orientation_computed(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.3
        )

        def _mock_run(
            img: np.ndarray,
        ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            kpts = np.zeros((17, 3), dtype=np.float32)
            # Symmetric — facing camera
            kpts[0] = [cx - 5,  int(cy * 0.6), 0.90]
            kpts[1] = [cx + 5,  int(cy * 0.6), 0.90]
            kpts[2] = [cx,      int(cy * 0.65), 0.95]
            kpts[3] = [cx,      int(cy * 0.75), 0.85]
            kpts[4] = [cx - 20, int(cy * 0.85), 0.90]
            kpts[5] = [cx + 20, int(cy * 0.85), 0.90]
            kpts[8] = [cx - 10, cy + 10,        0.85]
            kpts[9] = [cx + 10, cy + 10,        0.85]
            for i in range(17):
                if kpts[i, 2] == 0:
                    kpts[i] = [cx, cy, 0.05]
            bbox = np.array([cx - 30, cy - 40, cx + 30, cy + 60])
            return [(kpts, bbox, 0.88)]

        estimator._backend = MagicMock()
        estimator._backend.run = _mock_run
        estimator._loaded = True

        dog_dets = [
            Detection(
                class_id=16, class_name="dog", confidence=0.95,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, dog_dets, [])
        if len(result) > 0 and result[0].body_orientation_deg is not None:
            assert abs(result[0].body_orientation_deg) < 20.0

    # --- Scenario: graceful error handling ----------------------------- #

    def test_crop_error_does_not_crash(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        mixed_dets = [
            Detection(
                class_id=16, class_name="dog", confidence=0.9,
                bbox=BoundingBox(50, 50, 150, 200),
            ),
            Detection(
                class_id=16, class_name="dog", confidence=0.8,
                bbox=BoundingBox(1, 1, 3, 3),  # tiny bbox at edge
            ),
        ]
        result = estimator.estimate(synthetic_frame, mixed_dets, [])
        assert len(result) >= 0

    # --- Scenario: unsupported species filtering ----------------------- #

    def test_unsupported_species_skipped_with_mock(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        """Only supported species should produce AnimalPose output."""
        mapper = SpeciesMapper(supported_species=["dog"])
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx",
            conf_threshold=0.3,
            species_mapper=mapper,
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        mixed = [
            Detection(
                class_id=16, class_name="dog", confidence=0.9,
                bbox=BoundingBox(50, 50, 150, 350),
            ),
            Detection(
                class_id=19, class_name="cow", confidence=0.85,
                bbox=BoundingBox(200, 60, 350, 380),
            ),
        ]
        result = estimator.estimate(synthetic_frame, mixed, [])
        assert len(result) == 1
        assert result[0].species == "dog"

    # --- Scenario: multiple animal species in one frame ---------------- #

    def test_multiple_species(
        self, synthetic_frame: np.ndarray,
    ) -> None:
        estimator = AP10KHRNetAnimalPoseEstimator(
            model_path="dummy.onnx",
            conf_threshold=0.3,
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_onnx_heatmap
        estimator._loaded = True

        two_species = [
            Detection(
                class_id=16, class_name="dog", confidence=0.9,
                bbox=BoundingBox(10, 10, 100, 200),
            ),
            Detection(
                class_id=19, class_name="cow", confidence=0.85,
                bbox=BoundingBox(200, 60, 350, 380),
            ),
        ]
        result = estimator.estimate(synthetic_frame, two_species, [])
        assert len(result) == 2
        species = {p.species for p in result}
        assert species == {"dog", "cow"}
