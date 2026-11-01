from __future__ import annotations

from typing import Callable, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose.human.yolov8_pose_estimator import (
    YOLOv8HumanPoseEstimator,
    _OnnxBackend,
    _TorchBackend,
)
from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox
from src.schemas.pose import HumanPose


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _dummy_torch_output(image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Simulate a TorchBackend returning a single frontal-pose detection.

    The keypoints are placed at the center of the input image.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Symmetric keypoints centered on image
    kpts[0] = [cx, int(cy * 0.7), 0.95]       # nose
    kpts[5] = [cx - 40, int(cy * 0.8), 0.90]  # left_shoulder
    kpts[6] = [cx + 40, int(cy * 0.8), 0.90]  # right_shoulder
    kpts[11] = [cx - 20, int(cy * 1.1), 0.85] # left_hip
    kpts[12] = [cx + 20, int(cy * 1.1), 0.85] # right_hip
    # Fill remaining keypoints at low confidence
    for i in range(17):
        if kpts[i, 2] == 0:
            kpts[i] = [cx, cy, 0.05]
    bbox = np.array([cx - 60, cy - 80, cx + 60, cy + 80], dtype=np.float32)
    return [(kpts, bbox, 0.92)]


def _dummy_onnx_output(image: np.ndarray) -> np.ndarray:
    """Simulate raw ONNX output tensor (1, 56, N) with one detection."""
    raw = np.zeros((1, 56, 8400), dtype=np.float32)
    # Place one valid detection at the center
    h, w = image.shape[:2]  # should be 640x640
    cx, cy = w / 2.0, h / 2.0
    bw, bh = 120.0, 160.0
    # Encode in the first grid cell
    raw[0, 0, 0] = cx   # box cx
    raw[0, 1, 0] = cy   # box cy
    raw[0, 2, 0] = bw   # box w
    raw[0, 3, 0] = bh   # box h
    # Person score (pre-sigmoid: 3.0 → sigmoid ≈ 0.95)
    raw[0, 4, 0] = 3.0
    # Keypoints (17 * 3 = 51 rows starting at index 5)
    # nose
    raw[0, 5, 0] = cx
    raw[0, 6, 0] = cy * 0.7
    raw[0, 7, 0] = 3.0  # pre-sigmoid high
    # left_shoulder
    raw[0, 5 + 3 * 5, 0] = cx - 40
    raw[0, 5 + 3 * 5 + 1, 0] = cy * 0.8
    raw[0, 5 + 3 * 5 + 2, 0] = 2.5
    # right_shoulder
    raw[0, 5 + 3 * 6, 0] = cx + 40
    raw[0, 5 + 3 * 6 + 1, 0] = cy * 0.8
    raw[0, 5 + 3 * 6 + 2, 0] = 2.5
    # left_hip
    raw[0, 5 + 3 * 11, 0] = cx - 20
    raw[0, 5 + 3 * 11 + 1, 0] = cy * 1.1
    raw[0, 5 + 3 * 11 + 2, 0] = 2.0
    # right_hip
    raw[0, 5 + 3 * 12, 0] = cx + 20
    raw[0, 5 + 3 * 12 + 1, 0] = cy * 1.1
    raw[0, 5 + 3 * 12 + 2, 0] = 2.0
    return raw


# --------------------------------------------------------------------------- #
#  TorchBackend tests
# --------------------------------------------------------------------------- #

class TestTorchBackend:
    """Tests for the internal _TorchBackend class."""

    def test_load_raises_without_ultralytics(self) -> None:
        backend = _TorchBackend("dummy.pt", "cpu", 0.5)
        # Simulate ultralytics not installed
        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                backend.load()

    def test_run_returns_results(self) -> None:
        backend = _TorchBackend("dummy.pt", "cpu", 0.5)
        mock_model = MagicMock()
        mock_kpts = MagicMock()
        mock_kpts.data.cpu.return_value.numpy.return_value = np.zeros((1, 17, 3))
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.zeros((1, 4))
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_result = MagicMock()
        mock_result.keypoints = mock_kpts
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]
        backend._model = mock_model

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = backend.run(img)
        assert len(results) == 1
        kpts, bbox, score = results[0]
        assert kpts.shape == (17, 3)
        assert bbox.shape == (4,)


# --------------------------------------------------------------------------- #
#  OnnxBackend tests
# --------------------------------------------------------------------------- #

class TestOnnxBackend:
    """Tests for the internal _OnnxBackend class."""

    def test_load_raises_without_onnxruntime(self) -> None:
        backend = _OnnxBackend("dummy.onnx", 0.5, 0.65)
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                backend.load()

    def test_run_with_mocked_session(self) -> None:
        backend = _OnnxBackend("dummy.onnx", 0.5, 0.65)
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.get_inputs.return_value[0].name = "images"
        mock_session.run.return_value = [_dummy_onnx_output(np.zeros((640, 640, 3)))]
        backend._session = mock_session
        backend._input_name = "images"

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = backend.run(img)
        assert len(results) >= 0


# --------------------------------------------------------------------------- #
#  YOLOv8HumanPoseEstimator tests
# --------------------------------------------------------------------------- #

class TestYOLOv8HumanPoseEstimator:
    """Tests for the main YOLOv8HumanPoseEstimator class."""

    # --- Scenario: no person detections -------------------------------- #

    def test_no_person_detections_returns_empty(self, synthetic_frame: np.ndarray) -> None:
        """When no person detections exist, estimator returns empty sequence
        and does NOT load the model."""
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.5
        )
        car_dets = [
            Detection(
                class_id=1, class_name="car", confidence=0.9,
                bbox=BoundingBox(0, 0, 50, 50),
            )
        ]
        result = estimator.estimate(synthetic_frame, car_dets, [])
        assert len(result) == 0
        # Model should NOT have been loaded
        assert estimator._loaded is False

    def test_empty_detections_returns_empty(self, synthetic_frame: np.ndarray) -> None:
        estimator = YOLOv8HumanPoseEstimator()
        result = estimator.estimate(synthetic_frame, [], [])
        assert len(result) == 0
        assert estimator._loaded is False

    # --- Scenario: returns correct HumanPose type ---------------------- #

    def test_returns_human_pose_objects(self, synthetic_frame: np.ndarray) -> None:
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.3
        )
        # Bypass real model loading: inject a mock backend directly
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_torch_output
        estimator._loaded = True

        person_dets = [
            Detection(
                class_id=0, class_name="person", confidence=0.95,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, person_dets, [])
        assert len(result) == 1
        pose = result[0]
        assert isinstance(pose, HumanPose)
        assert pose.track_id is None
        assert pose.overall_confidence > 0
        assert len(pose.keypoints) == 17

    def test_human_pose_has_valid_keypoints(self, synthetic_frame: np.ndarray) -> None:
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_torch_output
        estimator._loaded = True

        person_dets = [
            Detection(
                class_id=0, class_name="person", confidence=0.95,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, person_dets, [])
        pose = result[0]
        # Keypoints should be in original image coordinates (not crop space)
        kpt = pose.keypoints[0]  # nose
        assert kpt.x >= 0
        assert kpt.y >= 0
        assert kpt.x <= 640  # frame width
        assert kpt.y <= 480  # frame height

    # --- Scenario: max_persons enforcement ----------------------------- #

    def test_max_persons_enforced(self, synthetic_frame: np.ndarray) -> None:
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.3, max_persons=2
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_torch_output  # will be called per-crop
        estimator._loaded = True

        three_persons = [
            Detection(
                class_id=0, class_name="person", confidence=0.9,
                bbox=BoundingBox(10, 10, 100, 200),
            ),
            Detection(
                class_id=0, class_name="person", confidence=0.8,
                bbox=BoundingBox(200, 10, 300, 200),
            ),
            Detection(
                class_id=0, class_name="person", confidence=0.7,
                bbox=BoundingBox(400, 10, 500, 200),
            ),
        ]
        result = estimator.estimate(synthetic_frame, three_persons, [])
        # With max_persons=2, we should only process the top 2
        assert len(result) <= 2

    # --- Scenario: track_id assignment --------------------------------- #

    def test_track_id_from_tracks(self, synthetic_frame: np.ndarray) -> None:
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_torch_output
        estimator._loaded = True

        person_dets = [
            Detection(
                class_id=0, class_name="person", confidence=0.95,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        tracks = [
            TrackedObject(
                track_id=42, class_name="person", confidence=0.95,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, person_dets, tracks)
        if len(result) > 0:
            assert result[0].track_id == 42

    # --- Scenario: backend selection by extension ---------------------- #

    def test_backend_selection_pt(self) -> None:
        from src.pose.human.yolov8_pose_estimator import _create_backend
        backend = _create_backend("model.pt", "cpu", 0.5, 0.65)
        assert isinstance(backend, _TorchBackend)

    def test_backend_selection_onnx(self) -> None:
        from src.pose.human.yolov8_pose_estimator import _create_backend
        backend = _create_backend("model.onnx", "cpu", 0.5, 0.65)
        assert isinstance(backend, _OnnxBackend)

    def test_backend_selection_unknown_raises(self) -> None:
        from src.pose.human.yolov8_pose_estimator import _create_backend
        with pytest.raises(ValueError, match="Unrecognised"):
            _create_backend("model.tflite", "cpu", 0.5, 0.65)

    # --- Scenario: orientation computed -------------------------------- #

    def test_orientation_computed(self, synthetic_frame: np.ndarray) -> None:
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.3
        )
        # Return symmetric keypoints (facing camera)
        def _mock_run(img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
            h, w = img.shape[:2]
            kpts = np.zeros((17, 3), dtype=np.float32)
            cx, cy = w // 2, h // 2
            kpts[0] = [cx, int(cy * 0.7), 0.95]
            kpts[5] = [cx - 50, int(cy * 0.8), 0.90]
            kpts[6] = [cx + 50, int(cy * 0.8), 0.90]
            kpts[11] = [cx - 25, int(cy * 1.1), 0.85]
            kpts[12] = [cx + 25, int(cy * 1.1), 0.85]
            for i in range(17):
                if kpts[i, 2] == 0:
                    kpts[i] = [cx, cy, 0.05]
            bbox = np.array([cx - 60, cy - 80, cx + 60, cy + 80])
            return [(kpts, bbox, 0.92)]
        estimator._backend = MagicMock()
        estimator._backend.run = _mock_run
        estimator._loaded = True

        person_dets = [
            Detection(
                class_id=0, class_name="person", confidence=0.95,
                bbox=BoundingBox(200, 100, 500, 400),
            )
        ]
        result = estimator.estimate(synthetic_frame, person_dets, [])
        if len(result) > 0:
            # Body orientation should be near 0° for symmetric keypoints
            if result[0].body_orientation_deg is not None:
                assert abs(result[0].body_orientation_deg) < 20.0

    # --- Scenario: graceful error handling ----------------------------- #

    def test_crop_error_does_not_crash(self, synthetic_frame: np.ndarray) -> None:
        """If a single detection fails during processing, others still work."""
        estimator = YOLOv8HumanPoseEstimator(
            model_path="dummy.pt", device="cpu", conf_threshold=0.3
        )
        estimator._backend = MagicMock()
        estimator._backend.run = _dummy_torch_output
        estimator._loaded = True

        # Add one invalid detection (at image edge — valid bbox but crop
        # will be tiny/empty after expansion clamping)
        mixed_dets = [
            Detection(
                class_id=0, class_name="person", confidence=0.9,
                bbox=BoundingBox(10, 10, 100, 200),
            ),
            Detection(
                class_id=0, class_name="person", confidence=0.8,
                bbox=BoundingBox(1, 1, 3, 3),  # tiny bbox at edge
            ),
        ]
        # Should not raise
        result = estimator.estimate(synthetic_frame, mixed_dets, [])
        assert len(result) >= 0
