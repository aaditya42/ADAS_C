from __future__ import annotations

import numpy as np
import pytest

from src.lane_detection.interfaces import LaneDetectionResult
from src.lane_detection.lane_detector_factory import LaneDetectorFactory
from src.lane_detection.ufld_lane_detector import UFLDLaneDetector


class TestUFLDFactory:
    def test_factory_returns_ufld(self) -> None:
        cfg = {"lane": {"enabled": True, "model_type": "ufld"}}
        det = LaneDetectorFactory.create(cfg)
        assert isinstance(det, UFLDLaneDetector)

    def test_factory_returns_none_when_disabled(self) -> None:
        cfg = {"lane": {"enabled": False}}
        det = LaneDetectorFactory.create(cfg)
        assert det is None

    def test_factory_raises_on_unknown(self) -> None:
        cfg = {"lane": {"enabled": True, "model_type": "nonexistent"}}
        with pytest.raises(ValueError, match="Unknown lane detector"):
            LaneDetectorFactory.create(cfg)


class TestUFLDLaneDetector:
    def test_importable(self) -> None:
        assert UFLDLaneDetector is not None

    def test_raises_on_missing_model(self) -> None:
        det = UFLDLaneDetector(model_path="nonexistent.onnx")
        dummy = np.zeros((288, 800, 3), dtype=np.uint8)
        with pytest.raises(FileNotFoundError):
            det.detect(dummy)

    def test_softmax_outputs_probabilities(self) -> None:
        logits = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
        probs = UFLDLaneDetector._softmax(logits, axis=-1)
        assert probs.shape == logits.shape
        for row in probs.reshape(-1, 3):
            assert abs(row.sum() - 1.0) < 1e-5

    def test_returns_empty_when_all_low_conf(self) -> None:
        logits = np.zeros((1, 4, 56, 200), dtype=np.float32)
        logits[0, 0, 0, 0] = -10.0
        det = UFLDLaneDetector.__new__(UFLDLaneDetector)
        det._conf_threshold = 0.9
        det._num_points = 56
        det._num_cols = 200
        det._num_lanes = 4

        result = det._postprocess(logits, 800, 288)
        assert isinstance(result, LaneDetectionResult)
        assert result.lane_count == 0
        assert result.confidence == 0.0
        assert not result.left_lane_points
        assert not result.right_lane_points

    def test_lane_assigns_sides_by_position(self) -> None:
        logits = np.zeros((1, 4, 56, 200), dtype=np.float32)
        logits[0, 0, :, 30] = 10.0
        logits[0, 1, :, 170] = 10.0

        det = UFLDLaneDetector.__new__(UFLDLaneDetector)
        det._conf_threshold = 0.0
        det._num_points = 56
        det._num_cols = 200
        det._num_lanes = 4

        result = det._postprocess(logits, 800, 288)
        assert result.left_lane_points
        assert result.right_lane_points

        left_x = result.left_lane_points[0][0]
        right_x = result.right_lane_points[0][0]
        assert left_x < right_x


class TestConfig:
    def test_ufld_config_formats(self) -> None:
        cfg = {"lane": {"enabled": True, "model_type": "ufld"}}
        det = LaneDetectorFactory.create(cfg)
        assert isinstance(det, UFLDLaneDetector)

    def test_custom_config_values(self) -> None:
        cfg = {
            "lane": {
                "enabled": True,
                "model_type": "ufld",
                "model_path": "weights/custom.onnx",
                "input_width": 640,
                "input_height": 360,
                "conf_threshold": 0.3,
            },
        }
        det = LaneDetectorFactory.create(cfg)
        assert isinstance(det, UFLDLaneDetector)
        assert det._model_path == "weights/custom.onnx"
        assert det._input_w == 640
        assert det._input_h == 360
        assert det._conf_threshold == 0.3
