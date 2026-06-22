from __future__ import annotations

import numpy as np
import pytest

from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector
from src.lane_detection.interfaces import LaneDetectionResult


class TestModelLoading:
    def test_raises_on_bad_config_path(self) -> None:
        det = CLRNetLaneDetector(config_path="nonexistent_config.py")
        with pytest.raises(FileNotFoundError, match="CLRNet config not found"):
            det._load_model()

    def test_raises_on_bad_checkpoint_path(self) -> None:
        det = CLRNetLaneDetector(
            model_path="nonexistent.pth",
            config_path=__file__,  # any existing file as fake config
        )
        with pytest.raises(FileNotFoundError, match="CLRNet checkpoint not found"):
            det._load_model()

    def test_detect_returns_empty_on_bad_path(self) -> None:
        det = CLRNetLaneDetector(model_path="nonexistent.pth")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert isinstance(result, LaneDetectionResult)
        assert result.lane_count == 0
        assert result.confidence == 0.0


class TestFailureRecovery:
    def test_preprocess_output_shape(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._input_w = 800
        det._input_h = 320
        det._device = "cpu"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.shape == (1, 3, 320, 800)
        assert tensor.device.type == "cpu"

    def test_preprocess_handles_resize(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._input_w = 400
        det._input_h = 200
        det._device = "cpu"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.shape == (1, 3, 200, 400)

    def test_postprocess_garbage_returns_empty(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._conf_threshold = 0.5
        det._max_lanes = 4
        det._input_w = 800
        det._input_h = 320
        # Empty tensor
        result = det._postprocess(torch_empty(), 640, 480)
        assert isinstance(result, LaneDetectionResult)
        assert result.lane_count == 0

    def test_detect_without_model_returns_empty(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._model_path = "nonexistent.pth"
        det._model = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert isinstance(result, LaneDetectionResult)
        assert result.lane_count == 0

    def test_temporal_smoothing_empty_buffer(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = True
        from collections import deque
        det._recent_results = deque(maxlen=5)
        r = LaneDetectionResult.empty()
        result = det._apply_smoothing(r)
        assert result.confidence == 0.0
        assert result.lane_count == 0


def torch_empty() -> ...:
    import torch
    return torch.empty((0,))


class TestSchemaValidation:
    def test_lane_detection_result_points_are_valid(self) -> None:
        result = LaneDetectionResult(
            left_lane_points=((0, 0), (10, 10)),
            right_lane_points=((100, 0), (90, 10)),
            center_lane_points=((50, 0), (50, 10)),
            lane_count=2,
            ego_lane_center_x=50.0,
            confidence=0.8,
        )
        for pts in (result.left_lane_points, result.right_lane_points,
                    result.center_lane_points):
            for x, y in pts:
                assert isinstance(x, int)
                assert isinstance(y, int)
                assert x >= 0
                assert y >= 0

    def test_empty_result_consistent(self) -> None:
        e1 = LaneDetectionResult.empty()
        e2 = LaneDetectionResult.empty()
        assert e1 == e2

    def test_non_empty_result_not_equal_empty(self) -> None:
        r = LaneDetectionResult(
            left_lane_points=((10, 20),),
            right_lane_points=((100, 20),),
            center_lane_points=((55, 20),),
            lane_count=2,
            ego_lane_center_x=55.0,
            confidence=0.9,
        )
        assert r != LaneDetectionResult.empty()
