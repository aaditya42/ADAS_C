from __future__ import annotations

import numpy as np
import pytest
import torch

from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector


def _make_detector(**kwargs) -> CLRNetLaneDetector:
    det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
    det._input_w = kwargs.get("input_width", 800)
    det._input_h = kwargs.get("input_height", 320)
    det._device = torch.device("cpu")
    return det


class TestPreprocessing:
    def test_preprocess_returns_float32_on_cpu(self) -> None:
        det = _make_detector(input_width=800, input_height=320)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.dtype == torch.float32

    def test_preprocess_output_shape(self) -> None:
        det = _make_detector(input_width=800, input_height=320)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.shape == (1, 3, 320, 800)

    def test_preprocess_resizes_to_target(self) -> None:
        det = _make_detector(input_width=400, input_height=200)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.shape == (1, 3, 200, 400)

    def test_preprocess_values_in_0_1_range(self) -> None:
        det = _make_detector(input_width=800, input_height=320)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_handles_odd_dimensions(self) -> None:
        det = _make_detector(input_width=801, input_height=321)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.shape == (1, 3, 321, 801)

    def test_preprocess_on_gpu(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        det = _make_detector(input_width=800, input_height=320)
        det._device = torch.device("cuda:0")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = det._preprocess(frame)
        assert tensor.device.type == "cuda"
