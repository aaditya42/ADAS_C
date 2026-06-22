from __future__ import annotations

import numpy as np
import pytest
import torch

from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector
from src.lane_detection.interfaces import LaneDetectionResult


@pytest.fixture
def detector() -> CLRNetLaneDetector:
    det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
    det._conf_threshold = 0.5
    det._input_w = 800
    det._input_h = 320
    det._lane_conf_threshold = 0.5
    det._max_lanes = 4
    det._cfg = type("cfg", (), {"test_parameters": type("tp", (), {"nms_thres": 50})()})()
    return det


class TestPostprocessing:
    def test_empty_result_when_no_predictions(self, detector) -> None:
        result = detector._postprocess(torch.empty((0,)), 640, 480)
        assert isinstance(result, LaneDetectionResult)
        assert result.lane_count == 0

    def test_empty_result_when_no_valid_lanes(self, detector) -> None:
        # All confidences very low
        preds = torch.full((1, 192, 78), -100.0)
        result = detector._postprocess(preds, 640, 480)
        assert result.lane_count == 0

    def test_detects_valid_lanes(self, detector) -> None:
        # 2 proposals with high confidence
        preds = torch.zeros((1, 192, 78))
        # Lane 0: left, high confidence
        preds[0, 0, 0] = -10.0  # neg logit
        preds[0, 0, 1] = 10.0   # pos logit
        preds[0, 0, 2] = 0.0    # start_y (top)
        preds[0, 0, 3] = 0.25   # start_x (200/800)
        preds[0, 0, 4] = 0.25   # theta
        preds[0, 0, 5] = 1.0    # length (full)
        preds[0, 0, 6:] = 0.25  # x offsets at 0.25
        # Lane 1: right, high confidence
        preds[0, 1, 0] = -10.0
        preds[0, 1, 1] = 10.0
        preds[0, 1, 2] = 0.0
        preds[0, 1, 3] = 0.75   # start_x (600/800)
        preds[0, 1, 4] = 0.25   # theta
        preds[0, 1, 5] = 1.0
        preds[0, 1, 6:] = 0.75  # x offsets at 0.75

        result = detector._postprocess(preds, 800, 320)
        assert result.lane_count == 2
        assert result.left_lane_points
        assert result.right_lane_points
        assert result.ego_lane_center_x is not None

    def test_lanes_sorted_left_to_right(self, detector) -> None:
        preds = torch.zeros((1, 192, 78))
        # Lane 0: right
        preds[0, 0, 1] = 10.0
        preds[0, 0, 2] = 0.0
        preds[0, 0, 3] = 0.75
        preds[0, 0, 4] = 0.25
        preds[0, 0, 5] = 1.0
        preds[0, 0, 6:] = 0.75
        # Lane 1: left
        preds[0, 1, 1] = 10.0
        preds[0, 1, 2] = 0.0
        preds[0, 1, 3] = 0.25
        preds[0, 1, 4] = 0.25
        preds[0, 1, 5] = 1.0
        preds[0, 1, 6:] = 0.25

        result = detector._postprocess(preds, 800, 320)
        assert result.left_lane_points
        assert result.right_lane_points
        left_x = result.left_lane_points[-1][0]
        right_x = result.right_lane_points[-1][0]
        assert left_x < right_x

    def test_low_confidence_filtered_out(self, detector) -> None:
        preds = torch.zeros((1, 192, 78))
        preds[0, 0, 1] = -100.0  # very low positive logit
        preds[0, 0, 6:] = 0.5
        result = detector._postprocess(preds, 800, 320)
        assert result.lane_count == 0

    def test_center_lane_is_average(self, detector) -> None:
        preds = torch.zeros((1, 192, 78))
        # Left lane
        preds[0, 0, 1] = 10.0
        preds[0, 0, 2] = 0.0
        preds[0, 0, 3] = 0.25
        preds[0, 0, 4] = 0.25
        preds[0, 0, 5] = 1.0
        preds[0, 0, 6:] = 0.25
        # Right lane
        preds[0, 1, 1] = 10.0
        preds[0, 1, 2] = 0.0
        preds[0, 1, 3] = 0.75
        preds[0, 1, 4] = 0.25
        preds[0, 1, 5] = 1.0
        preds[0, 1, 6:] = 0.75

        result = detector._postprocess(preds, 800, 320)
        assert result.center_lane_points
        lx = result.left_lane_points[-1][0]
        rx = result.right_lane_points[-1][0]
        cx = result.center_lane_points[-1][0]
        expected_cx = (lx + rx) // 2
        assert abs(cx - expected_cx) <= 2

    def test_only_one_lane_no_ego_center(self, detector) -> None:
        preds = torch.zeros((1, 192, 78))
        preds[0, 0, 1] = 10.0
        preds[0, 0, 2] = 0.0
        preds[0, 0, 3] = 0.5
        preds[0, 0, 4] = 0.25
        preds[0, 0, 5] = 1.0
        preds[0, 0, 6:] = 0.5

        result = detector._postprocess(preds, 800, 320)
        assert result.lane_count == 1
        assert result.ego_lane_center_x is None

    def test_max_lanes_caps_output(self, detector) -> None:
        detector._max_lanes = 2
        preds = torch.zeros((1, 192, 78))
        for i in range(5):
            x_val = 0.1 + i * 0.2
            preds[0, i, 1] = 10.0
            preds[0, i, 2] = 0.0
            preds[0, i, 3] = x_val
            preds[0, i, 4] = 0.25
            preds[0, i, 5] = 1.0
            preds[0, i, 6:] = x_val

        result = detector._postprocess(preds, 800, 320)
        assert result.lane_count <= 2

    def test_points_within_image_bounds(self, detector) -> None:
        preds = torch.zeros((1, 192, 78))
        preds[0, 0, 1] = 10.0
        preds[0, 0, 2] = 0.0
        preds[0, 0, 3] = 0.5
        preds[0, 0, 4] = 0.25
        preds[0, 0, 5] = 1.0
        preds[0, 0, 6:] = 0.5

        result = detector._postprocess(preds, 640, 480)
        for pts in (result.left_lane_points, result.right_lane_points,
                    result.center_lane_points):
            for x, y in pts:
                assert 0 <= x < 640
                assert 0 <= y < 480
