from __future__ import annotations

from collections import deque

import pytest

from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector
from src.lane_detection.interfaces import LaneDetectionResult


def _make_result(
    left_x: int = 100,
    right_x: int = 500,
    lane_count: int = 2,
    confidence: float = 0.9,
) -> LaneDetectionResult:
    img_h = 320
    return LaneDetectionResult(
        left_lane_points=tuple((left_x, y) for y in range(0, img_h, 10)),
        right_lane_points=tuple((right_x, y) for y in range(0, img_h, 10)),
        center_lane_points=tuple(((left_x + right_x) // 2, y) for y in range(0, img_h, 10)),
        lane_count=lane_count,
        ego_lane_center_x=(left_x + right_x) / 2.0,
        confidence=confidence,
    )


def _make_empty_result() -> LaneDetectionResult:
    return LaneDetectionResult.empty()


class TestTemporalSmoothing:
    def test_returns_original_when_disabled(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = False
        det._recent_results = deque(maxlen=5)
        r = _make_result()
        smoothed = det._apply_smoothing(r)
        assert smoothed is r

    def test_returns_original_with_single_frame(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = True
        det._recent_results = deque(maxlen=5)
        r = _make_result()
        smoothed = det._apply_smoothing(r)
        assert smoothed.left_lane_points == r.left_lane_points

    def test_smoothing_averages_two_frames(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = True
        det._recent_results = deque(maxlen=5)

        r1 = _make_result(left_x=100, right_x=500)
        r2 = _make_result(left_x=110, right_x=510)

        det._apply_smoothing(r1)
        smoothed = det._apply_smoothing(r2)

        # After smoothing with 2 frames, points should be averaged
        assert smoothed.left_lane_points
        assert smoothed.right_lane_points
        # First point is at (100+110)/2 = 105 for left
        smoothed_left_x = smoothed.left_lane_points[0][0]
        assert smoothed_left_x == 105

    def test_empty_frames_do_not_corrupt(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = True
        det._recent_results = deque(maxlen=5)

        det._apply_smoothing(_make_empty_result())
        r = _make_result(left_x=100, right_x=500)
        smoothed = det._apply_smoothing(r)
        # Should still work after empty frames
        assert smoothed.left_lane_points
        assert smoothed.left_lane_points[0][0] == 100

    def test_smoothing_over_multiple_frames(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = True
        det._recent_results = deque(maxlen=5)

        for left_x in [100, 104, 108, 112, 116]:
            det._apply_smoothing(_make_result(left_x=left_x, right_x=500))

        r = _make_result(left_x=120, right_x=500)
        smoothed = det._apply_smoothing(r)

        # deque stores last 5: [104, 108, 112, 116, 120] → avg = 112
        assert smoothed.left_lane_points[0][0] == 112

    def test_window_limited_correctly(self) -> None:
        det = CLRNetLaneDetector.__new__(CLRNetLaneDetector)
        det._temporal_smoothing = True
        det._smoothing_window = 3
        det._recent_results = deque(maxlen=3)

        for left_x in [100, 200, 300, 400, 500]:
            det._apply_smoothing(_make_result(left_x=left_x, right_x=600))

        r = _make_result(left_x=600, right_x=700)
        smoothed = det._apply_smoothing(r)

        # Only last 3 (400, 500, 600) → average = 500
        assert smoothed.left_lane_points[0][0] == 500
