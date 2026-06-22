from __future__ import annotations

import pytest

from src.lane_detection.interfaces import LaneDetectionResult


class TestLaneDetectionResultSchema:
    def test_creates_with_all_fields(self) -> None:
        result = LaneDetectionResult(
            left_lane_points=((100, 400), (120, 300), (140, 200)),
            right_lane_points=((500, 400), (480, 300), (460, 200)),
            center_lane_points=((300, 400), (300, 300), (300, 200)),
            lane_count=2,
            ego_lane_center_x=300.0,
            confidence=0.9,
            timestamp=123.456,
        )
        assert len(result.left_lane_points) == 3
        assert len(result.right_lane_points) == 3
        assert len(result.center_lane_points) == 3
        assert result.lane_count == 2
        assert result.ego_lane_center_x == 300.0
        assert result.confidence == 0.9
        assert result.timestamp == pytest.approx(123.456)

    def test_empty_method_returns_safe_default(self) -> None:
        empty = LaneDetectionResult.empty()
        assert empty.left_lane_points == ()
        assert empty.right_lane_points == ()
        assert empty.center_lane_points == ()
        assert empty.lane_count == 0
        assert empty.ego_lane_center_x is None
        assert empty.confidence == 0.0
        assert empty.timestamp is None

    def test_is_frozen(self) -> None:
        result = LaneDetectionResult.empty()
        with pytest.raises(AttributeError):
            result.lane_count = 5  # type: ignore[misc]

    def test_has_slots(self) -> None:
        result = LaneDetectionResult.empty()
        with pytest.raises(AttributeError):
            result.new_attr = "test"  # type: ignore[attr-defined]

    def test_multiple_empty_calls_return_independent_instances(self) -> None:
        a = LaneDetectionResult.empty()
        b = LaneDetectionResult.empty()
        assert a is not b

    def test_timestamp_defaults_to_none(self) -> None:
        result = LaneDetectionResult(
            left_lane_points=(),
            right_lane_points=(),
            center_lane_points=(),
            lane_count=0,
            ego_lane_center_x=None,
            confidence=0.0,
        )
        assert result.timestamp is None

    def test_empty_lane_points_are_tuples(self) -> None:
        empty = LaneDetectionResult.empty()
        assert isinstance(empty.left_lane_points, tuple)
        assert isinstance(empty.right_lane_points, tuple)
        assert isinstance(empty.center_lane_points, tuple)

    def test_non_empty_lane_points_contain_integer_pairs(self) -> None:
        result = LaneDetectionResult(
            left_lane_points=((100, 400),),
            right_lane_points=((500, 400),),
            center_lane_points=((300, 400),),
            lane_count=1,
            ego_lane_center_x=300.0,
            confidence=0.8,
        )
        for x, y in result.left_lane_points:
            assert isinstance(x, int)
            assert isinstance(y, int)
