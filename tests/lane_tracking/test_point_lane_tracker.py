from __future__ import annotations

import numpy as np

from src.lane_detection.interfaces import LaneDetectionResult
from src.lane_tracking.tracker import (
    PointLaneTracker,
    _extend_lane,
    _get_bottom_x,
    _get_top_x,
    _smooth_points,
)


def _make_result(
    left_pts: list[tuple[int, int]],
    right_pts: list[tuple[int, int]],
    conf: float = 0.9,
) -> LaneDetectionResult:
    return LaneDetectionResult(
        left_lane_points=tuple(left_pts),
        right_lane_points=tuple(right_pts),
        center_lane_points=(),
        lane_count=2 if left_pts and right_pts else 1 if left_pts or right_pts else 0,
        ego_lane_center_x=None,
        confidence=conf,
    )


def _straight_pts(x_start: int, y_range: tuple[int, int, int]) -> list[tuple[int, int]]:
    """Generate straight vertical lane points from *y_range* ``(start, stop, step)``."""
    return [(x_start, y) for y in range(*y_range)]


def _curved_pts(
    coeffs: tuple[float, float, float],
    y_vals: list[int],
) -> list[tuple[int, int]]:
    """Generate points along ``x = a*y^2 + b*y + c`` at given *y_vals*."""
    a, b, c = coeffs
    return [(int(round(a * y * y + b * y + c)), y) for y in y_vals]


# ======================================================================
# Internal helpers
# ======================================================================


class TestGetBottomX:
    def test_returns_bottom_x(self) -> None:
        pts = [(200, 300), (220, 500), (250, 700)]
        assert _get_bottom_x(pts) == 250.0

    def test_returns_none_empty(self) -> None:
        assert _get_bottom_x([]) is None

    def test_single_point(self) -> None:
        assert _get_bottom_x([(300, 400)]) == 300.0


class TestGetTopX:
    def test_returns_top_x(self) -> None:
        pts = [(200, 300), (220, 500), (250, 700)]
        assert _get_top_x(pts) == 200.0

    def test_returns_none_empty(self) -> None:
        assert _get_top_x([]) is None


class TestSmoothPoints:
    def test_first_frame_uses_raw(self) -> None:
        new = [(200, 300), (220, 400)]
        result = _smooth_points(new, [], alpha=0.25)
        assert result == [(200, 300), (220, 400)]

    def test_ema_applied_per_y(self) -> None:
        new = [(200, 300), (220, 400)]
        old = [(180, 300), (210, 400)]
        result = _smooth_points(new, old, alpha=0.5)
        # y=300: 0.5*200 + 0.5*180 = 190
        # y=400: 0.5*220 + 0.5*210 = 215
        assert result == [(190, 300), (215, 400)]

    def test_custom_alpha(self) -> None:
        new = [(200, 300)]
        old = [(100, 300)]
        result = _smooth_points(new, old, alpha=0.75)
        # 0.75*200 + 0.25*100 = 175
        assert result == [(175, 300)]

    def test_new_y_added(self) -> None:
        new = [(200, 300), (220, 400)]
        old = [(180, 300)]
        result = _smooth_points(new, old, alpha=0.5)
        # y=300: 0.5*200 + 0.5*180 = 190
        # y=400: new point → 220
        assert result == [(190, 300), (220, 400)]

    def test_old_y_dropped(self) -> None:
        new = [(200, 400)]
        old = [(180, 300), (210, 400)]
        result = _smooth_points(new, old, alpha=0.5)
        # y=300 not in new → dropped
        # y=400: 0.5*200 + 0.5*210 = 205
        assert result == [(205, 400)]


class TestExtendLane:
    def test_extends_upward(self) -> None:
        # 6 points from y=500 down to y=400
        pts = [(200, y) for y in range(500, 399, -20)]
        result = _extend_lane(pts, img_w=1280, extend_above=30, min_allowed_y=350)
        ys = [p[1] for p in result]
        assert min(ys) <= 370  # top_y=400, 400-30=370

    def test_clamps_to_min_allowed(self) -> None:
        pts = [(200, y) for y in range(500, 419, -20)]
        result = _extend_lane(pts, img_w=1280, extend_above=200, min_allowed_y=360)
        ys = [p[1] for p in result]
        assert min(ys) >= 360

    def test_does_not_extend_below_min_allowed(self) -> None:
        pts = [(200, y) for y in range(500, 399, -20)]
        result = _extend_lane(pts, img_w=1280, extend_above=5, min_allowed_y=350)
        ys = [p[1] for p in result]
        # min_y of pts = 400, 400-5 = 395, which is >= 350
        assert min(ys) >= 395

    def test_clips_x_to_image_width(self) -> None:
        # x increases with y; at top x should be bounded
        pts = [(1200, y) for y in range(500, 399, -20)]
        result = _extend_lane(pts, img_w=1280, extend_above=40, min_allowed_y=350)
        xs = [p[0] for p in result]
        assert all(0 <= x < 1280 for x in xs)

    def test_requires_min_points(self) -> None:
        pts = [(200, y) for y in range(500, 460, -20)]
        assert len(pts) < 4  # only 3 points
        result = _extend_lane(pts, img_w=1280, extend_above=40, n_slope_points=4)
        assert len(result) == len(pts)

    def test_linear_slope_preserves_direction(self) -> None:
        # Lane sloping left as it goes up (x decreases with y)
        pts = [(300, 500), (280, 470), (260, 440), (240, 410)]
        result = _extend_lane(pts, img_w=1280, extend_above=30, n_slope_points=4)
        assert len(result) == len(pts) + 1
        top_x = result[-1][0]
        # Should extend leftward, so top_x < 240
        assert top_x < 240


# ======================================================================
# PointLaneTracker
# ======================================================================


class TestPointLaneTracker:
    def _tracker(self, **overrides: dict) -> PointLaneTracker:
        cfg = {
            "smoothing_alpha": 0.25,
            "max_missing_frames": 5,
            "confidence_threshold": 0.5,
            "min_points": 3,  # minimal for tests
            "max_bottom_jump": 80.0,
            "max_geometry_delta": 600.0,
            "extend_above_pixels": 0,  # off by default in tests
            "min_allowed_y": 350,
        }
        cfg.update(overrides)
        return PointLaneTracker(cfg)

    # ---- Basic flow ----

    def test_process_none_returns_none(self) -> None:
        tracker = self._tracker()
        result = tracker.process(None, img_h=720, img_w=1280)
        assert result is None

    def test_process_empty_result(self) -> None:
        tracker = self._tracker()
        empty = LaneDetectionResult.empty()
        result = tracker.process(empty, img_h=720, img_w=1280)
        assert result is not None
        assert result.lane_count == 0
        assert result.confidence == 0.0
        assert result.left_lane_points == ()
        assert result.right_lane_points == ()

    def test_first_frame_both_lanes(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        out = tracker.process(_make_result(left, right), 720, 1280)
        assert out is not None
        assert out.lane_count == 2
        assert len(out.left_lane_points) >= 2
        assert len(out.right_lane_points) >= 2

    def test_first_frame_left_only(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        out = tracker.process(_make_result(left, []), 720, 1280)
        assert out is not None
        assert out.lane_count == 1
        assert len(out.left_lane_points) >= 2
        assert len(out.right_lane_points) == 0

    def test_first_frame_right_only(self) -> None:
        tracker = self._tracker()
        right = _straight_pts(800, (300, 701, 50))
        out = tracker.process(_make_result([], right), 720, 1280)
        assert out is not None
        assert out.lane_count == 1
        assert len(out.right_lane_points) >= 2
        assert len(out.left_lane_points) == 0

    # ---- EMA smoothing ----

    def test_ema_smoothing_stabilises(self) -> None:
        tracker = self._tracker(smoothing_alpha=0.5)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))

        # Frame 1: establish tracks
        tracker.process(_make_result(left, right), 720, 1280)

        # Frame 2: noisy detection
        left_noisy = [(201, y) for y in range(300, 701, 50)]
        right_noisy = [(798, y) for y in range(300, 701, 50)]
        out = tracker.process(_make_result(left_noisy, right_noisy), 720, 1280)
        assert out is not None and out.lane_count == 2
        # With alpha=0.5, smoothed should be between noisy and original
        for lx, _ in out.left_lane_points:
            assert 200 <= lx <= 201
        for rx, _ in out.right_lane_points:
            assert 798 <= rx <= 800

    # ---- Lane memory ----

    def test_lane_disappears_then_returns(self) -> None:
        tracker = self._tracker(max_missing_frames=3)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))

        # Frame 1: both detected
        r1 = tracker.process(_make_result(left, right), 720, 1280)
        assert r1 is not None and r1.lane_count == 2

        # Frames 2-3: no detection — lanes should persist via memory
        empty = LaneDetectionResult.empty()
        r2 = tracker.process(empty, 720, 1280)
        assert r2 is not None and r2.lane_count == 2
        r3 = tracker.process(empty, 720, 1280)
        assert r3 is not None and r3.lane_count == 2

    def test_lane_dies_after_max_missed(self) -> None:
        tracker = self._tracker(max_missing_frames=2)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))

        r1 = tracker.process(_make_result(left, right), 720, 1280)
        assert r1 is not None and r1.lane_count == 2

        empty = LaneDetectionResult.empty()
        # Frame 2: missed 1
        tracker.process(empty, 720, 1280)
        # Frame 3: missed 2
        tracker.process(empty, 720, 1280)
        # Frame 4: missed 3 — exceeds max 2
        r4 = tracker.process(empty, 720, 1280)
        assert r4 is not None and r4.lane_count == 0

    def test_lane_reappears_resets_missed(self) -> None:
        tracker = self._tracker(max_missing_frames=3)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))

        r1 = tracker.process(_make_result(left, right), 720, 1280)
        assert r1 is not None and r1.lane_count == 2

        # One missed frame
        tracker.process(LaneDetectionResult.empty(), 720, 1280)

        # Lane reappears
        r3 = tracker.process(_make_result(left, right), 720, 1280)
        assert r3 is not None and r3.lane_count == 2

    def test_left_only_memory_right_dies(self) -> None:
        tracker = self._tracker(max_missing_frames=2)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))

        r1 = tracker.process(_make_result(left, right), 720, 1280)
        assert r1 is not None and r1.lane_count == 2

        # Right disappears, left stays
        r2 = tracker.process(_make_result(left, []), 720, 1280)
        assert r2 is not None
        assert r2.lane_count == 2  # right in memory
        assert len(r2.left_lane_points) > 0
        assert len(r2.right_lane_points) > 0

        # Right stays gone past max_missed
        r3 = tracker.process(_make_result(left, []), 720, 1280)
        assert r3 is not None and r3.lane_count == 2
        r4 = tracker.process(_make_result(left, []), 720, 1280)
        assert r4 is not None and r4.lane_count == 1
        assert len(r4.right_lane_points) == 0

    # ---- Geometry validation ----

    def test_crossing_rejected(self) -> None:
        tracker = self._tracker()
        # Left lane is to the right of right lane → crossing
        left = _straight_pts(900, (300, 701, 50))
        right = _straight_pts(100, (300, 701, 50))
        out = tracker.process(_make_result(left, right), 720, 1280)
        assert out is not None
        assert out.lane_count == 0

    def test_wide_geometry_rejected(self) -> None:
        tracker = self._tracker(max_geometry_delta=600.0)
        # Lane with huge delta (diagonal across entire image)
        pts = [(50, 300), (950, 700)]
        out = tracker.process(_make_result(left_pts=pts, right_pts=[]), 720, 1280)
        assert out is not None
        assert out.lane_count == 0  # too few points too, but still

    def test_too_few_points_rejected(self) -> None:
        tracker = self._tracker(min_points=5)
        pts = [(200, y) for y in range(300, 501, 50)]  # 5 points, ok
        out = tracker.process(_make_result(left_pts=pts, right_pts=[]), 720, 1280)
        assert out is not None
        assert len(out.left_lane_points) > 0

        tracker2 = self._tracker(min_points=6)
        pts2 = [(200, y) for y in range(300, 501, 50)]  # 5 points, not ok
        out2 = tracker2.process(_make_result(left_pts=pts2, right_pts=[]), 720, 1280)
        assert out2 is not None
        assert len(out2.left_lane_points) == 0

    # ---- Confidence gate ----

    def test_low_confidence_rejected(self) -> None:
        tracker = self._tracker(confidence_threshold=0.6)
        pts = _straight_pts(200, (300, 701, 50))
        # confidence of 0.5 < 0.6 threshold
        out = tracker.process(_make_result(pts, pts, conf=0.5), 720, 1280)
        assert out is not None
        assert out.lane_count == 0

    def test_high_confidence_accepted(self) -> None:
        tracker = self._tracker(confidence_threshold=0.6)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        out = tracker.process(_make_result(left, right, conf=0.7), 720, 1280)
        assert out is not None
        assert out.lane_count == 2

    # ---- Extend lane ----

    def test_extension_adds_points(self) -> None:
        tracker = self._tracker(extend_above_pixels=30, min_allowed_y=350)
        pts = _straight_pts(200, (400, 701, 50))
        out = tracker.process(_make_result(pts, []), 720, 1280)
        assert out is not None
        assert len(out.left_lane_points) > len(pts)

    # ---- Centre lane ----

    def test_center_lane_generated(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        out = tracker.process(_make_result(left, right), 720, 1280)
        assert out is not None
        assert len(out.center_lane_points) > 0

    def test_no_center_lane_when_one_missing(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        out = tracker.process(_make_result(left, []), 720, 1280)
        assert out is not None
        assert len(out.center_lane_points) == 0

    # ---- Ego lane centre ----

    def test_ego_lane_center_x(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        out = tracker.process(_make_result(left, right), 720, 1280)
        assert out is not None
        assert out.ego_lane_center_x is not None
        assert out.ego_lane_center_x == 500.0  # (200+800)/2

    # ---- Reset ----

    def test_reset_clears_state(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        out = tracker.process(_make_result(left, []), 720, 1280)
        assert out is not None and out.lane_count == 1

        tracker.reset()
        assert len(tracker._left_points) == 0
        assert len(tracker._right_points) == 0
        assert tracker._left_missed == 0
        assert tracker._right_missed == 0
        assert tracker._frame_count == 0

    # ---- Multi-frame ----

    def test_multiple_frames_same_position(self) -> None:
        tracker = self._tracker(smoothing_alpha=0.5)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))

        for _ in range(5):
            out = tracker.process(_make_result(left, right), 720, 1280)
            assert out is not None and out.lane_count == 2

    def test_sudden_jump_uses_previous(self) -> None:
        """When detection jumps > max_bottom_jump, the current detection
        should be rejected and the previous smoothed points retained.
        """
        tracker = self._tracker(
            smoothing_alpha=0.5,
            max_bottom_jump=50.0,
            min_points=3,
        )
        # Stable position first
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        r1 = tracker.process(_make_result(left, right), 720, 1280)
        assert r1 is not None and r1.lane_count == 2

        # Sudden jump on the next frame
        left_jump = _straight_pts(400, (300, 701, 50))
        right_jump = _straight_pts(1000, (300, 701, 50))
        r2 = tracker.process(_make_result(left_jump, right_jump), 720, 1280)
        # The tracker should reject and use previous
        assert r2 is not None and r2.lane_count == 2
        lx_r2 = [p[0] for p in r2.left_lane_points]
        # Should NOT reflect the jumped position
        assert all(x < 300 for x in lx_r2)

    # ---- Configurability ----

    def test_custom_alpha(self) -> None:
        tracker = self._tracker(smoothing_alpha=0.1)
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        r1 = tracker.process(_make_result(left, right), 720, 1280)
        assert r1 is not None and r1.lane_count == 2

        # Very different detection with low alpha → mostly previous
        left2 = _straight_pts(300, (300, 701, 50))
        r2 = tracker.process(_make_result(left2, right), 720, 1280)
        assert r2 is not None and r2.lane_count == 2
        # With alpha=0.1, smoothed x should be close to original 200
        lx = [p[0] for p in r2.left_lane_points]
        assert all(200 <= x <= 210 for x in lx)

    def test_confidence_in_output(self) -> None:
        tracker = self._tracker(confidence_threshold=0.3)
        left = _straight_pts(200, (300, 701, 50))
        out = tracker.process(_make_result(left, [], conf=0.8), 720, 1280)
        assert out is not None
        assert out.confidence == 0.8

    def test_confidence_zero_for_empty(self) -> None:
        tracker = self._tracker()
        out = tracker.process(LaneDetectionResult.empty(), 720, 1280)
        assert out is not None
        assert out.confidence == 0.0

    # ---- API contract ----

    def test_preserves_lane_detection_result_interface(self) -> None:
        tracker = self._tracker()
        left = _straight_pts(200, (300, 701, 50))
        right = _straight_pts(800, (300, 701, 50))
        out = tracker.process(_make_result(left, right), 720, 1280)
        assert out is not None
        # Interface fields
        assert isinstance(out.left_lane_points, tuple)
        assert isinstance(out.right_lane_points, tuple)
        assert isinstance(out.center_lane_points, tuple)
        assert isinstance(out.lane_count, int)
        assert out.ego_lane_center_x is None or isinstance(out.ego_lane_center_x, float)
        assert isinstance(out.confidence, float)
        # Properties
        assert hasattr(out, "left_lane")
        assert hasattr(out, "right_lane")
