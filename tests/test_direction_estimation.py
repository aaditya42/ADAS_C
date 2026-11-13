"""Unit tests for camera-relative DirectionEstimator.

Covers:
- Front / Front-Left / Front-Right directional assignment with lane corridor.
- Optical periphery Left / Right detection in lower-lateral image regions.
- Fallback to image center geometric corridor when lane info is unavailable.
- Temporal smoothing and confirmation hysteresis across boundary jitter.
- Rejection and non-existence of rear / blind-spot directional warnings.
- Graceful handling of invalid or out-of-bounds bounding boxes.
- Stale track eviction and reset.
"""

from __future__ import annotations

import pytest

from src.schemas.bbox import BoundingBox
from src.threat.direction import DirectionEstimator
from src.threat.schemas import CameraDirection, LaneRelationship


class DummyLaneBoundary:
    """Mock lane boundary providing polynomial evaluation."""

    def __init__(self, x_offset: float):
        self.x_offset = x_offset

    def x_at_y(self, y: float) -> float:
        return self.x_offset


class DummyLaneDetectionResult:
    """Mock LaneDetectionResult with left and right lane boundaries."""

    def __init__(self, left_x: float, right_x: float):
        self.left_boundary = DummyLaneBoundary(left_x)
        self.right_boundary = DummyLaneBoundary(right_x)


class TestDirectionEstimator:
    """Comprehensive test suite for DirectionEstimator."""

    def test_front_direction_with_lane(self):
        """Object centered between lane lines is classified as FRONT."""
        estimator = DirectionEstimator()
        lane_info = DummyLaneDetectionResult(left_x=450.0, right_x=830.0)
        # Ego center = 640. Target box from x=600 to 680 (cx=640)
        bbox = BoundingBox(x1=600.0, y1=400.0, x2=680.0, y2=520.0)

        # Frame 1: initial observation
        d1 = estimator.estimate(track_id=1, bbox=bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        assert d1 == CameraDirection.FRONT

        # Frame 2: confirmed
        d2 = estimator.estimate(track_id=1, bbox=bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        assert d2 == CameraDirection.FRONT

    def test_front_left_direction_with_lane(self):
        """Object to the left of the left lane boundary is classified as FRONT_LEFT."""
        estimator = DirectionEstimator()
        lane_info = DummyLaneDetectionResult(left_x=450.0, right_x=830.0)
        # Target box centered at x=320 (left of left lane boundary 450)
        bbox = BoundingBox(x1=280.0, y1=400.0, x2=360.0, y2=500.0)

        estimator.estimate(track_id=2, bbox=bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        d = estimator.estimate(track_id=2, bbox=bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        assert d == CameraDirection.FRONT_LEFT

    def test_front_right_direction_with_lane(self):
        """Object to the right of the right lane boundary is classified as FRONT_RIGHT."""
        estimator = DirectionEstimator()
        lane_info = DummyLaneDetectionResult(left_x=450.0, right_x=830.0)
        # Target box centered at x=950 (right of right lane boundary 830)
        bbox = BoundingBox(x1=900.0, y1=400.0, x2=1000.0, y2=500.0)

        estimator.estimate(track_id=3, bbox=bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        d = estimator.estimate(track_id=3, bbox=bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        assert d == CameraDirection.FRONT_RIGHT

    def test_optical_periphery_left_and_right(self):
        """Object passing into lower image periphery is classified as LEFT or RIGHT."""
        estimator = DirectionEstimator()

        # Left flank: cx <= 0.15 * 1280 = 192, bottom_y > 0.50 * 720 = 360
        left_bbox = BoundingBox(x1=50.0, y1=400.0, x2=150.0, y2=600.0)
        estimator.estimate(track_id=4, bbox=left_bbox, img_width=1280.0, img_height=720.0)
        d_left = estimator.estimate(track_id=4, bbox=left_bbox, img_width=1280.0, img_height=720.0)
        assert d_left == CameraDirection.LEFT

        # Right flank: cx >= 0.85 * 1280 = 1088, bottom_y > 360
        right_bbox = BoundingBox(x1=1100.0, y1=400.0, x2=1220.0, y2=620.0)
        estimator.estimate(track_id=5, bbox=right_bbox, img_width=1280.0, img_height=720.0)
        d_right = estimator.estimate(track_id=5, bbox=right_bbox, img_width=1280.0, img_height=720.0)
        assert d_right == CameraDirection.RIGHT

    def test_fallback_without_lane_detection(self):
        """When lane detection is None, geometric image center corridor is used."""
        estimator = DirectionEstimator()
        # Default corridor width = 0.30 * 1280 = 384 => [448, 832]

        center_bbox = BoundingBox(x1=600.0, y1=350.0, x2=680.0, y2=450.0)
        estimator.estimate(track_id=6, bbox=center_bbox, lane_info=None, img_width=1280.0, img_height=720.0)
        d_center = estimator.estimate(track_id=6, bbox=center_bbox, lane_info=None, img_width=1280.0, img_height=720.0)
        assert d_center == CameraDirection.FRONT

        left_bbox = BoundingBox(x1=300.0, y1=350.0, x2=400.0, y2=450.0)
        estimator.estimate(track_id=7, bbox=left_bbox, lane_info=None, img_width=1280.0, img_height=720.0)
        d_left = estimator.estimate(track_id=7, bbox=left_bbox, lane_info=None, img_width=1280.0, img_height=720.0)
        assert d_left == CameraDirection.FRONT_LEFT

    def test_temporal_smoothing_boundary_jitter(self):
        """Single-frame jitter across lane boundary does not trigger immediate direction flap."""
        estimator = DirectionEstimator()
        lane_info = DummyLaneDetectionResult(left_x=450.0, right_x=830.0)

        # Stable FRONT object for 3 frames
        front_bbox = BoundingBox(x1=600.0, y1=400.0, x2=680.0, y2=500.0)
        for _ in range(3):
            estimator.estimate(track_id=10, bbox=front_bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        assert estimator.get_direction(10) == CameraDirection.FRONT

        # Single frame noisy jitter just touching right lane (cx = 835)
        jitter_bbox = BoundingBox(x1=800.0, y1=400.0, x2=870.0, y2=500.0)
        d_jitter = estimator.estimate(track_id=10, bbox=jitter_bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        # Should remain FRONT due to confirmation buffer
        assert d_jitter == CameraDirection.FRONT

        # Second sustained frame confirms new direction FRONT_RIGHT
        d_confirmed = estimator.estimate(track_id=10, bbox=jitter_bbox, lane_info=lane_info, img_width=1280.0, img_height=720.0)
        assert d_confirmed == CameraDirection.FRONT_RIGHT

    def test_no_rear_or_blind_spot_warning(self):
        """Verify that camera-relative directions strictly exclude physically unsupported rear zones."""
        # Available directions enum verification
        all_enum_values = {d.value for d in CameraDirection}
        assert "REAR" not in all_enum_values
        assert "REAR_LEFT" not in all_enum_values
        assert "REAR_RIGHT" not in all_enum_values
        assert "BLIND_SPOT" not in all_enum_values

        estimator = DirectionEstimator()
        # Even with extreme coordinates, output can only be FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, or UNKNOWN
        extreme_bbox = BoundingBox(x1=10.0, y1=650.0, x2=80.0, y2=710.0)
        res = estimator.estimate(track_id=99, bbox=extreme_bbox, img_width=1280.0, img_height=720.0)
        assert res in (CameraDirection.UNKNOWN, CameraDirection.LEFT)

    def test_unknown_for_empty_bbox(self):
        """None bbox yields UNKNOWN direction."""
        estimator = DirectionEstimator()
        res = estimator.estimate(track_id=100, bbox=None, img_width=1280.0, img_height=720.0)
        assert res == CameraDirection.UNKNOWN

    def test_stale_track_eviction_and_reset(self):
        """Stale track histories are cleaned up on eviction or reset."""
        estimator = DirectionEstimator()
        bbox = BoundingBox(x1=600.0, y1=400.0, x2=680.0, y2=500.0)

        estimator.estimate(track_id=20, bbox=bbox, timestamp=1.0, img_width=1280.0, img_height=720.0)
        assert 20 in estimator._history

        # Purge stale tracks with delta > 2.0s
        estimator.purge_stale_tracks(current_timestamp=4.0, stale_timeout_s=2.0)
        assert 20 not in estimator._history

        # Reset clears everything
        estimator.estimate(track_id=21, bbox=bbox, timestamp=4.5, img_width=1280.0, img_height=720.0)
        assert len(estimator._history) == 1
        estimator.reset()
        assert len(estimator._history) == 0
