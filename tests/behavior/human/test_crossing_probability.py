from __future__ import annotations

import pytest

from src.behavior.base import LaneContext
from src.behavior.config import CrossingProbabilityConfig
from src.behavior.human.crossing_probability import (
    GeometricCrossingProbabilityEstimator,
)
from src.schemas.lane_info import LaneLine


def _make_lane_context(
    left_coeffs: tuple[float, ...] | None = (0.0, 1.0, 100.0),
    right_coeffs: tuple[float, ...] | None = (0.0, 1.0, 500.0),
) -> LaneContext:
    return LaneContext(
        left_boundary=LaneLine(coeffs=left_coeffs, confidence=0.9)
        if left_coeffs
        else None,
        right_boundary=LaneLine(coeffs=right_coeffs, confidence=0.9)
        if right_coeffs
        else None,
        confidence=0.9,
    )


_DEFAULT_CFG = CrossingProbabilityConfig(
    decay_distance_px=50.0,
    heading_weight=0.3,
    speed_weight=0.2,
)


class TestGeometricCrossingProbabilityEstimator:
    """Tests for ``GeometricCrossingProbabilityEstimator``."""

    def test_no_lane_context_returns_zero(self) -> None:
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        prob = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=None,
            heading_deg=0.0,
            speed_mps=1.0,
        )
        assert prob == 0.0

    def test_no_boundaries_returns_zero(self) -> None:
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(left_coeffs=None, right_coeffs=None)
        prob = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=1.0,
        )
        assert prob == 0.0

    def test_inside_lane_returns_one(self) -> None:
        """Entity between left and right lane boundaries → probability = 1.0."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        # At y=400, left lane x = 500, right lane x = 800
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        prob = est.estimate(
            foot_x=650.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        assert prob == 1.0

    def test_on_left_boundary_returns_one(self) -> None:
        """Entity exactly on left boundary."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 300.0),
            right_coeffs=(0.0, 0.0, 600.0),
        )
        prob = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        assert prob == 1.0

    def test_far_from_lane_returns_near_zero(self) -> None:
        """Entity very far from lane → probability ≈ 0."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        # 2000 pixels away from nearest lane edge
        prob = est.estimate(
            foot_x=0.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        # exp(-2000/50) ≈ 0
        assert prob < 0.01

    def test_heading_toward_road_increases_probability(self) -> None:
        """Heading toward road → higher than heading away."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        # Entity at x=300, lane at x=500, need to move right
        # heading_deg=90 means moving right in image → toward road
        prob_toward = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=90.0,
            speed_mps=0.0,
        )
        # heading_deg=-90 means moving left → away from road
        prob_away = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=-90.0,
            speed_mps=0.0,
        )
        assert prob_toward > prob_away

    def test_heading_parallel_has_lower_probability(self) -> None:
        """Walking parallel to road (heading 0°) → lower probability."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        prob_parallel = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        prob_toward = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=90.0,
            speed_mps=0.0,
        )
        assert prob_parallel < prob_toward

    def test_speed_improves_probability(self) -> None:
        """Higher speed toward road → higher probability."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        prob_no_speed = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=90.0,
            speed_mps=0.0,
        )
        prob_with_speed = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=90.0,
            speed_mps=2.0,
        )
        assert prob_with_speed >= prob_no_speed

    def test_distance_monotonic(self) -> None:
        """Closer to lane → strictly higher probability, all else equal."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        prob_far = est.estimate(
            foot_x=10.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=90.0,
            speed_mps=0.5,
        )
        prob_close = est.estimate(
            foot_x=400.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=90.0,
            speed_mps=0.5,
        )
        assert prob_close > prob_far

    def test_curved_lane(self) -> None:
        """Polynomial lane with curvature."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        # Curved lane: x = a*y² + b*y + c
        # At y=400: left=0.001*160000 - 0.5*400 + 300 = 160 - 200 + 300 = 260
        # right=0.001*160000 - 0.5*400 + 600 = 160 - 200 + 600 = 560
        ctx = _make_lane_context(
            left_coeffs=(0.001, -0.5, 300.0),
            right_coeffs=(0.001, -0.5, 600.0),
        )
        # Inside lane
        prob_inside = est.estimate(
            foot_x=400.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        assert prob_inside == pytest.approx(1.0)

        # Outside to the left
        prob_outside = est.estimate(
            foot_x=100.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        assert 0.0 < prob_outside < 1.0

    def test_only_left_boundary(self) -> None:
        """Only a left boundary is available."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=None,
        )
        prob = est.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        assert 0.0 < prob < 1.0

    def test_only_right_boundary(self) -> None:
        """Only a right boundary is available."""
        est = GeometricCrossingProbabilityEstimator(_DEFAULT_CFG)
        ctx = _make_lane_context(
            left_coeffs=None,
            right_coeffs=(0.0, 0.0, 500.0),
        )
        prob = est.estimate(
            foot_x=600.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=0.0,
            speed_mps=0.0,
        )
        assert 0.0 < prob < 1.0

    def test_config_custom_decay(self) -> None:
        """Shorter decay → probability drops faster with distance."""
        cfg_fast_decay = CrossingProbabilityConfig(
            decay_distance_px=10.0,
            heading_weight=0.0,
            speed_weight=0.0,
        )
        cfg_slow_decay = CrossingProbabilityConfig(
            decay_distance_px=200.0,
            heading_weight=0.0,
            speed_weight=0.0,
        )
        est_fast = GeometricCrossingProbabilityEstimator(cfg_fast_decay)
        est_slow = GeometricCrossingProbabilityEstimator(cfg_slow_decay)
        ctx = _make_lane_context(
            left_coeffs=(0.0, 0.0, 500.0),
            right_coeffs=(0.0, 0.0, 800.0),
        )
        p_fast = est_fast.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=None,
            speed_mps=None,
        )
        p_slow = est_slow.estimate(
            foot_x=300.0,
            foot_y=400.0,
            lane_context=ctx,
            heading_deg=None,
            speed_mps=None,
        )
        assert p_fast < p_slow
