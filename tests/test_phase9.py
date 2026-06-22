from __future__ import annotations

import math

import numpy as np
import pytest

from src.schemas import BoundingBox, PerceivedObject
from src.schemas.motion import RelativeMotion
from src.ttc.base import BaseTTCEstimator
from src.ttc.factory import build_ttc
from src.ttc.simple_ttc import SimpleTTCEstimator


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_perceived(
    track_id: int = 1,
    distance: float = 20.0,
    depth_confidence: float = 0.9,
) -> PerceivedObject:
    return PerceivedObject(
        track_id=track_id,
        class_name="car",
        bbox=BoundingBox(10, 20, 100, 200),
        depth=distance,
        distance=distance,
        depth_confidence=depth_confidence,
    )


def _make_motion(
    track_id: int = 1,
    speed: float = 10.0,
) -> RelativeMotion:
    return RelativeMotion(
        track_id=track_id,
        speed_mps=speed,
        acceleration_mps2=0.0,
        direction="approaching" if speed > 0.3 else "stationary",
        confidence=0.9,
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestSimpleTTCEstimator:
    def test_approaching_calculates_ttc(self) -> None:
        """20 m at 10 m/s → TTC = 2.0 s."""
        est = SimpleTTCEstimator()
        p = _make_perceived(track_id=1, distance=20.0)
        m = {1: _make_motion(track_id=1, speed=10.0)}
        result = est.estimate([p], m)
        assert result[1] == pytest.approx(2.0)

    def test_stationary_returns_inf(self) -> None:
        """Speed ≈ 0 → inf."""
        est = SimpleTTCEstimator(min_speed_mps=0.2)
        p = _make_perceived(track_id=1, distance=20.0)
        m = {1: _make_motion(track_id=1, speed=0.0)}
        result = est.estimate([p], m)
        assert result[1] == math.inf

    def test_speed_below_threshold_returns_inf(self) -> None:
        """Speed below min_speed_mps → inf."""
        est = SimpleTTCEstimator(min_speed_mps=1.0)
        p = _make_perceived(track_id=1, distance=20.0)
        m = {1: _make_motion(track_id=1, speed=0.5)}
        result = est.estimate([p], m)
        assert result[1] == math.inf

    def test_receding_speed_returns_inf(self) -> None:
        """Negative speed (receding) → inf."""
        est = SimpleTTCEstimator()
        p = _make_perceived(track_id=1, distance=20.0)
        m = {1: _make_motion(track_id=1, speed=-3.0)}
        result = est.estimate([p], m)
        assert result[1] == math.inf

    def test_no_motion_data_returns_none(self) -> None:
        """Missing motion entry → None."""
        est = SimpleTTCEstimator()
        p = _make_perceived(track_id=1, distance=20.0)
        result = est.estimate([p], {})
        assert result[1] is None

    def test_speed_none_returns_none(self) -> None:
        """Motion with None speed → None."""
        est = SimpleTTCEstimator()
        p = _make_perceived(track_id=1, distance=20.0)
        m = {1: RelativeMotion(
            track_id=1, speed_mps=None, acceleration_mps2=None,
            direction="unknown", confidence=0.0,
        )}
        result = est.estimate([p], m)
        assert result[1] is None

    def test_zero_distance_returns_none(self) -> None:
        """Zero or negative distance → None (invalid)."""
        est = SimpleTTCEstimator()
        p = _make_perceived(track_id=1, distance=0.0)
        m = {1: _make_motion(track_id=1, speed=5.0)}
        result = est.estimate([p], m)
        assert result[1] is None

    def test_clips_to_max_ttc(self) -> None:
        """Very large TTC clips to max_ttc_seconds."""
        est = SimpleTTCEstimator(max_ttc_seconds=10.0)
        p = _make_perceived(track_id=1, distance=100.0)
        m = {1: _make_motion(track_id=1, speed=1.0)}
        result = est.estimate([p], m)
        assert result[1] == pytest.approx(10.0)  # would be 100, clipped


class TestFactory:
    def test_returns_simple_ttc(self) -> None:
        est = build_ttc({"ttc": {"type": "simple"}})
        assert isinstance(est, SimpleTTCEstimator)

    def test_returns_simple_by_default(self) -> None:
        est = build_ttc({})
        assert isinstance(est, SimpleTTCEstimator)

    def test_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown TTC type"):
            build_ttc({"ttc": {"type": "trajectory"}})

    def test_pass_config_values(self) -> None:
        est = build_ttc({
            "ttc": {
                "type": "simple",
                "max_ttc_seconds": 15.0,
                "min_speed_mps": 0.5,
            }
        })
        assert isinstance(est, SimpleTTCEstimator)
        assert est._max_ttc == pytest.approx(15.0)
        assert est._min_speed == pytest.approx(0.5)

    def test_pipeline_integration(self) -> None:
        """Full pipeline process() returns ttc key."""
        from src.core.pipeline import PerceptionPipeline
        pipe = PerceptionPipeline({})
        frame = type(
            "Frame", (), {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "frame_id": 1, "timestamp": 0.0, "source": "test",
            }
        )()
        result = pipe.process(frame)
        assert "ttc" in result
        assert isinstance(result["ttc"], dict)
