from __future__ import annotations

import numpy as np
import pytest

from src.memory.track_history_manager import TrackHistoryManager
from src.motion.factory import build_motion_estimator
from src.motion.relative_motion import RelativeMotionEstimator
from src.schemas import BoundingBox, Frame, PerceivedObject


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_frame(frame_id: int, timestamp: float | None = None) -> Frame:
    if timestamp is None:
        timestamp = float(frame_id) / 30.0
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=timestamp,
        source="test",
    )


def _make_perceived(
    track_id: int = 1,
    distance: float = 10.0,
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


def _build_history(
    distances: list[float],
    track_id: int = 1,
    start_time: float = 0.0,
    fps: float = 30.0,
) -> TrackHistoryManager:
    """Build a TrackHistoryManager with a track at known distances."""
    mgr = TrackHistoryManager(max_history_per_track=60, stale_after_frames=100)
    for i, d in enumerate(distances):
        ts = start_time + i / fps
        frame = _make_frame(i + 1, ts)
        obj = _make_perceived(track_id=track_id, distance=d)
        mgr.update([obj], frame)
    return mgr


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestRelativeMotionEstimator:
    def test_approaching_constant_speed(self) -> None:
        """Distance decreasing at 5 m/s over 10 frames."""
        fps = 30.0
        d0 = 50.0
        speed = -5.0  # m/s (negative = decreasing distance)
        distances = [d0 + speed * (i / fps) for i in range(10)]
        mgr = _build_history(distances, start_time=0.0, fps=fps)

        est = RelativeMotionEstimator(min_samples=5)
        result = est.estimate(mgr)
        mot = result[1]

        assert mot.speed_mps is not None
        assert mot.speed_mps == pytest.approx(5.0, abs=0.5)  # approaching
        assert mot.direction == "approaching"

    def test_receding_constant_speed(self) -> None:
        """Distance increasing at 3 m/s."""
        fps = 30.0
        d0 = 10.0
        speed = 3.0
        distances = [d0 + speed * (i / fps) for i in range(10)]
        mgr = _build_history(distances, start_time=0.0, fps=fps)

        est = RelativeMotionEstimator(min_samples=5)
        result = est.estimate(mgr)
        mot = result[1]

        assert mot.speed_mps is not None
        assert mot.speed_mps == pytest.approx(-3.0, abs=0.5)  # receding
        assert mot.direction == "receding"

    def test_stationary(self) -> None:
        """Constant distance → stationary."""
        distances = [20.0] * 10
        mgr = _build_history(distances, start_time=0.0)

        est = RelativeMotionEstimator(
            min_samples=5, stationary_threshold_mps=0.5
        )
        result = est.estimate(mgr)
        mot = result[1]

        assert mot.speed_mps is not None
        assert abs(mot.speed_mps) < 0.5
        assert mot.direction == "stationary"

    def test_insufficient_samples(self) -> None:
        """Fewer than min_samples → unknown."""
        distances = [20.0] * 3
        mgr = _build_history(distances, start_time=0.0)

        est = RelativeMotionEstimator(min_samples=5)
        result = est.estimate(mgr)
        mot = result[1]

        assert mot.speed_mps is None
        assert mot.acceleration_mps2 is None
        assert mot.direction == "unknown"
        assert mot.confidence == pytest.approx(0.0)

    def test_no_valid_distance(self) -> None:
        """All snapshots have None distance → unknown."""
        mgr = TrackHistoryManager(max_history_per_track=30, stale_after_frames=100)
        for i in range(10):
            obj = _make_perceived(track_id=1, distance=0.0, depth_confidence=0.0)
            mgr.update([obj], _make_frame(i + 1))

        est = RelativeMotionEstimator(min_samples=5)
        result = est.estimate(mgr)
        mot = result[1]

        assert mot.speed_mps is None
        assert mot.direction == "unknown"

    def test_empty_history(self) -> None:
        """No tracks → empty result."""
        mgr = TrackHistoryManager()
        est = RelativeMotionEstimator()
        assert est.estimate(mgr) == {}

    def test_multiple_tracks(self) -> None:
        """Two tracks with different motions (same history manager)."""
        mgr = TrackHistoryManager(max_history_per_track=30, stale_after_frames=100)
        for i in range(10):
            ts = i / 30.0
            d1 = 50.0 - 5.0 * (i / 30.0)   # approaching at 5 m/s
            d2 = 10.0 + 2.0 * (i / 30.0)   # receding at 2 m/s
            frame = _make_frame(i + 1, ts)
            mgr.update([
                _make_perceived(track_id=1, distance=d1),
                _make_perceived(track_id=2, distance=d2),
            ], frame)

        est = RelativeMotionEstimator(min_samples=5)
        result = est.estimate(mgr)

        assert 1 in result
        assert 2 in result
        # Track 1 approaching (distance decreasing)
        assert result[1].speed_mps is not None
        assert result[1].speed_mps > 2.0  # should be ~5
        # Track 2 receding (distance increasing)
        assert result[2].speed_mps is not None
        assert result[2].speed_mps < -1.0  # should be ~-2

    def test_ema_smoothing(self) -> None:
        """EMA produces different result with alpha=1 vs alpha=0.1."""
        fps = 30.0
        # Noisy distances around a downward trend
        np.random.seed(42)
        trend = np.linspace(50.0, 20.0, 10)
        noise = np.random.uniform(-3, 3, 10)
        distances = list(trend + noise)
        mgr = _build_history(distances, start_time=0.0, fps=fps)

        est_smooth = RelativeMotionEstimator(
            min_samples=5, ema_alpha=0.1
        )
        est_raw = RelativeMotionEstimator(
            min_samples=5, ema_alpha=1.0
        )

        r1 = est_smooth.estimate(mgr)
        r2 = est_raw.estimate(mgr)

        assert r1[1].speed_mps is not None
        assert r2[1].speed_mps is not None
        # Both should indicate approaching (positive speed)
        assert r1[1].speed_mps > 0
        assert r2[1].speed_mps > 0


class TestFactory:
    def test_returns_relative_motion_estimator(self) -> None:
        est = build_motion_estimator({"motion": {"enabled": True}})
        assert isinstance(est, RelativeMotionEstimator)

    def test_returns_noop_when_disabled(self) -> None:
        from src.motion.factory import _NoopMotionEstimator
        est = build_motion_estimator({"motion": {"enabled": False}})
        assert isinstance(est, _NoopMotionEstimator)

    def test_returns_relative_motion_on_missing_section(self) -> None:
        est = build_motion_estimator({})
        assert isinstance(est, RelativeMotionEstimator)

    def test_pipeline_integration(self) -> None:
        """Full pipeline process() returns motion key."""
        from src.core.pipeline import PerceptionPipeline
        pipe = PerceptionPipeline({})
        frame = _make_frame(1, 0.0)
        result = pipe.process(frame)
        assert "motion" in result
        assert isinstance(result["motion"], dict)
