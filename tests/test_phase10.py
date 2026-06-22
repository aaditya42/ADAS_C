from __future__ import annotations

import math

import numpy as np
import pytest

from src.risk.base import BaseRiskEngine
from src.risk.factory import build_risk_engine
from src.risk.score_based_risk import ScoreBasedRiskEngine
from src.schemas import BoundingBox, PerceivedObject
from src.schemas.motion import RelativeMotion
from src.schemas.risk_object import RiskLevel, RiskObject, WarningLevel


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_perceived(
    track_id: int = 1,
    distance: float = 50.0,
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


def _make_motion(speed: float) -> RelativeMotion:
    direction = "approaching" if speed > 0.3 else "stationary"
    return RelativeMotion(
        track_id=1, speed_mps=speed, acceleration_mps2=0.0,
        direction=direction,  # type: ignore[arg-type]
        confidence=0.9,
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestScoreBasedRiskEngine:
    def test_far_slow_safe(self) -> None:
        """50 m away, 1 m/s → should be SAFE."""
        engine = ScoreBasedRiskEngine()
        p = [_make_perceived(track_id=1, distance=50.0)]
        m = {1: _make_motion(speed=1.0)}
        t = {1: 50.0}
        result = engine.evaluate(p, m, t)
        assert len(result) == 1
        r = result[0]
        assert r.risk_level == RiskLevel.SAFE
        assert r.risk_score < 25

    def test_close_high_ttc_high(self) -> None:
        """5 m away, 0.5 s TTC → HIGH."""
        engine = ScoreBasedRiskEngine()
        p = [_make_perceived(track_id=1, distance=5.0)]
        m = {1: _make_motion(speed=10.0)}
        t = {1: 0.5}
        result = engine.evaluate(p, m, t)
        assert len(result) == 1
        r = result[0]
        assert r.risk_level == RiskLevel.HIGH
        assert r.risk_score >= 75

    def test_distance_none_safe(self) -> None:
        """No distance data → lower risk."""
        engine = ScoreBasedRiskEngine()
        p = [_make_perceived(track_id=1, distance=0.0, depth_confidence=0.0)]
        m = {1: _make_motion(speed=0.0)}
        t = {1: math.inf}
        result = engine.evaluate(p, m, t)
        r = result[0]
        assert r.risk_level == RiskLevel.SAFE

    def test_score_distance_critical(self) -> None:
        assert ScoreBasedRiskEngine._score_distance(0.0, 5.0, 50.0) == 100.0
        assert ScoreBasedRiskEngine._score_distance(5.0, 5.0, 50.0) == 100.0

    def test_score_distance_safe(self) -> None:
        assert ScoreBasedRiskEngine._score_distance(50.0, 5.0, 50.0) == 0.0
        assert ScoreBasedRiskEngine._score_distance(100.0, 5.0, 50.0) == 0.0

    def test_score_distance_none(self) -> None:
        assert ScoreBasedRiskEngine._score_distance(None, 5.0, 50.0) == 0.0

    def test_score_distance_ramp(self) -> None:
        """27.5 m is halfway between 5 and 50 → score = 50."""
        score = ScoreBasedRiskEngine._score_distance(27.5, 5.0, 50.0)
        assert score == pytest.approx(50.0, abs=0.1)

    def test_score_ttc_critical(self) -> None:
        assert ScoreBasedRiskEngine._score_ttc(0.5, 1.0, 6.0) == 100.0
        assert ScoreBasedRiskEngine._score_ttc(1.0, 1.0, 6.0) == 100.0

    def test_score_ttc_safe(self) -> None:
        assert ScoreBasedRiskEngine._score_ttc(6.0, 1.0, 6.0) == 0.0
        assert ScoreBasedRiskEngine._score_ttc(10.0, 1.0, 6.0) == 0.0

    def test_score_ttc_inf_none(self) -> None:
        assert ScoreBasedRiskEngine._score_ttc(math.inf, 1.0, 6.0) == 0.0
        assert ScoreBasedRiskEngine._score_ttc(None, 1.0, 6.0) == 0.0

    def test_score_ttc_ramp(self) -> None:
        """3.5 s is halfway between 1 and 6 → score = 50."""
        score = ScoreBasedRiskEngine._score_ttc(3.5, 1.0, 6.0)
        assert score == pytest.approx(50.0, abs=0.1)

    def test_score_speed_none_or_zero(self) -> None:
        assert ScoreBasedRiskEngine._score_speed(None, 15.0) == 0.0
        assert ScoreBasedRiskEngine._score_speed(0.0, 15.0) == 0.0
        assert ScoreBasedRiskEngine._score_speed(-5.0, 15.0) == 0.0

    def test_score_speed_critical(self) -> None:
        assert ScoreBasedRiskEngine._score_speed(15.0, 15.0) == 100.0
        assert ScoreBasedRiskEngine._score_speed(20.0, 15.0) == 100.0

    def test_score_speed_ramp(self) -> None:
        """7.5 m/s is halfway to 15 → score = 50."""
        score = ScoreBasedRiskEngine._score_speed(7.5, 15.0)
        assert score == pytest.approx(50.0, abs=0.1)

    def test_lane_and_scene_placeholders(self) -> None:
        obj = _make_perceived(track_id=1)
        assert ScoreBasedRiskEngine._score_lane(None, obj) == 0.0
        assert ScoreBasedRiskEngine._score_scene(None, obj) == 0.0

    def test_multiple_objects(self) -> None:
        """Two objects with different risk levels."""
        engine = ScoreBasedRiskEngine()
        p = [
            _make_perceived(track_id=1, distance=5.0),
            _make_perceived(track_id=2, distance=60.0),
        ]
        m = {
            1: _make_motion(speed=10.0),
            2: _make_motion(speed=0.5),
        }
        t = {1: 0.5, 2: 120.0}
        result = engine.evaluate(p, m, t)
        assert len(result) == 2
        by_id = {r.track_id: r for r in result}
        assert by_id[1].risk_level == RiskLevel.HIGH
        assert by_id[2].risk_level == RiskLevel.SAFE

    def test_warning_mirrors_risk(self) -> None:
        engine = ScoreBasedRiskEngine()
        p = [_make_perceived(track_id=1, distance=5.0)]
        m = {1: _make_motion(speed=10.0)}
        t = {1: 0.5}
        result = engine.evaluate(p, m, t)
        r = result[0]
        assert r.warning_level == WarningLevel.CRITICAL

    def test_safe_warning_none(self) -> None:
        engine = ScoreBasedRiskEngine()
        p = [_make_perceived(track_id=1, distance=100.0)]
        m = {1: _make_motion(speed=0.0)}
        t = {1: math.inf}
        result = engine.evaluate(p, m, t)
        assert result[0].warning_level == WarningLevel.NONE


class TestFactory:
    def test_returns_score_based(self) -> None:
        eng = build_risk_engine({"risk": {"type": "score_based"}})
        assert isinstance(eng, ScoreBasedRiskEngine)

    def test_returns_score_based_by_default(self) -> None:
        eng = build_risk_engine({})
        assert isinstance(eng, ScoreBasedRiskEngine)

    def test_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown risk engine type"):
            build_risk_engine({"risk": {"type": "unknown"}})

    def test_pipeline_integration(self) -> None:
        from src.core.pipeline import PerceptionPipeline
        pipe = PerceptionPipeline({})
        frame = type(
            "Frame", (), {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "frame_id": 1, "timestamp": 0.0, "source": "test",
            }
        )()
        result = pipe.process(frame)
        assert "risk" in result
        assert isinstance(result["risk"], list)
