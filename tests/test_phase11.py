from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from src.schemas import RiskLevel, RiskObject, WarningLevel
from src.schemas.risk_object import RiskLevel as RL
from src.warning.channels import BaseWarningChannel, ConsoleChannel
from src.warning.factory import build_warning_manager
from src.warning.warning_manager import WarningManager


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

@dataclass
class _FakeFrame:
    frame_id: int = 1
    image: Any = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.uint8))
    timestamp: float = 0.0
    source: str = "test"


def _make_risk(
    track_id: int = 1,
    level: RiskLevel = RiskLevel.HIGH,
    score: float = 80.0,
) -> RiskObject:
    warning_map = {
        RiskLevel.SAFE: WarningLevel.NONE,
        RiskLevel.LOW: WarningLevel.ADVISORY,
        RiskLevel.MEDIUM: WarningLevel.CAUTION,
        RiskLevel.HIGH: WarningLevel.CRITICAL,
    }
    return RiskObject(
        track_id=track_id,
        risk_level=level,
        risk_score=score,
        warning_level=warning_map[level],
    )


class SpyChannel(BaseWarningChannel):
    """Records all emit calls for test assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, Any, str]] = []

    def emit(self, track_id: int, risk: Any, message: str) -> None:
        self.calls.append((track_id, risk, message))


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestWarningManager:
    def test_hysteresis_delays_first_emit(self) -> None:
        """HIGH risk must be seen 3 frames before first emit."""
        spy = SpyChannel()
        mgr = WarningManager(
            channels=[spy], hysteresis_frames=3, cooldown_sec=0.0
        )
        risk = [_make_risk(track_id=1)]

        # Frame 1 — not enough
        mgr.process(risk, _FakeFrame(1))
        assert len(spy.calls) == 0

        # Frame 2 — still not enough
        mgr.process(risk, _FakeFrame(2))
        assert len(spy.calls) == 0

        # Frame 3 — reaches threshold → emit
        mgr.process(risk, _FakeFrame(3))
        assert len(spy.calls) == 1

    def test_low_risk_never_emits_with_medium_min(self) -> None:
        """LOW risk with min_level=MEDIUM never emits."""
        spy = SpyChannel()
        mgr = WarningManager(
            channels=[spy], min_level="MEDIUM",
            hysteresis_frames=1, cooldown_sec=0.0,
        )
        risk = [_make_risk(track_id=1, level=RiskLevel.LOW, score=30.0)]

        for fid in range(10):
            mgr.process(risk, _FakeFrame(fid))

        assert len(spy.calls) == 0

    def test_cooldown_blocks_immediate_second_emit(self) -> None:
        """After emitting, cooldown prevents next-frame emit."""
        spy = SpyChannel()
        mgr = WarningManager(
            channels=[spy], hysteresis_frames=1,
            cooldown_sec=999.0,  # long cooldown
        )
        risk = [_make_risk(track_id=1)]

        mgr.process(risk, _FakeFrame(1))
        assert len(spy.calls) == 1  # first emit

        mgr.process(risk, _FakeFrame(2))
        assert len(spy.calls) == 1  # blocked by cooldown

    def test_multiple_channels(self) -> None:
        """All channels receive the emit."""
        spy1 = SpyChannel()
        spy2 = SpyChannel()
        mgr = WarningManager(
            channels=[spy1, spy2], hysteresis_frames=1, cooldown_sec=0.0,
        )
        risk = [_make_risk(track_id=1)]

        mgr.process(risk, _FakeFrame(1))
        assert len(spy1.calls) == 1
        assert len(spy2.calls) == 1

    def test_alert_contains_message_and_metadata(self) -> None:
        """Returned alert dict has expected fields."""
        spy = SpyChannel()
        mgr = WarningManager(
            channels=[spy], hysteresis_frames=1, cooldown_sec=0.0,
        )
        risk = [_make_risk(track_id=42, level=RiskLevel.HIGH, score=85.0)]

        alerts = mgr.process(risk, _FakeFrame(7))
        assert len(alerts) == 1
        a = alerts[0]
        assert a["track_id"] == 42
        assert a["risk_level"] == "HIGH"
        assert a["risk_score"] == 85.0
        assert "Track 42" in a["message"]
        assert a["frame_id"] == 7

    def test_risk_drops_below_threshold_resets_hysteresis(self) -> None:
        """After one HIGH then one SAFE, counter resets."""
        spy = SpyChannel()
        mgr = WarningManager(
            channels=[spy], hysteresis_frames=2, cooldown_sec=0.0,
        )
        high = [_make_risk(track_id=1, level=RiskLevel.HIGH)]
        safe = [_make_risk(track_id=1, level=RiskLevel.SAFE, score=0.0)]

        mgr.process(high, _FakeFrame(1))   # count=1
        mgr.process(safe, _FakeFrame(2))   # resets
        mgr.process(high, _FakeFrame(3))   # count=1 again
        assert len(spy.calls) == 0

        mgr.process(high, _FakeFrame(4))   # count=2 → emit
        assert len(spy.calls) == 1

    def test_stale_track_cleanup(self) -> None:
        """Track not seen anymore gets its state dropped."""
        spy = SpyChannel()
        mgr = WarningManager(
            channels=[spy], hysteresis_frames=1, cooldown_sec=0.0,
        )
        mgr.process([_make_risk(track_id=1)], _FakeFrame(1))
        mgr.process([], _FakeFrame(2))  # no risk objects
        # Consecutive counter should be empty for track 1
        assert 1 not in mgr._consecutive


class TestFactory:
    def test_builds_with_console_channel(self) -> None:
        mgr = build_warning_manager({
            "warning": {
                "min_level": "LOW",
                "channels": ["console"],
            }
        })
        assert isinstance(mgr, WarningManager)
        assert len(mgr._channels) == 1
        assert isinstance(mgr._channels[0], ConsoleChannel)

    def test_builds_with_empty_channels(self) -> None:
        mgr = build_warning_manager({
            "warning": {
                "channels": [],
            }
        })
        assert isinstance(mgr, WarningManager)
        assert len(mgr._channels) == 0

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
        assert "alerts" in result
        assert isinstance(result["alerts"], list)
