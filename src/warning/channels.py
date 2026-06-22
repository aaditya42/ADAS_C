from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

from src.schemas.risk_object import RiskLevel, WarningLevel


# ------------------------------------------------------------------ #
#  Base
# ------------------------------------------------------------------ #

class BaseWarningChannel(ABC):
    """Abstract warning output channel.

    Contract:
      - emit() is called for each risk object that passes the warning
        manager's hysteresis / cooldown logic.
      - Implementations MUST NOT raise if the output medium is unavailable
        (e.g. sound library missing) — they degrade gracefully.
    """

    @abstractmethod
    def emit(
        self, track_id: int, risk: Any, message: str
    ) -> None: ...


# ------------------------------------------------------------------ #
#  Console
# ------------------------------------------------------------------ #

_LEVEL_COLORS: dict[RiskLevel, str] = {
    RiskLevel.SAFE: "\033[32m",    # green
    RiskLevel.LOW: "\033[33m",     # yellow
    RiskLevel.MEDIUM: "\033[38;5;208m",  # orange
    RiskLevel.HIGH: "\033[31m",    # red
}
_RESET = "\033[0m"


class ConsoleChannel(BaseWarningChannel):
    """Writes coloured warning messages to stderr."""

    def emit(self, track_id: int, risk: Any, message: str) -> None:
        color = _LEVEL_COLORS.get(risk.risk_level, "")
        print(f"{color}{message}{_RESET}", file=sys.stderr)


# ------------------------------------------------------------------ #
#  Audio beep
# ------------------------------------------------------------------ #

class AudioBeepChannel(BaseWarningChannel):
    """Plays a beep sound for warnings.

    On Windows uses ``winsound``.  On other platforms tries
    ``simpleaudio``; if unavailable, prints the ASCII bell character.
    """

    _BEEP_DURATIONS: dict[RiskLevel, int] = {
        RiskLevel.LOW: 100,
        RiskLevel.MEDIUM: 200,
        RiskLevel.HIGH: 400,
    }
    _BEEP_FREQ: int = 880

    def __init__(self) -> None:
        self._beep = _get_beep_func()

    def emit(self, track_id: int, risk: Any, message: str) -> None:
        dur = self._BEEP_DURATIONS.get(risk.risk_level, 100)
        self._beep(dur)


def _get_beep_func():
    """Return a ``beep(duration_ms)`` callable."""
    try:
        import winsound  # noqa: PLC0415
        return lambda ms: winsound.Beep(AudioBeepChannel._BEEP_FREQ, ms)
    except ImportError:
        pass

    try:
        import simpleaudio  # noqa: PLC0415, F811
        import numpy as np  # noqa: PLC0415
        _SR = 44100

        def _simpleaudio_beep(ms: int) -> None:
            t = np.linspace(0, ms / 1000.0, int(_SR * ms / 1000.0), endpoint=False)
            wave = (np.sin(2 * np.pi * AudioBeepChannel._BEEP_FREQ * t) * 32767).astype(np.int16)
            try:
                simpleaudio.play_buffer(wave, 1, 2, _SR)
            except Exception:
                pass  # gracefully degrade

        return _simpleaudio_beep
    except ImportError:
        pass

    return lambda ms: print("\a", end="", flush=True)


# ------------------------------------------------------------------ #
#  Reserved channel stubs (future)
# ------------------------------------------------------------------ #

# class VoiceTTSChannel(BaseWarningChannel): ...
# class HUDChannel(BaseWarningChannel): ...
# class CANBusChannel(BaseWarningChannel): ...
