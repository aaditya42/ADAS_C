from __future__ import annotations

import time
from typing import Any

from src.schemas.risk_object import RiskLevel
from src.schemas import Frame
from src.warning.channels import BaseWarningChannel


_LEVEL_ORDER: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
}


class WarningManager:
    """Routes risk objects through warning channels with hysteresis & cooldown.

    Args:
        channels: List of output channels to emit through.
        min_level: Minimum risk level that triggers a warning (e.g. "MEDIUM").
        cooldown_sec: Minimum seconds between repeated warnings for the same
            track.
        hysteresis_frames: Consecutive frames at/above ``min_level`` before
            the first warning is emitted.
    """

    def __init__(
        self,
        channels: list[BaseWarningChannel],
        min_level: str = "MEDIUM",
        cooldown_sec: float = 2.0,
        hysteresis_frames: int = 3,
    ) -> None:
        self._channels = list(channels)
        try:
            self._min_order = _LEVEL_ORDER[RiskLevel(min_level.lower())]
        except (ValueError, AttributeError):
            self._min_order = 2  # MEDIUM fallback

        self._cooldown = cooldown_sec
        self._hysteresis = hysteresis_frames

        # Per-track state
        self._consecutive: dict[int, int] = {}
        self._last_warn_time: dict[int, float] = {}

    def process(
        self, risk_objects: list[Any], frame: Frame
    ) -> list[dict[str, Any]]:
        """Evaluate risk objects and emit warnings where appropriate.

        Args:
            risk_objects: List of ``RiskObject`` from the risk engine.
            frame: Current video frame (used for frame_id).

        Returns:
            List of alert dicts, one per emitted warning::

                {"track_id": int, "risk_level": str,
                 "risk_score": float, "message": str}
        """
        now = time.monotonic()
        alerts: list[dict[str, Any]] = []

        seen_this_frame: set[int] = set()

        for risk in risk_objects:
            tid = risk.track_id
            seen_this_frame.add(tid)

            order = _LEVEL_ORDER.get(risk.risk_level, 0)

            if order < self._min_order:
                self._consecutive.pop(tid, None)
                continue

            # Hysteresis
            self._consecutive[tid] = self._consecutive.get(tid, 0) + 1
            if self._consecutive[tid] < self._hysteresis:
                continue

            # Cooldown
            last = self._last_warn_time.get(tid, -self._cooldown)
            if now - last < self._cooldown:
                continue

            self._last_warn_time[tid] = now

            message = (
                f"[{risk.risk_level.name}] Track {risk.track_id} "
                f"score={risk.risk_score:.0f}"
            )

            for ch in self._channels:
                ch.emit(tid, risk, message)

            alerts.append({
                "track_id": tid,
                "risk_level": risk.risk_level.name,
                "risk_score": risk.risk_score,
                "message": message,
                "frame_id": frame.frame_id,
            })

        # Cleanup stale tracks
        stale = set(self._consecutive.keys()) - seen_this_frame
        for tid in stale:
            del self._consecutive[tid]
            self._last_warn_time.pop(tid, None)

        return alerts
