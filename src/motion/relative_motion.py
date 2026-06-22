from __future__ import annotations

from typing import Any

import numpy as np

from src.schemas.motion import Direction, RelativeMotion


class RelativeMotionEstimator:
    """Estimates per-track speed and acceleration from history snapshots.

    Uses linear regression on (timestamp, distance) over a sliding window
    to compute velocity, then derives acceleration and direction.

    Args:
        min_samples: Minimum number of snapshots with valid distance needed.
        smoothing: Smoothing method (only ``"ema"`` supported).
        ema_alpha: Exponential moving average factor for speed (0 = very
            smooth, 1 = no smoothing).
        stationary_threshold_mps: Absolute speed below which an object is
            considered stationary.
    """

    def __init__(
        self,
        min_samples: int = 5,
        smoothing: str = "ema",
        ema_alpha: float = 0.4,
        stationary_threshold_mps: float = 0.3,
    ) -> None:
        self._min_samples = min_samples
        self._smoothing = smoothing
        self._ema_alpha = ema_alpha
        self._stationary_threshold = stationary_threshold_mps
        self._prev_speed: dict[int, float] = {}

    def estimate(
        self, history_manager: Any
    ) -> dict[int, RelativeMotion]:
        """Compute motion estimates for all active tracks.

        Args:
            history_manager: A ``TrackHistoryManager`` instance.

        Returns:
            Dict mapping track_id to ``RelativeMotion``.
        """
        active = history_manager.active_tracks()
        result: dict[int, RelativeMotion] = {}

        for tid in active:
            snaps = history_manager.get_history(tid)
            valid = [s for s in snaps if s.distance is not None and s.distance > 0]

            if len(valid) < self._min_samples:
                result[tid] = RelativeMotion(
                    track_id=tid,
                    speed_mps=None,
                    acceleration_mps2=None,
                    direction="unknown",
                    confidence=0.0,
                )
                continue

            recent = valid[-self._min_samples :]
            times = np.array([s.timestamp for s in recent], dtype=np.float64)
            dists = np.array([s.distance for s in recent], dtype=np.float64)

            n = len(recent)
            # Linear regression: dist = m * time + b
            t_mean = times.mean()
            d_mean = dists.mean()
            num = ((times - t_mean) * (dists - d_mean)).sum()
            den = ((times - t_mean) ** 2).sum()

            if den == 0:
                result[tid] = RelativeMotion(
                    track_id=tid,
                    speed_mps=None,
                    acceleration_mps2=None,
                    direction="unknown",
                    confidence=0.0,
                )
                continue

            slope = num / den  # m = Δdist / Δtime (negative = approaching)
            intercept = d_mean - slope * t_mean

            # Residual standard deviation
            residuals = dists - (slope * times + intercept)
            residual_std = float(np.std(residuals, ddof=1)) if n >= 3 else 0.0

            # Speed: negative slope = approaching → positive speed
            raw_speed = -slope

            # EMA smoothing
            prev = self._prev_speed.get(tid, raw_speed)
            if self._smoothing == "ema":
                smoothed = self._ema_alpha * raw_speed + (1 - self._ema_alpha) * prev
            else:
                smoothed = raw_speed
            self._prev_speed[tid] = smoothed

            # Acceleration via split-window regression
            accel: float | None = None
            if n >= 4:
                mid = n // 2
                t0, t1 = times[:mid], times[mid:]
                d0, d1 = dists[:mid], dists[mid:]
                if len(t0) >= 2 and len(t1) >= 2:
                    s0 = ((t0 - t0.mean()) * (d0 - d0.mean())).sum() / max(
                        ((t0 - t0.mean()) ** 2).sum(), 1e-12
                    )
                    s1 = ((t1 - t1.mean()) * (d1 - d1.mean())).sum() / max(
                        ((t1 - t1.mean()) ** 2).sum(), 1e-12
                    )
                    dt_half = (times[-1] - times[0]) / 2.0
                    if dt_half > 0:
                        accel = (-s1 - (-s0)) / dt_half
            if accel is None:
                accel = 0.0

            # Direction
            direction: Direction
            if abs(smoothed) <= self._stationary_threshold:
                direction = "stationary"
            elif smoothed > 0:
                direction = "approaching"
            else:
                direction = "receding"

            # Confidence
            conf_n = min(1.0, n / 15.0)
            if residual_std > 0:
                confidence = conf_n * (1.0 / (1.0 + residual_std))
            else:
                confidence = conf_n
            confidence = max(0.0, min(1.0, confidence))

            result[tid] = RelativeMotion(
                track_id=tid,
                speed_mps=float(f"{smoothed:.4f}"),
                acceleration_mps2=float(f"{accel:.4f}"),
                direction=direction,
                confidence=float(f"{confidence:.4f}"),
            )

        # Cleanup stale tracks
        stale = set(self._prev_speed.keys()) - active
        for tid in stale:
            del self._prev_speed[tid]

        return result
