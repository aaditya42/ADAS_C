from __future__ import annotations

from typing import Any


def build_motion_estimator(config: dict[str, Any]) -> Any:
    """Build a motion estimator from the configuration.

    Args:
        config: The full pipeline configuration dictionary.

    Returns:
        A :class:`RelativeMotionEstimator` instance, or a noop fallback
        if the ``motion`` section is missing or disabled.
    """
    motion_cfg = config.get("motion", {})
    if not motion_cfg.get("enabled", True):
        return _NoopMotionEstimator()

    from src.motion.relative_motion import RelativeMotionEstimator  # noqa: PLC0415

    return RelativeMotionEstimator(
        min_samples=int(motion_cfg.get("min_samples", 5)),
        smoothing=str(motion_cfg.get("smoothing", "ema")),
        ema_alpha=float(motion_cfg.get("ema_alpha", 0.4)),
        stationary_threshold_mps=float(
            motion_cfg.get("stationary_threshold_mps", 0.3)
        ),
    )


class _NoopMotionEstimator:
    """Fallback that returns empty motion results."""

    def estimate(self, *args: Any, **kwargs: Any) -> dict[int, Any]:
        return {}
