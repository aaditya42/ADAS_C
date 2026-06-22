from __future__ import annotations

from typing import Any

from src.ttc.base import BaseTTCEstimator


def build_ttc(config: dict[str, Any]) -> BaseTTCEstimator:
    """Build a TTC estimator from the configuration.

    Args:
        config: The full pipeline configuration dictionary.

    Returns:
        A ``BaseTTCEstimator`` instance.
    """
    ttc_cfg = config.get("ttc", {})
    ttc_type = str(ttc_cfg.get("type", "simple"))

    if ttc_type == "simple":
        from src.ttc.simple_ttc import SimpleTTCEstimator  # noqa: PLC0415

        return SimpleTTCEstimator(
            max_ttc_seconds=float(ttc_cfg.get("max_ttc_seconds", 30.0)),
            min_speed_mps=float(ttc_cfg.get("min_speed_mps", 0.2)),
        )

    raise ValueError(f"Unknown TTC type: {ttc_type!r}")
