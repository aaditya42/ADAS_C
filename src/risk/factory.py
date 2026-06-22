from __future__ import annotations

from typing import Any

from src.risk.base import BaseRiskEngine


def build_risk_engine(config: dict[str, Any]) -> BaseRiskEngine:
    """Build a risk engine from the configuration.

    Args:
        config: The full pipeline configuration dictionary.

    Returns:
        A ``BaseRiskEngine`` instance.
    """
    risk_cfg = config.get("risk", {})
    rtype = str(risk_cfg.get("type", "score_based"))

    if rtype == "score_based":
        from src.risk.score_based_risk import ScoreBasedRiskEngine  # noqa: PLC0415

        w = risk_cfg.get("weights", {})
        t = risk_cfg.get("thresholds", {})
        return ScoreBasedRiskEngine(
            weights={
                "distance": float(w.get("distance", 0.4)),
                "ttc": float(w.get("ttc", 0.5)),
                "speed": float(w.get("speed", 0.1)),
                "lane": float(w.get("lane", 0.0)),
                "scene": float(w.get("scene", 0.0)),
            },
            dist_critical_m=float(t.get("dist_critical_m", 5.0)),
            dist_safe_m=float(t.get("dist_safe_m", 50.0)),
            ttc_critical_s=float(t.get("ttc_critical_s", 1.0)),
            ttc_safe_s=float(t.get("ttc_safe_s", 6.0)),
            speed_critical_mps=float(t.get("speed_critical_mps", 15.0)),
        )

    raise ValueError(f"Unknown risk engine type: {rtype!r}")
