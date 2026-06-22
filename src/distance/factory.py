from __future__ import annotations

from typing import Any, Optional

from src.distance.base import BaseDistanceEstimator
from src.distance.calibrated_distance import CalibratedDistanceEstimator
from src.distance.relative_distance import RelativeDistanceEstimator


class _NoopDistanceEstimator(BaseDistanceEstimator):
    """Fallback that always returns None."""

    def estimate(
        self,
        perceived: PerceivedObject,  # noqa: F821
        depth_map: Any,
        frame: Any,
    ) -> Optional[float]:
        return None


def build_distance_estimator(cfg: dict[str, Any]) -> BaseDistanceEstimator:
    """Build a distance estimator from the pipeline configuration.

    ``distance.type`` can be ``"relative"`` or ``"calibrated"``.

    If no ``distance`` section exists a no-op estimator returning None is
    returned.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        An initialised :class:`BaseDistanceEstimator` instance.
    """
    dist_cfg = cfg.get("distance")
    if dist_cfg is None:
        return _NoopDistanceEstimator()

    dist_type = dist_cfg.get("type", "relative")

    if dist_type == "relative":
        return RelativeDistanceEstimator()

    if dist_type == "calibrated":
        cal_cfg = dist_cfg.get("calibrated", {})
        return CalibratedDistanceEstimator(
            focal_length_px=float(cal_cfg.get("focal_length_px", 800.0)),
            class_real_heights=cal_cfg.get("class_real_heights"),
            fallback_height=float(cal_cfg.get("fallback_height", 1.5)),
        )

    raise ValueError(
        f"Unknown distance estimator type: '{dist_type}'. "
        f"Expected one of: 'relative', 'calibrated'."
    )
