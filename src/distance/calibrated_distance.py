from __future__ import annotations

from typing import Optional

import numpy as np

from src.distance.base import BaseDistanceEstimator
from src.schemas import Frame, PerceivedObject


class CalibratedDistanceEstimator(BaseDistanceEstimator):
    """Metric distance estimator using the pinhole height-prior method.

    ``distance ≈ (real_world_height * focal_length_px) / bbox_height_px``

    Each object class is associated with a prior real-world height (in
    meters). The focal length is a camera intrinsic read from configuration.

    The result is clipped to ``[0.5, 200.0]`` meters.

    Args:
        focal_length_px: Camera focal length in pixels (fx).
        class_real_heights: Mapping from class name to real-world height
            in meters, e.g. ``{"person": 1.7, "car": 1.5}``.
        fallback_height: Height used for unknown classes.
    """

    _CLIP_MIN = 0.5
    _CLIP_MAX = 200.0

    def __init__(
        self,
        focal_length_px: float = 800.0,
        class_real_heights: Optional[dict[str, float]] = None,
        fallback_height: float = 1.5,
    ) -> None:
        self._focal = focal_length_px
        self._heights = class_real_heights or {}
        self._fallback = fallback_height

    def estimate(
        self,
        perceived: PerceivedObject,
        depth_map: np.ndarray | None,
        frame: Frame,
    ) -> Optional[float]:
        bbox_h = perceived.bbox.height
        if bbox_h <= 0:
            return None

        real_h = self._heights.get(perceived.class_name, self._fallback)
        distance = (real_h * self._focal) / bbox_h

        return float(np.clip(distance, self._CLIP_MIN, self._CLIP_MAX))
