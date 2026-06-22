from __future__ import annotations

from typing import Optional

import numpy as np

from src.distance.base import BaseDistanceEstimator
from src.schemas import Frame, PerceivedObject


class RelativeDistanceEstimator(BaseDistanceEstimator):
    """Returns the raw depth value from the fusion layer unchanged.

    **WARNING:** The returned value is in *relative* depth units, not meters.
    This estimator is intended for development / debugging until a calibrated
    estimator is deployed.
    """

    def estimate(
        self,
        perceived: PerceivedObject,
        depth_map: np.ndarray | None,
        frame: Frame,
    ) -> Optional[float]:
        if perceived.depth_confidence > 0:
            return perceived.depth
        return None
