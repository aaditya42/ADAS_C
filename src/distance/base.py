from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.schemas import Frame, PerceivedObject


class BaseDistanceEstimator(ABC):
    """Abstract distance estimator that converts raw depth / geometry into
    metric distance in meters.

    Contract:
      - ``estimate()`` accepts a PerceivedObject, optional depth map, and
        the current Frame, and returns a float (meters) or None if distance
        cannot be determined.
      - Implementations MUST NOT mutate any of the inputs.
    """

    @abstractmethod
    def estimate(
        self,
        perceived: PerceivedObject,
        depth_map: np.ndarray | None,
        frame: Frame,
    ) -> Optional[float]:
        ...
