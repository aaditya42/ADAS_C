from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTTCEstimator(ABC):
    """Abstract time-to-collision estimator.

    Contract:
      - estimate() accepts the current perceived objects and their motion
        estimates, and returns a dict mapping track_id to TTC in seconds.
      - ``None`` means insufficient data (unknown).
      - ``float('inf')`` means no collision risk (receding / stationary).
      - Finite values are in seconds, always > 0.
    """

    @abstractmethod
    def estimate(
        self,
        perceived: list[Any],
        motion: dict[int, Any],
    ) -> dict[int, float | None]:
        ...
