from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLaneDetector(ABC):
    """Abstract lane detection module.

    Contract:
      - detect() takes a Frame and returns a LaneInfo object.
      - If no lanes are visible, returned LaneInfo has None for all lanes
        and low confidence.
    """

    @abstractmethod
    def detect(self, frame: Any) -> Any:  # Frame -> LaneInfo
        ...
