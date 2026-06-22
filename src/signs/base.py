from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSignDetector(ABC):
    """Abstract traffic sign detector.

    Contract:
      - detect() takes a Frame and returns a list of TrafficSignInfo objects.
      - The list may be empty if no signs are visible.
      - Implementations MUST NOT mutate the input Frame.
    """

    @abstractmethod
    def detect(self, frame: Any) -> Any:  # Frame -> list[TrafficSignInfo]
        ...
