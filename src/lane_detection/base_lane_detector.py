from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.lane_detection.interfaces import LaneDetectionResult


class BaseLaneDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        ...
