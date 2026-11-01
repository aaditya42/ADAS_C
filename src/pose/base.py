from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from src.schemas import Detection, TrackedObject
from src.schemas.pose import AnimalPose, HumanPose


class BaseHumanPoseEstimator(ABC):
    """Abstract human pose estimator.

    Contract:
      - estimate() accepts a raw image, detection list, and tracked-object list,
        and returns a sequence of HumanPose instances.
      - The returned sequence may be empty if no human poses are detected.
      - Implementations MUST NOT mutate the input arrays or lists.
      - Heavy model loading MUST be deferred to the first estimate() call.
    """

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
    ) -> Sequence[HumanPose]:
        """Run human pose estimation on a single frame.

        Args:
            frame: BGR image array of shape (H, W, 3), dtype uint8.
            detections: Detected objects in the current frame.
            tracks: Tracked objects with persistent IDs.

        Returns:
            A sequence of HumanPose instances (may be empty).
        """
        ...


class BaseAnimalPoseEstimator(ABC):
    """Abstract animal pose estimator.

    Contract:
      - estimate() accepts a raw image, detection list, and tracked-object list,
        and returns a sequence of AnimalPose instances.
      - The returned sequence may be empty if no animal poses are detected.
      - Implementations MUST NOT mutate the input arrays or lists.
      - Heavy model loading MUST be deferred to the first estimate() call.
    """

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
    ) -> Sequence[AnimalPose]:
        """Run animal pose estimation on a single frame.

        Args:
            frame: BGR image array of shape (H, W, 3), dtype uint8.
            detections: Detected objects in the current frame.
            tracks: Tracked objects with persistent IDs.

        Returns:
            A sequence of AnimalPose instances (may be empty).
        """
        ...
