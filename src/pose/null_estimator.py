from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.base import BaseAnimalPoseEstimator, BaseHumanPoseEstimator
from src.schemas import Detection, TrackedObject
from src.schemas.pose import AnimalPose, HumanPose
from src.utils.logger import get_logger


class NullHumanPoseEstimator(BaseHumanPoseEstimator):
    """Fallback human pose estimator that returns an empty result.

    Used when pose estimation is disabled in the configuration. All
    methods are no-ops — no model is loaded and no computation is
    performed.
    """

    def __init__(self) -> None:
        self._logger = get_logger(f"{__name__}.NullHumanPoseEstimator")

    def estimate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
    ) -> Sequence[HumanPose]:
        self._logger.debug("NullHumanPoseEstimator.estimate called — returning empty result")
        return ()


class NullAnimalPoseEstimator(BaseAnimalPoseEstimator):
    """Fallback animal pose estimator that returns an empty result.

    Used when pose estimation is disabled in the configuration. All
    methods are no-ops — no model is loaded and no computation is
    performed.
    """

    def __init__(self) -> None:
        self._logger = get_logger(f"{__name__}.NullAnimalPoseEstimator")

    def estimate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
    ) -> Sequence[AnimalPose]:
        self._logger.debug("NullAnimalPoseEstimator.estimate called — returning empty result")
        return ()
