from __future__ import annotations

import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from src.pose.base import BaseAnimalPoseEstimator, BaseHumanPoseEstimator
from src.pose.factory import PoseEstimatorBundle, build_pose_estimators
from src.pose.history import PoseHistoryManager
from src.schemas import Detection, TrackedObject
from src.schemas.pose import AnimalPose, HumanPose, PoseFrame
from src.utils.logger import get_logger


class PoseSubsystem:
    """Unified pose estimation subsystem for the ADAS pipeline.

    Owns the human and animal pose estimators plus the pose configuration.
    Provides a single ``process()`` method that the pipeline calls once per
    frame to produce a unified ``PoseFrame`` and optionally maintains a
    per-track pose history via :class:`PoseHistoryManager`.

    Errors inside an individual estimator are caught, logged at ERROR, and
    result in an empty pose list for that entity type — they never crash
    the pipeline.

    Args:
        config: Full pipeline configuration dictionary (the ``pose``
            section is extracted internally).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = get_logger(f"{__name__}.PoseSubsystem")

        bundle: PoseEstimatorBundle = build_pose_estimators(config)
        self._human_estimator: BaseHumanPoseEstimator = bundle.human
        self._animal_estimator: BaseAnimalPoseEstimator = bundle.animal

        # Optional per-track pose history
        pose_cfg = config.get("pose", {})
        history_cfg = pose_cfg.get("history", {})
        if history_cfg.get("enabled", True):
            self._history: Optional[PoseHistoryManager] = PoseHistoryManager(
                max_length=int(history_cfg.get("max_length", 30)),
                max_age_seconds=float(history_cfg.get("max_age_seconds", 10.0)),
            )
        else:
            self._history = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> Optional[PoseHistoryManager]:
        """The per-track pose history manager, or ``None`` if disabled."""
        return self._history

    def process(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
        frame_id: int,
        timestamp: float,
    ) -> PoseFrame:
        """Run pose estimation on a single frame.

        Args:
            frame: BGR image array of shape ``(H, W, 3)``, dtype uint8.
            detections: All detections produced by the upstream detector.
            tracks: All tracked objects from the upstream tracker.
            frame_id: Monotonic frame index.
            timestamp: Frame timestamp in seconds.

        Returns:
            A unified ``PoseFrame`` containing human and animal pose
            results.  Both lists may be empty.
        """
        humans: Sequence[HumanPose]
        animals: Sequence[AnimalPose]

        # --- Human pose -------------------------------------------------
        t0 = time.perf_counter()
        try:
            humans = self._human_estimator.estimate(frame, detections, tracks)
        except Exception:
            self._logger.exception(
                "Human pose estimator failed for frame %d; returning empty list",
                frame_id,
            )
            humans = ()
        elapsed = time.perf_counter() - t0
        self._logger.debug(
            "Human pose  |  frame=%d  count=%d  time=%.1f ms",
            frame_id,
            len(humans),
            elapsed * 1000,
        )

        # --- Animal pose ------------------------------------------------
        t0 = time.perf_counter()
        try:
            animals = self._animal_estimator.estimate(frame, detections, tracks)
        except Exception:
            self._logger.exception(
                "Animal pose estimator failed for frame %d; returning empty list",
                frame_id,
            )
            animals = ()
        elapsed = time.perf_counter() - t0
        self._logger.debug(
            "Animal pose  |  frame=%d  count=%d  time=%.1f ms",
            frame_id,
            len(animals),
            elapsed * 1000,
        )

        pose_frame = PoseFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            humans=tuple(humans),
            animals=tuple(animals),
        )

        # Feed into history manager (if enabled)
        if self._history is not None:
            self._history.update(pose_frame)
            self._history.prune(timestamp)

        return pose_frame
