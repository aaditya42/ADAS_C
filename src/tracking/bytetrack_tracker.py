from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.base import BaseTracker
from src.schemas import BoundingBox, Detection, Frame, TrackedObject
from src.utils.logger import get_logger

if TYPE_CHECKING:
    import supervision as sv


class ByteTrackTracker(BaseTracker):
    """Multi-object tracker wrapping supervision's ByteTrack.

    Converts the pipeline's ``Detection`` schemas into ``sv.Detections``,
    runs ``sv.ByteTrack``, and maps results back to ``TrackedObject``.

    Args:
        track_activation_threshold: Confidence threshold to start a track.
        lost_track_buffer: Frames to keep a lost track alive.
        minimum_matching_threshold: IoU threshold for matching.
        frame_rate: Expected video frame rate.
        minimum_consecutive_frames: Min frames a track must be seen to be
            returned.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self._track_activation_threshold = track_activation_threshold
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        self._frame_rate = frame_rate
        self._minimum_consecutive_frames = minimum_consecutive_frames
        self._logger = get_logger(f"{__name__}.ByteTrackTracker")

        self._bt: sv.ByteTrack | None = None
        self._class_names: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    #  BaseTracker
    # ------------------------------------------------------------------ #

    def update(
        self, detections: list[Detection], frame: Frame
    ) -> list[TrackedObject]:
        if self._bt is None:
            self._init_tracker()

        for det in detections:
            self._class_names[det.class_id] = det.class_name

        sv_dets = self._to_sv(detections)
        sv_tracked = self._bt.update_with_detections(sv_dets)
        return self._from_sv(sv_tracked)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _init_tracker(self) -> None:
        import supervision as sv

        self._bt = sv.ByteTrack(
            track_activation_threshold=self._track_activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=self._frame_rate,
            minimum_consecutive_frames=self._minimum_consecutive_frames,
        )
        self._logger.info(
            "ByteTrack initialised  |  activation_thresh=%.2f  "
            "lost_buffer=%d  match_thresh=%.2f  fps=%d",
            self._track_activation_threshold,
            self._lost_track_buffer,
            self._minimum_matching_threshold,
            self._frame_rate,
        )

    @staticmethod
    def _to_sv(detections: list[Detection]) -> sv.Detections:
        import numpy as np
        import supervision as sv

        n = len(detections)
        if n == 0:
            return sv.Detections.empty()

        xyxy = np.zeros((n, 4), dtype=np.float32)
        confidence = np.zeros((n,), dtype=np.float32)
        class_id = np.zeros((n,), dtype=int)

        for i, det in enumerate(detections):
            xyxy[i] = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            confidence[i] = det.confidence
            class_id[i] = det.class_id

        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def _from_sv(self, sv_dets: sv.Detections) -> list[TrackedObject]:
        tracked: list[TrackedObject] = []
        if len(sv_dets) == 0:
            return tracked

        for i in range(len(sv_dets)):
            tracker_id = sv_dets.tracker_id[i]
            if tracker_id is None or tracker_id < 0:
                continue

            x1, y1, x2, y2 = sv_dets.xyxy[i]
            class_id = int(sv_dets.class_id[i])
            conf = float(sv_dets.confidence[i])
            class_name = self._class_names.get(class_id, str(class_id))

            tracked.append(
                TrackedObject(
                    track_id=int(tracker_id),
                    class_name=class_name,
                    confidence=conf,
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )

        return tracked
