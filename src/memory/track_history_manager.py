from __future__ import annotations

from collections import deque
from typing import Optional

from src.schemas import BoundingBox, Frame, PerceivedObject


class TrackSnapshot:
    """A single observation of a track at a given frame.

    Attributes:
        frame_id: Monotonic frame counter.
        timestamp: Unix timestamp of the observation.
        bbox: Bounding box in pixel coordinates.
        distance: Estimated distance in meters (or relative units), or None.
        depth_confidence: Confidence of the depth estimate in [0, 1].
    """

    __slots__ = ("frame_id", "timestamp", "bbox", "distance", "depth_confidence")

    def __init__(
        self,
        frame_id: int,
        timestamp: float,
        bbox: BoundingBox,
        distance: float | None,
        depth_confidence: float,
    ) -> None:
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.bbox = bbox
        self.distance = distance
        self.depth_confidence = depth_confidence

    def __repr__(self) -> str:
        return (
            f"TrackSnapshot(fid={self.frame_id}, "
            f"dist={self.distance}, conf={self.depth_confidence:.2f})"
        )


class TrackHistoryManager:
    """Maintains per-track observation histories.

    Each track gets a fixed-length deque of ``TrackSnapshot`` objects.  Tracks
    that are not seen for ``stale_after_frames`` consecutive frames are
    automatically garbage-collected.

    Args:
        max_history_per_track: Maximum number of frames to keep per track.
        stale_after_frames: Frames without an update before a track is evicted.
    """

    def __init__(
        self,
        max_history_per_track: int = 60,
        stale_after_frames: int = 60,
    ) -> None:
        self._maxlen = max_history_per_track
        self._stale = stale_after_frames

        self._history: dict[int, deque[TrackSnapshot]] = {}
        self._last_seen: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(self, perceived: list[PerceivedObject], frame: Frame) -> None:
        """Append a snapshot for each perceived object and GC stale tracks.

        Args:
            perceived: Perceived objects from the fusion layer.
            frame: The current frame (used for frame_id and timestamp).
        """
        seen_this_frame: set[int] = set()

        for obj in perceived:
            tid = obj.track_id
            seen_this_frame.add(tid)

            if tid not in self._history:
                self._history[tid] = deque(maxlen=self._maxlen)
                self._last_seen[tid] = -1

            self._history[tid].append(
                TrackSnapshot(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    bbox=obj.bbox,
                    distance=obj.distance if obj.depth_confidence > 0 else None,
                    depth_confidence=obj.depth_confidence,
                )
            )
            self._last_seen[tid] = frame.frame_id

        # Update last_seen for tracks not seen this frame (keep their
        # last_seen as-is so GC can find them later).
        # GC: remove tracks not seen for stale_after_frames
        stale_ids = [
            tid
            for tid, last in self._last_seen.items()
            if tid not in seen_this_frame
            and (frame.frame_id - last) > self._stale
        ]
        for tid in stale_ids:
            del self._history[tid]
            del self._last_seen[tid]

    def get_history(self, track_id: int) -> list[TrackSnapshot]:
        """Return all snapshots for a track, most recent last."""
        deq = self._history.get(track_id)
        return list(deq) if deq is not None else []

    def get_recent(self, track_id: int, n: int) -> list[TrackSnapshot]:
        """Return the last *n* snapshots for a track (most recent last)."""
        deq = self._history.get(track_id)
        if deq is None:
            return []
        n = min(n, len(deq))
        return list(deq)[-n:]

    def active_tracks(self) -> set[int]:
        """Return the set of track IDs currently being tracked."""
        return set(self._history.keys())

    def __len__(self) -> int:
        return len(self._history)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._history
