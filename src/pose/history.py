from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict, Optional, Tuple

from src.schemas.pose import AnimalPose, HumanPose, PoseFrame
from src.utils.logger import get_logger


class PoseHistoryManager:
    """Per-track temporal buffer of pose observations.

    Maintains separate deques of ``(timestamp, HumanPose)`` and
    ``(timestamp, AnimalPose)`` entries for each track ID.  Entries
    are automatically evicted when they exceed ``max_age_seconds``
    or when the deque exceeds ``max_length``.

    Thread-safe: all public methods acquire ``threading.Lock``.

    Args:
        max_length: Maximum number of frames to keep per track.
        max_age_seconds: Maximum age (in seconds) before an entry is
            considered stale and eligible for pruning.
    """

    def __init__(
        self,
        max_length: int = 30,
        max_age_seconds: float = 10.0,
    ) -> None:
        self._max_length = max_length
        self._max_age_seconds = max_age_seconds
        self._logger = get_logger(f"{__name__}.PoseHistoryManager")

        self._human_history: Dict[int, deque[Tuple[float, HumanPose]]] = {}
        self._animal_history: Dict[int, deque[Tuple[float, AnimalPose]]] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(self, pose_frame: PoseFrame) -> None:
        """Append all pose observations from a completed frame.

        Poses whose ``track_id`` is ``None`` are silently ignored
        (they cannot be tracked across frames).

        Args:
            pose_frame: The unified pose result for the current frame.
        """
        with self._lock:
            timestamp = pose_frame.timestamp

            for human in pose_frame.humans:
                tid = human.track_id
                if tid is None:
                    continue
                if tid not in self._human_history:
                    self._human_history[tid] = deque(maxlen=self._max_length)
                self._human_history[tid].append((timestamp, human))

            for animal in pose_frame.animals:
                tid = animal.track_id
                if tid is None:
                    continue
                if tid not in self._animal_history:
                    self._animal_history[tid] = deque(maxlen=self._max_length)
                self._animal_history[tid].append((timestamp, animal))

    def get_human_history(self, track_id: int) -> Tuple[HumanPose, ...]:
        """Return all stored HumanPose observations for a track, oldest first.

        Args:
            track_id: Persistent track identifier.

        Returns:
            An ordered tuple of ``HumanPose`` instances (may be empty).
        """
        with self._lock:
            deq = self._human_history.get(track_id)
            if deq is None:
                return ()
            return tuple(pose for _, pose in deq)

    def get_animal_history(self, track_id: int) -> Tuple[AnimalPose, ...]:
        """Return all stored AnimalPose observations for a track, oldest first.

        Args:
            track_id: Persistent track identifier.

        Returns:
            An ordered tuple of ``AnimalPose`` instances (may be empty).
        """
        with self._lock:
            deq = self._animal_history.get(track_id)
            if deq is None:
                return ()
            return tuple(pose for _, pose in deq)

    def active_track_ids(self) -> Tuple[int, ...]:
        """Return all track IDs that have at least one stored pose."""
        with self._lock:
            ids: set[int] = set(self._human_history.keys())
            ids.update(self._animal_history.keys())
            return tuple(sorted(ids))

    def prune(self, current_timestamp: float) -> int:
        """Remove entries older than ``current_timestamp - max_age_seconds``.

        Tracks whose deque becomes empty after pruning are removed from
        the internal dictionaries to prevent unbounded memory growth.

        Args:
            current_timestamp: Reference timestamp in seconds.

        Returns:
            The number of individual entries pruned.
        """
        with self._lock:
            count = 0
            cutoff = current_timestamp - self._max_age_seconds

            for history_dict in (self._human_history, self._animal_history):
                stale_tracks: list[int] = []
                for tid, deq in history_dict.items():
                    while deq and deq[0][0] < cutoff:
                        deq.popleft()
                        count += 1
                    if not deq:
                        stale_tracks.append(tid)
                for tid in stale_tracks:
                    del history_dict[tid]

            if count > 0:
                self._logger.debug(
                    "Pruned %d stale pose entries  |  cutoff=%.2f",
                    count,
                    cutoff,
                )

            return count
