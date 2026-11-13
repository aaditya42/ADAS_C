from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from src.schemas import BoundingBox, Frame, PerceivedObject, TrackedObject
from src.trajectory.kalman_filter import (
    ConstantAccelerationKalmanFilter,
    ConstantVelocityKalmanFilter,
)
from src.utils.logger import get_logger


@dataclass(frozen=True)
class TrajectorySnapshot:
    """A single spatial-temporal observation of a track.

    Attributes:
        timestamp: Timestamp of the observation in seconds.
        frame_id: Frame counter.
        position: Reference coordinate (x, y) in image space.
        bbox: Bounding box of the object.
        velocity: Estimated velocity (vx, vy) at this instant.
        acceleration: Estimated acceleration (ax, ay) at this instant.
        track_instance_id: Unique continuous track episode identity.
    """

    timestamp: float
    frame_id: int
    position: tuple[float, float]
    bbox: BoundingBox
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    track_instance_id: str = ""


class TrajectoryHistoryManager:
    """Maintains time-indexed trajectory histories and Kalman filters per track.

    Responsibilities:
    - Extracts reference coordinate from bounding boxes (bottom-center or center).
    - Maintains independent history queues with duration-based pruning.
    - Runs independent CA-KF or CV-KF updates per track with variable dt.
    - Assigns unique continuous ``track_instance_id``s, detecting time gaps
      exceeding ``maximum_track_gap_seconds`` and resetting filters to prevent
      velocity explosion and false matches across track ID reuse.
    - Evicts stale tracks exceeding ``stale_track_timeout_seconds``.
    """

    def __init__(
        self,
        history_duration_seconds: float = 2.0,
        minimum_history_points: int = 10,
        maximum_track_gap_seconds: float = 0.5,
        stale_track_timeout_seconds: float = 1.0,
        position_reference: str = "bottom_center",
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
        model: str = "constant_acceleration",
    ) -> None:
        self.history_duration_seconds = max(0.1, float(history_duration_seconds))
        self.minimum_history_points = max(1, int(minimum_history_points))
        self.maximum_track_gap_seconds = max(0.01, float(maximum_track_gap_seconds))
        self.stale_track_timeout_seconds = max(0.01, float(stale_track_timeout_seconds))
        self.position_reference = position_reference
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.model = model

        self._logger = get_logger(f"{__name__}.TrajectoryHistoryManager")

        # Per-track state
        self._history: dict[int, deque[TrajectorySnapshot]] = {}
        self._filters: dict[
            int, ConstantAccelerationKalmanFilter | ConstantVelocityKalmanFilter
        ] = {}
        self._last_observed_time: dict[int, float] = {}
        self._class_names: dict[int, str] = {}
        self._consecutive_points: dict[int, int] = {}
        self._instance_counters: dict[int, int] = {}
        self._current_instance_id: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    #  Position Extraction
    # ------------------------------------------------------------------ #

    def extract_position(self, bbox: BoundingBox) -> tuple[float, float]:
        """Extract reference (x, y) point from a bounding box."""
        if self.position_reference == "bottom_center":
            x = (bbox.x1 + bbox.x2) / 2.0
            y = bbox.y2
            return (float(x), float(y))
        elif self.position_reference == "center":
            return (float(bbox.center[0]), float(bbox.center[1]))
        else:
            # Fallback to bottom-center
            x = (bbox.x1 + bbox.x2) / 2.0
            y = bbox.y2
            return (float(x), float(y))

    # ------------------------------------------------------------------ #
    #  Observation Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        objects: list[TrackedObject | PerceivedObject],
        frame: Frame,
    ) -> None:
        """Update histories and Kalman filters with latest frame observations."""
        current_time = frame.timestamp
        seen_tracks: set[int] = set()

        for obj in objects:
            tid = obj.track_id
            seen_tracks.add(tid)
            self._class_names[tid] = obj.class_name

            raw_pos = self.extract_position(obj.bbox)

            # Check for existing track & gap detection
            if tid in self._last_observed_time:
                last_time = self._last_observed_time[tid]
                dt = current_time - last_time

                if dt <= 0:
                    # Duplicate or out-of-order timestamp: clamp to small delta
                    dt = 1e-4

                if dt > self.maximum_track_gap_seconds:
                    # Detection gap exceeded threshold: start new track instance & reset filter
                    new_counter = self._instance_counters.get(tid, 1) + 1
                    self._instance_counters[tid] = new_counter
                    self._current_instance_id[tid] = f"{tid}_inst{new_counter}"
                    self._logger.debug(
                        "Track %d gap (%.3fs > %.3fs); incremented to instance %s, resetting filter.",
                        tid,
                        dt,
                        self.maximum_track_gap_seconds,
                        self._current_instance_id[tid],
                    )
                    self._filters[tid] = self._create_filter(raw_pos)
                    self._consecutive_points[tid] = 1
                else:
                    # Continuous update
                    kf = self._filters[tid]
                    kf.predict(dt)
                    kf.update(raw_pos)
                    self._consecutive_points[tid] = self._consecutive_points.get(tid, 0) + 1
            else:
                # New track initialization
                self._history[tid] = deque()
                self._instance_counters[tid] = 1
                self._current_instance_id[tid] = f"{tid}_inst1"
                self._filters[tid] = self._create_filter(raw_pos)
                self._consecutive_points[tid] = 1

            self._last_observed_time[tid] = current_time
            kf = self._filters[tid]
            instance_id = self._current_instance_id[tid]

            # Append snapshot with smoothed state
            snapshot = TrajectorySnapshot(
                timestamp=current_time,
                frame_id=frame.frame_id,
                position=raw_pos,
                bbox=obj.bbox,
                velocity=kf.velocity,
                acceleration=kf.acceleration,
                track_instance_id=instance_id,
            )
            self._history[tid].append(snapshot)

            # Prune snapshots older than history_duration_seconds
            cutoff_time = current_time - self.history_duration_seconds
            deq = self._history[tid]
            while deq and deq[0].timestamp < cutoff_time:
                deq.popleft()

        # Evict stale tracks
        self._prune_stale_tracks(current_time)

    def _prune_stale_tracks(self, current_time: float) -> None:
        """Evict tracks not updated within stale_track_timeout_seconds."""
        stale_ids = [
            tid
            for tid, last_t in self._last_observed_time.items()
            if (current_time - last_t) > self.stale_track_timeout_seconds
        ]
        for tid in stale_ids:
            self._history.pop(tid, None)
            self._filters.pop(tid, None)
            self._last_observed_time.pop(tid, None)
            self._class_names.pop(tid, None)
            self._consecutive_points.pop(tid, None)
            self._current_instance_id.pop(tid, None)

    def _create_filter(
        self, initial_position: tuple[float, float]
    ) -> ConstantAccelerationKalmanFilter | ConstantVelocityKalmanFilter:
        """Instantiate either CA-KF or CV-KF depending on configured model."""
        if self.model in ("constant_velocity", "cv", "linear"):
            return ConstantVelocityKalmanFilter(
                initial_position=initial_position,
                process_noise_std=self.process_noise_std,
                measurement_noise_std=self.measurement_noise_std,
            )
        return ConstantAccelerationKalmanFilter(
            initial_position=initial_position,
            process_noise_std=self.process_noise_std,
            measurement_noise_std=self.measurement_noise_std,
        )

    # ------------------------------------------------------------------ #
    #  Query API
    # ------------------------------------------------------------------ #

    def is_track_ready(self, track_id: int) -> bool:
        """Return True if track has accumulated sufficient continuous history."""
        return self._consecutive_points.get(track_id, 0) >= self.minimum_history_points

    def get_track_instance_id(self, track_id: int) -> str:
        """Return the current continuous track instance ID."""
        return self._current_instance_id.get(track_id, f"{track_id}_inst1")

    def get_filter(
        self, track_id: int
    ) -> ConstantAccelerationKalmanFilter | ConstantVelocityKalmanFilter | None:
        """Get the Kalman filter instance for a given track."""
        return self._filters.get(track_id)

    def get_history(self, track_id: int) -> list[TrajectorySnapshot]:
        """Return the observation snapshots for a track (oldest to newest)."""
        deq = self._history.get(track_id)
        return list(deq) if deq is not None else []

    def get_class_name(self, track_id: int) -> str:
        """Return the object class label."""
        return self._class_names.get(track_id, "unknown")

    def get_last_observed_time(self, track_id: int) -> float | None:
        """Return timestamp of most recent observation."""
        return self._last_observed_time.get(track_id)

    def active_track_ids(self) -> set[int]:
        """Return set of currently active track IDs."""
        return set(self._history.keys())

    # ------------------------------------------------------------------ #
    #  Interpolation / Evaluation Matching Helper
    # ------------------------------------------------------------------ #

    def get_position_at_timestamp(
        self,
        track_id: int,
        target_timestamp: float,
        match_tolerance_seconds: float = 0.05,
        track_instance_id: str | None = None,
    ) -> tuple[float, float, float] | None:
        """Retrieve actual observed position at target_timestamp.

        Interpolates between adjacent continuous snapshots if the target
        timestamp falls between two observations without a track gap.
        Otherwise returns closest observation if within tolerance.
        If track_instance_id is provided, only matches within that exact continuous
        track instance, avoiding identity switches across track gap/reuse.

        Returns:
            (actual_x, actual_y, actual_timestamp) or None if unavailable/gap.
        """
        snapshots = self.get_history(track_id)
        if not snapshots:
            return None

        # Filter by track_instance_id if specified
        if track_instance_id:
            snapshots = [s for s in snapshots if s.track_instance_id == track_instance_id]
            if not snapshots:
                return None

        # Check bounds
        first_t = snapshots[0].timestamp
        last_t = snapshots[-1].timestamp

        # Case 1: Exact or nearest within tolerance before/after bounds
        if target_timestamp < first_t:
            if (first_t - target_timestamp) <= match_tolerance_seconds:
                return (snapshots[0].position[0], snapshots[0].position[1], first_t)
            return None

        if target_timestamp > last_t:
            if (target_timestamp - last_t) <= match_tolerance_seconds:
                return (snapshots[-1].position[0], snapshots[-1].position[1], last_t)
            return None

        # Case 2: Within history range, find bounding snapshots
        for i in range(len(snapshots) - 1):
            s1 = snapshots[i]
            s2 = snapshots[i + 1]

            if s1.timestamp <= target_timestamp <= s2.timestamp:
                dt = s2.timestamp - s1.timestamp
                if dt > self.maximum_track_gap_seconds:
                    # Gap detected in observation range: cannot interpolate across gap
                    return None

                if dt == 0:
                    return (s1.position[0], s1.position[1], s1.timestamp)

                # Linear interpolation
                alpha = (target_timestamp - s1.timestamp) / dt
                interp_x = (1.0 - alpha) * s1.position[0] + alpha * s2.position[0]
                interp_y = (1.0 - alpha) * s1.position[1] + alpha * s2.position[1]
                return (interp_x, interp_y, target_timestamp)

        # Fallback closest snapshot
        closest = min(snapshots, key=lambda s: abs(s.timestamp - target_timestamp))
        if abs(closest.timestamp - target_timestamp) <= match_tolerance_seconds:
            return (closest.position[0], closest.position[1], closest.timestamp)

        return None
