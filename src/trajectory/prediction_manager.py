from __future__ import annotations

import math
import uuid
from collections import deque
from typing import Any

from src.schemas.trajectory import (
    EvaluationStatus,
    PredictionRecord,
    TrajectoryPoint,
    TrajectoryWaypointRecord,
)
from src.trajectory.history_manager import TrajectoryHistoryManager
from src.trajectory.metrics import calculate_trajectory_metrics
from src.utils.logger import get_logger


class PredictionManager:
    """Manages active prediction records and future-position evaluation matching.

    Every emitted trajectory forecast creates a ``PredictionRecord`` with status
    ``PENDING`` and detailed ``TrajectoryWaypointRecord``s. As incoming video
    frames advance in time, this manager:
    1. Evaluates intermediate waypoints against actual observations to calculate ADE.
    2. Evaluates the final endpoint at the target horizon to calculate FDE and endpoint error.
    3. Resolves statuses to MATCHED, TRACK_LOST, or MISSING.
    4. Buffers recently matched records for live visual error overlays.
    """

    def __init__(
        self,
        match_tolerance_seconds: float = 0.05,
        max_records: int = 10000,
        success_threshold_pixels: float = 20.0,
    ) -> None:
        self.match_tolerance_seconds = max(0.001, float(match_tolerance_seconds))
        self.max_records = max(100, int(max_records))
        self.success_threshold_pixels = max(0.1, float(success_threshold_pixels))
        self._logger = get_logger(f"{__name__}.PredictionManager")

        # Map prediction_id -> PredictionRecord
        self._records: dict[str, PredictionRecord] = {}
        # List of pending IDs to inspect efficiently
        self._pending_ids: list[str] = []
        # Ring buffer of recently resolved records for visualization
        self._recently_resolved: deque[tuple[float, PredictionRecord]] = deque(maxlen=200)

    # ------------------------------------------------------------------ #
    #  Record Registration
    # ------------------------------------------------------------------ #

    def add_prediction(
        self,
        track_id: int,
        class_name: str,
        prediction_timestamp: float,
        horizon_seconds: float,
        current_position: tuple[float, float],
        predicted_position: tuple[float, float],
        predictor: str = "constant_acceleration_kalman",
        uncertainty: dict[str, Any] | None = None,
        track_instance_id: str = "",
        waypoints: list[TrajectoryPoint] | None = None,
        following_track_id: str | int | None = None,
        leading_track_id: str | int | None = None,
        following_speed_mps: float | None = None,
        leading_speed_mps: float | None = None,
        inter_vehicle_gap_m: float | None = None,
        drac_mps2: float | None = None,
        drac_status: str = "UNAVAILABLE",
    ) -> PredictionRecord:
        """Register a new prediction record for future ground truth matching."""
        target_timestamp = prediction_timestamp + horizon_seconds
        rec_id = str(uuid.uuid4())
        inst_id = track_instance_id or f"{track_id}_inst1"

        # Create detailed waypoint records for all intermediate forecast points
        wp_records: list[TrajectoryWaypointRecord] = []
        if waypoints:
            for wp in waypoints:
                wp_records.append(
                    TrajectoryWaypointRecord(
                        future_timestamp=wp.future_timestamp,
                        time_offset=wp.time_offset,
                        predicted_position=wp.position,
                        variance=wp.variance,
                        status=EvaluationStatus.PENDING,
                    )
                )

        record = PredictionRecord(
            prediction_id=rec_id,
            track_id=track_id,
            track_instance_id=inst_id,
            class_name=class_name,
            prediction_timestamp=prediction_timestamp,
            current_position=current_position,
            horizon_seconds=horizon_seconds,
            target_timestamp=target_timestamp,
            predicted_position=predicted_position,
            predictor=predictor,
            uncertainty=uncertainty,
            status=EvaluationStatus.PENDING,
            waypoint_records=wp_records,
            following_track_id=following_track_id,
            leading_track_id=leading_track_id,
            following_speed_mps=following_speed_mps,
            leading_speed_mps=leading_speed_mps,
            inter_vehicle_gap_m=inter_vehicle_gap_m,
            drac_mps2=drac_mps2,
            drac_status=drac_status,
        )

        self._records[rec_id] = record
        self._pending_ids.append(rec_id)

        # Evict oldest resolved records if exceeding buffer capacity
        if len(self._records) > self.max_records:
            self._prune_old_records()

        return record

    def _prune_old_records(self) -> None:
        """Prune oldest completed records to prevent unbounded memory growth."""
        excess = len(self._records) - self.max_records
        resolved_keys = [
            k
            for k, rec in self._records.items()
            if rec.status != EvaluationStatus.PENDING
        ]
        for k in resolved_keys[:excess]:
            self._records.pop(k, None)

    # ------------------------------------------------------------------ #
    #  Evaluation Matching
    # ------------------------------------------------------------------ #

    def evaluate_pending(
        self,
        history_mgr: TrajectoryHistoryManager,
        current_timestamp: float,
    ) -> list[PredictionRecord]:
        """Evaluate pending prediction records whose target time has arrived.

        Args:
            history_mgr: TrajectoryHistoryManager holding actual observations.
            current_timestamp: Current video frame timestamp.

        Returns:
            List of records that transitioned out of PENDING in this step.
        """
        remaining_pending: list[str] = []
        resolved_records: list[PredictionRecord] = []

        for rec_id in self._pending_ids:
            record = self._records.get(rec_id)
            if record is None or record.status != EvaluationStatus.PENDING:
                continue

            # Step A: Evaluate intermediate waypoints whose future timestamp has arrived
            for wp in record.waypoint_records:
                if wp.status == EvaluationStatus.PENDING and current_timestamp >= (wp.future_timestamp - self.match_tolerance_seconds):
                    wp_match = history_mgr.get_position_at_timestamp(
                        track_id=record.track_id,
                        target_timestamp=wp.future_timestamp,
                        match_tolerance_seconds=self.match_tolerance_seconds,
                        track_instance_id=record.track_instance_id,
                    )
                    if wp_match is not None:
                        wp_x, wp_y, wp_t = wp_match
                        wp.actual_position = (wp_x, wp_y)
                        wp.actual_timestamp = wp_t
                        wp.displacement_error = math.hypot(
                            wp.predicted_position[0] - wp_x,
                            wp.predicted_position[1] - wp_y,
                        )
                        wp.status = EvaluationStatus.MATCHED
                    elif current_timestamp > (wp.future_timestamp + self.match_tolerance_seconds):
                        wp.status = EvaluationStatus.MISSING

            target_t = record.target_timestamp

            # Step B: Check if terminal horizon target timestamp has been reached
            if current_timestamp < (target_t - self.match_tolerance_seconds):
                remaining_pending.append(rec_id)
                continue

            # Terminal matching window
            match_res = history_mgr.get_position_at_timestamp(
                track_id=record.track_id,
                target_timestamp=target_t,
                match_tolerance_seconds=self.match_tolerance_seconds,
                track_instance_id=record.track_instance_id,
            )

            if match_res is not None:
                act_x, act_y, act_t = match_res
                err_x = record.predicted_position[0] - act_x
                err_y = record.predicted_position[1] - act_y
                err_dist = math.hypot(err_x, err_y)

                record.actual_position = (act_x, act_y)
                record.actual_timestamp = act_t
                record.error_components = (err_x, err_y)
                record.error_distance = err_dist
                record.endpoint_error = err_dist
                record.fde = err_dist
                record.status = EvaluationStatus.MATCHED

                # Compute ADE across matched waypoints
                matched_wps = [
                    w.displacement_error
                    for w in record.waypoint_records
                    if w.status == EvaluationStatus.MATCHED and w.displacement_error is not None
                ]
                if matched_wps:
                    record.ade = float(sum(matched_wps) / len(matched_wps))
                else:
                    record.ade = err_dist

                resolved_records.append(record)
                self._recently_resolved.append((current_timestamp, record))
            else:
                # No match yet: check if target window has strictly expired
                if current_timestamp > (target_t + self.match_tolerance_seconds):
                    last_seen = history_mgr.get_last_observed_time(record.track_id)
                    current_inst = history_mgr.get_track_instance_id(record.track_id)

                    if current_inst != record.track_instance_id:
                        # Track ID was reset or reused for a new object episode
                        record.status = EvaluationStatus.INVALID
                    elif last_seen is None or last_seen < (target_t - self.match_tolerance_seconds):
                        record.status = EvaluationStatus.TRACK_LOST
                    else:
                        record.status = EvaluationStatus.MISSING

                    resolved_records.append(record)
                    self._recently_resolved.append((current_timestamp, record))
                else:
                    # Still within tolerance window, keep pending for possible arrival
                    remaining_pending.append(rec_id)

        self._pending_ids = remaining_pending
        return resolved_records

    # ------------------------------------------------------------------ #
    #  Live Visual Overlay Query
    # ------------------------------------------------------------------ #

    def get_active_evaluations(
        self,
        current_time: float,
        max_age_seconds: float = 1.5,
    ) -> list[PredictionRecord]:
        """Return recently matched records within max_age_seconds for visual display."""
        cutoff = current_time - max_age_seconds
        return [
            rec
            for resolved_t, rec in self._recently_resolved
            if resolved_t >= cutoff and rec.status == EvaluationStatus.MATCHED
        ]

    # ------------------------------------------------------------------ #
    #  Query API
    # ------------------------------------------------------------------ #

    def get_records(
        self,
        status: EvaluationStatus | None = None,
        track_id: int | None = None,
    ) -> list[PredictionRecord]:
        """Query stored prediction records."""
        records = list(self._records.values())
        if status is not None:
            records = [r for r in records if r.status == status]
        if track_id is not None:
            records = [r for r in records if r.track_id == track_id]
        return records

    def get_summary_statistics(self) -> dict[str, Any]:
        """Compute aggregate evaluation summary metrics."""
        return calculate_trajectory_metrics(
            list(self._records.values()),
            success_threshold_pixels=self.success_threshold_pixels,
        )
