from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvaluationStatus(str, Enum):
    """Evaluation matching status for trajectory prediction records."""

    PENDING = "PENDING"
    MATCHED = "MATCHED"
    MISSING = "MISSING"
    INVALID = "INVALID"
    TRACK_LOST = "TRACK_LOST"


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single predicted waypoint in a future trajectory.

    Attributes:
        future_timestamp: Absolute target timestamp in seconds.
        time_offset: Forward time delta in seconds from prediction instant.
        position: Predicted (x, y) coordinates.
        variance: Optional estimated position variance (var_x, var_y).
    """

    future_timestamp: float
    time_offset: float
    position: tuple[float, float]
    variance: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert waypoint to serialization dictionary."""
        data: dict[str, Any] = {
            "future_timestamp": round(self.future_timestamp, 4),
            "time_offset": round(self.time_offset, 4),
            "position": [round(self.position[0], 2), round(self.position[1], 2)],
        }
        if self.variance is not None:
            data["variance"] = [round(self.variance[0], 4), round(self.variance[1], 4)]
        return data


@dataclass
class TrajectoryWaypointRecord:
    """Individual waypoint evaluation record for detailed future position tracking.

    Attributes:
        future_timestamp: Expected observation timestamp in seconds.
        time_offset: Time delta from prediction creation in seconds.
        predicted_position: Predicted (x, y) coordinates.
        variance: Positional covariance diagonal (var_x, var_y).
        actual_position: Observed actual position at future_timestamp, or None.
        actual_timestamp: Timestamp when actual observation was made.
        displacement_error: Euclidean error between predicted and actual in pixels.
        status: Individual waypoint matching status.
    """

    future_timestamp: float
    time_offset: float
    predicted_position: tuple[float, float]
    variance: tuple[float, float] | None = None
    actual_position: tuple[float, float] | None = None
    actual_timestamp: float | None = None
    displacement_error: float | None = None
    status: EvaluationStatus = EvaluationStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        """Convert waypoint record to dictionary."""
        return {
            "future_timestamp": round(self.future_timestamp, 4),
            "time_offset": round(self.time_offset, 4),
            "predicted_position": [
                round(self.predicted_position[0], 2),
                round(self.predicted_position[1], 2),
            ],
            "variance": (
                [round(self.variance[0], 4), round(self.variance[1], 4)]
                if self.variance is not None
                else None
            ),
            "actual_position": (
                [round(self.actual_position[0], 2), round(self.actual_position[1], 2)]
                if self.actual_position is not None
                else None
            ),
            "actual_timestamp": (
                round(self.actual_timestamp, 4)
                if self.actual_timestamp is not None
                else None
            ),
            "displacement_error": (
                round(self.displacement_error, 3)
                if self.displacement_error is not None
                else None
            ),
            "status": self.status.value,
        }


@dataclass
class TrajectoryPrediction:
    """Complete future trajectory forecast for a tracked object.

    Attributes:
        track_id: Persistent identity of the tracked object.
        class_name: Object classification label.
        prediction_timestamp: Timestamp when the forecast was computed.
        horizon_seconds: Configured future prediction horizon.
        current_position: Observed reference position (x, y) at prediction time.
        estimated_velocity: Filtered velocity (vx, vy) in px/sec.
        estimated_acceleration: Filtered acceleration (ax, ay) in px/sec^2.
        predicted_positions: Sequential future waypoints up to the horizon.
        final_predicted_position: Target position (x, y) at horizon_seconds.
        predictor: Identifier of prediction algorithm.
        uncertainty: Optional metadata or covariance summary.
        track_instance_id: Unique continuous track episode identity.
    """

    track_id: int
    class_name: str
    prediction_timestamp: float
    horizon_seconds: float
    current_position: tuple[float, float]
    estimated_velocity: tuple[float, float]
    estimated_acceleration: tuple[float, float]
    predicted_positions: list[TrajectoryPoint]
    final_predicted_position: tuple[float, float]
    predictor: str = "constant_acceleration_kalman"
    uncertainty: dict[str, Any] | None = None
    track_instance_id: str = ""

    def __post_init__(self) -> None:
        if not self.track_instance_id:
            self.track_instance_id = f"{self.track_id}_inst1"

    def to_dict(self) -> dict[str, Any]:
        """Format matching standard specification."""
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "track_instance_id": self.track_instance_id,
            "class_name": self.class_name,
            "prediction_timestamp": round(self.prediction_timestamp, 4),
            "horizon_seconds": round(self.horizon_seconds, 4),
            "current_position": [
                round(self.current_position[0], 2),
                round(self.current_position[1], 2),
            ],
            "estimated_velocity": [
                round(self.estimated_velocity[0], 3),
                round(self.estimated_velocity[1], 3),
            ],
            "estimated_acceleration": [
                round(self.estimated_acceleration[0], 3),
                round(self.estimated_acceleration[1], 3),
            ],
            "predicted_positions": [p.to_dict() for p in self.predicted_positions],
            "final_predicted_position": [
                round(self.final_predicted_position[0], 2),
                round(self.final_predicted_position[1], 2),
            ],
            "predictor": self.predictor,
        }
        if self.uncertainty is not None:
            data["uncertainty"] = self.uncertainty
        return data


@dataclass
class PredictionRecord:
    """Historical forecast record preserved for evaluation against ground truth.

    Attributes:
        prediction_id: Unique identifier for this prediction instance.
        track_id: ByteTrack ID of the tracked object.
        class_name: Category of the tracked object.
        prediction_timestamp: Time when forecast was emitted.
        current_position: Reference position at prediction time.
        horizon_seconds: Time interval into future.
        target_timestamp: Exact timestamp when prediction should be evaluated.
        predicted_position: Predicted position at target_timestamp.
        predictor: Model name.
        uncertainty: Positional covariance or error ellipse info.
        status: EvaluationStatus enum (PENDING, MATCHED, MISSING, etc.).
        actual_timestamp: Observed timestamp of actual position.
        actual_position: Actual observed position at target time.
        error_distance: Euclidean prediction error in pixels.
        error_components: (dx, dy) error vector.
        track_instance_id: Unique continuous track identity preventing ID reuse issues.
        waypoint_records: Detailed records for every intermediate future waypoint.
        ade: Average Displacement Error across valid matched waypoints.
        fde: Final Displacement Error at horizon_seconds.
        endpoint_error: Endpoint error in pixels (equivalent to error_distance).
    """

    prediction_id: str
    track_id: int
    class_name: str
    prediction_timestamp: float
    current_position: tuple[float, float]
    horizon_seconds: float
    target_timestamp: float
    predicted_position: tuple[float, float]
    predictor: str = "constant_acceleration_kalman"
    uncertainty: dict[str, Any] | None = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    actual_timestamp: float | None = None
    actual_position: tuple[float, float] | None = None
    error_distance: float | None = None
    error_components: tuple[float, float] | None = None
    track_instance_id: str = ""
    waypoint_records: list[TrajectoryWaypointRecord] = field(default_factory=list)
    ade: float | None = None
    fde: float | None = None
    endpoint_error: float | None = None
    following_track_id: str | int | None = None
    leading_track_id: str | int | None = None
    following_speed_mps: float | None = None
    leading_speed_mps: float | None = None
    inter_vehicle_gap_m: float | None = None
    drac_mps2: float | None = None
    drac_status: str = "UNAVAILABLE"

    def __post_init__(self) -> None:
        if not self.track_instance_id:
            self.track_instance_id = f"{self.track_id}_inst1"
        if self.endpoint_error is None and self.error_distance is not None:
            self.endpoint_error = self.error_distance
        elif self.error_distance is None and self.endpoint_error is not None:
            self.error_distance = self.endpoint_error
        if self.fde is None and self.endpoint_error is not None:
            self.fde = self.endpoint_error

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for logging or serialization."""
        return {
            "prediction_id": self.prediction_id,
            "track_id": self.track_id,
            "track_instance_id": self.track_instance_id,
            "class_name": self.class_name,
            "prediction_timestamp": round(self.prediction_timestamp, 4),
            "current_position": [
                round(self.current_position[0], 2),
                round(self.current_position[1], 2),
            ],
            "horizon_seconds": round(self.horizon_seconds, 4),
            "target_timestamp": round(self.target_timestamp, 4),
            "predicted_position": [
                round(self.predicted_position[0], 2),
                round(self.predicted_position[1], 2),
            ],
            "predictor": self.predictor,
            "uncertainty": self.uncertainty,
            "status": self.status.value,
            "actual_timestamp": (
                round(self.actual_timestamp, 4)
                if self.actual_timestamp is not None
                else None
            ),
            "actual_position": (
                [round(self.actual_position[0], 2), round(self.actual_position[1], 2)]
                if self.actual_position is not None
                else None
            ),
            "error_distance": (
                round(self.error_distance, 3)
                if self.error_distance is not None
                else None
            ),
            "endpoint_error": (
                round(self.endpoint_error, 3)
                if self.endpoint_error is not None
                else None
            ),
            "error_components": (
                [
                    round(self.error_components[0], 3),
                    round(self.error_components[1], 3),
                ]
                if self.error_components is not None
                else None
            ),
            "ade": round(self.ade, 3) if self.ade is not None else None,
            "fde": round(self.fde, 3) if self.fde is not None else None,
            "num_waypoints": len(self.waypoint_records),
            "following_track_id": self.following_track_id,
            "leading_track_id": self.leading_track_id,
            "following_speed_mps": (
                round(self.following_speed_mps, 2)
                if self.following_speed_mps is not None
                else None
            ),
            "leading_speed_mps": (
                round(self.leading_speed_mps, 2)
                if self.leading_speed_mps is not None
                else None
            ),
            "inter_vehicle_gap_m": (
                round(self.inter_vehicle_gap_m, 2)
                if self.inter_vehicle_gap_m is not None
                else None
            ),
            "drac_mps2": (
                round(self.drac_mps2, 3)
                if self.drac_mps2 is not None
                else None
            ),
            "drac_status": self.drac_status,
        }

