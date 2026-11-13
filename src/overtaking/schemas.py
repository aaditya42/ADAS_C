from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.schemas.bbox import BoundingBox


class OvertakingState(str, Enum):
    """Lifecycle state machine stages for an individual candidate vehicle."""

    VEHICLE_AHEAD = "VEHICLE_AHEAD"
    OVERTAKING_STARTED = "OVERTAKING_STARTED"
    PASSING = "PASSING"
    OVERTAKING_COMPLETED = "OVERTAKING_COMPLETED"
    INCOMPLETE = "INCOMPLETE"


class OvertakingEventStatus(str, Enum):
    """Overall outcome status of an overtaking event."""

    IN_PROGRESS = "IN_PROGRESS"
    ACTIVE = "IN_PROGRESS"  # Backward-compatible alias
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"
    UNCERTAIN = "UNCERTAIN"
    INVALIDATED = "INVALIDATED"


class IdentityContinuityStatus(str, Enum):
    """Continuity classification of tracker identity during a manoeuvre."""

    CONTINUOUS = "CONTINUOUS"
    RECOVERED = "RECOVERED"
    SUSPECTED_ID_SWITCH = "SUSPECTED_ID_SWITCH"
    DISCONTINUOUS = "DISCONTINUOUS"


class CorridorPosition(str, Enum):
    """Spatial relationship of an object relative to the ego vehicle forward corridor."""

    IN_EGO_CORRIDOR = "IN_EGO_CORRIDOR"
    LEFT_OF_CORRIDOR = "LEFT_OF_CORRIDOR"
    RIGHT_OF_CORRIDOR = "RIGHT_OF_CORRIDOR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OvertakingObservation:
    """Timestamped spatial-temporal snapshot of a candidate vehicle.

    Attributes:
        frame_id: Monotonic video frame counter.
        timestamp: Presentation or media elapsed timestamp in seconds.
        bbox: Axis-aligned bounding box in pixels.
        center_x: Horizontal center in pixels.
        center_y: Vertical center in pixels.
        bottom_y: Bottom edge (y2) in pixels (contact patch proximity).
        bbox_area: Bounding box area (w * h) in pixels^2.
        relative_depth: Normalized relative depth (or None if unavailable/uncalibrated).
        corridor_position: Position relative to ego travel corridor.
        in_ego_corridor: Whether the object center is within the ego corridor.
    """

    frame_id: int
    timestamp: float
    bbox: BoundingBox
    center_x: float
    center_y: float
    bottom_y: float
    bbox_area: float
    relative_depth: float | None
    corridor_position: CorridorPosition
    in_ego_corridor: bool


@dataclass
class OvertakingEvent:
    """Comprehensive structured record of an overtaking manoeuvre.

    Maintains distinct, unmixed timestamps for physical onset, confirmation,
    completion, and associated delays.
    """

    # Entity Identification
    event_id: str
    track_id: int
    track_instance_id: str
    vehicle_class: str

    # State & Lifecycle
    state: OvertakingState = OvertakingState.VEHICLE_AHEAD
    status: OvertakingEventStatus = OvertakingEventStatus.IN_PROGRESS
    identity_continuity_status: IdentityContinuityStatus = IdentityContinuityStatus.CONTINUOUS
    reassociated_track_ids: list[int] = field(default_factory=list)

    # Timing Data (strictly video source seconds, preserving high precision)
    first_observed_timestamp: float = 0.0
    candidate_start_timestamp: float | None = None
    start_timestamp: float | None = None
    start_confirmation_timestamp: float | None = None
    start_confirmation_delay_seconds: float | None = None

    candidate_end_timestamp: float | None = None
    end_timestamp: float | None = None
    end_confirmation_timestamp: float | None = None
    completion_confirmation_delay_seconds: float | None = None
    duration_seconds: float | None = None

    last_observed_timestamp: float = 0.0
    missing_observation_duration: float = 0.0

    # Passing Dynamics & Quality Metrics
    passing_side: str = "UNKNOWN"  # "LEFT", "RIGHT", or "UNKNOWN"
    completion_confidence: float = 0.0
    uncertainty_reason: str = ""

    # Spatial Geometry (Initial vs Latest)
    initial_position: tuple[float, float] | None = None
    latest_position: tuple[float, float] | None = None
    initial_bbox: BoundingBox | None = None
    latest_bbox: BoundingBox | None = None
    initial_depth: float | None = None
    latest_depth: float | None = None
    initial_corridor_position: CorridorPosition = CorridorPosition.UNKNOWN
    latest_corridor_position: CorridorPosition = CorridorPosition.UNKNOWN

    # Observation History & Health Flags
    observations: deque[OvertakingObservation] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    total_observations: int = 0
    valid_observation_count: int = 0
    lane_data_validity: bool = True
    depth_data_validity: bool = True

    def confirm_start(
        self,
        start_time: float,
        confirmation_time: float,
        candidate_start_time: float | None = None,
    ) -> None:
        """Confirm event start, establishing separate physical start and confirmation timestamps."""
        self.state = OvertakingState.OVERTAKING_STARTED
        self.status = OvertakingEventStatus.IN_PROGRESS
        self.start_timestamp = start_time
        self.start_confirmation_timestamp = confirmation_time
        self.candidate_start_timestamp = (
            candidate_start_time if candidate_start_time is not None else start_time
        )
        if self.start_confirmation_timestamp >= self.start_timestamp:
            self.start_confirmation_delay_seconds = round(
                self.start_confirmation_timestamp - self.start_timestamp, 4
            )
        else:
            self.start_confirmation_delay_seconds = 0.0

    def finalize_completion(
        self,
        end_time: float,
        confirmation_time: float | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Finalize event as successfully completed.

        Calculates duration strictly from (end_timestamp - start_timestamp)
        and preserves separate confirmation delay.
        """
        self.state = OvertakingState.OVERTAKING_COMPLETED
        self.status = OvertakingEventStatus.COMPLETED
        self.end_timestamp = end_time
        conf_time = confirmation_time if confirmation_time is not None else end_time
        self.end_confirmation_timestamp = conf_time

        if self.end_confirmation_timestamp >= self.end_timestamp:
            self.completion_confirmation_delay_seconds = round(
                self.end_confirmation_timestamp - self.end_timestamp, 4
            )
        else:
            self.completion_confirmation_delay_seconds = 0.0

        if self.start_timestamp is not None:
            # High-precision duration in seconds (non-negative)
            self.duration_seconds = max(0.0, self.end_timestamp - self.start_timestamp)

        self.completion_confidence = min(1.0, max(0.0, confidence))
        self.uncertainty_reason = ""

    def finalize_incomplete(
        self,
        reason: str,
        status: OvertakingEventStatus = OvertakingEventStatus.INCOMPLETE,
    ) -> None:
        """Mark event as incomplete, uncertain, or invalidated.

        Guarantees that no fabricated completion timestamp or duration is produced.
        """
        self.state = OvertakingState.INCOMPLETE
        self.status = status
        self.uncertainty_reason = reason
        self.end_timestamp = None
        self.end_confirmation_timestamp = None
        self.completion_confirmation_delay_seconds = None
        self.duration_seconds = None
        self.completion_confidence = 0.0

    @property
    def identity_continuity(self) -> IdentityContinuityStatus:
        """Alias for identity_continuity_status."""
        return self.identity_continuity_status

    @identity_continuity.setter
    def identity_continuity(self, val: IdentityContinuityStatus) -> None:
        self.identity_continuity_status = val

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary format for downstream reporting and serialization."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "track_instance_id": self.track_instance_id,
            "vehicle_class": self.vehicle_class,
            "state": self.state.value,
            "status": self.status.value,
            "identity_continuity": self.identity_continuity_status.value,
            "passing_side": self.passing_side,
            "candidate_start_timestamp": (
                round(self.candidate_start_timestamp, 3)
                if self.candidate_start_timestamp is not None
                else None
            ),
            "start_timestamp": (
                round(self.start_timestamp, 3)
                if self.start_timestamp is not None
                else None
            ),
            "start_confirmation_timestamp": (
                round(self.start_confirmation_timestamp, 3)
                if self.start_confirmation_timestamp is not None
                else None
            ),
            "start_confirmation_delay_seconds": (
                round(self.start_confirmation_delay_seconds, 3)
                if self.start_confirmation_delay_seconds is not None
                else None
            ),
            "candidate_end_timestamp": (
                round(self.candidate_end_timestamp, 3)
                if self.candidate_end_timestamp is not None
                else None
            ),
            "end_timestamp": (
                round(self.end_timestamp, 3)
                if self.end_timestamp is not None
                else None
            ),
            "end_confirmation_timestamp": (
                round(self.end_confirmation_timestamp, 3)
                if self.end_confirmation_timestamp is not None
                else None
            ),
            "completion_confirmation_delay_seconds": (
                round(self.completion_confirmation_delay_seconds, 3)
                if self.completion_confirmation_delay_seconds is not None
                else None
            ),
            "duration_seconds": (
                round(self.duration_seconds, 3)
                if self.duration_seconds is not None
                else None
            ),
            "last_observed_timestamp": round(self.last_observed_timestamp, 3),
            "missing_observation_duration": round(self.missing_observation_duration, 3),
            "completion_confidence": round(self.completion_confidence, 2),
            "uncertainty_reason": self.uncertainty_reason,
            "total_observations": self.total_observations,
            "valid_observation_count": self.valid_observation_count,
            "initial_position": (
                [round(self.initial_position[0], 1), round(self.initial_position[1], 1)]
                if self.initial_position is not None
                else None
            ),
            "latest_position": (
                [round(self.latest_position[0], 1), round(self.latest_position[1], 1)]
                if self.latest_position is not None
                else None
            ),
            "initial_bbox": (
                [
                    round(self.initial_bbox.x1, 1),
                    round(self.initial_bbox.y1, 1),
                    round(self.initial_bbox.x2, 1),
                    round(self.initial_bbox.y2, 1),
                ]
                if self.initial_bbox is not None
                else None
            ),
            "latest_bbox": (
                [
                    round(self.latest_bbox.x1, 1),
                    round(self.latest_bbox.y1, 1),
                    round(self.latest_bbox.x2, 1),
                    round(self.latest_bbox.y2, 1),
                ]
                if self.latest_bbox is not None
                else None
            ),
            "initial_depth": (
                round(self.initial_depth, 2)
                if self.initial_depth is not None
                else None
            ),
            "latest_depth": (
                round(self.latest_depth, 2)
                if self.latest_depth is not None
                else None
            ),
            "lane_data_validity": self.lane_data_validity,
            "depth_data_validity": self.depth_data_validity,
        }


@dataclass(frozen=True)
class OvertakingCycleResult:
    """Cycle summary of overtaking events emitted for the current frame.

    Attributes:
        active_events: Events currently in progress (STARTED or PASSING).
        newly_started: Events that transitioned to OVERTAKING_STARTED this frame.
        newly_completed: Events successfully completed this frame.
        newly_incomplete: Events terminated as INCOMPLETE or UNCERTAIN this frame.
    """

    active_events: list[OvertakingEvent]
    newly_started: list[OvertakingEvent]
    newly_completed: list[OvertakingEvent]
    newly_incomplete: list[OvertakingEvent]
