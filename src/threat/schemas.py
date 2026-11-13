"""Schemas and data structures for threat feature extraction and scenario classification.

This module defines typed records for threat features, scenario assessments,
and indicator validity for the monocular ADAS system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.schemas.bbox import BoundingBox


class ValidityStatus(str, Enum):
    """Validity status of an extracted threat indicator."""

    VALID_METRIC = "VALID_METRIC"
    VALID_PROXY = "VALID_PROXY"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "UNAVAILABLE"


class CameraDirection(str, Enum):
    """Camera-relative spatial direction of the object."""

    FRONT = "FRONT"
    FRONT_LEFT = "FRONT_LEFT"
    FRONT_RIGHT = "FRONT_RIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


class LaneRelationship(str, Enum):
    """Spatial relationship between the object and the ego travel corridor."""

    EGO_LANE = "EGO_LANE"
    LEFT_ADJACENT_LANE = "LEFT_ADJACENT_LANE"
    RIGHT_ADJACENT_LANE = "RIGHT_ADJACENT_LANE"
    ENTERING_EGO_LANE = "ENTERING_EGO_LANE"
    LEAVING_EGO_LANE = "LEAVING_EGO_LANE"
    OUTSIDE_RELEVANT_LANES = "OUTSIDE_RELEVANT_LANES"
    UNKNOWN = "UNKNOWN"


class TrajectoryConflictState(str, Enum):
    """Estimated conflict status between predicted object trajectory and ego path."""

    CONFLICT = "CONFLICT"
    NO_CONFLICT = "NO_CONFLICT"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "UNAVAILABLE"


class VulnerabilityLevel(str, Enum):
    """Physical vulnerability level of the road user class."""

    CRITICAL = "CRITICAL"      # Pedestrian, cyclist
    HIGH = "HIGH"              # Motorcycle
    STANDARD = "STANDARD"      # Car
    LOW_HEAVY = "LOW_HEAVY"    # Truck, bus
    UNKNOWN = "UNKNOWN"


class ScenarioType(str, Enum):
    """High-level interaction scenario category."""

    REAR_END = "REAR_END"
    PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"
    CYCLIST_CONFLICT = "CYCLIST_CONFLICT"
    VEHICLE_CUT_IN = "VEHICLE_CUT_IN"
    ADJACENT_LANE_THREAT = "ADJACENT_LANE_THREAT"
    ANIMAL_CROSSING = "ANIMAL_CROSSING"
    UNKNOWN = "UNKNOWN"


class ScenarioState(str, Enum):
    """Fine-grained sub-state within a scenario."""

    # Scenario 1: Rear-End
    REAR_END_CONFLICT = "REAR_END_CONFLICT"
    REAR_END_POTENTIAL = "REAR_END_POTENTIAL"
    NO_REAR_END_CONFLICT = "NO_REAR_END_CONFLICT"
    UNCERTAIN = "UNCERTAIN"

    # Scenario 2: Pedestrian Crossing
    PEDESTRIAN_CROSSING_CONFLICT = "PEDESTRIAN_CROSSING_CONFLICT"
    PEDESTRIAN_APPROACHING_PATH = "PEDESTRIAN_APPROACHING_PATH"
    PEDESTRIAN_ALONGSIDE_PATH = "PEDESTRIAN_ALONGSIDE_PATH"
    PEDESTRIAN_CLEAR_OF_PATH = "PEDESTRIAN_CLEAR_OF_PATH"

    # Scenario 3: Cyclist Conflict
    CYCLIST_CROSSING_CONFLICT = "CYCLIST_CROSSING_CONFLICT"
    CYCLIST_ENTERING_LANE = "CYCLIST_ENTERING_LANE"
    CYCLIST_LEAD_CONFLICT = "CYCLIST_LEAD_CONFLICT"
    CYCLIST_ADJACENT_LANE = "CYCLIST_ADJACENT_LANE"

    # Scenario 4: Vehicle Cut-In
    POTENTIAL_CUT_IN = "POTENTIAL_CUT_IN"
    CONFIRMED_CUT_IN = "CONFIRMED_CUT_IN"
    COMPLETED_CUT_IN = "COMPLETED_CUT_IN"
    UNCERTAIN_LANE_TRANSITION = "UNCERTAIN_LANE_TRANSITION"

    # Scenario 5: Adjacent-Lane Threat
    ADJACENT_LANE_ENCROACHING = "ADJACENT_LANE_ENCROACHING"
    ADJACENT_LANE_STABLE = "ADJACENT_LANE_STABLE"

    # Supporting: Animal
    ANIMAL_CROSSING_CONFLICT = "ANIMAL_CROSSING_CONFLICT"
    ANIMAL_APPROACHING_PATH = "ANIMAL_APPROACHING_PATH"
    ANIMAL_CLEAR_OF_PATH = "ANIMAL_CLEAR_OF_PATH"

    # General
    NONE = "NONE"


class ModelEvaluationStatus(str, Enum):
    """Evaluation validity status for the selected risk model."""

    VALID_FULL_MODEL = "VALID_FULL_MODEL"
    VALID_ADAPTED_MODEL = "VALID_ADAPTED_MODEL"
    INSUFFICIENT_INDICATORS = "INSUFFICIENT_INDICATORS"
    UNSUPPORTED_SCENARIO = "UNSUPPORTED_SCENARIO"
    INVALID_MEASUREMENTS = "INVALID_MEASUREMENTS"


class RiskModelType(str, Enum):
    """Identifier for risk assessment models."""

    OPTION_A_ADRS_2025 = "OPTION_A_ADRS_2025"
    OPTION_B_INTERSECTION_2025 = "OPTION_B_INTERSECTION_2025"
    FHWA_SSAM_DIAGNOSTIC = "FHWA_SSAM_DIAGNOSTIC"
    RULE_BASED_ASSESSMENT = "RULE_BASED_ASSESSMENT"
    UNSUPPORTED = "UNSUPPORTED"


class ReliabilityGrade(str, Enum):
    """Reliability assessment of threat measurements and scores."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    PROVISIONAL = "PROVISIONAL"
    UNRELIABLE = "UNRELIABLE"


class ThreatRiskLevel(str, Enum):
    """Provisional threat risk levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNASSESSED = "UNASSESSED"


@dataclass(frozen=True)
class IndicatorValidity:
    """Validity metadata and diagnostic reason for an individual risk indicator."""

    status: ValidityStatus
    value: Any = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert validity record to a serializable dictionary."""
        return {
            "status": self.status.value,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass
class ThreatFeatures:
    """Per-object threat feature record extracted from perception outputs.

    Maintains all individual kinematic, spatial, behavioral, and conflict
    indicators alongside their explicit physical validity metadata.
    """

    track_id: int
    tracker_id: int
    object_class: str
    bbox: BoundingBox
    timestamp: float

    # Spatial and direction attributes
    direction: CameraDirection
    lane_relationship: LaneRelationship
    lane_overlap: float = 0.0
    lateral_deviation: float | None = None

    # Surrogate safety indicators
    ttc: float | None = None
    pet: float | None = None
    drac: float | None = None

    # Motion kinematics
    relative_speed: float | None = None
    relative_acceleration: float | None = None
    motion_direction: str = "unknown"
    speed_variation: float | None = None
    acceleration_noise: float | None = None
    looming_rate: float | None = None

    # Trajectory conflict attributes
    trajectory_conflict: TrajectoryConflictState = TrajectoryConflictState.UNAVAILABLE
    trajectory_conflict_time: float | None = None

    # Vulnerability & Behavior
    vulnerability: VulnerabilityLevel = VulnerabilityLevel.STANDARD
    crossing_probability: float | None = None
    behavior_label: str | None = None

    # Tracking observation metadata
    observation_duration: float = 0.0
    observation_count: int = 1
    is_metric: bool = False

    # Detailed indicator validity registry
    indicator_validity: dict[str, IndicatorValidity] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert threat feature record to a dictionary."""
        return {
            "track_id": self.track_id,
            "tracker_id": self.tracker_id,
            "object_class": self.object_class,
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "timestamp": round(self.timestamp, 4),
            "direction": self.direction.value,
            "lane_relationship": self.lane_relationship.value,
            "lane_overlap": round(self.lane_overlap, 4),
            "lateral_deviation": (
                round(self.lateral_deviation, 4)
                if self.lateral_deviation is not None
                else None
            ),
            "ttc": round(self.ttc, 3) if self.ttc is not None else None,
            "pet": round(self.pet, 3) if self.pet is not None else None,
            "drac": round(self.drac, 3) if self.drac is not None else None,
            "relative_speed": (
                round(self.relative_speed, 3)
                if self.relative_speed is not None
                else None
            ),
            "relative_acceleration": (
                round(self.relative_acceleration, 3)
                if self.relative_acceleration is not None
                else None
            ),
            "motion_direction": self.motion_direction,
            "speed_variation": (
                round(self.speed_variation, 4)
                if self.speed_variation is not None
                else None
            ),
            "acceleration_noise": (
                round(self.acceleration_noise, 4)
                if self.acceleration_noise is not None
                else None
            ),
            "looming_rate": (
                round(self.looming_rate, 4)
                if self.looming_rate is not None
                else None
            ),
            "trajectory_conflict": self.trajectory_conflict.value,
            "trajectory_conflict_time": (
                round(self.trajectory_conflict_time, 3)
                if self.trajectory_conflict_time is not None
                else None
            ),
            "vulnerability": self.vulnerability.value,
            "crossing_probability": (
                round(self.crossing_probability, 3)
                if self.crossing_probability is not None
                else None
            ),
            "behavior_label": self.behavior_label,
            "observation_duration": round(self.observation_duration, 3),
            "observation_count": self.observation_count,
            "is_metric": self.is_metric,
            "indicator_validity": {
                k: v.to_dict() for k, v in self.indicator_validity.items()
            },
        }


@dataclass
class ScenarioAssessment:
    """Scenario classification evaluation for a single tracked object."""

    track_id: int
    scenario: ScenarioType
    state: str
    confidence: float
    direction: CameraDirection
    lane_relationship: LaneRelationship
    evidence: list[str] = field(default_factory=list)
    secondary_scenarios: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    validity: ValidityStatus = ValidityStatus.VALID_PROXY

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario assessment record to a dictionary."""
        return {
            "track_id": self.track_id,
            "scenario": self.scenario.value,
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "direction": self.direction.value,
            "lane_relationship": self.lane_relationship.value,
            "evidence": list(self.evidence),
            "secondary_scenarios": list(self.secondary_scenarios),
            "timestamp": round(self.timestamp, 4),
            "validity": self.validity.value,
        }


@dataclass(frozen=True)
class NormalizedIndicator:
    """A normalized risk indicator with traceability metadata."""

    raw_value: float | None
    normalized_value: float | None
    method: str
    reference_range: tuple[float, float]
    risk_direction: str  # "INVERSE" (lower is riskier) or "DIRECT" (higher is riskier)
    validity: ValidityStatus
    source_model: str
    adaptation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert normalized indicator to serializable dictionary."""
        return {
            "raw_value": round(self.raw_value, 4) if self.raw_value is not None else None,
            "normalized_value": (
                round(self.normalized_value, 4) if self.normalized_value is not None else None
            ),
            "method": self.method,
            "reference_range": [round(self.reference_range[0], 2), round(self.reference_range[1], 2)],
            "risk_direction": self.risk_direction,
            "validity": self.validity.value,
            "source_model": self.source_model,
            "adaptation_note": self.adaptation_note,
        }


@dataclass
class ThreatRiskAssessment:
    """Per-object threat risk assessment and prioritization evaluation."""

    track_id: int
    object_class: str
    scenario: ScenarioType
    scenario_state: str
    direction: CameraDirection
    lane_relationship: LaneRelationship

    model_name: RiskModelType
    model_status: ModelEvaluationStatus
    published_coefficients: dict[str, float] = field(default_factory=dict)
    raw_indicators: dict[str, float | None] = field(default_factory=dict)
    normalized_indicators: dict[str, NormalizedIndicator] = field(default_factory=dict)

    risk_score: float | None = None
    risk_level: ThreatRiskLevel = ThreatRiskLevel.UNASSESSED
    reliability: ReliabilityGrade = ReliabilityGrade.PROVISIONAL

    priority_rank: int | None = None
    is_primary_threat: bool = False
    collision_urgency: float | None = None

    evidence: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert risk assessment to serializable dictionary."""
        return {
            "track_id": self.track_id,
            "object_class": self.object_class,
            "scenario": self.scenario.value,
            "scenario_state": self.scenario_state,
            "direction": self.direction.value,
            "lane_relationship": self.lane_relationship.value,
            "model_name": self.model_name.value,
            "model_status": self.model_status.value,
            "published_coefficients": dict(self.published_coefficients),
            "raw_indicators": {
                k: round(v, 4) if v is not None else None for k, v in self.raw_indicators.items()
            },
            "normalized_indicators": {
                k: v.to_dict() for k, v in self.normalized_indicators.items()
            },
            "risk_score": round(self.risk_score, 4) if self.risk_score is not None else None,
            "risk_level": self.risk_level.value,
            "reliability": self.reliability.value,
            "priority_rank": self.priority_rank,
            "is_primary_threat": self.is_primary_threat,
            "collision_urgency": (
                round(self.collision_urgency, 3) if self.collision_urgency is not None else None
            ),
            "evidence": list(self.evidence),
            "rejection_reasons": list(self.rejection_reasons),
            "timestamp": round(self.timestamp, 4),
        }


class DirectionalWarningLevel(str, Enum):
    """Driver warning severity level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNASSESSED = "UNASSESSED"


class WarningEventState(str, Enum):
    """Lifecycle state of a warning event."""

    INACTIVE = "INACTIVE"
    CANDIDATE = "CANDIDATE"
    ACTIVE_LOW = "ACTIVE_LOW"
    ACTIVE_MEDIUM = "ACTIVE_MEDIUM"
    ACTIVE_HIGH = "ACTIVE_HIGH"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"


@dataclass
class WarningEvent:
    """Persistent warning event record tracking a specific threat over time."""

    event_id: str
    track_id: int
    object_class: str
    scenario: str
    direction: CameraDirection
    current_level: DirectionalWarningLevel
    previous_level: DirectionalWarningLevel
    state: WarningEventState

    first_detected_timestamp: float
    confirmation_timestamp: float | None = None
    activation_timestamp: float | None = None
    last_updated_timestamp: float = 0.0

    escalation_timestamps: list[tuple[float, str, str]] = field(default_factory=list)
    deescalation_timestamps: list[tuple[float, str, str]] = field(default_factory=list)
    resolution_timestamp: float | None = None
    resolution_reason: str = ""

    risk_score: float | None = None
    risk_model: str = ""
    reliability: str = "PROVISIONAL"
    ttc: float | None = None
    collision_urgency: float | None = None
    is_primary_threat: bool = False
    priority_rank: int | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Active warning duration in seconds (from activation to resolution or last update)."""
        if self.activation_timestamp is None:
            return None
        end_t = self.resolution_timestamp if self.resolution_timestamp is not None else self.last_updated_timestamp
        return max(0.0, end_t - self.activation_timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Convert warning event to dictionary."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "scenario": self.scenario,
            "direction": self.direction.value,
            "current_level": self.current_level.value,
            "previous_level": self.previous_level.value,
            "state": self.state.value,
            "first_detected_timestamp": round(self.first_detected_timestamp, 4),
            "confirmation_timestamp": (
                round(self.confirmation_timestamp, 4)
                if self.confirmation_timestamp is not None
                else None
            ),
            "activation_timestamp": (
                round(self.activation_timestamp, 4)
                if self.activation_timestamp is not None
                else None
            ),
            "last_updated_timestamp": round(self.last_updated_timestamp, 4),
            "escalation_timestamps": [
                [round(t, 4), l1, l2] for t, l1, l2 in self.escalation_timestamps
            ],
            "deescalation_timestamps": [
                [round(t, 4), l1, l2] for t, l1, l2 in self.deescalation_timestamps
            ],
            "resolution_timestamp": (
                round(self.resolution_timestamp, 4)
                if self.resolution_timestamp is not None
                else None
            ),
            "resolution_reason": self.resolution_reason,
            "risk_score": round(self.risk_score, 4) if self.risk_score is not None else None,
            "risk_model": self.risk_model,
            "reliability": self.reliability,
            "ttc": round(self.ttc, 3) if self.ttc is not None else None,
            "collision_urgency": (
                round(self.collision_urgency, 3)
                if self.collision_urgency is not None
                else None
            ),
            "is_primary_threat": self.is_primary_threat,
            "priority_rank": self.priority_rank,
            "evidence": list(self.evidence),
        }


@dataclass
class WarningOutput:
    """Real-time structured warning notification output for downstream consumption."""

    event_id: str
    track_id: int
    object_class: str
    scenario: str
    direction: CameraDirection
    level: DirectionalWarningLevel
    state: WarningEventState

    risk_score: float | None = None
    risk_model: str = ""
    reliability: str = "PROVISIONAL"
    ttc: float | None = None
    collision_urgency: float | None = None
    is_primary_threat: bool = False
    priority_rank: int | None = None
    timestamp: float = 0.0
    reason: str = ""
    escalated: bool = False
    deescalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert warning output to dictionary."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "scenario": self.scenario,
            "direction": self.direction.value,
            "level": self.level.value,
            "state": self.state.value,
            "risk_score": round(self.risk_score, 4) if self.risk_score is not None else None,
            "risk_model": self.risk_model,
            "reliability": self.reliability,
            "ttc": round(self.ttc, 3) if self.ttc is not None else None,
            "collision_urgency": (
                round(self.collision_urgency, 3)
                if self.collision_urgency is not None
                else None
            ),
            "is_primary_threat": self.is_primary_threat,
            "priority_rank": self.priority_rank,
            "timestamp": round(self.timestamp, 4),
            "reason": self.reason,
            "escalated": self.escalated,
            "deescalated": self.deescalated,
        }



