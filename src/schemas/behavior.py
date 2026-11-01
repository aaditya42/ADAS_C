from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


class BehaviorLabel(Enum):
    """Semantic behavior labels for tracked human and animal entities.

    Human behaviour categories cover pedestrian-vehicle interaction scenarios:
    standing, walking, running, crossing/approaching/leaving road, fallen, distracted.

    Animal behaviour categories cover common road-side observations:
    standing, walking, running, lying, grazing, crossing/approaching road.

    UNKNOWN is the fallback when the behaviour cannot be determined.
    """

    HUMAN_STANDING = "human_standing"
    HUMAN_WALKING = "human_walking"
    HUMAN_RUNNING = "human_running"
    HUMAN_CROSSING_ROAD = "human_crossing_road"
    HUMAN_APPROACHING_ROAD = "human_approaching_road"
    HUMAN_LEAVING_ROAD = "human_leaving_road"
    HUMAN_FALLEN = "human_fallen"
    HUMAN_DISTRACTED = "human_distracted"

    ANIMAL_STANDING = "animal_standing"
    ANIMAL_WALKING = "animal_walking"
    ANIMAL_RUNNING = "animal_running"
    ANIMAL_LYING = "animal_lying"
    ANIMAL_GRAZING = "animal_grazing"
    ANIMAL_CROSSING_ROAD = "animal_crossing_road"
    ANIMAL_APPROACHING_ROAD = "animal_approaching_road"

    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BehaviorState:
    """Behaviour state for a single tracked entity at a point in time.

    entity_type distinguishes human from animal subjects.
    primary_label is the most likely behaviour; secondary_labels hold alternatives.
    crossing_probability in [0, 1] when computed (None if unsupported).
    """

    track_id: int
    entity_type: Literal["human", "animal"]
    primary_label: BehaviorLabel
    secondary_labels: tuple[BehaviorLabel, ...]
    confidence: float
    crossing_probability: Optional[float] = None
    body_orientation_deg: Optional[float] = None
    head_orientation_deg: Optional[float] = None
    timestamp: float = 0.0
    schema_version: str = "1.0"


@dataclass(frozen=True, slots=True)
class BehaviorFrame:
    """Container for all behaviour states in a single frame."""

    frame_id: int
    timestamp: float
    states: tuple[BehaviorState, ...] = ()
    schema_version: str = "1.0"
