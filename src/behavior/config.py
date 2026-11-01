from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class HumanSpeedConfig:
    walking_min_mps: float = 0.3
    running_min_mps: float = 1.5


@dataclass(frozen=True, slots=True)
class HumanFallenConfig:
    torso_angle_threshold_deg: float = 45.0


@dataclass(frozen=True, slots=True)
class HumanDistractedConfig:
    head_motion_angle_threshold_deg: float = 60.0


@dataclass(frozen=True, slots=True)
class HumanRoadConfig:
    crossing_probability_high: float = 0.7
    approaching_distance_px: float = 100.0
    approaching_angle_threshold_deg: float = 45.0
    leaving_distance_px: float = 80.0


@dataclass(frozen=True, slots=True)
class HumanBehaviorSubConfig:
    speed: HumanSpeedConfig = field(default_factory=HumanSpeedConfig)
    fallen: HumanFallenConfig = field(default_factory=HumanFallenConfig)
    distracted: HumanDistractedConfig = field(default_factory=HumanDistractedConfig)
    road: HumanRoadConfig = field(default_factory=HumanRoadConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanBehaviorSubConfig:
        speed_data = data.get("speed", {})
        fallen_data = data.get("fallen", {})
        distracted_data = data.get("distracted", {})
        road_data = data.get("road", {})
        return cls(
            speed=HumanSpeedConfig(
                walking_min_mps=float(speed_data.get("walking_min_mps", 0.3)),
                running_min_mps=float(speed_data.get("running_min_mps", 1.5)),
            ),
            fallen=HumanFallenConfig(
                torso_angle_threshold_deg=float(
                    fallen_data.get("torso_angle_threshold_deg", 45.0)
                ),
            ),
            distracted=HumanDistractedConfig(
                head_motion_angle_threshold_deg=float(
                    distracted_data.get("head_motion_angle_threshold_deg", 60.0)
                ),
            ),
            road=HumanRoadConfig(
                crossing_probability_high=float(
                    road_data.get("crossing_probability_high", 0.7)
                ),
                approaching_distance_px=float(
                    road_data.get("approaching_distance_px", 100.0)
                ),
                approaching_angle_threshold_deg=float(
                    road_data.get("approaching_angle_threshold_deg", 45.0)
                ),
                leaving_distance_px=float(
                    road_data.get("leaving_distance_px", 80.0)
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class AnimalSpeedConfig:
    walking_min_mps: float = 0.2
    running_min_mps: float = 1.2


@dataclass(frozen=True, slots=True)
class AnimalLyingConfig:
    torso_height_threshold: float = 0.3
    spine_angle_threshold_deg: float = 30.0


@dataclass(frozen=True, slots=True)
class AnimalGrazingConfig:
    head_down_angle_threshold_deg: float = 45.0
    max_speed_mps: float = 0.3


@dataclass(frozen=True, slots=True)
class AnimalRoadConfig:
    crossing_probability_high: float = 0.7
    approaching_distance_px: float = 120.0
    approaching_angle_threshold_deg: float = 45.0


@dataclass(frozen=True, slots=True)
class AnimalBehaviorSubConfig:
    speed: AnimalSpeedConfig = field(default_factory=AnimalSpeedConfig)
    lying: AnimalLyingConfig = field(default_factory=AnimalLyingConfig)
    grazing: AnimalGrazingConfig = field(default_factory=AnimalGrazingConfig)
    road: AnimalRoadConfig = field(default_factory=AnimalRoadConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnimalBehaviorSubConfig:
        speed_data = data.get("speed", {})
        lying_data = data.get("lying", {})
        grazing_data = data.get("grazing", {})
        road_data = data.get("road", {})
        return cls(
            speed=AnimalSpeedConfig(
                walking_min_mps=float(speed_data.get("walking_min_mps", 0.2)),
                running_min_mps=float(speed_data.get("running_min_mps", 1.2)),
            ),
            lying=AnimalLyingConfig(
                torso_height_threshold=float(
                    lying_data.get("torso_height_threshold", 0.3)
                ),
                spine_angle_threshold_deg=float(
                    lying_data.get("spine_angle_threshold_deg", 30.0)
                ),
            ),
            grazing=AnimalGrazingConfig(
                head_down_angle_threshold_deg=float(
                    grazing_data.get("head_down_angle_threshold_deg", 45.0)
                ),
                max_speed_mps=float(grazing_data.get("max_speed_mps", 0.3)),
            ),
            road=AnimalRoadConfig(
                crossing_probability_high=float(
                    road_data.get("crossing_probability_high", 0.7)
                ),
                approaching_distance_px=float(
                    road_data.get("approaching_distance_px", 120.0)
                ),
                approaching_angle_threshold_deg=float(
                    road_data.get("approaching_angle_threshold_deg", 45.0)
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossingProbabilityConfig:
    decay_distance_px: float = 50.0
    heading_weight: float = 0.3
    speed_weight: float = 0.2


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    enabled: bool = False
    human: HumanBehaviorSubConfig = field(default_factory=HumanBehaviorSubConfig)
    animal: AnimalBehaviorSubConfig = field(default_factory=AnimalBehaviorSubConfig)
    crossing_probability: CrossingProbabilityConfig = field(
        default_factory=CrossingProbabilityConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehaviorConfig:
        if not data:
            return cls()
        human_data = data.get("human", {})
        animal_data = data.get("animal", {})
        cp_data = data.get("crossing_probability", {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            human=HumanBehaviorSubConfig.from_dict(human_data),
            animal=AnimalBehaviorSubConfig.from_dict(animal_data),
            crossing_probability=CrossingProbabilityConfig(
                decay_distance_px=float(
                    cp_data.get("decay_distance_px", 50.0)
                ),
                heading_weight=float(
                    cp_data.get("heading_weight", 0.3)
                ),
                speed_weight=float(
                    cp_data.get("speed_weight", 0.2)
                ),
            ),
        )
