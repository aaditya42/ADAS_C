from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from src.risk.contributors import RiskAdjustment, RiskContributor, RiskContext
from src.schemas.behavior import BehaviorLabel, BehaviorState
from src.utils.logger import get_logger


@dataclass(frozen=True)
class BehaviorRiskContributionConfig:
    fallen_boost: float = 30.0
    crossing_boost: float = 25.0
    approaching_boost: float = 20.0
    distracted_approaching_boost: float = 15.0
    animal_crossing_boost: float = 25.0
    animal_approaching_boost: float = 20.0
    crossing_prob_threshold: float = 0.5
    ttc_boost_scale: float = 0.3
    max_boost: float = 50.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehaviorRiskContributionConfig:
        return cls(
            fallen_boost=float(data.get("fallen_boost", 30.0)),
            crossing_boost=float(data.get("crossing_boost", 25.0)),
            approaching_boost=float(data.get("approaching_boost", 20.0)),
            distracted_approaching_boost=float(
                data.get("distracted_approaching_boost", 15.0)
            ),
            animal_crossing_boost=float(data.get("animal_crossing_boost", 25.0)),
            animal_approaching_boost=float(data.get("animal_approaching_boost", 20.0)),
            crossing_prob_threshold=float(
                data.get("crossing_prob_threshold", 0.5)
            ),
            ttc_boost_scale=float(data.get("ttc_boost_scale", 0.3)),
            max_boost=float(data.get("max_boost", 50.0)),
        )


ROAD_LABELS_WITH_DISTRACTED = frozenset({
    BehaviorLabel.HUMAN_CROSSING_ROAD,
    BehaviorLabel.HUMAN_APPROACHING_ROAD,
})


class BehaviorRiskContributor(RiskContributor):
    """Additive risk contributor that boosts risk scores based on behavior states.

    Rules (applied in order, cumulatively with a final cap):
      - FALLEN human → ``fallen_boost``
      - CROSSING_ROAD (human/animal) with crossing_prob ≥ threshold
        → ``crossing_boost`` + TTC-scaled increment
      - APPROACHING_ROAD (human/animal) with crossing_prob ≥ threshold
        → ``approaching_boost`` + TTC-scaled increment
      - DISTRACTED label combined with CROSSING/APPROACHING → additional boost
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._cfg = BehaviorRiskContributionConfig.from_dict(weights or {})
        self._logger = get_logger(f"{__name__}.BehaviorRiskContributor")

    def contribute(self, context: RiskContext) -> Sequence[RiskAdjustment]:
        if context.behavior_frame is None:
            return []

        adjustments: list[RiskAdjustment] = []
        for state in context.behavior_frame.states:
            boost = self._compute_boost(state, context)
            if abs(boost) > 0.001:
                adjustments.append(
                    RiskAdjustment(track_id=state.track_id, delta_score=boost)
                )

        return adjustments

    def _compute_boost(self, state: BehaviorState, context: RiskContext) -> float:
        label = state.primary_label
        ttc = context.ttc.get(state.track_id)
        prob = state.crossing_probability or 0.0
        boost = 0.0

        if label == BehaviorLabel.HUMAN_FALLEN:
            boost = self._cfg.fallen_boost

        elif label in (BehaviorLabel.HUMAN_CROSSING_ROAD, BehaviorLabel.ANIMAL_CROSSING_ROAD):
            if prob < self._cfg.crossing_prob_threshold:
                return 0.0
            is_human = label == BehaviorLabel.HUMAN_CROSSING_ROAD
            boost = self._cfg.crossing_boost if is_human else self._cfg.animal_crossing_boost
            boost += self._ttc_increment(ttc, boost)

            if is_human and BehaviorLabel.HUMAN_DISTRACTED in state.secondary_labels:
                boost += self._cfg.distracted_approaching_boost

        elif label in (BehaviorLabel.HUMAN_APPROACHING_ROAD, BehaviorLabel.ANIMAL_APPROACHING_ROAD):
            if prob < self._cfg.crossing_prob_threshold:
                return 0.0
            is_human = label == BehaviorLabel.HUMAN_APPROACHING_ROAD
            boost = self._cfg.approaching_boost if is_human else self._cfg.animal_approaching_boost
            boost += self._ttc_increment(ttc, boost)

            if is_human and BehaviorLabel.HUMAN_DISTRACTED in state.secondary_labels:
                boost += self._cfg.distracted_approaching_boost

        return min(boost, self._cfg.max_boost)

    def _ttc_increment(self, ttc: Optional[float], base_boost: float) -> float:
        if ttc is None or ttc <= 0:
            return 0.0
        return min(base_boost * 0.5, base_boost * self._cfg.ttc_boost_scale / max(ttc, 0.1))
