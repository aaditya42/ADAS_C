"""Dynamic threat prioritization and ranking engine.

Ranks all active road user threat assessments based on collision urgency,
model validity, normalized risk score, and temporal persistence.
"""

from __future__ import annotations

import math
from typing import Any

from src.threat.schemas import (
    ModelEvaluationStatus,
    ReliabilityGrade,
    ScenarioState,
    ThreatRiskAssessment,
    ThreatRiskLevel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreatPrioritizer:
    """Prioritizes and ranks tracked threats dynamically per processing frame."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        threat_cfg = self.config.get("threat_assessment", {})
        prio_cfg = threat_cfg.get("prioritization", {})

        self.enabled: bool = bool(threat_cfg.get("enabled", True) and prio_cfg.get("enabled", True))
        self.max_ranked_threats: int = int(prio_cfg.get("max_ranked_threats", 10))
        self.rank_swap_hysteresis: float = float(prio_cfg.get("rank_swap_hysteresis", 0.05))

        self._prev_primary_id: int | None = None
        self._prev_ranks: dict[int, int] = {}

    def reset(self) -> None:
        """Clear internal ranking memory."""
        self._prev_primary_id = None
        self._prev_ranks.clear()

    def prioritize(
        self, risk_assessments: dict[int, ThreatRiskAssessment]
    ) -> list[ThreatRiskAssessment]:
        """Rank and prioritize threat assessments for the current frame.

        Args:
            risk_assessments: Dict of ThreatRiskAssessment per track ID.

        Returns:
            List of prioritized ThreatRiskAssessment records sorted by priority (rank 1 first).
        """
        if not self.enabled or not risk_assessments:
            return []

        # Categorize into priority tiers
        # Tier 0: Confirmed Active Conflicts (imminent hazard)
        # Tier 1: Potential Conflicts / Transitions
        # Tier 2: Stable / Diverging
        # Tier 3: Unassessed / Insufficient Data
        active_conflict_states = {
            ScenarioState.REAR_END_CONFLICT.value,
            ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value,
            ScenarioState.CYCLIST_CROSSING_CONFLICT.value,
            ScenarioState.CYCLIST_LEAD_CONFLICT.value,
            ScenarioState.CONFIRMED_CUT_IN.value,
            ScenarioState.ANIMAL_CROSSING_CONFLICT.value,
        }

        potential_states = {
            ScenarioState.REAR_END_POTENTIAL.value,
            ScenarioState.POTENTIAL_CUT_IN.value,
            ScenarioState.PEDESTRIAN_APPROACHING_PATH.value,
            ScenarioState.CYCLIST_ENTERING_LANE.value,
            ScenarioState.ADJACENT_LANE_ENCROACHING.value,
            ScenarioState.ANIMAL_APPROACHING_PATH.value,
        }

        def _sort_key(item: tuple[int, ThreatRiskAssessment]) -> tuple:
            tid, ass = item
            is_valid = ass.model_status in (
                ModelEvaluationStatus.VALID_FULL_MODEL,
                ModelEvaluationStatus.VALID_ADAPTED_MODEL,
            )

            # 1. Tier determination (lower integer = higher priority)
            if is_valid and ass.scenario_state in active_conflict_states:
                tier = 0
            elif is_valid and ass.scenario_state in potential_states:
                tier = 1
            elif is_valid:
                tier = 2
            else:
                tier = 3

            # 2. Collision urgency (seconds to collision; lower = higher priority)
            urgency = ass.collision_urgency if (ass.collision_urgency is not None and math.isfinite(ass.collision_urgency)) else 999.0

            # 3. Risk score (inverted: higher score = higher priority)
            neg_score = -float(ass.risk_score) if ass.risk_score is not None else 0.0

            # 4. Reliability preference
            rel_rank = {
                ReliabilityGrade.HIGH: 0,
                ReliabilityGrade.MEDIUM: 1,
                ReliabilityGrade.PROVISIONAL: 2,
                ReliabilityGrade.UNRELIABLE: 3,
            }.get(ass.reliability, 4)

            # 5. Hysteresis bonus for previously primary threat
            hysteresis_penalty = -0.05 if tid == self._prev_primary_id else 0.0

            return (tier, round(urgency, 1), neg_score + hysteresis_penalty, rel_rank, tid)

        sorted_items = sorted(risk_assessments.items(), key=_sort_key)

        ranked_list: list[ThreatRiskAssessment] = []
        new_ranks: dict[int, int] = {}

        for rank_idx, (tid, ass) in enumerate(sorted_items[: self.max_ranked_threats], start=1):
            is_valid = ass.model_status in (
                ModelEvaluationStatus.VALID_FULL_MODEL,
                ModelEvaluationStatus.VALID_ADAPTED_MODEL,
            )

            # Only assign active rank if assessment has valid model or actionable state
            assigned_rank = rank_idx if is_valid else None
            is_primary = (rank_idx == 1) and is_valid and (ass.risk_score is not None and ass.risk_score >= 0.25)

            # Update assessment fields
            ass.priority_rank = assigned_rank
            ass.is_primary_threat = is_primary

            if is_primary:
                self._prev_primary_id = tid

            if assigned_rank is not None:
                new_ranks[tid] = assigned_rank

            ranked_list.append(ass)

        # Update remaining items beyond max_ranked_threats
        for tid, ass in sorted_items[self.max_ranked_threats :]:
            ass.priority_rank = None
            ass.is_primary_threat = False
            ranked_list.append(ass)

        self._prev_ranks = new_ranks
        return ranked_list

