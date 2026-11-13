"""Unit tests for dynamic threat prioritization and ranking.

Covers:
- Multi-tier threat ranking (active conflicts > potential > stable > unassessed)
- Sub-sorting by collision urgency and risk score
- Primary threat assignment (rank 1)
- Model identity retention without cross-model conflation
- Dynamic track additions and removals
- Ranking stability and hysteresis
- PerceptionPipeline integration
"""

import math
import numpy as np
import pytest

from src.core.pipeline import PerceptionPipeline
from src.schemas.bbox import BoundingBox
from src.schemas.frame import Frame
from src.threat.prioritizer import ThreatPrioritizer
from src.threat.schemas import (
    CameraDirection,
    LaneRelationship,
    ModelEvaluationStatus,
    ReliabilityGrade,
    RiskModelType,
    ScenarioState,
    ScenarioType,
    ThreatRiskAssessment,
    ThreatRiskLevel,
)


def _make_risk_assessment(
    track_id: int,
    scenario: ScenarioType = ScenarioType.REAR_END,
    state: str = ScenarioState.REAR_END_CONFLICT.value,
    model_name: RiskModelType = RiskModelType.OPTION_A_ADRS_2025,
    model_status: ModelEvaluationStatus = ModelEvaluationStatus.VALID_FULL_MODEL,
    risk_score: float | None = 0.80,
    collision_urgency: float | None = 2.0,
    reliability: ReliabilityGrade = ReliabilityGrade.HIGH,
) -> ThreatRiskAssessment:
    """Helper to construct ThreatRiskAssessment for prioritization tests."""
    return ThreatRiskAssessment(
        track_id=track_id,
        object_class="car",
        scenario=scenario,
        scenario_state=state,
        direction=CameraDirection.FRONT,
        lane_relationship=LaneRelationship.EGO_LANE,
        model_name=model_name,
        model_status=model_status,
        published_coefficients={"TTC": 0.25, "Spd": 0.24, "AN": 0.21, "DRAC": 0.16, "LD": 0.14},
        raw_indicators={"TTC": collision_urgency},
        normalized_indicators={},
        risk_score=risk_score,
        risk_level=ThreatRiskLevel.HIGH if (risk_score and risk_score >= 0.7) else ThreatRiskLevel.LOW,
        reliability=reliability,
        priority_rank=None,
        is_primary_threat=False,
        collision_urgency=collision_urgency,
        evidence=["TEST_EVALUATION"],
        rejection_reasons=[],
        timestamp=1.0,
    )


class TestThreatPrioritization:
    """Comprehensive test suite for ThreatPrioritizer."""

    def test_multi_tier_prioritization(self):
        """Test active conflict (Tier 0) > potential (Tier 1) > stable (Tier 2) > unassessed (Tier 3)."""
        prioritizer = ThreatPrioritizer()

        # Track 1: Active confirmed cut-in (Tier 0), urgency = 2.5s, score = 0.85
        t1 = _make_risk_assessment(
            track_id=1,
            scenario=ScenarioType.VEHICLE_CUT_IN,
            state=ScenarioState.CONFIRMED_CUT_IN.value,
            risk_score=0.85,
            collision_urgency=2.5,
        )

        # Track 2: Potential rear-end (Tier 1), urgency = 4.0s, score = 0.50
        t2 = _make_risk_assessment(
            track_id=2,
            scenario=ScenarioType.REAR_END,
            state=ScenarioState.REAR_END_POTENTIAL.value,
            risk_score=0.50,
            collision_urgency=4.0,
        )

        # Track 3: Stable adjacent traffic (Tier 2), score = 0.15
        t3 = _make_risk_assessment(
            track_id=3,
            scenario=ScenarioType.ADJACENT_LANE_THREAT,
            state=ScenarioState.ADJACENT_LANE_STABLE.value,
            risk_score=0.15,
            collision_urgency=None,
        )

        # Track 4: Unassessed / insufficient indicators (Tier 3), score = None
        t4 = _make_risk_assessment(
            track_id=4,
            model_status=ModelEvaluationStatus.INSUFFICIENT_INDICATORS,
            risk_score=None,
            collision_urgency=None,
        )

        ranked = prioritizer.prioritize({1: t1, 2: t2, 3: t3, 4: t4})

        # Track 1 should be Rank 1 and primary threat
        assert ranked[0].track_id == 1
        assert ranked[0].priority_rank == 1
        assert ranked[0].is_primary_threat is True

        # Track 2 should be Rank 2
        assert ranked[1].track_id == 2
        assert ranked[1].priority_rank == 2
        assert ranked[1].is_primary_threat is False

        # Track 3 should be Rank 3
        assert ranked[2].track_id == 3
        assert ranked[2].priority_rank == 3

        # Track 4 should have priority_rank = None (unassessed)
        assert ranked[3].track_id == 4
        assert ranked[3].priority_rank is None
        assert ranked[3].is_primary_threat is False

    def test_sub_sorting_by_urgency_and_score(self):
        """Test that within same tier, lower collision urgency (time to impact) takes precedence."""
        prioritizer = ThreatPrioritizer()

        # Track 10: Active rear-end conflict, TTC = 1.2 s, score = 0.90
        t10 = _make_risk_assessment(10, state=ScenarioState.REAR_END_CONFLICT.value, risk_score=0.90, collision_urgency=1.2)
        # Track 11: Active rear-end conflict, TTC = 2.8 s, score = 0.92 (higher score, but less urgent time)
        t11 = _make_risk_assessment(11, state=ScenarioState.REAR_END_CONFLICT.value, risk_score=0.92, collision_urgency=2.8)

        ranked = prioritizer.prioritize({10: t10, 11: t11})

        # Track 10 should be Rank 1 because TTC 1.2s is more critical
        assert ranked[0].track_id == 10
        assert ranked[0].priority_rank == 1
        assert ranked[0].is_primary_threat is True
        assert ranked[1].track_id == 11
        assert ranked[1].priority_rank == 2

    def test_model_identity_preservation_across_scenarios(self):
        """Test that different models retain their distinct identity without conflating scales."""
        prioritizer = ThreatPrioritizer()

        # Track 20 evaluated via Option A ADRS
        t20 = _make_risk_assessment(20, model_name=RiskModelType.OPTION_A_ADRS_2025, risk_score=0.75, collision_urgency=2.0)
        # Track 21 evaluated via Rule-Based Model
        t21 = _make_risk_assessment(21, scenario=ScenarioType.PEDESTRIAN_CROSSING, model_name=RiskModelType.RULE_BASED_ASSESSMENT, risk_score=0.85, collision_urgency=1.8)

        ranked = prioritizer.prioritize({20: t20, 21: t21})
        assert len(ranked) == 2

        # Each assessment preserves its distinct model_name
        lookup = {r.track_id: r for r in ranked}
        assert lookup[20].model_name == RiskModelType.OPTION_A_ADRS_2025
        assert lookup[21].model_name == RiskModelType.RULE_BASED_ASSESSMENT

    def test_dynamic_track_updates_and_hysteresis(self):
        """Test dynamic threat insertion, removal, and primary threat hysteresis."""
        prioritizer = ThreatPrioritizer()

        # Frame 1: Track 30 is primary (score = 0.80, urgency = 2.0s)
        t30 = _make_risk_assessment(30, risk_score=0.80, collision_urgency=2.0)
        res1 = prioritizer.prioritize({30: t30})
        assert res1[0].track_id == 30
        assert res1[0].is_primary_threat is True

        # Frame 2: New track 31 enters with marginal difference (same urgency, score = 0.82)
        # Hysteresis prevents rapid rank flapping
        t30_f2 = _make_risk_assessment(30, risk_score=0.80, collision_urgency=2.0)
        t31_f2 = _make_risk_assessment(31, risk_score=0.82, collision_urgency=2.0)

        res2 = prioritizer.prioritize({30: t30_f2, 31: t31_f2})
        assert res2[0].track_id == 30  # Retains top rank due to hysteresis

        # Frame 3: Track 31 becomes significantly more critical (urgency = 0.8s)
        t31_f3 = _make_risk_assessment(31, risk_score=0.95, collision_urgency=0.8)
        res3 = prioritizer.prioritize({30: t30_f2, 31: t31_f3})
        assert res3[0].track_id == 31  # Overtakes primary threat decisively
        assert res3[0].is_primary_threat is True

    def test_pipeline_integration(self):
        """Test that PerceptionPipeline exposes threat_assessments, prioritized_threats, and helper getters."""
        pipeline = PerceptionPipeline(config={})
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(image=dummy_img, frame_id=1, timestamp=0.033, source="test")

        result = pipeline.process(frame)
        assert "threat_assessments" in result
        assert "prioritized_threats" in result
        assert isinstance(result["threat_assessments"], dict)
        assert isinstance(result["prioritized_threats"], list)
        assert pipeline.get_threat_assessments() == result["threat_assessments"]
        assert pipeline.get_prioritized_threats() == result["prioritized_threats"]

