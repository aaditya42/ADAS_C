"""Unit tests for threat risk assessment models and indicator normalization.

Covers:
- Option A (ADRS 2025) published coefficients and calculation
- Option A missing indicator handling without value fabrication or weight redistribution
- Option A scenario scope rejection
- Option B (Composite Intersection 2025) published coefficients
- Option B unavailable PET handling (must not set PET=0 or renormalize weights)
- SSAM diagnostic conflict threshold gating
- Rule-based adapted model labeling
- Indicator normalizer bounds, risk directions, and non-finite handling
"""

import math
import pytest

from src.threat.models.option_a import OptionARiskModel
from src.threat.models.option_b import OptionBRiskModel
from src.threat.models.rule_based import RuleBasedRiskModel
from src.threat.models.ssam import SSAMDiagnosticModel
from src.threat.normalization import IndicatorNormalizer
from src.threat.risk_engine import ScenarioRiskEngine
from src.threat.schemas import (
    CameraDirection,
    IndicatorValidity,
    LaneRelationship,
    ModelEvaluationStatus,
    ReliabilityGrade,
    RiskModelType,
    ScenarioAssessment,
    ScenarioState,
    ScenarioType,
    ThreatFeatures,
    ThreatRiskLevel,
    TrajectoryConflictState,
    ValidityStatus,
    VulnerabilityLevel,
)
from src.schemas.bbox import BoundingBox


def _make_features(
    track_id: int = 1,
    object_class: str = "car",
    ttc: float | None = 2.5,
    drac: float | None = 1.5,
    speed_var: float | None = 0.6,
    accel_noise: float | None = 0.4,
    lat_dev: float | None = 0.2,
    pet: float | None = None,
    is_metric: bool = True,
    traj_conflict: TrajectoryConflictState = TrajectoryConflictState.CONFLICT,
    lane_overlap: float = 0.95,
) -> ThreatFeatures:
    """Helper to construct ThreatFeatures for risk model tests."""
    ind_v = {
        "ttc": IndicatorValidity(ValidityStatus.VALID_METRIC if is_metric else ValidityStatus.VALID_PROXY, ttc),
        "drac": IndicatorValidity(ValidityStatus.VALID_METRIC if (is_metric and drac is not None) else ValidityStatus.UNAVAILABLE, drac),
        "relative_motion": IndicatorValidity(ValidityStatus.VALID_METRIC if is_metric else ValidityStatus.VALID_PROXY, 5.0),
        "lane_overlap": IndicatorValidity(ValidityStatus.VALID_METRIC if is_metric else ValidityStatus.VALID_PROXY, lane_overlap),
        "pet": IndicatorValidity(ValidityStatus.UNAVAILABLE, pet),
    }
    return ThreatFeatures(
        track_id=track_id,
        tracker_id=track_id,
        object_class=object_class,
        bbox=BoundingBox(500, 400, 700, 600),
        timestamp=1.0,
        direction=CameraDirection.FRONT,
        lane_relationship=LaneRelationship.EGO_LANE,
        lane_overlap=lane_overlap,
        lateral_deviation=lat_dev,
        ttc=ttc,
        pet=pet,
        drac=drac,
        relative_speed=5.0,
        relative_acceleration=-0.5,
        motion_direction="approaching",
        speed_variation=speed_var,
        acceleration_noise=accel_noise,
        looming_rate=0.2,
        trajectory_conflict=traj_conflict,
        trajectory_conflict_time=1.5 if traj_conflict == TrajectoryConflictState.CONFLICT else None,
        vulnerability=VulnerabilityLevel.STANDARD,
        crossing_probability=None,
        behavior_label=None,
        observation_duration=1.5,
        observation_count=15,
        is_metric=is_metric,
        indicator_validity=ind_v,
    )


def _make_assessment(
    track_id: int = 1,
    scenario: ScenarioType = ScenarioType.REAR_END,
    state: str = ScenarioState.REAR_END_CONFLICT.value,
) -> ScenarioAssessment:
    """Helper to construct ScenarioAssessment for testing."""
    return ScenarioAssessment(
        track_id=track_id,
        scenario=scenario,
        state=state,
        confidence=0.92,
        direction=CameraDirection.FRONT,
        lane_relationship=LaneRelationship.EGO_LANE,
        evidence=["IN_EGO_LANE_AHEAD", "RAPID_CLOSING_VELOCITY"],
        timestamp=1.0,
    )


class TestThreatRiskModels:
    """Comprehensive test suite for published and adapted risk models."""

    def test_indicator_normalizer_ttc_bounds_and_direction(self):
        """Test TTC normalization: lower TTC -> higher risk, bounded in [0, 1]."""
        normalizer = IndicatorNormalizer()

        # TTC <= 1.0 s -> normalized = 1.0 (maximum risk)
        n1 = normalizer.normalize_ttc(0.8)
        assert n1.normalized_value == 1.0
        assert n1.risk_direction == "INVERSE"

        # TTC >= 6.0 s -> normalized = 0.0 (safe)
        n2 = normalizer.normalize_ttc(6.5)
        assert n2.normalized_value == 0.0

        # Midpoint: TTC = 3.5 s in [1.0, 6.0] -> (6.0 - 3.5) / 5.0 = 0.5
        n3 = normalizer.normalize_ttc(3.5)
        assert abs(n3.normalized_value - 0.5) < 1e-4

        # Unavailable / negative TTC
        n4 = normalizer.normalize_ttc(-2.0)
        assert n4.normalized_value is None
        assert n4.validity == ValidityStatus.UNAVAILABLE

    def test_indicator_normalizer_direct_indicators(self):
        """Test direct risk indicators: Spd, AN, DRAC, LD (higher value -> higher risk)."""
        normalizer = IndicatorNormalizer()

        # DRAC: [0.0, 5.0] m/s^2
        ndrac = normalizer.normalize_drac(2.5)
        assert abs(ndrac.normalized_value - 0.5) < 1e-4
        assert ndrac.risk_direction == "DIRECT"

        # Speed std dev: [0.0, 3.0] m/s
        nspd = normalizer.normalize_spd(1.5)
        assert abs(nspd.normalized_value - 0.5) < 1e-4

        # Acceleration noise: [0.0, 2.0] m/s^2
        nan = normalizer.normalize_an(1.0)
        assert abs(nan.normalized_value - 0.5) < 1e-4

        # Lateral deviation: [0.0, 1.0]
        nld = normalizer.normalize_ld(0.4)
        assert abs(nld.normalized_value - 0.4) < 1e-4

    def test_option_a_published_coefficients_and_full_evaluation(self):
        """Test Option A published AHP weights and ADRS score computation."""
        model = OptionARiskModel()
        normalizer = IndicatorNormalizer()

        # Check published coefficients strictly
        assert model.published_coefficients == {
            "TTC": 0.25,
            "Spd": 0.24,
            "AN": 0.21,
            "DRAC": 0.16,
            "LD": 0.14,
        }
        assert abs(sum(model.published_coefficients.values()) - 1.00) < 1e-6

        # Setup features with known values:
        # TTC = 3.5s -> TTC_N = 0.50
        # Spd = 1.5 m/s -> Spd_N = 0.50
        # AN = 1.0 m/s^2 -> AN_N = 0.50
        # DRAC = 2.5 m/s^2 -> DRAC_N = 0.50
        # LD = 0.50 -> LD_N = 0.50
        # Expected ADRS = 0.50 * (0.25 + 0.24 + 0.21 + 0.16 + 0.14) = 0.50
        feats = _make_features(ttc=3.5, drac=2.5, speed_var=1.5, accel_noise=1.0, lat_dev=0.5, is_metric=True)
        ass = _make_assessment(scenario=ScenarioType.REAR_END)

        result = model.evaluate(feats, ass, normalizer)
        assert result.model_status == ModelEvaluationStatus.VALID_FULL_MODEL
        assert result.risk_score is not None
        assert abs(result.risk_score - 0.50) < 1e-4
        assert result.risk_level == ThreatRiskLevel.MEDIUM
        assert result.reliability == ReliabilityGrade.HIGH
        assert result.model_name == RiskModelType.OPTION_A_ADRS_2025

    def test_option_a_missing_indicator_no_fabrication_or_redistribution(self):
        """Test that Option A reports INSUFFICIENT_INDICATORS if DRAC is missing (e.g. uncalibrated relative depth)."""
        model = OptionARiskModel()
        normalizer = IndicatorNormalizer()

        # DRAC is None (e.g. relative depth mode)
        feats = _make_features(ttc=2.0, drac=None, is_metric=False)
        ass = _make_assessment(scenario=ScenarioType.REAR_END)

        result = model.evaluate(feats, ass, normalizer)
        assert result.model_status == ModelEvaluationStatus.INSUFFICIENT_INDICATORS
        assert result.risk_score is None
        assert result.risk_level == ThreatRiskLevel.UNASSESSED
        assert any("DRAC" in r for r in result.rejection_reasons)
        assert any("not redistributed" in r for r in result.rejection_reasons)

    def test_option_a_unsupported_scenario_rejection(self):
        """Test that Option A rejects crossing pedestrians or cyclists as UNSUPPORTED_SCENARIO."""
        model = OptionARiskModel()
        normalizer = IndicatorNormalizer()

        feats = _make_features(object_class="person")
        ass = _make_assessment(scenario=ScenarioType.PEDESTRIAN_CROSSING)

        result = model.evaluate(feats, ass, normalizer)
        assert result.model_status == ModelEvaluationStatus.UNSUPPORTED_SCENARIO
        assert result.risk_score is None
        assert result.risk_level == ThreatRiskLevel.UNASSESSED
        assert "Option A (ADRS 2025) is formulated for mixed-traffic car-following" in result.rejection_reasons[0]

    def test_option_b_published_coefficients_and_pet_unavailability(self):
        """Test Option B published weights and verified rejection when PET is unavailable."""
        model = OptionBRiskModel()
        normalizer = IndicatorNormalizer()

        # Check published coefficients
        assert model.published_coefficients == {
            "PET": 0.30,
            "TTC": 0.30,
            "V_comb": 0.20,
            "DRAC": 0.10,
            "ROW": 0.05,
            "UB": 0.05,
        }
        assert abs(sum(model.published_coefficients.values()) - 1.00) < 1e-6

        # In monocular vision, PET is unavailable
        feats = _make_features(object_class="bicycle", pet=None)
        ass = _make_assessment(scenario=ScenarioType.CYCLIST_CONFLICT)

        result = model.evaluate(feats, ass, normalizer)
        assert result.model_status == ModelEvaluationStatus.INSUFFICIENT_INDICATORS
        assert result.risk_score is None
        assert result.risk_level == ThreatRiskLevel.UNASSESSED
        assert any("PET" in r for r in result.rejection_reasons)
        assert any("not redistributed" in r for r in result.rejection_reasons)

    def test_option_b_unsupported_scenario_rejection(self):
        """Test that Option B rejects highway car rear-end or pedestrian interactions."""
        model = OptionBRiskModel()
        normalizer = IndicatorNormalizer()

        feats = _make_features(object_class="car")
        ass = _make_assessment(scenario=ScenarioType.REAR_END)

        result = model.evaluate(feats, ass, normalizer)
        assert result.model_status == ModelEvaluationStatus.UNSUPPORTED_SCENARIO
        assert result.risk_score is None

    def test_ssam_diagnostic_model(self):
        """Test SSAM diagnostic conflict threshold gating without arbitrary linear weights."""
        model = SSAMDiagnosticModel()
        normalizer = IndicatorNormalizer()

        # Critical TTC < 1.5s
        feats_crit = _make_features(ttc=1.2, drac=2.0)
        ass = _make_assessment()
        res_crit = model.evaluate(feats_crit, ass, normalizer)
        assert res_crit.model_status == ModelEvaluationStatus.VALID_FULL_MODEL
        assert res_crit.risk_score >= 0.85
        assert "SSAM_CRITICAL_CONFLICT_TTC_SUB_1_5S" in res_crit.evidence
        assert res_crit.risk_level == ThreatRiskLevel.HIGH

        # Safe TTC > 6.0s
        feats_safe = _make_features(ttc=7.5, drac=0.2)
        res_safe = model.evaluate(feats_safe, ass, normalizer)
        assert res_safe.risk_score <= 0.20
        assert "SSAM_NO_SIGNIFICANT_CONFLICT" in res_safe.evidence

    def test_rule_based_risk_model_distinct_labeling(self):
        """Test that adapted rule-based models are explicitly labeled as VALID_ADAPTED_MODEL."""
        model = RuleBasedRiskModel()
        normalizer = IndicatorNormalizer()

        # Pedestrian crossing conflict
        feats_ped = _make_features(object_class="person", ttc=2.0, lane_overlap=0.4)
        feats_ped.crossing_probability = 0.80
        ass_ped = _make_assessment(scenario=ScenarioType.PEDESTRIAN_CROSSING, state=ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value)

        res_ped = model.evaluate(feats_ped, ass_ped, normalizer)
        assert res_ped.model_name == RiskModelType.RULE_BASED_ASSESSMENT
        assert res_ped.model_status == ModelEvaluationStatus.VALID_ADAPTED_MODEL
        assert res_ped.risk_score >= 0.70
        assert res_ped.risk_level == ThreatRiskLevel.HIGH

    def test_scenario_risk_engine_routing(self):
        """Test ScenarioRiskEngine routes scenarios and handles fallback adapted models."""
        engine = ScenarioRiskEngine({"threat_assessment": {"enabled": True, "risk_scoring": {"allow_adapted_fallback": True}}})

        # Car rear-end with metric measurements -> Option A
        feats_car = _make_features(track_id=1, object_class="car", ttc=3.0, drac=2.0, is_metric=True)
        ass_car = _make_assessment(track_id=1, scenario=ScenarioType.REAR_END)

        evals = engine.evaluate({1: feats_car}, {1: ass_car})
        assert 1 in evals
        assert evals[1].model_name == RiskModelType.OPTION_A_ADRS_2025
        assert evals[1].model_status == ModelEvaluationStatus.VALID_FULL_MODEL

        # Car rear-end under relative uncalibrated depth -> Option A lacks DRAC -> adapted fallback evaluated
        feats_proxy = _make_features(track_id=2, object_class="car", ttc=3.0, drac=None, is_metric=False)
        ass_proxy = _make_assessment(track_id=2, scenario=ScenarioType.REAR_END)

        evals_proxy = engine.evaluate({2: feats_proxy}, {2: ass_proxy})
        assert 2 in evals_proxy
        assert evals_proxy[2].model_name == RiskModelType.RULE_BASED_ASSESSMENT
        assert evals_proxy[2].model_status == ModelEvaluationStatus.VALID_ADAPTED_MODEL
        assert "ADAPTED_FALLBACK_EVALUATION" in evals_proxy[2].evidence

