"""Mathematical verification tests for threat risk assessment models against published literature.

Verifies:
1. Option A (2025 Comprehensive) exact published coefficients (TTC:0.25, Spd:0.24, AN:0.21, DRAC:0.16, LD:0.14 = 1.00).
2. Option B (2025 Direct Linear) exact published coefficients (PET:0.30, TTC:0.30, V_comb:0.20, DRAC:0.10, ROW:0.05, UB:0.05 = 1.00).
3. SSAM (FHWA) diagnostic conflict thresholds (TTC <= 1.5s, PET <= 5.0s, DRAC >= 3.4 m/s²).
4. Normalization bounds and monotonicity.
5. Boundary conditions, missing values, and zero-fabrication safety.
"""

from src.threat.models.option_a import OptionARiskModel
from src.threat.models.option_b import OptionBRiskModel
from src.threat.models.ssam import SSAMDiagnosticModel
from src.threat.normalization import IndicatorNormalizer
from src.threat.schemas import (
    CameraDirection,
    IndicatorValidity,
    LaneRelationship,
    ModelEvaluationStatus,
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
    """Helper to construct ThreatFeatures for math verification."""
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


class TestThreatMathVerification:
    """Rigorous verification of mathematical formulas against published sources."""

    def test_option_a_coefficients_sum_to_unity(self):
        """Option A (ADRS 2025) published coefficients must sum strictly to 1.0."""
        model = OptionARiskModel()
        coeffs = model.published_coefficients

        assert coeffs["TTC"] == 0.25
        assert coeffs["Spd"] == 0.24
        assert coeffs["AN"] == 0.21
        assert coeffs["DRAC"] == 0.16
        assert coeffs["LD"] == 0.14

        total = sum(coeffs.values())
        assert abs(total - 1.00) < 1e-9

    def test_option_a_hand_calculated_scenario(self):
        """Verify Option A exact weighted calculation against hand-computed values.

        Given normalized indicators:
        - TTC = 3.5 s -> norm = (6.0 - 3.5)/5.0 = 0.500
        - Spd = 1.5 m/s -> norm = 1.5/3.0 = 0.500
        - AN = 1.0 m/s^2 -> norm = 1.0/2.0 = 0.500
        - DRAC = 2.5 m/s^2 -> norm = 2.5/5.0 = 0.500
        - LD = 0.5 -> norm = 0.500

        Hand Calculation:
          0.25 * 0.500 = 0.1250
        + 0.24 * 0.500 = 0.1200
        + 0.21 * 0.500 = 0.1050
        + 0.16 * 0.500 = 0.0800
        + 0.14 * 0.500 = 0.0700
        -----------------------
        Total Score    = 0.5000
        """
        model = OptionARiskModel()
        normalizer = IndicatorNormalizer()

        features = _make_features(
            ttc=3.5,
            speed_var=1.5,
            accel_noise=1.0,
            drac=2.5,
            lat_dev=0.5,
            is_metric=True,
        )
        scenario = _make_assessment(scenario=ScenarioType.REAR_END)

        assessment = model.evaluate(features, scenario, normalizer)
        assert assessment.model_status == ModelEvaluationStatus.VALID_FULL_MODEL
        assert assessment.risk_score is not None
        assert abs(assessment.risk_score - 0.500) < 1e-3
        assert assessment.risk_level == ThreatRiskLevel.MEDIUM

    def test_option_b_coefficients_sum_to_unity(self):
        """Option B (Composite Intersection 2025) published coefficients must sum strictly to 1.0."""
        model = OptionBRiskModel()
        coeffs = model.published_coefficients

        assert coeffs["PET"] == 0.30
        assert coeffs["TTC"] == 0.30
        assert coeffs["V_comb"] == 0.20
        assert coeffs["DRAC"] == 0.10
        assert coeffs["ROW"] == 0.05
        assert coeffs["UB"] == 0.05

        total = sum(coeffs.values())
        assert abs(total - 1.00) < 1e-9

    def test_ssam_diagnostic_thresholds(self):
        """Verify FHWA SSAM conflict severity categorization."""
        model = SSAMDiagnosticModel()
        normalizer = IndicatorNormalizer()

        # Critical SSAM Conflict: TTC <= 1.5s
        critical_features = _make_features(ttc=1.2, drac=3.5)
        crit_ass = model.evaluate(critical_features, _make_assessment(), normalizer)
        assert crit_ass.model_status == ModelEvaluationStatus.VALID_FULL_MODEL
        assert crit_ass.risk_level == ThreatRiskLevel.HIGH
        assert any("SSAM_CRITICAL_CONFLICT_TTC_SUB_1_5S" in ev for ev in crit_ass.evidence)

        # Safe SSAM: TTC > 6.0s
        safe_features = _make_features(ttc=7.5, drac=0.2)
        safe_ass = model.evaluate(safe_features, _make_assessment(), normalizer)
        assert safe_ass.risk_level == ThreatRiskLevel.LOW
        assert safe_ass.risk_score <= 0.20

    def test_zero_fabrication_on_missing_indicators(self):
        """Missing indicators must never be fabricated as 0.0 or 1.0.

        If DRAC is missing in Option A, model must mark status as INSUFFICIENT_INDICATORS
        rather than fabricating arbitrary scores.
        """
        model = OptionARiskModel()
        normalizer = IndicatorNormalizer()

        empty_features = _make_features(drac=None)
        ass = _make_assessment(scenario=ScenarioType.REAR_END)

        res = model.evaluate(empty_features, ass, normalizer)
        assert res.model_status == ModelEvaluationStatus.INSUFFICIENT_INDICATORS
        assert res.risk_score is None
        assert any("DRAC" in r for r in res.rejection_reasons)

    def test_indicator_normalizer_monotonicity_and_bounds(self):
        """Verify that normalized indicators are strictly monotonic and bounded in [0.0, 1.0]."""
        normalizer = IndicatorNormalizer()

        # TTC normalization is inverse: lower TTC = higher risk
        norm_close = normalizer.normalize_ttc(1.5)
        norm_far = normalizer.normalize_ttc(4.5)
        assert 0.0 <= norm_close.normalized_value <= 1.0
        assert 0.0 <= norm_far.normalized_value <= 1.0
        assert norm_close.normalized_value > norm_far.normalized_value

        # DRAC normalization is direct: higher deceleration requirement = higher risk
        norm_drac_high = normalizer.normalize_drac(4.0)
        norm_drac_low = normalizer.normalize_drac(1.0)
        assert 0.0 <= norm_drac_high.normalized_value <= 1.0
        assert 0.0 <= norm_drac_low.normalized_value <= 1.0
        assert norm_drac_high.normalized_value > norm_drac_low.normalized_value
