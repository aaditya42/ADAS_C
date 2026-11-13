"""Unit tests for DirectionalWarningGenerator and WarningEvent lifecycle management.

Covers:
- Warning-level decisions: HIGH, MEDIUM, LOW, and UNASSESSED without fabrication.
- Fast-track bypass for critical imminent hazards (TTC <= 1.5s).
- Single-frame false-alarm rejection (temporal confirmation buffer).
- Escalation hysteresis (LOW -> MEDIUM -> HIGH).
- De-escalation hysteresis requiring sustained lower risk (0.5s) to avoid flicker.
- Duplicate warning suppression (one persistent event per track & scenario).
- Track loss grace period (0.4s) and stale expiration (2.0s).
- Tracker ID reuse detection with clean event resolution.
- Simultaneous multi-threat independence across distinct directions and classes.
- Full PerceptionPipeline integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.pipeline import PerceptionPipeline
from src.schemas.bbox import BoundingBox
from src.schemas.frame import Frame
from src.threat.direction import DirectionEstimator
from src.threat.schemas import (
    CameraDirection,
    DirectionalWarningLevel,
    IndicatorValidity,
    LaneRelationship,
    ModelEvaluationStatus,
    ReliabilityGrade,
    RiskModelType,
    ScenarioAssessment,
    ScenarioState,
    ScenarioType,
    ThreatFeatures,
    ThreatRiskAssessment,
    ThreatRiskLevel,
    TrajectoryConflictState,
    ValidityStatus,
    VulnerabilityLevel,
    WarningEvent,
    WarningEventState,
    WarningOutput,
)
from src.threat.warning_generator import DirectionalWarningGenerator


def _make_dummy_frame(timestamp: float = 0.0, frame_id: int = 1) -> Frame:
    """Helper to create dummy video Frame."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=frame_id, timestamp=timestamp, source="test")


def _make_threat_data(
    track_id: int,
    object_class: str = "car",
    scenario: ScenarioType = ScenarioType.REAR_END,
    scenario_state: str = ScenarioState.REAR_END_CONFLICT.value,
    risk_score: float | None = 0.75,
    ttc: float | None = 2.0,
    relative_speed: float | None = 3.0,
    direction: CameraDirection = CameraDirection.FRONT,
    bbox: BoundingBox | None = None,
    lane_relationship: LaneRelationship = LaneRelationship.EGO_LANE,
    model_status: ModelEvaluationStatus = ModelEvaluationStatus.VALID_FULL_MODEL,
    reliability: ReliabilityGrade = ReliabilityGrade.HIGH,
    timestamp: float = 1.0,
) -> tuple[ThreatRiskAssessment, ThreatFeatures]:
    """Helper to construct paired ThreatRiskAssessment and ThreatFeatures."""
    if bbox is None:
        bbox = BoundingBox(x1=580.0, y1=350.0, x2=700.0, y2=490.0)

    ind_v = {
        "ttc": IndicatorValidity(ValidityStatus.VALID_METRIC, ttc),
        "drac": IndicatorValidity(ValidityStatus.VALID_METRIC, 2.2),
        "relative_motion": IndicatorValidity(ValidityStatus.VALID_METRIC, relative_speed),
        "lane_overlap": IndicatorValidity(ValidityStatus.VALID_METRIC, 0.85 if lane_relationship == LaneRelationship.EGO_LANE else 0.0),
        "pet": IndicatorValidity(ValidityStatus.UNAVAILABLE, None),
    }

    feat = ThreatFeatures(
        track_id=track_id,
        tracker_id=track_id,
        object_class=object_class,
        bbox=bbox,
        timestamp=timestamp,
        direction=direction,
        lane_relationship=lane_relationship,
        lane_overlap=0.85 if lane_relationship == LaneRelationship.EGO_LANE else 0.0,
        lateral_deviation=0.0,
        ttc=ttc,
        pet=None,
        drac=2.2,
        relative_speed=relative_speed,
        relative_acceleration=-0.5,
        motion_direction="approaching" if (relative_speed and relative_speed > 0) else "steady",
        speed_variation=0.5,
        acceleration_noise=0.3,
        looming_rate=0.2,
        trajectory_conflict=TrajectoryConflictState.CONFLICT,
        trajectory_conflict_time=ttc,
        vulnerability=VulnerabilityLevel.STANDARD,
        crossing_probability=None,
        behavior_label=None,
        observation_duration=1.5,
        observation_count=15,
        is_metric=True,
        indicator_validity=ind_v,
    )

    ass = ThreatRiskAssessment(
        track_id=track_id,
        object_class=object_class,
        scenario=scenario,
        scenario_state=scenario_state,
        direction=direction,
        lane_relationship=lane_relationship,
        model_name=RiskModelType.OPTION_A_ADRS_2025,
        model_status=model_status,
        published_coefficients={"TTC": 0.25, "Spd": 0.24, "AN": 0.21, "DRAC": 0.16, "LD": 0.14},
        raw_indicators={"TTC": ttc},
        normalized_indicators={},
        risk_score=risk_score,
        risk_level=ThreatRiskLevel.HIGH if (risk_score and risk_score >= 0.7) else ThreatRiskLevel.LOW,
        reliability=reliability,
        priority_rank=1,
        is_primary_threat=True,
        collision_urgency=ttc,
        evidence=["TEST_ALERT"],
        rejection_reasons=[],
        timestamp=timestamp,
    )

    return ass, feat


class TestDirectionalWarningGenerator:
    """Comprehensive test suite for DirectionalWarningGenerator."""

    def test_warning_level_decisions(self):
        """Test HIGH, MEDIUM, LOW, and UNASSESSED candidate level determination."""
        generator = DirectionalWarningGenerator()

        # HIGH: score 0.75, active conflict
        ass_h, feat_h = _make_threat_data(track_id=1, risk_score=0.75)
        lvl_h, _ = generator._determine_candidate_level(ass_h, feat_h)
        assert lvl_h == DirectionalWarningLevel.HIGH

        # MEDIUM: score 0.52
        ass_m, feat_m = _make_threat_data(
            track_id=2,
            risk_score=0.52,
            scenario_state=ScenarioState.REAR_END_POTENTIAL.value,
        )
        lvl_m, _ = generator._determine_candidate_level(ass_m, feat_m)
        assert lvl_m == DirectionalWarningLevel.MEDIUM

        # LOW: score 0.28, adjacent lane encroaching
        ass_l, feat_l = _make_threat_data(
            track_id=3,
            risk_score=0.28,
            scenario=ScenarioType.ADJACENT_LANE_THREAT,
            scenario_state=ScenarioState.ADJACENT_LANE_ENCROACHING.value,
            lane_relationship=LaneRelationship.LEFT_ADJACENT_LANE,
        )
        lvl_l, _ = generator._determine_candidate_level(ass_l, feat_l)
        assert lvl_l == DirectionalWarningLevel.LOW

        # UNASSESSED: missing indicators or score None
        ass_u, feat_u = _make_threat_data(
            track_id=4,
            risk_score=None,
            model_status=ModelEvaluationStatus.INSUFFICIENT_INDICATORS,
        )
        lvl_u, _ = generator._determine_candidate_level(ass_u, feat_u)
        assert lvl_u == DirectionalWarningLevel.UNASSESSED

    def test_flicker_rejection_temporal_confirmation(self):
        """Single-frame risk spike does not trigger warning; second frame confirms."""
        generator = DirectionalWarningGenerator()
        ass, feat = _make_threat_data(track_id=1, risk_score=0.55, ttc=2.5, timestamp=1.0)
        f1 = _make_dummy_frame(timestamp=1.0)

        # Frame 1: Candidate buffered, no active output
        out1 = generator.process({1: ass}, {1: feat}, frame=f1)
        assert len(out1) == 0
        assert len(generator.get_active_events()) == 0

        # Frame 2: Sustained at t=1.2s -> Confirmed and Activated
        ass.timestamp = 1.2
        feat.timestamp = 1.2
        f2 = _make_dummy_frame(timestamp=1.2)
        out2 = generator.process({1: ass}, {1: feat}, frame=f2)
        assert len(out2) == 1
        warn = out2[0]
        assert warn.level == DirectionalWarningLevel.MEDIUM
        assert warn.state == WarningEventState.ACTIVE_MEDIUM
        assert warn.direction == CameraDirection.FRONT

        # Verify event timestamps
        events = generator.get_all_events()
        assert len(events) == 1
        assert events[0].first_detected_timestamp == 1.0
        assert events[0].activation_timestamp == 1.2

    def test_fast_track_escalation_for_critical_hazard(self):
        """Imminent critical hazard (TTC <= 1.5s, HIGH level) activates on frame 1 without delay."""
        generator = DirectionalWarningGenerator()
        ass, feat = _make_threat_data(track_id=5, risk_score=0.88, ttc=1.2, timestamp=2.0)
        feat.lane_overlap = 0.50
        f1 = _make_dummy_frame(timestamp=2.0)

        out = generator.process({5: ass}, {5: feat}, frame=f1)
        assert len(out) == 1
        assert out[0].level == DirectionalWarningLevel.HIGH
        assert out[0].state == WarningEventState.ACTIVE_HIGH
        assert out[0].event_id.startswith("warn_rear_end_5_")

    def test_escalation_and_deescalation_hysteresis(self):
        """Test escalation (LOW -> MEDIUM -> HIGH) and delayed de-escalation (HIGH -> MEDIUM)."""
        generator = DirectionalWarningGenerator()

        # Step 1: Start at LOW level at t=0.0s and t=0.2s
        ass_l, feat_l = _make_threat_data(
            track_id=10,
            risk_score=0.30,
            scenario_state="STEADY_FOLLOWING",
            ttc=4.0,
            timestamp=0.0,
        )
        generator.process({10: ass_l}, {10: feat_l}, frame=_make_dummy_frame(timestamp=0.0))
        ass_l.timestamp = 0.2
        feat_l.timestamp = 0.2
        out_init = generator.process({10: ass_l}, {10: feat_l}, frame=_make_dummy_frame(timestamp=0.2))
        assert len(out_init) == 1
        assert out_init[0].level == DirectionalWarningLevel.LOW

        # Step 2: Escalate to HIGH level at t=0.3s (2 frames confirmation)
        ass_h, feat_h = _make_threat_data(track_id=10, risk_score=0.85, ttc=1.8, timestamp=0.3)
        generator.process({10: ass_h}, {10: feat_h}, frame=_make_dummy_frame(timestamp=0.3))
        ass_h.timestamp = 0.4
        feat_h.timestamp = 0.4
        out_esc = generator.process({10: ass_h}, {10: feat_h}, frame=_make_dummy_frame(timestamp=0.4))
        assert out_esc[0].level == DirectionalWarningLevel.HIGH
        assert out_esc[0].escalated is True

        # Step 3: De-escalation attempt: risk drops to MEDIUM at t=0.5s
        ass_m, feat_m = _make_threat_data(track_id=10, risk_score=0.50, ttc=3.2, timestamp=0.5)
        out_deesc1 = generator.process({10: ass_m}, {10: feat_m}, frame=_make_dummy_frame(timestamp=0.5))
        # Hysteresis must hold warning at HIGH (requires 0.50s sustained lower risk)
        assert out_deesc1[0].level == DirectionalWarningLevel.HIGH
        assert out_deesc1[0].deescalated is False

        # Step 4: Sustained lower risk at t=1.1s (delta = 0.60s >= 0.50s)
        ass_m.timestamp = 1.1
        feat_m.timestamp = 1.1
        out_deesc2 = generator.process({10: ass_m}, {10: feat_m}, frame=_make_dummy_frame(timestamp=1.1))
        # Now de-escalates to MEDIUM
        assert out_deesc2[0].level == DirectionalWarningLevel.MEDIUM
        assert out_deesc2[0].deescalated is True

    def test_duplicate_suppression_and_single_event_id(self):
        """Continuous frames update the same WarningEvent without creating duplicate entries."""
        generator = DirectionalWarningGenerator()
        ass, feat = _make_threat_data(track_id=12, risk_score=0.60, ttc=2.8, timestamp=1.0)

        # Activate event over 2 frames
        generator.process({12: ass}, {12: feat}, frame=_make_dummy_frame(timestamp=1.0))
        ass.timestamp = 1.2
        feat.timestamp = 1.2
        out = generator.process({12: ass}, {12: feat}, frame=_make_dummy_frame(timestamp=1.2))
        event_id = out[0].event_id

        # Feed 10 consecutive frames
        for t in np.linspace(1.3, 2.5, 10):
            ass.timestamp = float(t)
            feat.timestamp = float(t)
            curr_out = generator.process({12: ass}, {12: feat}, frame=_make_dummy_frame(timestamp=float(t)))
            assert len(curr_out) == 1
            assert curr_out[0].event_id == event_id

        # Total events logged must remain exactly 1
        assert len(generator.get_all_events()) == 1

    def test_track_loss_grace_and_stale_expiration(self):
        """Temporary track loss (<=0.4s) retains active warning; stale (>2.0s) expires."""
        generator = DirectionalWarningGenerator()
        ass, feat = _make_threat_data(track_id=15, risk_score=0.75, ttc=1.4, timestamp=1.0)

        # Frame 1: fast-track active
        generator.process({15: ass}, {15: feat}, frame=_make_dummy_frame(timestamp=1.0))
        assert len(generator.get_active_events()) == 1

        # Track momentarily disappears at t=1.2s (delta=0.2s <= 0.4s grace)
        out_grace = generator.process({}, {}, frame=_make_dummy_frame(timestamp=1.2))
        assert len(out_grace) == 1
        assert out_grace[0].reason == "TEMPORARY_TRACK_LOSS_GRACE"

        # Track still missing at t=3.5s (delta=2.5s > 2.0s stale timeout)
        generator.process({}, {}, frame=_make_dummy_frame(timestamp=3.5))
        assert len(generator.get_active_events()) == 0
        all_ev = generator.get_all_events()
        assert len(all_ev) == 1
        assert all_ev[0].state == WarningEventState.EXPIRED

    def test_tracker_id_reuse_detection(self):
        """Reusing track ID with different object class resolves old event and starts fresh."""
        generator = DirectionalWarningGenerator()

        # Step 1: Track 20 is a car
        ass_car, feat_car = _make_threat_data(
            track_id=20, object_class="car", risk_score=0.85, ttc=1.2, timestamp=1.0
        )
        generator.process({20: ass_car}, {20: feat_car}, frame=_make_dummy_frame(timestamp=1.0))
        events1 = generator.get_all_events()
        assert len(events1) == 1
        assert events1[0].object_class == "car"

        # Step 2: Track 20 suddenly re-appears as a pedestrian at t=2.0s
        ass_ped, feat_ped = _make_threat_data(
            track_id=20,
            object_class="pedestrian",
            scenario=ScenarioType.PEDESTRIAN_CROSSING,
            scenario_state=ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value,
            risk_score=0.90,
            ttc=1.1,
            timestamp=2.0,
        )
        generator.process({20: ass_ped}, {20: feat_ped}, frame=_make_dummy_frame(timestamp=2.0))

        all_ev = generator.get_all_events()
        assert len(all_ev) == 2
        # First car event was resolved due to class change
        assert all_ev[0].state == WarningEventState.RESOLVED
        assert all_ev[0].resolution_reason == "TRACKER_ID_REUSE_CLASS_CHANGED"
        # Second event is pedestrian
        assert all_ev[1].object_class == "pedestrian"

    def test_multiple_simultaneous_threats(self):
        """Simultaneous threats across distinct directions and classes maintain independent events."""
        generator = DirectionalWarningGenerator()

        # Threat 1: Lead braking car in FRONT (HIGH)
        ass1, feat1 = _make_threat_data(
            track_id=101, object_class="car", risk_score=0.82, ttc=1.3, direction=CameraDirection.FRONT, timestamp=1.0
        )
        # Threat 2: Pedestrian crossing from FRONT_LEFT (HIGH)
        bbox_ped = BoundingBox(x1=250.0, y1=380.0, x2=320.0, y2=510.0)
        ass2, feat2 = _make_threat_data(
            track_id=102,
            object_class="pedestrian",
            scenario=ScenarioType.PEDESTRIAN_CROSSING,
            scenario_state=ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value,
            risk_score=0.78,
            ttc=1.4,
            direction=CameraDirection.FRONT_LEFT,
            bbox=bbox_ped,
            lane_relationship=LaneRelationship.LEFT_ADJACENT_LANE,
            timestamp=1.0,
        )
        # Threat 3: Cyclist encroaching from FRONT_RIGHT (MEDIUM)
        bbox_cyc = BoundingBox(x1=920.0, y1=400.0, x2=990.0, y2=520.0)
        ass3, feat3 = _make_threat_data(
            track_id=103,
            object_class="bicycle",
            scenario=ScenarioType.CYCLIST_CONFLICT,
            scenario_state=ScenarioState.CYCLIST_ENTERING_LANE.value,
            risk_score=0.55,
            ttc=2.8,
            direction=CameraDirection.FRONT_RIGHT,
            bbox=bbox_cyc,
            lane_relationship=LaneRelationship.RIGHT_ADJACENT_LANE,
            timestamp=1.0,
        )

        threat_assessments = {101: ass1, 102: ass2, 103: ass3}
        threat_features = {101: feat1, 102: feat2, 103: feat3}

        # Frame 1: 101 and 102 fast-track activate; 103 is buffered
        out1 = generator.process(threat_assessments, threat_features, frame=_make_dummy_frame(timestamp=1.0))
        assert len(out1) == 2
        active_tids = {w.track_id for w in out1}
        assert active_tids == {101, 102}

        # Frame 2 at t=1.2s confirms 103 as well
        for ass in threat_assessments.values():
            ass.timestamp = 1.2
        for feat in threat_features.values():
            feat.timestamp = 1.2
        out2 = generator.process(threat_assessments, threat_features, frame=_make_dummy_frame(timestamp=1.2))
        assert len(out2) == 3

        levels_by_tid = {w.track_id: w.level for w in out2}
        assert levels_by_tid[101] == DirectionalWarningLevel.HIGH
        assert levels_by_tid[102] == DirectionalWarningLevel.HIGH
        assert levels_by_tid[103] == DirectionalWarningLevel.MEDIUM

        dirs_by_tid = {w.track_id: w.direction for w in out2}
        assert dirs_by_tid[101] == CameraDirection.FRONT
        assert dirs_by_tid[102] == CameraDirection.FRONT_LEFT
        assert dirs_by_tid[103] == CameraDirection.FRONT_RIGHT

    def test_pipeline_integration(self):
        """PerceptionPipeline executes directional warning generation and exposes results."""
        config = {
            "threat_assessment": {
                "enabled": True,
                "directional_warnings": {
                    "enabled": True,
                },
            },
            "detector": {"enabled": False},
            "tracker": {"enabled": False},
            "depth": {"enabled": False},
            "lane": {"enabled": False},
            "signs": {"enabled": False},
            "scene": {"enabled": False},
            "behavior": {"enabled": False},
            "visualization": {"enabled": False},
        }
        pipeline = PerceptionPipeline(config)
        assert pipeline.directional_warning_generator is not None

        # Reset threat state clears warning events
        pipeline.reset_threat_state()
        assert len(pipeline.get_directional_warnings()) == 0
        assert len(pipeline.get_directional_warning_events()) == 0
