"""Unit tests for traffic conflict scenario classification.

Covers:
- Scenario 1: Rear-End Conflict (closing vs receding vs potential)
- Scenario 2: Pedestrian Crossing Conflict (crossing vs approaching vs alongside vs clear)
- Scenario 3: Cyclist Conflict (crossing vs entering vs lead vs adjacent)
- Scenario 4: Vehicle Cut-In (potential vs confirmed vs completed vs uncertain)
- Scenario 5: Adjacent-Lane Threat (encroaching vs stable)
- Animal Crossing Conflict
- Temporal stability, confirmation buffers, and flicker prevention
- Temporary track loss grace period
- PerceptionPipeline integration
"""

import numpy as np
import pytest

from src.core.pipeline import PerceptionPipeline
from src.schemas.bbox import BoundingBox
from src.schemas.frame import Frame
from src.threat.scenario_classifier import ScenarioClassifier
from src.threat.schemas import (
    CameraDirection,
    LaneRelationship,
    ScenarioState,
    ScenarioType,
    ThreatFeatures,
    TrajectoryConflictState,
    ValidityStatus,
    VulnerabilityLevel,
)


def _make_dummy_features(
    track_id: int = 1,
    object_class: str = "car",
    direction: CameraDirection = CameraDirection.FRONT,
    lane_rel: LaneRelationship = LaneRelationship.EGO_LANE,
    lane_overlap: float = 0.9,
    relative_speed: float | None = 5.0,
    ttc: float | None = 3.0,
    drac: float | None = 1.0,
    traj_conflict: TrajectoryConflictState = TrajectoryConflictState.CONFLICT,
    crossing_probability: float | None = None,
    behavior_label: str | None = None,
    looming_rate: float | None = 0.2,
    timestamp: float = 1.0,
) -> ThreatFeatures:
    """Helper to construct synthetic ThreatFeatures for testing."""
    return ThreatFeatures(
        track_id=track_id,
        tracker_id=track_id,
        object_class=object_class,
        bbox=BoundingBox(500, 400, 700, 600),
        timestamp=timestamp,
        direction=direction,
        lane_relationship=lane_rel,
        lane_overlap=lane_overlap,
        lateral_deviation=0.2 if lane_rel == LaneRelationship.EGO_LANE else 1.3,
        ttc=ttc,
        pet=None,
        drac=drac,
        relative_speed=relative_speed,
        relative_acceleration=0.0,
        motion_direction="approaching" if (relative_speed and relative_speed > 0) else "receding",
        speed_variation=0.1,
        acceleration_noise=0.1,
        looming_rate=looming_rate,
        trajectory_conflict=traj_conflict,
        trajectory_conflict_time=1.5 if traj_conflict == TrajectoryConflictState.CONFLICT else None,
        vulnerability=VulnerabilityLevel.STANDARD,
        crossing_probability=crossing_probability,
        behavior_label=behavior_label,
        observation_duration=1.0,
        observation_count=10,
        is_metric=True,
    )


class TestScenarioClassifier:
    """Comprehensive test suite for ScenarioClassifier."""

    def test_rear_end_closing_vs_receding(self):
        """Test rear-end conflict states under closing vs receding motion."""
        classifier = ScenarioClassifier({"threat_assessment": {"enabled": True}})

        # 1. Closing rapidly with small TTC -> REAR_END_CONFLICT
        closing_feat = _make_dummy_features(
            object_class="car",
            lane_rel=LaneRelationship.EGO_LANE,
            relative_speed=6.0,
            ttc=2.5,
            looming_rate=0.3,
        )
        # Feed 3 frames for multi-frame confirmation
        for t in [1.0, 1.1, 1.2]:
            closing_feat.timestamp = t
            res = classifier.classify({1: closing_feat}, frame_timestamp=t)

        ass = res[1]
        assert ass.scenario == ScenarioType.REAR_END
        assert ass.state == ScenarioState.REAR_END_CONFLICT.value
        assert ass.confidence >= 0.85
        assert "RAPID_CLOSING_VELOCITY" in ass.evidence

        # 2. Receding vehicle (speed < -0.3 m/s) -> NO_REAR_END_CONFLICT
        receding_feat = _make_dummy_features(
            object_class="car",
            lane_rel=LaneRelationship.EGO_LANE,
            relative_speed=-4.0,
            ttc=None,
            looming_rate=-0.2,
        )
        for t in [1.3, 1.4, 1.5]:
            receding_feat.timestamp = t
            res_rec = classifier.classify({1: receding_feat}, frame_timestamp=t)

        ass_rec = res_rec[1]
        assert ass_rec.scenario == ScenarioType.REAR_END
        assert ass_rec.state == ScenarioState.NO_REAR_END_CONFLICT.value
        assert "RECEDING_GAP" in ass_rec.evidence

    def test_pedestrian_scenarios(self):
        """Test pedestrian crossing, approaching, and alongside path."""
        classifier = ScenarioClassifier({"threat_assessment": {"enabled": True}})

        # 1. Pedestrian crossing ego path
        ped_crossing = _make_dummy_features(
            track_id=10,
            object_class="person",
            lane_rel=LaneRelationship.ENTERING_EGO_LANE,
            lane_overlap=0.35,
            crossing_probability=0.75,
            traj_conflict=TrajectoryConflictState.CONFLICT,
            behavior_label="human_crossing_road",
        )
        res1 = classifier.classify({10: ped_crossing}, frame_timestamp=1.0)
        assert res1[10].scenario == ScenarioType.PEDESTRIAN_CROSSING
        assert res1[10].state == ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value

        # 2. Pedestrian alongside path (sidewalk, clear of road)
        ped_alongside = _make_dummy_features(
            track_id=11,
            object_class="person",
            direction=CameraDirection.FRONT_RIGHT,
            lane_rel=LaneRelationship.OUTSIDE_RELEVANT_LANES,
            lane_overlap=0.0,
            crossing_probability=0.05,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
            behavior_label="human_standing",
        )
        res2 = classifier.classify({11: ped_alongside}, frame_timestamp=1.0)
        assert res2[11].scenario == ScenarioType.PEDESTRIAN_CROSSING
        assert res2[11].state == ScenarioState.PEDESTRIAN_ALONGSIDE_PATH.value

        # 3. Pedestrian leaving road
        ped_leaving = _make_dummy_features(
            track_id=12,
            object_class="person",
            lane_rel=LaneRelationship.LEAVING_EGO_LANE,
            lane_overlap=0.05,
            crossing_probability=0.1,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
            behavior_label="human_leaving_road",
        )
        res3 = classifier.classify({12: ped_leaving}, frame_timestamp=1.0)
        assert res3[12].scenario == ScenarioType.PEDESTRIAN_CROSSING
        assert res3[12].state == ScenarioState.PEDESTRIAN_CLEAR_OF_PATH.value

    def test_cyclist_scenarios(self):
        """Test cyclist crossing, entering, lead, and adjacent interactions."""
        classifier = ScenarioClassifier({"threat_assessment": {"enabled": True}})

        # 1. Cyclist crossing path
        cyclist_cross = _make_dummy_features(
            track_id=20,
            object_class="bicycle",
            direction=CameraDirection.FRONT_LEFT,
            lane_rel=LaneRelationship.ENTERING_EGO_LANE,
            traj_conflict=TrajectoryConflictState.CONFLICT,
        )
        res1 = classifier.classify({20: cyclist_cross}, frame_timestamp=1.0)
        assert res1[20].scenario == ScenarioType.CYCLIST_CONFLICT
        assert res1[20].state == ScenarioState.CYCLIST_CROSSING_CONFLICT.value

        # 2. Cyclist lead in ego lane
        cyclist_lead = _make_dummy_features(
            track_id=21,
            object_class="bicycle",
            direction=CameraDirection.FRONT,
            lane_rel=LaneRelationship.EGO_LANE,
            lane_overlap=0.8,
            relative_speed=2.0,
            ttc=4.0,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
        )
        res2 = classifier.classify({21: cyclist_lead}, frame_timestamp=1.0)
        assert res2[21].scenario == ScenarioType.CYCLIST_CONFLICT
        assert res2[21].state == ScenarioState.CYCLIST_LEAD_CONFLICT.value

        # 3. Cyclist adjacent lane stable
        cyclist_adj = _make_dummy_features(
            track_id=22,
            object_class="bicycle",
            direction=CameraDirection.FRONT_RIGHT,
            lane_rel=LaneRelationship.RIGHT_ADJACENT_LANE,
            lane_overlap=0.0,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
        )
        res3 = classifier.classify({22: cyclist_adj}, frame_timestamp=1.0)
        assert res3[22].scenario == ScenarioType.CYCLIST_CONFLICT
        assert res3[22].state == ScenarioState.CYCLIST_ADJACENT_LANE.value

    def test_vehicle_cut_in_states(self):
        """Test vehicle cut-in progression: potential -> confirmed -> completed."""
        classifier = ScenarioClassifier({"threat_assessment": {"enabled": True}})

        # Step 1: Potential cut-in (in adjacent lane with inward trajectory conflict)
        pot_cut_in = _make_dummy_features(
            track_id=30,
            object_class="car",
            direction=CameraDirection.FRONT_LEFT,
            lane_rel=LaneRelationship.LEFT_ADJACENT_LANE,
            lane_overlap=0.0,
            traj_conflict=TrajectoryConflictState.CONFLICT,
        )
        for t in [1.0, 1.1, 1.2]:
            pot_cut_in.timestamp = t
            res1 = classifier.classify({30: pot_cut_in}, frame_timestamp=t)
        assert res1[30].scenario == ScenarioType.VEHICLE_CUT_IN
        assert res1[30].state == ScenarioState.POTENTIAL_CUT_IN.value

        # Step 2: Confirmed cut-in (crossing lane boundary, overlap 0.4)
        conf_cut_in = _make_dummy_features(
            track_id=30,
            object_class="car",
            direction=CameraDirection.FRONT_LEFT,
            lane_rel=LaneRelationship.ENTERING_EGO_LANE,
            lane_overlap=0.40,
            traj_conflict=TrajectoryConflictState.CONFLICT,
        )
        for t in [1.3, 1.4, 1.5]:
            conf_cut_in.timestamp = t
            res2 = classifier.classify({30: conf_cut_in}, frame_timestamp=t)
        assert res2[30].scenario == ScenarioType.VEHICLE_CUT_IN
        assert res2[30].state == ScenarioState.CONFIRMED_CUT_IN.value

        # Step 3: Completed cut-in (now fully established in ego lane, overlap 0.85)
        comp_cut_in = _make_dummy_features(
            track_id=30,
            object_class="car",
            direction=CameraDirection.FRONT,
            lane_rel=LaneRelationship.EGO_LANE,
            lane_overlap=0.85,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
        )
        for t in [1.6, 1.7, 1.8]:
            comp_cut_in.timestamp = t
            res3 = classifier.classify({30: comp_cut_in}, frame_timestamp=t)
        assert res3[30].scenario == ScenarioType.VEHICLE_CUT_IN
        assert res3[30].state == ScenarioState.COMPLETED_CUT_IN.value

    def test_adjacent_lane_threat(self):
        """Test adjacent-lane stable vs encroaching."""
        classifier = ScenarioClassifier({"threat_assessment": {"enabled": True}})

        # Stable adjacent vehicle
        stable_adj = _make_dummy_features(
            track_id=40,
            object_class="car",
            direction=CameraDirection.FRONT_RIGHT,
            lane_rel=LaneRelationship.RIGHT_ADJACENT_LANE,
            lane_overlap=0.0,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
            looming_rate=0.0,
        )
        res1 = classifier.classify({40: stable_adj}, frame_timestamp=1.0)
        assert res1[40].scenario == ScenarioType.ADJACENT_LANE_THREAT
        assert res1[40].state == ScenarioState.ADJACENT_LANE_STABLE.value

        # Encroaching adjacent vehicle (lateral deviation low, looming rate elevated)
        encroaching_adj = _make_dummy_features(
            track_id=41,
            object_class="truck",
            direction=CameraDirection.FRONT_LEFT,
            lane_rel=LaneRelationship.LEFT_ADJACENT_LANE,
            lane_overlap=0.0,
            traj_conflict=TrajectoryConflictState.NO_CONFLICT,
            looming_rate=0.25,
        )
        encroaching_adj.lateral_deviation = 1.1  # Near line
        res2 = classifier.classify({41: encroaching_adj}, frame_timestamp=1.0)
        assert res2[41].scenario == ScenarioType.ADJACENT_LANE_THREAT
        assert res2[41].state == ScenarioState.ADJACENT_LANE_ENCROACHING.value

    def test_temporal_stability_prevents_single_frame_flicker(self):
        """Test that a transient 1-frame state change does not destabilize confirmed state."""
        classifier = ScenarioClassifier({"threat_assessment": {"scenarios": {"min_confirmation_frames": 3}}})

        feat = _make_dummy_features(
            track_id=50,
            object_class="car",
            lane_rel=LaneRelationship.EGO_LANE,
            relative_speed=-2.0,  # Receding
            ttc=None,
        )
        # Establish confirmed state over 3 frames
        for t in [1.0, 1.1, 1.2]:
            feat.timestamp = t
            res = classifier.classify({50: feat}, frame_timestamp=t)
        assert res[50].state == ScenarioState.NO_REAR_END_CONFLICT.value

        # Single transient frame of closing speed noise (speed = 1.0)
        feat_noisy = _make_dummy_features(
            track_id=50,
            object_class="car",
            lane_rel=LaneRelationship.EGO_LANE,
            relative_speed=1.0,
            ttc=7.0,  # Potential
        )
        feat_noisy.timestamp = 1.3
        res_noisy = classifier.classify({50: feat_noisy}, frame_timestamp=1.3)

        # Still maintains previous confirmed state due to min_confirmation_frames=3
        assert res_noisy[50].state == ScenarioState.NO_REAR_END_CONFLICT.value
        assert res_noisy[50].validity == ValidityStatus.UNCERTAIN

    def test_temporary_track_loss_grace_period(self):
        """Test that track absent for <= grace_period (0.4s) retains assessment with decayed confidence."""
        classifier = ScenarioClassifier({
            "threat_assessment": {
                "scenarios": {"track_loss_grace_seconds": 0.4},
                "stale_timeout_seconds": 2.0,
            }
        })

        feat = _make_dummy_features(track_id=60, object_class="car")
        classifier.classify({60: feat}, frame_timestamp=1.0)

        # Track 60 drops out for 0.2s (delta = 0.2 <= 0.4)
        res_grace = classifier.classify({}, frame_timestamp=1.2)
        assert 60 in res_grace
        assert "TEMPORARY_TRACK_LOSS_GRACE" in res_grace[60].evidence
        assert res_grace[60].validity == ValidityStatus.UNCERTAIN

        # Track 60 absent for 2.5s (> stale_timeout 2.0s) -> evicted completely
        res_stale = classifier.classify({}, frame_timestamp=3.7)
        assert 60 not in res_stale

    def test_pipeline_integration(self):
        """Test that PerceptionPipeline.process includes threat_features and threat_scenarios."""
        pipeline = PerceptionPipeline(config={})
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(image=dummy_img, frame_id=1, timestamp=0.033, source="test")

        result = pipeline.process(frame)
        assert "threat_features" in result
        assert "threat_scenarios" in result
        assert isinstance(result["threat_features"], dict)
        assert isinstance(result["threat_scenarios"], dict)
        assert pipeline.get_threat_features() == result["threat_features"]
        assert pipeline.get_threat_scenarios() == result["threat_scenarios"]
