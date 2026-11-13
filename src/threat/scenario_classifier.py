"""Modular traffic conflict scenario classifier.

Classifies interactions between the ego vehicle and tracked road users
into distinct conflict scenarios with temporal stabilization and
multi-frame persistence.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from src.threat.schemas import (
    CameraDirection,
    LaneRelationship,
    ScenarioAssessment,
    ScenarioState,
    ScenarioType,
    ThreatFeatures,
    TrajectoryConflictState,
    ValidityStatus,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioClassifier:
    """Classifies interactions into scenario-specific conflict categories.

    Implements multi-cue classification for:
    - Scenario 1: Rear-End Conflict
    - Scenario 2: Pedestrian Crossing Conflict
    - Scenario 3: Cyclist Conflict
    - Scenario 4: Vehicle Cut-In
    - Scenario 5: Adjacent-Lane Threat
    - Supporting: Animal Crossing Conflict

    Enforces temporal confirmation buffers to prevent frame-to-frame flicker,
    handles temporary track loss, and evicts stale histories.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        threat_cfg = self.config.get("threat_assessment", {})
        scen_cfg = threat_cfg.get("scenarios", {})

        self.enabled: bool = bool(threat_cfg.get("enabled", True))
        self.min_confirmation_frames: int = int(
            scen_cfg.get("min_confirmation_frames", 3)
        )
        self.track_loss_grace_seconds: float = float(
            scen_cfg.get("track_loss_grace_seconds", 0.4)
        )
        self.stale_timeout_seconds: float = float(
            threat_cfg.get("stale_timeout_seconds", 2.0)
        )

        # Configurable conflict thresholds
        self.ttc_conflict_s: float = float(scen_cfg.get("ttc_conflict_seconds", 4.5))
        self.ttc_potential_s: float = float(scen_cfg.get("ttc_potential_seconds", 8.0))
        self.drac_conflict_mps2: float = float(scen_cfg.get("drac_conflict_mps2", 2.0))
        self.closing_speed_threshold_mps: float = float(
            scen_cfg.get("closing_speed_threshold_mps", 0.3)
        )

        # Internal state tracking
        self._candidate_history: dict[int, deque[tuple[ScenarioType, str]]] = {}
        self._confirmed_assessments: dict[int, ScenarioAssessment] = {}
        self._last_seen_timestamp: dict[int, float] = {}

    def reset(self) -> None:
        """Reset all internal scenario tracking states."""
        self._candidate_history.clear()
        self._confirmed_assessments.clear()
        self._last_seen_timestamp.clear()

    def evict_stale_tracks(self, current_timestamp: float) -> None:
        """Evict tracks absent longer than stale_timeout_seconds."""
        stale_ids = [
            tid
            for tid, last_t in self._last_seen_timestamp.items()
            if (current_timestamp - last_t) > self.stale_timeout_seconds
            or (current_timestamp < last_t)  # Video rewound
        ]
        for tid in stale_ids:
            self._candidate_history.pop(tid, None)
            self._confirmed_assessments.pop(tid, None)
            self._last_seen_timestamp.pop(tid, None)

    def classify(
        self,
        features_map: dict[int, ThreatFeatures],
        frame_timestamp: float = 0.0,
    ) -> dict[int, ScenarioAssessment]:
        """Classify each tracked road user into a stabilized conflict scenario.

        Args:
            features_map: Extracted threat features per track ID.
            frame_timestamp: Current video presentation timestamp.

        Returns:
            Dictionary mapping track_id to ScenarioAssessment.
        """
        if not self.enabled:
            return {}

        self.evict_stale_tracks(frame_timestamp)

        assessments: dict[int, ScenarioAssessment] = {}
        active_tids = set(features_map.keys())

        # 1. Process active tracks
        for tid, feat in features_map.items():
            self._last_seen_timestamp[tid] = frame_timestamp

            # Determine candidate scenario and state from current frame features
            candidate_scenario, candidate_state, evidence, secondaries, raw_conf = (
                self._evaluate_candidate_scenario(feat)
            )

            # Update temporal candidate buffer
            if tid not in self._candidate_history:
                self._candidate_history[tid] = deque(maxlen=10)
            self._candidate_history[tid].append((candidate_scenario, candidate_state))

            # Temporal stabilization logic
            assessment = self._stabilize_assessment(
                tid=tid,
                feat=feat,
                candidate_scenario=candidate_scenario,
                candidate_state=candidate_state,
                evidence=evidence,
                secondaries=secondaries,
                raw_confidence=raw_conf,
                timestamp=frame_timestamp,
            )

            self._confirmed_assessments[tid] = assessment
            assessments[tid] = assessment

        # 2. Handle missing tracks within grace period
        for tid, last_assessment in list(self._confirmed_assessments.items()):
            if tid not in active_tids:
                dt = frame_timestamp - self._last_seen_timestamp.get(tid, 0.0)
                if 0.0 < dt <= self.track_loss_grace_seconds:
                    # Retain with slightly decayed confidence
                    decayed_conf = max(0.2, last_assessment.confidence * 0.85)
                    decayed = ScenarioAssessment(
                        track_id=tid,
                        scenario=last_assessment.scenario,
                        state=last_assessment.state,
                        confidence=decayed_conf,
                        direction=last_assessment.direction,
                        lane_relationship=last_assessment.lane_relationship,
                        evidence=list(last_assessment.evidence) + ["TEMPORARY_TRACK_LOSS_GRACE"],
                        secondary_scenarios=list(last_assessment.secondary_scenarios),
                        timestamp=frame_timestamp,
                        validity=ValidityStatus.UNCERTAIN,
                    )
                    assessments[tid] = decayed

        return assessments

    def _evaluate_candidate_scenario(
        self, feat: ThreatFeatures
    ) -> tuple[ScenarioType, str, list[str], list[str], float]:
        """Evaluate raw candidate scenario, state, and evidence for a single track."""
        c = feat.object_class.lower()

        # Route by road user category
        if c in ("person", "pedestrian"):
            return self._classify_pedestrian(feat)
        elif c in ("bicycle", "cyclist"):
            return self._classify_cyclist(feat)
        elif c in ("animal",):
            return self._classify_animal(feat)
        elif c in ("car", "truck", "bus", "motorcycle", "motorbike"):
            return self._classify_vehicle(feat)

        # Unknown / other
        return (
            ScenarioType.UNKNOWN,
            ScenarioState.NONE.value,
            ["UNRECOGNIZED_ROAD_USER_CLASS"],
            [],
            0.1,
        )

    # -------------------------------------------------------------------------
    # Scenario 1, 4, 5: Motor Vehicles
    # -------------------------------------------------------------------------
    def _classify_vehicle(
        self, feat: ThreatFeatures
    ) -> tuple[ScenarioType, str, list[str], list[str], float]:
        """Classify motorized vehicles into Rear-End, Cut-In, or Adjacent-Lane."""
        lane_rel = feat.lane_relationship
        overlap = feat.lane_overlap
        direction = feat.direction
        closing_speed = feat.relative_speed
        ttc = feat.ttc
        drac = feat.drac
        traj_conflict = feat.trajectory_conflict
        looming = feat.looming_rate or 0.0

        prev_assessment = self._confirmed_assessments.get(feat.track_id)
        prev_scenario = prev_assessment.scenario if prev_assessment else None
        prev_state = prev_assessment.state if prev_assessment else None

        # Check Cut-In first if vehicle is transitioning into ego lane or in adjacent lane moving inward
        is_adjacent = lane_rel in (
            LaneRelationship.LEFT_ADJACENT_LANE,
            LaneRelationship.RIGHT_ADJACENT_LANE,
        )
        is_entering = lane_rel == LaneRelationship.ENTERING_EGO_LANE

        # Cut-In: Has evidence of lateral transition toward or into ego corridor
        if is_entering or (is_adjacent and traj_conflict == TrajectoryConflictState.CONFLICT):
            evidence: list[str] = []
            secondaries: list[str] = []

            # Sub-state: Confirmed cut-in (actively crossing boundary)
            if overlap >= 0.15 and overlap <= 0.70:
                evidence.append("LANE_BOUNDARY_STRADDLE")
                evidence.append("CONFIRMED_LANE_INTRUSION")
                confidence = 0.85
                return (
                    ScenarioType.VEHICLE_CUT_IN,
                    ScenarioState.CONFIRMED_CUT_IN.value,
                    evidence,
                    secondaries,
                    confidence,
                )

            # Sub-state: Potential cut-in (in adjacent lane, predicted path crosses line)
            elif is_adjacent and traj_conflict == TrajectoryConflictState.CONFLICT:
                evidence.append("ADJACENT_LANE_POSITION")
                evidence.append("PREDICTED_LANE_CROSSING")
                confidence = 0.75
                return (
                    ScenarioType.VEHICLE_CUT_IN,
                    ScenarioState.POTENTIAL_CUT_IN.value,
                    evidence,
                    secondaries,
                    confidence,
                )

            # Sub-state: Uncertain lane transition
            else:
                evidence.append("BOUNDARY_HOVERING")
                confidence = 0.50
                return (
                    ScenarioType.VEHICLE_CUT_IN,
                    ScenarioState.UNCERTAIN_LANE_TRANSITION.value,
                    evidence,
                    secondaries,
                    confidence,
                )

        # Cut-In completion: Was cutting in, now established in ego lane
        if (
            prev_scenario == ScenarioType.VEHICLE_CUT_IN
            and prev_state in (ScenarioState.CONFIRMED_CUT_IN.value, ScenarioState.POTENTIAL_CUT_IN.value)
            and overlap >= 0.75
        ):
            evidence = ["FULL_EGO_LANE_ESTABLISHED", "LATERAL_VELOCITY_SUBSIDED"]
            return (
                ScenarioType.VEHICLE_CUT_IN,
                ScenarioState.COMPLETED_CUT_IN.value,
                evidence,
                [ScenarioType.REAR_END.value],
                0.90,
            )

        # Scenario 1: Rear-End Conflict (In ego lane ahead)
        if lane_rel == LaneRelationship.EGO_LANE or overlap >= 0.50:
            evidence = ["IN_EGO_LANE_AHEAD"]
            secondaries = []

            # Check for active conflict vs receding motion
            is_receding = (
                (closing_speed is not None and closing_speed < -self.closing_speed_threshold_mps)
                or feat.motion_direction == "receding"
            )

            is_closing = (
                not is_receding
                and (
                    (closing_speed is not None and closing_speed > self.closing_speed_threshold_mps)
                    or feat.motion_direction == "approaching"
                    or looming > 0.15
                )
            )

            if is_closing:
                if (
                    (ttc is not None and ttc <= self.ttc_conflict_s)
                    or (drac is not None and drac >= self.drac_conflict_mps2)
                    or (traj_conflict == TrajectoryConflictState.CONFLICT)
                    or (looming > 0.25)
                ):
                    evidence.append("RAPID_CLOSING_VELOCITY")
                    if ttc is not None:
                        evidence.append(f"TTC_CRITICAL_{ttc:.1f}S")
                    if drac is not None:
                        evidence.append(f"DRAC_ELEVATED_{drac:.1f}MPS2")
                    return (
                        ScenarioType.REAR_END,
                        ScenarioState.REAR_END_CONFLICT.value,
                        evidence,
                        secondaries,
                        0.92,
                    )
                else:
                    # Closing slowly or moderate TTC
                    evidence.append("MODERATE_CLOSING_RATE")
                    if ttc is not None:
                        evidence.append(f"TTC_MODERATE_{ttc:.1f}S")
                    return (
                        ScenarioType.REAR_END,
                        ScenarioState.REAR_END_POTENTIAL.value,
                        evidence,
                        secondaries,
                        0.75,
                    )

            elif is_receding:
                evidence.append("RECEDING_GAP")
                evidence.append("NO_COLLISION_COURSE")
                return (
                    ScenarioType.REAR_END,
                    ScenarioState.NO_REAR_END_CONFLICT.value,
                    evidence,
                    secondaries,
                    0.88,
                )

            else:
                # Stationary / neutral headway or insufficient history
                if feat.observation_count < 3:
                    evidence.append("INSUFFICIENT_OBSERVATION_HISTORY")
                    return (
                        ScenarioType.REAR_END,
                        ScenarioState.UNCERTAIN.value,
                        evidence,
                        secondaries,
                        0.50,
                    )
                else:
                    evidence.append("NEUTRAL_HEADWAY")
                    return (
                        ScenarioType.REAR_END,
                        ScenarioState.REAR_END_POTENTIAL.value,
                        evidence,
                        secondaries,
                        0.70,
                    )

        # Scenario 5: Adjacent-Lane Threat
        if is_adjacent:
            evidence = ["ADJACENT_LANE_TRAFFIC"]
            secondaries = []

            # Encroaching: vehicle drifting toward lane boundary (lateral deviation decreasing)
            # or lateral margin looming
            if feat.lateral_deviation is not None and feat.lateral_deviation < 1.25 and looming > 0.10:
                evidence.append("LATERAL_ENCROACHMENT_DRIFT")
                return (
                    ScenarioType.ADJACENT_LANE_THREAT,
                    ScenarioState.ADJACENT_LANE_ENCROACHING.value,
                    evidence,
                    secondaries,
                    0.78,
                )
            else:
                evidence.append("PARALLEL_STABLE_COURSE")
                return (
                    ScenarioType.ADJACENT_LANE_THREAT,
                    ScenarioState.ADJACENT_LANE_STABLE.value,
                    evidence,
                    secondaries,
                    0.85,
                )

        # Lateral periphery or outside relevant lanes
        return (
            ScenarioType.UNKNOWN,
            ScenarioState.NONE.value,
            ["OUTSIDE_CONFLICT_ZONES"],
            [],
            0.40,
        )

    # -------------------------------------------------------------------------
    # Scenario 2: Pedestrian Crossing Conflict
    # -------------------------------------------------------------------------
    def _classify_pedestrian(
        self, feat: ThreatFeatures
    ) -> tuple[ScenarioType, str, list[str], list[str], float]:
        """Classify pedestrian interactions into crossing, approaching, or alongside."""
        lane_rel = feat.lane_relationship
        overlap = feat.lane_overlap
        cross_prob = feat.crossing_probability
        traj_conflict = feat.trajectory_conflict
        behavior = feat.behavior_label or ""
        evidence: list[str] = ["PEDESTRIAN_VULNERABLE_ROAD_USER"]
        secondaries: list[str] = []

        is_in_lane = lane_rel == LaneRelationship.EGO_LANE or overlap > 0.10
        is_entering = lane_rel == LaneRelationship.ENTERING_EGO_LANE
        is_leaving = (
            lane_rel == LaneRelationship.LEAVING_EGO_LANE
            or "leaving" in behavior
        )

        # 1. Active Crossing Conflict
        if (
            (is_in_lane or is_entering)
            and (
                (cross_prob is not None and cross_prob >= 0.35)
                or traj_conflict == TrajectoryConflictState.CONFLICT
                or "crossing" in behavior
                or feat.direction == CameraDirection.FRONT
            )
        ):
            evidence.append("PEDESTRIAN_IN_OR_ENTERING_PATH")
            if cross_prob is not None:
                evidence.append(f"CROSSING_PROB_{cross_prob:.2f}")
            if traj_conflict == TrajectoryConflictState.CONFLICT:
                evidence.append("PREDICTED_PATH_INTERSECTION")
            return (
                ScenarioType.PEDESTRIAN_CROSSING,
                ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value,
                evidence,
                secondaries,
                0.95,
            )

        # 2. Clear of Path (leaving road or diverging path)
        if is_leaving or "leaving" in behavior:
            evidence.append("PEDESTRIAN_LEAVING_ROAD")
            evidence.append("DIVERGING_PATH")
            return (
                ScenarioType.PEDESTRIAN_CROSSING,
                ScenarioState.PEDESTRIAN_CLEAR_OF_PATH.value,
                evidence,
                secondaries,
                0.85,
            )

        # 3. Approaching Path
        if (
            (not is_in_lane)
            and (
                (cross_prob is not None and 0.20 <= cross_prob < 0.45)
                or "approaching" in behavior
                or traj_conflict == TrajectoryConflictState.CONFLICT
            )
        ):
            evidence.append("PEDESTRIAN_LATERAL_APPROACH")
            if cross_prob is not None:
                evidence.append(f"APPROACHING_PROB_{cross_prob:.2f}")
            return (
                ScenarioType.PEDESTRIAN_CROSSING,
                ScenarioState.PEDESTRIAN_APPROACHING_PATH.value,
                evidence,
                secondaries,
                0.80,
            )

        # 4. Alongside Path (sidewalk / stationary)
        evidence.append("PEDESTRIAN_PARALLEL_TRAJECTORY")
        evidence.append("CLEAR_OF_TRAVEL_CORRIDOR")
        return (
            ScenarioType.PEDESTRIAN_CROSSING,
            ScenarioState.PEDESTRIAN_ALONGSIDE_PATH.value,
            evidence,
            secondaries,
            0.75,
        )

    # -------------------------------------------------------------------------
    # Scenario 3: Cyclist Conflict
    # -------------------------------------------------------------------------
    def _classify_cyclist(
        self, feat: ThreatFeatures
    ) -> tuple[ScenarioType, str, list[str], list[str], float]:
        """Classify cyclist interactions into crossing, entering, lead, or adjacent."""
        lane_rel = feat.lane_relationship
        overlap = feat.lane_overlap
        traj_conflict = feat.trajectory_conflict
        ttc = feat.ttc
        speed = feat.relative_speed
        evidence: list[str] = ["CYCLIST_VULNERABLE_ROAD_USER"]
        secondaries: list[str] = []

        is_in_lane = lane_rel == LaneRelationship.EGO_LANE or overlap >= 0.40
        is_entering = lane_rel == LaneRelationship.ENTERING_EGO_LANE
        is_adjacent = lane_rel in (
            LaneRelationship.LEFT_ADJACENT_LANE,
            LaneRelationship.RIGHT_ADJACENT_LANE,
        )

        # 1. Cyclist Crossing Conflict
        if traj_conflict == TrajectoryConflictState.CONFLICT and (
            feat.direction in (CameraDirection.FRONT_LEFT, CameraDirection.FRONT_RIGHT)
            or is_entering
        ):
            evidence.append("CYCLIST_CROSSING_PATH")
            evidence.append("TRAJECTORY_INTERSECTION")
            return (
                ScenarioType.CYCLIST_CONFLICT,
                ScenarioState.CYCLIST_CROSSING_CONFLICT.value,
                evidence,
                secondaries,
                0.90,
            )

        # 2. Cyclist Entering Lane
        if is_entering or (is_adjacent and overlap > 0.05):
            evidence.append("CYCLIST_ENTERING_EGO_LANE")
            evidence.append("INWARD_LATERAL_DRIFT")
            return (
                ScenarioType.CYCLIST_CONFLICT,
                ScenarioState.CYCLIST_ENTERING_LANE.value,
                evidence,
                secondaries,
                0.85,
            )

        # 3. Cyclist Lead Conflict (Riding ahead in ego lane)
        if is_in_lane:
            evidence.append("CYCLIST_LEAD_IN_EGO_LANE")
            if (speed is not None and speed > 0.1) or (ttc is not None and ttc <= 5.0):
                evidence.append("CLOSING_GAP")
            return (
                ScenarioType.CYCLIST_CONFLICT,
                ScenarioState.CYCLIST_LEAD_CONFLICT.value,
                evidence,
                secondaries,
                0.88,
            )

        # 4. Cyclist in Adjacent Lane
        evidence.append("CYCLIST_ADJACENT_STABLE")
        evidence.append("PARALLEL_TRAJECTORY")
        return (
            ScenarioType.CYCLIST_CONFLICT,
            ScenarioState.CYCLIST_ADJACENT_LANE.value,
            evidence,
            secondaries,
            0.82,
        )

    # -------------------------------------------------------------------------
    # Supporting: Animal Crossing Conflict
    # -------------------------------------------------------------------------
    def _classify_animal(
        self, feat: ThreatFeatures
    ) -> tuple[ScenarioType, str, list[str], list[str], float]:
        """Classify animal road interaction."""
        lane_rel = feat.lane_relationship
        overlap = feat.lane_overlap
        behavior = feat.behavior_label or ""
        evidence: list[str] = ["ANIMAL_ROAD_USER"]

        if lane_rel == LaneRelationship.EGO_LANE or overlap > 0.10 or "crossing" in behavior:
            evidence.append("ANIMAL_IN_TRAVEL_PATH")
            return (
                ScenarioType.ANIMAL_CROSSING,
                ScenarioState.ANIMAL_CROSSING_CONFLICT.value,
                evidence,
                [],
                0.85,
            )
        elif lane_rel == LaneRelationship.ENTERING_EGO_LANE or "approaching" in behavior:
            evidence.append("ANIMAL_APPROACHING_TRAVEL_PATH")
            return (
                ScenarioType.ANIMAL_CROSSING,
                ScenarioState.ANIMAL_APPROACHING_PATH.value,
                evidence,
                [],
                0.75,
            )
        else:
            evidence.append("ANIMAL_OUTSIDE_TRAVEL_PATH")
            return (
                ScenarioType.ANIMAL_CROSSING,
                ScenarioState.ANIMAL_CLEAR_OF_PATH.value,
                evidence,
                [],
                0.70,
            )

    # -------------------------------------------------------------------------
    # Temporal Stabilization
    # -------------------------------------------------------------------------
    def _stabilize_assessment(
        self,
        tid: int,
        feat: ThreatFeatures,
        candidate_scenario: ScenarioType,
        candidate_state: str,
        evidence: list[str],
        secondaries: list[str],
        raw_confidence: float,
        timestamp: float,
    ) -> ScenarioAssessment:
        """Apply multi-frame persistence to filter transient flickers."""
        prev = self._confirmed_assessments.get(tid)
        history = self._candidate_history.get(tid, deque())

        # If no previous assessment, initialize confirmed state
        if prev is None:
            return ScenarioAssessment(
                track_id=tid,
                scenario=candidate_scenario,
                state=candidate_state,
                confidence=raw_confidence,
                direction=feat.direction,
                lane_relationship=feat.lane_relationship,
                evidence=evidence,
                secondary_scenarios=secondaries,
                timestamp=timestamp,
                validity=ValidityStatus.VALID_METRIC if feat.is_metric else ValidityStatus.VALID_PROXY,
            )

        # Critical fast-track exception: imminent critical conflict transitions immediately
        is_critical_fast_track = (
            (candidate_state == ScenarioState.REAR_END_CONFLICT.value and feat.ttc is not None and feat.ttc <= 2.0)
            or (candidate_state == ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value and feat.lane_overlap > 0.20)
        )

        if is_critical_fast_track:
            return ScenarioAssessment(
                track_id=tid,
                scenario=candidate_scenario,
                state=candidate_state,
                confidence=raw_confidence,
                direction=feat.direction,
                lane_relationship=feat.lane_relationship,
                evidence=evidence + ["CRITICAL_IMMEDIATE_TRANSITION"],
                secondary_scenarios=secondaries,
                timestamp=timestamp,
                validity=ValidityStatus.VALID_METRIC if feat.is_metric else ValidityStatus.VALID_PROXY,
            )

        # Standard transition check: count consecutive occurrences in candidate history
        recent_matches = sum(
            1 for s, st in history if s == candidate_scenario and st == candidate_state
        )

        if (candidate_scenario == prev.scenario and candidate_state == prev.state) or (
            recent_matches >= self.min_confirmation_frames
        ):
            # Transition confirmed or persisting
            stabilized_conf = min(0.98, max(raw_confidence, prev.confidence + 0.05))
            return ScenarioAssessment(
                track_id=tid,
                scenario=candidate_scenario,
                state=candidate_state,
                confidence=stabilized_conf,
                direction=feat.direction,
                lane_relationship=feat.lane_relationship,
                evidence=evidence,
                secondary_scenarios=secondaries,
                timestamp=timestamp,
                validity=ValidityStatus.VALID_METRIC if feat.is_metric else ValidityStatus.VALID_PROXY,
            )
        else:
            # Candidate not yet confirmed across min_confirmation_frames; retain previous confirmed state
            return ScenarioAssessment(
                track_id=tid,
                scenario=prev.scenario,
                state=prev.state,
                confidence=max(0.3, prev.confidence - 0.03),
                direction=feat.direction,
                lane_relationship=feat.lane_relationship,
                evidence=prev.evidence,
                secondary_scenarios=[candidate_scenario.value],
                timestamp=timestamp,
                validity=ValidityStatus.UNCERTAIN,
            )
