"""Directional warning generation and event management system.

Consumes threat assessments and spatial geometry to assign camera-relative
directional zones, evaluate LOW/MEDIUM/HIGH warning levels, enforce temporal
confirmation and escalation/de-escalation hysteresis, suppress duplicates, and
manage warning event lifecycles.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from src.schemas.frame import Frame
from src.threat.direction import DirectionEstimator
from src.threat.schemas import (
    CameraDirection,
    DirectionalWarningLevel,
    ModelEvaluationStatus,
    ReliabilityGrade,
    ScenarioState,
    ThreatFeatures,
    ThreatRiskAssessment,
    ThreatRiskLevel,
    TrajectoryConflictState,
    ValidityStatus,
    WarningEvent,
    WarningEventState,
    WarningOutput,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LEVEL_ORDER = {
    DirectionalWarningLevel.UNASSESSED: 0,
    DirectionalWarningLevel.LOW: 1,
    DirectionalWarningLevel.MEDIUM: 2,
    DirectionalWarningLevel.HIGH: 3,
}


class DirectionalWarningGenerator:
    """Manages directional warning generation and persistent warning event lifecycles.

    Features:
    - Dedicated camera-relative direction assignment (FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT).
    - LOW, MEDIUM, HIGH, and UNASSESSED warning level decisions.
    - Multi-threat management: independent events per tracked object.
    - Temporal confirmation filtering to reject single-frame false alarms.
    - Escalation and de-escalation hysteresis to prevent warning flapping.
    - Duplicate warning suppression and track-loss grace periods.
    - Strict media timestamping using frame presentation time.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        threat_cfg = self.config.get("threat_assessment", {})
        warn_cfg = threat_cfg.get("directional_warnings", {})

        self.enabled: bool = bool(threat_cfg.get("enabled", True) and warn_cfg.get("enabled", True))
        self.direction_estimator = DirectionEstimator(self.config)

        # Thresholds
        thresh_cfg = warn_cfg.get("thresholds", {})
        self.high_risk_threshold: float = float(thresh_cfg.get("high_risk_score", 0.70))
        self.medium_risk_threshold: float = float(thresh_cfg.get("medium_risk_score", 0.40))
        self.low_risk_threshold: float = float(thresh_cfg.get("low_risk_score", 0.20))
        self.critical_ttc_override: float = float(thresh_cfg.get("critical_ttc_override_s", 1.8))

        # Timing and hysteresis
        time_cfg = warn_cfg.get("timing", {})
        self.confirmation_frames: int = int(time_cfg.get("confirmation_frames", 2))
        self.confirmation_duration_s: float = float(time_cfg.get("confirmation_duration_s", 0.15))
        self.escalation_frames: int = int(time_cfg.get("escalation_frames", 2))
        self.deescalation_duration_s: float = float(time_cfg.get("deescalation_duration_s", 0.50))
        self.resolution_duration_s: float = float(time_cfg.get("resolution_duration_s", 0.80))
        self.track_loss_grace_s: float = float(time_cfg.get("track_loss_grace_s", 0.40))
        self.stale_timeout_s: float = float(threat_cfg.get("stale_timeout_seconds", 2.0))
        self.id_reuse_gap_s: float = float(threat_cfg.get("id_reuse_gap_seconds", 2.0))

        # Internal state
        self._event_counter: int = 0
        self._active_events: dict[tuple[int, str], WarningEvent] = {}  # (track_id, scenario) -> WarningEvent
        self._all_events: list[WarningEvent] = []
        self._last_seen: dict[int, float] = {}
        self._track_classes: dict[int, str] = {}

        # Candidate buffer for temporal confirmation: (track_id, scenario) -> deque[(timestamp, level, reason)]
        self._candidate_buffer: dict[tuple[int, str], deque[tuple[float, DirectionalWarningLevel, str]]] = {}
        # De-escalation buffer: (track_id, scenario) -> deque[(timestamp, level)]
        self._deescalation_buffer: dict[tuple[int, str], deque[tuple[float, DirectionalWarningLevel]]] = {}
        # Low risk duration buffer for resolution: (track_id, scenario) -> float (first time below threshold)
        self._low_risk_start_time: dict[tuple[int, str], float] = {}

        # Evaluation & reporting history buffers
        self._warning_history: list[dict[str, Any]] = []
        self._suppressed_duplicate_count: int = 0
        self._risk_assessment_details: list[dict[str, Any]] = []
        self._max_risk_details: int = 10000

    def reset(self) -> None:
        """Clear all warning events and internal state."""
        self.direction_estimator.reset()
        self._event_counter = 0
        self._active_events.clear()
        self._all_events.clear()
        self._last_seen.clear()
        self._track_classes.clear()
        self._candidate_buffer.clear()
        self._deescalation_buffer.clear()
        self._low_risk_start_time.clear()
        self._warning_history.clear()
        self._suppressed_duplicate_count = 0
        self._risk_assessment_details.clear()

    def process(
        self,
        threat_assessments: dict[int, ThreatRiskAssessment],
        threat_features: dict[int, ThreatFeatures],
        lane_info: Any = None,
        frame: Frame | None = None,
    ) -> list[WarningOutput]:
        """Process threat assessments and manage active directional warning events.

        Args:
            threat_assessments: Risk assessments from Stage 3.
            threat_features: Threat features from Stage 2.
            lane_info: Lane detection geometry.
            frame: Video frame (used for timestamp and image dimensions).

        Returns:
            List of WarningOutput records for all currently active warnings.
        """
        if not self.enabled:
            return []

        timestamp = float(frame.timestamp) if frame is not None else 0.0
        if frame is not None and hasattr(frame, "image") and frame.image is not None and len(frame.image.shape) >= 2:
            img_w, img_h = float(frame.image.shape[1]), float(frame.image.shape[0])
        else:
            img_w, img_h = 1280.0, 720.0

        # Purge expired and stale tracks
        self._purge_stale_events(timestamp)

        active_outputs: list[WarningOutput] = []
        current_seen_tracks: set[int] = set()

        for tid, ass in threat_assessments.items():
            feat = threat_features.get(tid)
            if feat is None:
                continue

            current_seen_tracks.add(tid)
            self._last_seen[tid] = timestamp

            # Tracker ID reuse check
            prev_class = self._track_classes.get(tid)
            if prev_class is not None and prev_class != feat.object_class.lower():
                self._resolve_events_for_track(tid, timestamp, reason="TRACKER_ID_REUSE_CLASS_CHANGED")
            self._track_classes[tid] = feat.object_class.lower()

            # Estimate / verify camera-relative direction
            direction = self.direction_estimator.estimate(
                track_id=tid,
                bbox=feat.bbox,
                lane_info=lane_info,
                lane_relationship=feat.lane_relationship,
                img_width=img_w,
                img_height=img_h,
                timestamp=timestamp,
            )

            # Determine candidate warning level and supporting reason
            candidate_level, candidate_reason = self._determine_candidate_level(ass, feat)

            # Record detailed risk assessment diagnostics (capped to avoid unbounded memory)
            if len(self._risk_assessment_details) < self._max_risk_details:
                contrib_parts = []
                for k, v in ass.normalized_indicators.items():
                    val = getattr(v, "normalized_value", v)
                    coeff = ass.published_coefficients.get(k, 0.0)
                    if val is not None and coeff is not None:
                        contrib_parts.append(f"{k}:{float(val) * float(coeff):.4f}")
                    elif val is not None:
                        contrib_parts.append(f"{k}:{float(val):.4f}")

                def _get_norm_val(key: str) -> float | None:
                    obj = ass.normalized_indicators.get(key)
                    if obj is None:
                        return None
                    return getattr(obj, "normalized_value", obj)

                detail_row = {
                    "timestamp": timestamp,
                    "event_id": f"warn_{ass.scenario.value.lower()}_{tid}" if (tid, ass.scenario.value) in self._active_events else "",
                    "track_id": tid,
                    "object_class": feat.object_class,
                    "scenario": ass.scenario.value,
                    "risk_model": ass.model_name.value,
                    "model_status": ass.model_status.value,
                    "raw_ttc": feat.ttc,
                    "norm_ttc": _get_norm_val("TTC"),
                    "raw_pet": feat.pet,
                    "norm_pet": _get_norm_val("PET"),
                    "raw_drac": feat.drac,
                    "norm_drac": _get_norm_val("DRAC"),
                    "raw_dist": getattr(feat, "distance", None),
                    "norm_dist": _get_norm_val("Distance") or _get_norm_val("Gap"),
                    "raw_rel_speed": feat.relative_speed,
                    "norm_rel_speed": _get_norm_val("RelSpeed") or _get_norm_val("Spd") or _get_norm_val("V_comb"),
                    "raw_encroachment": feat.lane_overlap,
                    "norm_encroachment": _get_norm_val("LaneOverlap") or _get_norm_val("LD"),
                    "relative_speed": feat.relative_speed,
                    "speed_std_dev": feat.speed_variation,
                    "acceleration_noise": feat.acceleration_noise,
                    "lateral_deviation": feat.lateral_deviation,
                    "combined_speed": feat.relative_speed,
                    "right_of_way": True,
                    "urgent_brake": False,
                    "trajectory_conflict": feat.trajectory_conflict.value,
                    "lane_overlap": feat.lane_overlap,
                    "measurement_type": "METRIC" if feat.is_metric else "PROXY",
                    "indicator_validity": ", ".join(f"{k}:{v.status.value}" for k, v in feat.indicator_validity.items()) if hasattr(feat, "indicator_validity") else "",
                    "reliability": ass.reliability.value,
                    "published_coefficients": ", ".join(f"{k}:{v}" for k, v in ass.published_coefficients.items()),
                    "weighted_contributions": ", ".join(contrib_parts),
                    "risk_score": ass.risk_score,
                    "priority_rank": ass.priority_rank,
                }
                self._risk_assessment_details.append(detail_row)

            # Update warning event lifecycle
            scenario_key = (tid, ass.scenario.value)
            output = self._update_event_lifecycle(
                scenario_key=scenario_key,
                tid=tid,
                feat=feat,
                ass=ass,
                direction=direction,
                candidate_level=candidate_level,
                candidate_reason=candidate_reason,
                timestamp=timestamp,
            )

            if output is not None:
                active_outputs.append(output)

        # Handle dropped tracks within grace period
        for scenario_key, event in list(self._active_events.items()):
            tid = scenario_key[0]
            if tid not in current_seen_tracks:
                dt = timestamp - self._last_seen.get(tid, 0.0)
                if 0.0 < dt <= self.track_loss_grace_s:
                    # Retain active event during temporary dropout
                    active_outputs.append(
                        WarningOutput(
                            event_id=event.event_id,
                            track_id=tid,
                            object_class=event.object_class,
                            scenario=event.scenario,
                            direction=event.direction,
                            level=event.current_level,
                            state=event.state,
                            risk_score=event.risk_score,
                            risk_model=event.risk_model,
                            reliability="PROVISIONAL",
                            ttc=event.ttc,
                            collision_urgency=event.collision_urgency,
                            is_primary_threat=event.is_primary_threat,
                            priority_rank=event.priority_rank,
                            timestamp=timestamp,
                            reason="TEMPORARY_TRACK_LOSS_GRACE",
                            escalated=False,
                            deescalated=False,
                        )
                    )
                elif dt > self.track_loss_grace_s and dt <= self.stale_timeout_s:
                    # Resolve event as target departed
                    self._resolve_event(scenario_key, timestamp, reason="TRACK_LOST_EXCEEDED_GRACE_PERIOD")

        return active_outputs

    def _determine_candidate_level(
        self, ass: ThreatRiskAssessment, feat: ThreatFeatures
    ) -> tuple[DirectionalWarningLevel, str]:
        """Determine candidate warning level based on risk score, urgency override, and validity."""
        # Unassessed check: if model status is invalid or measurements missing
        if ass.model_status in (
            ModelEvaluationStatus.INSUFFICIENT_INDICATORS,
            ModelEvaluationStatus.UNSUPPORTED_SCENARIO,
            ModelEvaluationStatus.INVALID_MEASUREMENTS,
        ) or ass.risk_score is None:
            return DirectionalWarningLevel.UNASSESSED, "Unassessed due to insufficient or uncalibrated indicators"

        score = ass.risk_score
        ttc = feat.ttc
        is_closing = (feat.relative_speed and feat.relative_speed > 0.2) or feat.motion_direction == "approaching"
        is_receding = (feat.relative_speed and feat.relative_speed < -0.3) or feat.motion_direction == "receding"

        # Suppress warnings for vehicles receding away from ego vehicle
        if is_receding and ass.scenario_state == ScenarioState.NO_REAR_END_CONFLICT.value:
            return DirectionalWarningLevel.UNASSESSED, "Receding vehicle with diverging trajectory"

        # 1. HIGH Warning Decision
        is_critical_urgency = (
            ttc is not None
            and ttc <= self.critical_ttc_override
            and is_closing
            and (feat.lane_overlap > 0.15 or feat.trajectory_conflict == TrajectoryConflictState.CONFLICT)
        )
        is_critical_conflict_state = ass.scenario_state in (
            ScenarioState.REAR_END_CONFLICT.value,
            ScenarioState.PEDESTRIAN_CROSSING_CONFLICT.value,
            ScenarioState.CONFIRMED_CUT_IN.value,
            ScenarioState.CYCLIST_CROSSING_CONFLICT.value,
        ) and (score >= 0.65 or is_critical_urgency)

        if (score >= self.high_risk_threshold or is_critical_urgency or is_critical_conflict_state) and (
            ass.reliability != ReliabilityGrade.UNRELIABLE
        ):
            reason = "Imminent collision risk"
            if is_critical_urgency:
                reason += f" (Urgent TTC={ttc:.1f}s)"
            elif score >= self.high_risk_threshold:
                reason += f" (Risk score={score:.2f})"
            return DirectionalWarningLevel.HIGH, reason

        # 2. MEDIUM Warning Decision
        is_developing_conflict = ass.scenario_state in (
            ScenarioState.POTENTIAL_CUT_IN.value,
            ScenarioState.PEDESTRIAN_APPROACHING_PATH.value,
            ScenarioState.CYCLIST_ENTERING_LANE.value,
            ScenarioState.CYCLIST_LEAD_CONFLICT.value,
            ScenarioState.REAR_END_POTENTIAL.value,
        ) and is_closing

        is_moderate_urgency = bool(ttc is not None and 1.8 < ttc <= 3.0 and is_closing and score >= 0.35)
        if score >= self.medium_risk_threshold or is_developing_conflict or is_moderate_urgency:
            return DirectionalWarningLevel.MEDIUM, f"Developing potential conflict (Score={score:.2f})"

        # 3. LOW Warning Decision
        if score >= self.low_risk_threshold or ass.scenario_state == ScenarioState.ADJACENT_LANE_ENCROACHING.value:
            return DirectionalWarningLevel.LOW, f"Advisory potential threat (Score={score:.2f})"

        return DirectionalWarningLevel.UNASSESSED, "Risk below advisory threshold"

    def _update_event_lifecycle(
        self,
        scenario_key: tuple[int, str],
        tid: int,
        feat: ThreatFeatures,
        ass: ThreatRiskAssessment,
        direction: CameraDirection,
        candidate_level: DirectionalWarningLevel,
        candidate_reason: str,
        timestamp: float,
    ) -> WarningOutput | None:
        """Manage confirmation, activation, escalation, de-escalation, and duplicate suppression."""
        active_event = self._active_events.get(scenario_key)

        # ---------------------------------------------------------------------
        # Case A: No active event exists yet -> Evaluate Candidate Confirmation
        # ---------------------------------------------------------------------
        if active_event is None:
            if candidate_level == DirectionalWarningLevel.UNASSESSED:
                self._candidate_buffer.pop(scenario_key, None)
                return None

            # Add to candidate confirmation buffer
            if scenario_key not in self._candidate_buffer:
                self._candidate_buffer[scenario_key] = deque(maxlen=10)
            self._candidate_buffer[scenario_key].append((timestamp, candidate_level, candidate_reason))

            buffer = self._candidate_buffer[scenario_key]
            first_t = buffer[0][0]
            dt = timestamp - first_t

            # Fast-track bypass for critical imminent hazards (e.g. HIGH warning with TTC <= 1.5s)
            is_fast_track = candidate_level == DirectionalWarningLevel.HIGH and (
                (feat.ttc is not None and feat.ttc <= 1.5) or feat.lane_overlap > 0.40
            )

            is_confirmed = is_fast_track or (
                len(buffer) >= self.confirmation_frames and dt >= self.confirmation_duration_s
            )

            if is_confirmed:
                # Activate new warning event
                self._event_counter += 1
                event_id = f"warn_{ass.scenario.value.lower()}_{tid}_{self._event_counter:04d}"
                state_map = {
                    DirectionalWarningLevel.LOW: WarningEventState.ACTIVE_LOW,
                    DirectionalWarningLevel.MEDIUM: WarningEventState.ACTIVE_MEDIUM,
                    DirectionalWarningLevel.HIGH: WarningEventState.ACTIVE_HIGH,
                }
                new_state = state_map.get(candidate_level, WarningEventState.ACTIVE_LOW)

                event = WarningEvent(
                    event_id=event_id,
                    track_id=tid,
                    object_class=feat.object_class,
                    scenario=ass.scenario.value,
                    direction=direction,
                    current_level=candidate_level,
                    previous_level=DirectionalWarningLevel.UNASSESSED,
                    state=new_state,
                    first_detected_timestamp=first_t,
                    confirmation_timestamp=timestamp,
                    activation_timestamp=timestamp,
                    last_updated_timestamp=timestamp,
                    risk_score=ass.risk_score,
                    risk_model=ass.model_name.value,
                    reliability=ass.reliability.value,
                    ttc=feat.ttc,
                    collision_urgency=ass.collision_urgency,
                    is_primary_threat=ass.is_primary_threat,
                    priority_rank=ass.priority_rank,
                    evidence=list(ass.evidence),
                )
                self._active_events[scenario_key] = event
                self._all_events.append(event)
                self._candidate_buffer.pop(scenario_key, None)

                # Record activation transition in warning history
                self._warning_history.append({
                    "event_id": event.event_id,
                    "track_id": tid,
                    "timestamp": timestamp,
                    "previous_level": "UNASSESSED",
                    "new_level": candidate_level.value,
                    "direction": direction.value,
                    "scenario": ass.scenario.value,
                    "transition_type": "ACTIVATION",
                    "triggering_evidence": candidate_reason,
                    "risk_score": ass.risk_score,
                    "risk_model": ass.model_name.value,
                    "reliability": ass.reliability.value,
                    "ttc": feat.ttc,
                    "event_state": new_state.value,
                })

                logger.info(
                    "Warning Activated: [%s] Event %s (Track %d, Scenario %s, Dir %s)",
                    candidate_level.value,
                    event_id,
                    tid,
                    ass.scenario.value,
                    direction.value,
                )

                return WarningOutput(
                    event_id=event.event_id,
                    track_id=tid,
                    object_class=event.object_class,
                    scenario=event.scenario,
                    direction=event.direction,
                    level=event.current_level,
                    state=event.state,
                    risk_score=event.risk_score,
                    risk_model=event.risk_model,
                    reliability=event.reliability,
                    ttc=event.ttc,
                    collision_urgency=event.collision_urgency,
                    is_primary_threat=event.is_primary_threat,
                    priority_rank=event.priority_rank,
                    timestamp=timestamp,
                    reason=candidate_reason,
                    escalated=False,
                    deescalated=False,
                )
            else:
                return None

        # ---------------------------------------------------------------------
        # Case B: Active event exists -> Update and Apply Hysteresis
        # ---------------------------------------------------------------------
        active_event.last_updated_timestamp = timestamp
        active_event.direction = direction
        active_event.risk_score = ass.risk_score
        active_event.ttc = feat.ttc
        active_event.collision_urgency = ass.collision_urgency
        active_event.is_primary_threat = ass.is_primary_threat
        active_event.priority_rank = ass.priority_rank
        active_event.evidence = list(ass.evidence)

        curr_order = _LEVEL_ORDER.get(active_event.current_level, 0)
        cand_order = _LEVEL_ORDER.get(candidate_level, 0)

        escalated = False
        deescalated = False

        # 1. Escalation check (candidate > current)
        if cand_order > curr_order:
            self._deescalation_buffer.pop(scenario_key, None)
            self._low_risk_start_time.pop(scenario_key, None)

            # Fast-track escalation for urgent threat (TTC <= 1.5s)
            is_fast_escalate = bool(
                candidate_level == DirectionalWarningLevel.HIGH
                and feat.ttc is not None
                and feat.ttc <= 1.5
            )

            # Accumulate escalation confirmation
            if scenario_key not in self._candidate_buffer:
                self._candidate_buffer[scenario_key] = deque(maxlen=5)
            self._candidate_buffer[scenario_key].append((timestamp, candidate_level, candidate_reason))

            if is_fast_escalate or len(self._candidate_buffer[scenario_key]) >= self.escalation_frames:
                prev_lvl = active_event.current_level
                active_event.previous_level = prev_lvl
                active_event.current_level = candidate_level
                active_event.escalation_timestamps.append((timestamp, prev_lvl.value, candidate_level.value))

                state_map = {
                    DirectionalWarningLevel.MEDIUM: WarningEventState.ACTIVE_MEDIUM,
                    DirectionalWarningLevel.HIGH: WarningEventState.ACTIVE_HIGH,
                }
                active_event.state = state_map.get(candidate_level, active_event.state)
                escalated = True
                self._candidate_buffer.pop(scenario_key, None)

                # Record escalation transition in warning history
                self._warning_history.append({
                    "event_id": active_event.event_id,
                    "track_id": tid,
                    "timestamp": timestamp,
                    "previous_level": prev_lvl.value,
                    "new_level": candidate_level.value,
                    "direction": direction.value,
                    "scenario": active_event.scenario,
                    "transition_type": "ESCALATION",
                    "triggering_evidence": candidate_reason,
                    "risk_score": ass.risk_score,
                    "risk_model": ass.model_name.value,
                    "reliability": ass.reliability.value,
                    "ttc": feat.ttc,
                    "event_state": active_event.state.value,
                })

                logger.info(
                    "Warning Escalated: Event %s [%s -> %s] Track %d",
                    active_event.event_id,
                    prev_lvl.value,
                    candidate_level.value,
                    tid,
                )

        # 2. De-escalation check (candidate < current, but still active)
        elif 0 < cand_order < curr_order:
            self._candidate_buffer.pop(scenario_key, None)
            self._low_risk_start_time.pop(scenario_key, None)

            if scenario_key not in self._deescalation_buffer:
                self._deescalation_buffer[scenario_key] = deque(maxlen=10)
            self._deescalation_buffer[scenario_key].append((timestamp, candidate_level))

            dt_deesc = timestamp - self._deescalation_buffer[scenario_key][0][0]
            if dt_deesc >= self.deescalation_duration_s:
                prev_lvl = active_event.current_level
                active_event.previous_level = prev_lvl
                active_event.current_level = candidate_level
                active_event.deescalation_timestamps.append((timestamp, prev_lvl.value, candidate_level.value))

                state_map = {
                    DirectionalWarningLevel.LOW: WarningEventState.ACTIVE_LOW,
                    DirectionalWarningLevel.MEDIUM: WarningEventState.ACTIVE_MEDIUM,
                }
                active_event.state = state_map.get(candidate_level, active_event.state)
                deescalated = True
                self._deescalation_buffer.pop(scenario_key, None)

                # Record de-escalation transition in warning history
                self._warning_history.append({
                    "event_id": active_event.event_id,
                    "track_id": tid,
                    "timestamp": timestamp,
                    "previous_level": prev_lvl.value,
                    "new_level": candidate_level.value,
                    "direction": direction.value,
                    "scenario": active_event.scenario,
                    "transition_type": "DEESCALATION",
                    "triggering_evidence": f"Sustained lower risk for {dt_deesc:.2f}s",
                    "risk_score": ass.risk_score,
                    "risk_model": ass.model_name.value,
                    "reliability": ass.reliability.value,
                    "ttc": feat.ttc,
                    "event_state": active_event.state.value,
                })

                logger.info(
                    "Warning De-escalated: Event %s [%s -> %s] Track %d",
                    active_event.event_id,
                    prev_lvl.value,
                    candidate_level.value,
                    tid,
                )

        # 3. Resolution check (candidate level is UNASSESSED / Safe)
        elif cand_order == 0:
            self._candidate_buffer.pop(scenario_key, None)
            self._deescalation_buffer.pop(scenario_key, None)

            if scenario_key not in self._low_risk_start_time:
                self._low_risk_start_time[scenario_key] = timestamp

            dt_resolve = timestamp - self._low_risk_start_time[scenario_key]
            if dt_resolve >= self.resolution_duration_s:
                self._resolve_event(scenario_key, timestamp, reason="THREAT_SUBSIDED_BELOW_ADVISORY")
                return None

        # Same level persists (Duplicate warning suppressed from re-alerting)
        else:
            self._candidate_buffer.pop(scenario_key, None)
            self._deescalation_buffer.pop(scenario_key, None)
            self._low_risk_start_time.pop(scenario_key, None)
            self._suppressed_duplicate_count += 1

        return WarningOutput(
            event_id=active_event.event_id,
            track_id=tid,
            object_class=active_event.object_class,
            scenario=active_event.scenario,
            direction=active_event.direction,
            level=active_event.current_level,
            state=active_event.state,
            risk_score=active_event.risk_score,
            risk_model=active_event.risk_model,
            reliability=active_event.reliability,
            ttc=active_event.ttc,
            collision_urgency=active_event.collision_urgency,
            is_primary_threat=active_event.is_primary_threat,
            priority_rank=active_event.priority_rank,
            timestamp=timestamp,
            reason=candidate_reason,
            escalated=escalated,
            deescalated=deescalated,
        )

    def _resolve_event(self, scenario_key: tuple[int, str], timestamp: float, reason: str = "") -> None:
        """Resolve and archive an active warning event."""
        event = self._active_events.pop(scenario_key, None)
        if event is not None:
            event.state = WarningEventState.RESOLVED
            event.resolution_timestamp = timestamp
            event.resolution_reason = reason
            if reason:
                event.evidence.append(f"RESOLVED_{reason}")

            # Record resolution in warning history
            self._warning_history.append({
                "event_id": event.event_id,
                "track_id": event.track_id,
                "timestamp": timestamp,
                "previous_level": event.current_level.value,
                "new_level": "UNASSESSED",
                "direction": event.direction.value,
                "scenario": event.scenario,
                "transition_type": "RESOLUTION",
                "triggering_evidence": reason or "Threat subsided below threshold",
                "risk_score": event.risk_score,
                "risk_model": event.risk_model,
                "reliability": event.reliability,
                "ttc": event.ttc,
                "event_state": WarningEventState.RESOLVED.value,
            })
            logger.info("Warning Resolved: Event %s Track %d (%s)", event.event_id, scenario_key[0], reason)

        self._candidate_buffer.pop(scenario_key, None)
        self._deescalation_buffer.pop(scenario_key, None)
        self._low_risk_start_time.pop(scenario_key, None)

    def _resolve_events_for_track(self, track_id: int, timestamp: float, reason: str = "") -> None:
        """Resolve all events associated with a specific track ID."""
        matching_keys = [k for k in self._active_events if k[0] == track_id]
        for k in matching_keys:
            self._resolve_event(k, timestamp, reason)

    def _purge_stale_events(self, current_time: float) -> None:
        """Expire events whose tracks have exceeded stale timeout without recovery."""
        for scenario_key, event in list(self._active_events.items()):
            tid = scenario_key[0]
            last_t = self._last_seen.get(tid, 0.0)
            if (current_time - last_t) > self.stale_timeout_s or (current_time < last_t):
                event.state = WarningEventState.EXPIRED
                event.resolution_timestamp = current_time
                event.resolution_reason = "STALE_TIMEOUT_EXPIRED"
                self._warning_history.append({
                    "event_id": event.event_id,
                    "track_id": tid,
                    "timestamp": current_time,
                    "previous_level": event.current_level.value,
                    "new_level": "UNASSESSED",
                    "direction": event.direction.value,
                    "scenario": event.scenario,
                    "transition_type": "EXPIRATION",
                    "triggering_evidence": "STALE_TIMEOUT_EXPIRED",
                    "risk_score": event.risk_score,
                    "risk_model": event.risk_model,
                    "reliability": event.reliability,
                    "ttc": event.ttc,
                    "event_state": WarningEventState.EXPIRED.value,
                })
                self._active_events.pop(scenario_key, None)
                self._candidate_buffer.pop(scenario_key, None)
                self._deescalation_buffer.pop(scenario_key, None)
                self._low_risk_start_time.pop(scenario_key, None)
                logger.info("Warning Expired: Event %s Track %d (Stale timeout)", event.event_id, tid)

    def finish_session(self, final_timestamp: float) -> None:
        """Gracefully close all active warning events upon video stream completion."""
        for scenario_key in list(self._active_events.keys()):
            self._resolve_event(scenario_key, final_timestamp, reason="VIDEO_STREAM_TERMINATED")

    def get_active_events(self) -> list[WarningEvent]:
        """Return all currently active warning events."""
        return list(self._active_events.values())

    def get_all_events(self) -> list[WarningEvent]:
        """Return complete historical log of all warning events."""
        return list(self._all_events)

    def get_warning_history(self) -> list[dict[str, Any]]:
        """Return chronological record of all warning state transitions."""
        return list(self._warning_history)

    def get_suppressed_duplicate_count(self) -> int:
        """Return total count of duplicate warnings suppressed."""
        return self._suppressed_duplicate_count

    def get_risk_assessment_details(self) -> list[dict[str, Any]]:
        """Return detailed per-indicator diagnostics and contributions for all evaluated objects."""
        return list(self._risk_assessment_details)
