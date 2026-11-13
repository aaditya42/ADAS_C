from __future__ import annotations

import math
import uuid
from collections import deque
from typing import Any, Optional

from src.lane_detection.interfaces import LaneDetectionResult
from src.overtaking.schemas import (
    CorridorPosition,
    IdentityContinuityStatus,
    OvertakingCycleResult,
    OvertakingEvent,
    OvertakingEventStatus,
    OvertakingObservation,
    OvertakingState,
)
from src.schemas import BoundingBox, Frame, PerceivedObject, TrackedObject
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _interpolate_x_at_y(
    points: tuple[tuple[int, int], ...],
    target_y: float,
) -> float | None:
    """Interpolate x-coordinate along lane points at given y."""
    if len(points) < 2:
        return None
    sorted_pts = sorted(points, key=lambda p: p[1])
    if target_y <= sorted_pts[0][1]:
        return float(sorted_pts[0][0])
    if target_y >= sorted_pts[-1][1]:
        return float(sorted_pts[-1][0])
    for i in range(len(sorted_pts) - 1):
        y0, y1 = float(sorted_pts[i][1]), float(sorted_pts[i + 1][1])
        if y0 <= target_y <= y1:
            x0, x1 = float(sorted_pts[i][0]), float(sorted_pts[i + 1][0])
            t = (target_y - y0) / (y1 - y0) if y1 != y0 else 0.0
            return x0 + t * (x1 - x0)
    return None


class OvertakingDetector:
    """Accurate timing and lifecycle manager for ego-vehicle overtaking detection.

    Responsibilities:
    - Maintains separate, unmixed timestamps for candidate start, confirmed start,
      confirmation arrival, candidate end, confirmed end, and confirmation delays.
    - Manages concurrent and sequential overtaking candidates independently.
    - Handles tracking interruptions via configurable grace periods and continuity checks.
    - Resolves continuous track instances preventing ID-switch corruption.
    - Enforces timestamp monotonicity and provides video frame-rate fallback.
    - Exposes a query interface for downstream reporting and visualization.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        ot_cfg = cfg.get("overtaking", {})

        self.enabled: bool = bool(ot_cfg.get("enabled", True))

        # Target classes
        target_classes_cfg = ot_cfg.get(
            "target_classes", ["car", "truck", "bus", "motorcycle", "van"]
        )
        self.target_classes: set[str] = {c.lower() for c in target_classes_cfg}

        # Detection & timing parameters
        det_cfg = ot_cfg.get("detection", {})
        self.min_initial_observations: int = int(det_cfg.get("min_initial_observations", 3))
        self.scale_growth_start_threshold: float = float(det_cfg.get("scale_growth_start_threshold", 1.20))
        self.scale_growth_passing_threshold: float = float(det_cfg.get("scale_growth_passing_threshold", 1.45))
        self.vertical_drop_ratio: float = float(det_cfg.get("vertical_drop_ratio", 0.05))
        self.ahead_max_y_ratio: float = float(det_cfg.get("ahead_max_y_ratio", 0.65))
        self.passing_y_ratio: float = float(det_cfg.get("passing_y_ratio", 0.55))
        self.exit_margin_pixels: float = float(det_cfg.get("exit_margin_pixels", 60.0))
        self.min_event_duration_seconds: float = float(det_cfg.get("min_event_duration_seconds", 0.4))
        self.max_event_duration_seconds: float = float(det_cfg.get("max_event_duration_seconds", 15.0))
        self.start_confirmation_duration: float = float(det_cfg.get("start_confirmation_duration", 0.25))
        self.completion_confirmation_duration: float = float(det_cfg.get("completion_confirmation_duration", 0.20))

        # Tracking & identity parameters
        track_cfg = ot_cfg.get("tracking", {})
        self.maximum_track_gap_seconds: float = float(track_cfg.get("maximum_track_gap_seconds", 0.5))
        self.track_loss_grace_seconds: float = float(track_cfg.get("track_loss_grace_seconds", 0.4))
        self.maximum_active_events: int = int(track_cfg.get("maximum_active_events", 10))
        self.identity_reassociation_enabled: bool = bool(track_cfg.get("identity_reassociation_enabled", True))
        self.identity_reassociation_max_gap: float = float(track_cfg.get("identity_reassociation_max_gap", 0.4))
        self.timestamp_fallback_enabled: bool = bool(track_cfg.get("timestamp_fallback_enabled", True))
        self.fallback_fps: float = float(track_cfg.get("fallback_fps", 30.0))

        # Per-candidate state
        self._candidates: dict[int, OvertakingEvent] = {}
        self._last_seen_timestamp: dict[int, float] = {}
        self._last_seen_frame_id: dict[int, int] = {}
        self._instance_counters: dict[int, int] = {}
        self._current_instance_id: dict[int, str] = {}
        self._track_histories: dict[int, Any] = {}
        self._last_frame_timestamp: float = -1.0

        # Cycle notification buffers
        self._last_cycle_started: list[OvertakingEvent] = []
        self._last_cycle_completed: list[OvertakingEvent] = []
        self._last_cycle_incomplete: list[OvertakingEvent] = []

        # Session-wide storage
        self._all_events: list[OvertakingEvent] = []
        self._completed_events: list[OvertakingEvent] = []
        self._incomplete_events: list[OvertakingEvent] = []
        self._uncertain_events: list[OvertakingEvent] = []

    # ------------------------------------------------------------------ #
    #  Query Interface
    # ------------------------------------------------------------------ #

    @property
    def active_events(self) -> list[OvertakingEvent]:
        """All currently active events (STARTED or PASSING)."""
        return [
            e for e in self._candidates.values()
            if e.state in (OvertakingState.OVERTAKING_STARTED, OvertakingState.PASSING)
        ]

    @property
    def completed_events(self) -> list[OvertakingEvent]:
        """All successfully completed events recorded in the current session."""
        return list(self._completed_events)

    @property
    def incomplete_events(self) -> list[OvertakingEvent]:
        """All events terminated as INCOMPLETE in the current session."""
        return list(self._incomplete_events)

    @property
    def uncertain_events(self) -> list[OvertakingEvent]:
        """All events marked UNCERTAIN due to ambiguous evidence."""
        return list(self._uncertain_events)

    @property
    def all_events(self) -> list[OvertakingEvent]:
        """All events (active, completed, incomplete) recorded in this session."""
        return list(self._all_events)

    def get_newly_started_events(self) -> list[OvertakingEvent]:
        return list(self._last_cycle_started)

    def get_active_events(self) -> list[OvertakingEvent]:
        return self.active_events

    def get_newly_completed_events(self) -> list[OvertakingEvent]:
        return list(self._last_cycle_completed)

    def get_incomplete_events(self) -> list[OvertakingEvent]:
        return self.incomplete_events

    def get_uncertain_events(self) -> list[OvertakingEvent]:
        return self.uncertain_events

    def get_finalized_events(self) -> list[OvertakingEvent]:
        """All resolved events (completed + incomplete + uncertain) for reporting."""
        return [
            e for e in self._all_events
            if e.status in (
                OvertakingEventStatus.COMPLETED,
                OvertakingEventStatus.INCOMPLETE,
                OvertakingEventStatus.UNCERTAIN,
            )
        ]

    def get_event_by_id(self, event_id: str) -> OvertakingEvent | None:
        """Find an event record by its unique event_id."""
        for e in self._all_events:
            if e.event_id == event_id:
                return e
        return None

    # ------------------------------------------------------------------ #
    #  Session Lifecycle
    # ------------------------------------------------------------------ #

    def start_session(
        self, session_id: str | None = None, initial_timestamp: float | None = None
    ) -> None:
        """Initialize or reset state at the beginning of a video session."""
        self.reset()
        if initial_timestamp is not None:
            self._last_frame_timestamp = initial_timestamp

    def reset(self) -> None:
        """Clear all session tracking state and histories."""
        self._candidates.clear()
        self._last_seen_timestamp.clear()
        self._last_seen_frame_id.clear()
        self._instance_counters.clear()
        self._current_instance_id.clear()
        self._track_histories.clear()
        self._last_frame_timestamp = -1.0
        self._last_cycle_started.clear()
        self._last_cycle_completed.clear()
        self._last_cycle_incomplete.clear()
        self._all_events.clear()
        self._completed_events.clear()
        self._incomplete_events.clear()
        self._uncertain_events.clear()

    def finish_session(self, final_timestamp: float | None = None) -> None:
        """Finalize all active events upon video stream completion.

        Ensures active manoeuvres are terminated as INCOMPLETE with reason VIDEO_ENDED,
        producing no fabricated completion timestamp or duration.
        """
        for tid, event in list(self._candidates.items()):
            if event.status == OvertakingEventStatus.IN_PROGRESS:
                event.finalize_incomplete("VIDEO_ENDED", status=OvertakingEventStatus.INCOMPLETE)
                self._incomplete_events.append(event)
        self._candidates.clear()

    # ------------------------------------------------------------------ #
    #  Frame Processing Cycle
    # ------------------------------------------------------------------ #

    def process(
        self,
        objects: list[TrackedObject | PerceivedObject],
        lane_info: LaneDetectionResult | None,
        scene_context: Any | None,
        frame: Frame,
    ) -> OvertakingCycleResult:
        """Execute one online detection and lifecycle update cycle.

        Args:
            objects: Detected and tracked objects in current frame.
            lane_info: Optional lane detection result.
            scene_context: Optional synthesised scene context.
            frame: Video frame with image, frame_id, and timestamp.

        Returns:
            OvertakingCycleResult summary of active and newly resolved events.
        """
        if not self.enabled:
            return OvertakingCycleResult([], [], [], [])

        current_time = self._resolve_frame_timestamp(frame)
        h, w = frame.image.shape[:2]

        newly_started: list[OvertakingEvent] = []
        newly_completed: list[OvertakingEvent] = []
        newly_incomplete: list[OvertakingEvent] = []

        seen_track_ids: set[int] = set()

        # Step 1: Ingest active objects
        for obj in objects:
            tid = obj.track_id
            seen_track_ids.add(tid)

            if obj.class_name.lower() not in self.target_classes:
                continue

            bbox = obj.bbox
            if bbox.width <= 0 or bbox.height <= 0:
                continue

            # Resolve track instance identity (detect gaps > maximum_track_gap_seconds)
            inst_id = self._resolve_track_instance(tid, current_time)

            # Spatial corridor evaluation
            c_pos, in_corr = self._evaluate_corridor_position(bbox, lane_info, w)

            # Depth validity check
            rel_depth: float | None = None
            depth_valid = False
            if hasattr(obj, "depth_confidence") and getattr(obj, "depth_confidence", 0) > 0:
                rel_depth = float(getattr(obj, "distance", 0.0))
                depth_valid = True

            lane_valid = lane_info is not None and getattr(lane_info, "confidence", 0.0) >= 0.2

            obs = OvertakingObservation(
                frame_id=frame.frame_id,
                timestamp=current_time,
                bbox=bbox,
                center_x=float(bbox.center[0]),
                center_y=float(bbox.center[1]),
                bottom_y=float(bbox.y2),
                bbox_area=float(bbox.area),
                relative_depth=rel_depth,
                corridor_position=c_pos,
                in_ego_corridor=in_corr,
            )

            # Candidate retrieval or creation
            event = self._candidates.get(tid)
            if event is None:
                # Check for ID-switch reassociation with an unassigned lost candidate
                if self.identity_reassociation_enabled:
                    event = self._try_reassociate_candidate(tid, inst_id, obj, obs, current_time)

            if event is None:
                # Check eligibility for new candidate (must be ahead in or near forward corridor)
                if obs.bottom_y <= self.ahead_max_y_ratio * h:
                    if len(self._candidates) < self.maximum_active_events:
                        event = OvertakingEvent(
                            event_id=f"ovt_{uuid.uuid4().hex[:8]}",
                            track_id=tid,
                            track_instance_id=inst_id,
                            vehicle_class=obj.class_name,
                            state=OvertakingState.VEHICLE_AHEAD,
                            status=OvertakingEventStatus.IN_PROGRESS,
                            first_observed_timestamp=current_time,
                            last_observed_timestamp=current_time,
                            initial_position=(obs.center_x, obs.bottom_y),
                            latest_position=(obs.center_x, obs.bottom_y),
                            initial_bbox=bbox,
                            latest_bbox=bbox,
                            initial_depth=rel_depth,
                            latest_depth=rel_depth,
                            initial_corridor_position=c_pos,
                            latest_corridor_position=c_pos,
                            lane_data_validity=lane_valid,
                            depth_data_validity=depth_valid,
                        )
                        event.observations.append(obs)
                        event.total_observations = 1
                        event.valid_observation_count = 1
                        self._candidates[tid] = event
                        self._all_events.append(event)
            else:
                # Update existing candidate
                event.observations.append(obs)
                event.total_observations += 1
                event.valid_observation_count += 1
                event.latest_bbox = bbox
                event.latest_position = (obs.center_x, obs.bottom_y)
                if rel_depth is not None:
                    event.latest_depth = rel_depth
                event.latest_corridor_position = c_pos
                event.last_observed_timestamp = current_time
                event.missing_observation_duration = 0.0
                event.lane_data_validity = lane_valid
                event.depth_data_validity = depth_valid

                # Advance state machine
                self._update_candidate_fsm(
                    event=event,
                    obs=obs,
                    current_time=current_time,
                    frame_w=w,
                    frame_h=h,
                    newly_started=newly_started,
                    newly_completed=newly_completed,
                    newly_incomplete=newly_incomplete,
                )

            self._last_seen_timestamp[tid] = current_time
            self._last_seen_frame_id[tid] = frame.frame_id

        # Step 2: Handle candidates not observed in this frame
        self._check_unseen_candidates(
            seen_track_ids=seen_track_ids,
            current_time=current_time,
            frame_w=w,
            frame_h=h,
            newly_completed=newly_completed,
            newly_incomplete=newly_incomplete,
        )

        self._last_cycle_started = newly_started
        self._last_cycle_completed = newly_completed
        self._last_cycle_incomplete = newly_incomplete

        return OvertakingCycleResult(
            active_events=self.active_events,
            newly_started=newly_started,
            newly_completed=newly_completed,
            newly_incomplete=newly_incomplete,
        )

    # ------------------------------------------------------------------ #
    #  Internal State Machine & Timing Logic
    # ------------------------------------------------------------------ #

    def _update_candidate_fsm(
        self,
        event: OvertakingEvent,
        obs: OvertakingObservation,
        current_time: float,
        frame_w: int,
        frame_h: int,
        newly_started: list[OvertakingEvent],
        newly_completed: list[OvertakingEvent],
        newly_incomplete: list[OvertakingEvent],
    ) -> None:
        if event.status != OvertakingEventStatus.IN_PROGRESS:
            return

        baseline_obs = event.observations[0]
        base_area = max(1.0, baseline_obs.bbox_area)
        base_y2 = baseline_obs.bottom_y
        scale_ratio = obs.bbox_area / base_area
        vertical_drop = (obs.bottom_y - base_y2) / float(frame_h)

        # ------------------------------------------------------------- #
        # Stage 1: VEHICLE_AHEAD -> OVERTAKING_STARTED
        # ------------------------------------------------------------- #
        if event.state == OvertakingState.VEHICLE_AHEAD:
            if event.total_observations >= self.min_initial_observations:
                approaching = (
                    scale_ratio >= self.scale_growth_start_threshold
                    or vertical_drop >= self.vertical_drop_ratio
                )

                depth_consistent = True
                if (
                    baseline_obs.relative_depth is not None
                    and obs.relative_depth is not None
                ):
                    depth_consistent = obs.relative_depth <= (baseline_obs.relative_depth * 1.05)

                if approaching and depth_consistent:
                    # Record candidate start timestamp at first indication
                    if event.candidate_start_timestamp is None:
                        event.candidate_start_timestamp = current_time

                    # Backtrack to find the earliest observation exhibiting approach
                    earliest_start_t = baseline_obs.timestamp
                    for o in event.observations:
                        if (
                            o.bbox_area >= base_area * 1.08
                            or (o.bottom_y - base_y2) / float(frame_h) >= 0.02
                        ):
                            earliest_start_t = o.timestamp
                            break

                    elapsed_approach = current_time - earliest_start_t
                    if elapsed_approach >= self.start_confirmation_duration:
                        # Confirmed start!
                        event.confirm_start(
                            start_time=earliest_start_t,
                            confirmation_time=current_time,
                            candidate_start_time=event.candidate_start_timestamp,
                        )
                        newly_started.append(event)
                        logger.debug(
                            "Overtaking STARTED for track %d (%s) at t=%.3fs (backtracked to %.3fs, delay=%.3fs)",
                            event.track_id,
                            event.vehicle_class,
                            current_time,
                            earliest_start_t,
                            event.start_confirmation_delay_seconds or 0.0,
                        )
                else:
                    # Reset candidate start if approach ceased
                    event.candidate_start_timestamp = None

        # ------------------------------------------------------------- #
        # Stage 2: OVERTAKING_STARTED -> PASSING (or INCOMPLETE)
        # ------------------------------------------------------------- #
        elif event.state == OvertakingState.OVERTAKING_STARTED:
            # Check for aborted approach
            if scale_ratio < 0.95 and vertical_drop < -0.05:
                event.finalize_incomplete("ABORTED_APPROACH_TARGET_PULLING_AWAY")
                newly_incomplete.append(event)
                self._incomplete_events.append(event)
                return

            # Check timeout
            if (
                event.start_timestamp is not None
                and (current_time - event.start_timestamp) > self.max_event_duration_seconds
            ):
                event.finalize_incomplete("TIMEOUT_EXCEEDED")
                newly_incomplete.append(event)
                self._incomplete_events.append(event)
                return

            is_passing_zone = (obs.bottom_y / float(frame_h)) >= self.passing_y_ratio
            is_large_scale = scale_ratio >= self.scale_growth_passing_threshold
            corridor_cx = float(frame_w) / 2.0

            if is_passing_zone and is_large_scale:
                event.state = OvertakingState.PASSING
                event.passing_side = "LEFT" if obs.center_x < corridor_cx else "RIGHT"
                logger.debug(
                    "Overtaking PASSING for track %d on %s side at t=%.3fs",
                    event.track_id,
                    event.passing_side,
                    current_time,
                )

        # ------------------------------------------------------------- #
        # Stage 3: PASSING -> OVERTAKING_COMPLETED (or INCOMPLETE)
        # ------------------------------------------------------------- #
        elif event.state == OvertakingState.PASSING:
            # Check timeout
            if (
                event.start_timestamp is not None
                and (current_time - event.start_timestamp) > self.max_event_duration_seconds
            ):
                event.finalize_incomplete("TIMEOUT_EXCEEDED")
                newly_incomplete.append(event)
                self._incomplete_events.append(event)
                return

            # Detect candidate completion conditions while in view
            at_left_exit = (
                event.passing_side == "LEFT"
                and obs.bbox.x1 <= self.exit_margin_pixels
                and (obs.bottom_y / float(frame_h)) >= self.passing_y_ratio
            )
            at_right_exit = (
                event.passing_side == "RIGHT"
                and obs.bbox.x2 >= (frame_w - self.exit_margin_pixels)
                and (obs.bottom_y / float(frame_h)) >= self.passing_y_ratio
            )
            cleared_bottom = (obs.bottom_y / float(frame_h)) >= 0.95

            if at_left_exit or at_right_exit or cleared_bottom:
                if event.candidate_end_timestamp is None:
                    event.candidate_end_timestamp = current_time

    def _check_unseen_candidates(
        self,
        seen_track_ids: set[int],
        current_time: float,
        frame_w: int,
        frame_h: int,
        newly_completed: list[OvertakingEvent],
        newly_incomplete: list[OvertakingEvent],
    ) -> None:
        """Handle candidates missing from current frame, enforcing grace period and exit verification."""
        for tid, event in list(self._candidates.items()):
            if tid in seen_track_ids or event.status != OvertakingEventStatus.IN_PROGRESS:
                continue

            last_t = event.last_observed_timestamp
            missing_dur = current_time - last_t
            event.missing_observation_duration = missing_dur

            # 1. Grace period: hold state without termination
            if missing_dur <= self.track_loss_grace_seconds:
                continue

            # 2. Exceeded grace period: evaluate resolution
            if event.state == OvertakingState.PASSING:
                latest = event.observations[-1]
                at_left_boundary = (
                    event.passing_side == "LEFT"
                    and latest.bbox.x1 <= self.exit_margin_pixels * 1.5
                    and (latest.bottom_y / float(frame_h)) >= (self.passing_y_ratio * 0.9)
                )
                at_right_boundary = (
                    event.passing_side == "RIGHT"
                    and latest.bbox.x2 >= (frame_w - self.exit_margin_pixels * 1.5)
                    and (latest.bottom_y / float(frame_h)) >= (self.passing_y_ratio * 0.9)
                )
                cleared_bottom = (latest.bottom_y / float(frame_h)) >= 0.90

                if at_left_boundary or at_right_boundary or cleared_bottom:
                    # Earliest supported completion is the observation when it reached the exit boundary
                    end_obs_time = (
                        event.candidate_end_timestamp
                        if event.candidate_end_timestamp is not None
                        else last_t
                    )
                    event.finalize_completion(
                        end_time=end_obs_time,
                        confirmation_time=current_time,
                        confidence=0.95,
                    )
                    newly_completed.append(event)
                    self._completed_events.append(event)
                    logger.info(
                        "Overtaking COMPLETED for track %d (%s) on %s side! Duration: %.3fs (delay=%.3fs)",
                        event.track_id,
                        event.vehicle_class,
                        event.passing_side,
                        event.duration_seconds or 0.0,
                        event.completion_confirmation_delay_seconds or 0.0,
                    )
                else:
                    event.finalize_incomplete("LOST_WHILE_PASSING_NOT_AT_EXIT_BOUNDARY")
                    newly_incomplete.append(event)
                    self._incomplete_events.append(event)

            elif event.state == OvertakingState.OVERTAKING_STARTED:
                event.finalize_incomplete("LOST_IN_MIDFIELD")
                newly_incomplete.append(event)
                self._incomplete_events.append(event)

            elif event.state == OvertakingState.VEHICLE_AHEAD:
                event.finalize_incomplete("LEFT_WITHOUT_OVERTAKING")
                self._incomplete_events.append(event)

            # Evict from active candidate tracking
            del self._candidates[tid]

    # ------------------------------------------------------------------ #
    #  Helper Methods
    # ------------------------------------------------------------------ #

    def _resolve_frame_timestamp(self, frame: Frame) -> float:
        """Resolve monotonic presentation timestamp with source FPS fallback."""
        ts: float | None = None
        if frame.timestamp is not None:
            try:
                raw_ts = float(frame.timestamp)
                if raw_ts >= 0.0:
                    ts = raw_ts
            except (ValueError, TypeError):
                pass

        if ts is None and self.timestamp_fallback_enabled:
            # Fallback to frame_id / fps
            ts = float(frame.frame_id) / max(1.0, self.fallback_fps)

        if ts is None:
            ts = max(0.0, self._last_frame_timestamp + 1.0 / self.fallback_fps)

        # Enforce strict timestamp monotonicity
        if self._last_frame_timestamp >= 0.0 and ts < self._last_frame_timestamp:
            logger.debug(
                "Non-monotonic timestamp detected (ts=%.4f < last=%.4f); clamping to last + 1e-4",
                ts,
                self._last_frame_timestamp,
            )
            ts = self._last_frame_timestamp + 1e-4

        self._last_frame_timestamp = ts
        return ts

    def _resolve_track_instance(self, track_id: int, current_time: float) -> str:
        """Assign unique continuous track episode identity, incrementing upon large gaps."""
        last_t = self._last_seen_timestamp.get(track_id)
        if last_t is None:
            self._instance_counters[track_id] = 1
            inst = f"{track_id}_inst1"
            self._current_instance_id[track_id] = inst
            return inst

        dt = current_time - last_t
        if dt > self.maximum_track_gap_seconds:
            cnt = self._instance_counters.get(track_id, 1) + 1
            self._instance_counters[track_id] = cnt
            inst = f"{track_id}_inst{cnt}"
            self._current_instance_id[track_id] = inst
            return inst

        return self._current_instance_id.get(track_id, f"{track_id}_inst1")

    def _try_reassociate_candidate(
        self,
        new_tid: int,
        inst_id: str,
        obj: TrackedObject | PerceivedObject,
        obs: OvertakingObservation,
        current_time: float,
    ) -> OvertakingEvent | None:
        """Attempt to reassociate an active candidate under a new ID if track was lost briefly."""
        for tid, cand in list(self._candidates.items()):
            if cand.status != OvertakingEventStatus.IN_PROGRESS:
                continue
            if cand.state not in (OvertakingState.OVERTAKING_STARTED, OvertakingState.PASSING):
                continue

            time_gap = current_time - cand.last_observed_timestamp
            if time_gap <= self.identity_reassociation_max_gap:
                # Check class match and spatial proximity
                if cand.vehicle_class.lower() == obj.class_name.lower():
                    last_obs = cand.observations[-1]
                    dist_x = abs(obs.center_x - last_obs.center_x)
                    dist_y = abs(obs.bottom_y - last_obs.bottom_y)
                    # Plausible spatial continuity during short gap
                    if dist_x < 150.0 and dist_y < 120.0:
                        logger.info(
                            "Reassociating lost candidate track %d -> %d (%s) at t=%.3fs",
                            tid,
                            new_tid,
                            cand.vehicle_class,
                            current_time,
                        )
                        del self._candidates[tid]
                        cand.track_id = new_tid
                        cand.track_instance_id = inst_id
                        cand.identity_continuity_status = IdentityContinuityStatus.SUSPECTED_ID_SWITCH
                        if tid not in cand.reassociated_track_ids:
                            cand.reassociated_track_ids.append(tid)
                        self._candidates[new_tid] = cand
                        return cand
        return None

    @staticmethod
    def _evaluate_corridor_position(
        bbox: BoundingBox,
        lane_info: LaneDetectionResult | None,
        frame_width: int,
    ) -> tuple[CorridorPosition, bool]:
        """Determine whether the object center is within the ego corridor or adjacent."""
        cx = float(bbox.center[0])
        by = float(bbox.y2)

        if lane_info is not None and getattr(lane_info, "confidence", 0.0) >= 0.2:
            left_pts = getattr(lane_info, "left_lane_points", ())
            right_pts = getattr(lane_info, "right_lane_points", ())
            if left_pts and right_pts:
                lx = _interpolate_x_at_y(left_pts, by)
                rx = _interpolate_x_at_y(right_pts, by)
                if lx is not None and rx is not None:
                    l_bound = min(lx, rx)
                    r_bound = max(lx, rx)
                    if l_bound <= cx <= r_bound:
                        return CorridorPosition.IN_EGO_CORRIDOR, True
                    elif cx < l_bound:
                        return CorridorPosition.LEFT_OF_CORRIDOR, False
                    else:
                        return CorridorPosition.RIGHT_OF_CORRIDOR, False

        # Fallback to central 50% image corridor
        w = float(frame_width)
        c_left = 0.25 * w
        c_right = 0.75 * w
        if c_left <= cx <= c_right:
            return CorridorPosition.IN_EGO_CORRIDOR, True
        elif cx < c_left:
            return CorridorPosition.LEFT_OF_CORRIDOR, False
        else:
            return CorridorPosition.RIGHT_OF_CORRIDOR, False
