from __future__ import annotations

import math
from typing import Any, Optional

from src.behavior.base import (
    BaseAnimalBehaviorClassifier,
    BaseHumanBehaviorClassifier,
    LaneContext,
    MotionEstimate,
    TrackHistory,
)
from src.behavior.config import BehaviorConfig
from src.behavior.factory import build_behavior_classifiers
from src.memory.track_history_manager import TrackHistoryManager
from src.pose.history import PoseHistoryManager
from src.schemas.behavior import BehaviorFrame, BehaviorState
from src.schemas.lane_info import LaneInfo
from src.schemas.pose import PoseFrame
from src.utils.logger import get_logger


class BehaviorSubsystem:
    """Orchestrates per-track behavior classification for humans and animals.

    Owns the human/animal classifiers constructed via the factory and
    iterates over a ``PoseFrame``, pulling pose history, track history,
    motion estimates, and lane context for each tracked entity before
    delegating to the appropriate classifier.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._logger = get_logger(f"{__name__}.BehaviorSubsystem")

        raw_behavior = config.get("behavior", {})
        self._cfg = BehaviorConfig.from_dict(raw_behavior)
        bundle = build_behavior_classifiers(config)
        self._human_classifier: Optional[BaseHumanBehaviorClassifier] = bundle.human
        self._animal_classifier: Optional[BaseAnimalBehaviorClassifier] = bundle.animal

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def process(
        self,
        pose_frame: Optional[PoseFrame],
        pose_history_mgr: Optional[PoseHistoryManager],
        track_history_mgr: TrackHistoryManager,
        motion_frame: dict[int, Any],
        lane_info: Optional[LaneInfo],
        frame_id: int,
        timestamp: float,
    ) -> BehaviorFrame:
        if pose_frame is None:
            return BehaviorFrame(frame_id=frame_id, timestamp=timestamp)

        lane_context = self._build_lane_context(lane_info)
        states: list[BehaviorState] = []

        if self._human_classifier is not None:
            for human_pose in pose_frame.humans:
                tid = human_pose.track_id
                if tid is None:
                    continue
                p_hist = (
                    pose_history_mgr.get_human_history(tid)
                    if pose_history_mgr is not None
                    else ()
                )
                state = self._human_classifier.classify(
                    pose_history=p_hist,
                    track_history=self._build_track_history(track_history_mgr, tid),
                    motion=self._build_motion_estimate(motion_frame, tid, track_history_mgr),
                    lane_context=lane_context,
                    timestamp=timestamp,
                )
                states.append(state)

        if self._animal_classifier is not None:
            for animal_pose in pose_frame.animals:
                tid = animal_pose.track_id
                if tid is None:
                    continue
                p_hist = (
                    pose_history_mgr.get_animal_history(tid)
                    if pose_history_mgr is not None
                    else ()
                )
                state = self._animal_classifier.classify(
                    pose_history=p_hist,
                    track_history=self._build_track_history(track_history_mgr, tid),
                    motion=self._build_motion_estimate(motion_frame, tid, track_history_mgr),
                    lane_context=lane_context,
                    timestamp=timestamp,
                )
                states.append(state)

        return BehaviorFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            states=tuple(states),
        )

    @staticmethod
    def _build_lane_context(lane_info: Optional[LaneInfo]) -> Optional[LaneContext]:
        if lane_info is None:
            return None
        return LaneContext(
            left_boundary=lane_info.left_lane,
            right_boundary=lane_info.right_lane,
            confidence=lane_info.confidence,
        )

    @staticmethod
    def _build_track_history(
        track_history_mgr: TrackHistoryManager,
        track_id: int,
    ) -> Optional[TrackHistory]:
        snapshots = track_history_mgr.get_history(track_id)
        if not snapshots:
            return None
        positions: list[tuple[float, float]] = []
        timestamps: list[float] = []
        for s in snapshots:
            cx = (s.bbox.x1 + s.bbox.x2) / 2.0
            by = s.bbox.y2
            positions.append((cx, by))
            timestamps.append(s.timestamp)
        return TrackHistory(
            track_id=track_id,
            positions=tuple(positions),
            timestamps=tuple(timestamps),
        )

    @staticmethod
    def _build_motion_estimate(
        motion_frame: dict[int, Any],
        track_id: int,
        track_history_mgr: TrackHistoryManager,
    ) -> Optional[MotionEstimate]:
        rel_motion = motion_frame.get(track_id)
        speed = rel_motion.speed_mps if rel_motion is not None else None
        accel = rel_motion.acceleration_mps2 if rel_motion is not None else None
        conf = rel_motion.confidence if rel_motion is not None else 0.0
        heading = _compute_heading_from_track_history(track_history_mgr, track_id)
        if speed is None and heading is None and accel is None:
            return None
        return MotionEstimate(
            track_id=track_id,
            speed_mps=speed,
            heading_deg=heading,
            acceleration_mps2=accel,
            confidence=conf,
        )


def _compute_heading_from_track_history(
    track_history_mgr: TrackHistoryManager,
    track_id: int,
) -> Optional[float]:
    snapshots = track_history_mgr.get_history(track_id)
    if len(snapshots) < 2:
        return None
    first = snapshots[0]
    last = snapshots[-1]
    dx = ((last.bbox.x1 + last.bbox.x2) / 2.0) - ((first.bbox.x1 + first.bbox.x2) / 2.0)
    dy = last.bbox.y2 - first.bbox.y2
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return None
    heading = math.degrees(math.atan2(dx, -dy))
    return heading
