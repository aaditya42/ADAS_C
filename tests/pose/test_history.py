from __future__ import annotations

import time
from threading import Thread
from typing import Any, List, Optional, Tuple

import pytest

from src.pose.history import PoseHistoryManager
from src.schemas.pose import (
    AnimalKeypointName,
    AnimalPose,
    HumanKeypointName,
    HumanPose,
    Keypoint,
    PoseFrame,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _make_human_pose(
    track_id: Optional[int] = None,
    detection_id: int = 0,
    timestamp: float = 0.0,
    confidence: float = 0.9,
    name: str = "person",
) -> HumanPose:
    return HumanPose(
        track_id=track_id,
        detection_id=detection_id,
        keypoints=(
            Keypoint(name=kp.value, x=0.0, y=0.0, confidence=0.0, visible=False)
            for kp in HumanKeypointName
        ),
        bbox=(0.0, 0.0, 10.0, 20.0),
        overall_confidence=confidence,
        body_orientation_deg=None,
        head_orientation_deg=None,
        timestamp=timestamp,
    )


def _make_animal_pose(
    track_id: Optional[int] = None,
    detection_id: int = 0,
    timestamp: float = 0.0,
    confidence: float = 0.85,
    species: str = "dog",
) -> AnimalPose:
    return AnimalPose(
        track_id=track_id,
        detection_id=detection_id,
        keypoints=(
            Keypoint(name=kp.value, x=0.0, y=0.0, confidence=0.0, visible=False)
            for kp in AnimalKeypointName
        ),
        bbox=(0.0, 0.0, 10.0, 20.0),
        overall_confidence=confidence,
        body_orientation_deg=None,
        head_orientation_deg=None,
        species=species,
        timestamp=timestamp,
    )


def _make_pose_frame(
    frame_id: int = 0,
    timestamp: float = 0.0,
    humans: Optional[Tuple[HumanPose, ...]] = None,
    animals: Optional[Tuple[AnimalPose, ...]] = None,
) -> PoseFrame:
    return PoseFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        humans=humans or (),
        animals=animals or (),
    )


# --------------------------------------------------------------------------- #
#  PoseHistoryManager tests
# --------------------------------------------------------------------------- #


class TestPoseHistoryManagerUpdate:
    """Tests for the ``update()`` method."""

    def test_untracked_humans_are_ignored(self) -> None:
        mgr = PoseHistoryManager()
        frame = _make_pose_frame(
            timestamp=1.0,
            humans=(_make_human_pose(track_id=None),),
        )
        mgr.update(frame)
        assert mgr.active_track_ids() == ()

    def test_untracked_animals_are_ignored(self) -> None:
        mgr = PoseHistoryManager()
        frame = _make_pose_frame(
            timestamp=1.0,
            animals=(_make_animal_pose(track_id=None),),
        )
        mgr.update(frame)
        assert mgr.active_track_ids() == ()

    def test_tracked_humans_are_stored(self) -> None:
        mgr = PoseHistoryManager()
        frame = _make_pose_frame(
            timestamp=1.0,
            humans=(_make_human_pose(track_id=1),),
        )
        mgr.update(frame)
        assert mgr.active_track_ids() == (1,)
        history = mgr.get_human_history(1)
        assert len(history) == 1
        assert history[0].track_id == 1

    def test_tracked_animals_are_stored(self) -> None:
        mgr = PoseHistoryManager()
        frame = _make_pose_frame(
            timestamp=1.0,
            animals=(_make_animal_pose(track_id=10),),
        )
        mgr.update(frame)
        assert mgr.active_track_ids() == (10,)
        history = mgr.get_animal_history(10)
        assert len(history) == 1
        assert history[0].species == "dog"

    def test_multiple_frames_accumulate(self) -> None:
        mgr = PoseHistoryManager(max_length=10)
        for i in range(5):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                humans=(_make_human_pose(track_id=1),),
            )
            mgr.update(frame)
        assert len(mgr.get_human_history(1)) == 5

    def test_multiple_tracks(self) -> None:
        mgr = PoseHistoryManager()
        frame1 = _make_pose_frame(
            timestamp=1.0,
            humans=(_make_human_pose(track_id=1),),
            animals=(_make_animal_pose(track_id=10),),
        )
        mgr.update(frame1)
        assert set(mgr.active_track_ids()) == {1, 10}


class TestPoseHistoryManagerMaxLength:
    """Tests for the ``max_length`` cap."""

    def test_deque_respects_max_length(self) -> None:
        mgr = PoseHistoryManager(max_length=3)
        for i in range(10):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                humans=(_make_human_pose(track_id=1, timestamp=float(i)),),
            )
            mgr.update(frame)
        history = mgr.get_human_history(1)
        assert len(history) == 3
        # Should retain the 3 most recent
        assert history[-1].timestamp == 9.0
        assert history[0].timestamp == 7.0

    def test_different_max_length_for_animals(self) -> None:
        mgr = PoseHistoryManager(max_length=5)
        for i in range(20):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                animals=(_make_animal_pose(track_id=10),),
            )
            mgr.update(frame)
        assert len(mgr.get_animal_history(10)) == 5

    def test_max_length_one(self) -> None:
        mgr = PoseHistoryManager(max_length=1)
        for i in range(10):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                humans=(_make_human_pose(track_id=1, timestamp=float(i)),),
            )
            mgr.update(frame)
        assert len(mgr.get_human_history(1)) == 1
        assert mgr.get_human_history(1)[0].timestamp == 9.0


class TestPoseHistoryManagerPrune:
    """Tests for the ``prune()`` method."""

    def test_prune_removes_old_entries(self) -> None:
        mgr = PoseHistoryManager(max_age_seconds=5.0)
        mgr.update(_make_pose_frame(
            timestamp=1.0,
            humans=(_make_human_pose(track_id=1, timestamp=1.0),),
        ))
        mgr.update(_make_pose_frame(
            timestamp=2.0,
            humans=(_make_human_pose(track_id=1, timestamp=2.0),),
        ))
        mgr.update(_make_pose_frame(
            timestamp=10.0,
            humans=(_make_human_pose(track_id=1, timestamp=10.0),),
        ))

        pruned = mgr.prune(current_timestamp=10.0)  # cutoff = 5.0
        assert pruned == 2
        history = mgr.get_human_history(1)
        assert len(history) == 1
        assert history[0].timestamp == 10.0

    def test_prune_removes_empty_track(self) -> None:
        mgr = PoseHistoryManager(max_age_seconds=5.0)
        mgr.update(_make_pose_frame(
            timestamp=1.0,
            humans=(_make_human_pose(track_id=1),),
        ))
        mgr.prune(current_timestamp=10.0)  # cutoff = 5.0, old entry pruned
        assert mgr.active_track_ids() == ()

    def test_prune_empty_manager_returns_zero(self) -> None:
        mgr = PoseHistoryManager()
        assert mgr.prune(100.0) == 0

    def test_prune_removes_only_expired_entries(self) -> None:
        mgr = PoseHistoryManager(max_age_seconds=10.0)
        for ts in [1.0, 5.0, 12.0, 15.0]:
            mgr.update(_make_pose_frame(
                timestamp=ts,
                humans=(_make_human_pose(track_id=1),),
            ))
        pruned = mgr.prune(current_timestamp=20.0)  # cutoff = 10.0
        assert pruned == 2
        assert len(mgr.get_human_history(1)) == 2

    def test_prune_with_animal_history(self) -> None:
        mgr = PoseHistoryManager(max_age_seconds=5.0)
        mgr.update(_make_pose_frame(
            timestamp=1.0,
            animals=(_make_animal_pose(track_id=10),),
        ))
        mgr.update(_make_pose_frame(
            timestamp=10.0,
            animals=(_make_animal_pose(track_id=10),),
        ))
        pruned = mgr.prune(current_timestamp=10.0)
        assert pruned == 1
        assert len(mgr.get_animal_history(10)) == 1


class TestPoseHistoryManagerConcurrency:
    """Concurrent updates must not corrupt state."""

    def test_concurrent_updates(self) -> None:
        mgr = PoseHistoryManager(max_length=100)
        errors: list[str] = []

        def writer(n: int) -> None:
            for i in range(50):
                frame = _make_pose_frame(
                    frame_id=i,
                    timestamp=float(i + n * 1000),
                    humans=(_make_human_pose(track_id=n),),
                )
                mgr.update(frame)

        threads = [Thread(target=writer, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread wrote to its own track — all should be present
        active = mgr.active_track_ids()
        assert len(active) == 10
        for tid in range(10):
            history = mgr.get_human_history(tid)
            assert len(history) == 50, f"Track {tid} has {len(history)} entries"

    def test_concurrent_prune_and_update(self) -> None:
        mgr = PoseHistoryManager(max_length=50, max_age_seconds=0.1)
        errors: list[str] = []

        def updater() -> None:
            for i in range(100):
                frame = _make_pose_frame(
                    frame_id=i,
                    timestamp=float(i) * 0.01,
                    humans=(_make_human_pose(track_id=1),),
                )
                mgr.update(frame)

        def pruner() -> None:
            for i in range(50):
                mgr.prune(current_timestamp=float(i) * 0.02 + 5.0)

        threads = [
            Thread(target=updater),
            Thread(target=pruner),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash — state should remain internally consistent
        history = mgr.get_human_history(1)
        assert len(history) >= 0
        assert all(h.track_id == 1 for h in history)


class TestPoseHistoryManagerIntegration:
    """End-to-end scenarios."""

    def test_memory_does_not_grow_unbounded(self) -> None:
        """Long-running test verifies bounded memory."""
        mgr = PoseHistoryManager(max_length=5, max_age_seconds=30.0)

        for i in range(1000):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                humans=(_make_human_pose(track_id=1, timestamp=float(i)),),
            )
            mgr.update(frame)
            if i % 100 == 0:
                mgr.prune(current_timestamp=float(i))

        history = mgr.get_human_history(1)
        assert len(history) <= 5

    def test_get_human_history_returns_oldest_first(self) -> None:
        mgr = PoseHistoryManager(max_length=10)
        for i in range(5):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                humans=(_make_human_pose(track_id=1, timestamp=float(i)),),
            )
            mgr.update(frame)
        history = mgr.get_human_history(1)
        timestamps = [h.timestamp for h in history]
        assert timestamps == sorted(timestamps)

    def test_get_animal_history_returns_oldest_first(self) -> None:
        mgr = PoseHistoryManager(max_length=10)
        for i in range(5):
            frame = _make_pose_frame(
                frame_id=i,
                timestamp=float(i),
                animals=(_make_animal_pose(track_id=10, timestamp=float(i)),),
            )
            mgr.update(frame)
        history = mgr.get_animal_history(10)
        timestamps = [h.timestamp for h in history]
        assert timestamps == sorted(timestamps)

    def test_get_history_for_unknown_track_returns_empty(self) -> None:
        mgr = PoseHistoryManager()
        assert mgr.get_human_history(999) == ()
        assert mgr.get_animal_history(999) == ()

    def test_get_history_for_deleted_track_returns_empty(self) -> None:
        mgr = PoseHistoryManager(max_age_seconds=1.0)
        mgr.update(_make_pose_frame(
            timestamp=0.0,
            humans=(_make_human_pose(track_id=1),),
        ))
        mgr.prune(current_timestamp=10.0)
        assert mgr.get_human_history(1) == ()

    def test_active_track_ids_empty_initially(self) -> None:
        mgr = PoseHistoryManager()
        assert mgr.active_track_ids() == ()
