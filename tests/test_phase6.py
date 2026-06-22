from __future__ import annotations

import numpy as np

from src.memory.track_history_manager import TrackHistoryManager, TrackSnapshot
from src.schemas import BoundingBox, Frame, PerceivedObject


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_frame(frame_id: int) -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=float(frame_id) / 30.0,
        source="test",
    )


def _make_perceived(
    track_id: int,
    distance: float = 10.0,
    confidence: float = 0.9,
) -> PerceivedObject:
    return PerceivedObject(
        track_id=track_id,
        class_name="car",
        bbox=BoundingBox(10, 20, 100, 200),
        depth=distance,
        distance=distance,
        depth_confidence=confidence,
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestTrackSnapshot:
    def test_snapshot_creation(self) -> None:
        bbox = BoundingBox(10, 20, 100, 200)
        snap = TrackSnapshot(
            frame_id=5, timestamp=1.0, bbox=bbox,
            distance=15.0, depth_confidence=0.8,
        )
        assert snap.frame_id == 5
        assert snap.distance == 15.0
        assert snap.depth_confidence == 0.8


class TestTrackHistoryManager:
    def test_maxlen_cap(self) -> None:
        """Pushing more than max_history_per_track should truncate oldest."""
        mgr = TrackHistoryManager(max_history_per_track=10, stale_after_frames=100)

        for fid in range(1, 101):
            frame = _make_frame(fid)
            obj = _make_perceived(track_id=1, distance=float(fid))
            mgr.update([obj], frame)

        history = mgr.get_history(1)
        assert len(history) == 10  # capped at 10
        # Oldest should be frames 91–100
        assert history[0].frame_id == 91
        assert history[-1].frame_id == 100

    def test_gc_stale_tracks(self) -> None:
        """Track not seen for stale_after_frames should be GC'd."""
        mgr = TrackHistoryManager(max_history_per_track=30, stale_after_frames=5)

        # Update track 1 for 5 frames
        for fid in range(1, 6):
            mgr.update([_make_perceived(track_id=1)], _make_frame(fid))

        assert 1 in mgr
        assert mgr.active_tracks() == {1}

        # Advance 10 frames without updating track 1
        mgr.update([], _make_frame(15))

        assert 1 not in mgr
        assert mgr.active_tracks() == set()

    def test_track_isolation(self) -> None:
        """Two tracks updated alternately maintain independent histories."""
        mgr = TrackHistoryManager(max_history_per_track=30, stale_after_frames=50)

        # 5 updates for track 1, 3 for track 2
        for fid in range(1, 6):
            mgr.update([_make_perceived(track_id=1)], _make_frame(fid))
        for fid in range(6, 9):
            mgr.update([_make_perceived(track_id=2)], _make_frame(fid))

        assert len(mgr.get_history(1)) == 5
        assert len(mgr.get_history(2)) == 3
        assert mgr.active_tracks() == {1, 2}

    def test_get_recent(self) -> None:
        mgr = TrackHistoryManager(max_history_per_track=30, stale_after_frames=50)

        for fid in range(1, 21):
            mgr.update([_make_perceived(track_id=1)], _make_frame(fid))

        recent = mgr.get_recent(1, 5)
        assert len(recent) == 5
        assert [s.frame_id for s in recent] == [16, 17, 18, 19, 20]

    def test_get_history_missing_track(self) -> None:
        mgr = TrackHistoryManager()
        assert mgr.get_history(999) == []
        assert mgr.get_recent(999, 5) == []

    def test_len_and_contains(self) -> None:
        mgr = TrackHistoryManager()
        assert len(mgr) == 0

        mgr.update([_make_perceived(track_id=1)], _make_frame(1))
        assert len(mgr) == 1
        assert 1 in mgr
        assert 2 not in mgr

    def test_none_distance_when_no_depth(self) -> None:
        """When depth_confidence is 0, snapshot distance should be None."""
        mgr = TrackHistoryManager()

        obj = _make_perceived(track_id=1, distance=0.0, confidence=0.0)
        mgr.update([obj], _make_frame(1))

        snap = mgr.get_history(1)[0]
        assert snap.distance is None
