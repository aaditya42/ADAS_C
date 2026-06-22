from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.schemas import BoundingBox, Detection, Frame, TrackedObject
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.tracking.factory import build_tracker


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_frame() -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=1,
        timestamp=0.0,
        source="test",
    )


def _make_detection(
    x1: float, y1: float, x2: float, y2: float, class_id: int = 0,
    class_name: str = "car", conf: float = 0.9,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bbox=BoundingBox(x1, y1, x2, y2),
    )


def _mock_sv_detections(
    n: int,
    track_ids: list[int],
    class_ids: list[int] | None = None,
) -> MagicMock:
    """Build a mock sv.Detections with the given number of entries."""
    cids = class_ids if class_ids is not None else [0] * n
    dets = MagicMock()
    dets.__len__.return_value = n
    dets.tracker_id = np.array(track_ids, dtype=float)
    dets.xyxy = np.array(
        [[10, 20, 100, 200] for _ in range(n)], dtype=np.float32
    )
    dets.confidence = np.array([0.9] * n, dtype=np.float32)
    dets.class_id = np.array(cids, dtype=int)
    return dets


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestByteTrackTracker:
    def test_update_returns_tracked_objects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_bt = MagicMock()
        # Return 2 tracked detections with distinct class_ids
        mock_sv_dets = _mock_sv_detections(2, [1, 2], class_ids=[0, 1])
        mock_bt.update_with_detections.return_value = mock_sv_dets

        import supervision as sv
        monkeypatch.setattr(sv, "ByteTrack", lambda **kw: mock_bt)

        tracker = ByteTrackTracker()
        frame = _make_frame()
        detections = [
            _make_detection(10, 20, 100, 200, class_id=0, class_name="car"),
            _make_detection(50, 60, 150, 180, class_id=1, class_name="person"),
        ]

        tracked = tracker.update(detections, frame)

        assert len(tracked) == 2
        for t in tracked:
            assert isinstance(t, TrackedObject)
            assert isinstance(t.bbox, BoundingBox)
            assert t.track_id in (1, 2)

        ids = {t.track_id: t.class_name for t in tracked}
        assert ids[1] == "car"
        assert ids[2] == "person"

    def test_update_track_id_persistence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two consecutive updates should assign the same track_id."""
        mock_bt = MagicMock()

        dets1 = _mock_sv_detections(1, [5])
        dets2 = _mock_sv_detections(1, [5])
        mock_bt.update_with_detections.side_effect = [dets1, dets2]

        import supervision as sv
        monkeypatch.setattr(sv, "ByteTrack", lambda **kw: mock_bt)

        tracker = ByteTrackTracker()
        frame = _make_frame()

        det = _make_detection(10, 20, 100, 200, class_id=0, class_name="car")

        result1 = tracker.update([det], frame)
        result2 = tracker.update([det], frame)

        assert len(result1) == 1
        assert len(result2) == 1
        assert result1[0].track_id == result2[0].track_id == 5

    def test_update_empty_detections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_bt = MagicMock()
        import supervision as sv

        empty_dets = MagicMock()
        empty_dets.__len__.return_value = 0

        mock_bt.update_with_detections.return_value = empty_dets
        monkeypatch.setattr(sv, "ByteTrack", lambda **kw: mock_bt)

        tracker = ByteTrackTracker()
        frame = _make_frame()

        tracked = tracker.update([], frame)
        assert tracked == []

    def test_update_skips_negative_tracker_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Detections with tracker_id < 0 should be skipped."""
        mock_bt = MagicMock()
        dets = MagicMock()
        dets.__len__.return_value = 2
        dets.tracker_id = np.array([1, -1], dtype=float)
        dets.xyxy = np.array(
            [[10, 20, 100, 200], [50, 60, 150, 180]], dtype=np.float32
        )
        dets.confidence = np.array([0.9, 0.8], dtype=np.float32)
        dets.class_id = np.array([0, 1], dtype=int)
        mock_bt.update_with_detections.return_value = dets

        import supervision as sv
        monkeypatch.setattr(sv, "ByteTrack", lambda **kw: mock_bt)

        tracker = ByteTrackTracker()
        frame = _make_frame()

        detections = [
            _make_detection(10, 20, 100, 200, class_id=0, class_name="car"),
            _make_detection(50, 60, 150, 180, class_id=1, class_name="person"),
        ]

        tracked = tracker.update(detections, frame)
        assert len(tracked) == 1
        assert tracked[0].track_id == 1


class TestFactory:
    def test_factory_returns_bytetrack(self) -> None:
        cfg = {"tracker": {"type": "bytetrack"}}
        trk = build_tracker(cfg)
        assert isinstance(trk, ByteTrackTracker)

    def test_factory_returns_noop_on_missing_section(self) -> None:
        cfg: dict = {}
        trk = build_tracker(cfg)
        assert not isinstance(trk, ByteTrackTracker)

    def test_factory_raises_on_unknown(self) -> None:
        cfg = {"tracker": {"type": "sift"}}
        with pytest.raises(ValueError, match="sift"):
            build_tracker(cfg)
