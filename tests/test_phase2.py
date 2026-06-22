from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.factory import build_detector
from src.detection.yolo_detector import YoloDetector
from src.schemas import BoundingBox, Detection, Frame


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


def _make_mock_model(names: dict[int, str]) -> MagicMock:
    """Return a MagicMock that mimics a YOLO model with the given names."""
    model = MagicMock()
    model.names = names
    return model


def _make_mock_result(
    cls_ids: list[int],
    confs: list[float],
    xyxy: list[list[float]],
) -> MagicMock:
    """Build a mock ultralytics Results object."""
    box_mock = MagicMock()
    box_mock.cls.int.return_value.tolist.return_value = cls_ids
    box_mock.conf.float.return_value.tolist.return_value = confs
    box_mock.xyxy.float.return_value.tolist.return_value = xyxy

    result = MagicMock()
    result.boxes = box_mock
    return result


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestYoloDetector:
    def test_detect_returns_detections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = {0: "person", 1: "car"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[0, 1],
            confs=[0.95, 0.80],
            xyxy=[[10, 20, 100, 200], [50, 60, 150, 180]],
        )
        mock_model.predict.return_value = [mock_result]

        # Prevent actual YOLO import / instantiation
        monkeypatch.setattr(
            "src.detection.yolo_detector.YoloDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloDetector(model_path="fake.pt")
        frame = _make_frame()
        detections = detector.detect(frame)

        assert len(detections) == 2
        for d in detections:
            assert isinstance(d, Detection)
            assert isinstance(d.bbox, BoundingBox)

        assert detections[0].class_name == "person"
        assert detections[0].confidence == 0.95
        assert detections[1].class_name == "car"

    def test_detect_class_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "person", 1: "car", 2: "truck"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[0, 1, 2],
            confs=[0.9, 0.8, 0.7],
            xyxy=[[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
        )
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.detection.yolo_detector.YoloDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloDetector(
            model_path="fake.pt", class_filter=["car", "truck"]
        )
        frame = _make_frame()
        detections = detector.detect(frame)

        assert len(detections) == 2
        assert all(d.class_name in ("car", "truck") for d in detections)

    def test_detect_empty_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "person"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[], confs=[], xyxy=[]
        )
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.detection.yolo_detector.YoloDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloDetector(model_path="fake.pt")
        frame = _make_frame()
        detections = detector.detect(frame)

        assert detections == []

    def test_detect_no_boxes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "person"}
        mock_model = _make_mock_model(names)
        # boxes is None — no detections at all
        result = MagicMock()
        result.boxes = None
        mock_model.predict.return_value = [result]

        monkeypatch.setattr(
            "src.detection.yolo_detector.YoloDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloDetector(model_path="fake.pt")
        frame = _make_frame()
        detections = detector.detect(frame)

        assert detections == []


class TestFactory:
    def test_factory_returns_yolo(self) -> None:
        cfg = {"detector": {"type": "yolo", "model_path": "yolo11n.pt"}}
        detector = build_detector(cfg)
        assert isinstance(detector, YoloDetector)

    def test_factory_raises_on_unknown(self) -> None:
        cfg = {"detector": {"type": "faster_rcnn"}}
        with pytest.raises(ValueError, match="faster_rcnn"):
            build_detector(cfg)
