from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.schemas import BoundingBox, Frame
from src.schemas.traffic_sign_info import SignType, TrafficSignInfo
from src.signs.base import BaseSignDetector
from src.signs.factory import build_sign_detector
from src.signs.yolo_sign_detector import YoloSignDetector


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
    model = MagicMock()
    model.names = names
    return model


def _make_mock_result(
    cls_ids: list[int],
    confs: list[float],
    xyxy: list[list[float]],
) -> MagicMock:
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

class TestYoloSignDetector:
    def test_detect_returns_signs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "stop", 1: "speed_limit_50"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[0, 1],
            confs=[0.95, 0.80],
            xyxy=[[10, 20, 100, 200], [50, 60, 150, 180]],
        )
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloSignDetector(model_path="fake_signs.pt")
        frame = _make_frame()
        signs = detector.detect(frame)

        assert len(signs) == 2
        for s in signs:
            assert isinstance(s, TrafficSignInfo)
            assert isinstance(s.bbox, BoundingBox)

        assert signs[0].sign_type == SignType.STOP
        assert signs[0].confidence == pytest.approx(0.95)
        assert signs[0].value is None

        assert signs[1].sign_type == SignType.SPEED_LIMIT
        assert signs[1].value == 50.0
        assert signs[1].confidence == pytest.approx(0.80)

    def test_parses_numeric_suffix_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """speed_limit_30 -> value=30.0, speed_limit_120 -> value=120.0"""
        names = {0: "speed_limit_30", 1: "speed_limit_120"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[0, 1],
            confs=[0.9, 0.85],
            xyxy=[[0, 0, 20, 20], [30, 30, 60, 60]],
        )
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloSignDetector(model_path="fake.pt")
        signs = detector.detect(_make_frame())

        assert len(signs) == 2
        assert signs[0].value == 30.0
        assert signs[1].value == 120.0

    def test_unmapped_class_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "some_weird_sign"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[0],
            confs=[0.7],
            xyxy=[[10, 10, 50, 50]],
        )
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloSignDetector(model_path="fake.pt")
        signs = detector.detect(_make_frame())

        assert len(signs) == 1
        assert signs[0].sign_type == SignType.UNKNOWN
        assert signs[0].value is None

    def test_empty_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "stop"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(cls_ids=[], confs=[], xyxy=[])
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloSignDetector(model_path="fake.pt")
        signs = detector.detect(_make_frame())
        assert signs == []

    def test_no_boxes_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "stop"}
        mock_model = _make_mock_model(names)
        result = MagicMock()
        result.boxes = None
        mock_model.predict.return_value = [result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        detector = YoloSignDetector(model_path="fake.pt")
        signs = detector.detect(_make_frame())
        assert signs == []

    def test_custom_name_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = {0: "custom_stop"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(
            cls_ids=[0],
            confs=[0.9],
            xyxy=[[0, 0, 10, 10]],
        )
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        custom_map = {"custom_stop": SignType.STOP}
        detector = YoloSignDetector(
            model_path="fake.pt", class_name_to_sign=custom_map
        )
        signs = detector.detect(_make_frame())
        assert len(signs) == 1
        assert signs[0].sign_type == SignType.STOP


class TestFactory:
    def test_returns_none_when_disabled(self) -> None:
        assert build_sign_detector({"signs": {"enabled": False}}) is None

    def test_returns_none_when_no_signs_section(self) -> None:
        assert build_sign_detector({}) is None

    def test_returns_yolo_sign_detector(self) -> None:
        cfg = {
            "signs": {
                "enabled": True,
                "type": "yolo_signs",
                "model_path": "fake.pt",
            }
        }
        detector = build_sign_detector(cfg)
        assert isinstance(detector, YoloSignDetector)

    def test_raises_on_unknown_type(self) -> None:
        cfg = {
            "signs": {
                "enabled": True,
                "type": "faster_rcnn",
            }
        }
        with pytest.raises(ValueError, match="faster_rcnn"):
            build_sign_detector(cfg)


class TestPipelineIntegration:
    def test_signs_key_in_output(self) -> None:
        """Pipeline process() returns 'signs' key even when disabled."""
        from src.core.pipeline import PerceptionPipeline

        pipe = PerceptionPipeline({"signs": {"enabled": False}})
        frame = _make_frame()
        result = pipe.process(frame)
        assert "signs" in result
        assert result["signs"] == []

    def test_signs_key_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pipeline with signs enabled returns 'signs' key."""
        from src.core.pipeline import PerceptionPipeline

        names = {0: "stop"}
        mock_model = _make_mock_model(names)
        mock_result = _make_mock_result(cls_ids=[], confs=[], xyxy=[])
        mock_model.predict.return_value = [mock_result]

        monkeypatch.setattr(
            "src.signs.yolo_sign_detector.YoloSignDetector._load_model",
            lambda self: setattr(self, "_model", mock_model)
            or setattr(self, "_names", names),
        )

        cfg = {
            "signs": {
                "enabled": True,
                "type": "yolo_signs",
                "model_path": "fake.pt",
            }
        }
        pipe = PerceptionPipeline(cfg)
        frame = _make_frame()
        result = pipe.process(frame)
        assert "signs" in result
        assert isinstance(result["signs"], list)


class TestBaseSignDetector:
    def test_abc_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseSignDetector()  # type: ignore[abstract]
