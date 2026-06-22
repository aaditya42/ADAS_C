from __future__ import annotations

import pytest

from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector
from src.lane_detection.lane_detector_factory import LaneDetectorFactory


class TestCLRNetFactory:
    def test_factory_returns_clrnet_by_default(self) -> None:
        det = LaneDetectorFactory.create({"lane": {"enabled": True}})
        assert isinstance(det, CLRNetLaneDetector)

    def test_factory_returns_clrnet_explicit(self) -> None:
        det = LaneDetectorFactory.create(
            {"lane": {"enabled": True, "model_type": "clrnet"}}
        )
        assert isinstance(det, CLRNetLaneDetector)

    def test_factory_returns_none_when_disabled(self) -> None:
        det = LaneDetectorFactory.create({"lane": {"enabled": False}})
        assert det is None

    def test_factory_returns_clrnet_with_empty_config(self) -> None:
        det = LaneDetectorFactory.create({})
        assert det is not None
        assert isinstance(det, CLRNetLaneDetector)

    def test_factory_raises_on_unknown(self) -> None:
        cfg = {"lane": {"enabled": True, "model_type": "nonexistent"}}
        with pytest.raises(ValueError, match="Unknown lane detector"):
            LaneDetectorFactory.create(cfg)

    def test_register_new_detector(self) -> None:
        class _FakeDetector:
            def detect(self, frame):
                from src.lane_detection.interfaces import LaneDetectionResult
                return LaneDetectionResult.empty()

        LaneDetectorFactory.register("fake", _FakeDetector)
        try:
            det = LaneDetectorFactory.create(
                {"lane": {"enabled": True, "model_type": "fake"}}
            )
            assert isinstance(det, _FakeDetector)
        finally:
            del LaneDetectorFactory._registry["fake"]

    def test_passes_clrnet_config_parameters(self) -> None:
        cfg = {
            "lane": {
                "enabled": True,
                "model_type": "clrnet",
                "model_path": "weights/my_checkpoint.pth",
                "input_width": 640,
                "input_height": 360,
                "conf_threshold": 0.3,
                "lane_confidence_threshold": 0.4,
                "max_lanes": 3,
                "temporal_smoothing": False,
                "smoothing_window": 2,
            },
        }
        det = LaneDetectorFactory.create(cfg)
        assert isinstance(det, CLRNetLaneDetector)
        assert det._model_path == "weights/my_checkpoint.pth"
        assert det._input_w == 640
        assert det._input_h == 360
        assert det._conf_threshold == 0.3
        assert det._lane_conf_threshold == 0.4
        assert det._max_lanes == 3
        assert not det._temporal_smoothing
        assert det._smoothing_window == 2
