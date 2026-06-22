from __future__ import annotations

import numpy as np
import pytest

from src.core.base import (
    BaseDepthEstimator,
    BaseDetector,
    BaseDistanceEstimator,
    BaseFrameSource,
    BaseLaneDetector,
    BaseSignDetector,
    BaseTracker,
)
from src.core.pipeline import PerceptionPipeline
from src.schemas import (
    BoundingBox,
    Detection,
    Frame,
    LaneInfo,
    LaneLine,
    PerceivedObject,
    RiskLevel,
    RiskObject,
    SignType,
    TrackedObject,
    TrafficSignInfo,
    WarningLevel,
)


class TestSchemas:
    def test_bounding_box(self) -> None:
        b = BoundingBox(10.0, 20.0, 100.0, 200.0)
        assert b.width == 90.0
        assert b.height == 180.0
        assert b.center == (55.0, 110.0)
        assert b.area == 16200.0

    def test_bounding_box_validation(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(100, 0, 50, 200)

    def test_bounding_box_iou(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 15, 15)
        assert a.iou(b) == pytest.approx(25 / 175, rel=1e-6)

    def test_bounding_box_iou_no_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert a.iou(b) == 0.0

    def test_frame(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        f = Frame(image=img, frame_id=1, timestamp=0.0, source="test")
        assert f.image.shape == (480, 640, 3)

    def test_detection(self) -> None:
        b = BoundingBox(0, 0, 10, 10)
        d = Detection(class_id=0, class_name="car", confidence=0.9, bbox=b)
        assert d.class_name == "car"

    def test_tracked_object(self) -> None:
        b = BoundingBox(0, 0, 10, 10)
        t = TrackedObject(track_id=1, class_name="car", confidence=0.9, bbox=b)
        assert t.track_id == 1

    def test_perceived_object(self) -> None:
        b = BoundingBox(0, 0, 10, 10)
        p = PerceivedObject(
            track_id=1,
            class_name="car",
            bbox=b,
            depth=5.0,
            distance=3.0,
            depth_confidence=0.8,
        )
        assert p.depth == 5.0

    def test_risk_object(self) -> None:
        r = RiskObject(
            track_id=1,
            risk_level=RiskLevel.HIGH,
            risk_score=85.0,
            warning_level=WarningLevel.CRITICAL,
        )
        assert r.risk_score == 85.0
        assert r.risk_level == RiskLevel.HIGH

    def test_lane_info(self) -> None:
        ll = LaneLine(coeffs=(0.0, 0.0, 320.0), confidence=0.9)
        info = LaneInfo(left_lane=ll, right_lane=ll, lane_center=ll, confidence=0.85)
        assert info.confidence == 0.85
        assert info.left_lane is not None

    def test_traffic_sign_info(self) -> None:
        b = BoundingBox(0, 0, 10, 10)
        s = TrafficSignInfo(
            sign_type=SignType.SPEED_LIMIT, bbox=b, confidence=0.95, value=60.0
        )
        assert s.sign_type == SignType.SPEED_LIMIT
        assert s.value == 60.0


class TestABCs:
    def test_base_frame_source_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseFrameSource()  # type: ignore[abstract]

    def test_base_detector_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDetector()  # type: ignore[abstract]

    def test_base_tracker_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseTracker()  # type: ignore[abstract]

    def test_base_depth_estimator_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDepthEstimator()  # type: ignore[abstract]

    def test_base_distance_estimator_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDistanceEstimator()  # type: ignore[abstract]

    def test_base_lane_detector_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLaneDetector()  # type: ignore[abstract]

    def test_base_sign_detector_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseSignDetector()  # type: ignore[abstract]


class TestPipeline:
    def test_pipeline_create(self) -> None:
        pipeline = PerceptionPipeline({})
        assert pipeline.config == {}

    def test_pipeline_process(self) -> None:
        pipeline = PerceptionPipeline({"test": 1})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert "frame" in result
        assert result["frame"] is frame
