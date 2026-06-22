from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.lane.classical_lane_detector import ClassicalLaneDetector
from src.lane.factory import build_lane_detector
from src.schemas import BoundingBox, Frame, PerceivedObject
from src.schemas.lane_info import LaneInfo, LaneLine


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _synthetic_lane_frame(
    h: int = 480, w: int = 640,
) -> Frame:
    """Create a black frame with two diagonal white lane lines."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Left lane: from (180, 290) to (60, 479) — negative slope
    cv2.line(img, (180, 290), (60, 479), (255, 255, 255), thickness=8)
    # Right lane: from (460, 290) to (580, 479) — positive slope
    cv2.line(img, (460, 290), (580, 479), (255, 255, 255), thickness=8)
    return Frame(image=img, frame_id=1, timestamp=0.0, source="test")


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestClassicalLaneDetector:
    def test_detect_returns_lane_info(self) -> None:
        det = ClassicalLaneDetector()
        frame = _synthetic_lane_frame()
        info = det.detect(frame)
        assert isinstance(info, LaneInfo)

    def test_detects_both_lanes(self) -> None:
        det = ClassicalLaneDetector(
            canny_low=20, canny_high=60,
            hough_min_votes=5, hough_min_len=10, hough_max_gap=100,
        )
        frame = _synthetic_lane_frame()
        info = det.detect(frame)
        assert info.left_lane is not None
        assert info.right_lane is not None

    def test_center_roughly_center(self) -> None:
        """Lane center at bottom should be near image-center x (~320)."""
        det = ClassicalLaneDetector(
            canny_low=20, canny_high=60,
            hough_min_votes=5, hough_min_len=10, hough_max_gap=100,
        )
        frame = _synthetic_lane_frame()
        info = det.detect(frame)
        assert info.lane_center is not None
        c = info.lane_center.coeffs
        cx = c[0] * 479 * 479 + c[1] * 479 + c[2]
        assert 200 <= cx <= 450

    def test_confidence_nonzero(self) -> None:
        det = ClassicalLaneDetector(
            canny_low=20, canny_high=60,
            hough_min_votes=5, hough_min_len=10, hough_max_gap=100,
        )
        frame = _synthetic_lane_frame()
        info = det.detect(frame)
        assert info.confidence > 0

    def test_detect_blank_image(self) -> None:
        """All-black frame should return low confidence LaneInfo."""
        det = ClassicalLaneDetector()
        blank = Frame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=1, timestamp=0.0, source="test",
        )
        info = det.detect(blank)
        assert isinstance(info, LaneInfo)
        assert info.left_lane is None
        assert info.right_lane is None
        assert info.confidence == 0.0


class TestFactory:
    def test_returns_classical(self) -> None:
        det = build_lane_detector({"lane": {"enabled": True, "type": "classical"}})
        assert isinstance(det, ClassicalLaneDetector)

    def test_returns_classical_by_default(self) -> None:
        det = build_lane_detector({"lane": {"enabled": True}})
        assert isinstance(det, ClassicalLaneDetector)

    def test_returns_none_when_disabled(self) -> None:
        det = build_lane_detector({"lane": {"enabled": False}})
        assert det is None

    def test_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown lane detector type"):
            build_lane_detector({"lane": {"enabled": True, "type": "lanenet"}})

    def test_pipeline_integration(self) -> None:
        from src.core.pipeline import PerceptionPipeline
        pipe = PerceptionPipeline({})
        frame = type(
            "Frame", (), {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "frame_id": 1, "timestamp": 0.0, "source": "test",
            }
        )()
        result = pipe.process(frame)
        assert "lane" in result


class TestLaneRiskScore:
    """Test _score_lane integration via the risk engine."""

    def test_lane_risk_center_is_low(self) -> None:
        """Object near lane center → low lane risk contribution."""
        from src.risk.score_based_risk import ScoreBasedRiskEngine
        # Left lane at y=120: -0.5*120+350 = 290
        # Right lane at y=120: 0.5*120+270 = 330
        # Center=310, half-width=20
        info = LaneInfo(
            left_lane=LaneLine(coeffs=(0.0, -0.5, 350.0), confidence=0.9),
            right_lane=LaneLine(coeffs=(0.0, 0.5, 270.0), confidence=0.9),
            lane_center=LaneLine(coeffs=(0.0, 0.0, 310.0), confidence=0.9),
            confidence=0.9,
        )
        # bbox bottom-center at x=300 (offset=10 → score=25)
        obj = PerceivedObject(
            track_id=1, class_name="car",
            bbox=BoundingBox(290, 100, 310, 120),
            depth=20.0, distance=20.0, depth_confidence=0.9,
        )
        score = ScoreBasedRiskEngine._score_lane(info, obj)
        assert score == pytest.approx(25.0, abs=1.0)

    def test_lane_risk_edge_is_high(self) -> None:
        """Object near lane edge → elevated lane risk."""
        from src.risk.score_based_risk import ScoreBasedRiskEngine
        info = LaneInfo(
            left_lane=LaneLine(coeffs=(0.0, -0.5, 350.0), confidence=0.9),
            right_lane=LaneLine(coeffs=(0.0, 0.5, 270.0), confidence=0.9),
            lane_center=LaneLine(coeffs=(0.0, 0.0, 310.0), confidence=0.9),
            confidence=0.9,
        )
        # bbox bottom-center at x=340 (offset=30, half-width=20, (30/20)*50=75)
        obj = PerceivedObject(
            track_id=1, class_name="car",
            bbox=BoundingBox(330, 100, 350, 120),
            depth=20.0, distance=20.0, depth_confidence=0.9,
        )
        score = ScoreBasedRiskEngine._score_lane(info, obj)
        assert score == pytest.approx(75.0, abs=1.0)

    def test_lane_risk_none_lane_info(self) -> None:
        """None LaneInfo → score 0."""
        from src.risk.score_based_risk import ScoreBasedRiskEngine
        obj = PerceivedObject(
            track_id=1, class_name="car",
            bbox=BoundingBox(300, 100, 320, 120),
            depth=20.0, distance=20.0, depth_confidence=0.9,
        )
        score = ScoreBasedRiskEngine._score_lane(None, obj)
        assert score == 0.0
