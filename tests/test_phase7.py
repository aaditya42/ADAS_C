from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from src.distance.base import BaseDistanceEstimator
from src.distance.calibrated_distance import CalibratedDistanceEstimator
from src.distance.factory import build_distance_estimator
from src.distance.relative_distance import RelativeDistanceEstimator
from src.schemas import BoundingBox, Frame, PerceivedObject


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_perceived(
    class_name: str = "car",
    x1: float = 50,
    y1: float = 50,
    x2: float = 150,
    y2: float = 150,
    depth: float = 10.0,
    distance: float = 10.0,
    depth_confidence: float = 0.9,
) -> PerceivedObject:
    return PerceivedObject(
        track_id=1,
        class_name=class_name,
        bbox=BoundingBox(x1, y1, x2, y2),
        depth=depth,
        distance=distance,
        depth_confidence=depth_confidence,
    )


def _make_frame() -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=1,
        timestamp=0.0,
        source="test",
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestRelativeDistanceEstimator:
    def test_returns_depth_value(self) -> None:
        est = RelativeDistanceEstimator()
        p = _make_perceived(depth=7.5, distance=7.5)
        result = est.estimate(p, None, _make_frame())
        assert result == pytest.approx(7.5)

    def test_returns_none_when_no_depth(self) -> None:
        est = RelativeDistanceEstimator()
        p = _make_perceived(depth=0.0, distance=0.0, depth_confidence=0.0)
        result = est.estimate(p, None, _make_frame())
        assert result is None


class TestCalibratedDistanceEstimator:
    def test_person_height_prior(self) -> None:
        """100 px tall person, focal=800, real_h=1.7 → distance ≈ 13.6 m."""
        est = CalibratedDistanceEstimator(
            focal_length_px=800.0,
            class_real_heights={"person": 1.7},
            fallback_height=1.5,
        )
        # bbox height = 100 px → y2 - y1 = 100
        p = _make_perceived(
            class_name="person", x1=50, y1=50, x2=150, y2=150
        )
        result = est.estimate(p, None, _make_frame())
        # (1.7 * 800) / 100 = 13.6
        assert result == pytest.approx(13.6, abs=0.1)

    def test_car_height_prior(self) -> None:
        """200 px tall car, focal=800, real_h=1.5 → distance ≈ 6.0 m."""
        est = CalibratedDistanceEstimator(
            focal_length_px=800.0,
            class_real_heights={"car": 1.5},
        )
        p = _make_perceived(
            class_name="car", x1=50, y1=50, x2=150, y2=250
        )
        result = est.estimate(p, None, _make_frame())
        assert result == pytest.approx(6.0, abs=0.1)

    def test_unknown_class_uses_fallback(self) -> None:
        """Unrecognised class uses fallback_height = 1.5."""
        est = CalibratedDistanceEstimator(
            focal_length_px=800.0,
            class_real_heights={},
            fallback_height=2.0,
        )
        p = _make_perceived(
            class_name="unknown_vehicle",
            x1=50, y1=50, x2=150, y2=100,
        )
        result = est.estimate(p, None, _make_frame())
        # (2.0 * 800) / 50 = 32.0
        assert result == pytest.approx(32.0, abs=0.1)

    def test_clip_min(self) -> None:
        """Extremely tall bbox should be clipped to 0.5."""
        est = CalibratedDistanceEstimator(
            focal_length_px=1000.0,
            class_real_heights={"car": 1.5},
        )
        # 1.5 * 1000 / 10000 = 0.15 → clipped to 0.5
        p = _make_perceived(
            class_name="car", x1=0, y1=0, x2=100, y2=10000,
        )
        result = est.estimate(p, None, _make_frame())
        assert result == pytest.approx(0.5)

    def test_clip_max(self) -> None:
        """Tiny bbox should be clipped to 200.0."""
        est = CalibratedDistanceEstimator(
            focal_length_px=1000.0,
            class_real_heights={"car": 1.5},
        )
        # 1.5 * 1000 / 1 = 1500 → clipped to 200
        p = _make_perceived(
            class_name="car", x1=0, y1=0, x2=100, y2=1,
        )
        result = est.estimate(p, None, _make_frame())
        assert result == pytest.approx(200.0)

    def test_clip_max_tiny_box(self) -> None:
        """Very small bbox (height=2) should be clipped to 200.0."""
        est = CalibratedDistanceEstimator(
            focal_length_px=1000.0,
            class_real_heights={"car": 1.5},
        )
        # (1.5 * 1000) / 2 = 750 → clipped to 200
        p = _make_perceived(
            class_name="car", x1=0, y1=0, x2=100, y2=2,
        )
        result = est.estimate(p, None, _make_frame())
        assert result == pytest.approx(200.0)


class TestFactory:
    def test_relative(self) -> None:
        est = build_distance_estimator({"distance": {"type": "relative"}})
        assert isinstance(est, RelativeDistanceEstimator)

    def test_calibrated(self) -> None:
        cfg = {
            "distance": {
                "type": "calibrated",
                "calibrated": {
                    "focal_length_px": 800,
                    "class_real_heights": {"person": 1.7},
                },
            }
        }
        est = build_distance_estimator(cfg)
        assert isinstance(est, CalibratedDistanceEstimator)

    def test_missing_section(self) -> None:
        est = build_distance_estimator({})
        assert not isinstance(est, (RelativeDistanceEstimator, CalibratedDistanceEstimator))
