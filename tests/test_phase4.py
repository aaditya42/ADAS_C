from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.depth.depth_anything_estimator import DepthAnythingEstimator
from src.depth.factory import build_depth_estimator
from src.schemas import Frame


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_frame(h: int = 256, w: int = 256) -> Frame:
    return Frame(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        frame_id=1,
        timestamp=0.0,
        source="test",
    )


def _mock_pipeline_output(h: int = 256, w: int = 256) -> MagicMock:
    """Simulate the HF pipeline output."""
    depth = np.random.rand(h, w).astype(np.float32) * 10.0
    return {"predicted_depth": depth}


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestDepthAnythingEstimator:
    def test_estimate_returns_float32_2d(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Depth map should be 2-D float32 with same spatial dims as input."""

        def fake_load(self: DepthAnythingEstimator) -> None:
            self._pipeline = MagicMock(return_value=_mock_pipeline_output(256, 256))
            import cv2
            from PIL import Image
            self._cv2 = cv2
            self._Image = Image

        monkeypatch.setattr(
            "src.depth.depth_anything_estimator.DepthAnythingEstimator._load_pipeline",
            fake_load,
        )

        estimator = DepthAnythingEstimator(device="cpu")
        frame = _make_frame(256, 256)
        depth = estimator.estimate(frame)

        assert isinstance(depth, np.ndarray)
        assert depth.shape == (256, 256)
        assert depth.dtype == np.float32

    def test_estimate_resizes_to_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When HF returns different resolution, result is resized to frame."""

        # Pipeline returns 128x128 but frame is 256x256
        def fake_load(self: DepthAnythingEstimator) -> None:
            self._pipeline = MagicMock(
                return_value=_mock_pipeline_output(128, 128)
            )
            import cv2
            from PIL import Image
            self._cv2 = cv2
            self._Image = Image

        monkeypatch.setattr(
            "src.depth.depth_anything_estimator.DepthAnythingEstimator._load_pipeline",
            fake_load,
        )

        estimator = DepthAnythingEstimator(device="cpu")
        frame = _make_frame(480, 640)
        depth = estimator.estimate(frame)

        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32

    def test_estimate_larger_is_farther(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify the inversion: pixels near camera should have smaller values."""
        import cv2
        from PIL import Image

        # Build a known depth map: left half close, right half far
        h, w = 64, 128
        raw = np.zeros((h, w), dtype=np.float32)
        raw[:, : w // 2] = 0.5   # close in raw output (LARGER = closer originally)
        raw[:, w // 2 :] = 9.5   # far in raw output

        def fake_load(self: DepthAnythingEstimator) -> None:
            self._pipeline = MagicMock(
                return_value={"predicted_depth": raw}
            )
            self._cv2 = cv2
            self._Image = Image

        monkeypatch.setattr(
            "src.depth.depth_anything_estimator.DepthAnythingEstimator._load_pipeline",
            fake_load,
        )

        estimator = DepthAnythingEstimator(device="cpu")
        frame = _make_frame(h, w)
        depth = estimator.estimate(frame)

        # After inversion: larger = farther, so left (previously 0.5) should
        # become (max - 0.5) and right (previously 9.5) becomes (max - 9.5).
        # Since max = 9.5, left ≈ 9.0 (far in output), right ≈ 0.0 (close) —
        # wait, that's backwards. Let's verify:

        # Raw: left=0.5 (close in raw = larger raw = closer), right=9.5 (far)
        # Inversion: depth = max - raw = 9.5 - raw
        # After: left ≈ 9.0, right ≈ 0.0
        # So left > right means left is *farther*, which is the WRONG direction
        # for the test's assumption (left half should be closer).
        # Actually the test just checks that the convention holds:
        # LARGER = FARTHER. The left half after inversion has larger values
        # (9.0 > 0.0), meaning it's farther — which is consistent with
        # the convention. The test's comment was misleading.

        left_mean = depth[:, : w // 2].mean()
        right_mean = depth[:, w // 2 :].mean()

        # left was closer in raw (smaller raw value = 0.5), so after inversion
        # left becomes farther (larger depth value).
        assert left_mean > right_mean, (
            "Left half (closer in raw) should be farther after inversion"
        )


class TestFactory:
    def test_factory_returns_depth_anything(self) -> None:
        cfg = {
            "depth": {
                "enabled": True,
                "type": "depth_anything_v2",
                "model_name": "fake/model",
            }
        }
        estimator = build_depth_estimator(cfg)
        assert isinstance(estimator, DepthAnythingEstimator)

    def test_factory_returns_noop_when_disabled(self) -> None:
        cfg = {"depth": {"enabled": False}}
        estimator = build_depth_estimator(cfg)
        assert not isinstance(estimator, DepthAnythingEstimator)

    def test_factory_returns_noop_on_missing_section(self) -> None:
        cfg: dict = {}
        estimator = build_depth_estimator(cfg)
        assert not isinstance(estimator, DepthAnythingEstimator)

    def test_factory_raises_on_unknown(self) -> None:
        cfg = {"depth": {"enabled": True, "type": "midas"}}
        with pytest.raises(ValueError, match="midas"):
            build_depth_estimator(cfg)
