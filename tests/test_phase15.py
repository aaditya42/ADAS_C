from __future__ import annotations

import numpy as np
import pytest

from src.schemas import BoundingBox, Frame, PerceivedObject
from src.schemas.risk_object import RiskLevel, RiskObject
from src.visualization.factory import build_renderer
from src.visualization.layers.base_layer import BaseLayer
from src.visualization.layers.depth_layer import DepthLayer
from src.visualization.layers.fps_layer import FPSLayer
from src.visualization.layers.hud_layer import HUDLayer
from src.visualization.layers.perceived_layer import PerceivedLayer
from src.visualization.renderer import Renderer


def _make_frame() -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=1, timestamp=0.0, source="test",
    )


def _make_output(add_perceived: bool = False, add_risk: bool = False) -> dict:
    frame = _make_frame()
    out: dict = {
        "frame": frame, "detections": [], "tracked": [],
        "depth_map": np.zeros((480, 640), dtype=np.float32),
        "perceived": [], "history": None, "motion": {}, "ttc": {},
        "risk": [], "alerts": [], "lane": None, "signs": [],
        "scene": None, "_fps": 30.0,
    }
    if add_perceived:
        out["perceived"] = [
            PerceivedObject(
                track_id=1, class_name="car",
                bbox=BoundingBox(100, 150, 300, 400),
                depth=20.0, distance=20.0, depth_confidence=0.9,
            )
        ]
    if add_risk:
        out["risk"] = [
            RiskObject(track_id=1, risk_level=RiskLevel.HIGH, risk_score=85.0, warning_level="critical"),
        ]
    return out


class TestRenderer:
    def test_returns_same_shape(self) -> None:
        layers: list[BaseLayer] = [
            PerceivedLayer(enabled=True), HUDLayer(enabled=True), FPSLayer(enabled=True),
        ]
        result = Renderer(layers).render(_make_output(add_perceived=True, add_risk=True))
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_differs_from_raw_frame(self) -> None:
        layers: list[BaseLayer] = [
            PerceivedLayer(enabled=True), HUDLayer(enabled=True), FPSLayer(enabled=True),
        ]
        out = _make_output(add_perceived=True, add_risk=True)
        result = Renderer(layers).render(out)
        diff = np.abs(result.astype(np.int16) - out["frame"].image.astype(np.int16)).sum()
        assert diff > 0

    def test_all_disabled_returns_raw(self) -> None:
        layers: list[BaseLayer] = [
            PerceivedLayer(enabled=False), HUDLayer(enabled=False), FPSLayer(enabled=False),
        ]
        out = _make_output()
        result = Renderer(layers).render(out)
        np.testing.assert_array_equal(result, out["frame"].image)

    def test_toggle_layer(self) -> None:
        d = DepthLayer(enabled=False)
        r = Renderer([d])
        assert not d.enabled
        assert r.toggle("depth")
        assert d.enabled
        assert not r.toggle("depth")
        assert not d.enabled

    def test_toggle_unknown(self) -> None:
        assert not Renderer([]).toggle("nonexistent")


class TestFactory:
    def test_builds_renderer(self) -> None:
        r = build_renderer({"visualization": {"enabled": True, "layers": ["fps"]}})
        assert isinstance(r, Renderer)
        assert len(r.layers) == 1

    def test_unknown_layer_skipped(self) -> None:
        r = build_renderer({"visualization": {"enabled": True, "layers": ["nonexistent"]}})
        assert len(r.layers) == 0


class TestPipelineIntegration:
    def test_render_returns_image(self) -> None:
        from src.core.pipeline import PerceptionPipeline
        pipe = PerceptionPipeline({})
        frame = _make_frame()
        result = pipe.process(frame)
        annotated = pipe.render(result)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (480, 640, 3)
