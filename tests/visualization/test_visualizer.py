from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.visualization.overlays.behavior_overlay import BehaviorOverlay
from src.visualization.overlays.pose_overlay import PoseOverlay
from src.visualization.renderer import Renderer
from src.visualization.visualizer import build_visualizer


class TestBuildVisualizerConfig:

    def test_overlays_enabled_by_default(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
            },
        }
        renderer = build_visualizer(cfg)
        assert isinstance(renderer, Renderer)
        layer_types = [type(l).__name__ for l in renderer.layers]
        assert "PoseOverlay" in layer_types
        assert "BehaviorOverlay" in layer_types

    def test_pose_overlay_disabled(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
                "pose_overlay": {"enabled": False},
            },
        }
        renderer = build_visualizer(cfg)
        layer_types = [type(l).__name__ for l in renderer.layers]
        assert "PoseOverlay" not in layer_types
        assert "BehaviorOverlay" in layer_types

    def test_behavior_overlay_disabled(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
                "behavior_overlay": {"enabled": False},
            },
        }
        renderer = build_visualizer(cfg)
        layer_types = [type(l).__name__ for l in renderer.layers]
        assert "PoseOverlay" in layer_types
        assert "BehaviorOverlay" not in layer_types

    def test_both_overlays_disabled(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
                "pose_overlay": {"enabled": False},
                "behavior_overlay": {"enabled": False},
            },
        }
        renderer = build_visualizer(cfg)
        layer_types = [type(l).__name__ for l in renderer.layers]
        assert "PoseOverlay" not in layer_types
        assert "BehaviorOverlay" not in layer_types

    def test_overlay_renders_fallback_when_no_frame(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
            },
        }
        renderer = build_visualizer(cfg)
        result = renderer.render({"frame": None})
        assert result.shape == (480, 640, 3)  # renderer default fallback
        assert result.dtype == np.uint8

    def test_pose_overlay_accepts_custom_colors(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
                "pose_overlay": {
                    "enabled": True,
                    "high_conf_color": [255, 0, 0],
                    "low_conf_color": [0, 0, 255],
                    "human_edge_color": [0, 255, 255],
                    "animal_edge_color": [255, 255, 0],
                    "keypoint_radius": 6,
                    "edge_thickness": 3,
                    "high_conf_threshold": 0.7,
                },
            },
        }
        renderer = build_visualizer(cfg)
        overlay = [l for l in renderer.layers if isinstance(l, PoseOverlay)][0]
        assert overlay._kp_radius == 6
        assert overlay._edge_thickness == 3
        assert overlay._high_conf_color == (255, 0, 0)
        assert overlay._low_conf_color == (0, 0, 255)
        assert overlay._human_edge_color == (0, 255, 255)
        assert overlay._animal_edge_color == (255, 255, 0)
        assert overlay._conf_thresh == 0.7

    def test_behavior_overlay_accepts_custom_colors(self) -> None:
        cfg: dict[str, Any] = {
            "visualization": {
                "layers": [],
                "behavior_overlay": {
                    "enabled": True,
                    "font_scale": 0.6,
                    "font_thickness": 2,
                    "offset_x": -100,
                    "offset_y": -5,
                    "label_colors": {
                        "human.human_fallen": [0, 0, 200],
                    },
                },
            },
        }
        renderer = build_visualizer(cfg)
        overlay = [l for l in renderer.layers if isinstance(l, BehaviorOverlay)][0]
        assert overlay._font_scale == 0.6
        assert overlay._font_thickness == 2
        assert overlay._offset_x == -100
        assert overlay._offset_y == -5
        assert overlay._label_colors.get(("human", "human_fallen")) == (0, 0, 200)
