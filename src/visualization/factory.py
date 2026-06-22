from __future__ import annotations

from typing import Any

from src.visualization.layers.base_layer import BaseLayer
from src.visualization.layers.depth_layer import DepthLayer
from src.visualization.layers.detection_layer import DetectionLayer
from src.visualization.layers.fps_layer import FPSLayer
from src.visualization.layers.hud_layer import HUDLayer
from src.visualization.layers.lane_layer import LaneLayer
from src.visualization.layers.perceived_layer import PerceivedLayer
from src.visualization.layers.risk_layer import RiskLayer
from src.visualization.layers.sign_layer import SignLayer
from src.visualization.layers.tracking_layer import TrackingLayer
from src.visualization.renderer import Renderer


_LAYER_BUILDERS: dict[str, tuple[type[BaseLayer], dict[str, Any]]] = {
    "detection": (DetectionLayer, {}),
    "tracking": (TrackingLayer, {"color_by": "id"}),
    "depth": (DepthLayer, {"alpha": 0.4, "colormap": "MAGMA"}),
    "perceived": (PerceivedLayer, {"show_distance": True, "show_speed": True, "show_ttc": True}),
    "lane": (LaneLayer, {"fill_alpha": 0.2}),
    "sign": (SignLayer, {}),
    "risk": (RiskLayer, {}),
    "hud": (HUDLayer, {"alpha": 0.5}),
    "fps": (FPSLayer, {}),
}


def build_renderer(cfg: dict[str, Any]) -> Renderer:
    """Build a ``Renderer`` from the pipeline configuration.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        A ``Renderer`` instance.
    """
    vis_cfg = cfg.get("visualization", {})
    layer_names: list[str] = vis_cfg.get("layers", [
        "depth", "lane", "perceived", "risk", "sign", "hud", "fps",
    ])

    layers: list[BaseLayer] = []
    for name in layer_names:
        if name not in _LAYER_BUILDERS:
            continue
        cls, defaults = _LAYER_BUILDERS[name]

        # Merge per-layer config over defaults
        layer_opts = dict(vis_cfg.get(name, {}))
        kwargs = {**defaults, **layer_opts}

        layers.append(cls(**kwargs))

    return Renderer(layers)
