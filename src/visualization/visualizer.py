from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

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
from src.visualization.overlays.behavior_overlay import BehaviorOverlay
from src.visualization.overlays.pose_overlay import PoseOverlay
from src.visualization.renderer import Renderer

_CORE_LAYER_BUILDERS: dict[str, tuple[type[BaseLayer], dict[str, Any]]] = {
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


def build_visualizer(cfg: dict[str, Any]) -> Renderer:
    """Build a ``Renderer`` with core layers and optional pose/behaviour overlays.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        A ``Renderer`` instance with all enabled layers and overlays.
    """
    vis_cfg = cfg.get("visualization", {})
    layer_names: list[str] = vis_cfg.get("layers", [
        "depth", "lane", "perceived", "risk", "sign", "hud", "fps",
    ])

    layers: list[BaseLayer] = []
    for name in layer_names:
        if name not in _CORE_LAYER_BUILDERS:
            continue
        cls, defaults = _CORE_LAYER_BUILDERS[name]
        layer_opts = dict(vis_cfg.get(name, {}))
        kwargs = {**defaults, **layer_opts}
        layers.append(cls(**kwargs))

    # --- Optional pose overlay -------------------------------------------- #
    pose_cfg = vis_cfg.get("pose_overlay", {})
    if pose_cfg.get("enabled", True):
        layers.append(PoseOverlay(
            keypoint_radius=int(pose_cfg.get("keypoint_radius", 4)),
            edge_thickness=int(pose_cfg.get("edge_thickness", 2)),
            high_conf_threshold=float(pose_cfg.get("high_conf_threshold", 0.5)),
            high_conf_color=_parse_color(pose_cfg.get("high_conf_color", [0, 255, 0])),
            low_conf_color=_parse_color(pose_cfg.get("low_conf_color", [0, 100, 0])),
            human_edge_color=_parse_color(pose_cfg.get("human_edge_color", [255, 255, 0])),
            animal_edge_color=_parse_color(pose_cfg.get("animal_edge_color", [255, 0, 255])),
        ))

    # --- Optional behavior overlay ---------------------------------------- #
    beh_cfg = vis_cfg.get("behavior_overlay", {})
    if beh_cfg.get("enabled", True):
        raw_colors: Optional[dict[str, list[int]]] = beh_cfg.get("label_colors")
        label_colors = None
        if raw_colors is not None:
            label_colors = {k: tuple(v) for k, v in raw_colors.items()}
        layers.append(BehaviorOverlay(
            label_colors=label_colors,
            font_scale=float(beh_cfg.get("font_scale", 0.4)),
            font_thickness=int(beh_cfg.get("font_thickness", 1)),
            offset_x=int(beh_cfg.get("offset_x", -140)),
            offset_y=int(beh_cfg.get("offset_y", -10)),
        ))

    return Renderer(layers)


def _parse_color(val: Any) -> tuple[int, int, int]:
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        return (int(val[0]), int(val[1]), int(val[2]))
    return (0, 255, 0)
