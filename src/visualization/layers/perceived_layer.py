from __future__ import annotations

import math

import numpy as np

from src.schemas import PerceivedObject, RiskObject
from src.visualization.layers._utils import color_for_track, draw_label, risk_color
from src.visualization.layers.base_layer import BaseLayer


class PerceivedLayer(BaseLayer):
    """Draw perceived objects with depth/distance/speed/TTC labels.

    Args:
        enabled: Whether this layer is active.
        show_distance: Show distance (m) label.
        show_speed: Show speed (m/s) label.
        show_ttc: Show TTC (s) label.
    """

    def __init__(
        self,
        enabled: bool = True,
        show_distance: bool = True,
        show_speed: bool = True,
        show_ttc: bool = True,
    ) -> None:
        super().__init__(enabled)
        self._show_distance = show_distance
        self._show_speed = show_speed
        self._show_ttc = show_ttc

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        perceived: list[PerceivedObject] = ctx.get("perceived", [])
        risk_list: list[RiskObject] | None = ctx.get("risk")
        motion: dict = ctx.get("motion", {})
        ttc_dict: dict = ctx.get("ttc", {})

        risk_by_id: dict[int, RiskObject] = {}
        if risk_list is not None:
            for r in risk_list:
                risk_by_id[r.track_id] = r

        for obj in perceived:
            r = risk_by_id.get(obj.track_id)
            color = risk_color(r.risk_level if r is not None else None)

            x1 = int(round(obj.bbox.x1))
            y1 = int(round(obj.bbox.y1))
            x2 = int(round(obj.bbox.x2))
            y2 = int(round(obj.bbox.y2))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{obj.track_id} {obj.class_name}"

            if r is not None:
                label += f" risk={r.risk_score:.0f}"

            if self._show_distance and obj.depth_confidence > 0:
                label += f" d={obj.distance:.2f}"

            if self._show_speed:
                m = motion.get(obj.track_id)
                if m is not None and m.speed_mps is not None:
                    label += f" v={m.speed_mps:+.1f}"

            if self._show_ttc:
                t = ttc_dict.get(obj.track_id)
                if t is not None and math.isfinite(t):
                    label += f" TTC:{t:.1f}s"

            draw_label(canvas, x1, y1, label, color)

        return canvas
