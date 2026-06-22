from __future__ import annotations

import numpy as np

from src.schemas import TrackedObject
from src.visualization.layers._utils import color_for_track, draw_label, risk_color
from src.visualization.layers.base_layer import BaseLayer


class TrackingLayer(BaseLayer):
    """Draw tracked objects with per-track colours.

    Args:
        enabled: Whether this layer is active.
        color_by: ``"id"`` (per-track) or ``"class"`` (per-class).
    """

    def __init__(self, enabled: bool = True, color_by: str = "id") -> None:
        super().__init__(enabled)
        self._color_by = color_by

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        tracked: list[TrackedObject] = ctx.get("tracked", [])
        for obj in tracked:
            color = color_for_track(obj.track_id)

            x1 = int(round(obj.bbox.x1))
            y1 = int(round(obj.bbox.y1))
            x2 = int(round(obj.bbox.x2))
            y2 = int(round(obj.bbox.y2))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            draw_label(
                canvas,
                x1,
                y1,
                f"ID:{obj.track_id} {obj.class_name} {obj.confidence:.2f}",
                color,
            )

        return canvas
