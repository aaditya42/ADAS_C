from __future__ import annotations

import numpy as np

from src.schemas import Detection
from src.visualization.layers._utils import _COLOR_GREEN, draw_label
from src.visualization.layers.base_layer import BaseLayer


class DetectionLayer(BaseLayer):
    """Draw raw detection bounding boxes (green)."""

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        detections: list[Detection] = ctx.get("detections", [])
        for det in detections:
            x1 = int(round(det.bbox.x1))
            y1 = int(round(det.bbox.y1))
            x2 = int(round(det.bbox.x2))
            y2 = int(round(det.bbox.y2))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_GREEN, 2)
            draw_label(
                canvas, x1, y1, f"{det.class_name} {det.confidence:.2f}", _COLOR_GREEN
            )

        return canvas
