from __future__ import annotations

import numpy as np

from src.schemas.traffic_sign_info import TrafficSignInfo
from src.visualization.layers._utils import draw_label
from src.visualization.layers.base_layer import BaseLayer


class SignLayer(BaseLayer):
    """Draw traffic sign bounding boxes (blue) with labels."""

    _SIGN_COLOR: tuple[int, int, int] = (255, 0, 0)

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        signs: list[TrafficSignInfo] = ctx.get("signs", [])
        for sign in signs:
            x1 = int(round(sign.bbox.x1))
            y1 = int(round(sign.bbox.y1))
            x2 = int(round(sign.bbox.x2))
            y2 = int(round(sign.bbox.y2))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), self._SIGN_COLOR, 2)

            label = sign.sign_type.value
            if sign.value is not None:
                label += f" {sign.value:.0f}"

            draw_label(canvas, x1, y1, label, self._SIGN_COLOR)

        return canvas
