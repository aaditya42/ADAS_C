from __future__ import annotations

import numpy as np

from src.visualization.layers.base_layer import BaseLayer


class FPSLayer(BaseLayer):
    """Draw FPS and frame ID in the top-right corner.

    Args:
        enabled: Whether this layer is active.
    """

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        fps = ctx.get("_fps", 0.0)
        frame = ctx.get("frame")
        fid = frame.frame_id if frame is not None else 0

        text = f"{fps:.1f} FPS  |  Frame {fid}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        h, w = canvas.shape[:2]
        x = w - tw - 8
        y = h - 8

        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return canvas
