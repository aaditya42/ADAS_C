from __future__ import annotations

import numpy as np

from src.visualization.layers.base_layer import BaseLayer


class DepthLayer(BaseLayer):
    """Overlay a colourised depth map onto the canvas.

    Args:
        enabled: Whether this layer is active.
        alpha: Blend factor for the depth overlay.
        colormap: OpenCV colour map constant name (e.g. ``"MAGMA"``).
    """

    def __init__(
        self,
        enabled: bool = True,
        alpha: float = 0.4,
        colormap: str = "MAGMA",
    ) -> None:
        super().__init__(enabled)
        self._alpha = alpha
        self._colormap_name = colormap

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        depth_map = ctx.get("depth_map")
        if depth_map is None:
            return canvas

        cmap_attr = getattr(cv2, f"COLORMAP_{self._colormap_name}", cv2.COLORMAP_MAGMA)

        norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(norm, cmap_attr)

        return cv2.addWeighted(canvas, 1.0 - self._alpha, colored, self._alpha, 0.0)
