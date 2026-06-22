from __future__ import annotations

import numpy as np

from src.visualization.layers.base_layer import BaseLayer


class Renderer:
    """Compositing renderer for the ADAS perception pipeline.

    Maintains an ordered list of ``BaseLayer`` instances applied in z-order.
    Supports per-layer runtime toggle via ``toggle(layer_name)``.
    """

    def __init__(self, layers: list[BaseLayer]) -> None:
        self._layers: list[BaseLayer] = layers
        self._name_map: dict[str, BaseLayer] = {}
        for layer in layers:
            name = type(layer).__name__.lower().replace("layer", "")
            self._name_map[name] = layer

    @property
    def layers(self) -> list[BaseLayer]:
        return list(self._layers)

    def toggle(self, layer_name: str) -> bool:
        """Toggle a layer by short name (e.g. ``"depth"``, ``"lane"``).

        Returns the new enabled state, or ``False`` if the layer is unknown.
        """
        layer = self._name_map.get(layer_name)
        if layer is None:
            return False
        layer.enabled = not layer.enabled
        return layer.enabled

    def render(self, pipeline_output: dict) -> np.ndarray:
        """Render all enabled layers onto a clone of the raw frame.

        Args:
            pipeline_output: The dict returned by
                ``PerceptionPipeline.process()``.

        Returns:
            Annotated BGR image array.
        """
        frame = pipeline_output.get("frame")
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        canvas = frame.image.copy()

        for layer in self._layers:
            if not layer.enabled:
                continue
            canvas = layer(canvas, pipeline_output)

        return canvas
