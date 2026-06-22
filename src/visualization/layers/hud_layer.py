from __future__ import annotations

import numpy as np

from src.schemas.scene_context import SceneContext
from src.visualization.layers.base_layer import BaseLayer


class HUDLayer(BaseLayer):
    """Draw a top HUD strip: speed limit, traffic light, stop/yield, in-lane count.

    Args:
        enabled: Whether this layer is active.
        alpha: Transparency of the background bar.
    """

    def __init__(self, enabled: bool = True, alpha: float = 0.5) -> None:
        super().__init__(enabled)
        self._alpha = alpha

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        scene: SceneContext | None = ctx.get("scene")
        h, w = canvas.shape[:2]
        bar_h = 28

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        canvas = cv2.addWeighted(canvas, 1.0 - self._alpha, overlay, self._alpha, 0.0)

        if scene is None:
            cv2.putText(canvas, "Scene: N/A", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            return canvas

        parts: list[str] = []

        if scene.active_speed_limit is not None:
            parts.append(f"SPD:{scene.active_speed_limit:.0f}")

        tl = scene.traffic_light_state
        if tl != "none":
            tl_colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
            tc = tl_colors.get(tl, (255, 255, 255))
            cv2.putText(canvas, tl.upper(), (w - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2)

        if scene.stop_required:
            parts.append("STOP")
        if scene.yield_required:
            parts.append("YIELD")
        if scene.crosswalk_pedestrian:
            parts.append("PED")

        parts.append(f"LANE:{scene.ego_lane_object_count}")

        if scene.nearest_in_lane_distance_m is not None:
            parts.append(f"NEAR:{scene.nearest_in_lane_distance_m:.1f}m")

        text = "  |  ".join(parts)
        cv2.putText(canvas, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return canvas
