from __future__ import annotations

import numpy as np

from src.lane_detection.interfaces import LaneDetectionResult
from src.visualization.layers.base_layer import BaseLayer


class LaneLayer(BaseLayer):
    """Draw lane boundaries and the ego driving corridor.

    Supports both the new point-based ``LaneDetectionResult`` and legacy
    polynomial-based ``LaneInfo``.

    Args:
        enabled: Whether this layer is active.
        fill_alpha: Transparency of the lane polygon fill.
    """

    def __init__(self, enabled: bool = True, fill_alpha: float = 0.2) -> None:
        super().__init__(enabled)
        self._fill_alpha = fill_alpha

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        import cv2  # noqa: PLC0415

        if not self._enabled:
            return canvas

        lane_info: LaneDetectionResult | None = ctx.get("lane")
        if lane_info is None or lane_info.confidence < 0.1:
            return canvas

        h, w = canvas.shape[:2]

        # --- Try new point-based schema first ---
        if hasattr(lane_info, "left_lane_points"):
            return self._draw_points(canvas, lane_info)

        # --- Legacy polynomial-based schema ---
        y_vals = np.linspace(0, h - 1, num=50, dtype=np.float64)

        left_x: np.ndarray | None = None
        right_x: np.ndarray | None = None

        if lane_info.left_lane is not None:
            c = lane_info.left_lane.coeffs
            left_x = c[0] * y_vals * y_vals + c[1] * y_vals + c[2]

        if lane_info.right_lane is not None:
            c = lane_info.right_lane.coeffs
            right_x = c[0] * y_vals * y_vals + c[1] * y_vals + c[2]

        canvas = self._draw_polynomial_lanes(
            canvas, left_x, right_x, y_vals, w, h,
        )

        # Legacy centre line
        if lane_info.lane_center is not None:
            c = lane_info.lane_center.coeffs
            cx = c[0] * y_vals * y_vals + c[1] * y_vals + c[2]
            pts = np.column_stack([
                np.clip(cx, 0, w - 1).astype(np.int32),
                y_vals.astype(np.int32),
            ])
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    #  Point-based drawing (new schema)
    # ------------------------------------------------------------------

    def _draw_points(self, canvas: np.ndarray, lane_info: LaneDetectionResult) -> np.ndarray:
        import cv2  # noqa: PLC0415

        left_pts = lane_info.left_lane_points
        right_pts = lane_info.right_lane_points
        center_pts = lane_info.center_lane_points

        color_fill: tuple[int, int, int] = (0, 255, 0)

        # Ego driving corridor (translucent polygon)
        if left_pts and right_pts:
            left_arr = np.array(left_pts, dtype=np.int32)
            right_arr = np.array(right_pts, dtype=np.int32)
            polygon = np.vstack([left_arr, right_arr[::-1]])
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [polygon], color_fill)
            canvas = cv2.addWeighted(canvas, 1.0 - self._fill_alpha, overlay, self._fill_alpha, 0.0)

        # Lane lines
        for pts in (left_pts, right_pts):
            if pts:
                arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [arr], isClosed=False, color=color_fill, thickness=2)

        # Centre line (white)
        if center_pts:
            arr = np.array(center_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [arr], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    #  Polynomial-based drawing (legacy schema)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_polynomial_lanes(
        canvas: np.ndarray,
        left_x: np.ndarray | None,
        right_x: np.ndarray | None,
        y_vals: np.ndarray,
        w: int,
        h: int,
    ) -> np.ndarray:
        import cv2  # noqa: PLC0415

        color: tuple[int, int, int] = (0, 255, 255)
        fill_alpha = 0.3

        # Fill between lanes
        if left_x is not None and right_x is not None:
            pts = np.column_stack([
                np.clip(left_x, 0, w - 1),
                y_vals,
            ]).astype(np.int32)
            pts = np.vstack([
                pts,
                np.column_stack([
                    np.clip(right_x[::-1], 0, w - 1),
                    y_vals[::-1],
                ]).astype(np.int32),
            ])
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            canvas = cv2.addWeighted(canvas, 1.0 - fill_alpha, overlay, fill_alpha, 0.0)

        # Lane lines
        for xs in (left_x, right_x):
            if xs is not None:
                pts = np.column_stack([
                    np.clip(xs, 0, w - 1).astype(np.int32),
                    y_vals.astype(np.int32),
                ])
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

        return canvas
