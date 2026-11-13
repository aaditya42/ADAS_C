from __future__ import annotations

import cv2
import numpy as np

from src.schemas.trajectory import EvaluationStatus, PredictionRecord, TrajectoryPrediction
from src.visualization.layers._utils import color_for_track, draw_label
from src.visualization.layers.base_layer import BaseLayer


class TrajectoryLayer(BaseLayer):
    """Visualizes predicted future trajectories, endpoints, and live accuracy errors.

    Args:
        enabled: Whether this layer is active.
        show_predicted_path: Draw trajectory curve connecting current to future positions.
        show_predicted_endpoint: Draw prominent marker and label at horizon endpoint.
        show_actual_endpoint: Draw actual observed position when target timestamp arrives.
        show_error_line: Draw error displacement line between predicted and actual positions.
        show_error_label: Display numeric endpoint error in pixels.
    """

    def __init__(
        self,
        enabled: bool = True,
        show_predicted_path: bool = True,
        show_predicted_endpoint: bool = True,
        show_actual_endpoint: bool = True,
        show_error_line: bool = True,
        show_error_label: bool = True,
    ) -> None:
        super().__init__(enabled)
        self.show_predicted_path = show_predicted_path
        self.show_predicted_endpoint = show_predicted_endpoint
        self.show_actual_endpoint = show_actual_endpoint
        self.show_error_line = show_error_line
        self.show_error_label = show_error_label

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        if not self._enabled:
            return canvas

        trajectories: list[TrajectoryPrediction] = ctx.get("trajectories", [])
        evaluations: list[PredictionRecord] = ctx.get("trajectory_evaluations", [])

        # ------------------------------------------------------------- #
        # 1. Visualize Live Predicted Trajectories
        # ------------------------------------------------------------- #
        for pred in trajectories:
            color = color_for_track(pred.track_id)
            cur_x = int(round(pred.current_position[0]))
            cur_y = int(round(pred.current_position[1]))

            # Collect pixel points along predicted trajectory
            pts: list[tuple[int, int]] = [(cur_x, cur_y)]
            for wp in pred.predicted_positions:
                px = int(round(wp.position[0]))
                py = int(round(wp.position[1]))
                pts.append((px, py))

            # Draw predicted trajectory path
            if self.show_predicted_path and len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, pts[i + 1], 2, color, -1, cv2.LINE_AA)

            # Draw prominent final predicted endpoint
            if self.show_predicted_endpoint and len(pts) > 1:
                end_x, end_y = pts[-1]

                # Distinct multi-ring crosshair marker
                cv2.circle(canvas, (end_x, end_y), 7, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (end_x, end_y), 2, (255, 255, 255), -1, cv2.LINE_AA)

                # Crosshair tick marks
                tick_len = 5
                cv2.line(canvas, (end_x - tick_len, end_y), (end_x + tick_len, end_y), color, 1, cv2.LINE_AA)
                cv2.line(canvas, (end_x, end_y - tick_len), (end_x, end_y + tick_len), color, 1, cv2.LINE_AA)

                # Label marker with Track ID and Horizon
                label = f"ID:{pred.track_id} Pred ({pred.horizon_seconds:.1f}s)"
                draw_label(canvas, end_x + 8, end_y, label, color)

        # ------------------------------------------------------------- #
        # 2. Visualize Live Target Arrival & Error Comparison
        # ------------------------------------------------------------- #
        for rec in evaluations:
            if rec.status != EvaluationStatus.MATCHED or rec.actual_position is None:
                continue

            pred_x = int(round(rec.predicted_position[0]))
            pred_y = int(round(rec.predicted_position[1]))
            act_x = int(round(rec.actual_position[0]))
            act_y = int(round(rec.actual_position[1]))

            # Draw actual observed position (vibrant green marker)
            if self.show_actual_endpoint:
                actual_color = (0, 255, 128)  # Lime green
                cv2.circle(canvas, (act_x, act_y), 6, actual_color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (act_x, act_y), 2, actual_color, -1, cv2.LINE_AA)
                draw_label(canvas, act_x + 6, act_y, f"ID:{rec.track_id} Actual", actual_color)

            # Draw error displacement line between predicted and actual
            if self.show_error_line:
                error_color = (0, 140, 255)  # Orange warning line
                cv2.line(canvas, (pred_x, pred_y), (act_x, act_y), error_color, 2, cv2.LINE_AA)

            # Display numerical error label
            if self.show_error_label and rec.endpoint_error is not None:
                mid_x = (pred_x + act_x) // 2
                mid_y = (pred_y + act_y) // 2
                err_text = f"ID:{rec.track_id} Err:{rec.endpoint_error:.1f}px"
                draw_label(canvas, mid_x, mid_y, err_text, (0, 140, 255))

        return canvas
