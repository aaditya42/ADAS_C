from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from src.overtaking.schemas import OvertakingCycleResult, OvertakingEvent, OvertakingEventStatus, OvertakingState
from src.visualization.layers._utils import color_for_track, draw_label
from src.visualization.layers.base_layer import BaseLayer


class OvertakingLayer(BaseLayer):
    """Visualizes live ego-vehicle overtaking manoeuvres, statuses, and notifications.

    Supports independent runtime toggling without disturbing underlying
    detection, state progression, timestamp recording, or Excel reporting.

    Args:
        enabled: Whether the visual overlay is active.
        show_target_marker: Draw minimal visual indicator around candidate target vehicles.
        show_elapsed_time: Display elapsed overtaking time on active badges.
        show_event_state: Display current state machine state.
        show_completion_notification: Display temporary completion banner when an event completes.
        notification_duration: How long (seconds) completion notifications remain visible.
        maximum_displayed_events: Maximum number of simultaneous active events shown in HUD.
    """

    def __init__(
        self,
        enabled: bool = True,
        show_target_marker: bool = True,
        show_elapsed_time: bool = True,
        show_event_state: bool = True,
        show_completion_notification: bool = True,
        notification_duration: float = 2.5,
        maximum_displayed_events: int = 5,
    ) -> None:
        super().__init__(enabled)
        self.show_target_marker = show_target_marker
        self.show_elapsed_time = show_elapsed_time
        self.show_event_state = show_event_state
        self.show_completion_notification = show_completion_notification
        self.notification_duration = notification_duration
        self.maximum_displayed_events = maximum_displayed_events

        # Buffer for recent notifications: list of dicts {text, color, expiry_time}
        self._notification_buffer: list[dict[str, Any]] = []

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        if not self._enabled:
            return canvas

        frame = ctx.get("frame")
        if frame is None:
            return canvas

        current_time = float(getattr(frame, "timestamp", 0.0) or 0.0)
        h, w = canvas.shape[:2]

        overtaking_res: OvertakingCycleResult | None = ctx.get("overtaking")
        if overtaking_res is None:
            return canvas

        # ------------------------------------------------------------- #
        # 1. Ingest newly completed / incomplete events into notification buffer
        # ------------------------------------------------------------- #
        if self.show_completion_notification:
            for ev in overtaking_res.newly_completed:
                dur_str = f"{ev.duration_seconds:.2f}s" if ev.duration_seconds is not None else "N/A"
                text = f"OVERTAKING COMPLETED | ID: {ev.track_id} ({ev.vehicle_class.upper()}) | TIME: {dur_str} | SIDE: {ev.passing_side}"
                self._notification_buffer.append({
                    "text": text,
                    "color": (0, 220, 0),  # Bright green
                    "expiry_time": current_time + self.notification_duration,
                })

            for ev in overtaking_res.newly_incomplete:
                if ev.state in (OvertakingState.OVERTAKING_STARTED, OvertakingState.PASSING):
                    reason = ev.uncertainty_reason or "INTERRUPTED"
                    text = f"OVERTAKING INCOMPLETE | ID: {ev.track_id} | {reason}"
                    self._notification_buffer.append({
                        "text": text,
                        "color": (0, 165, 255),  # Amber
                        "expiry_time": current_time + min(2.0, self.notification_duration),
                    })

        # Purge expired notifications
        self._notification_buffer = [
            n for n in self._notification_buffer if n["expiry_time"] >= current_time
        ]

        # ------------------------------------------------------------- #
        # 2. Draw Target Vehicle Visual Markers
        # ------------------------------------------------------------- #
        active_candidates = overtaking_res.active_events
        if self.show_target_marker:
            for ev in active_candidates:
                if ev.latest_bbox is not None:
                    bx1 = int(round(ev.latest_bbox.x1))
                    by1 = int(round(ev.latest_bbox.y1))
                    bx2 = int(round(ev.latest_bbox.x2))
                    by2 = int(round(ev.latest_bbox.y2))
                    color = (255, 140, 0) if ev.state == OvertakingState.PASSING else (0, 215, 255)

                    # Distinct corner brackets around target
                    c_len = max(8, min(20, (bx2 - bx1) // 4))
                    thick = 2
                    # Top-left
                    cv2.line(canvas, (bx1, by1), (bx1 + c_len, by1), color, thick, cv2.LINE_AA)
                    cv2.line(canvas, (bx1, by1), (bx1, by1 + c_len), color, thick, cv2.LINE_AA)
                    # Top-right
                    cv2.line(canvas, (bx2, by1), (bx2 - c_len, by1), color, thick, cv2.LINE_AA)
                    cv2.line(canvas, (bx2, by1), (bx2, by1 + c_len), color, thick, cv2.LINE_AA)
                    # Bottom-left
                    cv2.line(canvas, (bx1, by2), (bx1 + c_len, by2), color, thick, cv2.LINE_AA)
                    cv2.line(canvas, (bx1, by2), (bx1, by2 - c_len), color, thick, cv2.LINE_AA)
                    # Bottom-right
                    cv2.line(canvas, (bx2, by2), (bx2 - c_len, by2), color, thick, cv2.LINE_AA)
                    cv2.line(canvas, (bx2, by2), (bx2, by2 - c_len), color, thick, cv2.LINE_AA)

                    # Status tag above target box
                    elapsed = max(0.0, current_time - (ev.start_timestamp or current_time))
                    tag = f"OVT {ev.state.value.split('_')[-1]} {elapsed:.1f}s"
                    draw_label(canvas, bx1, max(15, by1 - 6), tag, color)

        # ------------------------------------------------------------- #
        # 3. Draw Active Overtaking HUD Cards (Top-Left under top HUD)
        # ------------------------------------------------------------- #
        start_y = 38
        card_x = 12
        card_h = 24
        pad = 6

        # Draw active cards
        displayed_count = 0
        for ev in active_candidates[: self.maximum_displayed_events]:
            elapsed = max(0.0, current_time - (ev.start_timestamp or current_time))
            side_str = f" | {ev.passing_side}" if ev.passing_side != "UNKNOWN" else ""
            line_text = f"OVERTAKING | ID: {ev.track_id} | {ev.vehicle_class.upper()} | {elapsed:.2f} s{side_str}"

            # Calculate box width
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.48
            thick = 1
            (tw, th), _ = cv2.getTextSize(line_text, font, scale, thick)
            card_w = tw + 20

            # Draw semi-transparent background badge
            overlay = canvas.copy()
            y1 = start_y + displayed_count * (card_h + pad)
            y2 = y1 + card_h
            cv2.rectangle(overlay, (card_x, y1), (card_x + card_w, y2), (25, 25, 30), -1)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0.0, canvas)

            # Left accent bar
            accent_col = (0, 165, 255) if ev.state == OvertakingState.PASSING else (0, 215, 255)
            cv2.rectangle(canvas, (card_x, y1), (card_x + 4, y2), accent_col, -1)
            cv2.rectangle(canvas, (card_x, y1), (card_x + card_w, y2), accent_col, 1)

            # Text
            cv2.putText(
                canvas,
                line_text,
                (card_x + 10, y1 + 17),
                font,
                scale,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )
            displayed_count += 1

        # ------------------------------------------------------------- #
        # 4. Draw Temporary Completion / Incomplete Notifications
        # ------------------------------------------------------------- #
        notif_y = start_y + displayed_count * (card_h + pad)
        for notif in self._notification_buffer:
            n_text = notif["text"]
            n_color = notif["color"]
            (tw, th), _ = cv2.getTextSize(n_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            n_w = tw + 22
            n_h = 26

            overlay = canvas.copy()
            cv2.rectangle(overlay, (card_x, notif_y), (card_x + n_w, notif_y + n_h), (15, 25, 20), -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0.0, canvas)

            cv2.rectangle(canvas, (card_x, notif_y), (card_x + 4, notif_y + n_h), n_color, -1)
            cv2.rectangle(canvas, (card_x, notif_y), (card_x + n_w, notif_y + n_h), n_color, 1)

            cv2.putText(
                canvas,
                n_text,
                (card_x + 10, notif_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                n_color,
                1,
                cv2.LINE_AA,
            )
            notif_y += n_h + pad

        return canvas

