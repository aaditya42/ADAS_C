"""Driver Warning HUD visualization layer.

Displays real-time directional collision warnings, primary threat details,
secondary threat lists, directional visual indicators, and target object highlights.
Supports independent ON/OFF runtime and configuration toggling without affecting
background threat assessment or warning event lifecycles.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from src.threat.schemas import (
    CameraDirection,
    DirectionalWarningLevel,
    WarningEventState,
    WarningOutput,
)
from src.visualization.layers.base_layer import BaseLayer

# Direction indicator symbols with unambiguous directional guidance
_DIRECTION_SYMBOLS: dict[CameraDirection, str] = {
    CameraDirection.FRONT: "[^]",
    CameraDirection.FRONT_LEFT: "[<^]",
    CameraDirection.FRONT_RIGHT: "[>^]",
    CameraDirection.LEFT: "[<-]",
    CameraDirection.RIGHT: "[->]",
    CameraDirection.UNKNOWN: "[?]",
}

_DIRECTION_LABELS: dict[CameraDirection, str] = {
    CameraDirection.FRONT: "FRONT",
    CameraDirection.FRONT_LEFT: "FRONT-LEFT",
    CameraDirection.FRONT_RIGHT: "FRONT-RIGHT",
    CameraDirection.LEFT: "LEFT",
    CameraDirection.RIGHT: "RIGHT",
    CameraDirection.UNKNOWN: "UNKNOWN",
}

# Color palettes (BGR for OpenCV)
_COLOR_HIGH_ACCENT = (40, 40, 225)       # Crimson / Vibrant Red
_COLOR_HIGH_BG = (15, 15, 35)           # Deep Dark Red
_COLOR_MEDIUM_ACCENT = (0, 140, 255)     # Amber / Vivid Orange
_COLOR_MEDIUM_BG = (15, 22, 35)         # Dark Amber/Navy
_COLOR_LOW_ACCENT = (220, 160, 0)        # Cyan / Light Azure
_COLOR_LOW_BG = (20, 28, 35)            # Dark Cyan/Slate
_COLOR_NEUTRAL_ACCENT = (140, 140, 140)  # Neutral Grey
_COLOR_NEUTRAL_BG = (25, 25, 25)        # Charcoal
_COLOR_WHITE = (255, 255, 255)
_COLOR_MUTED = (190, 190, 190)

_LEVEL_ORDER = {
    DirectionalWarningLevel.HIGH: 3,
    DirectionalWarningLevel.MEDIUM: 2,
    DirectionalWarningLevel.LOW: 1,
    DirectionalWarningLevel.UNASSESSED: 0,
}


def _alpha_blend_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bg_color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend a solid background rectangle in-place into canvas ROI without full-image copy."""
    h, w = canvas.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = canvas[y1:y2, x1:x2]
    bg_rect = np.full_like(sub, bg_color, dtype=np.uint8)
    cv2.addWeighted(bg_rect, alpha, sub, 1.0 - alpha, 0.0, sub)


class WarningLayer(BaseLayer):
    """Real-time Driver Warning HUD layer.

    Presents the highest-priority threat on a primary prominent panel,
    summarizes concurrent secondary warnings in a compact stack, marks
    threat targets on the road canvas, and provides an independent runtime toggle.
    """

    def __init__(
        self,
        enabled: bool = True,
        show_primary_warning: bool = True,
        show_secondary_warnings: bool = True,
        max_secondary_warnings: int = 3,
        show_direction_indicator: bool = True,
        highlight_threat_object: bool = True,
        show_ttc: bool = True,
        show_risk_score: bool = False,
        show_track_id: bool = True,
        show_scenario: bool = True,
        show_reliability: bool = True,
        neutral_status_indicator: bool = True,
        toggle_notification_duration: float = 1.5,
    ) -> None:
        super().__init__(enabled)
        self.show_primary_warning = show_primary_warning
        self.show_secondary_warnings = show_secondary_warnings
        self.max_secondary_warnings = max_secondary_warnings
        self.show_direction_indicator = show_direction_indicator
        self.highlight_threat_object = highlight_threat_object
        self.show_ttc = show_ttc
        self.show_risk_score = show_risk_score
        self.show_track_id = show_track_id
        self.show_scenario = show_scenario
        self.show_reliability = show_reliability
        self.neutral_status_indicator = neutral_status_indicator
        self.toggle_notification_duration = toggle_notification_duration

        # State tracking for hysteresis and notifications
        self._last_primary_event_id: str | None = None
        self._last_enabled_state: bool = enabled
        self._toggle_notification_expiry: float = 0.0
        self._toggle_notification_text: str = ""
        self._toggle_notification_color: tuple[int, int, int] = _COLOR_NEUTRAL_ACCENT

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if self._enabled != value:
            self._enabled = value
            self._toggle_notification_text = f"[WARNING HUD: {'ON' if value else 'OFF'}]"
            self._toggle_notification_color = (0, 220, 0) if value else (0, 165, 255)
            # Will be assigned actual timestamp in __call__
            self._toggle_notification_expiry = -1.0

    def select_primary_threat(
        self, warnings: list[WarningOutput]
    ) -> tuple[WarningOutput | None, list[WarningOutput]]:
        """Select the highest-priority active threat using deterministic ranking and stability.

        Filters out UNASSESSED or inactive warnings, prioritizes HIGH over MEDIUM/LOW,
        incorporates prioritizer rank and collision urgency, and applies persistence
        tie-breaking to avoid frame-to-frame panel flapping.
        """
        # Filter active actionable warnings only
        active = [
            w for w in warnings
            if w.level != DirectionalWarningLevel.UNASSESSED
            and w.state in (
                WarningEventState.ACTIVE_HIGH,
                WarningEventState.ACTIVE_MEDIUM,
                WarningEventState.ACTIVE_LOW,
            )
        ]

        if not active:
            self._last_primary_event_id = None
            return None, []

        def sort_key(w: WarningOutput) -> tuple[int, int, int, float, int, int]:
            # 1. Severity: Higher level first (HIGH: -3, MEDIUM: -2, LOW: -1)
            sev = -_LEVEL_ORDER.get(w.level, 0)
            # 2. Primary threat flag: is_primary_threat first
            prim = 0 if w.is_primary_threat else 1
            # 3. Priority rank from Stage 3 ThreatPrioritizer
            rank = w.priority_rank if w.priority_rank is not None else 999
            # 4. Collision urgency (TTC): Lower TTC first
            urgency = w.collision_urgency if w.collision_urgency is not None else 999.0
            # 5. Persistence bonus: Previous primary threat gets tie-breaker priority
            persist = 0 if w.event_id == self._last_primary_event_id else 1
            # 6. Stable ID
            tid = w.track_id
            return (sev, prim, rank, urgency, persist, tid)

        sorted_threats = sorted(active, key=sort_key)
        primary = sorted_threats[0]
        self._last_primary_event_id = primary.event_id
        secondary = sorted_threats[1:]
        return primary, secondary

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        frame = ctx.get("frame")
        current_time = float(getattr(frame, "timestamp", 0.0) or 0.0)

        # Check toggle state changes for notification duration
        if self._toggle_notification_expiry == -1.0:
            self._toggle_notification_expiry = current_time + self.toggle_notification_duration

        # -----------------------------------------------------------------
        # Critical Requirement: If HUD is disabled, skip all drawing operations
        # (Background risk calculation and event tracking already occurred in pipeline)
        # -----------------------------------------------------------------
        if not self._enabled:
            # Optionally show brief temporary toggle-off notification banner
            if current_time <= self._toggle_notification_expiry and self._toggle_notification_text:
                self._draw_toggle_banner(canvas, self._toggle_notification_text, self._toggle_notification_color)
            return canvas

        # Retrieve warnings generated by Stage 4
        warnings: list[WarningOutput] = ctx.get("warnings") or ctx.get("directional_warnings") or []
        primary, secondary = self.select_primary_threat(warnings)

        # 1. Draw Target Object Highlight on the road canvas
        if self.highlight_threat_object and primary is not None:
            self._highlight_target_object(canvas, primary, ctx)

        # 2. Draw Primary Warning HUD Panel
        if self.show_primary_warning:
            if primary is not None:
                self._draw_primary_panel(canvas, primary)
            elif self.neutral_status_indicator:
                self._draw_neutral_status(canvas)

        # 3. Draw Secondary Threat List
        if self.show_secondary_warnings and secondary and self.max_secondary_warnings > 0:
            self._draw_secondary_panel(canvas, secondary[: self.max_secondary_warnings])

        # 4. Draw Toggle Notification if active
        if current_time <= self._toggle_notification_expiry and self._toggle_notification_text:
            self._draw_toggle_banner(canvas, self._toggle_notification_text, self._toggle_notification_color)

        return canvas

    def _draw_primary_panel(self, canvas: np.ndarray, threat: WarningOutput) -> None:
        """Render the prominent primary warning panel centered at the upper display."""
        h, w = canvas.shape[:2]

        # Determine theme by warning level
        if threat.level == DirectionalWarningLevel.HIGH:
            accent_col = _COLOR_HIGH_ACCENT
            bg_col = _COLOR_HIGH_BG
            title_text = "CRITICAL WARNING"
            panel_h = 74
        elif threat.level == DirectionalWarningLevel.MEDIUM:
            accent_col = _COLOR_MEDIUM_ACCENT
            bg_col = _COLOR_MEDIUM_BG
            title_text = "CAUTION WARNING"
            panel_h = 70
        else:
            accent_col = _COLOR_LOW_ACCENT
            bg_col = _COLOR_LOW_BG
            title_text = "ADVISORY THREAT"
            panel_h = 56

        panel_w = 480
        x1 = max(10, (w - panel_w) // 2)
        y1 = 36  # Directly below top HUD strip (y=28)
        x2 = x1 + panel_w
        y2 = y1 + panel_h

        # Fast in-place ROI alpha blend (no full-frame copies)
        _alpha_blend_rect(canvas, x1, y1, x2, y2, bg_col, 0.88)

        # Outer border & thick left accent bar
        cv2.rectangle(canvas, (x1, y1), (x2, y2), accent_col, 1)
        cv2.rectangle(canvas, (x1, y1), (x1 + 6, y2), accent_col, -1)

        # Direction text and icon
        dir_sym = _DIRECTION_SYMBOLS.get(threat.direction, "[?]")
        dir_lbl = _DIRECTION_LABELS.get(threat.direction, "UNKNOWN")
        dir_full = f"{dir_lbl} {dir_sym}" if self.show_direction_indicator else dir_lbl

        # Line 1: [LEVEL] [DIRECTION] [OBJECT CLASS]
        class_str = threat.object_class.upper() if threat.object_class else "OBSTACLE"
        line1_left = f"{title_text} | {dir_full} | {class_str}"
        cv2.putText(
            canvas,
            line1_left,
            (x1 + 14, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            accent_col if threat.level != DirectionalWarningLevel.HIGH else _COLOR_WHITE,
            2 if threat.level == DirectionalWarningLevel.HIGH else 1,
            cv2.LINE_AA,
        )

        # Line 2: Details: Track ID, Scenario, TTC, Risk, Reliability
        details: list[str] = []
        if self.show_track_id:
            details.append(f"ID:{threat.track_id}")

        if self.show_scenario and threat.scenario:
            clean_scen = threat.scenario.replace("_", " ").title()
            details.append(clean_scen)

        if self.show_ttc:
            if threat.ttc is not None and threat.ttc > 0.0:
                details.append(f"TTC: {threat.ttc:.1f}s")
            else:
                details.append("TTC: N/A")

        if self.show_risk_score and threat.risk_score is not None:
            details.append(f"Score: {threat.risk_score:.2f}")

        if self.show_reliability and threat.reliability:
            details.append(f"Rel: {threat.reliability}")

        line2_text = "  |  ".join(details)
        cv2.putText(
            canvas,
            line2_text,
            (x1 + 14, y1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            _COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Line 3: Threat Reason / Explanation (for HIGH & MEDIUM)
        if panel_h > 60:
            reason = threat.reason if threat.reason else "Trajectory conflict detected"
            if len(reason) > 58:
                reason = reason[:55] + "..."
            cv2.putText(
                canvas,
                reason,
                (x1 + 14, y1 + 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                _COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )

    def _draw_secondary_panel(self, canvas: np.ndarray, secondary: list[WarningOutput]) -> None:
        """Render compact secondary warnings on the upper-right corner."""
        h, w = canvas.shape[:2]
        sec_w = 260
        row_h = 24
        pad = 4
        x1 = w - sec_w - 12
        start_y = 36

        # Draw header with fast ROI blend
        _alpha_blend_rect(canvas, x1, start_y - 2, x1 + sec_w, start_y + 16, (20, 20, 20), 0.70)
        cv2.putText(
            canvas,
            f"OTHER ACTIVE THREATS ({len(secondary)})",
            (x1 + 6, start_y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            _COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        curr_y = start_y + 20
        for i, threat in enumerate(secondary):
            if threat.level == DirectionalWarningLevel.HIGH:
                col = _COLOR_HIGH_ACCENT
            elif threat.level == DirectionalWarningLevel.MEDIUM:
                col = _COLOR_MEDIUM_ACCENT
            else:
                col = _COLOR_LOW_ACCENT

            dir_lbl = _DIRECTION_LABELS.get(threat.direction, "UNK")
            ttc_str = f"TTC:{threat.ttc:.1f}s" if (threat.ttc is not None and threat.ttc > 0) else "TTC:N/A"
            text = f"{i+2}. {threat.level.value[:3]} {dir_lbl} {threat.object_class.upper()} {ttc_str}"

            # Fast row background blend
            _alpha_blend_rect(canvas, x1, curr_y, x1 + sec_w, curr_y + row_h, (20, 20, 25), 0.80)

            # Left accent
            cv2.rectangle(canvas, (x1, curr_y), (x1 + 3, curr_y + row_h), col, -1)
            cv2.rectangle(canvas, (x1, curr_y), (x1 + sec_w, curr_y + row_h), col, 1)

            cv2.putText(
                canvas,
                text,
                (x1 + 8, curr_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                _COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )
            curr_y += row_h + pad

    def _highlight_target_object(self, canvas: np.ndarray, threat: WarningOutput, ctx: dict) -> None:
        """Draw distinct corner brackets and threat tag on the primary threat's bounding box."""
        tid = threat.track_id
        bbox = None

        # Look up bounding box from threat features or perceived objects
        threat_feats = ctx.get("threat_features", {})
        if tid in threat_feats and hasattr(threat_feats[tid], "bbox") and threat_feats[tid].bbox is not None:
            bbox = threat_feats[tid].bbox
        else:
            perceived = ctx.get("perceived", [])
            for p in perceived:
                if getattr(p, "track_id", None) == tid:
                    bbox = getattr(p, "bbox", None)
                    break

        if bbox is None:
            return

        bx1 = int(round(bbox.x1))
        by1 = int(round(bbox.y1))
        bx2 = int(round(bbox.x2))
        by2 = int(round(bbox.y2))

        # Check bounds
        h, w = canvas.shape[:2]
        if bx1 >= w or bx2 <= 0 or by1 >= h or by2 <= 0:
            return

        bx1, bx2 = max(0, bx1), min(w - 1, bx2)
        by1, by2 = max(0, by1), min(h - 1, by2)

        if threat.level == DirectionalWarningLevel.HIGH:
            col = _COLOR_HIGH_ACCENT
        elif threat.level == DirectionalWarningLevel.MEDIUM:
            col = _COLOR_MEDIUM_ACCENT
        else:
            col = _COLOR_LOW_ACCENT

        # Distinct corner brackets around target
        c_len = max(8, min(24, (bx2 - bx1) // 4))
        thick = 2 if threat.level == DirectionalWarningLevel.HIGH else 1

        # Top-left
        cv2.line(canvas, (bx1, by1), (bx1 + c_len, by1), col, thick, cv2.LINE_AA)
        cv2.line(canvas, (bx1, by1), (bx1, by1 + c_len), col, thick, cv2.LINE_AA)
        # Top-right
        cv2.line(canvas, (bx2, by1), (bx2 - c_len, by1), col, thick, cv2.LINE_AA)
        cv2.line(canvas, (bx2, by1), (bx2, by1 - c_len), col, thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(canvas, (bx1, by2), (bx1 + c_len, by2), col, thick, cv2.LINE_AA)
        cv2.line(canvas, (bx1, by2), (bx1, by2 - c_len), col, thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(canvas, (bx2, by2), (bx2 - c_len, by2), col, thick, cv2.LINE_AA)
        cv2.line(canvas, (bx2, by2), (bx2, by2 - c_len), col, thick, cv2.LINE_AA)

        # Status tag above box
        dir_lbl = _DIRECTION_LABELS.get(threat.direction, "")
        tag_text = f"[{threat.level.value} {dir_lbl}]"
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tag_y = max(18, by1 - 6)
        tag_x = bx1

        # Fast ROI blend
        _alpha_blend_rect(canvas, tag_x, tag_y - th - 2, tag_x + tw + 6, tag_y + 2, (15, 15, 20), 0.85)
        cv2.rectangle(canvas, (tag_x, tag_y - th - 2), (tag_x + tw + 6, tag_y + 2), col, 1)
        cv2.putText(
            canvas,
            tag_text,
            (tag_x + 3, tag_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            col if threat.level != DirectionalWarningLevel.HIGH else _COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    def _draw_neutral_status(self, canvas: np.ndarray) -> None:
        """Render a small, unobtrusive status badge when no threats are active."""
        h, w = canvas.shape[:2]
        text = "WARNINGS: ACTIVE MONITORING (0 THREATS)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        panel_w = tw + 20
        panel_h = 22
        x1 = (w - panel_w) // 2
        y1 = 34

        _alpha_blend_rect(canvas, x1, y1, x1 + panel_w, y1 + panel_h, _COLOR_NEUTRAL_BG, 0.70)
        cv2.rectangle(canvas, (x1, y1), (x1 + panel_w, y1 + panel_h), _COLOR_NEUTRAL_ACCENT, 1)

        cv2.putText(
            canvas,
            text,
            (x1 + 10, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            _COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

    def _draw_toggle_banner(self, canvas: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        """Render a temporary notification badge indicating keyboard HUD toggle state."""
        h, w = canvas.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bw = tw + 24
        bh = th + 16
        bx = (w - bw) // 2
        by = h - 60

        _alpha_blend_rect(canvas, bx, by, bx + bw, by + bh, (15, 15, 20), 0.85)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 2)

        cv2.putText(
            canvas,
            text,
            (bx + 12, by + bh - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
