from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.schemas.behavior import BehaviorFrame, BehaviorLabel, BehaviorState
from src.schemas.pose import AnimalPose, HumanPose, PoseFrame


_DEFAULT_LABEL_COLORS: Dict[BehaviorLabel, Tuple[int, int, int]] = {
    BehaviorLabel.HUMAN_STANDING: (0, 255, 0),
    BehaviorLabel.HUMAN_WALKING: (0, 255, 255),
    BehaviorLabel.HUMAN_RUNNING: (0, 0, 255),
    BehaviorLabel.HUMAN_CROSSING_ROAD: (0, 165, 255),
    BehaviorLabel.HUMAN_APPROACHING_ROAD: (0, 100, 255),
    BehaviorLabel.HUMAN_LEAVING_ROAD: (255, 255, 0),
    BehaviorLabel.HUMAN_FALLEN: (0, 0, 200),
    BehaviorLabel.HUMAN_DISTRACTED: (255, 0, 255),
    BehaviorLabel.ANIMAL_STANDING: (0, 255, 0),
    BehaviorLabel.ANIMAL_WALKING: (0, 255, 255),
    BehaviorLabel.ANIMAL_RUNNING: (0, 0, 255),
    BehaviorLabel.ANIMAL_LYING: (255, 0, 100),
    BehaviorLabel.ANIMAL_GRAZING: (100, 100, 255),
    BehaviorLabel.ANIMAL_CROSSING_ROAD: (0, 165, 255),
    BehaviorLabel.ANIMAL_APPROACHING_ROAD: (0, 100, 255),
    BehaviorLabel.UNKNOWN: (128, 128, 128),
}

_LABEL_SHORT_NAMES: Dict[BehaviorLabel, str] = {
    BehaviorLabel.HUMAN_STANDING: "STAND",
    BehaviorLabel.HUMAN_WALKING: "WALK",
    BehaviorLabel.HUMAN_RUNNING: "RUN",
    BehaviorLabel.HUMAN_CROSSING_ROAD: "CROSS",
    BehaviorLabel.HUMAN_APPROACHING_ROAD: "APPR",
    BehaviorLabel.HUMAN_LEAVING_ROAD: "LEAVE",
    BehaviorLabel.HUMAN_FALLEN: "FALL",
    BehaviorLabel.HUMAN_DISTRACTED: "DIST",
    BehaviorLabel.ANIMAL_STANDING: "STAND",
    BehaviorLabel.ANIMAL_WALKING: "WALK",
    BehaviorLabel.ANIMAL_RUNNING: "RUN",
    BehaviorLabel.ANIMAL_LYING: "LYING",
    BehaviorLabel.ANIMAL_GRAZING: "GRAZE",
    BehaviorLabel.ANIMAL_CROSSING_ROAD: "CROSS",
    BehaviorLabel.ANIMAL_APPROACHING_ROAD: "APPR",
    BehaviorLabel.UNKNOWN: "??",
}


def _draw_badge(
    canvas: np.ndarray,
    x: int,
    y: int,
    lines: Tuple[str, ...],
    bg_color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    import cv2  # noqa: PLC0415

    max_w = 0
    total_h = 0
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, tw)
        total_h += th + 4

    pad = 4
    bx1 = x
    by1 = y - total_h - pad
    bx2 = x + max_w + pad * 2
    by2 = y

    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_color, -1)

    cur_y = by1 + pad + int(round(cv2.getTextSize("T", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]))
    for text in lines:
        cv2.putText(
            canvas, text,
            (bx1 + pad, cur_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        cur_y += int(round(cv2.getTextSize("T", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1])) + 4


class BehaviorOverlay:
    """Draws behaviour label badges near each tracked entity.

    Args:
        label_colors: Per-label BGR colour overrides.
        font_scale: Font scale for badge text.
        font_thickness: Font thickness for badge text.
        offset_x: Horizontal offset from bbox right edge.
        offset_y: Vertical offset from bbox bottom edge.
    """

    def __init__(
        self,
        label_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        font_scale: float = 0.4,
        font_thickness: int = 1,
        offset_x: int = -140,
        offset_y: int = -10,
    ) -> None:
        self._label_colors: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        if label_colors is not None:
            for label_key, color in label_colors.items():
                parts = label_key.split(".", 1)
                if len(parts) == 2:
                    self._label_colors[(parts[0], parts[1])] = tuple(color)
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._offset_x = offset_x
        self._offset_y = offset_y

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        behavior_frame: Optional[BehaviorFrame] = ctx.get("behavior")
        if behavior_frame is None:
            return canvas

        pose_frame: Optional[PoseFrame] = ctx.get("pose")
        bbox_map: Dict[int, Tuple[float, float, float, float]] = {}
        if pose_frame is not None:
            for h in pose_frame.humans:
                if h.track_id is not None:
                    bbox_map[h.track_id] = h.bbox
            for a in pose_frame.animals:
                if a.track_id is not None:
                    bbox_map[a.track_id] = a.bbox

        for state in behavior_frame.states:
            bbox = bbox_map.get(state.track_id)
            if bbox is None:
                continue
            self._draw_state_badge(canvas, state, bbox)

        return canvas

    def _draw_state_badge(
        self,
        canvas: np.ndarray,
        state: BehaviorState,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        x2 = int(round(bbox[2])) + self._offset_x
        y2 = int(round(bbox[3])) + self._offset_y

        color = self._resolve_color(state)
        lines = [_LABEL_SHORT_NAMES.get(state.primary_label, "??")]
        conf_str = f"{state.confidence:.2f}"
        if state.crossing_probability is not None:
            conf_str += f" CP:{state.crossing_probability:.2f}"
        lines.append(conf_str)

        _draw_badge(canvas, x2, y2, tuple(lines), color, self._font_scale, self._font_thickness)

    def _resolve_color(self, state: BehaviorState) -> Tuple[int, int, int]:
        ent = state.entity_type
        label_str = state.primary_label.value
        override = self._label_colors.get((ent, label_str))
        if override is not None:
            return override
        return _DEFAULT_LABEL_COLORS.get(state.primary_label, (128, 128, 128))
