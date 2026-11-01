from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.schemas.pose import AnimalPose, HumanPose, Keypoint, PoseFrame
from src.visualization.skeletons import AP10K_EDGES, COCO_17_EDGES


def _keypoint_color(
    kp: Keypoint,
    high_conf_color: Tuple[int, int, int],
    low_conf_color: Tuple[int, int, int],
    threshold: float,
) -> Tuple[int, int, int]:
    if kp.confidence >= threshold:
        return high_conf_color
    return low_conf_color


def _draw_keypoint(
    canvas: np.ndarray,
    kp: Keypoint,
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    import cv2  # noqa: PLC0415

    if not kp.visible and kp.confidence < 0.01:
        return
    r = max(1, int(round(radius * kp.confidence)))
    cv2.circle(canvas, (int(round(kp.x)), int(round(kp.y))), r, color, -1)


def _draw_skeleton_edge(
    canvas: np.ndarray,
    kp_a: Keypoint,
    kp_b: Keypoint,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    import cv2  # noqa: PLC0415

    if not kp_a.visible and kp_a.confidence < 0.01:
        return
    if not kp_b.visible and kp_b.confidence < 0.01:
        return
    alpha = min(kp_a.confidence, kp_b.confidence)
    if alpha < 0.01:
        return
    t = max(1, int(round(thickness * alpha)))
    pt_a = (int(round(kp_a.x)), int(round(kp_a.y)))
    pt_b = (int(round(kp_b.x)), int(round(kp_b.y)))
    overlay = canvas.copy()
    cv2.line(overlay, pt_a, pt_b, color, t)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)


class PoseOverlay:
    """Draws human/animal keypoints and skeleton edges on the canvas.

    Args:
        keypoint_radius: Base radius of keypoint circles in pixels.
        edge_thickness: Base thickness of skeleton edge lines.
        high_conf_threshold: Confidence at/bove which ``high_conf_color`` is used.
        high_conf_color: BGR color for high-confidence keypoints & edges.
        low_conf_color: BGR color for low-confidence keypoints & edges.
        human_edge_color: BGR color for human skeleton edges.
        animal_edge_color: BGR color for animal skeleton edges.
    """

    def __init__(
        self,
        keypoint_radius: int = 4,
        edge_thickness: int = 2,
        high_conf_threshold: float = 0.5,
        high_conf_color: Tuple[int, int, int] = (0, 255, 0),
        low_conf_color: Tuple[int, int, int] = (0, 100, 0),
        human_edge_color: Tuple[int, int, int] = (255, 255, 0),
        animal_edge_color: Tuple[int, int, int] = (255, 0, 255),
    ) -> None:
        self._kp_radius = keypoint_radius
        self._edge_thickness = edge_thickness
        self._conf_thresh = high_conf_threshold
        self._high_conf_color = high_conf_color
        self._low_conf_color = low_conf_color
        self._human_edge_color = human_edge_color
        self._animal_edge_color = animal_edge_color

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        pose_frame: Optional[PoseFrame] = ctx.get("pose")
        if pose_frame is None:
            return canvas

        for human in pose_frame.humans:
            self._draw_human(canvas, human)
        for animal in pose_frame.animals:
            self._draw_animal(canvas, animal)

        return canvas

    def _draw_human(self, canvas: np.ndarray, human: HumanPose) -> None:
        kp_map = self._build_kp_map(human.keypoints)
        for i, j in COCO_17_EDGES:
            if i in kp_map and j in kp_map:
                _draw_skeleton_edge(
                    canvas, kp_map[i], kp_map[j],
                    self._human_edge_color, self._edge_thickness,
                )
        for kp in human.keypoints:
            color = _keypoint_color(kp, self._high_conf_color, self._low_conf_color, self._conf_thresh)
            _draw_keypoint(canvas, kp, self._kp_radius, color)

    def _draw_animal(self, canvas: np.ndarray, animal: AnimalPose) -> None:
        kp_map = self._build_kp_map(animal.keypoints)
        for i, j in AP10K_EDGES:
            if i in kp_map and j in kp_map:
                _draw_skeleton_edge(
                    canvas, kp_map[i], kp_map[j],
                    self._animal_edge_color, self._edge_thickness,
                )
        for kp in animal.keypoints:
            color = _keypoint_color(kp, self._high_conf_color, self._low_conf_color, self._conf_thresh)
            _draw_keypoint(canvas, kp, self._kp_radius, color)

    @staticmethod
    def _build_kp_map(
        keypoints: Tuple[Keypoint, ...],
    ) -> Dict[int, Keypoint]:
        return {i: kp for i, kp in enumerate(keypoints)}
