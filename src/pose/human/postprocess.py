from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from src.pose.human.orientation import (
    compute_body_orientation,
    compute_head_orientation,
)
from src.schemas.pose import HumanPose, HumanKeypointName, Keypoint

# COCO 17 keypoint names in canonical order (matches HumanKeypointName enum).
_COCO_17_NAMES: Tuple[str, ...] = tuple(kp.value for kp in HumanKeypointName)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid."""
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


def decode_onnx_output(
    raw: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Decode YOLOv8-Pose ONNX output into per-detection keypoints.

    Args:
        raw: Raw ONNX output tensor of shape ``(1, 56, N)`` or ``(56, N)``.
        conf_threshold: Minimum confidence for a valid detection.
        iou_threshold: NMS IoU threshold.

    Returns:
        List of ``(keypoints, bbox, score)`` tuples where keypoints has shape
        ``(17, 3)``, bbox has shape ``(4,)``, and score is the detection
        confidence.  May be empty.
    """
    import cv2  # noqa: PLC0415

    raw = np.squeeze(raw)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] != 56:
        raw = raw.T

    if raw.ndim != 2 or raw.shape[0] != 56:
        return []

    num_dets = raw.shape[1]

    # Box: (cx, cy, w, h) — first 4 rows
    boxes_xywh = raw[:4, :].T  # (N, 4)
    # Classification: row 4
    cls_scores = sigmoid(raw[4, :])  # (N,)
    # Keypoints: rows 5..55 (17 * 3)
    kpts_raw = raw[5:56, :].T  # (N, 51) → (N, 17, 3)

    kpts_raw = kpts_raw.reshape(num_dets, 17, 3)

    # Decode boxes from cx,cy,w,h to x1,y1,x2,y2
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0  # y2

    # Filter by confidence
    mask = cls_scores >= conf_threshold
    if not np.any(mask):
        return []

    boxes_xyxy = boxes_xyxy[mask]
    cls_scores = cls_scores[mask]
    kpts_raw = kpts_raw[mask]

    if len(cls_scores) == 0:
        return []

    # Apply NMS
    # Convert boxes to pixel int tuples for cv2.dnn.NMSBoxes
    box_list = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_xyxy]
    raw_indices = cv2.dnn.NMSBoxes(box_list, cls_scores.tolist(), conf_threshold, iou_threshold)
    if len(raw_indices) == 0:
        return []

    indices_arr = np.asarray(raw_indices)
    if indices_arr.ndim > 1:
        indices_arr = indices_arr.flatten()
    indices = indices_arr.tolist()

    results: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for idx in indices:
        kpts = kpts_raw[idx]  # (17, 3)
        # Apply sigmoid to keypoint confidence values
        kpts[:, 2] = sigmoid(kpts[:, 2])
        results.append((kpts, boxes_xyxy[idx], float(cls_scores[idx])))

    return results


def build_human_pose(
    kpts_array: np.ndarray,
    bbox_array: np.ndarray,
    overall_confidence: float,
    detection_id: int,
    track_id: int | None,
    timestamp: float,
    conf_threshold: float,
) -> HumanPose:
    """Build a single HumanPose from model output data.

    Args:
        kpts_array: Array of shape ``(17, 3)`` with ``(x, y, confidence)``
            in **original image** coordinates.
        bbox_array: Array of shape ``(4,)`` with ``(x1, y1, x2, y2)`` in
            original image coordinates.
        overall_confidence: Detection confidence score.
        detection_id: Index of the source detection.
        track_id: Persistent track ID or None.
        timestamp: Frame timestamp.
        conf_threshold: Minimum keypoint confidence to mark as visible.

    Returns:
        A fully populated ``HumanPose`` instance.
    """
    names = _COCO_17_NAMES
    keypoints: List[Keypoint] = []

    for i in range(min(len(kpts_array), 17)):
        x, y, conf = kpts_array[i]
        visible = conf >= conf_threshold
        keypoints.append(
            Keypoint(
                name=names[i],
                x=float(x),
                y=float(y),
                confidence=float(conf),
                visible=visible,
            )
        )

    # Pad if fewer than 17 keypoints
    while len(keypoints) < 17:
        idx = len(keypoints)
        keypoints.append(
            Keypoint(name=names[idx], x=0.0, y=0.0, confidence=0.0, visible=False)
        )

    # Compute orientation
    kpt_dict = {kp.name: kp for kp in keypoints}
    body_orient = compute_body_orientation(kpt_dict, conf_threshold)
    head_orient = compute_head_orientation(kpt_dict, conf_threshold)

    return HumanPose(
        track_id=track_id,
        detection_id=detection_id,
        keypoints=tuple(keypoints),
        bbox=(
            float(bbox_array[0]),
            float(bbox_array[1]),
            float(bbox_array[2]),
            float(bbox_array[3]),
        ),
        overall_confidence=float(overall_confidence),
        body_orientation_deg=body_orient,
        head_orientation_deg=head_orient,
        timestamp=timestamp,
    )
