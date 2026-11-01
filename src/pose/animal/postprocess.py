from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.schemas.pose import AnimalKeypointName, AnimalPose, Keypoint

# AP-10K 17 keypoint names in canonical order (matches AnimalKeypointName enum).
_AP10K_17_NAMES: tuple[str, ...] = tuple(kp.value for kp in AnimalKeypointName)

# Minimum confidence for a keypoint to be used in orientation estimation.
_MIN_ORIENT_CONF: float = 0.3


def compute_body_orientation(
    keypoints: Dict[str, Keypoint], conf_threshold: float
) -> Optional[float]:
    """Estimate animal body yaw (in degrees) from front/back shoulder keypoints.

    Uses the horizontal offset between the midpoint of the front shoulders
    and the midpoint of the back shoulders as a proxy for torso rotation.

    Returns:
        Angle in degrees where:
            - 0° = roughly facing the camera
            - positive = body rotated to the animal's right
            - negative = body rotated to the animal's left
            - ``None`` = insufficient confident keypoints
    """
    lfs = _kp(keypoints, "left_front_shoulder", conf_threshold)
    rfs = _kp(keypoints, "right_front_shoulder", conf_threshold)
    lbs = _kp(keypoints, "left_back_shoulder", conf_threshold)
    rbs = _kp(keypoints, "right_back_shoulder", conf_threshold)

    have_front = lfs is not None and rfs is not None
    have_back = lbs is not None and rbs is not None

    if not have_front and not have_back:
        return None

    front_mid_x: Optional[float] = None
    back_mid_x: Optional[float] = None
    front_width: Optional[float] = None

    if have_front and lfs is not None and rfs is not None:
        front_mid_x = float((lfs[0] + rfs[0]) / 2.0)
        front_width = float(abs(rfs[0] - lfs[0]))

    if have_back and lbs is not None and rbs is not None:
        back_mid_x = float((lbs[0] + rbs[0]) / 2.0)

    if front_mid_x is not None and back_mid_x is not None and front_width is not None and front_width > 1.0:
        offset = (front_mid_x - back_mid_x) / front_width
        offset = np.clip(offset, -1.0, 1.0)
        angle = float(np.degrees(np.arcsin(offset)))
        return angle

    if have_front and front_width is not None and front_width > 1.0:
        lfs_conf = keypoints["left_front_shoulder"].confidence
        rfs_conf = keypoints["right_front_shoulder"].confidence
        conf_ratio = (lfs_conf - rfs_conf) / (lfs_conf + rfs_conf + 1e-8)
        return float(np.clip(conf_ratio * 45.0, -45.0, 45.0))

    return None


def compute_head_orientation(
    keypoints: Dict[str, Keypoint], conf_threshold: float
) -> Optional[float]:
    """Estimate animal head yaw (in degrees) from eye, ear, and nose keypoints.

    Uses the asymmetry of visible eyes/ears to estimate which way the head
    is turned.

    Returns:
        Angle in degrees where:
            - 0° = facing the camera
            - positive = head turned to the animal's right
            - negative = head turned to the animal's left
            - ``None`` = insufficient confident keypoints
    """
    le = _kp(keypoints, "left_eye", conf_threshold)
    re = _kp(keypoints, "right_eye", conf_threshold)
    nose = _kp(keypoints, "nose", conf_threshold)

    if le is not None and re is not None:
        eye_mid_x = (le[0] + re[0]) / 2.0
        eye_width = abs(re[0] - le[0])

        if nose is not None and eye_width > 1.0:
            offset = (nose[0] - eye_mid_x) / (eye_width / 2.0)
            offset = np.clip(offset, -1.0, 1.0)
            return float(np.degrees(np.arcsin(offset * 0.7)))

        le_conf = keypoints["left_eye"].confidence
        re_conf = keypoints["right_eye"].confidence
        ratio = (le_conf - re_conf) / (le_conf + re_conf + 1e-8)
        return float(np.clip(ratio * 30.0, -30.0, 30.0))

    if le is not None or re is not None:
        if le is not None and re is None:
            return -45.0
        return 45.0

    lea = _kp(keypoints, "left_ear", conf_threshold)
    rea = _kp(keypoints, "right_ear", conf_threshold)

    if lea is not None and rea is not None:
        return 0.0

    if lea is not None:
        return -45.0
    if rea is not None:
        return 45.0

    return None


def _kp(
    d: Dict[str, Keypoint], name: str, min_conf: float = _MIN_ORIENT_CONF
) -> Optional[np.ndarray]:
    """Return ``(x, y)`` array if the named keypoint exists and has sufficient confidence."""
    kp = d.get(name)
    if kp is not None and kp.confidence >= min_conf:
        return np.array([kp.x, kp.y], dtype=np.float32)
    return None


def build_animal_pose(
    kpts_array: np.ndarray,
    bbox_array: np.ndarray,
    overall_confidence: float,
    detection_id: int,
    track_id: int | None,
    timestamp: float,
    conf_threshold: float,
    species: str,
) -> AnimalPose:
    """Build a single AnimalPose from model output data.

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
        species: Animal species string (e.g. ``"dog"``, ``"cow"``).

    Returns:
        A fully populated ``AnimalPose`` instance.
    """
    names = _AP10K_17_NAMES
    keypoints_list: List[Keypoint] = []

    for i in range(min(len(kpts_array), 17)):
        x, y, conf = kpts_array[i]
        visible = conf >= conf_threshold
        keypoints_list.append(
            Keypoint(
                name=names[i],
                x=float(x),
                y=float(y),
                confidence=float(conf),
                visible=visible,
            )
        )

    while len(keypoints_list) < 17:
        idx = len(keypoints_list)
        keypoints_list.append(
            Keypoint(name=names[idx], x=0.0, y=0.0, confidence=0.0, visible=False)
        )

    kpt_dict = {kp.name: kp for kp in keypoints_list}
    body_orient = compute_body_orientation(kpt_dict, conf_threshold)
    head_orient = compute_head_orientation(kpt_dict, conf_threshold)

    return AnimalPose(
        track_id=track_id,
        detection_id=detection_id,
        keypoints=tuple(keypoints_list),
        bbox=(
            float(bbox_array[0]),
            float(bbox_array[1]),
            float(bbox_array[2]),
            float(bbox_array[3]),
        ),
        overall_confidence=float(overall_confidence),
        body_orientation_deg=body_orient,
        head_orientation_deg=head_orient,
        species=species,
        timestamp=timestamp,
    )


__all__ = [
    "build_animal_pose",
    "compute_body_orientation",
    "compute_head_orientation",
]
