from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.schemas.pose import Keypoint

# Minimum confidence for a keypoint to be used in orientation estimation.
_MIN_ORIENT_CONF: float = 0.3


def _kp(
    d: Dict[str, Keypoint], name: str, min_conf: float = _MIN_ORIENT_CONF
) -> Optional[np.ndarray]:
    """Return ``(x, y)`` array if the named keypoint exists and has sufficient confidence."""
    kp = d.get(name)
    if kp is not None and kp.confidence >= min_conf:
        return np.array([kp.x, kp.y], dtype=np.float32)
    return None


def compute_body_orientation(
    keypoints: Dict[str, Keypoint], conf_threshold: float
) -> Optional[float]:
    """Estimate body yaw (in degrees) from shoulder and hip keypoints.

    Uses the horizontal offset between the shoulder midpoint and hip midpoint
    as a proxy for torso rotation.

    Returns:
        Angle in degrees where:
            - 0° = roughly facing the camera
            - positive = torso rotated to the subject's right
            - negative = torso rotated to the subject's left
            - ``None`` = insufficient confident keypoints
    """
    ls = _kp(keypoints, "left_shoulder", conf_threshold)
    rs = _kp(keypoints, "right_shoulder", conf_threshold)
    lh = _kp(keypoints, "left_hip", conf_threshold)
    rh = _kp(keypoints, "right_hip", conf_threshold)

    have_shoulders = ls is not None and rs is not None
    have_hips = lh is not None and rh is not None

    if not have_shoulders and not have_hips:
        return None

    shoulder_mid_x: Optional[float] = None
    hip_mid_x: Optional[float] = None
    shoulder_width: Optional[float] = None

    if have_shoulders and ls is not None and rs is not None:
        shoulder_mid_x = float((ls[0] + rs[0]) / 2.0)
        shoulder_width = float(abs(rs[0] - ls[0]))

    if have_hips and lh is not None and rh is not None:
        hip_mid_x = float((lh[0] + rh[0]) / 2.0)

    if shoulder_mid_x is not None and hip_mid_x is not None and shoulder_width is not None and shoulder_width > 1.0:
        offset = (shoulder_mid_x - hip_mid_x) / shoulder_width
        offset = np.clip(offset, -1.0, 1.0)
        # arcsin maps horizontal offset to an angle estimate
        angle = float(np.degrees(np.arcsin(offset)))
        return angle

    if have_shoulders and shoulder_width is not None and shoulder_width > 1.0:
        ls_conf = keypoints["left_shoulder"].confidence
        rs_conf = keypoints["right_shoulder"].confidence
        conf_ratio = (ls_conf - rs_conf) / (ls_conf + rs_conf + 1e-8)
        return float(np.clip(conf_ratio * 45.0, -45.0, 45.0))

    return None


def compute_head_orientation(
    keypoints: Dict[str, Keypoint], conf_threshold: float
) -> Optional[float]:
    """Estimate head yaw (in degrees) from eye, ear, and nose keypoints.

    Uses the asymmetry of visible eyes/ears to estimate which way the head
    is turned.

    Returns:
        Angle in degrees where:
            - 0° = facing the camera
            - positive = head turned to the subject's right
            - negative = head turned to the subject's left
            - ``None`` = insufficient confident keypoints
    """
    le = _kp(keypoints, "left_eye", conf_threshold)
    re = _kp(keypoints, "right_eye", conf_threshold)
    nose = _kp(keypoints, "nose", conf_threshold)

    if le is not None and re is not None:
        # Both eyes visible → head roughly toward camera.
        # Use horizontal offset of nose relative to eyes midpoint.
        eye_mid_x = (le[0] + re[0]) / 2.0
        eye_width = abs(re[0] - le[0])

        if nose is not None and eye_width > 1.0:
            offset = (nose[0] - eye_mid_x) / (eye_width / 2.0)
            offset = np.clip(offset, -1.0, 1.0)
            return float(np.degrees(np.arcsin(offset * 0.7)))

        # No nose: return small angle based on confidence symmetry.
        le_conf = keypoints["left_eye"].confidence
        re_conf = keypoints["right_eye"].confidence
        ratio = (le_conf - re_conf) / (le_conf + re_conf + 1e-8)
        return float(np.clip(ratio * 30.0, -30.0, 30.0))

    if le is not None or re is not None:
        # Only one eye visible → head is turned to that side.
        # Rough estimate: ±45° depending on which eye.
        if le is not None and re is None:
            return -45.0
        return 45.0

    lea = _kp(keypoints, "left_ear", conf_threshold)
    rea = _kp(keypoints, "right_ear", conf_threshold)

    if lea is not None and rea is not None:
        # Both ears visible → head roughly toward camera.
        return 0.0

    if lea is not None:
        return -45.0
    if rea is not None:
        return 45.0

    return None
