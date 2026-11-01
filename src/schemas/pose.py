from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class HumanKeypointName(Enum):
    """COCO 17-keypoint names in standard order.

    The order follows the COCO keypoint convention:
    nose, left_eye, right_eye, left_ear, right_ear,
    left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist, left_hip, right_hip,
    left_knee, right_knee, left_ankle, right_ankle.
    """

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


class AnimalKeypointName(Enum):
    """AP-10K 17-keypoint names in standard order.

    Follows the convention from the AP-10K dataset (Yu et al., NeurIPS 2022).
    """

    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    NECK = "neck"
    LEFT_FRONT_SHOULDER = "left_front_shoulder"
    RIGHT_FRONT_SHOULDER = "right_front_shoulder"
    LEFT_FRONT_PAW = "left_front_paw"
    RIGHT_FRONT_PAW = "right_front_paw"
    LEFT_BACK_SHOULDER = "left_back_shoulder"
    RIGHT_BACK_SHOULDER = "right_back_shoulder"
    LEFT_BACK_PAW = "left_back_paw"
    RIGHT_BACK_PAW = "right_back_paw"
    TAIL = "tail"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    HEAD_TOP = "head_top"
    MID_BACK = "mid_back"


@dataclass(frozen=True, slots=True)
class Keypoint:
    """A single detected keypoint with image-space coordinates.

    x, y: pixel coordinates in the image plane.
    confidence: detection confidence in [0, 1].
    visible: whether the keypoint is visible (not occluded or truncated).
    name: semantic label matching the corresponding keypoint enum.
    """

    name: str
    x: float
    y: float
    confidence: float
    visible: bool


@dataclass(frozen=True, slots=True)
class HumanPose:
    """Pose estimation for a single human subject.

    keypoints follow COCO 17-keypoint order.
    bbox is (x1, y1, x2, y2) in pixel coordinates.
    body_orientation_deg is the yaw angle in the image plane (0 = facing camera).
    """

    track_id: Optional[int]
    detection_id: int
    keypoints: tuple[Keypoint, ...]
    bbox: tuple[float, float, float, float]
    overall_confidence: float
    body_orientation_deg: Optional[float]
    head_orientation_deg: Optional[float]
    timestamp: float
    schema_version: str = "1.0"


@dataclass(frozen=True, slots=True)
class AnimalPose:
    """Pose estimation for a single animal subject.

    keypoints follow AP-10K 17-keypoint order.
    species identifies the animal type (e.g. "dog", "cow", "deer", "unknown").
    """

    track_id: Optional[int]
    detection_id: int
    keypoints: tuple[Keypoint, ...]
    bbox: tuple[float, float, float, float]
    overall_confidence: float
    body_orientation_deg: Optional[float]
    head_orientation_deg: Optional[float]
    species: str
    timestamp: float
    schema_version: str = "1.0"


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Container for all pose detections in a single frame."""

    frame_id: int
    timestamp: float
    humans: tuple[HumanPose, ...]
    animals: tuple[AnimalPose, ...]
    schema_version: str = "1.0"
