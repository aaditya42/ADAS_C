from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox


@pytest.fixture(scope="session")
def synthetic_frame() -> np.ndarray:
    """Return a small synthetic BGR image (480 x 640) for use in tests."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:400, 200:500] = (128, 128, 128)
    return img


@pytest.fixture(scope="session")
def single_dog_detection() -> List[Detection]:
    """Return one dog detection (COCO class 16)."""
    return [
        Detection(
            class_id=16,
            class_name="dog",
            confidence=0.92,
            bbox=BoundingBox(200.0, 100.0, 500.0, 400.0),
        ),
    ]


@pytest.fixture(scope="session")
def multi_animal_detections() -> List[Detection]:
    """Return multiple animal detections with varying confidence."""
    return [
        Detection(
            class_id=16,
            class_name="dog",
            confidence=0.95,
            bbox=BoundingBox(50.0, 50.0, 150.0, 350.0),
        ),
        Detection(
            class_id=19,
            class_name="cow",
            confidence=0.85,
            bbox=BoundingBox(200.0, 60.0, 350.0, 380.0),
        ),
        Detection(
            class_id=17,
            class_name="horse",
            confidence=0.75,
            bbox=BoundingBox(400.0, 80.0, 600.0, 400.0),
        ),
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.90,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
        ),
    ]


@pytest.fixture(scope="session")
def mock_animal_tracks() -> List[TrackedObject]:
    """Return tracked objects matching animal detections."""
    return [
        TrackedObject(
            track_id=10, class_name="dog", confidence=0.95,
            bbox=BoundingBox(50.0, 50.0, 150.0, 350.0),
        ),
        TrackedObject(
            track_id=11, class_name="cow", confidence=0.85,
            bbox=BoundingBox(200.0, 60.0, 350.0, 380.0),
        ),
    ]


@pytest.fixture(scope="session")
def no_animal_detections() -> List[Detection]:
    """Return detections with no animal class IDs."""
    return [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
        ),
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.90,
            bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
        ),
    ]


@pytest.fixture(scope="session")
def ap10k_keypoints_facing() -> np.ndarray:
    """Return synthetic AP-10K keypoints for an animal facing the camera.

    Shape (17, 3) with symmetric keypoints.
    AP-10K order:
      0=left_eye, 1=right_eye, 2=nose, 3=neck,
      4=left_front_shoulder, 5=right_front_shoulder,
      6=left_front_paw, 7=right_front_paw,
      8=left_back_shoulder, 9=right_back_shoulder,
      10=left_back_paw, 11=right_back_paw,
      12=tail, 13=left_ear, 14=right_ear, 15=head_top, 16=mid_back
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    cx, cy = 320.0, 240.0
    kpts[0] = [cx - 15, cy - 50, 0.90]   # left_eye
    kpts[1] = [cx + 15, cy - 50, 0.90]   # right_eye
    kpts[2] = [cx,      cy - 40, 0.95]   # nose
    kpts[3] = [cx,      cy - 20, 0.85]   # neck
    kpts[4] = [cx - 40, cy,      0.90]   # left_front_shoulder
    kpts[5] = [cx + 40, cy,      0.90]   # right_front_shoulder
    kpts[6] = [cx - 35, cy + 60, 0.80]   # left_front_paw
    kpts[7] = [cx + 35, cy + 60, 0.80]   # right_front_paw
    kpts[8] = [cx - 30, cy + 30, 0.85]   # left_back_shoulder
    kpts[9] = [cx + 30, cy + 30, 0.85]   # right_back_shoulder
    kpts[10] = [cx - 25, cy + 90, 0.75]  # left_back_paw
    kpts[11] = [cx + 25, cy + 90, 0.75]  # right_back_paw
    kpts[12] = [cx,      cy + 110, 0.70] # tail
    kpts[13] = [cx - 20, cy - 55, 0.80]  # left_ear
    kpts[14] = [cx + 20, cy - 55, 0.80]  # right_ear
    kpts[15] = [cx,      cy - 65, 0.85]  # head_top
    kpts[16] = [cx,      cy + 15, 0.75]  # mid_back
    return kpts


@pytest.fixture(scope="session")
def ap10k_keypoints_right_profile() -> np.ndarray:
    """Return synthetic AP-10K keypoints for an animal in right profile."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    cx, cy = 400.0, 240.0
    kpts[0] = [0, 0, 0.05]              # left_eye (not visible)
    kpts[1] = [cx,      cy - 50, 0.90]  # right_eye
    kpts[2] = [cx,      cy - 40, 0.95]  # nose
    kpts[3] = [cx,      cy - 20, 0.85]  # neck
    kpts[4] = [0, 0, 0.05]              # left_front_shoulder
    kpts[5] = [cx,      cy,      0.90]  # right_front_shoulder
    kpts[6] = [0, 0, 0.05]              # left_front_paw
    kpts[7] = [cx,      cy + 60, 0.80]  # right_front_paw
    kpts[8] = [0, 0, 0.05]              # left_back_shoulder
    kpts[9] = [cx,      cy + 30, 0.85]  # right_back_shoulder
    kpts[10] = [0, 0, 0.05]             # left_back_paw
    kpts[11] = [cx,     cy + 90, 0.75]  # right_back_paw
    kpts[12] = [cx,     cy + 110, 0.70] # tail
    kpts[13] = [0, 0, 0.05]             # left_ear
    kpts[14] = [cx - 5, cy - 55, 0.80]  # right_ear
    kpts[15] = [cx,     cy - 65, 0.85]  # head_top
    kpts[16] = [cx,     cy + 15, 0.75]  # mid_back
    return kpts
