from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.schemas import Detection, TrackedObject
from src.schemas.bbox import BoundingBox


@pytest.fixture(scope="session")
def synthetic_frame() -> np.ndarray:
    """Return a small synthetic BGR image (480 x 640) for use in tests."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple rectangle to simulate a person-like region
    img[100:400, 200:500] = (128, 128, 128)
    return img


@pytest.fixture(scope="session")
def single_person_detections() -> List[Detection]:
    """Return one person Detection."""
    return [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(200.0, 100.0, 500.0, 400.0),
        ),
    ]


@pytest.fixture(scope="session")
def multi_person_detections() -> List[Detection]:
    """Return multiple person Detections with varying confidence."""
    return [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(50.0, 50.0, 150.0, 350.0),
        ),
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=BoundingBox(200.0, 60.0, 350.0, 380.0),
        ),
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.75,
            bbox=BoundingBox(400.0, 80.0, 600.0, 400.0),
        ),
        Detection(
            class_id=1,
            class_name="car",
            confidence=0.90,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
        ),
    ]


@pytest.fixture(scope="session")
def mock_tracks() -> List[TrackedObject]:
    """Return tracked objects matching person detections."""
    return [
        TrackedObject(
            track_id=1, class_name="person", confidence=0.95,
            bbox=BoundingBox(50.0, 50.0, 150.0, 350.0),
        ),
        TrackedObject(
            track_id=2, class_name="person", confidence=0.85,
            bbox=BoundingBox(200.0, 60.0, 350.0, 380.0),
        ),
    ]


@pytest.fixture(scope="session")
def no_person_detections() -> List[Detection]:
    """Return detections with no person class."""
    return [
        Detection(
            class_id=1,
            class_name="car",
            confidence=0.95,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
        ),
    ]


@pytest.fixture(scope="session")
def coco17_keypoints_facing() -> np.ndarray:
    """Return synthetic COCO-17 keypoints for a person facing the camera.

    Keypoints are in (17, 3) format with rough pixel positions.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Symmetric keypoints for a person roughly in center
    kpts[0] = [320, 180, 0.95]   # nose
    kpts[1] = [300, 170, 0.90]   # left_eye
    kpts[2] = [340, 170, 0.90]   # right_eye
    kpts[3] = [290, 175, 0.80]   # left_ear
    kpts[4] = [350, 175, 0.80]   # right_ear
    kpts[5] = [260, 230, 0.95]   # left_shoulder
    kpts[6] = [380, 230, 0.95]   # right_shoulder
    kpts[7] = [240, 300, 0.90]   # left_elbow
    kpts[8] = [400, 300, 0.90]   # right_elbow
    kpts[9] = [230, 370, 0.85]   # left_wrist
    kpts[10] = [410, 370, 0.85]  # right_wrist
    kpts[11] = [280, 370, 0.90]  # left_hip
    kpts[12] = [360, 370, 0.90]  # right_hip
    kpts[13] = [270, 430, 0.85]  # left_knee
    kpts[14] = [370, 430, 0.85]  # right_knee
    kpts[15] = [260, 470, 0.80]  # left_ankle
    kpts[16] = [380, 470, 0.80]  # right_ankle
    return kpts


@pytest.fixture(scope="session")
def coco17_keypoints_right_profile() -> np.ndarray:
    """Return synthetic COCO-17 keypoints for a person in right profile."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Asymmetric: right side prominent, left side low confidence/visible
    kpts[0] = [380, 190, 0.95]   # nose (shifted right)
    kpts[1] = [0, 0, 0.10]       # left_eye (not visible)
    kpts[2] = [390, 180, 0.90]   # right_eye
    kpts[3] = [0, 0, 0.10]       # left_ear (not visible)
    kpts[4] = [400, 185, 0.85]   # right_ear
    kpts[5] = [0, 0, 0.10]       # left_shoulder (not visible)
    kpts[6] = [420, 240, 0.95]   # right_shoulder
    kpts[7] = [0, 0, 0.10]       # left_elbow
    kpts[8] = [440, 310, 0.90]   # right_elbow
    kpts[9] = [0, 0, 0.10]       # left_wrist
    kpts[10] = [450, 380, 0.85]  # right_wrist
    kpts[11] = [0, 0, 0.10]      # left_hip
    kpts[12] = [400, 380, 0.90]  # right_hip
    kpts[13] = [0, 0, 0.10]      # left_knee
    kpts[14] = [390, 440, 0.85]  # right_knee
    kpts[15] = [0, 0, 0.10]      # left_ankle
    kpts[16] = [380, 480, 0.80]  # right_ankle
    return kpts
