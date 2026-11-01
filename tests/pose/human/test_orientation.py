from __future__ import annotations

from typing import Dict

import pytest

from src.pose.human.orientation import (
    compute_body_orientation,
    compute_head_orientation,
)
from src.schemas.pose import Keypoint


def _make_kpt(name: str, x: float, y: float, conf: float) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, confidence=conf, visible=conf >= 0.5)


_FACING_DICT: Dict[str, Keypoint] = {
    "left_shoulder": _make_kpt("left_shoulder", 260, 230, 0.95),
    "right_shoulder": _make_kpt("right_shoulder", 380, 230, 0.95),
    "left_hip": _make_kpt("left_hip", 280, 370, 0.90),
    "right_hip": _make_kpt("right_hip", 360, 370, 0.90),
    "nose": _make_kpt("nose", 320, 180, 0.95),
    "left_eye": _make_kpt("left_eye", 300, 170, 0.90),
    "right_eye": _make_kpt("right_eye", 340, 170, 0.90),
}

_RIGHT_PROFILE_DICT: Dict[str, Keypoint] = {
    "left_shoulder": _make_kpt("left_shoulder", 350, 240, 0.70),
    "right_shoulder": _make_kpt("right_shoulder", 450, 240, 0.95),
    "left_hip": _make_kpt("left_hip", 300, 380, 0.60),
    "right_hip": _make_kpt("right_hip", 400, 380, 0.90),
    "nose": _make_kpt("nose", 410, 190, 0.95),
    "right_eye": _make_kpt("right_eye", 420, 180, 0.90),
    "right_ear": _make_kpt("right_ear", 430, 185, 0.85),
}


class TestComputeBodyOrientation:
    """Tests for :func:`compute_body_orientation`."""

    def test_facing_camera(self) -> None:
        """Symmetric shoulders/hips → angle near 0°."""
        angle = compute_body_orientation(_FACING_DICT, conf_threshold=0.3)
        assert angle is not None
        assert abs(angle) < 15.0, f"Expected near 0°, got {angle:.1f}°"

    def test_right_profile(self) -> None:
        """Shoulders shifted right relative to hips → positive yaw angle."""
        angle = compute_body_orientation(_RIGHT_PROFILE_DICT, conf_threshold=0.3)
        assert angle is not None
        assert angle > 0.0, f"Expected positive angle for right profile, got {angle:.1f}°"

    def test_no_keypoints_returns_none(self) -> None:
        angle = compute_body_orientation({}, conf_threshold=0.3)
        assert angle is None

    def test_low_confidence_returns_none(self) -> None:
        low_conf = {
            "left_shoulder": _make_kpt("left_shoulder", 100, 100, 0.05),
            "right_shoulder": _make_kpt("right_shoulder", 200, 100, 0.05),
        }
        angle = compute_body_orientation(low_conf, conf_threshold=0.3)
        assert angle is None

    def test_only_one_shoulder(self) -> None:
        kpts = {
            "left_shoulder": _make_kpt("left_shoulder", 100, 100, 0.95),
        }
        angle = compute_body_orientation(kpts, conf_threshold=0.3)
        assert angle is None

    def test_shoulder_only_fallback(self) -> None:
        """When only shoulders are available (no hips), use confidence ratio."""
        kpts = {
            "left_shoulder": _make_kpt("left_shoulder", 260, 230, 0.90),
            "right_shoulder": _make_kpt("right_shoulder", 380, 230, 0.50),
        }
        angle = compute_body_orientation(kpts, conf_threshold=0.3)
        assert angle is not None
        # Left shoulder higher conf → left side more visible → rotated to
        # subject's right → positive angle (per docstring convention)
        assert angle > 0.0, f"Expected positive angle, got {angle:.1f}°"


class TestComputeHeadOrientation:
    """Tests for :func:`compute_head_orientation`."""

    def test_facing_camera(self) -> None:
        """Both eyes visible, nose centered → angle near 0°."""
        angle = compute_head_orientation(_FACING_DICT, conf_threshold=0.3)
        assert angle is not None
        assert abs(angle) < 15.0, f"Expected near 0°, got {angle:.1f}°"

    def test_right_profile(self) -> None:
        """Only right eye visible → turned to right → positive angle."""
        angle = compute_head_orientation(_RIGHT_PROFILE_DICT, conf_threshold=0.3)
        assert angle is not None
        assert angle > 0.0, f"Expected positive for right profile, got {angle:.1f}°"

    def test_no_keypoints_returns_none(self) -> None:
        angle = compute_head_orientation({}, conf_threshold=0.3)
        assert angle is None

    def test_low_confidence_returns_none(self) -> None:
        kpts = {
            "left_eye": _make_kpt("left_eye", 300, 170, 0.05),
            "right_eye": _make_kpt("right_eye", 340, 170, 0.05),
        }
        angle = compute_head_orientation(kpts, conf_threshold=0.3)
        assert angle is None

    def test_both_ears_visible(self) -> None:
        """Both ears visible → head roughly facing camera."""
        kpts = {
            "left_ear": _make_kpt("left_ear", 280, 180, 0.80),
            "right_ear": _make_kpt("right_ear", 360, 180, 0.80),
        }
        angle = compute_head_orientation(kpts, conf_threshold=0.3)
        assert angle is not None
        assert abs(angle) < 10.0

    def test_only_left_ear(self) -> None:
        kpts = {
            "left_ear": _make_kpt("left_ear", 280, 180, 0.80),
        }
        angle = compute_head_orientation(kpts, conf_threshold=0.3)
        assert angle is not None
        assert angle < 0.0, f"Expected negative for left ear only, got {angle:.1f}°"
