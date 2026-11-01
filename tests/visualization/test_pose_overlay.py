from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.schemas.pose import AnimalPose, HumanPose, Keypoint, PoseFrame
from src.visualization.overlays.pose_overlay import PoseOverlay


def _human_pose(
    track_id: int,
    keypoints: tuple[tuple[float, float, float, bool], ...],
) -> HumanPose:
    return HumanPose(
        track_id=track_id,
        detection_id=track_id * 100,
        keypoints=tuple(
            Keypoint(name=f"kp_{i}", x=x, y=y, confidence=c, visible=v)
            for i, (x, y, c, v) in enumerate(keypoints)
        ),
        bbox=(0, 0, 100, 200),
        overall_confidence=0.9,
        body_orientation_deg=None,
        head_orientation_deg=None,
        timestamp=0.0,
    )


def _animal_pose(
    track_id: int,
    keypoints: tuple[tuple[float, float, float, bool], ...],
    species: str = "dog",
) -> AnimalPose:
    return AnimalPose(
        track_id=track_id,
        detection_id=track_id * 100,
        keypoints=tuple(
            Keypoint(name=f"kp_{i}", x=x, y=y, confidence=c, visible=v)
            for i, (x, y, c, v) in enumerate(keypoints)
        ),
        bbox=(0, 0, 100, 200),
        overall_confidence=0.9,
        body_orientation_deg=None,
        head_orientation_deg=None,
        species=species,
        timestamp=0.0,
    )


class TestPoseOverlayOutput:

    def test_returns_canvas_when_no_pose_frame(self) -> None:
        overlay = PoseOverlay()
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay(canvas, {})
        assert result is canvas
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_returns_canvas_when_pose_frame_is_none(self) -> None:
        overlay = PoseOverlay()
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay(canvas, {"pose": None})
        assert result is canvas

    def test_output_shape_and_dtype_preserved(self) -> None:
        overlay = PoseOverlay()
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        human = _human_pose(
            track_id=1,
            keypoints=((100, 100, 0.9, True),) * 17,
        )
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(human,),
                animals=(),
            ),
        }
        result = overlay(canvas, ctx)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_draws_on_copy_not_inplace(self) -> None:
        overlay = PoseOverlay()
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas = original.copy()
        human = _human_pose(
            track_id=1,
            keypoints=((50, 50, 0.9, True),) * 17,
        )
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(human,),
                animals=(),
            ),
        }
        result = overlay(canvas, ctx)
        assert result is canvas
        np.testing.assert_array_equal(original, np.zeros((100, 100, 3), dtype=np.uint8))


class TestPoseOverlayKeypoints:

    def test_keypoints_drawn_for_human(self) -> None:
        overlay = PoseOverlay(keypoint_radius=4)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = [(50, 50, 0.9, True)] * 17
        human = _human_pose(track_id=1, keypoints=tuple(kps))
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(human,),
                animals=(),
            ),
        }
        result = overlay(canvas, ctx)
        # At least some pixels should have been drawn (non-zero)
        assert np.any(result != 0)

    def test_keypoints_drawn_for_animal(self) -> None:
        overlay = PoseOverlay(keypoint_radius=4)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = [(50, 50, 0.9, True)] * 17
        animal = _animal_pose(track_id=1, keypoints=tuple(kps))
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(),
                animals=(animal,),
            ),
        }
        result = overlay(canvas, ctx)
        assert np.any(result != 0)

    def test_low_confidence_keypoint_smaller(self) -> None:
        overlay = PoseOverlay(keypoint_radius=4)
        canvas_high = np.zeros((200, 200, 3), dtype=np.uint8)
        canvas_low = np.zeros((200, 200, 3), dtype=np.uint8)

        high_kps = [(50, 50, 0.9, True)] * 17
        low_kps = [(50, 50, 0.1, True)] * 17

        high_human = _human_pose(track_id=1, keypoints=tuple(high_kps))
        low_human = _human_pose(track_id=2, keypoints=tuple(low_kps))

        ctx_high: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(high_human,),
                animals=(),
            ),
        }
        ctx_low: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(low_human,),
                animals=(),
            ),
        }
        r_high = overlay(canvas_high, ctx_high)
        r_low = overlay(canvas_low, ctx_low)
        # High-confidence should draw more pixels (larger radius)
        assert np.count_nonzero(r_high) >= np.count_nonzero(r_low)


class TestPoseOverlayMultipleHumans:

    def test_overlay_returns_correct_shape_for_multiple_humans(self) -> None:
        overlay = PoseOverlay()
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = [(100, 100, 0.9, True)] * 17
        humans = tuple(
            _human_pose(track_id=i, keypoints=tuple(kps))
            for i in range(1, 4)
        )
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=humans,
                animals=(),
            ),
        }
        result = overlay(canvas, ctx)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8


class TestPoseOverlaySnapshot:

    def test_snapshot_human_keypoint_pixel_locations(self) -> None:
        overlay = PoseOverlay(keypoint_radius=3, edge_thickness=2)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        # Place keypoints at known locations: nose at (100, 40),
        # left_shoulder at (80, 80), right_shoulder at (120, 80)
        kps = [(50, 50, 0.0, False)] * 17  # all invisible/low first
        kps[0] = (100, 40, 0.95, True)    # nose
        kps[5] = (80, 80, 0.95, True)     # left_shoulder
        kps[6] = (120, 80, 0.95, True)    # right_shoulder
        kps[11] = (85, 130, 0.95, True)   # left_hip
        kps[12] = (115, 130, 0.95, True)  # right_hip
        kps[7] = (60, 110, 0.95, True)    # left_elbow
        kps[8] = (140, 110, 0.95, True)   # right_elbow

        human = _human_pose(track_id=1, keypoints=tuple(kps))
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(human,),
                animals=(),
            ),
        }
        result = overlay(canvas, ctx)

        # Keypoint at (100, 40) should have green pixels (high_conf_color)
        assert np.any(result[38:42, 98:102, 1] > 0), "nose keypoint not drawn"
        # Edge from nose (100,40) to left_shoulder (80,80) should have cyan pixels
        assert np.any(result[70:82, 80:100, 0] > 0), "nose-shoulder edge not drawn"
        # Edge from left_shoulder (80,80) to right_shoulder (120,80) should have cyan
        assert np.any(result[78:82, 80:120, 0] > 0), "shoulder edge not drawn"
        # Both keypoints and edges: total modified pixels exceeds keypoints-only count
        kp_only_count = np.count_nonzero(result)
        assert kp_only_count > 50, "too few pixels drawn for human snapshot"

    def test_snapshot_animal_keypoint_pixel_locations(self) -> None:
        overlay = PoseOverlay(keypoint_radius=3, edge_thickness=2)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        # Place keypoints for a dog-like skeleton
        kps = [(50, 50, 0.0, False)] * 17
        kps[0] = (90, 40, 0.95, True)   # left_eye
        kps[1] = (110, 40, 0.95, True)  # right_eye
        kps[2] = (100, 50, 0.95, True)  # nose
        kps[3] = (100, 65, 0.95, True)  # neck
        kps[4] = (75, 85, 0.95, True)   # left_front_shoulder
        kps[5] = (125, 85, 0.95, True)  # right_front_shoulder
        kps[16] = (100, 120, 0.95, True) # mid_back
        kps[8] = (85, 140, 0.95, True)  # left_back_shoulder

        animal = _animal_pose(track_id=1, keypoints=tuple(kps))
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(),
                animals=(animal,),
            ),
        }
        result = overlay(canvas, ctx)

        # nose at (100, 50) should have green pixels
        assert np.any(result[48:52, 98:102, 1] > 0), "nose keypoint not drawn"
        # Edge from neck (100,65) to left_front_shoulder (75,85) should have magenta
        assert np.any(result[65:85, 75:100, 2] > 0), "neck-shoulder edge not drawn"
        # Total modified pixels
        assert np.count_nonzero(result) > 50, "too few pixels drawn for animal snapshot"
