from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.schemas.behavior import BehaviorFrame, BehaviorLabel, BehaviorState
from src.schemas.pose import HumanPose, Keypoint, PoseFrame
from src.visualization.overlays.behavior_overlay import BehaviorOverlay


def _state(
    track_id: int,
    label: BehaviorLabel,
    confidence: float = 0.85,
    crossing_prob: float | None = None,
) -> BehaviorState:
    return BehaviorState(
        track_id=track_id,
        entity_type="human" if "human" in label.value else "animal",
        primary_label=label,
        secondary_labels=(),
        confidence=confidence,
        crossing_probability=crossing_prob,
        timestamp=0.0,
    )


def _human_pose(track_id: int) -> HumanPose:
    return HumanPose(
        track_id=track_id,
        detection_id=track_id * 100,
        keypoints=(),
        bbox=(50, 50, 200, 300),
        overall_confidence=0.9,
        body_orientation_deg=None,
        head_orientation_deg=None,
        timestamp=0.0,
    )


class TestBehaviorOverlayNoop:

    def test_no_behavior_frame(self) -> None:
        overlay = BehaviorOverlay()
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay(canvas, {})
        assert result is canvas

    def test_behavior_frame_is_none(self) -> None:
        overlay = BehaviorOverlay()
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay(canvas, {"behavior": None})
        assert result is canvas


class TestBehaviorOverlayOutput:

    def test_output_shape_and_dtype_preserved(self) -> None:
        overlay = BehaviorOverlay()
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(1),),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(
                    _state(1, BehaviorLabel.HUMAN_WALKING),
                ),
            ),
        }
        result = overlay(canvas, ctx)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_skips_state_without_matching_pose(self) -> None:
        overlay = BehaviorOverlay()
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(
                    _state(99, BehaviorLabel.HUMAN_WALKING),
                ),
            ),
        }
        result = overlay(canvas, ctx)
        np.testing.assert_array_equal(result, np.zeros((100, 100, 3), dtype=np.uint8))


class TestBehaviorOverlayLabels:

    def test_badge_drawn_for_human_label(self) -> None:
        overlay = BehaviorOverlay(offset_x=-20, offset_y=-10)
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(1),),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(
                    _state(1, BehaviorLabel.HUMAN_FALLEN),
                ),
            ),
        }
        result = overlay(canvas, ctx)
        assert np.any(result != 0)

    def test_badge_drawn_with_crossing_probability(self) -> None:
        overlay = BehaviorOverlay(offset_x=-20, offset_y=-10)
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(1),),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(
                    _state(1, BehaviorLabel.HUMAN_CROSSING_ROAD, crossing_prob=0.85),
                ),
            ),
        }
        result = overlay(canvas, ctx)
        assert np.any(result != 0)

    def test_multiple_states(self) -> None:
        overlay = BehaviorOverlay(offset_x=-20, offset_y=-10)
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(1), _human_pose(2)),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(
                    _state(1, BehaviorLabel.HUMAN_STANDING),
                    _state(2, BehaviorLabel.HUMAN_RUNNING),
                ),
            ),
        }
        result = overlay(canvas, ctx)
        assert np.any(result != 0)


class TestBehaviorOverlayConfig:

    def test_custom_label_color(self) -> None:
        overlay = BehaviorOverlay(
            label_colors={"human.human_fallen": [0, 0, 255]},
            offset_x=-20, offset_y=-10,
        )
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(1),),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(
                    _state(1, BehaviorLabel.HUMAN_FALLEN),
                ),
            ),
        }
        result = overlay(canvas, ctx)
        assert np.any(result != 0)


class TestBehaviorOverlaySnapshot:

    def test_snapshot_badge_pixel_locations(self) -> None:
        overlay = BehaviorOverlay(offset_x=-20, offset_y=-10, font_scale=0.5, font_thickness=1)
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        state = _state(1, BehaviorLabel.HUMAN_WALKING, confidence=0.85, crossing_prob=0.75)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(1),),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(state,),
            ),
        }
        result = overlay(canvas, ctx)

        # The bbox is (50, 50, 200, 300); offset_x=-20, offset_y=-10
        # Badge bottom-right is at x=180, y=290
        # The badge rectangle should be visible (non-zero pixels in that region)
        badge_region = result[270:300, 30:200, :]
        assert np.any(badge_region != 0), "badge region has no drawn pixels"
        # The badge background is filled - should have many non-zero pixels
        assert np.count_nonzero(badge_region) > 50, "badge too sparse"

    def test_snapshot_badge_with_crossing_prob(self) -> None:
        overlay = BehaviorOverlay(offset_x=-20, offset_y=-10, font_scale=0.5, font_thickness=1)
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        state = _state(2, BehaviorLabel.HUMAN_CROSSING_ROAD, confidence=0.9, crossing_prob=0.8)
        ctx: dict[str, Any] = {
            "pose": PoseFrame(
                frame_id=0, timestamp=0.0,
                humans=(_human_pose(2),),
                animals=(),
            ),
            "behavior": BehaviorFrame(
                frame_id=0, timestamp=0.0,
                states=(state,),
            ),
        }
        result = overlay(canvas, ctx)
        badge_region = result[270:300, 30:200, :]
        assert np.any(badge_region != 0), "CP badge region has no drawn pixels"
        assert np.count_nonzero(badge_region) > 50, "CP badge too sparse"
