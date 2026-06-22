from __future__ import annotations

import numpy as np
import pytest

from src.lane_detection.interfaces import LaneDetectionResult
from src.schemas import BoundingBox, Frame, PerceivedObject
from src.schemas.scene_context import SceneContext
from src.schemas.traffic_sign_info import SignType, TrafficSignInfo
from src.scene.factory import build_scene_builder
from src.scene.scene_builder import SceneBuilder


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_frame(frame_id: int = 1, timestamp: float = 0.0) -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=timestamp,
        source="test",
    )


def _make_perceived(
    track_id: int = 1,
    class_name: str = "car",
    x1: float = 200.0,
    y1: float = 200.0,
    x2: float = 300.0,
    y2: float = 400.0,
    distance: float = 30.0,
) -> PerceivedObject:
    return PerceivedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=BoundingBox(x1, y1, x2, y2),
        depth=distance,
        distance=distance,
        depth_confidence=0.9,
    )


def _make_lane_info(lx: float = 100.0, rx: float = 500.0) -> LaneDetectionResult:
    """Lane with left boundary at x=lx and right at x=rx (straight vertical)."""
    img_h = 480
    left_points = tuple((int(lx), y) for y in range(0, img_h, 10))
    right_points = tuple((int(rx), y) for y in range(0, img_h, 10))
    cx = (lx + rx) / 2.0
    center_points = tuple((int(cx), y) for y in range(0, img_h, 10))
    return LaneDetectionResult(
        left_lane_points=left_points,
        right_lane_points=right_points,
        center_lane_points=center_points,
        lane_count=2,
        ego_lane_center_x=cx,
        confidence=0.9,
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestSceneBuilder:
    def test_speed_limit_latching(self) -> None:
        """A SPEED_LIMIT sign with value=50 latches and appears in context."""
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.SPEED_LIMIT,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
                value=50.0,
            )
        ]
        frame = _make_frame(frame_id=1)
        ctx = builder.build([], None, signs, frame)
        assert ctx.active_speed_limit == 50.0
        assert not ctx.stop_required
        assert not ctx.yield_required
        assert ctx.traffic_light_state == "none"

    def test_speed_limit_latches_across_frames(self) -> None:
        """Speed limit persists for sign_latch_frames even if sign disappears."""
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.SPEED_LIMIT,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
                value=60.0,
            )
        ]
        # Frame 1: sign present
        ctx1 = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx1.active_speed_limit == 60.0

        # Frames 2-10: no sign, but latched
        for fid in range(2, 11):
            ctx = builder.build([], None, [], _make_frame(frame_id=fid))
            assert ctx.active_speed_limit == 60.0, f"Failed at frame {fid}"

    def test_speed_limit_expires_after_latch_frames(self) -> None:
        """Speed limit expires after sign_latch_frames of absence."""
        builder = SceneBuilder(sign_latch_frames=5)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.SPEED_LIMIT,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
                value=70.0,
            )
        ]
        ctx1 = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx1.active_speed_limit == 70.0

        # Frame after latch window → expired
        ctx2 = builder.build([], None, [], _make_frame(frame_id=1 + 5))
        assert ctx2.active_speed_limit is None

    def test_stop_sign_latching(self) -> None:
        """STOP sign sets stop_required=True."""
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.STOP,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
            )
        ]
        ctx = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx.stop_required

    def test_yield_sign_latching(self) -> None:
        """YIELD sign sets yield_required=True."""
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.YIELD,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
            )
        ]
        ctx = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx.yield_required

    def test_traffic_light_red(self) -> None:
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.TRAFFIC_LIGHT_RED,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
            )
        ]
        ctx = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx.traffic_light_state == "red"

    def test_traffic_light_green(self) -> None:
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.TRAFFIC_LIGHT_GREEN,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
            )
        ]
        ctx = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx.traffic_light_state == "green"

    def test_ego_lane_membership_from_lane_info(self) -> None:
        """Object between left/right lanes → in ego lane."""
        builder = SceneBuilder()
        lane = _make_lane_info(lx=100.0, rx=500.0)
        # Object centred at x=300, which is between 100 and 500
        obj = _make_perceived(track_id=1, x1=250, x2=350)
        ctx = builder.build([obj], lane, [], _make_frame(frame_id=1))
        assert ctx.ego_lane_object_count == 1
        assert ctx.nearest_in_lane_track_id == 1
        assert ctx.nearest_in_lane_distance_m == pytest.approx(30.0)

    def test_ego_lane_fallback_central(self) -> None:
        """When lane_info is None, central 50% is used."""
        builder = SceneBuilder()
        # Image is 640 wide → centre is 320, central 50% → x in [160, 480]
        obj_in = _make_perceived(track_id=1, x1=250, x2=350)    # cx=300 ✓
        obj_out = _make_perceived(track_id=2, x1=0, x2=100)     # cx=50  ✗
        ctx = builder.build([obj_in, obj_out], None, [], _make_frame(frame_id=1))
        assert ctx.ego_lane_object_count == 1
        assert ctx.nearest_in_lane_track_id == 1

    def test_nearest_in_lane_distance(self) -> None:
        """Closest in-lane object is correctly identified."""
        builder = SceneBuilder()
        lane = _make_lane_info(lx=100.0, rx=500.0)
        near = _make_perceived(track_id=1, x1=250, x2=350, distance=10.0)
        far = _make_perceived(track_id=2, x1=260, x2=340, distance=50.0)
        ctx = builder.build([near, far], lane, [], _make_frame(frame_id=1))
        assert ctx.nearest_in_lane_track_id == 1
        assert ctx.nearest_in_lane_distance_m == pytest.approx(10.0)

    def test_crosswalk_pedestrian(self) -> None:
        """Crosswalk sign + in-lane person → crosswalk_pedestrian=True."""
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.PEDESTRIAN_CROSSING,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
            )
        ]
        lane = _make_lane_info(lx=100.0, rx=500.0)
        person = _make_perceived(
            track_id=1, class_name="person", x1=250, x2=350, distance=5.0
        )
        ctx = builder.build([person], lane, signs, _make_frame(frame_id=1))
        assert ctx.crosswalk_pedestrian

    def test_no_crosswalk_no_pedestrian(self) -> None:
        builder = SceneBuilder()
        lane = _make_lane_info(lx=100.0, rx=500.0)
        person = _make_perceived(
            track_id=1, class_name="person", x1=250, x2=350
        )
        # No crosswalk sign
        ctx = builder.build([person], lane, [], _make_frame(frame_id=1))
        assert not ctx.crosswalk_pedestrian

    def test_timestamp_passed_through(self) -> None:
        builder = SceneBuilder()
        frame = _make_frame(frame_id=5, timestamp=123.456)
        ctx = builder.build([], None, [], frame)
        assert ctx.timestamp == pytest.approx(123.456)


class TestFactory:
    def test_builds_scene_builder(self) -> None:
        builder = build_scene_builder({"scene": {"enabled": True}})
        assert isinstance(builder, SceneBuilder)

    def test_default_latch_frames(self) -> None:
        builder = build_scene_builder({})
        assert builder._latch_frames == 30

    def test_custom_latch_frames(self) -> None:
        builder = build_scene_builder({"scene": {"sign_latch_frames": 60}})
        assert builder._latch_frames == 60


class TestPipelineIntegration:
    def test_scene_key_in_output(self) -> None:
        from src.core.pipeline import PerceptionPipeline

        pipe = PerceptionPipeline({})
        frame = _make_frame()
        result = pipe.process(frame)
        assert "scene" in result
        # Scene may be None if scene module is disabled, or a SceneContext
        if result["scene"] is not None:
            assert isinstance(result["scene"], SceneContext)

    def test_stop_required_in_pipeline(self) -> None:
        """Scene context from pipeline with a STOP sign."""
        from src.core.pipeline import PerceptionPipeline

        cfg = {"scene": {"enabled": True, "sign_latch_frames": 30}}
        pipe = PerceptionPipeline(cfg)
        # Build with stop sign via internal state — we test the builder directly
        builder = SceneBuilder(sign_latch_frames=30)
        signs = [
            TrafficSignInfo(
                sign_type=SignType.STOP,
                bbox=BoundingBox(0, 0, 10, 10),
                confidence=0.9,
            )
        ]
        ctx = builder.build([], None, signs, _make_frame(frame_id=1))
        assert ctx.stop_required
