from __future__ import annotations

import numpy as np
import pytest

from src.lane_detection.interfaces import LaneDetectionResult
from src.lane_detection.lane_detector_factory import LaneDetectorFactory


class TestPipelineIntegration:
    def test_pipeline_returns_lane_key(self) -> None:
        from src.core.pipeline import PerceptionPipeline

        pipe = PerceptionPipeline({})
        frame = type(
            "Frame", (), {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "frame_id": 1, "timestamp": 0.0, "source": "test",
            }
        )()
        result = pipe.process(frame)
        assert "lane" in result

    def test_lane_output_is_lane_detection_result(self) -> None:
        from src.core.pipeline import PerceptionPipeline

        # Empty config → lane enabled by default → detector created
        # but missing model file → detector returns LaneDetectionResult.empty()
        pipe = PerceptionPipeline({})
        frame = type(
            "Frame", (), {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "frame_id": 1, "timestamp": 0.0, "source": "test",
            }
        )()
        result = pipe.process(frame)
        lane = result.get("lane")
        assert isinstance(lane, LaneDetectionResult)
        assert lane.lane_count == 0
        assert lane.confidence == 0.0

    def test_enabled_config_creates_detector(self) -> None:
        # Lane enabled with config — detector is created
        cfg = {"lane": {"enabled": True, "model_type": "clrnet"}}
        det = LaneDetectorFactory.create(cfg)
        assert det is not None
        from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector
        assert isinstance(det, CLRNetLaneDetector)

    def test_visualization_consumes_lane_result(self) -> None:
        """Verify that LaneLayer.render() handles LaneDetectionResult."""
        from src.visualization.layers.lane_layer import LaneLayer

        layer = LaneLayer(enabled=True)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # New schema result
        lane_result = LaneDetectionResult(
            left_lane_points=((100, 400), (120, 300), (140, 200)),
            right_lane_points=((500, 400), (480, 300), (460, 200)),
            center_lane_points=((300, 350), (300, 250)),
            lane_count=2,
            ego_lane_center_x=300.0,
            confidence=0.9,
        )
        ctx = {"lane": lane_result}
        output = layer(canvas, ctx)
        assert output.shape == (480, 640, 3)
        assert output.dtype == np.uint8

    def test_visualization_skips_low_confidence(self) -> None:
        from src.visualization.layers.lane_layer import LaneLayer

        layer = LaneLayer(enabled=True)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        lane_result = LaneDetectionResult.empty()
        ctx = {"lane": lane_result}
        output = layer(canvas, ctx)
        np.testing.assert_array_equal(output, canvas)

    def test_visualization_without_lane_key(self) -> None:
        from src.visualization.layers.lane_layer import LaneLayer

        layer = LaneLayer(enabled=True)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        output = layer(canvas, {})
        np.testing.assert_array_equal(output, canvas)

    def test_visualization_disabled_layer(self) -> None:
        from src.visualization.layers.lane_layer import LaneLayer

        layer = LaneLayer(enabled=False)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        lane_result = LaneDetectionResult(
            left_lane_points=((100, 400),),
            right_lane_points=((500, 400),),
            center_lane_points=((300, 400),),
            lane_count=2,
            ego_lane_center_x=300.0,
            confidence=0.9,
        )
        output = layer(canvas, {"lane": lane_result})
        np.testing.assert_array_equal(output, canvas)

    def test_renderer_with_lane_layer(self) -> None:
        """Renderer integration with LaneDetectionResult."""
        from src.visualization.renderer import Renderer
        from src.visualization.layers.lane_layer import LaneLayer

        renderer = Renderer([LaneLayer(enabled=True)])
        output = renderer.render({
            "frame": type(
                "Frame", (), {
                    "image": np.zeros((480, 640, 3), dtype=np.uint8),
                    "frame_id": 1, "timestamp": 0.0, "source": "test",
                }
            )(),
            "lane": LaneDetectionResult(
                left_lane_points=((100, 400), (120, 300)),
                right_lane_points=((500, 400), (480, 300)),
                center_lane_points=((300, 350),),
                lane_count=2,
                ego_lane_center_x=300.0,
                confidence=0.9,
            ),
            "detections": [], "tracked": [], "depth_map": None,
            "perceived": [], "history": None, "motion": {}, "ttc": {},
            "risk": [], "alerts": [], "signs": [], "scene": None,
            "_fps": 30.0,
        })
        assert isinstance(output, np.ndarray)
        assert output.shape == (480, 640, 3)
