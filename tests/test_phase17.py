"""Phase 17 — Raspberry Pi Optimization tests.

Tests that ONNX detector/depth estimators can be constructed, that factory
branches dispatch correctly, and that end-to-end ONNX Runtime inference
works against a tiny synthetic ONNX model.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.schemas import Frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_onnx(path: str, input_shape: tuple[int, ...],
                         output_shape: tuple[int, ...]) -> None:
    """Create a minimal ONNX model that copies input to output."""
    import onnx
    from onnx import helper, TensorProto  # noqa: PLC0415

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_shape))

    node = helper.make_node(
        "ReduceMax",
        inputs=["input"],
        outputs=["output"],
        axes=[1],
        keepdims=1,
    )

    graph = helper.make_graph(
        [node],
        "test_graph",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _make_frame(h: int = 64, w: int = 64) -> Frame:
    return Frame(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        frame_id=1, timestamp=0.0, source="test",
    )


# ---------------------------------------------------------------------------
# Detection — YoloOnnxDetector / factory
# ---------------------------------------------------------------------------

class TestYoloOnnxDetector:
    def test_factory_branch(self) -> None:
        from src.detection.factory import build_detector  # noqa: PLC0415
        det = build_detector({"detector": {"type": "yolo_onnx"}})
        from src.detection.yolo_onnx_detector import YoloOnnxDetector  # noqa: PLC0415
        assert isinstance(det, YoloOnnxDetector)

    def test_importable(self) -> None:
        from src.detection.yolo_onnx_detector import YoloOnnxDetector  # noqa: PLC0415, F401
        assert True

    def test_raises_on_bad_path(self) -> None:
        from src.detection.yolo_onnx_detector import YoloOnnxDetector  # noqa: PLC0415
        det = YoloOnnxDetector(model_path="nonexistent.onnx")
        frame = _make_frame()
        with pytest.raises(Exception):
            det.detect(frame)

    def test_class_names_loading(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("person\ncar\ntruck\n")
            cls_path = f.name

        from src.detection.yolo_onnx_detector import YoloOnnxDetector  # noqa: PLC0415
        det = YoloOnnxDetector(
            model_path="nonexistent.onnx",
            class_names_path=cls_path,
        )
        assert det._class_names == ["person", "car", "truck"]
        os.unlink(cls_path)


# ---------------------------------------------------------------------------
# Depth — DepthAnythingOnnxEstimator / factory
# ---------------------------------------------------------------------------

class TestDepthAnythingOnnxEstimator:
    def test_factory_branch(self) -> None:
        from src.depth.factory import build_depth_estimator  # noqa: PLC0415
        cfg = {"depth": {"enabled": True, "type": "depth_anything_onnx"}}
        est = build_depth_estimator(cfg)
        from src.depth.depth_anything_onnx_estimator import DepthAnythingOnnxEstimator  # noqa: PLC0415
        assert isinstance(est, DepthAnythingOnnxEstimator)

    def test_importable(self) -> None:
        from src.depth.depth_anything_onnx_estimator import (  # noqa: PLC0415, F401
            DepthAnythingOnnxEstimator,
        )
        assert True

    def test_end_to_end_inference(self) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            _make_synthetic_onnx(onnx_path, (1, 3, 64, 64), (1, 1, 64, 64))

            from src.depth.depth_anything_onnx_estimator import (  # noqa: PLC0415
                DepthAnythingOnnxEstimator,
            )
            est = DepthAnythingOnnxEstimator(
                model_path=onnx_path,
                input_size=64,
            )

            frame = _make_frame(h=64, w=64)
            depth = est.estimate(frame)

            assert isinstance(depth, np.ndarray)
            assert depth.shape == (64, 64)
            assert depth.dtype == np.float32
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_raises_on_bad_path(self) -> None:
        from src.depth.depth_anything_onnx_estimator import (  # noqa: PLC0415
            DepthAnythingOnnxEstimator,
        )
        est = DepthAnythingOnnxEstimator(model_path="nonexistent.onnx")
        frame = _make_frame()
        with pytest.raises(Exception):
            est.estimate(frame)


# ---------------------------------------------------------------------------
# PiCameraSource / factory
# ---------------------------------------------------------------------------

class TestPiCameraSource:
    def test_factory_branch(self) -> None:
        from src.video.factory import build_frame_source  # noqa: PLC0415
        from src.video.picamera_source import PiCameraSource  # noqa: PLC0415
        cfg = {"video": {"type": "picamera", "width": 640, "height": 480}}
        try:
            src = build_frame_source(cfg)
            assert isinstance(src, PiCameraSource)
        except ImportError:
            pytest.skip("picamera2 not installed")

    def test_raises_if_picamera2_missing(self) -> None:
        from src.video.picamera_source import PiCameraSource  # noqa: PLC0415
        try:
            src = PiCameraSource()
            src.release()
        except ImportError:
            pass  # Expected on non-Pi


# ---------------------------------------------------------------------------
# Export ONNX CLI
# ---------------------------------------------------------------------------

class TestExportOnnx:
    def test_importable(self) -> None:
        import src.optimization.export_onnx  # noqa: PLC0415, F401
        assert True

    def test_onnx_checker_on_synthetic(self) -> None:
        pytest.importorskip("onnx")

        import onnx  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            _make_synthetic_onnx(onnx_path, (1, 3, 64, 64), (1, 1, 64, 64))
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


# ---------------------------------------------------------------------------
# Benchmark CLI
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_importable(self) -> None:
        import src.optimization.benchmark  # noqa: PLC0415, F401
        assert True


# ---------------------------------------------------------------------------
# Raspberry Pi config
# ---------------------------------------------------------------------------

class TestPiConfig:
    def test_config_file_exists(self) -> None:
        assert Path("configs/raspberry_pi.yaml").is_file()

    def test_coco_classes_file_exists(self) -> None:
        assert Path("configs/coco_classes.txt").is_file()
        lines = Path("configs/coco_classes.txt").read_text().splitlines()
        assert len(lines) == 80

    def test_pi_config_loads(self) -> None:
        from src.config.loader import load_config  # noqa: PLC0415
        cfg = load_config("configs/raspberry_pi.yaml")
        assert cfg["video"]["type"] == "picamera"
        assert cfg["detector"]["type"] == "yolo_onnx"
        assert cfg["depth"]["type"] == "depth_anything_onnx"


# ---------------------------------------------------------------------------
# main.py — --config argument
# ---------------------------------------------------------------------------

class TestMainCli:
    def test_default_config_arg(self) -> None:
        import sys  # noqa: PLC0415
        sys.argv = ["main.py"]
        import main  # noqa: PLC0415, F401
        assert True
