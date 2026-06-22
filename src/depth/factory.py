from __future__ import annotations

from typing import Any

from src.core.base import BaseDepthEstimator
from src.depth.depth_anything_estimator import DepthAnythingEstimator
from src.depth.depth_anything_onnx_estimator import DepthAnythingOnnxEstimator


class _NoopDepthEstimator(BaseDepthEstimator):
    """Fallback depth estimator that returns a zero depth map."""

    def estimate(self, frame: Frame) -> Any:  # noqa: F821
        import numpy as np

        return np.zeros(
            (frame.image.shape[0], frame.image.shape[1]), dtype=np.float32
        )


def build_depth_estimator(cfg: dict[str, Any]) -> BaseDepthEstimator:
    """Build a depth estimator from the pipeline configuration.

    The configuration must contain a ``depth.type`` field. Currently
    supported: ``"depth_anything_v2"``, ``"depth_anything_onnx"``.

    If no ``depth`` section exists a no-op estimator is returned.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        An initialised :class:`BaseDepthEstimator` instance.
    """
    depth_cfg = cfg.get("depth")
    if depth_cfg is None or not depth_cfg.get("enabled", False):
        return _NoopDepthEstimator()

    depth_type = depth_cfg.get("type", "depth_anything_v2")

    if depth_type == "depth_anything_v2":
        return DepthAnythingEstimator(
            model_name=str(
                depth_cfg.get(
                    "model_name",
                    "depth-anything/Depth-Anything-V2-Small-hf",
                )
            ),
            device=str(depth_cfg.get("device", "cpu")),
            input_size=int(depth_cfg.get("input_size", 518)),
        )

    if depth_type == "depth_anything_onnx":
        return DepthAnythingOnnxEstimator(
            model_path=str(
                depth_cfg.get("model_path", "weights/depth_anything_v2_small.onnx")
            ),
            input_size=int(depth_cfg.get("input_size", 224)),
            providers=depth_cfg.get("providers", ["CPUExecutionProvider"]),
        )

    raise ValueError(
        f"Unknown depth estimator type: '{depth_type}'. "
        f"Expected one of: 'depth_anything_v2', 'depth_anything_onnx'."
    )
