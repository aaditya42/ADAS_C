from __future__ import annotations

from typing import Any, Optional

from src.core.base import BaseDetector
from src.detection.yolo_detector import YoloDetector
from src.detection.yolo_onnx_detector import YoloOnnxDetector


class _NoopDetector(BaseDetector):
    """Fallback detector that returns an empty list."""

    def detect(self, frame: Frame) -> list:  # noqa: F821
        return []


def build_detector(cfg: dict[str, Any]) -> BaseDetector:
    """Build a detector from the pipeline configuration.

    The configuration must contain a ``detector.type`` field. Currently
    supported: ``"yolo"``, ``"yolo_onnx"``.

    If no ``detector`` section exists in the config a no-op detector is
    returned so that the pipeline can still run safely.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        An initialised :class:`BaseDetector` instance.
    """
    det_cfg = cfg.get("detector")
    if det_cfg is None:
        return _NoopDetector()

    det_type = det_cfg.get("type", "yolo")

    if det_type == "yolo":
        class_filter: Optional[list[str]] = det_cfg.get("class_filter")
        return YoloDetector(
            model_path=str(det_cfg.get("model_path", "yolo11n.pt")),
            conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            device=str(det_cfg.get("device", "cpu")),
            class_filter=class_filter,
        )

    if det_type == "yolo_onnx":
        return YoloOnnxDetector(
            model_path=str(det_cfg.get("model_path", "yolo11n.onnx")),
            conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            providers=det_cfg.get("providers", ["CPUExecutionProvider"]),
            class_names_path=det_cfg.get("class_names_path"),
        )

    raise ValueError(
        f"Unknown detector type: '{det_type}'. "
        f"Expected one of: 'yolo', 'yolo_onnx'."
    )
