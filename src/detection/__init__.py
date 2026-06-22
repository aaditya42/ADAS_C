from __future__ import annotations

from src.detection.factory import build_detector
from src.detection.yolo_detector import YoloDetector

__all__ = [
    "build_detector",
    "YoloDetector",
]


