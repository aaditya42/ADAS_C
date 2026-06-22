from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.core.base import BaseDetector
from src.schemas import BoundingBox, Detection, Frame
from src.utils.logger import get_logger


class YoloDetector(BaseDetector):
    """Object detector powered by Ultralytics YOLO (v11).

    The model is loaded lazily on the first call to ``detect()`` so that
    importing this module does not require ``ultralytics`` to be installed
    unless the detector is actually used.

    Args:
        model_path: Path or name of the YOLO weights file
            (e.g. ``"yolo11n.pt"``, ``"yolo11s.pt"``).
        conf_threshold: Confidence threshold for detections.
        iou_threshold: NMS IoU threshold.
        device: Inference device (``"cpu"``, ``"cuda:0"``, ``"mps"``, etc.).
        class_filter: If given, only return detections whose class name is in
            this list. ``None`` means return all classes.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        class_filter: Optional[list[str]] = None,
    ) -> None:
        self._model_path = model_path
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._device = device
        self._class_filter: Optional[set[str]] = (
            set(class_filter) if class_filter is not None else None
        )
        self._logger = get_logger(f"{__name__}.YoloDetector")

        self._model: Any = None
        self._names: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    #  BaseDetector
    # ------------------------------------------------------------------ #

    def detect(self, frame: Frame) -> list[Detection]:
        if self._model is None:
            self._load_model()

        results = self._model.predict(
            frame.image,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        detections: list[Detection] = []
        cls_ids: list[int] = boxes.cls.int().tolist()
        confs: list[float] = boxes.conf.float().tolist()
        xyxy: list[list[float]] = boxes.xyxy.float().tolist()

        for cls_id, conf, (x1, y1, x2, y2) in zip(cls_ids, confs, xyxy):
            class_name = self._names.get(cls_id, str(cls_id))

            if self._class_filter is not None and class_name not in self._class_filter:
                continue

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )

        return detections

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        from ultralytics import YOLO  # noqa: PLC0415 — isolated import

        self._logger.info("Loading YOLO model from %s ...", self._model_path)
        self._model = YOLO(self._model_path)
        self._names = self._model.names  # dict[int, str]
        self._logger.info(
            "YOLO loaded  |  device=%s  |  %d classes",
            self._device,
            len(self._names),
        )
