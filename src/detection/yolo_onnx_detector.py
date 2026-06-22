from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.base import BaseDetector
from src.schemas import BoundingBox, Detection, Frame
from src.utils.logger import get_logger


class YoloOnnxDetector(BaseDetector):
    """YOLO object detector running via ONNX Runtime.

    Same interface as ``YoloDetector`` but loads from an exported ``.onnx``
    file instead of a PyTorch weights file.  Preprocessing uses letterbox
    resizing; postprocessing uses ``cv2.dnn.NMSBoxes``.

    Args:
        model_path: Path to the ``.onnx`` model file.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: NMS IoU threshold.
        providers: ONNX Runtime execution providers
            (e.g. ``["CPUExecutionProvider"]``,
            ``["OpenVINOExecutionProvider"]``).
        class_names_path: Path to a text file with one class name per line.
            If ``None``, numeric class indices are used as names.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.onnx",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: Optional[list[str]] = None,
        class_names_path: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._providers = providers or ["CPUExecutionProvider"]
        self._class_names_path = class_names_path
        self._logger = get_logger(f"{__name__}.YoloOnnxDetector")

        self._session: Any = None
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._class_names: list[str] = []
        if self._class_names_path:
            path = Path(self._class_names_path)
            if path.is_file():
                self._class_names = [
                    line.strip() for line in path.read_text().splitlines()
                    if line.strip()
                ]
                self._logger.info("Loaded %d class names from %s",
                                  len(self._class_names), path)
            else:
                self._logger.warning("Class names file not found: %s", path)

    def detect(self, frame: Frame) -> list[Detection]:
        img_h, img_w = frame.image.shape[:2]

        if self._session is None:
            self._load_session()

        preprocessed, scale, pad_left, pad_top = self._preprocess(frame.image)

        ort_inputs = {self._input_name: preprocessed}
        raw = self._session.run(self._output_names, ort_inputs)[0]

        detections = self._postprocess(raw, img_w, img_h, scale, pad_left, pad_top)
        return detections

    def _load_session(self) -> None:
        import onnxruntime  # noqa: PLC0415

        self._logger.info("Loading ONNX model from %s ...", self._model_path)
        self._session = onnxruntime.InferenceSession(
            self._model_path, providers=self._providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        input_shape = self._session.get_inputs()[0].shape
        self._logger.info("ONNX session ready  |  input=%s  |  providers=%s",
                          input_shape, self._providers)

    def _preprocess(
        self, image: np.ndarray,
    ) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]

        input_w = 640
        input_h = 640

        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        import cv2  # noqa: PLC0415
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_left = (input_w - new_w) // 2
        pad_top = (input_h - new_h) // 2

        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, pad_left, pad_top

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_left: int,
        pad_top: int,
    ) -> list[Detection]:
        import cv2  # noqa: PLC0415

        raw = np.squeeze(raw)
        if raw.ndim == 2:
            raw = raw.T

        boxes: list[tuple[int, int, int, int]] = []
        confs: list[float] = []
        class_ids: list[int] = []

        for pred in raw:
            if len(pred) < 6:
                continue
            scores = pred[4:]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])

            if score < self._conf:
                continue

            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x1 = int((cx - bw / 2 - pad_left) / scale)
            y1 = int((cy - bh / 2 - pad_top) / scale)
            x2 = int((cx + bw / 2 - pad_left) / scale)
            y2 = int((cy + bh / 2 - pad_top) / scale)

            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2, y2))
            confs.append(score)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confs, self._conf, self._iou)
        if len(indices) == 0:
            return []

        if indices.ndim > 1:
            indices = indices.flatten()

        detections: list[Detection] = []
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            cls_id = class_ids[idx]
            cls_name: str = (
                self._class_names[cls_id]
                if cls_id < len(self._class_names)
                else str(cls_id)
            )

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=confs[idx],
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        return detections
