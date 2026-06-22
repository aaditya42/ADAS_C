from __future__ import annotations

from typing import Any

import numpy as np

from src.core.base import BaseDepthEstimator
from src.schemas import Frame
from src.utils.logger import get_logger


class DepthAnythingOnnxEstimator(BaseDepthEstimator):
    """Depth Anything V2 running via ONNX Runtime.

    Same interface and depth convention as ``DepthAnythingEstimator``
    (LARGER = FARTHER) but loads from an exported ``.onnx`` file
    instead of a HuggingFace ``transformers`` pipeline.

    Args:
        model_path: Path to the exported ``.onnx`` model file.
        input_size: Inference resolution (shorter side) in pixels.
        providers: ONNX Runtime execution providers.
    """

    def __init__(
        self,
        model_path: str = "depth_anything_v2_small.onnx",
        input_size: int = 224,
        providers: list[str] | None = None,
    ) -> None:
        self._model_path = model_path
        self._input_size = input_size
        self._providers = providers or ["CPUExecutionProvider"]
        self._logger = get_logger(f"{__name__}.DepthAnythingOnnxEstimator")

        self._session: Any = None
        self._input_name: str = ""
        self._output_name: str = ""

    def estimate(self, frame: Frame) -> np.ndarray:
        if self._session is None:
            self._load_session()

        import cv2  # noqa: PLC0415

        orig_h, orig_w = frame.image.shape[:2]

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        scale = self._input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        blob = (blob - mean) / std
        blob = np.expand_dims(blob, axis=0)

        ort_inputs = {self._input_name: blob}
        raw = self._session.run([self._output_name], ort_inputs)[0]

        depth = np.squeeze(raw)
        if depth.shape[:2] != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)

        depth = depth.astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)

        depth = depth.max() - depth

        return depth

    def _load_session(self) -> None:
        import onnxruntime  # noqa: PLC0415

        self._logger.info("Loading ONNX depth model from %s ...", self._model_path)
        self._session = onnxruntime.InferenceSession(
            self._model_path, providers=self._providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        self._logger.info("ONNX depth session ready  |  input=%s  |  providers=%s",
                          input_shape, self._providers)
