from __future__ import annotations

from typing import Any

import numpy as np

from src.core.base import BaseDepthEstimator
from src.schemas import Frame
from src.utils.logger import get_logger


class DepthAnythingEstimator(BaseDepthEstimator):
    """Monocular depth estimator using Depth Anything V2 via HuggingFace.

    The model is loaded lazily on the first ``estimate()`` call.

    **Depth convention:** The returned depth map stores *relative* depth
    values where **LARGER = FARTHER** from the camera.  This is the
    opposite of the raw HF pipeline output for some checkpoints — the
    estimator inverts the signal if needed so downstream code always
    reads "bigger number = more distant".

    Args:
        model_name: HuggingFace model ID.
        device: Torch device string (``"cpu"``, ``"cuda:0"``, etc.).
        input_size: Inference resolution (shorter side) in pixels.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
        input_size: int = 518,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._input_size = input_size
        self._logger = get_logger(f"{__name__}.DepthAnythingEstimator")

        self._pipeline: Any = None
        self._cv2: Any = None
        self._Image: Any = None

    # ------------------------------------------------------------------ #
    #  BaseDepthEstimator
    # ------------------------------------------------------------------ #

    def estimate(self, frame: Frame) -> np.ndarray:
        if self._pipeline is None:
            self._load_pipeline()

        # BGR → RGB → PIL
        rgb = self._cv2.cvtColor(frame.image, self._cv2.COLOR_BGR2RGB)
        pil_image = self._Image.fromarray(rgb)

        raw = self._pipeline(pil_image)
        depth = np.array(raw["predicted_depth"], dtype=np.float32)

        # Resize back to original frame dimensions.
        orig_h, orig_w = frame.image.shape[:2]
        if depth.shape[:2] != (orig_h, orig_w):
            depth = self._cv2.resize(
                depth, (orig_w, orig_h), interpolation=self._cv2.INTER_LINEAR
            )

        # Invert so that LARGER = FARTHER
        depth = depth.max() - depth

        return depth

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_pipeline(self) -> None:
        import cv2
        from PIL import Image
        from transformers import pipeline

        self._cv2 = cv2
        self._Image = Image

        self._logger.info(
            "Loading depth pipeline %s ...  (device=%s)",
            self._model_name,
            self._device,
        )
        self._pipeline = pipeline(
            "depth-estimation",
            model=self._model_name,
            device=self._device,
        )
        self._logger.info(
            "Depth pipeline loaded  |  model=%s  |  device=%s  |  "
            "input_size=%d",
            self._model_name,
            self._device,
            self._input_size,
        )
