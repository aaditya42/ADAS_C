from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.core.base import BaseFrameSource
from src.schemas import Frame
from src.utils.logger import get_logger


class PiCameraSource(BaseFrameSource):
    """Frame source for Raspberry Pi Camera Module v2 / HQ via ``picamera2``.

    If ``picamera2`` is not installed the constructor raises ``ImportError``
    with an installation hint.

    Args:
        width: Capture width in pixels.
        height: Capture height in pixels.
        framerate: Target framerate.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
    ) -> None:
        self._width = width
        self._height = height
        self._framerate = framerate
        self._logger = get_logger(f"{__name__}.PiCameraSource")

        self._camera: Any = None
        self._frame_id: int = 0

    @property
    def fps(self) -> float:
        return float(self._framerate)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def read(self) -> Frame | None:
        if self._camera is None:
            self._open_camera()

        raw = self._camera.capture_array()
        if raw is None:
            return None

        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]

        if raw.ndim == 3 and raw.shape[2] == 3:
            import cv2  # noqa: PLC0415
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            bgr = raw

        self._frame_id += 1
        return Frame(
            image=bgr,
            frame_id=self._frame_id,
            timestamp=time.time(),
            source="picamera",
        )

    def release(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception:
                pass
            self._camera = None
            self._logger.debug("PiCamera released")

    def _open_camera(self) -> None:
        try:
            from picamera2 import Picamera2  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "picamera2 is required for PiCameraSource. "
                "Install it with: pip install picamera2"
            )

        self._logger.info(
            "Opening PiCamera  |  %d x %d  |  %d fps",
            self._width, self._height, self._framerate,
        )
        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (self._width, self._height)},
        )
        self._camera.configure(config)
        self._camera.set_controls({"FrameRate": self._framerate})
        self._camera.start()
        self._logger.info("PiCamera started")
