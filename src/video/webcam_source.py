from __future__ import annotations

import time

import cv2

from src.core.base import BaseFrameSource
from src.schemas import Frame
from src.utils.logger import get_logger


class WebcamSource(BaseFrameSource):
    """Reads frames from a webcam via OpenCV.

    Args:
        device_id: Index of the camera device (0 = default).
        width: Desired capture width in pixels.
        height: Desired capture height in pixels.
    """

    def __init__(
        self, device_id: int = 0, width: int = 1280, height: int = 720
    ) -> None:
        self._device_id = device_id
        self._logger = get_logger(f"{__name__}.WebcamSource")

        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open webcam (device_id={device_id}). "
                "Make sure the camera is connected and not in use."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._frame_id: int = 0
        self._logger.info(
            "Opened webcam %d  [%d x %d, %.2f fps]",
            device_id,
            self.width,
            self.height,
            self.fps,
        )

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return -1  # live stream — unknown length

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------ #
    #  BaseFrameSource interface
    # ------------------------------------------------------------------ #

    def read(self) -> Frame | None:
        ret, raw = self._cap.read()
        if not ret:
            return None

        self._frame_id += 1
        return Frame(
            image=raw,
            frame_id=self._frame_id,
            timestamp=time.time(),
            source=f"webcam:{self._device_id}",
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._logger.debug("Released webcam %d", self._device_id)
