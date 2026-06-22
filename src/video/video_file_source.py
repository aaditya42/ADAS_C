from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.core.base import BaseFrameSource
from src.schemas import Frame
from src.utils.logger import get_logger


class VideoFileSource(BaseFrameSource):
    """Reads frames from a video file via OpenCV.

    Args:
        path: Path to the video file.
        loop: If True, restart the video when the stream ends.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """

    def __init__(self, path: str, loop: bool = False) -> None:
        self._loop = loop
        self._logger = get_logger(f"{__name__}.VideoFileSource")

        resolved = Path(path).resolve() if path else Path()
        if not path or not resolved.is_file():
            raise FileNotFoundError(
                f"Video file not found: {path!r}  (resolved: {resolved})"
            )
        self._path = str(resolved)

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self._path}")

        self._frame_id: int = 0
        self._logger.info(
            "Opened %s  [%d x %d, %.2f fps, %d frames]",
            self._path,
            self.width,
            self.height,
            self.fps,
            self.total_frames,
        )

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                ret, raw = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        self._frame_id += 1
        # OpenCV reads BGR natively — keep as-is.
        return Frame(
            image=raw,
            frame_id=self._frame_id,
            timestamp=time.time(),
            source=self._path,
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._logger.debug("Released %s", self._path)
