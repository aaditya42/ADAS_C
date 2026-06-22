from __future__ import annotations

from typing import Any

from src.core.base import BaseFrameSource  # noqa: F401 — used only in return type
from src.video.video_file_source import VideoFileSource
from src.video.webcam_source import WebcamSource
from src.video.picamera_source import PiCameraSource


def build_frame_source(cfg: dict[str, Any]) -> BaseFrameSource:
    """Build a frame source from a configuration dictionary.

    The configuration must contain a ``video.type`` field set to ``"file"``,
    ``"webcam"``, or ``"picamera"``.  Additional key-value pairs are passed
    as keyword arguments to the respective source constructor.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        An initialised :class:`BaseFrameSource` instance.

    Raises:
        ValueError: If ``video.type`` is unknown or missing.
    """
    video_cfg = cfg.get("video", {})
    source_type = video_cfg.get("type", "file")

    kwargs = {k: v for k, v in video_cfg.items() if k != "type"}

    if source_type == "file":
        return VideoFileSource(
            path=kwargs.get("path", ""),
            loop=bool(kwargs.get("loop", False)),
        )
    if source_type == "webcam":
        return WebcamSource(
            device_id=int(kwargs.get("webcam_device", 0)),
            width=int(kwargs.get("width", 1280)),
            height=int(kwargs.get("height", 720)),
        )
    if source_type == "picamera":
        return PiCameraSource(
            width=int(kwargs.get("width", 640)),
            height=int(kwargs.get("height", 480)),
            framerate=int(kwargs.get("framerate", 30)),
        )

    raise ValueError(
        f"Unknown video source type: '{source_type}'. "
        f"Expected one of: 'file', 'webcam', 'picamera'."
    )
