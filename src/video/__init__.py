from __future__ import annotations

from src.video.factory import build_frame_source
from src.video.video_file_source import VideoFileSource
from src.video.webcam_source import WebcamSource

__all__ = [
    "build_frame_source",
    "VideoFileSource",
    "WebcamSource",
]
