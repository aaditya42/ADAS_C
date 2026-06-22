from __future__ import annotations

from pathlib import Path

import pytest

from src.video.factory import build_frame_source
from src.video.video_file_source import VideoFileSource
from src.video.webcam_source import WebcamSource


class TestVideoFileSource:
    def test_raises_on_bad_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            VideoFileSource(path="./nonexistent_video.mp4")

    def test_raises_on_empty_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            VideoFileSource(path="")

    def test_read_frames(self, test_video_path: Path) -> None:
        source = VideoFileSource(str(test_video_path))
        frames: list = []
        with source:
            while True:
                frame = source.read()
                if frame is None:
                    break
                frames.append(frame)
        assert len(frames) == 10


class TestFactory:
    def test_factory_returns_video_file_source(
        self, test_video_path: Path
    ) -> None:
        cfg = {
            "video": {
                "type": "file",
                "path": str(test_video_path),
                "loop": False,
            }
        }
        source = build_frame_source(cfg)
        assert isinstance(source, VideoFileSource)
        source.release()

    def test_factory_returns_webcam_source(self) -> None:
        cfg = {
            "video": {
                "type": "webcam",
                "webcam_device": 0,
                "width": 640,
                "height": 480,
            }
        }
        source = build_frame_source(cfg)
        assert isinstance(source, WebcamSource)
        source.release()

    def test_factory_raises_on_unknown_type(self) -> None:
        cfg = {"video": {"type": "rtsp"}}
        with pytest.raises(ValueError, match="rtsp"):
            build_frame_source(cfg)

    def test_factory_handles_missing_video_section(self) -> None:
        cfg: dict = {}
        with pytest.raises(FileNotFoundError):
            build_frame_source(cfg)
