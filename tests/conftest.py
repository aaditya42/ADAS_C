from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny valid MP4 file for use in tests."""
    path = tmp_path_factory.mktemp("video") / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for _ in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
