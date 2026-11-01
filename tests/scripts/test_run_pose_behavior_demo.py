from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def three_frame_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("fixture") / "test_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))
    for _ in range(3):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.putText(
            frame, "test", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def minimal_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("config") / "test_config.yaml"
    config_yaml = """\
depth:
  enabled: false

distance:
  type: relative

motion:
  enabled: false

ttc:
  type: simple

risk:
  type: score_based

warning:
  min_level: NONE
  cooldown_sec: 2.0
  hysteresis_frames: 3
  channels:
    - console

lane:
  enabled: false

signs:
  enabled: false

scene:
  enabled: false

pose:
  enabled: false

behavior:
  enabled: false

visualization:
  enabled: false
"""
    path.write_text(config_yaml)
    return path


class TestRunPoseBehaviorDemo:
    def test_default_mode_creates_output(
        self,
        three_frame_video: Path,
        minimal_config: Path,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "output.mp4"
        result = subprocess.run(
            [
                sys.executable,
                "-m", "scripts.run_pose_behavior_demo",
                "--input", str(three_frame_video),
                "--output", str(output_path),
                "--config", str(minimal_config),
                "--no-overlay",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_path.exists(), f"Output file not found: {output_path}"
        assert output_path.stat().st_size > 0, "Output file is empty"

    def test_pose_only_flag(
        self,
        three_frame_video: Path,
        minimal_config: Path,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "pose_only.mp4"
        result = subprocess.run(
            [
                sys.executable,
                "-m", "scripts.run_pose_behavior_demo",
                "--input", str(three_frame_video),
                "--output", str(output_path),
                "--config", str(minimal_config),
                "--pose-only",
                "--no-overlay",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_behavior_only_flag(
        self,
        three_frame_video: Path,
        minimal_config: Path,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "behavior_only.mp4"
        result = subprocess.run(
            [
                sys.executable,
                "-m", "scripts.run_pose_behavior_demo",
                "--input", str(three_frame_video),
                "--output", str(output_path),
                "--config", str(minimal_config),
                "--behavior-only",
                "--no-overlay",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_pose_only_without_no_overlay_graceful(
        self,
        three_frame_video: Path,
        minimal_config: Path,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "pose_no_overlay_flag.mp4"
        result = subprocess.run(
            [
                sys.executable,
                "-m", "scripts.run_pose_behavior_demo",
                "--input", str(three_frame_video),
                "--output", str(output_path),
                "--config", str(minimal_config),
                "--pose-only",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_nonexistent_input_reports_error(
        self,
        minimal_config: Path,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "nonexistent.mp4"
        result = subprocess.run(
            [
                sys.executable,
                "-m", "scripts.run_pose_behavior_demo",
                "--input", str(tmp_path / "no_such_video.mp4"),
                "--output", str(output_path),
                "--config", str(minimal_config),
                "--no-overlay",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "not found" in result.stdout.lower()
