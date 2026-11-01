#!/usr/bin/env python3
"""End-to-end pose + behaviour classification demo.

Processes a video through the full pipeline and produces an annotated
output video with pose keypoints, skeleton edges, behaviour label
badges, and risk overlays.

Usage::

    # Full pipeline with pose + behaviour + risk overlays
    python scripts/run_pose_behavior_demo.py \
        --input data/video.mp4 \
        --output data/output.mp4 \
        --config configs/default.yaml

    # Pose only (behaviour disabled)
    python scripts/run_pose_behavior_demo.py \
        --input data/video.mp4 \
        --output data/output.mp4 \
        --pose-only

    # Behaviour only (pose disabled)
    python scripts/run_pose_behavior_demo.py \
        --input data/video.mp4 \
        --output data/output.mp4 \
        --behavior-only

    # Raw output without visual overlays
    python scripts/run_pose_behavior_demo.py \
        --input data/video.mp4 \
        --output data/output.mp4 \
        --no-overlay

Accepts standard video formats supported by OpenCV (MP4, AVI, MOV, etc.).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from src.config.loader import load_config
from src.core.pipeline import PerceptionPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose + Behaviour Classification Demo",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output video file",
    )
    parser.add_argument(
        "--config", "-c", default="configs/default.yaml",
        help="Path to configuration YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--pose-only", action="store_true",
        help="Enable only pose estimation (behaviour disabled)",
    )
    parser.add_argument(
        "--behavior-only", "--behaviour-only", action="store_true",
        dest="behavior_only",
        help="Enable only behaviour classification (pose disabled)",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable all visual overlays (raw output only)",
    )
    return parser.parse_args()


def _apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = _deep_merge(config, {"video": {"type": "file", "path": args.input, "loop": False}})

    if args.pose_only:
        config["behavior"] = _deep_merge(
            config.get("behavior", {}), {"enabled": False},
        )
        config["pose"] = _deep_merge(
            config.get("pose", {}), {"enabled": True},
        )
        vis = config.get("visualization", {})
        vis["pose_overlay"] = _deep_merge(
            vis.get("pose_overlay", {}), {"enabled": True},
        )
        vis["behavior_overlay"] = _deep_merge(
            vis.get("behavior_overlay", {}), {"enabled": False},
        )
        config["visualization"] = vis

    if args.behavior_only:
        config["pose"] = _deep_merge(
            config.get("pose", {}), {"enabled": False},
        )
        config["behavior"] = _deep_merge(
            config.get("behavior", {}), {"enabled": True},
        )
        vis = config.get("visualization", {})
        vis["pose_overlay"] = _deep_merge(
            vis.get("pose_overlay", {}), {"enabled": False},
        )
        vis["behavior_overlay"] = _deep_merge(
            vis.get("behavior_overlay", {}), {"enabled": True},
        )
        config["visualization"] = vis

    if args.no_overlay:
        config["visualization"] = _deep_merge(
            config.get("visualization", {}), {"enabled": False},
        )

    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[demo] Config not found: {config_path}")
        return

    config = load_config(config_path)
    config = _apply_cli_overrides(config, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.input).exists():
        print(f"[demo] Input video not found: {args.input}")
        return

    pipeline = PerceptionPipeline(config)

    source = pipeline.frame_source
    width = source.width
    height = source.height
    fps = source.fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames: int = 0
    start_time = time.perf_counter()
    print(f"[demo] Processing {args.input} -> {args.output}")

    with source:
        while True:
            frame = source.read()
            if frame is None:
                break

            result = pipeline.process(frame)
            total_frames += 1

            overlay = args.no_overlay
            if not overlay:
                display = pipeline.render(result)
            else:
                display = frame.image.copy()

            writer.write(display)

            if total_frames % 30 == 0:
                elapsed = time.perf_counter() - start_time
                fps_cur = total_frames / elapsed if elapsed > 0 else 0.0
                pose_frame = result.get("pose")
                behavior_frame = result.get("behavior")
                n_humans = len(pose_frame.humans) if pose_frame else 0
                n_animals = len(pose_frame.animals) if pose_frame else 0
                n_states = len(behavior_frame.states) if behavior_frame else 0
                print(
                    f"[demo] Frame {total_frames} | "
                    f"{fps_cur:.1f} FPS | "
                    f"humans={n_humans} animals={n_animals} "
                    f"behaviour_states={n_states}"
                )

    elapsed = time.perf_counter() - start_time
    writer.release()
    print(f"[demo] Done — {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames / elapsed:.1f} FPS)")
    print(f"[demo] Output saved to {output_path}")


if __name__ == "__main__":
    main()
