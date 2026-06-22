from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.config.loader import load_config
from src.core.pipeline import PerceptionPipeline
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="ADAS Perception System")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to configuration YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--video", default=None,
        help="Override video path (overrides config value)",
    )
    args = parser.parse_args()

    logger = get_logger("main")

    config_path = Path(args.config)
    config = load_config(config_path)

    if args.video:
        config["video"] = {"type": "file", "path": args.video, "loop": False}

    pipeline = PerceptionPipeline(config)
    logger.info("Pipeline ready (Phase 15)")

    source = pipeline.frame_source
    window_name = "Perception"

    frame_count = 0
    fps_timer = time.perf_counter()

    with source:
        while True:
            frame = source.read()
            if frame is None:
                break

            result = pipeline.process(frame)
            frame_count += 1

            # Compute FPS every frame for the FPS layer
            now = time.perf_counter()
            elapsed = now - fps_timer
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_timer = now
            result["_fps"] = fps

            if frame_count % 30 == 0:
                history = result["history"]
                active = {tid: len(history.get_history(tid)) for tid in history.active_tracks()}
                logger.debug("Active tracks: %s", active)

            if config.get("visualization", {}).get("display", True):
                display = pipeline.render(result)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("d"):
                    pipeline.renderer.toggle("depth")
                    logger.info("Toggled depth layer")
                if key == ord("l"):
                    pipeline.renderer.toggle("lane")
                    logger.info("Toggled lane layer")
                if key == ord("s"):
                    pipeline.renderer.toggle("sign")
                    logger.info("Toggled sign layer")
                if key == ord("r"):
                    pipeline.renderer.toggle("risk")
                    logger.info("Toggled risk layer")
                if key == ord("p"):
                    pipeline.renderer.toggle("perceived")
                    logger.info("Toggled perceived layer")
                if key == ord("h"):
                    pipeline.renderer.toggle("hud")
                    logger.info("Toggled HUD layer")
                if key == ord("f"):
                    pipeline.renderer.toggle("fps")
                    logger.info("Toggled FPS layer")

    cv2.destroyWindow(window_name)
    logger.info("Exited after processing %d frames", frame_count)


if __name__ == "__main__":
    main()
