#!/usr/bin/env python3
"""Benchmark the ADAS pipeline on a Raspberry Pi or desktop.

Runs *N* frames through the pipeline with the given config and prints
per-stage latency, overall FPS, and peak memory.

Usage::

    python -m src.optimization.benchmark \\
        --config configs/raspberry_pi.yaml --frames 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config.loader import load_config
from src.core.pipeline import PerceptionPipeline
from src.evaluation.profiler import PipelineProfiler
from src.evaluation.system_metrics import SystemEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--frames", type=int, default=300,
        help="Number of frames to process",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[bench] Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    if config.get("video", {}).get("type") != "file":
        config["video"] = {
            "type": "file",
            "path": str(config_path.parent / "data" / "test.mp4"),
            "loop": True,
        }

    pipeline = PerceptionPipeline(config)

    syst = SystemEvaluator()

    try:
        pipeline.frame_source
    except Exception as e:
        source_cfg = config.get("video", {})
        if source_cfg.get("type") in ("picamera", "webcam"):
            print(f"[bench] Cannot open camera source for benchmarking. "
                  f"Use a file source instead.\n  Error: {e}")
            sys.exit(1)
        print(f"[bench] Pipeline init failed: {e}")
        sys.exit(1)

    print(f"[bench] Running {args.frames} frames through pipeline ...")

    perf_t0 = time.perf_counter()
    frame_count = 0

    profiler = PipelineProfiler(pipeline)

    with pipeline.frame_source as source:
        with profiler:
            for _ in range(args.frames):
                frame = source.read()
                if frame is None:
                    break

                t0 = time.perf_counter()
                result = pipeline.process(frame)
                elapsed = (time.perf_counter() - t0) * 1000.0
                syst.record(elapsed)
                frame_count += 1

    total_elapsed = time.perf_counter() - perf_t0

    print()
    print("=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Config:          {config_path}")
    print(f"  Frames:          {frame_count}")
    print(f"  Total time:      {total_elapsed:.2f}s")
    print(f"  Mean FPS:        {frame_count / total_elapsed:.1f}")
    print()

    sys_summary = syst.summary()
    print("  --- System ---")
    print(f"  Mean latency:    {sys_summary['mean_latency_ms']:.1f} ms")
    print(f"  P50 / P95 / P99: {sys_summary['p50_latency_ms']:.1f} / "
          f"{sys_summary['p95_latency_ms']:.1f} / "
          f"{sys_summary['p99_latency_ms']:.1f} ms")
    print(f"  Peak RSS:        {sys_summary['peak_rss_mb']:.1f} MB")
    print()

    prof_summary = profiler.summary()
    print(profiler.summary_table(prof_summary))


if __name__ == "__main__":
    main()
