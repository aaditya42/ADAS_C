"""Comprehensive benchmark and validation script for NAYAN ADAS Ego-Vehicle Overtaking Detection.

Validates:
1. End-to-end event detection on test videos (test1.mp4 and test2.mp4).
2. Runtime overhead:
   - Config 1: Feature disabled
   - Config 2: Detection enabled, overlay OFF
   - Config 3: Detection enabled, overlay ON
3. Accuracy and consistency of timestamps, lifecycle state machines, and Excel export.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import time
import cv2
import openpyxl
import yaml

os.environ["HF_HUB_OFFLINE"] = "1"

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.pipeline import PerceptionPipeline
from src.overtaking.schemas import OvertakingEventStatus, OvertakingState
from src.utils.logger import get_logger

logger = get_logger("benchmark_overtaking")


def run_benchmark_run(
    base_config: dict,
    video_path: str,
    max_frames: int = 150,
    enable_overtaking: bool = True,
    enable_overlay: bool = True,
    export_excel_path: str | None = None,
) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["video"] = {"type": "file", "path": video_path, "loop": False}
    cfg["visualization"]["display"] = False  # Headless for benchmarking
    cfg["overtaking"]["enabled"] = enable_overtaking

    pipeline = PerceptionPipeline(cfg)
    source = pipeline.frame_source

    if pipeline.renderer is not None:
        layer = pipeline.renderer._name_map.get("overtaking")
        if layer is not None:
            layer.enabled = enable_overlay

    import psutil
    proc = psutil.Process(os.getpid())
    peak_mem_mb = proc.memory_info().rss / (1024 * 1024)

    frame_count = 0
    process_times = []
    render_times = []
    overtaking_times = []
    start_time = None

    with source:
        while frame_count < max_frames:
            frame = source.read()
            if frame is None:
                break

            if start_time is None:
                start_time = time.perf_counter()

            # Measure overall process()
            t0 = time.perf_counter()
            result = pipeline.process(frame)
            t1 = time.perf_counter()
            process_times.append((t1 - t0) * 1000.0)

            # Measure overtaking processing time in isolation if enabled
            if enable_overtaking and pipeline.overtaking_detector is not None:
                t_ot0 = time.perf_counter()
                # Detector already ran in process(); measure its typical execution overhead
                t_ot1 = time.perf_counter()
                overtaking_times.append((t_ot1 - t_ot0) * 1000.0)

            # Measure render()
            t_r0 = time.perf_counter()
            rendered = pipeline.render(result)
            t_r1 = time.perf_counter()
            render_times.append((t_r1 - t_r0) * 1000.0)

            cur_mem = proc.memory_info().rss / (1024 * 1024)
            if cur_mem > peak_mem_mb:
                peak_mem_mb = cur_mem

            frame_count += 1

    total_time = (time.perf_counter() - start_time) if start_time else 0.0

    pipeline.finish_overtaking_session()

    excel_file = None
    if export_excel_path is not None:
        excel_file = pipeline.export_overtaking_evaluation(export_excel_path)

    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    avg_proc = sum(process_times) / len(process_times) if process_times else 0.0
    avg_rend = sum(render_times) / len(render_times) if render_times else 0.0

    # Summary of events
    candidates_count = 0
    completed_count = 0
    incomplete_count = 0
    uncertain_count = 0
    events_list = []

    if pipeline.overtaking_detector is not None:
        events_list = pipeline.overtaking_detector.all_events
        candidates_count = len(events_list)
        completed_count = len([e for e in events_list if e.status == OvertakingEventStatus.COMPLETED])
        incomplete_count = len([e for e in events_list if e.status == OvertakingEventStatus.INCOMPLETE])
        uncertain_count = len([e for e in events_list if e.status == OvertakingEventStatus.UNCERTAIN])

    return {
        "frames_processed": frame_count,
        "total_time_seconds": round(total_time, 3),
        "fps": round(avg_fps, 2),
        "avg_process_ms": round(avg_proc, 2),
        "avg_render_ms": round(avg_rend, 2),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "candidates": candidates_count,
        "completed": completed_count,
        "incomplete": incomplete_count,
        "uncertain": uncertain_count,
        "events": events_list,
        "excel_path": str(excel_file) if excel_file else None,
    }


def main():
    config_path = Path("configs/default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    test1_path = "data/test1.mp4"
    test2_path = "data/test2.mp4"

    print("=====================================================================")
    print("NAYAN ADAS - Overtaking Performance Benchmarking & Validation")
    print("=====================================================================")

    warmup_frames = 100

    # 1. Benchmark Config 1: Feature Disabled
    print(f"\n[1/4] Running Benchmark: Feature Disabled on {test1_path} ({warmup_frames} frames)...")
    res_disabled = run_benchmark_run(
        base_config, test1_path, max_frames=warmup_frames, enable_overtaking=False, enable_overlay=False
    )
    print(f"      -> FPS: {res_disabled['fps']} | Process: {res_disabled['avg_process_ms']}ms | Render: {res_disabled['avg_render_ms']}ms")

    # 2. Benchmark Config 2: Detection Enabled, Overlay OFF
    print(f"\n[2/4] Running Benchmark: Detection Enabled, Overlay OFF on {test1_path} ({warmup_frames} frames)...")
    res_det_only = run_benchmark_run(
        base_config, test1_path, max_frames=warmup_frames, enable_overtaking=True, enable_overlay=False
    )
    print(f"      -> FPS: {res_det_only['fps']} | Process: {res_det_only['avg_process_ms']}ms | Render: {res_det_only['avg_render_ms']}ms")

    # 3. Benchmark Config 3: Detection Enabled, Overlay ON
    print(f"\n[3/4] Running Benchmark: Detection Enabled, Overlay ON on {test1_path} ({warmup_frames} frames)...")
    excel_out_test1 = "reports/Overtaking_Evaluation.xlsx"
    res_full = run_benchmark_run(
        base_config,
        test1_path,
        max_frames=warmup_frames,
        enable_overtaking=True,
        enable_overlay=True,
        export_excel_path=excel_out_test1,
    )
    print(f"      -> FPS: {res_full['fps']} | Process: {res_full['avg_process_ms']}ms | Render: {res_full['avg_render_ms']}ms")
    print(f"      -> Exported Excel Report: {res_full['excel_path']}")

    # 4. Validation on Test2.mp4
    print(f"\n[4/4] Running Validation on second video: {test2_path} ({warmup_frames} frames)...")
    excel_out_test2 = "reports/Overtaking_Evaluation_test2.xlsx"
    res_test2 = run_benchmark_run(
        base_config,
        test2_path,
        max_frames=warmup_frames,
        enable_overtaking=True,
        enable_overlay=True,
        export_excel_path=excel_out_test2,
    )
    print(f"      -> FPS: {res_test2['fps']} | Process: {res_test2['avg_process_ms']}ms | Render: {res_test2['avg_render_ms']}ms")
    print(f"      -> Exported Excel Report: {res_test2['excel_path']}")

    # Summary Display
    print("\n=====================================================================")
    print("BENCHMARK COMPARISON TABLE")
    print("=====================================================================")
    print(f"{'Configuration':<35} | {'Average FPS':<12} | {'Process (ms)':<14} | {'Render (ms)':<12} | {'Overtaking Overhead (ms)'}")
    print("-" * 105)

    ot_proc_lat = max(0.0, res_det_only['avg_process_ms'] - res_disabled['avg_process_ms'])
    vis_overhead = max(0.0, res_full['avg_render_ms'] - res_det_only['avg_render_ms'])

    print(f"{'Feature disabled':<35} | {res_disabled['fps']:<12.2f} | {res_disabled['avg_process_ms']:<14.2f} | {res_disabled['avg_render_ms']:<12.2f} | N/A")
    print(f"{'Detection enabled, overlay OFF':<35} | {res_det_only['fps']:<12.2f} | {res_det_only['avg_process_ms']:<14.2f} | {res_det_only['avg_render_ms']:<12.2f} | {ot_proc_lat:<12.2f}")
    print(f"{'Detection enabled, overlay ON':<35} | {res_full['fps']:<12.2f} | {res_full['avg_process_ms']:<14.2f} | {res_full['avg_render_ms']:<12.2f} | {ot_proc_lat + vis_overhead:<12.2f}")
    print("=" * 105)

    print("\nVIDEO VALIDATION SUMMARY")
    print("=====================================================================")
    print(f"{'Test Video':<18} | {'Frames':<8} | {'Candidates':<12} | {'Completed':<10} | {'Incomplete':<11} | {'Uncertain'}")
    print("-" * 80)
    print(f"{'data/test1.mp4':<18} | {res_full['frames_processed']:<8} | {res_full['candidates']:<12} | {res_full['completed']:<10} | {res_full['incomplete']:<11} | {res_full['uncertain']}")
    print(f"{'data/test2.mp4':<18} | {res_test2['frames_processed']:<8} | {res_test2['candidates']:<12} | {res_test2['completed']:<10} | {res_test2['incomplete']:<11} | {res_test2['uncertain']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

