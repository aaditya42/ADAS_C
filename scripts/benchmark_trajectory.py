"""Benchmark script comparing NAYAN ADAS pipeline performance with and without Trajectory Prediction.

Measures:
- Baseline vs. Enabled FPS
- Per-component latency breakdown (Detection, Tracking, Depth, Trajectory, Evaluation, Rendering)
- Peak memory usage
- Excel export duration
- Generates annotated sample video and Excel report
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
import cv2
import openpyxl
import yaml

os.environ["HF_HUB_OFFLINE"] = "1"

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.pipeline import PerceptionPipeline
from src.utils.logger import get_logger

logger = get_logger("benchmark_trajectory")


def run_pipeline_benchmark(
    config: dict,
    video_path: str,
    max_frames: int = 120,
    save_video_path: str | None = None,
    export_excel_path: str | None = None,
) -> dict:
    """Run pipeline on video_path and return precise timing metrics."""
    config["video"] = {"type": "file", "path": video_path, "loop": False}
    pipeline = PerceptionPipeline(config)
    source = pipeline.frame_source

    writer = None
    if save_video_path is not None:
        p = Path(save_video_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(p), fourcc, 30.0, (source.width, source.height))

    import psutil
    proc = psutil.Process(os.getpid())
    peak_mem_mb = proc.memory_info().rss / (1024 * 1024)

    frame_count = 0
    render_times = []
    process_times = []
    start_time = None

    with source:
        while frame_count < max_frames:
            frame = source.read()
            if frame is None:
                break

            if start_time is None:
                start_time = time.perf_counter()

            t0 = time.perf_counter()
            result = pipeline.process(frame)
            t1 = time.perf_counter()
            process_times.append((t1 - t0) * 1000.0)

            t_r0 = time.perf_counter()
            rendered = pipeline.render(result)
            t_r1 = time.perf_counter()
            render_times.append((t_r1 - t_r0) * 1000.0)

            if writer is not None:
                writer.write(rendered)

            cur_mem = proc.memory_info().rss / (1024 * 1024)
            if cur_mem > peak_mem_mb:
                peak_mem_mb = cur_mem

            frame_count += 1

    total_time = (time.perf_counter() - start_time) if start_time else 0.0

    if writer is not None:
        writer.release()

    excel_export_duration_ms = 0.0
    exported_file = None
    if export_excel_path is not None:
        t_ex0 = time.perf_counter()
        exported_file = pipeline.export_trajectory_evaluation(export_excel_path)
        excel_export_duration_ms = (time.perf_counter() - t_ex0) * 1000.0

    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    avg_proc = sum(process_times) / len(process_times) if process_times else 0.0
    avg_rend = sum(render_times) / len(render_times) if render_times else 0.0

    # Extract internal predictor telemetry if present
    pred_lat_ms = 0.0
    eval_lat_ms = 0.0
    tracked_count = 0.0
    total_preds = 0
    matched_preds = 0

    if pipeline.trajectory_predictor is not None:
        tp = pipeline.trajectory_predictor
        if tp._prediction_latencies:
            pred_lat_ms = sum(tp._prediction_latencies) / len(tp._prediction_latencies)
        if tp._evaluation_latencies:
            eval_lat_ms = sum(tp._evaluation_latencies) / len(tp._evaluation_latencies)
        if tp._tracked_counts:
            tracked_count = sum(tp._tracked_counts) / len(tp._tracked_counts)
        records = tp.prediction_manager.get_records()
        total_preds = len(records)
        matched_preds = sum(1 for r in records if r.status.value == "matched")

    return {
        "frames_processed": frame_count,
        "total_time_seconds": round(total_time, 3),
        "fps": round(avg_fps, 2),
        "avg_process_ms": round(avg_proc, 2),
        "avg_render_ms": round(avg_rend, 2),
        "avg_prediction_ms": round(pred_lat_ms, 3),
        "avg_evaluation_ms": round(eval_lat_ms, 3),
        "avg_tracked_objects": round(tracked_count, 1),
        "total_predictions": total_preds,
        "matched_predictions": matched_preds,
        "peak_memory_mb": round(peak_mem_mb, 2),
        "excel_export_duration_ms": round(excel_export_duration_ms, 2),
        "exported_file": str(exported_file) if exported_file else None,
        "video_output": save_video_path,
    }


def main() -> None:
    config_path = Path("configs/default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    video_path = "data/test2.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)

    print("==================================================================")
    print(" NAYAN ADAS: Trajectory Prediction & Pipeline Benchmark")
    print(f" Video: {video_path}")
    print("==================================================================")

    # 1. Baseline Run (Trajectory Prediction Disabled)
    print("\n[Run 1/2] Benchmarking BASELINE Pipeline (Trajectory Disabled)...")
    cfg_baseline = copy.deepcopy(base_cfg)
    cfg_baseline.setdefault("trajectory", {})["enabled"] = False
    cfg_baseline.setdefault("visualization", {})["enabled"] = True
    cfg_baseline["visualization"]["display"] = False
    cfg_baseline["visualization"]["save_video"] = False

    res_baseline = run_pipeline_benchmark(cfg_baseline, video_path, max_frames=80)
    print(f"  Baseline Frames: {res_baseline['frames_processed']}")
    print(f"  Baseline Time:   {res_baseline['total_time_seconds']}s")
    print(f"  Baseline FPS:    {res_baseline['fps']} FPS")
    print(f"  Avg Process:     {res_baseline['avg_process_ms']} ms/frame")
    print(f"  Avg Render:      {res_baseline['avg_render_ms']} ms/frame")
    print(f"  Peak Memory:     {res_baseline['peak_memory_mb']} MB")

    # 2. Enabled Run (Trajectory Prediction + Live Evaluation + Visualization + Excel)
    print("\n[Run 2/2] Benchmarking FULL Pipeline (Trajectory Prediction Enabled)...")
    cfg_enabled = copy.deepcopy(base_cfg)
    cfg_enabled.setdefault("trajectory", {})["enabled"] = True
    cfg_enabled.setdefault("visualization", {})["enabled"] = True
    cfg_enabled["visualization"]["display"] = False
    cfg_enabled["visualization"]["save_video"] = False

    demo_video = "data/output/trajectory_validation_demo.mp4"
    excel_path = "reports/NAYAN_Trajectory_Evaluation.xlsx"

    res_enabled = run_pipeline_benchmark(
        cfg_enabled,
        video_path,
        max_frames=80,
        save_video_path=demo_video,
        export_excel_path=excel_path,
    )
    print(f"  Enabled Frames:     {res_enabled['frames_processed']}")
    print(f"  Enabled Time:       {res_enabled['total_time_seconds']}s")
    print(f"  Enabled FPS:        {res_enabled['fps']} FPS")
    print(f"  Avg Process:        {res_enabled['avg_process_ms']} ms/frame")
    print(f"  Avg Render:         {res_enabled['avg_render_ms']} ms/frame")
    print(f"  Avg Traj Predict:   {res_enabled['avg_prediction_ms']} ms/frame")
    print(f"  Avg Traj Eval:      {res_enabled['avg_evaluation_ms']} ms/frame")
    print(f"  Avg Tracked Objs:   {res_enabled['avg_tracked_objects']}")
    print(f"  Total Predictions:  {res_enabled['total_predictions']}")
    print(f"  Matched Preds:      {res_enabled['matched_predictions']}")
    print(f"  Peak Memory:        {res_enabled['peak_memory_mb']} MB")
    print(f"  Excel Export Time:  {res_enabled['excel_export_duration_ms']} ms")
    print(f"  Workbook Output:    {res_enabled['exported_file']}")
    print(f"  Annotated Video:    {res_enabled['video_output']}")

    # 3. Comparative Summary
    fps_delta = res_enabled["fps"] - res_baseline["fps"]
    fps_impact_pct = (fps_delta / res_baseline["fps"] * 100.0) if res_baseline["fps"] > 0 else 0.0
    traj_overhead_ms = res_enabled["avg_prediction_ms"] + res_enabled["avg_evaluation_ms"]

    print("\n==================================================================")
    print(" Comparative Performance Summary")
    print("==================================================================")
    print(f" Baseline FPS:                   {res_baseline['fps']:.2f}")
    print(f" Trajectory-Enabled FPS:         {res_enabled['fps']:.2f}")
    print(f" Delta FPS:                      {fps_delta:+.2f} ({fps_impact_pct:+.1f}%)")
    print(f" Trajectory Prediction Latency:  {res_enabled['avg_prediction_ms']:.3f} ms")
    print(f" Evaluation / Matching Latency:  {res_enabled['avg_evaluation_ms']:.3f} ms")
    print(f" Total Trajectory Overhead:      {traj_overhead_ms:.3f} ms/frame")
    print(f" Peak Memory Overhead:           {res_enabled['peak_memory_mb'] - res_baseline['peak_memory_mb']:+.2f} MB")
    print(f" Excel Export Duration:          {res_enabled['excel_export_duration_ms']:.2f} ms")
    print("==================================================================")

    # 4. Programmatic Verification of Exported Excel
    if res_enabled["exported_file"] and os.path.exists(res_enabled["exported_file"]):
        wb = openpyxl.load_workbook(res_enabled["exported_file"], data_only=True)
        print(f"\n[Verification] Successfully verified Excel workbook: {res_enabled['exported_file']}")
        print(f"  Sheets present: {wb.sheetnames}")
        ws1 = wb["Prediction Records"]
        print(f"  Total raw prediction records in sheet 1: {ws1.max_row - 1}")
        ws2 = wb["Detailed Future Positions"]
        print(f"  Total future waypoints in sheet 2: {ws2.max_row - 1}")


if __name__ == "__main__":
    main()
