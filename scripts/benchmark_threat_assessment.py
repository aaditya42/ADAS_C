"""Benchmark and evaluation script for Priority-Based Warning System and Threat Assessment.

Evaluates:
1. Full video execution on test1.mp4 and exports reports/Threat_Assessment_Evaluation.xlsx
2. Runtime overhead benchmarking across 6 configurations:
   - Config 1: Baseline pipeline (threat assessment disabled)
   - Config 2: Threat assessment enabled (features + scenarios + risk scoring + prioritization)
   - Config 3: Warning management enabled (directional warnings + lifecycle management)
   - Config 4: Warning HUD OFF (visual layer present, toggle=OFF)
   - Config 5: Warning HUD ON (full visual rendering and blending)
   - Config 6: Full reporting enabled (in-memory diagnostics + Excel export)
3. Metric calculations: Mean latency, P95/P99 latency, FPS, RAM usage, and export time.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import openpyxl
import yaml

os.environ["HF_HUB_OFFLINE"] = "1"

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.core.pipeline import PerceptionPipeline
from src.threat.schemas import DirectionalWarningLevel, WarningEventState
from src.utils.logger import get_logger

logger = get_logger("benchmark_threat_assessment")


def run_benchmark_config(
    base_config: dict,
    video_path: str,
    max_frames: int = 150,
    enable_threat: bool = True,
    enable_warnings: bool = True,
    enable_hud: bool = True,
    export_excel: bool = False,
    output_excel_name: str | None = None,
) -> dict:
    import psutil

    cfg = copy.deepcopy(base_config)
    cfg["video"] = {"type": "file", "path": video_path, "loop": False}
    cfg["visualization"]["display"] = False  # Headless execution
    cfg["visualization"]["save_video"] = False

    # Configure threat subsystems
    cfg.setdefault("threat_assessment", {})
    cfg["threat_assessment"]["enabled"] = enable_threat
    cfg["threat_assessment"].setdefault("directional_warnings", {})
    cfg["threat_assessment"]["directional_warnings"]["enabled"] = enable_warnings
    cfg["threat_assessment"]["directional_warnings"].setdefault("export", {})
    cfg["threat_assessment"]["directional_warnings"]["export"]["export_on_session_end"] = export_excel
    if output_excel_name:
        cfg["threat_assessment"]["directional_warnings"]["export"]["filename"] = output_excel_name

    pipeline = PerceptionPipeline(cfg)
    source = pipeline.frame_source

    if pipeline.renderer is not None:
        layer = pipeline.renderer._name_map.get("warning")
        if layer is not None:
            layer.enabled = enable_hud

    proc = psutil.Process(os.getpid())
    peak_mem_mb = proc.memory_info().rss / (1024 * 1024)

    frame_count = 0
    process_latencies = []
    render_latencies = []
    threat_latencies = []
    warning_draw_latencies = []
    start_time = None

    with source:
        while frame_count < max_frames:
            frame = source.read()
            if frame is None:
                break

            if start_time is None:
                start_time = time.perf_counter()

            # Process latency
            t0 = time.perf_counter()
            result = pipeline.process(frame)
            t1 = time.perf_counter()
            process_latencies.append((t1 - t0) * 1000.0)

            # Render latency
            t_r0 = time.perf_counter()
            rendered = pipeline.render(result)
            t_r1 = time.perf_counter()
            render_latencies.append((t_r1 - t_r0) * 1000.0)

            cur_mem = proc.memory_info().rss / (1024 * 1024)
            if cur_mem > peak_mem_mb:
                peak_mem_mb = cur_mem

            frame_count += 1

    total_time = (time.perf_counter() - start_time) if start_time else 0.0
    effective_fps = (frame_count / total_time) if total_time > 0 else 0.0

    export_duration_ms = 0.0
    exported_file_path = None
    if export_excel:
        t_exp0 = time.perf_counter()
        pipeline.finish_threat_session()
        exported_file_path = pipeline.export_threat_evaluation()
        t_exp1 = time.perf_counter()
        export_duration_ms = (t_exp1 - t_exp0) * 1000.0

    p_lat = np.array(process_latencies) if process_latencies else np.array([0.0])
    r_lat = np.array(render_latencies) if render_latencies else np.array([0.0])

    events = pipeline.get_directional_warning_events()
    active_warnings = pipeline.get_directional_warnings()

    return {
        "frames_processed": frame_count,
        "total_time_s": total_time,
        "effective_fps": effective_fps,
        "process_mean_ms": float(np.mean(p_lat)),
        "process_p95_ms": float(np.percentile(p_lat, 95)),
        "process_p99_ms": float(np.percentile(p_lat, 99)),
        "render_mean_ms": float(np.mean(r_lat)),
        "render_p95_ms": float(np.percentile(r_lat, 95)),
        "total_frame_mean_ms": float(np.mean(p_lat) + np.mean(r_lat)),
        "peak_mem_mb": peak_mem_mb,
        "export_duration_ms": export_duration_ms,
        "exported_file": str(exported_file_path) if exported_file_path else None,
        "total_events": len(events),
        "active_warnings": len(active_warnings),
        "warning_generator": pipeline.directional_warning_generator,
    }


def main():
    config_path = Path("configs/default.yaml")
    base_cfg = load_config(config_path)

    test1_path = "./data/test1.mp4"
    if not Path(test1_path).exists():
        print(f"Error: Video file {test1_path} not found.")
        return

    print("=" * 80)
    print("STEP 1: FULL RUN ON test1.mp4 WITH EXCEL REPORT GENERATION")
    print("=" * 80)
    full_result = run_benchmark_config(
        base_config=base_cfg,
        video_path=test1_path,
        max_frames=1200,  # Process entire video (~1165 frames)
        enable_threat=True,
        enable_warnings=True,
        enable_hud=True,
        export_excel=True,
        output_excel_name="Threat_Assessment_Evaluation.xlsx",
    )
    print(f"Full Run Completed:")
    print(f"  Frames Processed: {full_result['frames_processed']}")
    print(f"  Total Duration: {full_result['total_time_s']:.2f} s")
    print(f"  Effective FPS: {full_result['effective_fps']:.2f}")
    print(f"  Mean Process Latency: {full_result['process_mean_ms']:.2f} ms")
    print(f"  Mean Render Latency: {full_result['render_mean_ms']:.2f} ms")
    print(f"  Total Warning Events: {full_result['total_events']}")
    print(f"  Export Duration: {full_result['export_duration_ms']:.2f} ms")
    print(f"  Exported Workbook: {full_result['exported_file']}")

    if full_result["warning_generator"]:
        wg = full_result["warning_generator"]
        events = wg.get_all_events()
        history = wg.get_warning_history()
        details = wg.get_risk_assessment_details()
        suppressed = wg.get_suppressed_duplicate_count()

        high_ev = sum(1 for e in events if e.current_level == DirectionalWarningLevel.HIGH)
        med_ev = sum(1 for e in events if e.current_level == DirectionalWarningLevel.MEDIUM)
        low_ev = sum(1 for e in events if e.current_level == DirectionalWarningLevel.LOW)

        print(f"\nWarning Event Breakdown:")
        print(f"  HIGH Warnings: {high_ev}")
        print(f"  MEDIUM Warnings: {med_ev}")
        print(f"  LOW Warnings: {low_ev}")
        print(f"  State Transitions Logged: {len(history)}")
        print(f"  Detailed Assessments Logged: {len(details)}")
        print(f"  Duplicate Warnings Suppressed: {suppressed}")

    print("\n" + "=" * 80)
    print("STEP 2: RUNTIME BENCHMARKING ACROSS 6 ARCHITECTURAL CONFIGURATIONS")
    print("=" * 80)
    benchmark_frames = 150

    configs_to_test = [
        ("Config 1: Baseline Pipeline (Threat OFF)", False, False, False, False),
        ("Config 2: Threat Assessment Enabled (Warnings OFF)", True, False, False, False),
        ("Config 3: Warning Management Enabled (HUD OFF)", True, True, False, False),
        ("Config 4: Warning HUD Toggle OFF", True, True, False, False),
        ("Config 5: Warning HUD Toggle ON", True, True, True, False),
        ("Config 6: Full Reporting Enabled (Excel Export)", True, True, True, True),
    ]

    results = []
    for name, en_th, en_wn, en_hud, en_exp in configs_to_test:
        print(f"Benchmarking {name} ({benchmark_frames} frames)...")
        res = run_benchmark_config(
            base_config=base_cfg,
            video_path=test1_path,
            max_frames=benchmark_frames,
            enable_threat=en_th,
            enable_warnings=en_wn,
            enable_hud=en_hud,
            export_excel=en_exp,
            output_excel_name="Threat_Benchmark_Temp.xlsx",
        )
        results.append((name, res))
        print(f"  -> FPS: {res['effective_fps']:.1f} | Frame Latency: {res['total_frame_mean_ms']:.2f} ms (P95: {res['process_p95_ms']:.2f} ms) | RAM: {res['peak_mem_mb']:.1f} MB")

    print("\n" + "=" * 80)
    print("STEP 3: OVERHEAD SUMMARY TABLE")
    print("=" * 80)
    print(f"| {'Configuration':<50} | {'FPS':>6} | {'Proc (ms)':>10} | {'P95 (ms)':>9} | {'P99 (ms)':>9} | {'Render':>8} | {'RAM (MB)':>9} |")
    print(f"|{'-'*52}|{'-'*8}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*10}|{'-'*11}|")
    for name, r in results:
        print(
            f"| {name:<50} | {r['effective_fps']:>6.1f} | {r['process_mean_ms']:>10.2f} | "
            f"{r['process_p95_ms']:>9.2f} | {r['process_p99_ms']:>9.2f} | {r['render_mean_ms']:>8.2f} | "
            f"{r['peak_mem_mb']:>9.1f} |"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()

