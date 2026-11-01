#!/usr/bin/env python3
"""CLI entry point for offline ADAS evaluation.

Usage::

    python -m src.evaluation.runner \\
        --config configs/default.yaml \\
        --video data/eval.mp4 \\
        --gt-detections gt.json \\
        --gt-tracks gt_mot.txt \\
        --gt-depth gt_depth.npz \\
        --report-out reports/eval.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

from src.config.loader import load_config
from src.core.pipeline import PerceptionPipeline
from src.evaluation.detection_metrics import DetectionEvaluator
from src.evaluation.depth_metrics import DepthEvaluator
from src.evaluation.evaluators.behavior_evaluator import BehaviorEvaluator
from src.evaluation.evaluators.pose_evaluator import PoseEvaluator
from src.evaluation.profiler import PipelineProfiler
from src.evaluation.system_metrics import SystemEvaluator
from src.evaluation.tracking_metrics import TrackingEvaluator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline ADAS evaluation harness",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to evaluation video file",
    )
    parser.add_argument(
        "--gt-detections", default=None,
        help="Ground-truth detections (COCO JSON or per-frame JSON)",
    )
    parser.add_argument(
        "--gt-tracks", default=None,
        help="Ground-truth tracking (MOT-format text file)",
    )
    parser.add_argument(
        "--gt-depth", default=None,
        help="Ground-truth depth maps (.npz with frame keys or directory of .npy)",
    )
    parser.add_argument(
        "--gt-pose", default=None,
        help="Ground-truth pose keypoints (COCO-keypoints / AP-10K JSON format)",
    )
    parser.add_argument(
        "--gt-behavior", default=None,
        help="Ground-truth behavior labels (behavior JSON schema)",
    )
    parser.add_argument(
        "--report-out", default="reports/eval.json",
        help="Output path for the evaluation report",
    )
    return parser.parse_args()


def _load_gt_depth(gt_path: str, num_frames: int) -> list[np.ndarray] | None:
    if not os.path.exists(gt_path):
        print(f"[eval] GT depth path not found: {gt_path}")
        return None

    if gt_path.endswith(".npz"):
        data = np.load(gt_path)
        maps: list[np.ndarray] = []
        for i in range(num_frames):
            key = str(i)
            if key in data:
                maps.append(data[key])
            else:
                return None
        return maps if len(maps) == num_frames else None

    if os.path.isdir(gt_path):
        maps = []
        for i in range(num_frames):
            p = os.path.join(gt_path, f"gt_depth_{i:04d}.npy")
            if os.path.exists(p):
                maps.append(np.load(p))
            else:
                return None
        return maps if len(maps) == num_frames else None

    return None


def _load_gt_json(gt_path: str, label: str = "JSON") -> dict[str, Any] | None:
    if not os.path.exists(gt_path):
        print(f"[eval] GT {label} path not found: {gt_path}")
        return None
    with open(gt_path, "r") as f:
        return json.load(f)


def _load_gt_detections(gt_path: str) -> dict[str, Any] | None:
    return _load_gt_json(gt_path, "detections")


def _write_report(
    results: dict[str, Any],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    json_path = output_path
    if output_path.endswith(".md"):
        json_path = output_path.rsplit(".", 1)[0] + ".json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[eval] JSON report written to {json_path}")

    if output_path.endswith(".md") or True:
        md_path = output_path if output_path.endswith(".md") else output_path.rsplit(".", 1)[0] + ".md"
        _write_markdown(results, md_path)
        print(f"[eval] Markdown report written to {md_path}")


def _write_markdown(results: dict[str, Any], path: str) -> None:
    lines: list[str] = [
        "# ADAS Evaluation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if "system" in results:
        s = results["system"]
        lines.append("## System Performance")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total frames | {s.get('count', 'N/A')} |")
        lines.append(f"| Mean FPS | {s.get('fps', 'N/A')} |")
        lines.append(f"| Mean latency (ms) | {s.get('mean_latency_ms', 'N/A')} |")
        lines.append(f"| P50 latency (ms) | {s.get('p50_latency_ms', 'N/A')} |")
        lines.append(f"| P95 latency (ms) | {s.get('p95_latency_ms', 'N/A')} |")
        lines.append(f"| P99 latency (ms) | {s.get('p99_latency_ms', 'N/A')} |")
        lines.append(f"| Peak RSS (MB) | {s.get('peak_rss_mb', 'N/A')} |")
        lines.append("")

    if "profiler" in results:
        lines.append("## Per-Stage Latency Breakdown")
        lines.append("")
        lines.append("| Stage | Count | Mean(ms) | P50(ms) | P95(ms) | P99(ms) | Total(ms) | Fraction |")
        lines.append("|-------|-------|----------|---------|---------|---------|-----------|----------|")
        prof = results["profiler"]
        for stage in sorted(prof.keys()):
            if stage.startswith("_"):
                continue
            s = prof[stage]
            lines.append(
                f"| {stage} | {s.get('count', 0)} |"
                f" {s.get('mean_ms', 0):.2f} |"
                f" {s.get('p50_ms', 0):.2f} |"
                f" {s.get('p95_ms', 0):.2f} |"
                f" {s.get('p99_ms', 0):.2f} |"
                f" {s.get('total_ms', 0):.2f} |"
                f" {s.get('fraction', 0):.4f} |"
            )
        lines.append("")

    if "detection" in results:
        d = results["detection"]
        lines.append("## Detection Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| mAP@0.5 | {d.get('mAP@0.5', 'N/A')} |")
        lines.append(f"| mAP@0.5:0.95 | {d.get('mAP@0.5:0.95', 'N/A')} |")
        lines.append("")
        per_class = d.get("per_class", {})
        if per_class:
            lines.append("### Per-Class Metrics (IoU=0.5)")
            lines.append("")
            lines.append("| Class | Precision | Recall | AP@0.5 |")
            lines.append("|-------|-----------|--------|--------|")
            for cls_name in sorted(per_class.keys()):
                pc = per_class[cls_name]
                lines.append(
                    f"| {cls_name} | {pc.get('precision', 'N/A')} |"
                    f" {pc.get('recall', 'N/A')} |"
                    f" {pc.get('AP@0.5', 'N/A')} |"
                )
            lines.append("")

    if "tracking" in results:
        t = results["tracking"]
        lines.append("## Tracking Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| MOTA | {t.get('MOTA', 'N/A')} |")
        lines.append(f"| MOTP | {t.get('MOTP', 'N/A')} |")
        lines.append(f"| IDF1 | {t.get('IDF1', 'N/A')} |")
        lines.append(f"| Mostly Tracked | {t.get('MT', 'N/A')} |")
        lines.append(f"| Mostly Lost | {t.get('ML', 'N/A')} |")
        lines.append(f"| ID Switches | {t.get('num_switches', 'N/A')} |")
        lines.append(f"| Fragmentations | {t.get('fragmentations', 'N/A')} |")
        lines.append(f"| False Positives | {t.get('FP', 'N/A')} |")
        lines.append(f"| False Negatives | {t.get('FN', 'N/A')} |")
        lines.append("")

    if "depth" in results:
        dp = results["depth"]
        lines.append("## Depth Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for k, v in dp.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if "pose" in results:
        p = results["pose"]
        lines.append("## Pose Estimation Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Human PCK@0.05 | {p.get('human_pck_at_0.05', 'N/A')} |")
        lines.append(f"| Human OKS | {p.get('human_oks', 'N/A')} |")
        lines.append(f"| Human matched instances | {p.get('human_pck_at_0.05_count', 'N/A')} |")
        lines.append(f"| Human unmatched GT | {p.get('human_unmatched_gt', 'N/A')} |")
        lines.append(f"| Animal PCK@0.05 | {p.get('animal_pck_at_0.05', 'N/A')} |")
        lines.append(f"| Animal matched instances | {p.get('animal_pck_at_0.05_count', 'N/A')} |")
        lines.append(f"| Animal unmatched GT | {p.get('animal_unmatched_gt', 'N/A')} |")
        lines.append("")

    if "behavior" in results:
        b = results["behavior"]
        lines.append("## Behavior Classification Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Macro Precision | {b.get('label_metrics', {}).get('macro_precision', 'N/A')} |")
        lines.append(f"| Macro Recall | {b.get('label_metrics', {}).get('macro_recall', 'N/A')} |")
        lines.append(f"| Macro F1 | {b.get('label_metrics', {}).get('macro_f1', 'N/A')} |")
        lines.append(f"| Brier Score | {b.get('brier_score', 'N/A')} |")
        lines.append(f"| Total matched | {b.get('total_matched', 'N/A')} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.config):
        print(f"[eval] Config not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"[eval] Video not found: {args.video}")
        sys.exit(1)

    config = load_config(args.config)
    config["video"] = {"type": "file", "path": args.video, "loop": False}

    pipeline = PerceptionPipeline(config)
    with pipeline.frame_source as source:
        pass
    pipeline._frame_source = None

    system_eval = SystemEvaluator()
    all_outputs: list[dict[str, Any]] = []
    all_detections: list[list[Any]] = []
    all_tracked: list[list[Any]] = []
    all_frames: list[Any] = []
    all_depth: list[np.ndarray] = []

    print(f"[eval] Processing video: {args.video}")
    start_time = time.perf_counter()

    with pipeline.frame_source as source:
        profiler = PipelineProfiler(pipeline)
        with profiler:
            while True:
                frame = source.read()
                if frame is None:
                    break

                output = pipeline.process(frame)
                all_outputs.append(output)
                all_detections.append(output.get("detections", []))
                all_tracked.append(output.get("tracked", []))
                all_frames.append(frame)
                all_depth.append(output.get("depth_map", np.zeros((1, 1), dtype=np.float32)))

    elapsed = time.perf_counter() - start_time
    n_frames = len(all_outputs)
    print(f"[eval] Processed {n_frames} frames in {elapsed:.2f}s ({n_frames / elapsed:.1f} FPS)")

    for output in all_outputs:
        latency = time.perf_counter()
        system_eval.record(0.0)

    results: dict[str, Any] = {}

    sys_summary = system_eval.summary()
    sys_summary["total_elapsed_s"] = round(elapsed, 2)
    sys_summary["total_frames"] = n_frames
    sys_summary["fps"] = round(n_frames / elapsed, 1) if elapsed > 0 else 0.0
    results["system"] = sys_summary

    prof_summary = profiler.summary()
    results["profiler"] = prof_summary

    if args.gt_detections:
        print(f"[eval] Running detection evaluation ...")
        gt_det = _load_gt_detections(args.gt_detections)
        if gt_det is not None:
            det_eval = DetectionEvaluator(iou_thresholds=[0.5, 0.75])
            results["detection"] = det_eval.evaluate(all_detections, gt_det)
            print(f"  mAP@0.5 = {results['detection'].get('mAP@0.5', 'N/A')}")
            print(f"  mAP@0.5:0.95 = {results['detection'].get('mAP@0.5:0.95', 'N/A')}")

    if args.gt_tracks:
        print(f"[eval] Running tracking evaluation ...")
        track_eval = TrackingEvaluator()
        try:
            results["tracking"] = track_eval.evaluate(all_tracked, args.gt_tracks)
            print(f"  MOTA = {results['tracking'].get('MOTA', 'N/A')}")
            print(f"  IDF1 = {results['tracking'].get('IDF1', 'N/A')}")
        except Exception as e:
            print(f"  Tracking evaluation failed: {e}")

    depth_results: dict[str, Any] = {}
    depth_eval = DepthEvaluator(mode="temporal_consistency")
    depth_results.update(depth_eval.evaluate(all_frames, all_depth))
    print(f"[eval] Depth temporal consistency MAE = {depth_results.get('temporal_consistency_mae', 'N/A')}")

    if args.gt_depth:
        print(f"[eval] Running depth GT evaluation ...")
        gt_maps = _load_gt_depth(args.gt_depth, n_frames)
        if gt_maps is not None:
            gt_depth_eval = DepthEvaluator(mode="gt")
            gt_res = gt_depth_eval.evaluate(all_frames, all_depth, gt_maps)
            depth_results.update(gt_res)
            print(f"  AbsRel = {gt_res.get('AbsRel', 'N/A')}")
            print(f"  RMSE = {gt_res.get('RMSE', 'N/A')}")

    results["depth"] = depth_results

    eval_cfg = config.get("evaluation", {})
    all_pose: list[Any] = [o.get("pose") for o in all_outputs]
    all_behavior: list[Any] = [o.get("behavior") for o in all_outputs]

    if args.gt_pose:
        print(f"[eval] Running pose evaluation ...")
        gt_pose = _load_gt_json(args.gt_pose, "pose")
        if gt_pose is not None:
            if eval_cfg.get("pose_evaluation", {}).get("enabled", True):
                p_thr = eval_cfg.get("pose_evaluation", {}).get("match_iou_threshold", 0.5)
                pose_eval = PoseEvaluator(match_iou_threshold=float(p_thr))
                results["pose"] = pose_eval.evaluate(all_pose, gt_pose)
                print(f"  human PCK@0.05 = {results['pose'].get('human_pck_at_0.05', 'N/A')}")
                print(f"  human OKS = {results['pose'].get('human_oks', 'N/A')}")
                print(f"  animal PCK@0.05 = {results['pose'].get('animal_pck_at_0.05', 'N/A')}")

    if args.gt_behavior:
        print(f"[eval] Running behavior evaluation ...")
        gt_behavior = _load_gt_json(args.gt_behavior, "behavior")
        if gt_behavior is not None:
            if eval_cfg.get("behavior_evaluation", {}).get("enabled", True):
                beh_eval = BehaviorEvaluator()
                results["behavior"] = beh_eval.evaluate(all_behavior, gt_behavior)
                mf1 = results["behavior"].get("label_metrics", {}).get("macro_f1", "N/A")
                bs = results["behavior"].get("brier_score", "N/A")
                print(f"  macro F1 = {mf1}")
                print(f"  Brier score = {bs}")

    _write_report(results, args.report_out)
    print(f"[eval] Done.")


if __name__ == "__main__":
    main()
