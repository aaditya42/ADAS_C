from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable

import numpy as np


class PipelineProfiler:
    """Context manager that monkey-patches each pipeline stage to record
    per-call latency.

    Usage::

        pipe = PerceptionPipeline(config)
        with PipelineProfiler(pipe) as prof:
            for _ in range(total_frames):
                pipe.process(frame)

        print(prof.summary())
    """

    STAGES = [
        "detector.detect",
        "tracker.update",
        "depth_estimator.estimate",
        "fusion.fuse",
        "distance_estimator.estimate",
        "history.update",
        "motion_estimator.estimate",
        "ttc_estimator.estimate",
        "lane_detector.detect",
        "sign_detector.detect",
        "scene_builder.build",
        "risk_engine.evaluate",
        "warning_manager.process",
    ]

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.stage_times: dict[str, list[float]] = defaultdict(list)
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> PipelineProfiler:
        self._patch()
        return self

    def __exit__(self, *args: Any) -> None:
        self._restore()

    def _patch(self) -> None:
        self._patch_module("detector", "detect")
        self._patch_module("tracker", "update")
        self._patch_module("depth_estimator", "estimate", "estimate_depth")
        self._patch_module("fusion", "fuse")
        self._patch_module("distance_estimator", "estimate", "enrich_distances")
        self._patch_module("history", "update", "history_update")
        self._patch_module("motion_estimator", "estimate", "compute_motion")
        self._patch_module("ttc_estimator", "estimate", "compute_ttc")
        self._patch_module("lane_detector", "detect", "detect_lanes")
        self._patch_module("warning_manager", "process", "process_warnings")
        self._patch_method("risk_engine", "evaluate")
        self._patch_method("_scene_builder", "build", "scene_builder.build")
        self._patch_method("_sign_detector", "detect", "sign_detector.detect")

    def _patch_module(self, attr: str, method: str, stage_name: str | None = None) -> None:
        name = stage_name or f"{attr}.{method}"
        module = getattr(self.pipeline, attr, None)
        if module is None:
            return
        original = getattr(module, method, None)
        if original is None:
            return
        self._originals[name] = (module, method, original)

        def _make_wrapper(stage: str, orig: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return orig(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    self.stage_times[stage].append(elapsed)
            return wrapper

        setattr(module, method, _make_wrapper(name, original))

    def _patch_method(self, attr: str, method: str, stage_name: str | None = None) -> None:
        name = stage_name or f"{attr}.{method}"
        obj = getattr(self.pipeline, attr, None)
        if obj is None:
            return
        original = getattr(obj, method, None)
        if original is None:
            return
        self._originals[name] = (obj, method, original)

        def _make_wrapper(stage: str, orig: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return orig(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    self.stage_times[stage].append(elapsed)
            return wrapper

        setattr(obj, method, _make_wrapper(name, original))

    def _restore(self) -> None:
        for name, (obj, method, original) in self._originals.items():
            setattr(obj, method, original)
        self._originals.clear()

    def summary(self) -> dict[str, Any]:
        """Compute per-stage latency statistics.

        Returns:
            Dict keyed by stage name with values containing ``count``,
            ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``, ``total_ms``
            and ``fraction`` (proportion of total pipeline time).
        """
        total_all = sum(
            sum(times) for times in self.stage_times.values()
        )

        result: dict[str, Any] = {}
        for stage, times in sorted(self.stage_times.items()):
            arr = sorted(times)
            n = len(arr)
            if n == 0:
                result[stage] = {
                    "count": 0, "mean_ms": 0.0, "p50_ms": 0.0,
                    "p95_ms": 0.0, "p99_ms": 0.0,
                    "total_ms": 0.0, "fraction": 0.0,
                }
                continue
            mean_ms = float(np.mean(arr))
            p50 = float(arr[int(n * 0.50)])
            p95 = float(arr[int(n * 0.95)])
            p99 = float(arr[int(n * 0.99)])
            total = float(sum(arr))
            fraction = total / total_all if total_all > 0 else 0.0

            result[stage] = {
                "count": n,
                "mean_ms": round(mean_ms, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "total_ms": round(total, 2),
                "fraction": round(fraction, 4),
            }

        result["_total_all_ms"] = round(total_all, 2)
        return result

    def summary_table(self, results: dict[str, Any] | None = None) -> str:
        """Pretty-print per-stage latency breakdown."""
        if results is None:
            results = self.summary()

        lines = [
            "+----------------------------+-------+----------+----------+----------+----------+----------+----------+",
            "| Stage                      | Count | Mean(ms) | P50(ms)  | P95(ms)  | P99(ms)  | Total(ms)| Fraction |",
            "+----------------------------+-------+----------+----------+----------+----------+----------+----------+",
        ]

        for stage in sorted(self.stage_times.keys()):
            s = results.get(stage, {})
            lines.append(
                f"| {stage:<26s} | {s.get('count', 0):>5d} |"
                f" {s.get('mean_ms', 0):>8.2f} |"
                f" {s.get('p50_ms', 0):>8.2f} |"
                f" {s.get('p95_ms', 0):>8.2f} |"
                f" {s.get('p99_ms', 0):>8.2f} |"
                f" {s.get('total_ms', 0):>8.2f} |"
                f" {s.get('fraction', 0):>8.4f} |"
            )

        lines.append(
            "+----------------------------+-------+----------+----------+----------+----------+----------+----------+"
        )
        lines.append(
            f"| TOTAL                      |       |          |          |          |          |"
            f" {results.get('_total_all_ms', 0):>8.2f} |   1.0000 |"
        )
        lines.append(
            "+----------------------------+-------+----------+----------+----------+----------+----------+----------+"
        )
        return "\n".join(lines)
