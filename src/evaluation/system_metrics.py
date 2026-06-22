from __future__ import annotations

import functools
import time
from typing import Any, Callable

import numpy as np
import psutil


class SystemEvaluator:
    """Collects per-call latency and memory snapshots for any callable.

    Can be used directly or as a decorator::

        eval = SystemEvaluator()

        @eval.measure("my_func")
        def my_func(): ...

        for _ in range(100):
            my_func()

        report = eval.summary()
    """

    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.memory_snapshots: list[int] = []
        self._names: list[str] = []

    def measure(self, name: str = "") -> Callable:
        """Decorator factory — wraps a callable to record latency and memory."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    self.latencies.append(elapsed)
                    self._names.append(name or func.__name__)
                    try:
                        mem = psutil.Process().memory_info().rss
                        self.memory_snapshots.append(mem)
                    except Exception:
                        self.memory_snapshots.append(0)

            return wrapper

        return decorator

    def record(self, latency_ms: float, name: str = "") -> None:
        """Manually record a latency value (useful when the decorator cannot be
        applied directly)."""
        self.latencies.append(latency_ms)
        self._names.append(name)
        try:
            mem = psutil.Process().memory_info().rss
            self.memory_snapshots.append(mem)
        except Exception:
            self.memory_snapshots.append(0)

    def summary(self) -> dict[str, Any]:
        """Compute aggregate statistics over all recorded calls.

        Returns:
            Dict with keys ``count``, ``mean_latency_ms``,
            ``p50_latency_ms``, ``p95_latency_ms``, ``p99_latency_ms``,
            ``min_latency_ms``, ``max_latency_ms``, ``fps``,
            ``mean_rss_mb``, ``peak_rss_mb``.
        """
        if not self.latencies:
            return {
                "count": 0, "mean_latency_ms": 0.0,
                "p50_latency_ms": 0.0, "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0, "min_latency_ms": 0.0,
                "max_latency_ms": 0.0, "fps": 0.0,
                "mean_rss_mb": 0.0, "peak_rss_mb": 0.0,
            }

        arr = sorted(self.latencies)
        n = len(arr)
        mean_ms = float(np.mean(arr))
        p50 = float(arr[int(n * 0.50)])
        p95 = float(arr[int(n * 0.95)])
        p99 = float(arr[int(n * 0.99)])
        min_ms = float(arr[0])
        max_ms = float(arr[-1])
        mean_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        mem_arr = [m for m in self.memory_snapshots if m > 0]
        mean_mem_mb = float(np.mean(mem_arr)) / (1024 * 1024) if mem_arr else 0.0
        peak_mem_mb = float(max(mem_arr)) / (1024 * 1024) if mem_arr else 0.0

        return {
            "count": n,
            "mean_latency_ms": round(mean_ms, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "min_latency_ms": round(min_ms, 2),
            "max_latency_ms": round(max_ms, 2),
            "fps": round(mean_fps, 1),
            "mean_rss_mb": round(mean_mem_mb, 2),
            "peak_rss_mb": round(peak_mem_mb, 2),
        }
