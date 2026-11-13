from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.schemas.trajectory import EvaluationStatus, PredictionRecord


def calculate_trajectory_metrics(
    records: list[PredictionRecord],
    success_threshold_pixels: float = 20.0,
) -> dict[str, Any]:
    """Compute comprehensive trajectory prediction evaluation metrics.

    Args:
        records: List of all stored PredictionRecords.
        success_threshold_pixels: Maximum endpoint error in pixels to consider
            a prediction successful in threshold-based success rate.

    Returns:
        Structured dictionary with counts, aggregate displacement errors,
        rates, and breakdowns by object class and prediction horizon.
    """
    total = len(records)
    if total == 0:
        return {
            "total_predictions": 0,
            "matched_predictions": 0,
            "unmatched_predictions": 0,
            "track_lost_predictions": 0,
            "missing_predictions": 0,
            "invalid_predictions": 0,
            "pending_predictions": 0,
            "matching_success_rate_pct": None,
            "threshold_success_rate_pct": None,
            "success_threshold_pixels": success_threshold_pixels,
            "mean_endpoint_error_px": None,
            "median_endpoint_error_px": None,
            "min_endpoint_error_px": None,
            "max_endpoint_error_px": None,
            "std_endpoint_error_px": None,
            "mean_ade_px": None,
            "mean_fde_px": None,
            "by_class": {},
            "by_horizon": {},
            "sample_counts": {"total": 0, "matched": 0},
        }

    matched = [r for r in records if r.status == EvaluationStatus.MATCHED]
    unmatched = [r for r in records if r.status != EvaluationStatus.MATCHED and r.status != EvaluationStatus.PENDING]
    track_lost = [r for r in records if r.status == EvaluationStatus.TRACK_LOST]
    missing = [r for r in records if r.status == EvaluationStatus.MISSING]
    invalid = [r for r in records if r.status == EvaluationStatus.INVALID]
    pending = [r for r in records if r.status == EvaluationStatus.PENDING]

    matching_success_rate = (len(matched) / (total - len(pending)) * 100.0) if (total - len(pending)) > 0 else 0.0

    endpoint_errors = [
        r.endpoint_error for r in matched if r.endpoint_error is not None
    ]
    ades = [r.ade for r in matched if r.ade is not None]
    fdes = [r.fde for r in matched if r.fde is not None]

    if endpoint_errors:
        mean_endpoint = float(np.mean(endpoint_errors))
        median_endpoint = float(np.median(endpoint_errors))
        min_endpoint = float(np.min(endpoint_errors))
        max_endpoint = float(np.max(endpoint_errors))
        std_endpoint = float(np.std(endpoint_errors, ddof=1)) if len(endpoint_errors) > 1 else 0.0
        within_thresh = sum(1 for e in endpoint_errors if e <= success_threshold_pixels)
        threshold_success_rate = (within_thresh / len(endpoint_errors)) * 100.0
    else:
        mean_endpoint = None
        median_endpoint = None
        min_endpoint = None
        max_endpoint = None
        std_endpoint = None
        threshold_success_rate = None

    mean_ade = float(np.mean(ades)) if ades else None
    mean_fde = float(np.mean(fdes)) if fdes else None

    # Breakdown by object class
    classes = sorted(set(r.class_name for r in records))
    by_class: dict[str, dict[str, Any]] = {}
    for cls in classes:
        cls_records = [r for r in records if r.class_name == cls]
        cls_matched = [r for r in cls_records if r.status == EvaluationStatus.MATCHED]
        cls_errors = [r.endpoint_error for r in cls_matched if r.endpoint_error is not None]
        cls_ades = [r.ade for r in cls_matched if r.ade is not None]
        by_class[cls] = {
            "total": len(cls_records),
            "matched": len(cls_matched),
            "match_rate_pct": (len(cls_matched) / len(cls_records) * 100.0) if cls_records else 0.0,
            "mean_endpoint_error_px": float(np.mean(cls_errors)) if cls_errors else None,
            "median_endpoint_error_px": float(np.median(cls_errors)) if cls_errors else None,
            "mean_ade_px": float(np.mean(cls_ades)) if cls_ades else None,
        }

    # Breakdown by prediction horizon
    horizons = sorted(set(round(r.horizon_seconds, 2) for r in records))
    by_horizon: dict[float, dict[str, Any]] = {}
    for h in horizons:
        h_records = [r for r in records if round(r.horizon_seconds, 2) == h]
        h_matched = [r for r in h_records if r.status == EvaluationStatus.MATCHED]
        h_errors = [r.endpoint_error for r in h_matched if r.endpoint_error is not None]
        h_ades = [r.ade for r in h_matched if r.ade is not None]
        by_horizon[h] = {
            "total": len(h_records),
            "matched": len(h_matched),
            "match_rate_pct": (len(h_matched) / len(h_records) * 100.0) if h_records else 0.0,
            "mean_endpoint_error_px": float(np.mean(h_errors)) if h_errors else None,
            "median_endpoint_error_px": float(np.median(h_errors)) if h_errors else None,
            "mean_ade_px": float(np.mean(h_ades)) if h_ades else None,
        }

    return {
        "total_predictions": total,
        "matched_predictions": len(matched),
        "unmatched_predictions": len(unmatched),
        "track_lost_predictions": len(track_lost),
        "missing_predictions": len(missing),
        "invalid_predictions": len(invalid),
        "pending_predictions": len(pending),
        "matching_success_rate_pct": round(matching_success_rate, 2) if matching_success_rate is not None else None,
        "threshold_success_rate_pct": round(threshold_success_rate, 2) if threshold_success_rate is not None else None,
        "success_threshold_pixels": success_threshold_pixels,
        "mean_endpoint_error_px": round(mean_endpoint, 3) if mean_endpoint is not None else None,
        "median_endpoint_error_px": round(median_endpoint, 3) if median_endpoint is not None else None,
        "min_endpoint_error_px": round(min_endpoint, 3) if min_endpoint is not None else None,
        "max_endpoint_error_px": round(max_endpoint, 3) if max_endpoint is not None else None,
        "std_endpoint_error_px": round(std_endpoint, 3) if std_endpoint is not None else None,
        "mean_ade_px": round(mean_ade, 3) if mean_ade is not None else None,
        "mean_fde_px": round(mean_fde, 3) if mean_fde is not None else None,
        "by_class": by_class,
        "by_horizon": by_horizon,
        "sample_counts": {
            "total": total,
            "matched": len(matched),
            "valid_errors": len(endpoint_errors),
        },
    }
