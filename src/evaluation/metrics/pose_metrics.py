from __future__ import annotations

from typing import Any

import numpy as np


COCO_SIGMAS: np.ndarray = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35,
    0.79, 0.79, 0.72, 0.72, 0.62, 0.62,
    1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
], dtype=np.float64)


def compute_pck(
    predicted_keypoints: np.ndarray,
    ground_truth_keypoints: np.ndarray,
    bbox_diagonal: float,
    threshold: float = 0.05,
    visibility_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if bbox_diagonal <= 0:
        return {"pck": 0.0, "correct": 0, "total": 0}

    distances = np.sqrt(
        np.sum((predicted_keypoints - ground_truth_keypoints) ** 2, axis=1)
    )

    if visibility_mask is not None:
        visible = visibility_mask.astype(bool)
        distances = distances[visible]
    else:
        visible = np.ones(len(distances), dtype=bool)

    if len(distances) == 0:
        return {"pck": 0.0, "correct": 0, "total": 0}

    correct = int(np.sum(distances <= threshold * bbox_diagonal))
    total = int(len(distances))
    return {
        "pck": float(correct) / float(total) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


def compute_pck_at_thresholds(
    predicted_keypoints: np.ndarray,
    ground_truth_keypoints: np.ndarray,
    bbox_diagonal: float,
    thresholds: list[float] | None = None,
    visibility_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2]
    results: dict[str, Any] = {}
    for t in thresholds:
        key = f"PCK@{t:.2f}"
        results[key] = compute_pck(
            predicted_keypoints, ground_truth_keypoints,
            bbox_diagonal, threshold=t,
            visibility_mask=visibility_mask,
        )["pck"]
    return results


def compute_oks(
    predicted_keypoints: np.ndarray,
    ground_truth_keypoints: np.ndarray,
    bbox_area: float,
    visibility_mask: np.ndarray,
    sigmas: np.ndarray | None = None,
) -> float:
    if sigmas is None:
        sigmas = COCO_SIGMAS
    if len(sigmas) != predicted_keypoints.shape[0]:
        sigmas = np.full(predicted_keypoints.shape[0], 0.1, dtype=np.float64)

    visible = visibility_mask.astype(bool)
    if not visible.any():
        return 0.0

    pred_vis = predicted_keypoints[visible]
    gt_vis = ground_truth_keypoints[visible]
    sigmas_vis = sigmas[visible]

    d2 = np.sum((pred_vis - gt_vis) ** 2, axis=1)
    vars = (2.0 * sigmas_vis) ** 2
    s2 = max(bbox_area, 1.0)

    oks_val = np.exp(-d2 / (2.0 * s2 * vars))
    return float(np.mean(oks_val))
