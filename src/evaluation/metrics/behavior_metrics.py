from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def compute_confusion_matrix(
    predicted_labels: list[str],
    ground_truth_labels: list[str],
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError(
            f"Length mismatch: {len(predicted_labels)} vs {len(ground_truth_labels)}"
        )

    all_labels: list[str] = sorted(
        set(predicted_labels) | set(ground_truth_labels)
    ) if label_names is None else label_names

    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=np.int64)

    for pred, gt in zip(predicted_labels, ground_truth_labels):
        i = label_to_idx.get(pred, 0)
        j = label_to_idx.get(gt, 0)
        matrix[j, i] += 1

    return {
        "label_names": all_labels,
        "matrix": matrix.tolist(),
        "matrix_shape": list(matrix.shape),
    }


def compute_per_label_metrics(
    predicted_labels: list[str],
    ground_truth_labels: list[str],
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError(
            f"Length mismatch: {len(predicted_labels)} vs {len(ground_truth_labels)}"
        )

    all_labels: list[str] = sorted(
        set(predicted_labels) | set(ground_truth_labels)
    ) if label_names is None else label_names

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for pred, gt in zip(predicted_labels, ground_truth_labels):
        if pred == gt:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    metrics: dict[str, dict[str, float]] = {}
    for label in all_labels:
        t = tp.get(label, 0)
        p = fp.get(label, 0)
        n = fn.get(label, 0)
        precision = float(t) / float(t + p) if (t + p) > 0 else 0.0
        recall = float(t) / float(t + n) if (t + n) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = t + n
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    if all_labels:
        macro_precision = float(np.mean([m["precision"] for m in metrics.values()]))
        macro_recall = float(np.mean([m["recall"] for m in metrics.values()]))
        macro_f1 = float(np.mean([m["f1"] for m in metrics.values()]))
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    return {
        "per_label": metrics,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
    }


def compute_brier_score(
    predicted_probabilities: np.ndarray,
    ground_truth_outcomes: np.ndarray,
) -> float:
    pred = np.asarray(predicted_probabilities, dtype=np.float64)
    gt = np.asarray(ground_truth_outcomes, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: {pred.shape} vs {gt.shape}"
        )
    if pred.size == 0:
        return 0.0
    return float(np.mean((pred - gt) ** 2))


def compute_reliability_diagram_data(
    predicted_probabilities: np.ndarray,
    ground_truth_outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    pred = np.asarray(predicted_probabilities, dtype=np.float64).ravel()
    gt = np.asarray(ground_truth_outcomes, dtype=np.float64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: {pred.shape} vs {gt.shape}"
        )

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (pred >= lo) & (pred < hi)
        if i == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            bin_confidences.append(float(np.mean(pred[mask])))
            bin_accuracies.append(float(np.mean(gt[mask])))
        else:
            bin_confidences.append(float(lo + (hi - lo) / 2))
            bin_accuracies.append(0.0)

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "bin_edges": bins.tolist(),
        "n_bins": n_bins,
        "total_samples": int(len(pred)),
    }
