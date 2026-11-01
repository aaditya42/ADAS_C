from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics.behavior_metrics import (
    compute_brier_score,
    compute_confusion_matrix,
    compute_per_label_metrics,
    compute_reliability_diagram_data,
)


class TestComputeConfusionMatrix:
    def test_perfect_classification(self) -> None:
        pred = ["A", "A", "B", "B"]
        gt = ["A", "A", "B", "B"]
        result = compute_confusion_matrix(pred, gt)
        assert result["matrix_shape"] == [2, 2]
        mat = result["matrix"]
        assert mat[0][0] == 2
        assert mat[1][1] == 2
        assert mat[0][1] == 0
        assert mat[1][0] == 0

    def test_with_confusion(self) -> None:
        pred = ["A", "A", "B", "B"]
        gt = ["A", "B", "A", "B"]
        result = compute_confusion_matrix(pred, gt)
        mat = result["matrix"]
        assert mat[0][0] == 1
        assert mat[0][1] == 1
        assert mat[1][0] == 1
        assert mat[1][1] == 1

    def test_three_classes(self) -> None:
        pred = ["X", "Y", "Z", "X", "Y"]
        gt = ["X", "Y", "Z", "Z", "X"]
        result = compute_confusion_matrix(pred, gt)
        assert result["matrix_shape"] == [3, 3]
        assert result["matrix"][0][0] >= 1

    def test_custom_label_names(self) -> None:
        pred = ["A", "B"]
        gt = ["A", "B"]
        result = compute_confusion_matrix(pred, gt, label_names=["A", "B", "C"])
        assert result["matrix_shape"] == [3, 3]

    def test_empty_lists(self) -> None:
        result = compute_confusion_matrix([], [])
        assert result["matrix_shape"] == [0, 0]
        assert result["matrix"] == []


class TestComputePerLabelMetrics:
    def test_perfect(self) -> None:
        pred = ["A", "A", "B", "B"]
        gt = ["A", "A", "B", "B"]
        result = compute_per_label_metrics(pred, gt)
        assert result["per_label"]["A"]["precision"] == 1.0
        assert result["per_label"]["A"]["recall"] == 1.0
        assert result["per_label"]["A"]["f1"] == 1.0
        assert result["per_label"]["A"]["support"] == 2
        assert result["macro_f1"] == 1.0

    def test_imperfect(self) -> None:
        pred = ["A", "A", "B", "B", "A"]
        gt = ["A", "B", "A", "B", "C"]
        result = compute_per_label_metrics(pred, gt)
        assert result["per_label"]["A"]["precision"] < 1.0
        assert result["per_label"]["C"]["support"] == 1
        assert result["per_label"]["C"]["recall"] == 0.0

    def test_missing_class(self) -> None:
        pred = ["A", "A"]
        gt = ["A", "B"]
        result = compute_per_label_metrics(pred, gt)
        assert result["per_label"]["B"]["precision"] == 0.0
        assert result["per_label"]["B"]["recall"] == 0.0
        assert result["per_label"]["B"]["f1"] == 0.0

    def test_empty_lists(self) -> None:
        result = compute_per_label_metrics([], [])
        assert result["macro_f1"] == 0.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_per_label_metrics(["A"], ["A", "B"])


class TestComputeBrierScore:
    def test_perfect(self) -> None:
        pred = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        gt = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        assert compute_brier_score(pred, gt) == 0.0

    def test_all_wrong(self) -> None:
        pred = np.array([1.0, 0.0], dtype=np.float64)
        gt = np.array([0.0, 1.0], dtype=np.float64)
        assert compute_brier_score(pred, gt) == 1.0

    def known_values(self) -> None:
        pred = np.array([0.8, 0.3, 0.6, 0.1], dtype=np.float64)
        gt = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        expected = ((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2
                    + (0.6 - 1.0) ** 2 + (0.1 - 0.0) ** 2) / 4.0
        assert compute_brier_score(pred, gt) == pytest.approx(expected)

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_brier_score(np.array([1.0]), np.array([1.0, 0.0]))

    def test_empty(self) -> None:
        assert compute_brier_score(np.array([]), np.array([])) == 0.0


class TestComputeReliabilityDiagramData:
    def test_perfect_calibration(self) -> None:
        rng = np.random.default_rng(42)
        pred = rng.uniform(0.0, 1.0, 1000)
        gt = (rng.uniform(0.0, 1.0, 1000) < pred).astype(np.float64)
        result = compute_reliability_diagram_data(pred, gt, n_bins=10)
        assert len(result["bin_confidences"]) == 10
        assert len(result["bin_accuracies"]) == 10
        assert len(result["bin_counts"]) == 10
        assert result["total_samples"] == 1000
        assert sum(result["bin_counts"]) == 1000

    def test_uncalibrated(self) -> None:
        pred = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
        gt = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        result = compute_reliability_diagram_data(pred, gt, n_bins=5)
        assert result["total_samples"] == 8

    def test_empty(self) -> None:
        result = compute_reliability_diagram_data(np.array([]), np.array([]), n_bins=5)
        assert result["total_samples"] == 0
        assert all(c == 0 for c in result["bin_counts"])

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_reliability_diagram_data(np.array([0.5]), np.array([1.0, 0.0]))
