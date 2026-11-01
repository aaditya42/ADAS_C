from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics.pose_metrics import (
    COCO_SIGMAS,
    compute_oks,
    compute_pck,
    compute_pck_at_thresholds,
)


class TestComputePCK:
    def test_perfect_match(self) -> None:
        pred = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        result = compute_pck(pred, gt, bbox_diagonal=100.0, threshold=0.05)
        assert result["pck"] == 1.0
        assert result["correct"] == 2
        assert result["total"] == 2

    def test_all_wrong(self) -> None:
        pred = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        gt = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float64)
        result = compute_pck(pred, gt, bbox_diagonal=100.0, threshold=0.05)
        assert result["pck"] == 0.0
        assert result["correct"] == 0
        assert result["total"] == 2

    def test_partial_match(self) -> None:
        pred = np.array([[10.0, 20.0], [500.0, 600.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        result = compute_pck(pred, gt, bbox_diagonal=200.0, threshold=0.05)
        assert result["correct"] == 1
        assert result["total"] == 2
        assert result["pck"] == 0.5

    def test_threshold_edge(self) -> None:
        pred = np.array([[10.0, 10.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0]], dtype=np.float64)
        diag = 200.0
        result_high = compute_pck(pred, gt, diag, threshold=0.1)
        assert result_high["correct"] == 1
        result_low = compute_pck(pred, gt, diag, threshold=0.01)
        assert result_low["correct"] == 0

    def test_visibility_mask(self) -> None:
        pred = np.array([[10.0, 20.0], [0.0, 0.0], [50.0, 60.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0], [100.0, 100.0], [50.0, 60.0]], dtype=np.float64)
        vis = np.array([True, False, True])
        result = compute_pck(pred, gt, bbox_diagonal=200.0, visibility_mask=vis)
        assert result["correct"] == 2
        assert result["total"] == 2

    def test_zero_diagonal(self) -> None:
        pred = np.array([[10.0, 20.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0]], dtype=np.float64)
        result = compute_pck(pred, gt, bbox_diagonal=0.0)
        assert result["pck"] == 0.0

    def test_empty_keypoints(self) -> None:
        pred = np.zeros((0, 2), dtype=np.float64)
        gt = np.zeros((0, 2), dtype=np.float64)
        result = compute_pck(pred, gt, bbox_diagonal=100.0, visibility_mask=np.array([], dtype=bool))
        assert result["pck"] == 0.0


class TestComputeOKS:
    def test_perfect_match(self) -> None:
        pred = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        vis = np.array([True, True])
        oks = compute_oks(pred, gt, bbox_area=10000.0, visibility_mask=vis)
        assert oks == pytest.approx(1.0)

    def test_visible_only(self) -> None:
        pred = np.array([[10.0, 20.0], [0.0, 0.0]], dtype=np.float64)
        gt = np.array([[10.0, 20.0], [100.0, 100.0]], dtype=np.float64)
        vis = np.array([True, False])
        oks = compute_oks(pred, gt, bbox_area=10000.0, visibility_mask=vis)
        assert oks == pytest.approx(1.0)

    def test_no_visible(self) -> None:
        pred = np.array([[10.0, 20.0]], dtype=np.float64)
        gt = np.array([[50.0, 60.0]], dtype=np.float64)
        vis = np.array([False])
        oks = compute_oks(pred, gt, bbox_area=10000.0, visibility_mask=vis)
        assert oks == 0.0

    def known_error_small_bbox(self) -> None:
        pred = np.array([[10.0, 10.0]], dtype=np.float64)
        gt = np.array([[12.0, 12.0]], dtype=np.float64)
        vis = np.array([True])
        oks_small = compute_oks(pred, gt, bbox_area=1.0, visibility_mask=vis)
        oks_large = compute_oks(pred, gt, bbox_area=10000.0, visibility_mask=vis)
        assert oks_small < oks_large

    def test_custom_sigmas(self) -> None:
        pred = np.array([[10.0, 10.0]], dtype=np.float64)
        gt = np.array([[15.0, 15.0]], dtype=np.float64)
        vis = np.array([True])
        narrow = compute_oks(pred, gt, 100.0, vis, sigmas=np.array([0.01]))
        wide = compute_oks(pred, gt, 100.0, vis, sigmas=np.array([1.0]))
        assert narrow < wide

    def test_default_sigmas_length_mismatch(self) -> None:
        pred = np.zeros((5, 2), dtype=np.float64)
        gt = np.zeros((5, 2), dtype=np.float64)
        vis = np.ones(5, dtype=bool)
        oks = compute_oks(pred, gt, 1000.0, vis)
        assert oks == pytest.approx(1.0)

    def test_realistic_oks(self) -> None:
        rng = np.random.default_rng(42)
        pred = np.array([
            [100.0, 120.0], [105.0, 115.0], [95.0, 118.0],
            [110.0, 100.0], [90.0, 102.0],
            [100.0, 80.0], [100.0, 82.0],
            [95.0, 60.0], [105.0, 58.0],
            [92.0, 40.0], [108.0, 38.0],
            [98.0, 90.0], [102.0, 88.0],
            [96.0, 110.0], [104.0, 112.0],
            [94.0, 130.0], [106.0, 128.0],
        ], dtype=np.float64)
        gt = pred + rng.normal(0, 2.0, pred.shape)
        vis = np.ones(17, dtype=bool)
        oks = compute_oks(pred, gt, 3000.0, vis, sigmas=COCO_SIGMAS)
        assert 0.0 < oks < 1.0


class TestComputePCKAtThresholds:
    def test_multiple_thresholds(self) -> None:
        pred = np.array([[10.0, 10.0], [100.0, 100.0]], dtype=np.float64)
        gt = np.array([[10.0, 10.0], [102.0, 102.0]], dtype=np.float64)
        results = compute_pck_at_thresholds(
            pred, gt, bbox_diagonal=200.0,
            thresholds=[0.01, 0.02, 0.05],
        )
        assert "PCK@0.01" in results
        assert "PCK@0.02" in results
        assert "PCK@0.05" in results
        assert results["PCK@0.05"] == 1.0

    def test_default_thresholds(self) -> None:
        pred = np.array([[10.0, 10.0]], dtype=np.float64)
        gt = np.array([[10.0, 10.0]], dtype=np.float64)
        results = compute_pck_at_thresholds(pred, gt, bbox_diagonal=100.0)
        assert len(results) == 4
