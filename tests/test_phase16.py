from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
import pytest

from src.evaluation.detection_metrics import DetectionEvaluator
from src.evaluation.depth_metrics import DepthEvaluator
from src.evaluation.profiler import PipelineProfiler
from src.evaluation.system_metrics import SystemEvaluator
from src.evaluation.tracking_metrics import TrackingEvaluator
from src.schemas import BoundingBox, Detection, Frame, TrackedObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    class_name: str,
    x1: float, y1: float, x2: float, y2: float,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _make_tracked(
    track_id: int,
    class_name: str,
    x1: float, y1: float, x2: float, y2: float,
    confidence: float = 0.9,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _make_frame(image: np.ndarray | None = None) -> Frame:
    if image is None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
    return Frame(image=image, frame_id=0, timestamp=0.0, source="test")


# ---------------------------------------------------------------------------
# DetectionEvaluator
# ---------------------------------------------------------------------------

class TestDetectionEvaluator:
    def test_all_correct(self) -> None:
        preds: list[list[Detection]] = [
            [_make_detection("car", 10, 10, 50, 50, 0.95)],
        ]
        gt = {
            "0": [{"class_name": "car", "bbox": [10, 10, 50, 50]}],
        }
        ev = DetectionEvaluator(iou_thresholds=[0.5])
        res = ev.evaluate(preds, gt)
        assert res["mAP@0.5"] > 0.99
        assert res["per_class"]["car"]["recall"] > 0.99

    def test_all_wrong(self) -> None:
        preds: list[list[Detection]] = [
            [_make_detection("car", 0, 0, 10, 10, 0.9)],
        ]
        gt = {
            "0": [{"class_name": "car", "bbox": [100, 100, 150, 150]}],
        }
        ev = DetectionEvaluator(iou_thresholds=[0.5])
        res = ev.evaluate(preds, gt)
        assert res["mAP@0.5"] < 0.1

    def test_empty_predictions(self) -> None:
        ev = DetectionEvaluator(iou_thresholds=[0.5])
        gt = {"0": [{"class_name": "car", "bbox": [10, 10, 50, 50]}]}
        res = ev.evaluate([[]], gt)
        assert res["mAP@0.5"] == 0.0

    def test_empty_ground_truth(self) -> None:
        preds = [[_make_detection("car", 10, 10, 50, 50, 0.9)]]
        ev = DetectionEvaluator(iou_thresholds=[0.5])
        res = ev.evaluate(preds, {})
        assert res["mAP@0.5"] == 0.0

    def test_multiple_classes(self) -> None:
        preds: list[list[Detection]] = [
            [
                _make_detection("car", 10, 10, 50, 50, 0.95),
                _make_detection("person", 60, 60, 90, 90, 0.85),
            ],
        ]
        gt = {
            "0": [
                {"class_name": "car", "bbox": [10, 10, 50, 50]},
                {"class_name": "person", "bbox": [60, 60, 90, 90]},
            ],
        }
        ev = DetectionEvaluator(iou_thresholds=[0.5])
        res = ev.evaluate(preds, gt)
        assert res["mAP@0.5"] > 0.99
        assert "car" in res["per_class"]
        assert "person" in res["per_class"]

    def test_coco_format_gt(self) -> None:
        preds: list[list[Detection]] = [
            [_make_detection("car", 10, 10, 50, 50, 0.95)],
        ]
        gt = {
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [
                {"image_id": 0, "category_id": 1,
                 "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
            ],
        }
        ev = DetectionEvaluator(iou_thresholds=[0.5])
        res = ev.evaluate(preds, gt)
        assert res["mAP@0.5"] > 0.9

    def test_mAP_05_095(self) -> None:
        preds: list[list[Detection]] = [
            [_make_detection("car", 10, 10, 50, 50, 0.95)],
        ]
        gt = {
            "0": [{"class_name": "car", "bbox": [10, 10, 50, 50]}],
        }
        ev = DetectionEvaluator(iou_thresholds=[0.5, 0.75])
        res = ev.evaluate(preds, gt)
        assert "mAP@0.5:0.95" in res
        assert res["mAP@0.5:0.95"] >= 0.0


# ---------------------------------------------------------------------------
# TrackingEvaluator
# ---------------------------------------------------------------------------

class TestTrackingEvaluator:
    def test_perfect_tracking(self) -> None:
        preds: list[list[TrackedObject]] = [
            [_make_tracked(1, "car", 10, 10, 50, 50)],
            [_make_tracked(1, "car", 15, 15, 55, 55)],
        ]
        gt_records = [
            {"frame": 0, "id": 1, "x": 10, "y": 10, "w": 40, "h": 40, "conf": 1.0},
            {"frame": 1, "id": 1, "x": 15, "y": 15, "w": 40, "h": 40, "conf": 1.0},
        ]
        ev = TrackingEvaluator()
        try:
            res = ev.evaluate(preds, gt_records)
        except ImportError:
            pytest.skip("motmetrics not installed")

        assert res["MOTA"] >= 0.9
        assert res["IDF1"] >= 0.9
        assert res["FP"] == 0

    def test_empty_predictions(self) -> None:
        preds: list[list[TrackedObject]] = [[], []]
        gt_records = [
            {"frame": 0, "id": 1, "x": 10, "y": 10, "w": 40, "h": 40, "conf": 1.0},
        ]
        ev = TrackingEvaluator()
        try:
            res = ev.evaluate(preds, gt_records)
        except ImportError:
            pytest.skip("motmetrics not installed")
        assert isinstance(res, dict)
        assert "MOTA" in res

    def test_mot_file_format(self) -> None:
        preds: list[list[TrackedObject]] = [
            [_make_tracked(1, "car", 10, 10, 50, 50)],
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("0,1,10,10,40,40,1.0,-1,-1,-1\n")
            mot_path = f.name

        ev = TrackingEvaluator()
        try:
            res = ev.evaluate(preds, mot_path)
        except ImportError:
            os.unlink(mot_path)
            pytest.skip("motmetrics not installed")

        os.unlink(mot_path)
        assert isinstance(res, dict)
        assert "MOTA" in res

    def test_summary_table(self) -> None:
        ev = TrackingEvaluator()
        sample = {
            "MOTA": 0.75, "MOTP": 0.8, "IDF1": 0.7,
            "MT": 2, "ML": 1, "num_switches": 0,
            "fragmentations": 1, "FP": 1, "FN": 2,
            "num_frames": 10, "num_objects": 5, "num_predictions": 4,
        }
        table = ev.summary_table(sample)
        assert "MOTA" in table
        assert "0.7500" in table


# ---------------------------------------------------------------------------
# DepthEvaluator
# ---------------------------------------------------------------------------

class TestDepthEvaluator:
    def test_temporal_consistency_identical(self) -> None:
        depth = np.ones((50, 50), dtype=np.float32) * 5.0
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        frames = [_make_frame(img), _make_frame(img)]
        ev = DepthEvaluator(mode="temporal_consistency")
        res = ev.evaluate(frames, [depth, depth])
        assert res["frames_evaluated"] == 1
        assert res["temporal_consistency_mae"] < 0.001

    def test_temporal_consistency_different(self) -> None:
        d1 = np.ones((50, 50), dtype=np.float32) * 5.0
        d2 = np.ones((50, 50), dtype=np.float32) * 10.0
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        frames = [_make_frame(img), _make_frame(img)]
        ev = DepthEvaluator(mode="temporal_consistency")
        res = ev.evaluate(frames, [d1, d2])
        assert res["frames_evaluated"] == 1

    def test_gt_metrics_perfect(self) -> None:
        depth = np.ones((20, 20), dtype=np.float32) * 5.0
        gt = np.ones((20, 20), dtype=np.float32) * 5.0
        ev = DepthEvaluator(mode="gt")
        res = ev.evaluate(
            [_make_frame(), _make_frame()],
            [depth, depth],
            [gt, gt],
        )
        assert res["AbsRel"] == 0.0
        assert res["RMSE"] == 0.0
        assert res["delta_1.25"] == 1.0

    def test_gt_metrics_with_error(self) -> None:
        pred = np.ones((20, 20), dtype=np.float32) * 5.0
        gt = np.ones((20, 20), dtype=np.float32) * 10.0
        ev = DepthEvaluator(mode="gt")
        res = ev.evaluate(
            [_make_frame()], [pred], [gt],
        )
        assert res["AbsRel"] == 0.5
        assert res["RMSE"] == 5.0

    def test_gt_metrics_masked(self) -> None:
        pred = np.ones((20, 20), dtype=np.float32) * 5.0
        gt = np.zeros((20, 20), dtype=np.float32)
        gt[5:15, 5:15] = 10.0
        ev = DepthEvaluator(mode="gt")
        res = ev.evaluate(
            [_make_frame()], [pred], [gt],
        )
        assert res["frames_evaluated"] == 1
        assert res["AbsRel"] > 0.0

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown depth evaluation mode"):
            DepthEvaluator(mode="invalid")


# ---------------------------------------------------------------------------
# SystemEvaluator
# ---------------------------------------------------------------------------

class TestSystemEvaluator:
    def test_decorator_records_latency(self) -> None:
        ev = SystemEvaluator()

        @ev.measure("test_func")
        def dummy() -> None:
            pass

        for _ in range(10):
            dummy()

        s = ev.summary()
        assert s["count"] == 10
        assert s["mean_latency_ms"] >= 0

    def test_manual_record(self) -> None:
        ev = SystemEvaluator()
        ev.record(5.0, "step1")
        ev.record(10.0, "step2")
        s = ev.summary()
        assert s["count"] == 2
        assert s["mean_latency_ms"] == 7.5
        assert s["p50_latency_ms"] == 10.0

    def test_empty_summary(self) -> None:
        ev = SystemEvaluator()
        s = ev.summary()
        assert s["count"] == 0
        assert s["mean_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# PipelineProfiler (unit-level smoke test)
# ---------------------------------------------------------------------------

class TestPipelineProfiler:
    def test_profiler_can_be_instantiated(self) -> None:
        from src.core.pipeline import PerceptionPipeline
        pipe = PerceptionPipeline({})
        prof = PipelineProfiler(pipe)
        assert isinstance(prof, PipelineProfiler)

    def test_summary_table(self) -> None:
        prof = PipelineProfiler.__new__(PipelineProfiler)
        prof.stage_times = {
            "detector.detect": [10.0, 12.0, 8.0],
            "tracker.update": [5.0, 6.0],
        }
        prof._originals = {}
        s = prof.summary()
        assert "detector.detect" in s
        assert s["detector.detect"]["mean_ms"] == 10.0
        assert s["tracker.update"]["p50_ms"] == 6.0
        table = prof.summary_table(s)
        assert "detector.detect" in table
        assert "TOTAL" in table

    def test_empty_profiler_summary(self) -> None:
        prof = PipelineProfiler.__new__(PipelineProfiler)
        prof.stage_times = {}
        prof._originals = {}
        s = prof.summary()
        assert "_total_all_ms" in s


# ---------------------------------------------------------------------------
# Runner (import smoke test)
# ---------------------------------------------------------------------------

class TestRunner:
    def test_importable(self) -> None:
        import src.evaluation.runner  # noqa: PLC0415, F401
        assert True
