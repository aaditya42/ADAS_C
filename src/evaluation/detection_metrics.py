from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.schemas import BoundingBox, Detection


class DetectionEvaluator:
    """Evaluate object detection against ground truth.

    Supports self-contained VOC-style AP computation (no pycocotools
    required) across configurable IoU thresholds.
    """

    def __init__(self, iou_thresholds: list[float] | None = None) -> None:
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5, 0.75]

    def evaluate(
        self,
        predictions: list[list[Detection]],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute detection metrics.

        Args:
            predictions: Per-frame list of detections.
            ground_truth: Dict with per-frame GT. Supported formats:

                COCO-style::
                    {
                        "categories": [{"id": 1, "name": "person"}, ...],
                        "annotations": [
                            {"image_id": 0, "category_id": 1,
                             "bbox": [x, y, w, h], "iscrowd": 0},
                            ...
                        ],
                    }

                Per-frame simple::
                    {
                        "0": [{"class_name": "car", "bbox": [x1, y1, x2, y2]}, ...],
                        ...
                    }

        Returns:
            Dict with keys ``mAP@0.5``, ``mAP@0.5:0.95``, ``per_class``.
        """
        gt_per_frame, cat_name_from_id = self._normalise_gt(ground_truth)

        class_names: set[str] = set()
        for dets in predictions:
            for d in dets:
                class_names.add(d.class_name)
        for frame_gt in gt_per_frame.values():
            for g in frame_gt:
                class_names.add(g["class_name"])
        class_names = {c for c in class_names if c != "__background__"}
        if not class_names:
            return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "per_class": {}}

        n_frames = max(len(predictions), len(gt_per_frame))

        results: dict[str, Any] = {"per_class": {}}
        for iou_thr in self.iou_thresholds:
            aps: list[float] = []
            for cls_name in sorted(class_names):
                ap = self._compute_ap(predictions, gt_per_frame, cls_name,
                                      iou_thr, cat_name_from_id, n_frames)
                aps.append(ap)
                cls_key = f"AP@{iou_thr}"
                if cls_name not in results["per_class"]:
                    results["per_class"][cls_name] = {}
                results["per_class"][cls_name][cls_key] = round(float(ap), 4)

            key = f"mAP@{iou_thr}"
            results[key] = round(float(np.mean(aps)) if aps else 0.0, 4)

        aps_05 = [results["per_class"][c].get("AP@0.5", 0.0)
                  for c in sorted(class_names)]

        results["mAP@0.5"] = results.get("mAP@0.5", 0.0)

        iou_05_95 = [t for t in self.iou_thresholds if 0.5 <= t <= 0.95]
        if len(iou_05_95) > 1:
            aps_all: list[float] = []
            for iou_thr in iou_05_95:
                for cls_name in sorted(class_names):
                    ap = self._compute_ap(predictions, gt_per_frame, cls_name,
                                          iou_thr, cat_name_from_id, n_frames)
                    aps_all.append(ap)
            results["mAP@0.5:0.95"] = round(float(np.mean(aps_all)) if aps_all else 0.0, 4)
        else:
            results["mAP@0.5:0.95"] = round(float(np.mean(aps_05)) if aps_05 else 0.0, 4)

        for cls_name in sorted(class_names):
            prec, rec = self._compute_precision_recall(
                predictions, gt_per_frame, cls_name, 0.5,
                cat_name_from_id, n_frames
            )
            results["per_class"][cls_name]["precision"] = round(float(prec), 4)
            results["per_class"][cls_name]["recall"] = round(float(rec), 4)

        return results

    def _normalise_gt(
        self, gt: dict[str, Any]
    ) -> tuple[dict[int, list[dict[str, Any]]], dict[int, str]]:
        gt_per_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)

        if "annotations" in gt:
            categories = {c["id"]: c["name"] for c in gt.get("categories", [])}
            cat_id_to_name: dict[int, str] = {}
            for ann in gt["annotations"]:
                image_id = int(ann["image_id"])
                cat_id = int(ann["category_id"])
                cat_id_to_name[cat_id] = categories.get(cat_id, str(cat_id))
                bbox_coco = ann["bbox"]
                gt_per_frame[image_id].append({
                    "class_name": categories.get(cat_id, str(cat_id)),
                    "bbox": BoundingBox(
                        x1=float(bbox_coco[0]),
                        y1=float(bbox_coco[1]),
                        x2=float(bbox_coco[0] + bbox_coco[2]),
                        y2=float(bbox_coco[1] + bbox_coco[3]),
                    ),
                    "iscrowd": ann.get("iscrowd", 0),
                })
            return gt_per_frame, categories
        else:
            cat_id_to_name = {0: "object"}
            for frame_key, anns in gt.items():
                frame_id = int(frame_key) if isinstance(frame_key, str) else frame_key
                if isinstance(anns, list):
                    for ann in anns:
                        cls_name = ann.get("class_name", "object")
                        if "bbox" in ann:
                            bb = ann["bbox"]
                            if len(bb) == 4:
                                gt_per_frame[frame_id].append({
                                    "class_name": cls_name,
                                    "bbox": BoundingBox(
                                        x1=float(bb[0]), y1=float(bb[1]),
                                        x2=float(bb[2]), y2=float(bb[3]),
                                    ),
                                })
            return gt_per_frame, cat_id_to_name

    def _compute_ap(
        self,
        predictions: list[list[Detection]],
        gt_per_frame: dict[int, list[dict[str, Any]]],
        class_name: str,
        iou_threshold: float,
        cat_name_from_id: dict[int, str],
        n_frames: int,
    ) -> float:
        detections: list[tuple[float, bool, int]] = []

        matched_gt: dict[int, set[int]] = defaultdict(set)

        for frame_id in range(n_frames):
            preds = predictions[frame_id] if frame_id < len(predictions) else []
            cls_dets = [d for d in preds if d.class_name == class_name]
            cls_gt = gt_per_frame.get(frame_id, [])
            cls_gt_annotations = [g for g in cls_gt
                                  if g["class_name"] == class_name]

            gt_used: set[int] = set()

            for det in sorted(cls_dets, key=lambda x: x.confidence, reverse=True):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_ann in enumerate(cls_gt_annotations):
                    if gt_idx in gt_used:
                        continue
                    iou_val = det.bbox.iou(gt_ann["bbox"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx

                is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
                if is_tp:
                    gt_used.add(best_gt_idx)
                    detections.append((det.confidence, True, frame_id))
                else:
                    detections.append((det.confidence, False, frame_id))

        detections.sort(key=lambda x: x[0], reverse=True)

        n_gt = sum(
            1 for frame_gt in gt_per_frame.values()
            for g in frame_gt if g["class_name"] == class_name
        )

        if n_gt == 0 or not detections:
            return 0.0

        tp = np.array([1 if d[1] else 0 for d in detections], dtype=np.float64)
        fp = np.array([0 if d[1] else 1 for d in detections], dtype=np.float64)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / n_gt
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

        ap = self._voc_ap(rec, prec)
        return ap

    def _compute_precision_recall(
        self,
        predictions: list[list[Detection]],
        gt_per_frame: dict[int, list[dict[str, Any]]],
        class_name: str,
        iou_threshold: float,
        cat_name_from_id: dict[int, str],
        n_frames: int,
    ) -> tuple[float, float]:
        detections: list[tuple[float, bool, int]] = []
        for frame_id in range(n_frames):
            preds = predictions[frame_id] if frame_id < len(predictions) else []
            cls_dets = [d for d in preds if d.class_name == class_name]
            cls_gt = gt_per_frame.get(frame_id, [])
            cls_gt_annotations = [g for g in cls_gt
                                  if g["class_name"] == class_name]

            gt_used: set[int] = set()
            for det in sorted(cls_dets, key=lambda x: x.confidence, reverse=True):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_ann in enumerate(cls_gt_annotations):
                    if gt_idx in gt_used:
                        continue
                    iou_val = det.bbox.iou(gt_ann["bbox"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx
                is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
                if is_tp:
                    gt_used.add(best_gt_idx)
                    detections.append((det.confidence, True, frame_id))
                else:
                    detections.append((det.confidence, False, frame_id))

        detections.sort(key=lambda x: x[0], reverse=True)

        n_gt = sum(
            1 for frame_gt in gt_per_frame.values()
            for g in frame_gt if g["class_name"] == class_name
        )

        if n_gt == 0 or not detections:
            return 0.0, 0.0

        tp = np.array([1 if d[1] else 0 for d in detections], dtype=np.float64)
        fp = np.array([0 if d[1] else 1 for d in detections], dtype=np.float64)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        n_pos = tp_cum[-1]
        precision = n_pos / (tp_cum[-1] + fp_cum[-1]) if (tp_cum[-1] + fp_cum[-1]) > 0 else 0.0
        recall = n_pos / n_gt if n_gt > 0 else 0.0

        return precision, recall

    @staticmethod
    def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        if len(idx) == 0:
            return 0.0
        ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
        return float(ap)
