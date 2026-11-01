from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.evaluation.metrics.pose_metrics import (
    compute_oks,
    compute_pck,
    compute_pck_at_thresholds,
)
from src.schemas.pose import AnimalPose, HumanPose, PoseFrame


def _coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _compute_bbox_diagonal(x1: float, y1: float, x2: float, y2: float) -> float:
    w = x2 - x1
    h = y2 - y1
    return float(np.sqrt(w * w + h * h))


def _compute_bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    w = x2 - x1
    h = y2 - y1
    return float(w * h)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class PoseEvaluator:
    def __init__(self, match_iou_threshold: float = 0.5) -> None:
        self.match_iou_threshold = match_iou_threshold

    def evaluate(
        self,
        predictions: list[PoseFrame | None],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        gt_by_frame = self._index_ground_truth(ground_truth)

        human_pck_list: list[float] = []
        human_oks_list: list[float] = []
        animal_pck_list: list[float] = []
        matched_human_count = 0
        matched_animal_count = 0
        unmatched_gt_humans = 0
        unmatched_gt_animals = 0

        seen_frames: set[int] = set()
        for pred_frame in predictions:
            if pred_frame is None:
                continue
            frame_id = pred_frame.frame_id
            seen_frames.add(frame_id)
            gt_anns = gt_by_frame.get(frame_id, [])
            gt_humans = [a for a in gt_anns if a["category_id"] == 1]
            gt_animals = [a for a in gt_anns if a["category_id"] != 1]

            used_gt_human: set[int] = set()
            for human in pred_frame.humans:
                best_idx, best_iou = self._find_best_match(
                    human.bbox, gt_humans, used_gt_human
                )
                if best_idx is None or best_iou < self.match_iou_threshold:
                    continue
                used_gt_human.add(best_idx)
                gt = gt_humans[best_idx]
                gt_kps = np.array(gt["keypoints"], dtype=np.float64).reshape(-1, 3)
                gt_xy = gt_kps[:, :2]
                gt_vis = gt_kps[:, 2] > 0
                pred_xy = self._pose_to_array(human)

                if pred_xy.shape[0] != gt_xy.shape[0]:
                    continue

                gt_bbox = gt["bbox"]
                gt_bbox_xyxy = _coco_bbox_to_xyxy(gt_bbox)
                diag = _compute_bbox_diagonal(*gt_bbox_xyxy)
                area = _compute_bbox_area(*gt_bbox_xyxy)

                pck_result = compute_pck(pred_xy, gt_xy, diag, visibility_mask=gt_vis)
                human_pck_list.append(pck_result["pck"])
                matched_human_count += 1

                oks_val = compute_oks(pred_xy, gt_xy, area, gt_vis)
                human_oks_list.append(oks_val)

            unmatched_gt_humans += len(gt_humans) - len(used_gt_human)

            used_gt_animal: set[int] = set()
            for animal in pred_frame.animals:
                best_idx, best_iou = self._find_best_match(
                    animal.bbox, gt_animals, used_gt_animal
                )
                if best_idx is None or best_iou < self.match_iou_threshold:
                    continue
                used_gt_animal.add(best_idx)
                gt = gt_animals[best_idx]
                gt_kps = np.array(gt["keypoints"], dtype=np.float64).reshape(-1, 3)
                gt_xy = gt_kps[:, :2]
                gt_vis = gt_kps[:, 2] > 0
                pred_xy = self._pose_to_array(animal)

                if pred_xy.shape[0] != gt_xy.shape[0]:
                    continue

                gt_bbox = gt["bbox"]
                gt_bbox_xyxy = _coco_bbox_to_xyxy(gt_bbox)
                diag = _compute_bbox_diagonal(*gt_bbox_xyxy)

                pck_result = compute_pck(pred_xy, gt_xy, diag, visibility_mask=gt_vis)
                animal_pck_list.append(pck_result["pck"])
                matched_animal_count += 1

            unmatched_gt_animals += len(gt_animals) - len(used_gt_animal)

        for frame_id, gt_anns in gt_by_frame.items():
            if frame_id in seen_frames:
                continue
            gt_humans = [a for a in gt_anns if a["category_id"] == 1]
            gt_animals = [a for a in gt_anns if a["category_id"] != 1]
            unmatched_gt_humans += len(gt_humans)
            unmatched_gt_animals += len(gt_animals)

        human_pck_05 = float(np.mean(human_pck_list)) if human_pck_list else 0.0
        human_oks = float(np.mean(human_oks_list)) if human_oks_list else 0.0
        animal_pck_05 = float(np.mean(animal_pck_list)) if animal_pck_list else 0.0

        return {
            "human_pck_at_0.05": round(human_pck_05, 6),
            "human_pck_at_0.05_count": matched_human_count,
            "human_oks": round(human_oks, 6),
            "human_oks_count": matched_human_count,
            "human_unmatched_gt": unmatched_gt_humans,
            "animal_pck_at_0.05": round(animal_pck_05, 6),
            "animal_pck_at_0.05_count": matched_animal_count,
            "animal_unmatched_gt": unmatched_gt_animals,
            "matching_iou_threshold": self.match_iou_threshold,
        }

    def _find_best_match(
        self,
        pred_bbox: tuple[float, float, float, float],
        gt_annotations: list[dict[str, Any]],
        used_indices: set[int],
    ) -> tuple[int | None, float]:
        best_iou = 0.0
        best_idx = None
        for i, gt in enumerate(gt_annotations):
            if i in used_indices:
                continue
            gt_bbox = gt["bbox"]
            gt_xyxy = _coco_bbox_to_xyxy(gt_bbox)
            iou = _bbox_iou(pred_bbox, gt_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx, best_iou

    @staticmethod
    def _pose_to_array(pose: HumanPose | AnimalPose) -> np.ndarray:
        return np.array([[kp.x, kp.y] for kp in pose.keypoints], dtype=np.float64)

    @staticmethod
    def _index_ground_truth(
        ground_truth: dict[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        gt_by_frame: dict[int, list[dict[str, Any]]] = {}
        for ann in ground_truth.get("annotations", []):
            image_id = int(ann["image_id"])
            if image_id not in gt_by_frame:
                gt_by_frame[image_id] = []
            gt_by_frame[image_id].append(ann)
        return gt_by_frame
