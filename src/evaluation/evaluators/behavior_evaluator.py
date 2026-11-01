"""Behavior ground-truth JSON schema.

.. code-block:: json

    {
        "frames": [
            {
                "frame_id": 0,
                "timestamp": 0.0,
                "annotations": [
                    {
                        "track_id": 1,
                        "entity_type": "human",
                        "primary_label": "human_crossing_road",
                        "crossing_probability": 0.85,
                        "bbox": [100, 200, 300, 500]
                    }
                ]
            }
        ]
    }

- ``frame_id``: int, matches ``PipelineContext.frame_id``.
- ``timestamp``: float, seconds.
- ``annotations``: list of ground-truth behavior annotations per frame.
- ``track_id``: int, unique per track across frames.
- ``entity_type``: ``"human"`` or ``"animal"``.
- ``primary_label``: string matching a ``BehaviorLabel`` enum value.
- ``crossing_probability``: optional float in [0, 1]; present when the
  entity is crossing or approaching the road.
- ``bbox``: optional ``[x1, y1, x2, y2]`` pixel coordinates.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.evaluation.metrics.behavior_metrics import (
    compute_brier_score,
    compute_confusion_matrix,
    compute_per_label_metrics,
    compute_reliability_diagram_data,
)
from src.schemas.behavior import BehaviorFrame, BehaviorLabel


class BehaviorEvaluator:
    def evaluate(
        self,
        predictions: list[BehaviorFrame | None],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        gt_by_frame = self._index_ground_truth(ground_truth)

        pred_labels: list[str] = []
        gt_labels: list[str] = []
        pred_cp: list[float] = []
        gt_cp: list[float] = []
        matched_count = 0
        unmatched_pred_count = 0
        unmatched_gt_count = 0

        for pred_frame in predictions:
            if pred_frame is None:
                continue

            frame_id = pred_frame.frame_id
            frame_gt_list = gt_by_frame.get(frame_id, [])
            gt_by_track: dict[int, dict[str, Any]] = {}
            for ann in frame_gt_list:
                gt_by_track[int(ann["track_id"])] = ann

            used_gt_tracks: set[int] = set()
            for state in pred_frame.states:
                track_id = state.track_id
                if track_id in gt_by_track:
                    gt_ann = gt_by_track[track_id]
                    gt_labels.append(gt_ann["primary_label"])
                    pred_labels.append(state.primary_label.value)

                    if (
                        state.crossing_probability is not None
                        and "crossing_probability" in gt_ann
                        and gt_ann["crossing_probability"] is not None
                    ):
                        pred_cp.append(state.crossing_probability)
                        gt_cp.append(float(gt_ann["crossing_probability"]))

                    used_gt_tracks.add(track_id)
                    matched_count += 1
                else:
                    unmatched_pred_count += 1

            unmatched_gt_count += len(frame_gt_list) - len(used_gt_tracks)

        label_names = sorted(
            set(pred_labels) | set(gt_labels)
        ) if pred_labels or gt_labels else []

        label_metrics = compute_per_label_metrics(pred_labels, gt_labels, label_names)
        confusion = compute_confusion_matrix(pred_labels, gt_labels, label_names)

        brier = 0.0
        reliability = {}
        if pred_cp:
            brier = compute_brier_score(
                np.array(pred_cp, dtype=np.float64),
                np.array(gt_cp, dtype=np.float64),
            )
            reliability = compute_reliability_diagram_data(
                np.array(pred_cp, dtype=np.float64),
                np.array(gt_cp, dtype=np.float64),
                n_bins=10,
            )

        return {
            "label_metrics": label_metrics,
            "confusion_matrix": confusion,
            "brier_score": round(brier, 6),
            "crossing_probability_calibration": reliability,
            "total_matched": matched_count,
            "total_unmatched_predictions": unmatched_pred_count,
            "total_unmatched_ground_truth": unmatched_gt_count,
        }

    @staticmethod
    def _index_ground_truth(
        ground_truth: dict[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        gt_by_frame: dict[int, list[dict[str, Any]]] = {}
        for frame_entry in ground_truth.get("frames", []):
            frame_id = int(frame_entry["frame_id"])
            gt_by_frame[frame_id] = frame_entry.get("annotations", [])
        return gt_by_frame
