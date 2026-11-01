from __future__ import annotations

import json

import pytest

from src.evaluation.evaluators.behavior_evaluator import BehaviorEvaluator
from src.schemas.behavior import BehaviorFrame, BehaviorLabel, BehaviorState


def _make_behavior_frame(
    frame_id: int,
    states: list[tuple[int, str, str, float, float | None]],
) -> BehaviorFrame:
    behavior_states: list[BehaviorState] = []
    for track_id, entity_type, label_str, confidence, cp in states:
        label = BehaviorLabel(label_str)
        behavior_states.append(BehaviorState(
            track_id=track_id,
            entity_type=entity_type,
            primary_label=label,
            secondary_labels=(),
            confidence=confidence,
            crossing_probability=cp,
            timestamp=float(frame_id) * 0.033,
        ))
    return BehaviorFrame(
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.033,
        states=tuple(behavior_states),
    )


def _make_gt(entries: list[dict]) -> dict:
    frames_map: dict[int, list[dict]] = {}
    for entry in entries:
        fid = entry["frame_id"]
        if fid not in frames_map:
            frames_map[fid] = []
        ann = {
            "track_id": entry["track_id"],
            "entity_type": entry.get("entity_type", "human"),
            "primary_label": entry["label"],
            "crossing_probability": entry.get("crossing_probability"),
        }
        if "bbox" in entry:
            ann["bbox"] = entry["bbox"]
        frames_map[fid].append(ann)

    frames_list = [
        {"frame_id": fid, "timestamp": float(fid) * 0.033, "annotations": anns}
        for fid, anns in sorted(frames_map.items())
    ]
    return {"frames": frames_list}


class TestBehaviorEvaluator:
    def test_perfect_match(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_walking", 0.95, None),
            (2, "animal", "animal_standing", 0.90, None),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking"},
            {"frame_id": 0, "track_id": 2, "label": "animal_standing"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        assert result["label_metrics"]["macro_f1"] == 1.0
        assert result["total_matched"] == 2

    def test_confused_labels(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_crossing_road", 0.80, None),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_standing"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        assert result["label_metrics"]["per_label"]["human_crossing_road"]["precision"] == 0.0
        assert result["label_metrics"]["per_label"]["human_crossing_road"]["recall"] == 0.0
        assert result["label_metrics"]["per_label"]["human_standing"]["recall"] == 0.0
        assert result["total_matched"] == 1

    def test_unmatched_predictions(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_walking", 0.95, None),
            (99, "human", "human_standing", 0.80, None),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        assert result["total_matched"] == 1
        assert result["total_unmatched_predictions"] == 1

    def test_unmatched_ground_truth(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_walking", 0.95, None),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking"},
            {"frame_id": 0, "track_id": 2, "label": "human_standing"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        assert result["total_matched"] == 1
        assert result["total_unmatched_ground_truth"] == 1

    def test_no_predictions(self) -> None:
        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([], {"frames": []})
        assert result["total_matched"] == 0
        assert result["total_unmatched_ground_truth"] == 0

    def test_brier_score_with_cp(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_crossing_road", 0.80, 0.85),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_crossing_road",
             "crossing_probability": 0.90},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        assert result["brier_score"] > 0.0
        assert result["crossing_probability_calibration"]["total_samples"] == 1

    def test_brier_score_skipped_when_no_cp(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_walking", 0.95, None),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        assert result["brier_score"] == 0.0

    def test_multiple_frames(self) -> None:
        frames = [
            _make_behavior_frame(0, [(1, "human", "human_walking", 0.95, None)]),
            _make_behavior_frame(1, [(1, "human", "human_walking", 0.95, None)]),
        ]
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking"},
            {"frame_id": 1, "track_id": 1, "label": "human_walking"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate(frames, gt)
        assert result["total_matched"] == 2

    def test_none_frame_skipped(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_walking", 0.95, None),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking"},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([None, pred_frame, None], gt)
        assert result["total_matched"] == 1

    def test_json_serializable(self) -> None:
        pred_frame = _make_behavior_frame(0, [
            (1, "human", "human_walking", 0.95, 0.8),
        ])
        gt = _make_gt([
            {"frame_id": 0, "track_id": 1, "label": "human_walking",
             "crossing_probability": 0.7},
        ])

        evaluator = BehaviorEvaluator()
        result = evaluator.evaluate([pred_frame], gt)
        json.dumps(result)
