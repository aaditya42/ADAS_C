from __future__ import annotations

import json

import pytest

from src.evaluation.evaluators.pose_evaluator import PoseEvaluator
from src.schemas.pose import AnimalPose, HumanPose, Keypoint, PoseFrame


def _make_human_pose(
    track_id: int,
    keypoints: list[tuple[float, float, float, bool]],
    bbox: tuple[float, float, float, float],
    frame_id: int = 0,
) -> tuple[HumanPose, PoseFrame]:
    kps = tuple(
        Keypoint(name=f"kp{i}", x=x, y=y, confidence=c, visible=v)
        for i, (x, y, c, v) in enumerate(keypoints)
    )
    human = HumanPose(
        track_id=track_id,
        detection_id=track_id,
        keypoints=kps,
        bbox=bbox,
        overall_confidence=1.0,
        body_orientation_deg=None,
        head_orientation_deg=None,
        timestamp=0.0,
    )
    frame = PoseFrame(
        frame_id=frame_id,
        timestamp=0.0,
        humans=(human,),
        animals=(),
    )
    return human, frame


def _make_animal_pose(
    track_id: int,
    keypoints: list[tuple[float, float, float, bool]],
    bbox: tuple[float, float, float, float],
    frame_id: int = 0,
) -> tuple[AnimalPose, PoseFrame]:
    kps = tuple(
        Keypoint(name=f"kp{i}", x=x, y=y, confidence=c, visible=v)
        for i, (x, y, c, v) in enumerate(keypoints)
    )
    animal = AnimalPose(
        track_id=track_id,
        detection_id=track_id,
        keypoints=kps,
        bbox=bbox,
        overall_confidence=1.0,
        body_orientation_deg=None,
        head_orientation_deg=None,
        species="dog",
        timestamp=0.0,
    )
    frame = PoseFrame(
        frame_id=frame_id,
        timestamp=0.0,
        humans=(),
        animals=(animal,),
    )
    return animal, frame


def _make_gt_json(
    humans: list[dict],
    animals: list[dict] | None = None,
) -> dict:
    annotations = []
    for h in humans:
        annotations.append({
            "image_id": h.get("frame_id", 0),
            "category_id": 1,
            "keypoints": h["keypoints"],
            "num_keypoints": len(h["keypoints"]) // 3,
            "bbox": h["bbox"],
            "area": h["bbox"][2] * h["bbox"][3],
        })
    if animals:
        for a in animals:
            annotations.append({
                "image_id": a.get("frame_id", 0),
                "category_id": 2,
                "keypoints": a["keypoints"],
                "num_keypoints": len(a["keypoints"]) // 3,
                "bbox": a["bbox"],
                "area": a["bbox"][2] * a["bbox"][3],
            })
    return {
        "images": [{"id": 0, "file_name": "frame_0000.jpg", "width": 1920, "height": 1080}],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "animal", "supercategory": "animal"},
        ],
    }


class TestPoseEvaluator:
    def test_perfect_human_match(self) -> None:
        kps_flat = [100.0, 120.0, 2] * 17
        bbox_coco = [50.0, 50.0, 100.0, 150.0]
        gt_json = _make_gt_json(humans=[{
            "frame_id": 0, "keypoints": kps_flat, "bbox": bbox_coco,
        }])

        kps = [(100.0, 120.0, 1.0, True)] * 17
        _, frame = _make_human_pose(1, kps, (50.0, 50.0, 150.0, 200.0), frame_id=0)

        evaluator = PoseEvaluator(match_iou_threshold=0.5)
        result = evaluator.evaluate([frame], gt_json)
        assert result["human_pck_at_0.05"] > 0.99
        assert result["human_oks"] > 0.99
        assert result["human_pck_at_0.05_count"] == 1
        assert result["human_unmatched_gt"] == 0

    def test_no_match_low_iou(self) -> None:
        kps_flat = [100.0, 120.0, 2] * 17
        bbox_coco = [500.0, 500.0, 50.0, 50.0]
        gt_json = _make_gt_json(humans=[{
            "frame_id": 0, "keypoints": kps_flat, "bbox": bbox_coco,
        }])

        kps = [(100.0, 120.0, 1.0, True)] * 17
        _, frame = _make_human_pose(1, kps, (0.0, 0.0, 100.0, 100.0), frame_id=0)

        evaluator = PoseEvaluator(match_iou_threshold=0.5)
        result = evaluator.evaluate([frame], gt_json)
        assert result["human_pck_at_0.05_count"] == 0
        assert result["human_unmatched_gt"] == 1

    def test_missing_frame_no_crash(self) -> None:
        gt_json = _make_gt_json(humans=[{
            "frame_id": 0, "keypoints": [100.0, 120.0, 2] * 17,
            "bbox": [50.0, 50.0, 100.0, 150.0],
        }])
        evaluator = PoseEvaluator()
        result = evaluator.evaluate([None], gt_json)
        assert result["human_unmatched_gt"] == 1

    def test_animal_evaluation(self) -> None:
        kps_flat = [200.0, 300.0, 2] * 17
        bbox_coco = [100.0, 100.0, 80.0, 120.0]
        gt_json = _make_gt_json(
            humans=[],
            animals=[{
                "frame_id": 0, "keypoints": kps_flat, "bbox": bbox_coco,
            }],
        )

        kps = [(200.0, 300.0, 1.0, True)] * 17
        _, frame = _make_animal_pose(1, kps, (100.0, 100.0, 180.0, 220.0), frame_id=0)

        evaluator = PoseEvaluator()
        result = evaluator.evaluate([frame], gt_json)
        assert result["animal_pck_at_0.05"] > 0.99
        assert result["animal_pck_at_0.05_count"] == 1

    def test_multiple_frames(self) -> None:
        gt_anns = []
        frames = []
        for fid in range(3):
            x = fid * 100.0
            gt_anns.append({
                "frame_id": fid, "keypoints": [x + 50.0, 100.0, 2] * 17,
                "bbox": [x, 50.0, 100.0, 150.0],
            })
            kps = [(x + 50.0, 100.0, 1.0, True)] * 17
            _, frame = _make_human_pose(
                fid, kps, (x, 50.0, x + 100.0, 200.0), frame_id=fid,
            )
            frames.append(frame)

        gt_json = _make_gt_json(humans=gt_anns)
        evaluator = PoseEvaluator()
        result = evaluator.evaluate(frames, gt_json)
        assert result["human_pck_at_0.05_count"] == 3

    def test_json_serializable_output(self) -> None:
        kps_flat = [100.0, 120.0, 2] * 17
        bbox_coco = [50.0, 50.0, 100.0, 150.0]
        gt_json = _make_gt_json(humans=[{
            "frame_id": 0, "keypoints": kps_flat, "bbox": bbox_coco,
        }])

        kps = [(100.0, 120.0, 1.0, True)] * 17
        _, frame = _make_human_pose(1, kps, (50.0, 50.0, 150.0, 200.0), frame_id=0)

        evaluator = PoseEvaluator()
        result = evaluator.evaluate([frame], gt_json)
        json.dumps(result)
