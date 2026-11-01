from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.schemas.pose import (
    AnimalKeypointName,
    AnimalPose,
    HumanKeypointName,
    HumanPose,
    Keypoint,
    PoseFrame,
)


class TestHumanKeypointName:
    def test_coco_17_count(self) -> None:
        assert len(HumanKeypointName) == 17

    def test_first_and_last(self) -> None:
        assert HumanKeypointName.NOSE.value == "nose"
        assert HumanKeypointName.RIGHT_ANKLE.value == "right_ankle"

    def test_middle_members(self) -> None:
        assert HumanKeypointName.LEFT_SHOULDER.value == "left_shoulder"
        assert HumanKeypointName.LEFT_HIP.value == "left_hip"
        assert HumanKeypointName.LEFT_KNEE.value == "left_knee"


class TestAnimalKeypointName:
    def test_ap10k_17_count(self) -> None:
        assert len(AnimalKeypointName) == 17

    def test_first_and_last(self) -> None:
        assert AnimalKeypointName.LEFT_EYE.value == "left_eye"
        assert AnimalKeypointName.MID_BACK.value == "mid_back"

    def test_representative_members(self) -> None:
        assert AnimalKeypointName.NOSE.value == "nose"
        assert AnimalKeypointName.TAIL.value == "tail"
        assert AnimalKeypointName.LEFT_FRONT_PAW.value == "left_front_paw"
        assert AnimalKeypointName.RIGHT_BACK_SHOULDER.value == "right_back_shoulder"


class TestKeypoint:
    def test_construction(self) -> None:
        kp = Keypoint(name="nose", x=100.0, y=200.0, confidence=0.95, visible=True)
        assert kp.name == "nose"
        assert kp.x == 100.0
        assert kp.y == 200.0
        assert kp.confidence == 0.95
        assert kp.visible is True

    def test_immutable(self) -> None:
        kp = Keypoint(name="nose", x=100.0, y=200.0, confidence=0.95, visible=True)
        with pytest.raises(FrozenInstanceError):
            kp.x = 150.0  # type: ignore[misc]

    def test_immutable_after_reassignment(self) -> None:
        kp = Keypoint(name="nose", x=100.0, y=200.0, confidence=0.95, visible=True)
        with pytest.raises(FrozenInstanceError):
            kp.name = "left_eye"  # type: ignore[misc]

    def test_not_visible(self) -> None:
        kp = Keypoint(name="left_elbow", x=0.0, y=0.0, confidence=0.0, visible=False)
        assert kp.visible is False
        assert kp.confidence == 0.0


class TestHumanPose:
    def test_minimal_construction(self) -> None:
        kps = (
            Keypoint("nose", 100.0, 200.0, 0.95, True),
            Keypoint("left_eye", 90.0, 190.0, 0.90, True),
        )
        pose = HumanPose(
            track_id=1,
            detection_id=0,
            keypoints=kps,
            bbox=(50.0, 100.0, 200.0, 400.0),
            overall_confidence=0.85,
            body_orientation_deg=None,
            head_orientation_deg=None,
            timestamp=0.0,
        )
        assert pose.track_id == 1
        assert pose.detection_id == 0
        assert len(pose.keypoints) == 2
        assert pose.bbox == (50.0, 100.0, 200.0, 400.0)
        assert pose.overall_confidence == 0.85
        assert pose.body_orientation_deg is None
        assert pose.head_orientation_deg is None
        assert pose.timestamp == 0.0

    def test_schema_version_default(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = HumanPose(
            track_id=None,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.0,
            body_orientation_deg=None,
            head_orientation_deg=None,
            timestamp=0.0,
        )
        assert pose.schema_version == "1.0"

    def test_with_orientation(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = HumanPose(
            track_id=5,
            detection_id=2,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.9,
            body_orientation_deg=45.0,
            head_orientation_deg=-10.0,
            timestamp=1.5,
        )
        assert pose.body_orientation_deg == 45.0
        assert pose.head_orientation_deg == -10.0

    def test_immutable(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = HumanPose(
            track_id=1,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.0,
            body_orientation_deg=None,
            head_orientation_deg=None,
            timestamp=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            pose.track_id = 2  # type: ignore[misc]

    def test_none_track_id(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = HumanPose(
            track_id=None,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.0,
            body_orientation_deg=None,
            head_orientation_deg=None,
            timestamp=0.0,
        )
        assert pose.track_id is None


class TestAnimalPose:
    def test_minimal_construction(self) -> None:
        kps = (
            Keypoint("left_eye", 100.0, 200.0, 0.95, True),
            Keypoint("nose", 120.0, 210.0, 0.93, True),
        )
        pose = AnimalPose(
            track_id=2,
            detection_id=1,
            keypoints=kps,
            bbox=(80.0, 150.0, 250.0, 400.0),
            overall_confidence=0.80,
            body_orientation_deg=None,
            head_orientation_deg=None,
            species="dog",
            timestamp=0.0,
        )
        assert pose.track_id == 2
        assert pose.species == "dog"
        assert len(pose.keypoints) == 2
        assert pose.bbox == (80.0, 150.0, 250.0, 400.0)

    def test_schema_version_default(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = AnimalPose(
            track_id=None,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.0,
            body_orientation_deg=None,
            head_orientation_deg=None,
            species="cow",
            timestamp=0.0,
        )
        assert pose.schema_version == "1.0"

    def test_immutable(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = AnimalPose(
            track_id=1,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.0,
            body_orientation_deg=None,
            head_orientation_deg=None,
            species="deer",
            timestamp=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            pose.species = "dog"  # type: ignore[misc]

    def test_unknown_species(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        pose = AnimalPose(
            track_id=None,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.5,
            body_orientation_deg=None,
            head_orientation_deg=None,
            species="unknown",
            timestamp=0.0,
        )
        assert pose.species == "unknown"


class TestPoseFrame:
    def test_empty_frame(self) -> None:
        frame = PoseFrame(frame_id=0, timestamp=0.0, humans=(), animals=())
        assert frame.frame_id == 0
        assert frame.timestamp == 0.0
        assert frame.humans == ()
        assert frame.animals == ()
        assert frame.schema_version == "1.0"

    def test_with_humans(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        human = HumanPose(
            track_id=1,
            detection_id=0,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.9,
            body_orientation_deg=None,
            head_orientation_deg=None,
            timestamp=0.0,
        )
        frame = PoseFrame(frame_id=1, timestamp=0.033, humans=(human,), animals=())
        assert len(frame.humans) == 1
        assert frame.humans[0].track_id == 1

    def test_with_animals(self) -> None:
        kps = (Keypoint("nose", 0.0, 0.0, 0.0, False),)
        animal = AnimalPose(
            track_id=2,
            detection_id=1,
            keypoints=kps,
            bbox=(0.0, 0.0, 1.0, 1.0),
            overall_confidence=0.8,
            body_orientation_deg=None,
            head_orientation_deg=None,
            species="dog",
            timestamp=0.0,
        )
        frame = PoseFrame(frame_id=2, timestamp=0.066, humans=(), animals=(animal,))
        assert len(frame.animals) == 1
        assert frame.animals[0].species == "dog"

    def test_immutable(self) -> None:
        frame = PoseFrame(frame_id=0, timestamp=0.0, humans=(), animals=())
        with pytest.raises(FrozenInstanceError):
            frame.frame_id = 1  # type: ignore[misc]


class TestStandaloneImports:
    def test_import_pose(self) -> None:
        from src.schemas.pose import HumanPose, AnimalPose, PoseFrame, Keypoint  # noqa: F811

        assert True
