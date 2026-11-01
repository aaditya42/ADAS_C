from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.schemas.behavior import BehaviorFrame, BehaviorLabel, BehaviorState


class TestBehaviorLabel:
    def test_human_labels_count(self) -> None:
        human_labels = {m for m in BehaviorLabel if m.value.startswith("human_")}
        assert len(human_labels) == 8

    def test_animal_labels_count(self) -> None:
        animal_labels = {m for m in BehaviorLabel if m.value.startswith("animal_")}
        assert len(animal_labels) == 7

    def test_unknown_label_exists(self) -> None:
        assert BehaviorLabel.UNKNOWN.value == "unknown"

    def test_human_label_values(self) -> None:
        assert BehaviorLabel.HUMAN_STANDING.value == "human_standing"
        assert BehaviorLabel.HUMAN_WALKING.value == "human_walking"
        assert BehaviorLabel.HUMAN_RUNNING.value == "human_running"
        assert BehaviorLabel.HUMAN_CROSSING_ROAD.value == "human_crossing_road"
        assert BehaviorLabel.HUMAN_APPROACHING_ROAD.value == "human_approaching_road"
        assert BehaviorLabel.HUMAN_LEAVING_ROAD.value == "human_leaving_road"
        assert BehaviorLabel.HUMAN_FALLEN.value == "human_fallen"
        assert BehaviorLabel.HUMAN_DISTRACTED.value == "human_distracted"

    def test_animal_label_values(self) -> None:
        assert BehaviorLabel.ANIMAL_STANDING.value == "animal_standing"
        assert BehaviorLabel.ANIMAL_WALKING.value == "animal_walking"
        assert BehaviorLabel.ANIMAL_RUNNING.value == "animal_running"
        assert BehaviorLabel.ANIMAL_LYING.value == "animal_lying"
        assert BehaviorLabel.ANIMAL_GRAZING.value == "animal_grazing"
        assert BehaviorLabel.ANIMAL_CROSSING_ROAD.value == "animal_crossing_road"
        assert BehaviorLabel.ANIMAL_APPROACHING_ROAD.value == "animal_approaching_road"

    def test_total_label_count(self) -> None:
        # 8 human + 7 animal + 1 unknown = 16
        assert len(BehaviorLabel) == 16


class TestBehaviorState:
    def test_minimal_construction(self) -> None:
        state = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_WALKING,
            secondary_labels=(),
            confidence=0.85,
        )
        assert state.track_id == 1
        assert state.entity_type == "human"
        assert state.primary_label == BehaviorLabel.HUMAN_WALKING
        assert state.secondary_labels == ()
        assert state.confidence == 0.85

    def test_default_values(self) -> None:
        state = BehaviorState(
            track_id=1,
            entity_type="animal",
            primary_label=BehaviorLabel.ANIMAL_STANDING,
            secondary_labels=(),
            confidence=0.9,
        )
        assert state.crossing_probability is None
        assert state.body_orientation_deg is None
        assert state.head_orientation_deg is None
        assert state.timestamp == 0.0
        assert state.schema_version == "1.0"

    def test_with_secondary_labels(self) -> None:
        state = BehaviorState(
            track_id=2,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_CROSSING_ROAD,
            secondary_labels=(
                BehaviorLabel.HUMAN_WALKING,
                BehaviorLabel.HUMAN_APPROACHING_ROAD,
            ),
            confidence=0.7,
            crossing_probability=0.85,
            body_orientation_deg=90.0,
            head_orientation_deg=30.0,
            timestamp=1.5,
        )
        assert len(state.secondary_labels) == 2
        assert state.crossing_probability == 0.85
        assert state.body_orientation_deg == 90.0
        assert state.head_orientation_deg == 30.0

    def test_immutable(self) -> None:
        state = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_STANDING,
            secondary_labels=(),
            confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            state.track_id = 2  # type: ignore[misc]

    def test_immutable_secondary_labels(self) -> None:
        state = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_STANDING,
            secondary_labels=(),
            confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            state.primary_label = BehaviorLabel.HUMAN_RUNNING  # type: ignore[misc]

    def test_crossing_probability_range(self) -> None:
        state = BehaviorState(
            track_id=3,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_CROSSING_ROAD,
            secondary_labels=(),
            confidence=0.8,
            crossing_probability=1.0,
        )
        assert state.crossing_probability == 1.0

    def test_animal_entity_type(self) -> None:
        state = BehaviorState(
            track_id=4,
            entity_type="animal",
            primary_label=BehaviorLabel.ANIMAL_CROSSING_ROAD,
            secondary_labels=(),
            confidence=0.6,
        )
        assert state.entity_type == "animal"

    def test_unknown_behavior(self) -> None:
        state = BehaviorState(
            track_id=5,
            entity_type="human",
            primary_label=BehaviorLabel.UNKNOWN,
            secondary_labels=(),
            confidence=0.0,
        )
        assert state.primary_label == BehaviorLabel.UNKNOWN


class TestBehaviorFrame:
    def test_empty_frame(self) -> None:
        frame = BehaviorFrame(frame_id=0, timestamp=0.0)
        assert frame.frame_id == 0
        assert frame.timestamp == 0.0
        assert frame.states == ()
        assert frame.schema_version == "1.0"

    def test_with_states(self) -> None:
        state = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_WALKING,
            secondary_labels=(),
            confidence=0.85,
        )
        frame = BehaviorFrame(frame_id=1, timestamp=0.033, states=(state,))
        assert len(frame.states) == 1
        assert frame.states[0].track_id == 1

    def test_multiple_states(self) -> None:
        s1 = BehaviorState(
            track_id=1,
            entity_type="human",
            primary_label=BehaviorLabel.HUMAN_WALKING,
            secondary_labels=(),
            confidence=0.85,
        )
        s2 = BehaviorState(
            track_id=2,
            entity_type="animal",
            primary_label=BehaviorLabel.ANIMAL_CROSSING_ROAD,
            secondary_labels=(),
            confidence=0.9,
            crossing_probability=0.75,
        )
        frame = BehaviorFrame(frame_id=2, timestamp=0.066, states=(s1, s2))
        assert len(frame.states) == 2
        assert frame.states[1].crossing_probability == 0.75

    def test_immutable(self) -> None:
        frame = BehaviorFrame(frame_id=0, timestamp=0.0)
        with pytest.raises(FrozenInstanceError):
            frame.frame_id = 1  # type: ignore[misc]


class TestStandaloneImports:
    def test_import_behavior(self) -> None:
        from src.schemas.behavior import BehaviorLabel, BehaviorState, BehaviorFrame  # noqa: F811

        assert True
