from __future__ import annotations

from typing import Any

import pytest

from src.risk.contributors import RiskAdjustment, RiskContext, RiskContributor
from src.risk.contributors.behavior_contributor import (
    BehaviorRiskContributionConfig,
    BehaviorRiskContributor,
)
from src.schemas.behavior import BehaviorFrame, BehaviorLabel, BehaviorState


def _make_state(
    track_id: int,
    label: BehaviorLabel,
    secondary: tuple[BehaviorLabel, ...] = (),
    crossing_prob: float | None = None,
) -> BehaviorState:
    return BehaviorState(
        track_id=track_id,
        entity_type="human" if "human" in label.value else "animal",
        primary_label=label,
        secondary_labels=secondary,
        confidence=0.9,
        crossing_probability=crossing_prob,
        timestamp=0.0,
    )


def _make_context(
    states: tuple[BehaviorState, ...],
    ttc_map: dict[int, float | None] | None = None,
) -> RiskContext:
    return RiskContext(
        perceived=[],
        motion={},
        ttc=ttc_map or {},
        lane_info=None,
        scene_context=None,
        behavior_frame=BehaviorFrame(
            frame_id=0, timestamp=0.0, states=states,
        ),
    )


class TestBehaviorRiskContributionConfig:

    def test_defaults(self) -> None:
        cfg = BehaviorRiskContributionConfig()
        assert cfg.fallen_boost == 30.0
        assert cfg.crossing_boost == 25.0
        assert cfg.approaching_boost == 20.0
        assert cfg.distracted_approaching_boost == 15.0
        assert cfg.animal_crossing_boost == 25.0
        assert cfg.animal_approaching_boost == 20.0
        assert cfg.crossing_prob_threshold == 0.5
        assert cfg.ttc_boost_scale == 0.3
        assert cfg.max_boost == 50.0

    def test_from_dict(self) -> None:
        cfg = BehaviorRiskContributionConfig.from_dict({
            "fallen_boost": 10.0,
            "crossing_boost": 15.0,
        })
        assert cfg.fallen_boost == 10.0
        assert cfg.crossing_boost == 15.0
        assert cfg.approaching_boost == 20.0  # default

    def test_from_dict_empty(self) -> None:
        cfg = BehaviorRiskContributionConfig.from_dict({})
        assert cfg.fallen_boost == 30.0  # default


class TestBehaviorRiskContributorNoBehavior:

    def test_no_behavior_frame_returns_empty(self) -> None:
        contrib = BehaviorRiskContributor()
        ctx = RiskContext(
            perceived=[], motion={}, ttc={},
            lane_info=None, scene_context=None,
            behavior_frame=None,
        )
        assert contrib.contribute(ctx) == []

    def test_empty_behavior_frame_returns_empty(self) -> None:
        contrib = BehaviorRiskContributor()
        ctx = _make_context(states=())
        assert contrib.contribute(ctx) == []


class TestBehaviorRiskContributorFallen:

    def test_fallen_human_gets_boost(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(track_id=1, label=BehaviorLabel.HUMAN_FALLEN)
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        assert adj[0].track_id == 1
        assert adj[0].delta_score == 30.0

    def test_fallen_boost_capped_by_max(self) -> None:
        contrib = BehaviorRiskContributor(weights={"fallen_boost": 100.0, "max_boost": 50.0})
        state = _make_state(track_id=1, label=BehaviorLabel.HUMAN_FALLEN)
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        assert adj[0].delta_score == 50.0  # capped

    def test_multiple_adjustments_in_one_frame(self) -> None:
        contrib = BehaviorRiskContributor()
        states = (
            _make_state(track_id=1, label=BehaviorLabel.HUMAN_FALLEN),
            _make_state(track_id=2, label=BehaviorLabel.HUMAN_FALLEN),
        )
        ctx = _make_context(states=states)
        adj = contrib.contribute(ctx)
        assert len(adj) == 2
        scores = {a.track_id: a.delta_score for a in adj}
        assert scores[1] == 30.0
        assert scores[2] == 30.0


class TestBehaviorRiskContributorCrossing:

    def test_human_crossing_road_with_high_prob(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_CROSSING_ROAD,
            crossing_prob=0.8,
        )
        ctx = _make_context(states=(state,), ttc_map={1: 3.0})
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        assert adj[0].track_id == 1
        # crossing_boost (25.0) + ttc increment
        assert adj[0].delta_score > 25.0
        assert adj[0].delta_score < 50.0

    def test_human_crossing_below_threshold_no_boost(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_CROSSING_ROAD,
            crossing_prob=0.3,
        )
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 0

    def test_human_crossing_with_distracted_secondary(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_CROSSING_ROAD,
            secondary=(BehaviorLabel.HUMAN_DISTRACTED,),
            crossing_prob=0.8,
        )
        ctx = _make_context(states=(state,), ttc_map={1: 3.0})
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        # crossing_boost (25) + ttc_increment + distracted_approaching_boost (15)
        assert adj[0].delta_score > 35.0  # 25 + 15 base minimum

    def test_animal_crossing_road(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.ANIMAL_CROSSING_ROAD,
            crossing_prob=0.8,
        )
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        assert adj[0].delta_score == 25.0  # animal_crossing_boost

    def test_animal_crossing_below_threshold_no_boost(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.ANIMAL_CROSSING_ROAD,
            crossing_prob=0.3,
        )
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 0


class TestBehaviorRiskContributorApproaching:

    def test_human_approaching_road(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_APPROACHING_ROAD,
            crossing_prob=0.7,
        )
        ctx = _make_context(states=(state,), ttc_map={1: 3.0})
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        assert adj[0].track_id == 1
        assert adj[0].delta_score > 20.0  # approaching_boost + ttc

    def test_human_approaching_with_distracted_secondary(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_APPROACHING_ROAD,
            secondary=(BehaviorLabel.HUMAN_DISTRACTED,),
            crossing_prob=0.7,
        )
        ctx = _make_context(states=(state,), ttc_map={1: 3.0})
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        # approaching_boost (20) + ttc_increment + distracted_approaching_boost (15)
        assert adj[0].delta_score > 30.0

    def test_animal_approaching_road(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.ANIMAL_APPROACHING_ROAD,
            crossing_prob=0.7,
        )
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 1
        assert adj[0].delta_score == 20.0  # animal_approaching_boost

    def test_unknown_label_no_boost(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(track_id=1, label=BehaviorLabel.UNKNOWN)
        ctx = _make_context(states=(state,))
        adj = contrib.contribute(ctx)
        assert len(adj) == 0


class TestBehaviorRiskContributorTTCScaling:

    def test_lower_ttc_gives_higher_boost(self) -> None:
        contrib = BehaviorRiskContributor()
        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_CROSSING_ROAD,
            crossing_prob=0.8,
        )
        ctx_fast = _make_context(states=(state,), ttc_map={1: 1.0})
        ctx_slow = _make_context(states=(state,), ttc_map={1: 5.0})
        adj_fast = contrib.contribute(ctx_fast)
        adj_slow = contrib.contribute(ctx_slow)
        assert adj_fast[0].delta_score > adj_slow[0].delta_score


class TestRiskEngineContributorIntegration:
    """Verify that ScoreBasedRiskEngine correctly uses contributors."""

    def test_empty_contributors_does_not_change_engine(self) -> None:
        from src.risk.score_based_risk import ScoreBasedRiskEngine
        from src.schemas import PerceivedObject
        from src.schemas.bbox import BoundingBox

        engine = ScoreBasedRiskEngine()
        obj = PerceivedObject(
            track_id=1, class_name="car",
            bbox=BoundingBox(0, 0, 100, 200),
            depth=10.0, distance=10.0, depth_confidence=0.9,
        )
        result_none = engine.evaluate(
            perceived=[obj],
            motion={},
            ttc={},
        )
        # With empty contributors, output should be identical
        engine._contributors = []
        result_empty = engine.evaluate(
            perceived=[obj],
            motion={},
            ttc={},
        )
        assert len(result_none) == len(result_empty)
        for r1, r2 in zip(result_none, result_empty):
            assert r1.track_id == r2.track_id
            assert r1.risk_score == r2.risk_score
            assert r1.risk_level == r2.risk_level

    def test_contributors_raise_risk_score(self) -> None:
        from src.risk.score_based_risk import ScoreBasedRiskEngine
        from src.schemas import PerceivedObject
        from src.schemas.bbox import BoundingBox

        contrib = BehaviorRiskContributor()
        engine = ScoreBasedRiskEngine(contributors=[contrib])
        obj = PerceivedObject(
            track_id=1, class_name="person",
            bbox=BoundingBox(0, 0, 100, 200),
            depth=10.0, distance=50.0, depth_confidence=0.9,
        )
        result_before = engine.evaluate(
            perceived=[obj],
            motion={},
            ttc={1: 5.0},
        )
        assert result_before[0].risk_score < 50  # far away, low base score

        state = _make_state(
            track_id=1, label=BehaviorLabel.HUMAN_FALLEN,
        )
        bf = BehaviorFrame(frame_id=0, timestamp=0.0, states=(state,))
        result_after = engine.evaluate(
            perceived=[obj],
            motion={},
            ttc={1: 5.0},
            behavior_frame=bf,
        )
        # Risk score should be higher due to FALLEN boost
        assert result_after[0].risk_score > result_before[0].risk_score
