from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.pipeline import PerceptionPipeline
from src.schemas import Frame
from src.schemas.behavior import BehaviorFrame


class TestPipelineBehaviorDisabled:
    """Pipeline output when ``behavior`` section is missing or disabled."""

    def test_behavior_key_not_in_result_when_missing(self) -> None:
        config: dict[str, Any] = {}
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert "behavior" in result
        assert result["behavior"] is None

    def test_behavior_is_none_when_explicitly_disabled(self) -> None:
        config: dict[str, Any] = {"behavior": {"enabled": False}}
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert result["behavior"] is None


class TestPipelineBehaviorEnabled:
    """Pipeline output when ``behavior.enabled=true`` with null pose."""

    def test_behavior_is_behaviorframe_when_enabled(self) -> None:
        config: dict[str, Any] = {
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            },
            "behavior": {
                "enabled": True,
            },
        }
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        behavior = result.get("behavior")
        assert isinstance(behavior, BehaviorFrame)
        assert len(behavior.states) == 0

    def test_behavior_has_correct_frame_id(self) -> None:
        config: dict[str, Any] = {
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            },
            "behavior": {
                "enabled": True,
            },
        }
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=42, timestamp=1.5, source="test")
        result = pipeline.process(frame)
        behavior = result["behavior"]
        assert behavior.frame_id == 42
        assert behavior.timestamp == 1.5


class TestPipelineGoldenOutput:
    """Backward-compatibility: enabling behavior does not alter other outputs
    when behaviour is enabled but there are no poses (null estimators)."""

    def _baseline_config(self) -> dict[str, Any]:
        return {}

    def _behavior_enabled_config(self) -> dict[str, Any]:
        return {
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            },
            "behavior": {
                "enabled": True,
            },
        }

    def test_behavior_enabled_does_not_change_existing_keys(self) -> None:
        pipeline_baseline = PerceptionPipeline(self._baseline_config())
        pipeline_enabled = PerceptionPipeline(self._behavior_enabled_config())

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")

        result_baseline = pipeline_baseline.process(frame)
        result_enabled = pipeline_enabled.process(frame)

        assert result_baseline["frame"] is result_enabled["frame"]

        baseline_keys = set(result_baseline.keys())
        enabled_keys = set(result_enabled.keys())

        # behavior key exists in both
        assert "behavior" in baseline_keys
        assert "behavior" in enabled_keys
        # pose key exists in both
        assert "pose" in baseline_keys
        assert "pose" in enabled_keys

        for key in baseline_keys:
            if key in ("pose", "behavior", "history"):
                continue

            assert key in result_enabled, f"Key {key!r} missing from enabled result"
            baseline_val = result_baseline[key]
            enabled_val = result_enabled[key]

            if isinstance(baseline_val, dict):
                assert baseline_val == enabled_val, f"Mismatch in dict {key!r}"
            elif isinstance(baseline_val, list):
                assert len(baseline_val) == len(enabled_val), f"Length mismatch in {key!r}"
                for b, e in zip(baseline_val, enabled_val):
                    assert type(b) is type(e), f"Type mismatch in {key!r}: {type(b)} vs {type(e)}"
            elif isinstance(baseline_val, np.ndarray):
                assert np.allclose(baseline_val, enabled_val), f"Array mismatch in {key!r}"
            elif baseline_val is None:
                assert enabled_val is None, f"Expected None for {key!r}, got {enabled_val}"
            else:
                assert baseline_val == enabled_val, f"Scalar mismatch in {key!r}: {baseline_val!r} vs {enabled_val!r}"

        # Verify structure
        assert result_baseline["pose"] is None
        assert result_baseline["behavior"] is None
        assert isinstance(result_enabled["pose"], object)
        assert isinstance(result_enabled["behavior"], BehaviorFrame)

    def test_baseline_behavior_key_is_none(self) -> None:
        """Verify the baseline output has ``behavior=None`` when disabled."""
        pipeline = PerceptionPipeline(self._baseline_config())
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert "behavior" in result
        assert result["behavior"] is None


class TestPipelineRobustness:

    def test_empty_config_does_not_crash(self) -> None:
        pipeline = PerceptionPipeline({})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert isinstance(result, dict)
        assert result["behavior"] is None

    def test_behavior_enabled_without_pose_enabled_is_safe(self) -> None:
        """When pose is not enabled, behavior should produce
        an empty BehaviorFrame because pose_frame is None."""
        pipeline = PerceptionPipeline({
            "behavior": {"enabled": True},
        })
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert isinstance(result["behavior"], BehaviorFrame)
        assert len(result["behavior"].states) == 0

    def test_behavior_enabled_empty_section_does_not_crash(self) -> None:
        pipeline = PerceptionPipeline({"behavior": {}})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert result["behavior"] is None
