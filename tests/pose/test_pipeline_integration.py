from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.core.pipeline import PerceptionPipeline
from src.schemas import Frame


# --------------------------------------------------------------------------- #
#  Integration tests: pose enabled via Null estimators
# --------------------------------------------------------------------------- #


class TestPipelineIntegrationNullEstimators:
    """Pipeline behaves correctly when ``pose.enabled=true`` with null estimators."""

    def test_pose_key_present_in_output(self) -> None:
        config: dict[str, Any] = {
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            }
        }
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert "pose" in result

    def test_pose_is_none_when_disabled(self) -> None:
        config: dict[str, Any] = {}
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert result.get("pose") is None

    def test_pose_is_poseframe_when_enabled(self) -> None:
        from src.schemas.pose import PoseFrame

        config: dict[str, Any] = {
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            }
        }
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        pose = result["pose"]
        assert isinstance(pose, PoseFrame)
        assert len(pose.humans) == 0
        assert len(pose.animals) == 0

    def test_pose_has_correct_frame_id(self) -> None:
        config: dict[str, Any] = {
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            }
        }
        pipeline = PerceptionPipeline(config)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=42, timestamp=1.5, source="test")
        result = pipeline.process(frame)
        pose = result["pose"]
        assert pose.frame_id == 42
        assert pose.timestamp == 1.5


# --------------------------------------------------------------------------- #
#  Golden output comparison
# --------------------------------------------------------------------------- #


class TestPipelineGoldenOutput:
    """Backward-compatibility: enabling null pose does not change other outputs.

    The result dict with ``pose.enabled=false`` (baseline) should be identical
    to the result with ``pose.enabled=true`` (null estimators) for all keys
    except the ``"pose"`` key (``None`` in baseline, a ``PoseFrame`` when
    enabled).
    """

    def _golden_config(self) -> dict[str, Any]:
        return {"test": 1}

    def _pose_enabled_config(self) -> dict[str, Any]:
        return {
            "test": 1,
            "pose": {
                "enabled": True,
                "human": {"estimator": "null"},
                "animal": {"estimator": "null"},
            },
        }

    def test_pose_enabled_does_not_change_existing_keys(self) -> None:
        pipeline_baseline = PerceptionPipeline(self._golden_config())
        pipeline_enabled = PerceptionPipeline(self._pose_enabled_config())

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")

        result_baseline = pipeline_baseline.process(frame)
        result_enabled = pipeline_enabled.process(frame)

        # Same frame object should be shared
        assert result_baseline["frame"] is result_enabled["frame"]

        # All non-pose keys should be equal
        for key in result_baseline:
            if key == "pose":
                # baseline has pose=None, enabled has a PoseFrame
                assert result_baseline[key] is None
                assert result_enabled[key] is not None
                continue

            assert key in result_enabled, f"Key {key!r} missing from enabled result"
            baseline_val = result_baseline[key]
            enabled_val = result_enabled[key]

            # Compare dicts, lists, and scalars (skip "history" which is stateful)
            if key == "history":
                continue

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

        # Pose key exists in both but with different values
        assert result_baseline["pose"] is None
        assert result_enabled["pose"] is not None

    def test_baseline_pose_key_is_none(self) -> None:
        """Verify the baseline output has ``pose=None`` when disabled."""
        pipeline = PerceptionPipeline(self._golden_config())
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert "pose" in result
        assert result["pose"] is None


# --------------------------------------------------------------------------- #
#  Edge-case: pipeline does not crash with pose subsystem
# --------------------------------------------------------------------------- #


class TestPipelineRobustness:
    """Edge cases for pose integration."""

    def test_empty_config_does_not_crash(self) -> None:
        pipeline = PerceptionPipeline({})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert isinstance(result, dict)
        assert result["pose"] is None

    def test_pose_enabled_with_no_pose_section_does_not_crash(self) -> None:
        """Config without a ``pose`` section defaults to disabled."""
        pipeline = PerceptionPipeline({"detector": {"type": "yolo"}})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert result.get("pose") is None

    def test_pose_enabled_empty_section_does_not_crash(self) -> None:
        """pose: {} should default to disabled (no enabled flag)."""
        pipeline = PerceptionPipeline({"pose": {}})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=img, frame_id=0, timestamp=0.0, source="test")
        result = pipeline.process(frame)
        assert result.get("pose") is None
