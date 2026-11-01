from __future__ import annotations

from typing import Any

import pytest

from src.behavior.animal.rule_based import RuleBasedAnimalBehaviorClassifier
from src.behavior.base import BaseAnimalBehaviorClassifier, BaseHumanBehaviorClassifier
from src.behavior.factory import BehaviorClassifierBundle, build_behavior_classifiers
from src.behavior.human.rule_based import RuleBasedHumanBehaviorClassifier


class TestBuildBehaviorClassifiers:
    """Tests for ``build_behavior_classifiers``."""

    def test_disabled_returns_none_classifiers(self) -> None:
        bundle = build_behavior_classifiers({"behavior": {"enabled": False}})
        assert bundle.human is None
        assert bundle.animal is None

    def test_missing_behavior_section_returns_none(self) -> None:
        bundle = build_behavior_classifiers({})
        assert bundle.human is None
        assert bundle.animal is None

    def test_empty_behavior_section_returns_none(self) -> None:
        bundle = build_behavior_classifiers({"behavior": {}})
        assert bundle.human is None
        assert bundle.animal is None

    def test_enabled_rule_based_returns_classifiers(self) -> None:
        config: dict[str, Any] = {
            "behavior": {
                "enabled": True,
                "human": {"classifier": "rule_based"},
                "animal": {"classifier": "rule_based"},
            }
        }
        bundle = build_behavior_classifiers(config)
        assert isinstance(bundle.human, RuleBasedHumanBehaviorClassifier)
        assert isinstance(bundle.animal, RuleBasedAnimalBehaviorClassifier)

    def test_enabled_without_subconfig_returns_rule_based(self) -> None:
        """When enabled without explicit sub-config, defaults to rule_based."""
        config: dict[str, Any] = {
            "behavior": {"enabled": True}
        }
        bundle = build_behavior_classifiers(config)
        assert isinstance(bundle.human, BaseHumanBehaviorClassifier)
        assert isinstance(bundle.animal, BaseAnimalBehaviorClassifier)

    def test_null_human_returns_none_human(self) -> None:
        config: dict[str, Any] = {
            "behavior": {
                "enabled": True,
                "human": {"classifier": "null"},
                "animal": {"classifier": "rule_based"},
            }
        }
        bundle = build_behavior_classifiers(config)
        assert bundle.human is None
        assert isinstance(bundle.animal, RuleBasedAnimalBehaviorClassifier)

    def test_unknown_human_classifier_raises(self) -> None:
        config: dict[str, Any] = {
            "behavior": {
                "enabled": True,
                "human": {"classifier": "magic_ml"},
            }
        }
        with pytest.raises(ValueError, match="magic_ml"):
            build_behavior_classifiers(config)

    def test_unknown_animal_classifier_raises(self) -> None:
        config: dict[str, Any] = {
            "behavior": {
                "enabled": True,
                "animal": {"classifier": "deep_learning"},
            }
        }
        with pytest.raises(ValueError, match="deep_learning"):
            build_behavior_classifiers(config)

    def test_bundle_is_frozen(self) -> None:
        bundle = BehaviorClassifierBundle(human=None, animal=None)
        with pytest.raises(AttributeError):
            bundle.human = object()  # type: ignore[misc]
