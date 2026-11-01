from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.behavior.animal.rule_based import RuleBasedAnimalBehaviorClassifier
from src.behavior.base import (
    BaseAnimalBehaviorClassifier,
    BaseHumanBehaviorClassifier,
)
from src.behavior.config import BehaviorConfig
from src.behavior.human.rule_based import RuleBasedHumanBehaviorClassifier
from src.utils.logger import get_logger


@dataclass(frozen=True, slots=True)
class BehaviorClassifierBundle:
    """Pair of human and animal behavior classifiers."""

    human: Optional[BaseHumanBehaviorClassifier]
    animal: Optional[BaseAnimalBehaviorClassifier]


def build_behavior_classifiers(
    config: dict[str, Any],
) -> BehaviorClassifierBundle:
    """Construct behavior classifiers from the pipeline configuration.

    Args:
        config: Full pipeline configuration dictionary (the ``behavior``
            section is read internally).

    Returns:
        A :class:`BehaviorClassifierBundle` whose ``human`` and ``animal``
        fields are ``None`` when the behavior module is disabled.
    """
    logger = get_logger(f"{__name__}.build_behavior_classifiers")

    raw_behavior = config.get("behavior", {})
    if not raw_behavior.get("enabled", False):
        logger.debug("Behavior classification disabled — returning None classifiers")
        return BehaviorClassifierBundle(human=None, animal=None)

    behavior_cfg = BehaviorConfig.from_dict(raw_behavior)

    human: Optional[BaseHumanBehaviorClassifier] = None
    animal: Optional[BaseAnimalBehaviorClassifier] = None

    # Human classifier (always rule_based for now)
    human_type = raw_behavior.get("human", {}).get("classifier", "rule_based")
    if human_type == "rule_based":
        human = RuleBasedHumanBehaviorClassifier(behavior_cfg)
    elif human_type != "null":
        raise ValueError(f"Unknown human behavior classifier: {human_type}")

    # Animal classifier (always rule_based for now)
    animal_type = raw_behavior.get("animal", {}).get("classifier", "rule_based")
    if animal_type == "rule_based":
        animal = RuleBasedAnimalBehaviorClassifier(behavior_cfg)
    elif animal_type != "null":
        raise ValueError(f"Unknown animal behavior classifier: {animal_type}")

    logger.debug(
        "Built behaviour classifiers: human=%s animal=%s",
        type(human).__name__ if human else "None",
        type(animal).__name__ if animal else "None",
    )
    return BehaviorClassifierBundle(human=human, animal=animal)
