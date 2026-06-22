from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRiskEngine(ABC):
    """Abstract risk assessment engine.

    Contract:
      - evaluate() accepts perceived objects, motion estimates, TTC values,
        and optional lane/scene context, returning a list of RiskObject.
      - One RiskObject per perceived track.
    """

    @abstractmethod
    def evaluate(
        self,
        perceived: list[Any],
        motion: dict[int, Any],
        ttc: dict[int, float | None],
        lane_info: Any = None,
        scene_context: Any = None,
    ) -> list[Any]:
        ...
