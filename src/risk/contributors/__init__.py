from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.schemas.behavior import BehaviorFrame


@dataclass(frozen=True)
class RiskAdjustment:
    track_id: int
    delta_score: float


@dataclass
class RiskContext:
    perceived: Sequence[Any]
    motion: dict[int, Any]
    ttc: dict[int, float | None]
    lane_info: Any
    scene_context: Any
    behavior_frame: Optional[BehaviorFrame]


class RiskContributor(ABC):

    @abstractmethod
    def contribute(self, context: RiskContext) -> Sequence[RiskAdjustment]:
        ...


__all__ = [
    "RiskAdjustment",
    "RiskContext",
    "RiskContributor",
]
