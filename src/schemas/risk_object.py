from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WarningLevel(Enum):
    NONE = "none"
    ADVISORY = "advisory"
    CAUTION = "caution"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RiskObject:
    """Risk assessment for a single perceived object.

    risk_score is a continuous value in [0, 100] where 0 is perfectly safe and
    100 is imminent collision. risk_level discretises this into four tiers.
    warning_level indicates what (if any) type of alert should be raised.
    """

    track_id: int
    risk_level: RiskLevel
    risk_score: float
    warning_level: WarningLevel
