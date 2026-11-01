from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.schemas.behavior import BehaviorFrame
from src.schemas.pose import PoseFrame


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Per-frame pipeline context carrying results from all subsystems.

    This is a lightweight container that subsystems can read from and
    write to during pipeline execution.  It is additive: new fields are
    added as each subsystem is integrated, without breaking existing code.

    Attributes:
        pose_frame: The unified pose estimation result for the current
            frame, or ``None`` if pose estimation is disabled.
        behavior_frame: The behaviour classification result for the current
            frame, or ``None`` if behaviour classification is disabled.
    """

    pose_frame: Optional[PoseFrame] = None
    behavior_frame: Optional[BehaviorFrame] = None
