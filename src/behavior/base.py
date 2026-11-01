from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.schemas.behavior import BehaviorState
from src.schemas.lane_info import LaneLine
from src.schemas.pose import AnimalPose, HumanPose


# --------------------------------------------------------------------------- #
#  Context types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrackHistory:
    """Bounding-box position history for a single track.

    Positions are bottom-center ``(cx, by)`` in image-pixel coordinates,
    ordered oldest to newest.
    """

    track_id: int
    positions: Tuple[Tuple[float, float], ...]
    timestamps: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.timestamps):
            raise ValueError(
                f"TrackHistory: {len(self.positions)} positions vs "
                f"{len(self.timestamps)} timestamps"
            )

    @property
    def duration(self) -> float:
        """Total time span covered by the history in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def displacement(self) -> Tuple[float, float]:
        """Displacement vector ``(dx, dy)`` from first to last position."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        return (
            self.positions[-1][0] - self.positions[0][0],
            self.positions[-1][1] - self.positions[0][1],
        )


@dataclass(frozen=True)
class MotionEstimate:
    """Motion estimate for a single tracked entity (its own motion).

    ``heading_deg`` uses the same convention as
    :attr:`HumanPose.body_orientation_deg` / :attr:`AnimalPose.body_orientation_deg`:
    0° = facing the ego vehicle, positive = clockwise.
    ``speed_mps`` is the entity's estimated speed in meters-per-second.
    """

    track_id: int
    speed_mps: Optional[float]
    heading_deg: Optional[float]
    acceleration_mps2: Optional[float]
    confidence: float


@dataclass(frozen=True)
class LaneContext:
    """Lane geometry for the current frame in image-pixel coordinates.

    Each boundary is a :class:`LaneLine` whose polynomial ``x = a*y² + b*y + c``
    is evaluated at the tracked entity's foot y-coordinate.
    """

    left_boundary: Optional[LaneLine]
    right_boundary: Optional[LaneLine]
    confidence: float

    @staticmethod
    def empty() -> LaneContext:
        return LaneContext(
            left_boundary=None,
            right_boundary=None,
            confidence=0.0,
        )

    def lane_x_at_y(self, line: LaneLine, y: float) -> Optional[float]:
        """Evaluate a lane-line polynomial at a given image y-coordinate.

        Returns the x-coordinate or ``None`` if the polynomial is empty.
        """
        if not line.coeffs:
            return None
        a, b, c = _ensure_degree2(line.coeffs)
        return a * y * y + b * y + c

    def is_inside(self, x: float, y: float) -> bool:
        """Whether a point ``(x, y)`` falls between the left and right lanes."""
        if self.left_boundary is None or self.right_boundary is None:
            return False
        left_x = self.lane_x_at_y(self.left_boundary, y)
        right_x = self.lane_x_at_y(self.right_boundary, y)
        if left_x is None or right_x is None:
            return False
        return left_x <= x <= right_x

    def distance_to_nearest_boundary(self, x: float, y: float) -> Optional[float]:
        """Shortest horizontal distance in pixels to either lane boundary."""
        if self.left_boundary is not None:
            left_x = self.lane_x_at_y(self.left_boundary, y)
        else:
            left_x = None
        if self.right_boundary is not None:
            right_x = self.lane_x_at_y(self.right_boundary, y)
        else:
            right_x = None

        dists: list[float] = []
        if left_x is not None:
            dists.append(abs(x - left_x))
        if right_x is not None:
            dists.append(abs(x - right_x))
        if not dists:
            return None
        return min(dists)


def _ensure_degree2(coeffs: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Pad or truncate coefficients to degree 2."""
    if len(coeffs) >= 3:
        return (coeffs[0], coeffs[1], coeffs[2])
    padded = [0.0, 0.0, 0.0]
    for i, c in enumerate(reversed(coeffs)):
        padded[2 - i] = c
    return (padded[0], padded[1], padded[2])


# --------------------------------------------------------------------------- #
#  Abstract Base Classes
# --------------------------------------------------------------------------- #


class BaseHumanBehaviorClassifier(ABC):
    """Interface for human behavior classifiers.

    Implementations should be stateless across tracks — all temporal
    context comes via ``pose_history`` and ``track_history``.
    """

    @abstractmethod
    def classify(
        self,
        pose_history: Tuple[HumanPose, ...],
        track_history: Optional[TrackHistory],
        motion: Optional[MotionEstimate],
        lane_context: Optional[LaneContext],
        timestamp: float,
    ) -> BehaviorState:
        """Classify the behavior of a single tracked human.

        Args:
            pose_history: Per-track pose observations, oldest first.
            track_history: Bbox position history for the track.
            motion: Estimated motion of the entity.
            lane_context: Lane geometry for the current frame.
            timestamp: Frame timestamp in seconds.

        Returns:
            A :class:`BehaviorState` for the entity.
        """


class BaseAnimalBehaviorClassifier(ABC):
    """Interface for animal behavior classifiers.

    Implementations should be stateless across tracks — all temporal
    context comes via ``pose_history`` and ``track_history``.
    """

    @abstractmethod
    def classify(
        self,
        pose_history: Tuple[AnimalPose, ...],
        track_history: Optional[TrackHistory],
        motion: Optional[MotionEstimate],
        lane_context: Optional[LaneContext],
        timestamp: float,
    ) -> BehaviorState:
        """Classify the behavior of a single tracked animal.

        Args:
            pose_history: Per-track pose observations, oldest first.
            track_history: Bbox position history for the track.
            motion: Estimated motion of the entity.
            lane_context: Lane geometry for the current frame.
            timestamp: Frame timestamp in seconds.

        Returns:
            A :class:`BehaviorState` for the entity.
        """
