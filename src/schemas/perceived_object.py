from __future__ import annotations

from dataclasses import dataclass

from src.schemas.bbox import BoundingBox


@dataclass(frozen=True)
class PerceivedObject:
    """A tracked object that has been enriched with depth and distance estimates.

    depth        — estimated depth in meters (positive).
    distance     — estimated lateral / radial distance in meters.
    depth_confidence — confidence of the depth estimate in [0, 1].
    """

    track_id: int
    class_name: str
    bbox: BoundingBox
    depth: float
    distance: float
    depth_confidence: float
