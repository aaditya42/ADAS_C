from __future__ import annotations

from dataclasses import dataclass

from src.schemas.bbox import BoundingBox


@dataclass(frozen=True)
class TrackedObject:
    """A detection that has been assigned a persistent track ID.

    The track_id is unique across the lifetime of the source video. The object
    retains its class label and bounding box from the latest associated detection.
    """

    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
