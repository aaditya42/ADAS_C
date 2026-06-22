from __future__ import annotations

from dataclasses import dataclass

from src.schemas.bbox import BoundingBox


@dataclass(frozen=True)
class Detection:
    """A single object detection produced by a detector model.

    class_id corresponds to the model's internal class index; class_name is the
    human-readable label. confidence is in [0, 1].
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
