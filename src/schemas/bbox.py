from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    All coordinates are in the image pixel space with origin at top-left.
    x1, y1 = top-left corner; x2, y2 = bottom-right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(
                f"Invalid BoundingBox: top-left ({self.x1}, {self.y1}) must be "
                f"strictly above bottom-right ({self.x2}, {self.y2})."
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        return (cx, cy)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-Union with another BoundingBox."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter = inter_w * inter_h

        union = self.area + other.area - inter
        if union == 0.0:
            return 0.0
        return inter / union
