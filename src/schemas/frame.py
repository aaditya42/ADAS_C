from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    """A single video frame with metadata.

    The image is an HWC uint8 numpy array (height, width, 3 channels, BGR order
    as produced by OpenCV). The frame_id is a monotonically increasing counter
    unique within a video source.
    """

    image: np.ndarray
    frame_id: int
    timestamp: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
