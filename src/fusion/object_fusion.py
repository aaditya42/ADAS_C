from __future__ import annotations

from typing import Callable

import numpy as np

from src.schemas import BoundingBox, PerceivedObject, TrackedObject
from src.utils.logger import get_logger

SamplingFn = Callable[[np.ndarray], float]


class ObjectFusion:
    """Fuses tracking + depth into PerceivedObject instances.

    Each tracked object gets a depth estimate sampled from the corresponding
    region of the depth map.  Multiple sampling strategies are supported.

    When no depth map is available the resulting ``PerceivedObject`` uses
    ``depth=0.0``, ``distance=0.0``, ``depth_confidence=0.0`` as a sentinel
    for "depth unknown".

    Args:
        sampling: Sampling strategy name.
            ``"median_center"`` (default), ``"mean"``, ``"min"``,
            ``"percentile_25"``.
        center_box_ratio: Fraction (0–1) of the bbox width/height to keep
            as the central sampling region.
        min_valid_depth_pixels: Minimum number of valid (non-NaN, non-zero)
            pixels required to produce a confident estimate.
    """

    _STRATEGIES: dict[str, SamplingFn] = {}

    def __init__(
        self,
        sampling: str = "median_center",
        center_box_ratio: float = 0.5,
        min_valid_depth_pixels: int = 10,
    ) -> None:
        self._sampling_name = sampling
        self._center_box_ratio = center_box_ratio
        self._min_valid_pixels = min_valid_depth_pixels
        self._logger = get_logger(f"{__name__}.ObjectFusion")

        if sampling not in self._STRATEGIES:
            raise ValueError(
                f"Unknown sampling strategy: '{sampling}'. "
                f"Available: {list(self._STRATEGIES)}"
            )
        self._sample_fn = self._STRATEGIES[sampling]

        self._logger.info(
            "ObjectFusion ready  |  sampling=%s  center_ratio=%.2f  "
            "min_pixels=%d",
            sampling,
            center_box_ratio,
            min_valid_depth_pixels,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        tracked: list[TrackedObject],
        depth_map: np.ndarray | None,
    ) -> list[PerceivedObject]:
        """Fuse tracking and depth into PerceivedObject list.

        Args:
            tracked: Tracked objects from the tracker.
            depth_map: 2-D float32 depth map or ``None``.

        Returns:
            One :class:`PerceivedObject` per tracked object.
        """
        perceived: list[PerceivedObject] = []

        for obj in tracked:
            if depth_map is None or depth_map.size == 0:
                perceived.append(
                    PerceivedObject(
                        track_id=obj.track_id,
                        class_name=obj.class_name,
                        bbox=obj.bbox,
                        depth=0.0,
                        distance=0.0,
                        depth_confidence=0.0,
                    )
                )
            else:
                depth_val, conf = self._sample_depth(depth_map, obj.bbox)
                perceived.append(
                    PerceivedObject(
                        track_id=obj.track_id,
                        class_name=obj.class_name,
                        bbox=obj.bbox,
                        depth=depth_val,
                        distance=depth_val,  # placeholder — Phase 7 adds calibration
                        depth_confidence=conf,
                    )
                )

        return perceived

    # ------------------------------------------------------------------ #
    #  Depth sampling
    # ------------------------------------------------------------------ #

    def _sample_depth(
        self, depth_map: np.ndarray, bbox: BoundingBox
    ) -> tuple[float, float]:
        """Extract a single depth value and confidence for a bounding box."""
        h_img, w_img = depth_map.shape[:2]

        # Compute centre sub-region
        cx, cy = bbox.center
        sub_w = bbox.width * self._center_box_ratio
        sub_h = bbox.height * self._center_box_ratio

        x1 = int(round(max(0, cx - sub_w / 2)))
        y1 = int(round(max(0, cy - sub_h / 2)))
        x2 = int(round(min(w_img, cx + sub_w / 2)))
        y2 = int(round(min(h_img, cy + sub_h / 2)))

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0

        patch = depth_map[y1:y2, x1:x2]
        valid = patch[(~np.isnan(patch)) & (patch > 0)]

        total = patch.size
        valid_count = valid.size
        confidence = min(1.0, valid_count / max(total, 1))

        if valid_count < self._min_valid_pixels:
            return 0.0, confidence

        depth_val = self._sample_fn(valid)
        return float(depth_val), float(confidence)

    # ------------------------------------------------------------------ #
    #  Sampling strategies
    # ------------------------------------------------------------------ #

    @staticmethod
    def _median(samples: np.ndarray) -> float:
        return float(np.median(samples))

    @staticmethod
    def _mean(samples: np.ndarray) -> float:
        return float(np.mean(samples))

    @staticmethod
    def _min(samples: np.ndarray) -> float:
        return float(np.min(samples))

    @staticmethod
    def _percentile_25(samples: np.ndarray) -> float:
        return float(np.percentile(samples, 25))


ObjectFusion._STRATEGIES = {
    "median_center": ObjectFusion._median,
    "mean": ObjectFusion._mean,
    "min": ObjectFusion._min,
    "percentile_25": ObjectFusion._percentile_25,
}
