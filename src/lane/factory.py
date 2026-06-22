from __future__ import annotations

from typing import Any

from src.lane.base import BaseLaneDetector


def build_lane_detector(config: dict[str, Any]) -> BaseLaneDetector | None:
    """Build a lane detector from the configuration.

    Args:
        config: The full pipeline configuration dictionary.

    Returns:
        A ``BaseLaneDetector`` instance, or ``None`` if lane detection
        is disabled.
    """
    lane_cfg = config.get("lane", {})
    if not lane_cfg.get("enabled", True):
        return None

    ltype = str(lane_cfg.get("type", "classical"))

    if ltype == "classical":
        from src.lane.classical_lane_detector import (  # noqa: PLC0415
            ClassicalLaneDetector,
        )

        return ClassicalLaneDetector(
            canny_low=int(lane_cfg.get("canny_low", 50)),
            canny_high=int(lane_cfg.get("canny_high", 150)),
            hough_min_votes=int(lane_cfg.get("hough_min_votes", 30)),
            hough_min_len=int(lane_cfg.get("hough_min_len", 40)),
            hough_max_gap=int(lane_cfg.get("hough_max_gap", 100)),
            roi_top_ratio=float(lane_cfg.get("roi_top_ratio", 0.6)),
            roi_bottom_ratio=float(lane_cfg.get("roi_bottom_ratio", 1.0)),
            roi_top_width_ratio=float(lane_cfg.get("roi_top_width_ratio", 0.15)),
        )

    raise ValueError(f"Unknown lane detector type: {ltype!r}")
