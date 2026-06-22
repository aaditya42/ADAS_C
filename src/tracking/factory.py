from __future__ import annotations

from typing import Any

from src.core.base import BaseTracker
from src.tracking.bytetrack_tracker import ByteTrackTracker


class _NoopTracker(BaseTracker):
    """Fallback tracker that passes through detections without tracking."""

    def update(
        self, detections: list, frame: Frame  # noqa: F821
    ) -> list:
        return []


def build_tracker(cfg: dict[str, Any]) -> BaseTracker:
    """Build a tracker from the pipeline configuration.

    The configuration must contain a ``tracker.type`` field. Currently
    supported: ``"bytetrack"``.

    If no ``tracker`` section exists a no-op tracker is returned so the
    pipeline remains safe to use.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        An initialised :class:`BaseTracker` instance.
    """
    trk_cfg = cfg.get("tracker")
    if trk_cfg is None:
        return _NoopTracker()

    trk_type = trk_cfg.get("type", "bytetrack")

    if trk_type == "bytetrack":
        return ByteTrackTracker(
            track_activation_threshold=float(
                trk_cfg.get("track_activation_threshold", 0.25)
            ),
            lost_track_buffer=int(
                trk_cfg.get("lost_track_buffer", 30)
            ),
            minimum_matching_threshold=float(
                trk_cfg.get("minimum_matching_threshold", 0.8)
            ),
            frame_rate=int(trk_cfg.get("frame_rate", 30)),
            minimum_consecutive_frames=int(
                trk_cfg.get("minimum_consecutive_frames", 1)
            ),
        )

    raise ValueError(
        f"Unknown tracker type: '{trk_type}'. Expected one of: 'bytetrack'."
    )
