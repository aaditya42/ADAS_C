from __future__ import annotations

from typing import Any, Optional

from src.signs.base import BaseSignDetector


def build_sign_detector(cfg: dict[str, Any]) -> Optional[BaseSignDetector]:
    """Build a sign detector from the pipeline configuration.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        A ``BaseSignDetector`` instance, or ``None`` if sign detection is
        disabled or no model path is configured.
    """
    signs_cfg = cfg.get("signs", {})
    if not signs_cfg.get("enabled", False):
        return None

    stype = str(signs_cfg.get("type", "yolo_signs"))

    if stype == "yolo_signs":
        from src.signs.yolo_sign_detector import YoloSignDetector  # noqa: PLC0415

        return YoloSignDetector(
            model_path=str(signs_cfg.get("model_path", "")),
            conf_threshold=float(signs_cfg.get("conf_threshold", 0.35)),
            device=str(signs_cfg.get("device", "cpu")),
        )

    raise ValueError(f"Unknown sign detector type: {stype!r}")
