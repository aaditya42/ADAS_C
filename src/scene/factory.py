from __future__ import annotations

from typing import Any

from src.scene.scene_builder import SceneBuilder


def build_scene_builder(cfg: dict[str, Any]) -> SceneBuilder:
    """Build a ``SceneBuilder`` from the pipeline configuration.

    Args:
        cfg: Full pipeline configuration dictionary.

    Returns:
        A ``SceneBuilder`` instance.
    """
    scene_cfg = cfg.get("scene", {})
    return SceneBuilder(
        sign_latch_frames=int(scene_cfg.get("sign_latch_frames", 30)),
    )
