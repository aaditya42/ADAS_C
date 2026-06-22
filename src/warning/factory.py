from __future__ import annotations

from typing import Any

from src.warning.channels import AudioBeepChannel, ConsoleChannel
from src.warning.warning_manager import WarningManager


_CHANNEL_MAP: dict[str, type] = {
    "console": ConsoleChannel,
    "audio_beep": AudioBeepChannel,
}


def build_warning_manager(config: dict[str, Any]) -> WarningManager:
    """Build a WarningManager from the configuration.

    Args:
        config: The full pipeline configuration dictionary.

    Returns:
        A configured ``WarningManager`` instance.
    """
    warn_cfg = config.get("warning", {})
    channel_names: list[str] = list(warn_cfg.get("channels", ["console"]))

    channels = []
    for name in channel_names:
        cls = _CHANNEL_MAP.get(name)
        if cls is not None:
            channels.append(cls())

    return WarningManager(
        channels=channels,
        min_level=str(warn_cfg.get("min_level", "MEDIUM")),
        cooldown_sec=float(warn_cfg.get("cooldown_sec", 2.0)),
        hysteresis_frames=int(warn_cfg.get("hysteresis_frames", 3)),
    )
