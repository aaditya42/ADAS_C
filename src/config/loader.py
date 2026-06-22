from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_RE.sub(_replace, value)


def _walk_and_interpolate(obj: Any) -> Any:
    """Recursively interpolate environment variables in string values."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_interpolate(v) for v in obj]
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with environment-variable interpolation.

    Patterns of the form ``${VAR_NAME}`` in string values are replaced with the
    corresponding environment variable at load time.

    Args:
        path: Path to the YAML config file.

    Returns:
        A dictionary containing the resolved configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return _walk_and_interpolate(raw)
