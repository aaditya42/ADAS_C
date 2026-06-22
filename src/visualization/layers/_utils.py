from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.schemas.risk_object import RiskLevel

_RISK_COLORS: dict[RiskLevel, tuple[int, int, int]] = {
    RiskLevel.SAFE: (0, 255, 0),
    RiskLevel.LOW: (0, 255, 255),
    RiskLevel.MEDIUM: (0, 165, 255),
    RiskLevel.HIGH: (0, 0, 255),
}

_COLOR_GREEN = (0, 255, 0)


def hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    hi = (h // 60) % 6
    f = (h % 60) / 60.0
    p = v * (255 - s) // 255
    q = v * (255 - s * f) // 255
    t = v * (255 - s * (1 - f)) // 255
    table = {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }
    r, g, b = table[hi]
    return (b, g, r)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    h = hashlib.md5(str(track_id).encode(), usedforsecurity=False)
    hue = int(h.hexdigest()[:6], 16) % 360
    return hsv_to_bgr(hue, 200, 220)


def risk_color(level: RiskLevel | None) -> tuple[int, int, int]:
    if level is None:
        return _COLOR_GREEN
    return _RISK_COLORS.get(level, _COLOR_GREEN)


def draw_label(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    import cv2  # noqa: PLC0415

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        canvas,
        (x1, y1 - th - 4),
        (x1 + tw + 4, y1),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
