from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseLayer(ABC):
    """Abstract visualisation layer.

    Each layer is a callable that receives a canvas (BGR image) and the
    full pipeline output dict, and returns the (possibly modified) canvas.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @abstractmethod
    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        ...
