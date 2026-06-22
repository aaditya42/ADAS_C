from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.schemas import Detection, Frame, TrackedObject


class BaseFrameSource(ABC):
    """Abstract video frame provider.

    Contract:
      - read() returns the next Frame or None when the stream ends.
      - read() blocks until a frame is available or the stream terminates.
      - release() frees any underlying resources (camera, file handle, etc.).
      - The source supports context-manager usage (with block).
      - Implementations should set Frame.source to a meaningful identifier.
    """

    @abstractmethod
    def read(self) -> Optional[Frame]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def __enter__(self) -> BaseFrameSource:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class BaseDetector(ABC):
    """Abstract object detector.

    Contract:
      - detect() accepts a Frame and returns a list of Detection objects.
      - The list may be empty if nothing is detected.
      - Implementations MUST NOT mutate the input Frame.
      - Implementations SHOULD be side-effect-free (no drawing, no logging to
        stdout beyond the configured logger).
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        ...


class BaseTracker(ABC):
    """Abstract multi-object tracker.

    Contract:
      - update() accepts the latest detections and the current Frame, and
        returns a list of TrackedObject instances with persistent track IDs.
      - The tracker maintains internal state across calls (tracks are
        propagated forward in time).
      - The returned list may be empty if there are no active tracks.
      - Implementations MUST NOT mutate the input Frame or detections.
    """

    @abstractmethod
    def update(
        self, detections: list[Detection], frame: Frame
    ) -> list[TrackedObject]:
        ...


class BaseDepthEstimator(ABC):
    """Abstract monocular depth estimator.

    Contract:
      - estimate() takes a Frame and returns a depth map as a 2-D numpy array
        of shape (H, W) with dtype float32. Each value represents the estimated
        depth in meters (or relative depth — the docstring of each implementation
        should clarify).
      - The output depth map has the same spatial resolution as the input image.
      - Implementations MUST NOT mutate the input Frame.
    """

    @abstractmethod
    def estimate(self, frame: Frame) -> np.ndarray:
        ...


class BaseDistanceEstimator(ABC):
    """Abstract distance estimator that converts depth to metric distances.

    Contract:
      - estimate() takes a depth map and a reference point (x, y) in pixel
        coordinates and returns the estimated distance in meters.
      - The implementation accounts for camera intrinsics if available.
    """

    @abstractmethod
    def estimate(
        self, depth_map: np.ndarray, x: float, y: float
    ) -> float:
        ...


class BaseLaneDetector(ABC):
    """Abstract lane detection module.

    Contract:
      - detect() takes a Frame and returns a LaneInfo object describing the
        lane boundaries in the current frame.
      - If no lanes are visible, the returned LaneInfo should have None for
        all lanes and a low confidence.
      - Implementations MUST NOT mutate the input Frame.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> LaneInfo:
        ...  # noqa: F821 — forward ref resolved at runtime


class BaseSignDetector(ABC):
    """Abstract traffic sign detector.

    Contract:
      - detect() takes a Frame and returns a list of TrafficSignInfo objects.
      - The list may be empty if no signs are visible.
      - Implementations MUST NOT mutate the input Frame.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[TrafficSignInfo]:
        ...  # noqa: F821 — forward ref resolved at runtime
