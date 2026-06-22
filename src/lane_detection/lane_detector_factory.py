from __future__ import annotations

import logging
from typing import Any

from src.lane_detection.base_lane_detector import BaseLaneDetector
from src.lane_detection.lane_models import ModelType

logger = logging.getLogger(__name__)


class _NoopLaneDetector(BaseLaneDetector):
    """Fallback detector that always returns an empty result."""

    def detect(self, frame: Any) -> Any:
        from src.lane_detection.interfaces import LaneDetectionResult
        return LaneDetectionResult.empty()


class LaneDetectorFactory:
    """Factory for lane detection models.

    Usage::

        detector = LaneDetectorFactory.create(config)

    To add a new model type:
      1. Create a new detector module in ``src/lane_detection/``.
      2. Register it with ``LaneDetectorFactory.register(name, cls)``.

    No pipeline or orchestrator changes are needed.
    """

    _registry: dict[str, type[BaseLaneDetector]] = {}

    @classmethod
    def register(cls, model_type: str, detector_cls: type[BaseLaneDetector]) -> None:
        cls._registry[model_type] = detector_cls

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseLaneDetector | None:
        lane_cfg = config.get("lane", {})
        if not lane_cfg.get("enabled", True):
            return None

        model_type = str(lane_cfg.get("model_type", ModelType.CLRNet.value))

        if model_type == ModelType.CLRNet.value:
            cls.register(ModelType.CLRNet.value, _get_clrnet_cls())

        # Keep UFLD registrable for backward compatibility
        if model_type == ModelType.UFLD.value:
            cls.register(ModelType.UFLD.value, _get_ufld_cls())

        detector_cls = cls._registry.get(model_type)
        if detector_cls is None:
            raise ValueError(
                f"Unknown lane detector model_type: '{model_type}'. "
                f"Supported: {list(cls._registry.keys())}"
            )

        return _build_detector(detector_cls, lane_cfg)


def _get_clrnet_cls() -> type[BaseLaneDetector]:
    from src.lane_detection.clrnet_lane_detector import CLRNetLaneDetector  # noqa: PLC0415
    return CLRNetLaneDetector


def _get_ufld_cls() -> type[BaseLaneDetector]:
    from src.lane_detection.ufld_lane_detector import UFLDLaneDetector  # noqa: PLC0415
    return UFLDLaneDetector


def _build_detector(
    detector_cls: type[BaseLaneDetector],
    lane_cfg: dict[str, Any],
) -> BaseLaneDetector:
    """Construct a lane detector from configuration, passing only
    the parameters that the concrete class accepts.
    """
    import inspect  # noqa: PLC0415

    sig = inspect.signature(detector_cls.__init__)
    param_names = set(sig.parameters.keys()) - {"self"}

    kwargs: dict[str, Any] = {}
    # Core parameters
    core_params: dict[str, str] = {
        "model_path": "model_path",
        "input_width": "input_width",
        "input_height": "input_height",
        "conf_threshold": "conf_threshold",
    }
    for cfg_key, param_name in core_params.items():
        if param_name in param_names and cfg_key in lane_cfg:
            kwargs[param_name] = lane_cfg[cfg_key]

    # CLRNet-specific parameters
    clrnet_params: dict[str, str] = {
        "lane_confidence_threshold": "lane_confidence_threshold",
        "max_lanes": "max_lanes",
        "temporal_smoothing": "temporal_smoothing",
        "smoothing_window": "smoothing_window",
        "cut_height": "cut_height",
    }
    for cfg_key, param_name in clrnet_params.items():
        if param_name in param_names and cfg_key in lane_cfg:
            kwargs[param_name] = lane_cfg[cfg_key]

    logger.debug(
        "Building lane detector %s with kwargs=%s",
        detector_cls.__name__,
        kwargs,
    )
    return detector_cls(**kwargs)
