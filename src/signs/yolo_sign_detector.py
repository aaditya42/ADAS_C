from __future__ import annotations

import re
from typing import Any, Optional

from src.schemas import BoundingBox, Frame
from src.schemas.traffic_sign_info import SignType, TrafficSignInfo
from src.signs.base import BaseSignDetector
from src.utils.logger import get_logger

DEFAULT_MAP: dict[str, SignType] = {
    "stop": SignType.STOP,
    "yield": SignType.YIELD,
    "no_entry": SignType.NO_ENTRY,
    "one_way": SignType.ONE_WAY,
    "pedestrian_crossing": SignType.PEDESTRIAN_CROSSING,
    "crosswalk": SignType.PEDESTRIAN_CROSSING,
    "speed_limit": SignType.SPEED_LIMIT,
    "traffic_light_red": SignType.TRAFFIC_LIGHT_RED,
    "traffic_light_green": SignType.TRAFFIC_LIGHT_GREEN,
    "traffic_light_yellow": SignType.TRAFFIC_LIGHT_YELLOW,
    "traffic_light": SignType.TRAFFIC_LIGHT_RED,
}


class YoloSignDetector(BaseSignDetector):
    """Traffic sign detector powered by a separate Ultralytics YOLO model.

    Mirrors the structure of ``YoloDetector`` (Phase 2) but is an independent
    class so the two can evolve separately.

    Class names from the model are mapped to ``SignType`` via
    ``class_name_to_sign``.  If a class name carries a numeric suffix (e.g.
    ``"speed_limit_50"``) the suffix is parsed as the sign's ``value``.

    The model is loaded lazily on the first call to ``detect()`` so that
    importing this module does not require ``ultralytics`` unless used.

    Args:
        model_path: Path or name of the YOLO weights file.
        conf_threshold: Confidence threshold for detections.
        device: Inference device (``"cpu"``, ``"cuda:0"``, etc.).
        class_name_to_sign: Mapping from YOLO class names to ``SignType``.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        device: str = "cpu",
        class_name_to_sign: Optional[dict[str, SignType]] = None,
    ) -> None:
        self._model_path = model_path
        self._conf = conf_threshold
        self._device = device
        self._name_map: dict[str, SignType] = (
            class_name_to_sign if class_name_to_sign is not None else DEFAULT_MAP
        )
        self._logger = get_logger(f"{__name__}.YoloSignDetector")

        self._model: Any = None
        self._names: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    #  BaseSignDetector
    # ------------------------------------------------------------------ #

    def detect(self, frame: Frame) -> list[TrafficSignInfo]:
        if self._model is None:
            self._load_model()

        results = self._model.predict(
            frame.image,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        signs: list[TrafficSignInfo] = []
        cls_ids: list[int] = boxes.cls.int().tolist()
        confs: list[float] = boxes.conf.float().tolist()
        xyxy: list[list[float]] = boxes.xyxy.float().tolist()

        for cls_id, conf, (x1, y1, x2, y2) in zip(cls_ids, confs, xyxy):
            class_name = self._names.get(cls_id, str(cls_id))
            sign_type, value = self._name_to_sign(class_name)

            signs.append(
                TrafficSignInfo(
                    sign_type=sign_type,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=conf,
                    value=value,
                )
            )

        return signs

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        from ultralytics import YOLO  # noqa: PLC0415 — isolated import

        self._logger.info(
            "Loading traffic-sign YOLO model from %s ...", self._model_path
        )
        self._model = YOLO(self._model_path)
        self._names = self._model.names
        self._logger.info(
            "Traffic-sign YOLO loaded  |  device=%s  |  %d classes",
            self._device,
            len(self._names),
        )

    def _name_to_sign(self, class_name: str) -> tuple[SignType, Optional[float]]:
        """Convert a YOLO class name to (SignType, value).

        Tries an exact match first.  If that fails, strips a trailing numeric
        suffix (e.g. ``"_50"``) and looks up the base name; the number becomes
        the sign's ``value``.
        """
        if class_name in self._name_map:
            return self._name_map[class_name], None

        m = re.match(r"^(.+?)_(\d+(?:\.\d+)?)$", class_name)
        if m:
            base, num_str = m.groups()
            if base in self._name_map:
                return self._name_map[base], float(num_str)

        self._logger.debug("Unmapped sign class name: %s", class_name)
        return SignType.UNKNOWN, None
