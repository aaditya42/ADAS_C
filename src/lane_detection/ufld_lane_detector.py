from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.lane_detection.base_lane_detector import BaseLaneDetector
from src.lane_detection.interfaces import LaneDetectionResult
from src.utils.logger import get_logger


class UFLDLaneDetector(BaseLaneDetector):
    """Ultra Fast Lane Detection (UFLD) running via ONNX Runtime.

    Expects a UFLD model exported to ONNX with:
      - Input:  ``(1, 3, input_height, input_width)``
      - Output: ``(1, num_lanes, num_points, num_cols)``

    Post-processing:
      1. Softmax over the last axis to get column probabilities.
      2. Argmax at each row anchor for each lane candidate.
      3. Confidence filtering via configurable threshold.
      4. Polynomial fitting (2nd order, ``x = a*y² + b*y + c``) to match
         the convention used by lane layers and risk scoring.

    Args:
        model_path: Path to the ``.onnx`` model file.
        input_width: Model input width in pixels.
        input_height: Model input height in pixels.
        conf_threshold: Minimum confidence for a lane to be accepted.
        num_points: Number of row anchors (must match model output).
        num_cols: Number of column grid cells (must match model output).
        num_lanes: Number of lane candidates in model output.
        providers: ONNX Runtime execution providers.
    """

    def __init__(
        self,
        model_path: str = "weights/ufld_v2.onnx",
        input_width: int = 800,
        input_height: int = 288,
        conf_threshold: float = 0.5,
        num_points: int = 56,
        num_cols: int = 200,
        num_lanes: int = 4,
        providers: list[str] | None = None,
    ) -> None:
        self._model_path = model_path
        self._input_w = input_width
        self._input_h = input_height
        self._conf_threshold = conf_threshold
        self._num_points = num_points
        self._num_cols = num_cols
        self._num_lanes = num_lanes
        self._providers = providers or ["CPUExecutionProvider"]
        self._logger = get_logger(f"{__name__}.UFLDLaneDetector")

        self._session: Any = None
        self._input_name: str = ""
        self._output_name: str = ""

    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        if self._session is None:
            self._load_session()

        blob = self._preprocess(frame)
        ort_inputs = {self._input_name: blob}
        raw = self._session.run([self._output_name], ort_inputs)[0]

        return self._postprocess(raw, frame.shape[1], frame.shape[0])

    def _load_session(self) -> None:
        import onnxruntime  # noqa: PLC0415

        model_path = Path(self._model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"UFLD model not found: {model_path.resolve()}"
            )

        self._logger.info(
            "Loading UFLD ONNX model from %s ...", self._model_path,
        )
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=self._providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        self._logger.info(
            "UFLD ONNX session ready  |  input=%s  |  providers=%s",
            input_shape, self._providers,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2  # noqa: PLC0415

        h, w = frame.shape[:2]
        if (h, w) != (self._input_h, self._input_w):
            resized = cv2.resize(
                frame, (self._input_w, self._input_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = frame

        blob = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        blob = (blob - mean) / std
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        return blob

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> LaneDetectionResult:
        raw = np.squeeze(raw)

        if raw.ndim == 4:
            raw = raw[0]

        probs = self._softmax(raw, axis=-1)

        x_scale = orig_w / self._num_cols
        y_scale = orig_h / self._num_points

        lane_lines: list[tuple[np.ndarray, float]] = []

        for lane_idx in range(min(self._num_lanes, probs.shape[0])):
            lane_probs = probs[lane_idx]
            col_idxs = np.argmax(lane_probs, axis=-1).astype(np.float64)
            max_probs = np.max(lane_probs, axis=-1)

            valid = max_probs >= self._conf_threshold
            if not valid.any():
                continue

            y_vals = np.where(valid)[0].astype(np.float64) * y_scale
            x_vals = col_idxs[valid] * x_scale

            if len(y_vals) < 3:
                continue

            A = np.column_stack([y_vals ** 2, y_vals, np.ones_like(y_vals)])
            coeffs, *_ = np.linalg.lstsq(A, x_vals, rcond=None)

            residuals = np.abs(A @ coeffs - x_vals)
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            confidence = float(np.clip(1.0 - rmse / (orig_w * 0.5), 0.0, 1.0))

            lane_lines.append((coeffs, confidence))

        if not lane_lines:
            return LaneDetectionResult.empty()

        lane_lines.sort(key=lambda x: x[1], reverse=True)

        img_center_x = orig_w / 2.0
        left_coeffs: tuple[float, float, float] | None = None
        right_coeffs: tuple[float, float, float] | None = None
        left_conf: float = 0.0
        right_conf: float = 0.0

        for coeffs, conf in lane_lines:
            center_x_at_bottom = (
                coeffs[0] * (orig_h ** 2) + coeffs[1] * orig_h + coeffs[2]
            )
            if center_x_at_bottom < img_center_x and left_coeffs is None:
                left_coeffs = tuple(float(c) for c in coeffs)
                left_conf = conf
            elif center_x_at_bottom >= img_center_x and right_coeffs is None:
                right_coeffs = tuple(float(c) for c in coeffs)
                right_conf = conf

        def _coeffs_to_points(
            coeffs: tuple[float, float, float] | None,
            h: int,
            num_pts: int = 50,
        ) -> tuple[tuple[int, int], ...]:
            if coeffs is None:
                return ()
            y_vals = np.linspace(0, h - 1, num=num_pts, dtype=np.float64)
            x_vals = coeffs[0] * y_vals * y_vals + coeffs[1] * y_vals + coeffs[2]
            pts: list[tuple[int, int]] = []
            for i in range(num_pts):
                xi = int(round(x_vals[i]))
                yi = int(round(y_vals[i]))
                if 0 <= xi < orig_w and 0 <= yi < orig_h:
                    pts.append((xi, yi))
            return tuple(pts)

        left_points = _coeffs_to_points(left_coeffs, orig_h)
        right_points = _coeffs_to_points(right_coeffs, orig_h)

        center_points: tuple[tuple[int, int], ...] = ()
        ego_lane_center_x: float | None = None
        if left_points and right_points:
            min_len = min(len(left_points), len(right_points))
            center_points = tuple(
                (
                    (left_points[i][0] + right_points[i][0]) // 2,
                    (left_points[i][1] + right_points[i][1]) // 2,
                )
                for i in range(min_len)
            )
            max_y_l = max(p[1] for p in left_points)
            max_y_r = max(p[1] for p in right_points)
            lx = next((p[0] for p in left_points if p[1] == max_y_l), None)
            rx = next((p[0] for p in right_points if p[1] == max_y_r), None)
            if lx is not None and rx is not None:
                ego_lane_center_x = (float(lx) + float(rx)) / 2.0

        confidences = [c for c in (left_conf, right_conf) if c > 0]
        overall_conf = float(np.mean(confidences)) if confidences else 0.0

        return LaneDetectionResult(
            left_lane_points=left_points,
            right_lane_points=right_points,
            center_lane_points=center_points,
            lane_count=len([c for c in (left_coeffs, right_coeffs) if c is not None]),
            ego_lane_center_x=ego_lane_center_x,
            confidence=overall_conf,
        )

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
