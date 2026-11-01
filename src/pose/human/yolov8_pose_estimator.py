from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.pose.base import BaseHumanPoseEstimator
from src.pose.human.postprocess import (
    build_human_pose,
    decode_onnx_output,
)
from src.pose.human.preprocess import (
    crop_person,
    letterbox,
    map_keypoints_to_original,
    normalize_for_onnx,
)
from src.schemas import Detection, TrackedObject
from src.schemas.pose import HumanPose
from src.utils.logger import get_logger

# Default inference size for YOLOv8-Pose models.
_DEFAULT_INPUT_SIZE: int = 640


# --------------------------------------------------------------------------- #
#  Backend protocol (internal)
# --------------------------------------------------------------------------- #


class _Backend(ABC):
    """Internal protocol for PyTorch / ONNX model execution."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before the first run."""

    @abstractmethod
    def run(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Run pose inference on a preprocessed image.

        Args:
            image: BGR uint8 image of shape ``(H, W, 3)``, already
                letterboxed to the model's input size.

        Returns:
            List of ``(keypoints, bbox, score)`` where keypoints has shape
            ``(17, 3)`` in the **model's output coordinate space** (which
            matches the letterboxed input space).  May be empty.
        """


# --------------------------------------------------------------------------- #
#  Torch backend
# --------------------------------------------------------------------------- #


class _TorchBackend(_Backend):
    """Backend that runs YOLOv8-Pose via the Ultralytics PyTorch engine."""

    def __init__(self, model_path: str, device: str, conf_threshold: float) -> None:
        self._model_path = model_path
        self._device = device
        self._conf_threshold = conf_threshold
        self._model: Any = None
        self._logger = get_logger(f"{__name__}._TorchBackend")

    def load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PyTorch backend requires 'ultralytics'. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._logger.info("Loading YOLOv8-Pose model (torch) from %s ...", self._model_path)
        self._model = YOLO(self._model_path)
        self._logger.info(
            "YOLOv8-Pose loaded  |  device=%s", self._device
        )

    def run(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        if self._model is None:
            self.load()

        results = self._model.predict(
            image,
            conf=self._conf_threshold,
            iou=0.65,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].keypoints is None or results[0].boxes is None:
            return []

        boxes_np: np.ndarray = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        confs_np: np.ndarray = results[0].boxes.conf.cpu().numpy()  # (N,)
        kpts_np: np.ndarray = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        out: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for i in range(len(kpts_np)):
            out.append((kpts_np[i], boxes_np[i], float(confs_np[i])))

        return out


# --------------------------------------------------------------------------- #
#  ONNX backend
# --------------------------------------------------------------------------- #


class _OnnxBackend(_Backend):
    """Backend that runs YOLOv8-Pose via ONNX Runtime."""

    def __init__(
        self, model_path: str, conf_threshold: float, iou_threshold: float
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._session: Any = None
        self._input_name: str = ""
        self._logger = get_logger(f"{__name__}._OnnxBackend")

    def load(self) -> None:
        try:
            import onnxruntime  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ONNX backend requires 'onnxruntime'. "
                "Install it with: pip install onnxruntime"
            ) from exc

        self._logger.info("Loading YOLOv8-Pose model (onnx) from %s ...", self._model_path)
        self._session = onnxruntime.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._logger.info("ONNX session ready  |  input=%s", self._session.get_inputs()[0].shape)

    def run(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        if self._session is None:
            self.load()

        blob = normalize_for_onnx(image)
        ort_inputs = {self._input_name: blob}
        raw = self._session.run(None, ort_inputs)[0]

        return decode_onnx_output(raw, self._conf_threshold, self._iou_threshold)


# --------------------------------------------------------------------------- #
#  Backend factory
# --------------------------------------------------------------------------- #


def _create_backend(
    model_path: str, device: str, conf_threshold: float, iou_threshold: float
) -> _Backend:
    """Select and instantiate the appropriate backend based on file extension."""
    ext = Path(model_path).suffix.lower()

    if ext in (".pt", ".pth", ".torchscript"):
        return _TorchBackend(model_path, device, conf_threshold)

    if ext in (".onnx",):
        return _OnnxBackend(model_path, conf_threshold, iou_threshold)

    raise ValueError(
        f"Unrecognised model file extension: {ext!r}. "
        f"Expected .pt, .pth, or .onnx.  Path: {model_path}"
    )


# --------------------------------------------------------------------------- #
#  YOLOv8HumanPoseEstimator
# --------------------------------------------------------------------------- #


class YOLOv8HumanPoseEstimator(BaseHumanPoseEstimator):
    """Human pose estimator powered by YOLOv8-Pose (top-down).

    Uses a top-down strategy: person detections from the upstream detector
    are used to crop the image, and the YOLOv8-Pose model runs on each crop
    to regress 17 COCO keypoints.

    Supports both PyTorch (``.pt``) and ONNX Runtime (``.onnx``) backends,
    selected automatically by the model file extension.

    The model is loaded **lazily** on the first ``estimate()`` call.  A
    ``threading.Lock`` ensures thread-safe lazy initialisation.

    Args:
        model_path: Path to the YOLOv8-Pose weights file.
        device: Inference device string (``"cpu"``, ``"cuda:0"``).
        conf_threshold: Minimum keypoint confidence (also used as detection
            confidence threshold).
        iou_threshold: NMS IoU threshold for pose output.
        input_size: Model input size in pixels (square).
        max_persons: Maximum number of persons to process per frame
            (highest-confidence detections are kept).
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.65,
        input_size: int = _DEFAULT_INPUT_SIZE,
        max_persons: int = 10,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._max_persons = max_persons
        self._logger = get_logger(f"{__name__}.YOLOv8HumanPoseEstimator")

        self._backend: Optional[_Backend] = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  BaseHumanPoseEstimator
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
    ) -> Sequence[HumanPose]:
        # Filter person-class detections and sort by confidence descending
        person_dets: List[Tuple[Detection, int]] = [
            (d, i) for i, d in enumerate(detections) if d.class_name.lower() == "person"
        ]
        person_dets.sort(key=lambda x: x[0].confidence, reverse=True)

        # Respect max_persons
        person_dets = person_dets[: self._max_persons]

        if not person_dets:
            return ()

        if not self._loaded:
            self._init_backend()

        timestamp = 0.0
        poses: List[HumanPose] = []

        for person_pos, (det, det_idx) in enumerate(person_dets):
            try:
                bbox = (
                    float(det.bbox.x1),
                    float(det.bbox.y1),
                    float(det.bbox.x2),
                    float(det.bbox.y2),
                )

                # 1. Crop person from image
                crop, crop_x1, crop_y1 = crop_person(frame, bbox, margin=1.2)
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                # 2. Letterbox crop to model input size
                padded, scale, pad_left, pad_top = letterbox(
                    crop, target_size=(self._input_size, self._input_size)
                )

                # 3. Run pose inference
                assert self._backend is not None  # mypy: guarded by self._loaded
                detections_out = self._backend.run(padded)

                if not detections_out:
                    continue

                # Take the highest-confidence detection from the pose model
                kpts, _, score = detections_out[0]

                # 4. Map keypoints from padded-crop space to original image space
                kpts_orig = map_keypoints_to_original(
                    kpts, scale, pad_left, pad_top, crop_x1, crop_y1
                )

                # 5. Build HumanPose using the original detection bbox.
                #    Match track by position in the filtered person list.
                matched_track_id: Optional[int] = (
                    tracks[person_pos].track_id
                    if person_pos < len(tracks)
                    else None
                )
                pose = build_human_pose(
                    kpts_array=kpts_orig,
                    bbox_array=np.array([bbox[0], bbox[1], bbox[2], bbox[3]]),
                    overall_confidence=score,
                    detection_id=det_idx,
                    track_id=matched_track_id,
                    timestamp=timestamp,
                    conf_threshold=self._conf_threshold,
                )
                poses.append(pose)

            except Exception:
                self._logger.exception(
                    "Failed to estimate pose for detection %d; skipping",
                    det_idx,
                )
                continue

        return poses

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _init_backend(self) -> None:
        """Thread-safe lazy initialisation of the inference backend."""
        with self._lock:
            if self._loaded:
                return
            self._backend = _create_backend(
                model_path=self._model_path,
                device=self._device,
                conf_threshold=self._conf_threshold,
                iou_threshold=self._iou_threshold,
            )
            self._backend.load()
            self._loaded = True
