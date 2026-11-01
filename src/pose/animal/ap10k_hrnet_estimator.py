from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.pose.animal.postprocess import build_animal_pose
from src.pose.animal.preprocess import (
    crop_animal_region,
    heatmap_to_keypoints,
    letterbox,
    map_keypoints_to_original,
    normalize_for_onnx,
    scale_keypoints_to_crop,
)
from src.pose.animal.species_mapper import SpeciesMapper
from src.pose.base import BaseAnimalPoseEstimator
from src.schemas import Detection, TrackedObject
from src.schemas.pose import AnimalPose
from src.utils.logger import get_logger

_DEFAULT_INPUT_SIZE: Tuple[int, int] = (256, 256)
_DEFAULT_HEATMAP_SIZE: Tuple[int, int] = (64, 64)


# --------------------------------------------------------------------------- #
#  Backend protocol (internal)
# --------------------------------------------------------------------------- #


class _Backend(ABC):
    """Internal protocol for ONNX / PyTorch model execution."""

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
#  ONNX backend (primary)
# --------------------------------------------------------------------------- #


class _OnnxBackend(_Backend):
    """Backend that runs an AP-10K HRNet model via ONNX Runtime.

    The model is expected to output heatmaps of shape ``(1, 17, H, W)``
    which are then argmax-decoded into image-space keypoints.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        conf_threshold: float,
    ) -> None:
        self._model_path = model_path
        self._input_size = input_size
        self._heatmap_size = heatmap_size
        self._conf_threshold = conf_threshold
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

        self._logger.info(
            "Loading AP-10K HRNet model (onnx) from %s ...", self._model_path
        )
        self._session = onnxruntime.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._logger.info(
            "ONNX session ready  |  input=%s  output=%s",
            self._session.get_inputs()[0].shape,
            self._session.get_outputs()[0].shape,
        )

    def run(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        if self._session is None:
            self.load()

        blob = normalize_for_onnx(image)
        ort_inputs = {self._input_name: blob}
        raw = self._session.run(None, ort_inputs)[0]  # (1, 17, H, W)

        # Decode heatmaps → (17, 3) keypoints in heatmap space
        kpts_heatmap = heatmap_to_keypoints(raw, self._conf_threshold)

        # Scale from heatmap space to crop-image space
        kpts_crop = scale_keypoints_to_crop(
            kpts_heatmap,
            heatmap_size=self._heatmap_size,
            crop_size=self._input_size,
        )

        # Build pseudo-bbox from keypoint extent
        visible_mask = kpts_crop[:, 2] >= self._conf_threshold
        if np.any(visible_mask):
            xs = kpts_crop[visible_mask, 0]
            ys = kpts_crop[visible_mask, 1]
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
        else:
            x1 = y1 = 0.0
            x2 = y2 = float(self._input_size[1])

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        score = float(np.mean(kpts_crop[:, 2]))

        return [(kpts_crop, bbox, score)]


# --------------------------------------------------------------------------- #
#  PyTorch backend (stub)
# --------------------------------------------------------------------------- #


class _TorchBackend(_Backend):
    """Backend stub that runs an AP-10K HRNet model via PyTorch.

    This is a minimal stub intended for completeness.  It uses
    ``torch.jit.load`` to run a TorchScript-serialised model.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        conf_threshold: float,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._input_size = input_size
        self._heatmap_size = heatmap_size
        self._conf_threshold = conf_threshold
        self._model: Any = None
        self._logger = get_logger(f"{__name__}._TorchBackend")

    def load(self) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PyTorch backend requires 'torch'. "
                "Install it with: pip install torch"
            ) from exc

        self._logger.info(
            "Loading AP-10K HRNet model (torch) from %s ...", self._model_path
        )
        self._model = torch.jit.load(self._model_path, map_location=self._device)  # type: ignore[no-untyped-call]
        self._model.eval()
        self._logger.info(
            "PyTorch model loaded  |  device=%s", self._device
        )

    def run(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        if self._model is None:
            self.load()

        import torch  # noqa: PLC0415

        blob = normalize_for_onnx(image)  # same shape (1, 3, H, W)
        tensor = torch.from_numpy(blob).to(self._device)
        with torch.no_grad():
            raw = self._model(tensor).cpu().numpy()  # (1, 17, H, W)

        kpts_heatmap = heatmap_to_keypoints(raw, self._conf_threshold)
        kpts_crop = scale_keypoints_to_crop(
            kpts_heatmap,
            heatmap_size=self._heatmap_size,
            crop_size=self._input_size,
        )

        visible_mask = kpts_crop[:, 2] >= self._conf_threshold
        if np.any(visible_mask):
            xs = kpts_crop[visible_mask, 0]
            ys = kpts_crop[visible_mask, 1]
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
        else:
            x1 = y1 = 0.0
            x2 = y2 = float(self._input_size[1])

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        score = float(np.mean(kpts_crop[:, 2]))

        return [(kpts_crop, bbox, score)]


# --------------------------------------------------------------------------- #
#  Backend factory
# --------------------------------------------------------------------------- #


def _create_backend(
    model_path: str,
    device: str,
    input_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    conf_threshold: float,
) -> _Backend:
    """Select and instantiate the appropriate backend based on file extension."""
    ext = Path(model_path).suffix.lower()

    if ext in (".onnx",):
        return _OnnxBackend(
            model_path=model_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            conf_threshold=conf_threshold,
        )

    if ext in (".pt", ".pth", ".torchscript"):
        return _TorchBackend(
            model_path=model_path,
            device=device,
            input_size=input_size,
            heatmap_size=heatmap_size,
            conf_threshold=conf_threshold,
        )

    raise ValueError(
        f"Unrecognised model file extension: {ext!r}. "
        f"Expected .pt, .pth, or .onnx.  Path: {model_path}"
    )


# --------------------------------------------------------------------------- #
#  AP10KHRNetAnimalPoseEstimator
# --------------------------------------------------------------------------- #


class AP10KHRNetAnimalPoseEstimator(BaseAnimalPoseEstimator):
    """Animal pose estimator powered by HRNet (AP-10K, top-down).

    Uses a top-down strategy: animal detections from the upstream detector
    are filtered by the configured species mapper, cropped, and the HRNet
    model runs on each crop to regress 17 AP-10K keypoints.

    Supports both ONNX Runtime (``.onnx``) and PyTorch (``.pt`` / ``.pth`` /
    ``.torchscript``) backends, selected automatically by the model file
    extension.  ONNX is the primary backend.

    The model is loaded **lazily** on the first ``estimate()`` call.  A
    ``threading.Lock`` ensures thread-safe lazy initialisation.

    Args:
        model_path: Path to the AP-10K HRNet weights file.
        device: Inference device string (``"cpu"``, ``"cuda:0"``).
        conf_threshold: Minimum keypoint confidence to mark as visible.
        input_size: ``(height, width)`` input resolution for the model.
        heatmap_size: ``(height, width)`` of the output heatmaps.
        max_animals: Maximum number of animals to process per frame.
        species_mapper: Mapper from COCO class IDs to species strings.
    """

    def __init__(
        self,
        model_path: str = "ap10k_hrnet.onnx",
        device: str = "cpu",
        conf_threshold: float = 0.5,
        input_size: Tuple[int, int] = _DEFAULT_INPUT_SIZE,
        heatmap_size: Tuple[int, int] = _DEFAULT_HEATMAP_SIZE,
        max_animals: int = 10,
        species_mapper: Optional[SpeciesMapper] = None,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._conf_threshold = conf_threshold
        self._input_size = input_size
        self._heatmap_size = heatmap_size
        self._max_animals = max_animals
        self._species_mapper = species_mapper or SpeciesMapper()
        self._logger = get_logger(f"{__name__}.AP10KHRNetAnimalPoseEstimator")

        self._backend: Optional[_Backend] = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  BaseAnimalPoseEstimator
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
    ) -> Sequence[AnimalPose]:
        # Filter animal detections by species mapper
        animal_dets: List[Tuple[Detection, int, str]] = []
        for i, d in enumerate(detections):
            species = self._species_mapper.map(d.class_id)
            if species is not None and self._species_mapper.is_supported(species):
                animal_dets.append((d, i, species))

        animal_dets.sort(key=lambda x: x[0].confidence, reverse=True)
        animal_dets = animal_dets[: self._max_animals]

        if not animal_dets:
            return ()

        if not self._loaded:
            self._init_backend()

        timestamp = time.time()
        poses: List[AnimalPose] = []
        animal_pos = 0

        for det, det_idx, species in animal_dets:
            try:
                bbox = (
                    float(det.bbox.x1),
                    float(det.bbox.y1),
                    float(det.bbox.x2),
                    float(det.bbox.y2),
                )

                crop, crop_x1, crop_y1 = crop_animal_region(
                    frame, bbox, margin=1.3
                )
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                padded, scale, pad_left, pad_top = letterbox(
                    crop, target_size=self._input_size
                )

                assert self._backend is not None
                t0 = time.perf_counter()
                detections_out = self._backend.run(padded)
                elapsed = time.perf_counter() - t0
                self._logger.debug(
                    "Animal pose inference for detection %d took %.1f ms",
                    det_idx,
                    elapsed * 1000,
                )

                if not detections_out:
                    continue

                kpts, _, score = detections_out[0]

                kpts_orig = map_keypoints_to_original(
                    kpts, scale, pad_left, pad_top, crop_x1, crop_y1
                )

                matched_track_id: Optional[int] = (
                    tracks[animal_pos].track_id
                    if animal_pos < len(tracks)
                    else None
                )
                pose = build_animal_pose(
                    kpts_array=kpts_orig,
                    bbox_array=np.array(
                        [bbox[0], bbox[1], bbox[2], bbox[3]]
                    ),
                    overall_confidence=score,
                    detection_id=det_idx,
                    track_id=matched_track_id,
                    timestamp=timestamp,
                    conf_threshold=self._conf_threshold,
                    species=species,
                )
                poses.append(pose)
                animal_pos += 1

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
                input_size=self._input_size,
                heatmap_size=self._heatmap_size,
                conf_threshold=self._conf_threshold,
            )
            self._backend.load()
            self._loaded = True
