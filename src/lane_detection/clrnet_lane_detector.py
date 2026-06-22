from __future__ import annotations

import logging
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.lane_detection.base_lane_detector import BaseLaneDetector
from src.lane_detection.interfaces import LaneDetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  CLRNet path setup — allows importing from external/CLRNet without manual
#  PYTHONPATH manipulation.
# ---------------------------------------------------------------------------
_CLRNET_ROOT = Path(__file__).resolve().parents[2] / "external" / "CLRNet"
if str(_CLRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLRNET_ROOT))

# ---------------------------------------------------------------------------
#  Mock for the un-compiled ``nms_impl`` CUDA extension.
#
#  CLRNet ships a custom CUDA NMS kernel that must be compiled via
#  setup.py.  Because we cannot rely on the compiled extension being
#  present, we install a pure-Python fallback *before* the CLRNet
#  package is imported.
# ---------------------------------------------------------------------------

def _install_nms_fallback() -> None:
    """Install a pure-Python ``nms_impl`` module in ``sys.modules`` so that
    ``clrnet.ops.nms`` can be imported without the compiled CUDA extension.
    """
    if "clrnet.ops.nms_impl" in sys.modules:
        return

    mod = types.ModuleType("clrnet.ops.nms_impl")

    def nms_forward(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        overlap: float,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure-Python lane NMS (mirrors the CUDA kernel logic).

        Args:
            boxes:  (N, 77) tensor
                    ``[cls_neg, cls_pos, start_y, start_x, length, x_offsets...]``
                    where x_offsets are scaled to pixel space and length is in
                    strip units.
            scores: (N,) confidence scores.
            overlap: Distance threshold in pixel space.
            top_k: Maximum number of lanes to keep.

        Returns:
            (keep, num_to_keep, parent_object_index)
        """
        n_offsets = 72
        n_strips = n_offsets - 1
        device = boxes.device
        N = boxes.shape[0]

        _, idx = scores.sort(descending=True)
        boxes = boxes[idx]

        keep_list: list[int] = []

        for i in range(N):
            if len(keep_list) >= top_k:
                break
            suppressed = False

            start_i = int(round(boxes[i, 2].item() * n_strips))
            len_i = int(round(boxes[i, 4].item()))
            end_i = min(start_i + len_i - 1, n_offsets - 1)

            for j in keep_list:
                start_j = int(round(boxes[j, 2].item() * n_strips))
                len_j = int(round(boxes[j, 4].item()))
                end_j = min(start_j + len_j - 1, n_offsets - 1)

                start = max(start_i, start_j)
                end = min(end_i, end_j)
                if end < start:
                    continue

                dist = 0.0
                for k in range(start, end + 1):
                    dist += abs(boxes[i, 5 + k].item() - boxes[j, 5 + k].item())
                avg = dist / (end - start + 1)

                if avg < overlap:
                    suppressed = True
                    break

            if not suppressed:
                keep_list.append(idx[i].item())

        keep = torch.tensor(keep_list, device=device, dtype=torch.long)
        num_to_keep = torch.tensor(len(keep_list), device=device, dtype=torch.long)
        parent = torch.zeros(N, device=device, dtype=torch.long)

        return keep, num_to_keep, parent

    mod.nms_forward = nms_forward
    sys.modules["clrnet.ops.nms_impl"] = mod


_install_nms_fallback()

# ---------------------------------------------------------------------------
#  CLRNet imports  (must come *after* the nms_impl mock)
# ---------------------------------------------------------------------------
from clrnet.utils.config import Config  # noqa: E402
from clrnet.models.registry import build_net  # noqa: E402


# ---------------------------------------------------------------------------
#  Detector
# ---------------------------------------------------------------------------

class CLRNetLaneDetector(BaseLaneDetector):
    """CLRNet lane detector using native PyTorch inference.

    Loads the CLRNet model directly from a ``.pth`` checkpoint and runs
    inference with PyTorch (GPU or CPU).  *No ONNX Runtime dependency.*

    Args:
        model_path: Path to the ``.pth`` checkpoint (default:
                    ``weights/culane_r18.pth``).
        config_path: Path to the CLRNet config file (default:
                     ``external/CLRNet/configs/clrnet/clr_resnet18_culane.py``).
        input_width: Width of the network input.
        input_height: Height of the network input.
        conf_threshold: Minimum softmax confidence for a lane proposal.
        lane_confidence_threshold: Minimum confidence for a lane to be displayed.
        max_lanes: Maximum number of lanes to output.
        temporal_smoothing: Whether to smooth results across frames.
        smoothing_window: Number of recent frames for temporal smoothing.
    """

    def __init__(
        self,
        model_path: str = "weights/culane_r18.pth",
        config_path: str = "",
        input_width: int = 800,
        input_height: int = 320,
        conf_threshold: float = 0.4,
        lane_confidence_threshold: float = 0.5,
        max_lanes: int = 4,
        temporal_smoothing: bool = True,
        smoothing_window: int = 5,
        cut_height: int = 270,
    ) -> None:
        self._model_path = model_path
        self._config_path = config_path or str(
            _CLRNET_ROOT / "configs" / "clrnet" / "clr_resnet18_culane.py"
        )
        self._input_w = input_width
        self._input_h = input_height
        self._conf_threshold = conf_threshold
        self._lane_conf_threshold = lane_confidence_threshold
        self._max_lanes = max_lanes
        self._temporal_smoothing = temporal_smoothing
        self._smoothing_window = smoothing_window

        self._cut_height = cut_height
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._model: torch.nn.Module | None = None
        self._cfg: Any = None
        self._recent_results: deque[LaneDetectionResult] = deque(
            maxlen=smoothing_window
        )
        self._logged_preprocess: bool = False

        # CLRNet test-time preprocessing: /255 only (no mean subtraction)
        # See GenerateLaneLine in clrnet/datasets/process/generate_lane_line.py

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Build the CLRNet network and load the pretrained checkpoint.

        Raises:
            FileNotFoundError: If the config or checkpoint file is missing.
            ValueError: If the checkpoint format is unrecognised.
        """
        cfg_path = Path(self._config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"CLRNet config not found: {cfg_path.resolve()}"
            )

        ckpt_path = Path(self._model_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"CLRNet checkpoint not found: {ckpt_path.resolve()}"
            )

        # --- Config ---
        logger.info("Loading CLRNet config: %s", cfg_path.name)
        cfg = Config.fromfile(str(cfg_path))
        cfg.img_w = self._input_w
        cfg.img_h = self._input_h
        self._cfg = cfg

        # --- Build network ---
        logger.info("Building CLRNet network ...")
        model = build_net(cfg)

        # --- Load checkpoint ---
        logger.info("Loading checkpoint: %s", ckpt_path.name)
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")

        state_dict = self._extract_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError(
                f"Unrecognised checkpoint format in {ckpt_path}. "
                "Expected a dict with a 'net', 'state_dict', or 'model' key, "
                "or a raw state dict."
            )

        state_dict = self._strip_dataparallel_prefix(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info("Missing keys in checkpoint: %d", len(missing))
        if unexpected:
            logger.info("Unexpected keys in checkpoint: %d", len(unexpected))

        model = model.to(self._device)
        model.eval()
        self._model = model

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "CLRNet model ready  |  device=%s  |  params=%d",
            self._device,
            total_params,
        )
        logger.info(
            "CLRNet config:\n"
            "  cut_height=%d\n"
            "  input_width=%d\n"
            "  input_height=%d",
            self._cut_height,
            self._input_w,
            self._input_h,
        )

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> dict[str, torch.Tensor] | None:
        """Try to extract a state dict from a checkpoint in various formats."""
        if not isinstance(checkpoint, dict):
            if hasattr(checkpoint, "state_dict"):
                return checkpoint.state_dict()
            return None

        for key in ("net", "state_dict", "model"):
            if key in checkpoint and isinstance(
                checkpoint[key], dict
            ):
                return checkpoint[key]

        # Maybe it *is* a raw state dict
        if all(
            isinstance(k, str) and isinstance(v, torch.Tensor)
            for k, v in checkpoint.items()
        ):
            return checkpoint

        return None

    @staticmethod
    def _strip_dataparallel_prefix(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remove ``module.`` prefix added by DataParallel."""
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            cleaned[k[7:] if k.startswith("module.") else k] = v
        return cleaned

    def _ensure_model(self) -> None:
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    #  Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert a BGR frame to a CHW tensor for CLRNet inference.

        Matches the official CLRNet CULane inference pipeline:
        crop top region → resize (INTER_CUBIC) → float32 / 255 → CHW → batch dim.

        See ``BaseDataset.__getitem__`` (base_dataset.py:40) and the
        ONNX demo (demo_onnx_new.py:278-284).

        No mean/std normalisation is applied at test time.
        """
        h, w = frame.shape[:2]

        cut_h = getattr(self, "_cut_height", 270)

        # 1. Crop sky region (official CULane pipeline)
        cropped = frame[cut_h:, :, :]

        # 2. Resize to model input size with INTER_CUBIC (matches ONNX demo)
        resized = cv2.resize(
            cropped,
            (self._input_w, self._input_h),
            interpolation=cv2.INTER_CUBIC,
        )

        # 3. Normalise to [0, 1]
        blob = resized.astype(np.float32, copy=False) / 255.0

        # Debug logging (first frame only)
        logged = getattr(self, "_logged_preprocess", False)
        if not logged:
            logger.info(
                "CLRNet preprocessing:\n"
                "  original=(%d,%d,%d)\n"
                "  cropped=(%d,%d,%d)\n"
                "  resized=(%d,%d,%d)\n"
                "  cut_height=%d",
                h,
                w,
                frame.shape[2],
                cropped.shape[0],
                cropped.shape[1],
                cropped.shape[2],
                resized.shape[0],
                resized.shape[1],
                resized.shape[2],
                cut_h,
            )
            self._logged_preprocess = True  # type: ignore[assignment]

        chw = np.ascontiguousarray(blob.transpose(2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self._device)
        return tensor

    # ------------------------------------------------------------------
    #  Post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        predictions: torch.Tensor,
        orig_w: int,
        orig_h: int,
    ) -> LaneDetectionResult:
        """Decode CLRNet model output into a ``LaneDetectionResult``.

        Mirrors official ``CLRHead.get_lanes`` then ``predictions_to_pred``.

        Args:
            predictions: Tensor of shape ``(1, num_priors, 78)`` from the
                         model forward pass.
            orig_w: Original frame width.
            orig_h: Original frame height.
        """
        if predictions is None or predictions.numel() == 0:
            return LaneDetectionResult.empty()

        preds = predictions[0]  # (num_priors, 2+2+2+72)

        # --- Softmax confidence ---
        scores = torch.softmax(preds[:, :2], dim=1)[:, 1]

        # --- Confidence threshold ---
        keep_mask = scores >= self._conf_threshold
        preds = preds[keep_mask]
        scores = scores[keep_mask]

        if preds.shape[0] == 0:
            return LaneDetectionResult.empty()

        # --- NMS (matches official get_lanes step) ---
        keep_idx = self._lane_nms(preds, scores)
        preds = preds[keep_idx]
        scores = scores[keep_idx]

        if preds.shape[0] == 0:
            return LaneDetectionResult.empty()

        # --- Pre-scale length to absolute strip units (official line 477) ---
        # predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
        n_strips = 71
        preds[:, 5] = torch.round(preds[:, 5] * n_strips)

        # --- Decode to pixel coords (official predictions_to_pred) ---
        lane_list = self._decode_lanes(preds, orig_w, orig_h)

        return self._build_result(lane_list, orig_w)

    def _lane_nms(
        self,
        predictions: torch.Tensor,
        scores: torch.Tensor,
    ) -> list[int]:
        """Lane NMS matching official CLRNet implementation.

        Official code (clr_head.py:458-469):
        - Removes theta (col 4) from the tensor
        - Scales length by n_strips and x_offsets by (img_w-1) into pixel space
        - Compares average pixel distance against nms_thres
        """
        n_strips = 71
        n_offsets = 72
        nms_thres = self._cfg.test_parameters.nms_thres

        # Build NMS tensor: skip theta column (index 4)
        # Original: [cls_neg, cls_pos, start_y, start_x, theta, length, x_offsets...]
        # NMS:      [cls_neg, cls_pos, start_y, start_x, length, x_offsets...]
        nms_input = predictions.detach().clone()
        nms_input = torch.cat(
            [nms_input[..., :4], nms_input[..., 5:]], dim=-1
        )
        # Scale length to absolute strips
        nms_input[..., 4] = nms_input[..., 4] * n_strips
        # Scale x offsets to pixel space
        nms_input[..., 5:] = nms_input[..., 5:] * (self._input_w - 1)

        _, idx = scores.sort(descending=True)
        nms_input = nms_input[idx]

        keep: list[int] = []
        for i in range(len(nms_input)):
            if len(keep) >= self._max_lanes:
                break

            suppressed = False
            a = nms_input[i]
            start_a = int(round(a[2].item() * n_strips))
            len_a = int(round(a[4].item()))
            end_a = min(start_a + len_a - 1, n_offsets - 1)

            for j in keep:
                b = nms_input[j]
                start_b = int(round(b[2].item() * n_strips))
                len_b = int(round(b[4].item()))
                end_b = min(start_b + len_b - 1, n_offsets - 1)

                start = max(start_a, start_b)
                end = min(end_a, end_b)
                if end < start:
                    continue

                dist = 0.0
                for k in range(start, end + 1):
                    dist += abs(
                        a[5 + k].item() - b[5 + k].item()
                    )
                avg = dist / (end - start + 1)

                if avg < nms_thres:
                    suppressed = True
                    break

            if not suppressed:
                keep.append(idx[i].item())

        return keep

    def _decode_lanes(
        self,
        predictions: torch.Tensor,
        orig_w: int,
        orig_h: int,
    ) -> list[list[tuple[int, int]]]:
        """Convert predictions to pixel-coordinate lane polylines.

        Faithfully replicates ``CLRHead.predictions_to_pred``
        (clr_head.py:279-319).

        Key details that MUST match the official logic:
        A) ``lane_xs = lane[6:]`` — normalised x offsets
        B) ``start = round(lane[2] * n_strips)`` — start strip index
        C) ``length = round(lane[5])`` — length in strips (pre-scaled)
        D) ``end = min(start + length - 1, len(prior_ys) - 1)``
        E) ``-2`` sentinel for invalid points
        F) Cumulative-product mask extends lane below start
        G) Double-precision ``prior_ys``
        H) Y conversion: ``y = (y * (ori_h - cut_h) + cut_h) / ori_h``
        I) Points in top→bottom order after flip
        """
        n_strips = 71
        n_offsets = 72
        device = predictions.device

        # Row-anchor y values: 1.0 (bottom) → 0.0 (top), double precision
        prior_ys = torch.linspace(
            1.0, 0.0, steps=n_offsets, device=device, dtype=torch.double
        )

        # CULane dataset constants for y-coordinate mapping
        ori_img_h = 590.0

        all_lanes: list[list[tuple[int, int]]] = []

        for lane in predictions:
            lane_xs = lane[6:]  # normalised x offsets

            start = min(
                max(0, int(round(lane[2].item() * n_strips))), n_strips
            )
            length = int(round(lane[5].item()))  # pre-scaled to strips
            end = start + length - 1
            end = min(end, n_offsets - 1)

            # Extend below start using cumulative-product mask
            mask = ~(
                (
                    (lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)
                )
                .cpu()
                .numpy()[::-1]
                .cumprod()[::-1]
                .astype(bool)
            )
            lane_xs[end + 1:] = -2.0
            lane_xs[:start][torch.from_numpy(mask).to(device)] = -2.0

            # Filter valid points
            lane_ys = prior_ys[lane_xs >= 0.0]
            lane_xs = lane_xs[lane_xs >= 0.0]

            if len(lane_xs) <= 1:
                continue

            # Flip to top → bottom order
            lane_xs = lane_xs.flip(0)
            lane_ys = lane_ys.flip(0)

            # Convert y from crop-space to full-image normalised
            # (matches official predictions_to_pred y conversion)
            cut_h = getattr(self, "_cut_height", 270)
            lane_ys = (
                lane_ys * (ori_img_h - cut_h) + cut_h
            ) / ori_img_h

            # Convert to pixel coordinates
            points: list[tuple[int, int]] = []
            for xi, yi in zip(lane_xs, lane_ys):
                # xi is normalised relative to model input width
                # Match official to_array: x_pixel = xi * ori_img_w
                # For arbitrary video, use orig_w directly
                x_pixel = int(round(xi.item() * orig_w))
                # yi is normalised relative to ori_img_h (full CULane image)
                # Scale to output frame height
                y_pixel = int(round(yi.item() * orig_h))
                y_pixel = min(y_pixel, orig_h - 1)

                if 0 <= x_pixel < orig_w and 0 <= y_pixel < orig_h:
                    points.append((x_pixel, y_pixel))

            # Diagnostic validation
            if len(points) >= 2:
                bottom = max(points, key=lambda p: p[1])
                top = min(points, key=lambda p: p[1])
                dx = abs(bottom[0] - top[0])
                if dx > orig_w * 0.7:
                    logger.warning(
                        "Potentially invalid lane geometry: "
                        "bottom=%s top=%s delta_x=%d",
                        bottom,
                        top,
                        dx,
                    )

            if len(points) >= 2:
                all_lanes.append(points)

        return all_lanes

    def _build_result(
        self,
        all_lane_points: list[list[tuple[int, int]]],
        img_w: int,
    ) -> LaneDetectionResult:
        """Sort lanes left-to-right and assign ego left / right / center."""
        if not all_lane_points:
            return LaneDetectionResult.empty()

        def _bottom_x(lane: list[tuple[int, int]]) -> float:
            return float(max(lane, key=lambda p: p[1])[0])

        sorted_lanes = sorted(all_lane_points, key=_bottom_x)
        sorted_lanes = sorted_lanes[: self._max_lanes]

        lane_count = len(sorted_lanes)
        center_x = img_w / 2.0

        left_lane: list[tuple[int, int]] | None = None
        right_lane: list[tuple[int, int]] | None = None

        left_of_center = [
            ln for ln in sorted_lanes if _bottom_x(ln) < center_x
        ]
        right_of_center = [
            ln for ln in sorted_lanes if _bottom_x(ln) >= center_x
        ]

        if left_of_center:
            left_lane = left_of_center[-1]
        if right_of_center:
            right_lane = right_of_center[0]

        center_lane: list[tuple[int, int]] | None = None
        if left_lane and right_lane:
            min_len = min(len(left_lane), len(right_lane))
            if min_len >= 2:
                left_sorted = sorted(left_lane, key=lambda p: p[1])
                right_sorted = sorted(right_lane, key=lambda p: p[1])
                center_lane = [
                    (
                        (left_sorted[i][0] + right_sorted[i][0]) // 2,
                        (left_sorted[i][1] + right_sorted[i][1]) // 2,
                    )
                    for i in range(min_len)
                ]

        ego_lane_center_x: float | None = None
        if left_lane and right_lane:
            max_y = max(p[1] for p in left_lane + right_lane)
            lx = [p[0] for p in left_lane if p[1] == max_y]
            rx = [p[0] for p in right_lane if p[1] == max_y]
            if lx and rx:
                ego_lane_center_x = (float(lx[0]) + float(rx[0])) / 2.0

        return LaneDetectionResult(
            left_lane_points=tuple(left_lane) if left_lane else (),
            right_lane_points=tuple(right_lane) if right_lane else (),
            center_lane_points=tuple(center_lane) if center_lane else (),
            lane_count=lane_count,
            ego_lane_center_x=ego_lane_center_x,
            confidence=(
                0.9
                if left_lane and right_lane
                else 0.3
                if left_lane or right_lane
                else 0.0
            ),
        )

    # ------------------------------------------------------------------
    #  Temporal smoothing
    # ------------------------------------------------------------------

    def _apply_smoothing(
        self, result: LaneDetectionResult
    ) -> LaneDetectionResult:
        """Moving-average smoothing across recent frames."""
        if not self._temporal_smoothing:
            return result

        self._recent_results.append(result)

        valid = [r for r in self._recent_results if r.lane_count > 0]
        if len(valid) < 2:
            return result

        def _avg_points(
            *point_groups: tuple[tuple[int, int], ...],
        ) -> tuple[tuple[int, int], ...]:
            non_empty = [g for g in point_groups if g]
            if not non_empty:
                return ()
            if len(non_empty) == 1:
                return non_empty[0]
            ref = non_empty[-1]
            out: list[tuple[int, int]] = []
            for i in range(len(ref)):
                xs: list[float] = []
                ys: list[float] = []
                for g in non_empty:
                    if i < len(g):
                        xs.append(float(g[i][0]))
                        ys.append(float(g[i][1]))
                if xs:
                    out.append(
                        (
                            int(round(sum(xs) / len(xs))),
                            int(round(sum(ys) / len(ys))),
                        )
                    )
            return tuple(out)

        left = _avg_points(*(r.left_lane_points for r in valid))
        right = _avg_points(*(r.right_lane_points for r in valid))
        center = _avg_points(*(r.center_lane_points for r in valid))

        return LaneDetectionResult(
            left_lane_points=left,
            right_lane_points=right,
            center_lane_points=center,
            lane_count=result.lane_count,
            ego_lane_center_x=result.ego_lane_center_x,
            confidence=result.confidence,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        """Run CLRNet lane detection on a BGR frame.

        Args:
            frame: BGR image array of shape ``(H, W, 3)``.

        Returns:
            A ``LaneDetectionResult``, or ``LaneDetectionResult.empty()``
            on any failure.
        """
        try:
            self._ensure_model()
        except FileNotFoundError:
            logger.exception(
                "CLRNet model/config file missing"
            )
            return LaneDetectionResult.empty()
        except Exception:
            logger.exception("CLRNet model loading failed")
            return LaneDetectionResult.empty()

        try:
            input_tensor = self._preprocess(frame)
        except Exception:
            logger.exception("CLRNet preprocessing failed")
            return LaneDetectionResult.empty()

        try:
            with torch.no_grad():
                output = self._model({"img": input_tensor})
        except Exception:
            logger.exception("CLRNet inference failed")
            return LaneDetectionResult.empty()

        try:
            result = self._postprocess(
                output,
                orig_w=frame.shape[1],
                orig_h=frame.shape[0],
            )
        except Exception:
            logger.exception("CLRNet postprocessing failed")
            return LaneDetectionResult.empty()

        if self._temporal_smoothing:
            try:
                result = self._apply_smoothing(result)
            except Exception:
                logger.exception("CLRNet temporal smoothing failed")

        return result
