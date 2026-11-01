from __future__ import annotations

from typing import Tuple

import numpy as np

from src.pose.human.preprocess import (
    crop_person as _crop_region,
    expand_bbox,
    letterbox as _letterbox,
    map_keypoints_to_original as _map_keypoints,
    normalize_for_onnx,
)

crop_animal_region = _crop_region
"""Crop an animal region from the image.  Alias for ``crop_person``."""

letterbox = _letterbox
"""Resize and pad while preserving aspect ratio."""

map_keypoints_to_original = _map_keypoints
"""Map model-output keypoints from letterbox-crop space to original image space."""

__all__ = [
    "crop_animal_region",
    "expand_bbox",
    "letterbox",
    "map_keypoints_to_original",
    "normalize_for_onnx",
    "heatmap_to_keypoints",
]


def heatmap_to_keypoints(
    heatmaps: np.ndarray,
    confidence_threshold: float = 0.1,
) -> np.ndarray:
    """Decode a batch-1 heatmap tensor into image-space keypoints.

    Each channel of the heatmap is argmax-decoded to a single ``(x, y)``
    coordinate.  The confidence value is taken as the peak heatmap value.

    Args:
        heatmaps: Array of shape ``(1, K, H, W)`` or ``(K, H, W)`` where
            ``K`` is the number of keypoints (17 for AP-10K).
        confidence_threshold: Minimum heatmap peak to accept a keypoint
            as "visible".  Keypoints below this threshold still receive
            coordinates but ``confidence`` will be the raw peak value.

    Returns:
        Array of shape ``(K, 3)`` with columns ``(x, y, confidence)`` in
        the **heatmap coordinate space**.  To convert to image space,
        multiply by the heatmap-to-image scale factor and add crop offset.
    """
    heatmaps = np.squeeze(heatmaps)
    if heatmaps.ndim == 3:
        pass  # (K, H, W) — expected
    elif heatmaps.ndim == 4 and heatmaps.shape[0] == 1:
        heatmaps = heatmaps[0]  # (1, K, H, W) → (K, H, W)
    else:
        raise ValueError(
            f"Expected heatmap shape (1, K, H, W) or (K, H, W), got {heatmaps.shape}"
        )

    K, H, W = heatmaps.shape
    keypoints = np.zeros((K, 3), dtype=np.float32)

    for k in range(K):
        heatmap = heatmaps[k]
        idx = int(np.argmax(heatmap))
        y = idx // W
        x = idx % W
        peak = float(heatmap[y, x])
        keypoints[k] = [float(x), float(y), peak]

    return keypoints


def scale_keypoints_to_crop(
    keypoints: np.ndarray,
    heatmap_size: Tuple[int, int],
    crop_size: Tuple[int, int],
) -> np.ndarray:
    """Scale keypoints from heatmap coordinates to crop-image coordinates.

    Args:
        keypoints: Array of shape ``(K, 3)`` in heatmap pixel space.
        heatmap_size: ``(H_heat, W_heat)`` dimensions of the heatmap.
        crop_size: ``(H_crop, W_crop)`` dimensions of the letterbox crop
            that was fed to the model.

    Returns:
        Array of shape ``(K, 3)`` scaled to crop-image coordinates.
    """
    h_heat, w_heat = heatmap_size
    h_crop, w_crop = crop_size
    scale_x = w_crop / w_heat
    scale_y = h_crop / h_heat

    mapped = keypoints.copy()
    mapped[:, 0] *= scale_x
    mapped[:, 1] *= scale_y
    return mapped
