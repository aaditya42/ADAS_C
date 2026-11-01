from __future__ import annotations

from typing import Tuple

import numpy as np

_COCO_MEAN = (0.485, 0.456, 0.406)
_COCO_STD = (0.229, 0.224, 0.225)


def expand_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_h: int,
    img_w: int,
    margin: float = 1.2,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box by a margin factor and clamp to image bounds.

    The box is expanded outward from its center.  A margin of 1.0 keeps
    the box unchanged; 1.2 adds 10 % on each side.

    Args:
        x1, y1, x2, y2: Original bounding box in pixel coordinates.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        margin: Expansion factor (>1 = larger box).

    Returns:
        Expanded box ``(ex1, ey1, ex2, ey2)`` clamped to image dimensions.
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * margin
    bh = (y2 - y1) * margin

    ex1 = int(round(cx - bw / 2.0))
    ey1 = int(round(cy - bh / 2.0))
    ex2 = int(round(cx + bw / 2.0))
    ey2 = int(round(cy + bh / 2.0))

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(img_w, ex2)
    ey2 = min(img_h, ey2)

    if ex2 <= ex1 or ey2 <= ey1:
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    return ex1, ey1, ex2, ey2


def crop_person(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    margin: float = 1.2,
) -> Tuple[np.ndarray, int, int]:
    """Crop a person from the image using an expanded bounding box.

    Args:
        image: BGR image of shape (H, W, 3).
        bbox: ``(x1, y1, x2, y2)`` in image pixel coordinates.
        margin: Expansion factor for the crop.

    Returns:
        Tuple of ``(crop, offset_x, offset_y)`` where the crop is the image
        region and offset_x/offset_y are the top-left corner of the crop in
        the original image (for mapping keypoints back).
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = expand_bbox(
        bbox[0], bbox[1], bbox[2], bbox[3], img_h, img_w, margin
    )
    crop = image[y1:y2, x1:x2].copy()
    return crop, x1, y1


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int]:
    """Resize and pad an image to a fixed target size while preserving aspect ratio.

    The output image is placed on a letterbox canvas of ``target_size``
    with ``color`` padding on the borders.

    Args:
        image: Input image (H, W, C).
        target_size: ``(height, width)`` of the output canvas.
        color: Padding colour (BGR).

    Returns:
        Tuple of ``(padded, scale, pad_left, pad_top)`` where:
        - ``padded`` is the target-sized uint8 image,
        - ``scale`` is the resize factor (original → resized),
        - ``pad_left`` / ``pad_top`` is the padding added on each side.
    """
    import cv2  # noqa: PLC0415

    h, w = image.shape[:2]
    tgt_h, tgt_w = target_size

    scale = min(tgt_w / w, tgt_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (tgt_w - new_w) // 2
    pad_top = (tgt_h - new_h) // 2

    canvas = np.full((tgt_h, tgt_w, 3), color, dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    return canvas, scale, pad_left, pad_top


def normalize_for_onnx(
    image: np.ndarray,
) -> np.ndarray:
    """Convert a uint8 BGR image to the ONNX input blob format.

    Performs:
        - BGR → RGB
        - uint8 → float32
        - 1/255 scaling
        - HWC → CHW transposition
        - Batch dimension insertion

    Args:
        image: Input image (H, W, 3) uint8 BGR.

    Returns:
        Blob of shape ``(1, 3, H, W)`` float32 suitable for ONNX inference.
    """
    blob = image.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob[:, :, ::-1])  # BGR → RGB
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    return blob


def map_keypoints_to_original(
    keypoints: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    crop_x1: int,
    crop_y1: int,
) -> np.ndarray:
    """Map model-output keypoints from letterbox-crop space to original image space.

    Each keypoint is transformed as::

        x_orig = (x_padded - pad_left) / scale + crop_x1
        y_orig = (y_padded - pad_top) / scale + crop_y1

    Args:
        keypoints: Array of shape ``(N, 3)`` where columns are ``(x, y, conf)``
            in the letterbox-padded crop coordinate space.
        scale: Resize factor from the letterbox transform.
        pad_left: Horizontal padding pixels added by letterbox.
        pad_top: Vertical padding pixels added by letterbox.
        crop_x1: Top-left x of the crop in the original image.
        crop_y1: Top-left y of the crop in the original image.

    Returns:
        Array of shape ``(N, 3)`` with coordinates mapped to the original
        full-frame image space.
    """
    mapped = keypoints.copy()
    mapped[:, 0] = (mapped[:, 0] - pad_left) / scale + crop_x1
    mapped[:, 1] = (mapped[:, 1] - pad_top) / scale + crop_y1
    return mapped
