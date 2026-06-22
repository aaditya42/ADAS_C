from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class DepthEvaluator:
    """Evaluate depth estimation quality.

    Two modes:
      - **temporal_consistency** (no GT): uses optical flow to measure
        per-pixel depth stability across consecutive frames.
      - **gt** (requires sparse or dense GT): computes AbsRel, RMSE,
        and δ<1.25|1.25²|1.25³.
    """

    def __init__(self, mode: str = "temporal_consistency") -> None:
        if mode not in ("temporal_consistency", "gt"):
            raise ValueError(f"Unknown depth evaluation mode: {mode}")
        self.mode = mode

    def evaluate(
        self,
        frames: list[Any],
        depth_maps: list[np.ndarray],
        gt_maps: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run depth evaluation.

        Args:
            frames: List of ``Frame`` objects (each has ``.image``).
            depth_maps: Corresponding list of predicted depth maps
                (same length as ``frames``).
            gt_maps: Ground-truth depth maps (same length). Required
                when ``mode="gt"``.

        Returns:
            Dict of metric names to values.
        """
        if self.mode == "temporal_consistency":
            return self._evaluate_temporal(frames, depth_maps)
        elif self.mode == "gt":
            if gt_maps is None:
                raise ValueError("gt_maps must be provided when mode='gt'")
            return self._evaluate_gt(depth_maps, gt_maps)
        else:
            return {}

    def _evaluate_temporal(
        self,
        frames: list[Any],
        depth_maps: list[np.ndarray],
    ) -> dict[str, Any]:
        errors: list[float] = []
        n_pairs = len(frames) - 1

        for i in range(n_pairs):
            prev_img = self._to_gray(frames[i].image)
            next_img = self._to_gray(frames[i + 1].image)
            prev_depth = depth_maps[i]
            next_depth = depth_maps[i + 1]

            if prev_depth.shape != next_depth.shape:
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_img, next_img, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )

            h, w = prev_depth.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            map_x = (x_coords + flow[..., 0]).astype(np.float32)
            map_y = (y_coords + flow[..., 1]).astype(np.float32)
            warped_prev = cv2.remap(
                prev_depth, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            valid = (
                (map_x >= 0) & (map_x < w - 1) &
                (map_y >= 0) & (map_y < h - 1) &
                (warped_prev > 0) & (next_depth > 0)
            )

            if not valid.any():
                continue

            diff = np.abs(warped_prev[valid] - next_depth[valid])

            p_min, p_max = np.percentile(diff, [1, 99])
            clipped = np.clip(diff, p_min, p_max)

            errors.append(float(np.mean(clipped)))

        if not errors:
            return {"temporal_consistency_mae": 0.0, "frames_evaluated": 0}

        return {
            "temporal_consistency_mae": round(float(np.mean(errors)), 6),
            "temporal_consistency_std": round(float(np.std(errors)), 6),
            "frames_evaluated": n_pairs,
        }

    def _evaluate_gt(
        self,
        depth_maps: list[np.ndarray],
        gt_maps: list[np.ndarray],
    ) -> dict[str, Any]:
        abs_rel_list: list[float] = []
        rmse_list: list[float] = []
        delta_125_list: list[float] = []
        delta_1252_list: list[float] = []
        delta_1253_list: list[float] = []

        for pred, gt in zip(depth_maps, gt_maps):
            mask = gt > 0
            if not mask.any():
                continue

            pred_valid = pred[mask].astype(np.float64)
            gt_valid = gt[mask].astype(np.float64)

            abs_rel_list.append(float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid)))
            rmse_list.append(float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2))))

            ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
            delta_125_list.append(float(np.mean(ratio < 1.25)))
            delta_1252_list.append(float(np.mean(ratio < 1.25 ** 2)))
            delta_1253_list.append(float(np.mean(ratio < 1.25 ** 3)))

        if not abs_rel_list:
            return {
                "AbsRel": 0.0, "RMSE": 0.0,
                "delta_1.25": 0.0, "delta_1.25^2": 0.0, "delta_1.25^3": 0.0,
                "frames_evaluated": 0,
            }

        return {
            "AbsRel": round(float(np.mean(abs_rel_list)), 6),
            "RMSE": round(float(np.mean(rmse_list)), 6),
            "delta_1.25": round(float(np.mean(delta_125_list)), 6),
            "delta_1.25^2": round(float(np.mean(delta_1252_list)), 6),
            "delta_1.25^3": round(float(np.mean(delta_1253_list)), 6),
            "frames_evaluated": len(depth_maps),
        }

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.uint8)
