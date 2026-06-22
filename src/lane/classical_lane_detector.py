from __future__ import annotations

import math

import cv2
import numpy as np

from src.lane.base import BaseLaneDetector
from src.schemas import Frame
from src.schemas.lane_info import LaneInfo, LaneLine


class ClassicalLaneDetector(BaseLaneDetector):
    """Lane detector using classical CV (Canny + Hough + linear fit).

    Processing pipeline::

        grayscale → Gaussian blur → Canny edge detection
        → trapezoidal ROI mask → HoughLinesP
        → split lines by slope sign → linear fit each side
        → LaneInfo

    Args:
        canny_low: Lower threshold for Canny.
        canny_high: Upper threshold for Canny.
        hough_min_votes: Minimum intersections (votes) for a line.
        hough_min_len: Minimum line length (pixels).
        hough_max_gap: Maximum gap between segments.
        roi_top_ratio: Y coordinate of ROI top as fraction of image height.
        roi_bottom_ratio: Y coordinate of ROI bottom as fraction of height.
        roi_top_width_ratio: How much to narrow the ROI top on each side
            (fraction of image width).
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_min_votes: int = 30,
        hough_min_len: int = 40,
        hough_max_gap: int = 100,
        roi_top_ratio: float = 0.6,
        roi_bottom_ratio: float = 1.0,
        roi_top_width_ratio: float = 0.15,
    ) -> None:
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._hough_votes = hough_min_votes
        self._hough_len = hough_min_len
        self._hough_gap = hough_max_gap
        self._roi_top = roi_top_ratio
        self._roi_bottom = roi_bottom_ratio
        self._roi_top_w = roi_top_width_ratio

    def detect(self, frame: Frame) -> LaneInfo:
        h, w = frame.image.shape[:2]
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self._canny_low, self._canny_high)

        # Trapezoidal ROI mask
        mask = self._make_roi_mask(h, w)
        masked = cv2.bitwise_and(edges, mask)

        # Hough lines
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=math.pi / 180.0,
            threshold=self._hough_votes,
            minLineLength=self._hough_len,
            maxLineGap=self._hough_gap,
        )

        return self._lines_to_lane_info(lines, h, w)

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _make_roi_mask(self, h: int, w: int) -> np.ndarray:
        roi_y1 = int(h * self._roi_top)
        roi_y2 = int(h * self._roi_bottom) - 1
        margin = int(w * self._roi_top_w)

        pts = np.array([
            (margin, roi_y1),
            (w - 1 - margin, roi_y1),
            (w - 1, roi_y2),
            (0, roi_y2),
        ], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _lines_to_lane_info(
        self, lines: np.ndarray | None, h: int, w: int
    ) -> LaneInfo:
        roi_y1 = int(h * self._roi_top)
        roi_y2 = int(h * self._roi_bottom) - 1

        left_segments: list[tuple[float, float]] = []  # (y1, x1), (y2, x2)
        right_segments: list[tuple[float, float]] = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y2 == y1:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                if abs(slope) < 0.2:
                    continue  # near-horizontal reject
                if slope < 0:
                    left_segments.append((y1, x1))
                    left_segments.append((y2, x2))
                else:
                    right_segments.append((y1, x1))
                    right_segments.append((y2, x2))

        left_line = self._fit_line(left_segments, roi_y1, roi_y2)
        right_line = self._fit_line(right_segments, roi_y1, roi_y2)

        left_lane = (
            LaneLine(coeffs=(0.0, left_line[0], left_line[1]), confidence=left_line[2])
            if left_line is not None else None
        )
        right_lane = (
            LaneLine(coeffs=(0.0, right_line[0], right_line[1]), confidence=right_line[2])
            if right_line is not None else None
        )

        # Lane center (midpoint at bottom of ROI)
        center_line: LaneLine | None = None
        if left_lane is not None and right_lane is not None:
            lx1 = left_line[0] * roi_y1 + left_line[1]
            lx2 = left_line[0] * roi_y2 + left_line[1]
            rx1 = right_line[0] * roi_y1 + right_line[1]
            rx2 = right_line[0] * roi_y2 + right_line[1]
            cx1 = (lx1 + rx1) / 2.0
            cx2 = (lx2 + rx2) / 2.0
            m = (cx2 - cx1) / (roi_y2 - roi_y1) if roi_y2 != roi_y1 else 0.0
            b = cx1 - m * roi_y1
            center_line = LaneLine(coeffs=(0.0, m, b), confidence=min(
                left_line[2], right_line[2]
            ))

        confidence = 0.0
        confs = []
        if left_line is not None:
            confs.append(left_line[2])
        if right_line is not None:
            confs.append(right_line[2])
        if confs:
            confidence = min(confs)

        return LaneInfo(
            left_lane=left_lane,
            right_lane=right_lane,
            lane_center=center_line,
            confidence=confidence,
        )

    @staticmethod
    def _fit_line(
        points: list[tuple[float, float]],
        roi_y1: int,
        roi_y2: int,
    ) -> tuple[float, float, float] | None:
        """Fit x = m * y + b via least squares. Returns (m, b, confidence)."""
        if len(points) < 4:  # at least 2 segments = 4 endpoints
            return None
        ys = np.array([p[0] for p in points], dtype=np.float64)
        xs = np.array([p[1] for p in points], dtype=np.float64)
        A = np.vstack([ys, np.ones_like(ys)]).T
        try:
            m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        # Clip endpoints to ROI
        x1 = m * roi_y1 + b
        x2 = m * roi_y2 + b

        n_inliers = len(points) // 2  # each segment gave 2 points
        confidence = min(1.0, n_inliers / 20.0)
        return (m, b, confidence)
