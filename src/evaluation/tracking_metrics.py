from __future__ import annotations

from typing import Any

import numpy as np

from src.schemas import TrackedObject


def _monkey_patch_motmetrics() -> None:
    """Add ``np.asfarray`` back for motmetrics compatibility with NumPy 2.x."""
    import numpy as np  # noqa: PLC0415
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # noqa: NPY001


class TrackingEvaluator:
    """Evaluate multi-object tracking against MOT-format ground truth.

    Uses ``motmetrics`` internally. If the library is unavailable all
    methods raise ``ImportError`` with an installation hint.
    """

    def __init__(self) -> None:
        _monkey_patch_motmetrics()

    @staticmethod
    def _check_motmetrics() -> None:
        try:
            import motmetrics  # noqa: PLC0415, F401
        except ImportError:
            raise ImportError(
                "motmetrics is required for tracking evaluation. "
                "Install it with: pip install motmetrics"
            )

    def evaluate(
        self,
        predictions: list[list[TrackedObject]],
        ground_truth: str | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute MOT metrics.

        Args:
            predictions: Per-frame list of tracked objects.
            ground_truth: Path to a MOT-challenge-format text file, or a
                list of per-frame ground-truth dicts with keys
                ``frame``, ``id``, ``x``, ``y``, ``w``, ``h``,
                ``conf`` (optional).

        Returns:
            Dict with keys ``MOTA``, ``MOTP``, ``IDF1``, ``IDSW``,
            ``MT``, ``ML``, ``FP``, ``FN``, ``num_switches``,
            ``fragmentations``.
        """
        import motmetrics as mm  # noqa: PLC0415

        gt_df = self._load_gt(ground_truth)
        pred_df = self._predictions_to_df(predictions)

        acc = mm.MOTAccumulator(auto_id=True)

        gt_frame_ids: set[int] = set()
        pred_frame_ids: set[int] = set()
        if not gt_df.empty and "frame" in gt_df.columns:
            gt_frame_ids = set(gt_df["frame"].unique())
        if not pred_df.empty and "frame" in pred_df.columns:
            pred_frame_ids = set(pred_df["frame"].unique())
        all_frame_ids = sorted(gt_frame_ids | pred_frame_ids)

        for frame_id in all_frame_ids:
            gt_frame = gt_df[gt_df["frame"] == frame_id] if "frame" in gt_df.columns else gt_df.iloc[:0]
            pred_frame = pred_df[pred_df["frame"] == frame_id] if "frame" in pred_df.columns else pred_df.iloc[:0]

            gt_ids: list[int] = []
            pred_ids: list[int] = []
            if not gt_frame.empty and "id" in gt_frame.columns:
                gt_ids = gt_frame["id"].values.tolist()
            if not pred_frame.empty and "id" in pred_frame.columns:
                pred_ids = pred_frame["id"].values.tolist()

            gt_bboxes = gt_frame[["x", "y", "w", "h"]].values if (
                not gt_frame.empty and all(c in gt_frame.columns for c in ("x", "y", "w", "h"))
            ) else np.empty((0, 4))
            pred_bboxes = pred_frame[["x", "y", "w", "h"]].values if (
                not pred_frame.empty and all(c in pred_frame.columns for c in ("x", "y", "w", "h"))
            ) else np.empty((0, 4))

            dist_matrix = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
            acc.update(gt_ids, pred_ids, dist_matrix)

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                "num_frames", "num_objects", "num_predictions",
                "mota", "motp", "idf1",
                "mostly_tracked", "mostly_lost",
                "num_switches", "num_fragmentations",
                "num_false_positives", "num_misses",
            ],
            name="eval",
        )

        row = summary.loc["eval"]

        return {
            "MOTA": round(float(row["mota"]), 4),
            "MOTP": round(float(row["motp"]), 4),
            "IDF1": round(float(row["idf1"]), 4),
            "MT": int(row["mostly_tracked"]),
            "ML": int(row["mostly_lost"]),
            "num_switches": int(row["num_switches"]),
            "fragmentations": int(row["num_fragmentations"]),
            "FP": int(row["num_false_positives"]),
            "FN": int(row["num_misses"]),
            "num_frames": int(row["num_frames"]),
            "num_objects": int(row["num_objects"]),
            "num_predictions": int(row["num_predictions"]),
        }

    def _load_gt(self, ground_truth: str | list[dict[str, Any]]) -> "Any":
        import pandas as pd  # noqa: PLC0415

        if isinstance(ground_truth, str):
            return self._parse_mot_file(ground_truth)
        else:
            return pd.DataFrame(ground_truth)

    @staticmethod
    def _parse_mot_file(path: str) -> "Any":
        import pandas as pd  # noqa: PLC0415

        df = pd.read_csv(
            path,
            header=None,
            names=[
                "frame", "id", "x", "y", "w", "h",
                "conf", "class_id", "visibility", "unused",
            ],
            sep="[,\\s]+",
            engine="python",
        )
        df = df[df["id"] > 0]
        df["frame"] = df["frame"].astype(int)
        df["id"] = df["id"].astype(int)
        return df

    @staticmethod
    def _predictions_to_df(
        predictions: list[list[TrackedObject]],
    ) -> "Any":
        import pandas as pd  # noqa: PLC0415

        rows: list[dict[str, Any]] = []
        for frame_id, tracks in enumerate(predictions):
            for t in tracks:
                rows.append({
                    "frame": frame_id,
                    "id": t.track_id,
                    "x": t.bbox.x1,
                    "y": t.bbox.y1,
                    "w": t.bbox.width,
                    "h": t.bbox.height,
                    "conf": t.confidence,
                })
        return pd.DataFrame(rows)

    def summary_table(self, results: dict[str, Any]) -> str:
        """Pretty-print a summary table of tracking metrics."""
        lines = [
            "+-------------------------+-----------+",
            "| Metric                 | Value     |",
            "+-------------------------+-----------+",
            f"| MOTA                   | {results['MOTA']:>8.4f} |",
            f"| MOTP                   | {results['MOTP']:>8.4f} |",
            f"| IDF1                   | {results['IDF1']:>8.4f} |",
            f"| Mostly Tracked         | {results['MT']:>8d} |",
            f"| Mostly Lost            | {results['ML']:>8d} |",
            f"| ID Switches            | {results['num_switches']:>8d} |",
            f"| Fragmentations         | {results['fragmentations']:>8d} |",
            f"| False Positives        | {results['FP']:>8d} |",
            f"| False Negatives        | {results['FN']:>8d} |",
            "+-------------------------+-----------+",
        ]
        return "\n".join(lines)
