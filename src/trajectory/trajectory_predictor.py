from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Any

from src.schemas import Frame, PerceivedObject, TrackedObject
from src.schemas.trajectory import (
    PredictionRecord,
    TrajectoryPoint,
    TrajectoryPrediction,
)
from src.trajectory.excel_exporter import TrajectoryExcelExporter
from src.trajectory.history_manager import TrajectoryHistoryManager
from src.trajectory.prediction_manager import PredictionManager
from src.utils.logger import get_logger


class TrajectoryPredictor:
    """Predicts future spatial trajectories for tracked objects using CA-KF.

    At each incoming video frame, the predictor:
    1. Updates historical observations and filters for active tracks.
    2. Resolves past pending prediction records against current ground truth observations,
       computing ADE, FDE, and endpoint displacement errors.
    3. Forecasts future positions over a dynamically computed grid of time offsets
       ending precisely at ``horizon_seconds``.
    4. Enqueues a detailed ``PredictionRecord`` with all waypoints for evaluation.
    5. Exposes active evaluation matches for visual overlays.
    6. Exports comprehensive Excel evaluation workbooks upon session completion.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        traj_cfg = config.get("trajectory", {})

        pred_cfg = traj_cfg.get("prediction", {})
        self.horizon_seconds: float = float(pred_cfg.get("horizon_seconds", 2.0))
        self.update_interval: float = float(pred_cfg.get("update_interval", 0.1))
        self.model_name: str = str(pred_cfg.get("model", "constant_acceleration"))
        self.history_duration_seconds: float = float(
            pred_cfg.get("history_duration_seconds", 2.0)
        )
        self.minimum_history_points: int = int(
            pred_cfg.get("minimum_history_points", 10)
        )
        self.process_noise_std: float = float(pred_cfg.get("process_noise_std", 1.0))
        self.measurement_noise_std: float = float(
            pred_cfg.get("measurement_noise_std", 2.0)
        )

        track_cfg = traj_cfg.get("tracking", {})
        self.maximum_track_gap_seconds: float = float(
            track_cfg.get("maximum_track_gap_seconds", 0.5)
        )
        self.stale_track_timeout_seconds: float = float(
            track_cfg.get("stale_track_timeout_seconds", 1.0)
        )

        coord_cfg = traj_cfg.get("coordinates", {})
        self.position_reference: str = str(
            coord_cfg.get("position_reference", "bottom_center")
        )

        eval_cfg = traj_cfg.get("evaluation", {})
        self.evaluation_enabled: bool = bool(eval_cfg.get("enabled", True))
        self.match_tolerance_seconds: float = float(
            eval_cfg.get("match_tolerance_seconds", 0.05)
        )
        self.max_records: int = int(eval_cfg.get("max_records", 10000))
        self.success_threshold_pixels: float = float(
            eval_cfg.get("success_threshold_pixels", 20.0)
        )

        export_cfg = traj_cfg.get("export", {})
        self.export_enabled: bool = bool(export_cfg.get("enabled", True))
        self.export_filename: str = str(
            export_cfg.get("filename", "NAYAN_Trajectory_Evaluation.xlsx")
        )
        self.export_output_dir: str = str(export_cfg.get("output_dir", "./reports"))
        self.export_on_session_end: bool = bool(
            export_cfg.get("export_on_session_end", True)
        )

        self._logger = get_logger(f"{__name__}.TrajectoryPredictor")

        # Core subsystems
        self.history_manager = TrajectoryHistoryManager(
            history_duration_seconds=self.history_duration_seconds,
            minimum_history_points=self.minimum_history_points,
            maximum_track_gap_seconds=self.maximum_track_gap_seconds,
            stale_track_timeout_seconds=self.stale_track_timeout_seconds,
            position_reference=self.position_reference,
            process_noise_std=self.process_noise_std,
            measurement_noise_std=self.measurement_noise_std,
            model=self.model_name,
        )

        self.prediction_manager = PredictionManager(
            match_tolerance_seconds=self.match_tolerance_seconds,
            max_records=self.max_records,
            success_threshold_pixels=self.success_threshold_pixels,
        )

        # Precompute or generate dynamic offsets
        self.offsets = self._generate_prediction_offsets(
            self.horizon_seconds, self.update_interval
        )

        # DRAC configuration and physical distance type
        self.drac_config: dict[str, Any] = self.config.get(
            "drac", self.config.get("trajectory", {}).get("drac", {})
        )
        self.distance_type: str = str(
            self.config.get("distance", {}).get("type", "relative")
        )

        # Runtime telemetry (bounded ring buffers to prevent memory growth)
        self.last_active_evaluations: list[PredictionRecord] = []
        self._prediction_latencies: deque[float] = deque(maxlen=10000)
        self._evaluation_latencies: deque[float] = deque(maxlen=10000)
        self._tracked_counts: deque[int] = deque(maxlen=10000)
        self._frames_processed: int = 0

    # ------------------------------------------------------------------ #
    #  Offset Generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_prediction_offsets(
        horizon_seconds: float, update_interval: float
    ) -> list[float]:
        """Dynamically compute forward time offsets ending exactly at horizon_seconds."""
        horizon = max(0.01, float(horizon_seconds))
        interval = max(0.005, float(update_interval))

        offsets: list[float] = []
        t = interval
        while t < horizon - 1e-6:
            offsets.append(round(t, 4))
            t += interval

        # Ensure the final horizon is always the terminal point
        offsets.append(round(horizon, 4))
        return offsets

    def set_prediction_horizon(
        self, horizon_seconds: float, update_interval: float | None = None
    ) -> None:
        """Dynamically update horizon and prediction interval at runtime."""
        self.horizon_seconds = max(0.01, float(horizon_seconds))
        if update_interval is not None:
            self.update_interval = max(0.005, float(update_interval))
        self.offsets = self._generate_prediction_offsets(
            self.horizon_seconds, self.update_interval
        )

    # ------------------------------------------------------------------ #
    #  Prediction Pipeline Step
    # ------------------------------------------------------------------ #

    def predict(
        self,
        objects: list[TrackedObject | PerceivedObject],
        frame: Frame,
        motion: dict[int, Any] | None = None,
        lane_info: Any = None,
        scene_context: Any = None,
    ) -> list[TrajectoryPrediction]:
        """Execute history ingestion, evaluation matching, and future trajectory prediction.

        Args:
            objects: Detected and tracked objects in the current frame.
            frame: Current video frame.
            motion: Optional motion estimation dictionary keyed by track_id.
            lane_info: Optional detected lane information.
            scene_context: Optional synthesised scene context snapshot.

        Returns:
            List of TrajectoryPrediction instances for eligible tracks.
        """
        self._frames_processed += 1
        self._tracked_counts.append(len(objects))
        current_time = frame.timestamp

        # Step 1: Update observations and Kalman filters
        self.history_manager.update(objects, frame)

        # Step 2: Evaluate pending predictions against actual observations
        if self.evaluation_enabled:
            eval_start = time.perf_counter()
            self.prediction_manager.evaluate_pending(
                self.history_manager, current_time
            )
            eval_lat_ms = (time.perf_counter() - eval_start) * 1000.0
            self._evaluation_latencies.append(eval_lat_ms)
            self.last_active_evaluations = self.prediction_manager.get_active_evaluations(
                current_time, max_age_seconds=1.5
            )

        # Step 3: Generate future forecasts for ready tracks
        pred_start = time.perf_counter()
        predictions: list[TrajectoryPrediction] = []

        from src.trajectory.drac import associate_drac_pair

        for obj in objects:
            tid = obj.track_id
            if not self.history_manager.is_track_ready(tid):
                continue

            kf = self.history_manager.get_filter(tid)
            if kf is None:
                continue

            # Extrapolate future positions along dynamic offsets
            future_points_data = kf.predict_future(self.offsets)

            trajectory_points: list[TrajectoryPoint] = []
            for offset, (px, py, (vx, vy)) in zip(self.offsets, future_points_data):
                trajectory_points.append(
                    TrajectoryPoint(
                        future_timestamp=current_time + offset,
                        time_offset=offset,
                        position=(px, py),
                        variance=(vx, vy),
                    )
                )

            final_pos = trajectory_points[-1].position
            cur_pos = kf.position
            cur_vel = kf.velocity
            cur_acc = kf.acceleration
            instance_id = self.history_manager.get_track_instance_id(tid)

            prediction = TrajectoryPrediction(
                track_id=tid,
                track_instance_id=instance_id,
                class_name=self.history_manager.get_class_name(tid),
                prediction_timestamp=current_time,
                horizon_seconds=self.horizon_seconds,
                current_position=cur_pos,
                estimated_velocity=cur_vel,
                estimated_acceleration=cur_acc,
                predicted_positions=trajectory_points,
                final_predicted_position=final_pos,
                predictor=self.model_name,
                uncertainty={
                    "position_variance": [
                        round(kf.position_variance[0], 4),
                        round(kf.position_variance[1], 4),
                    ]
                },
            )
            predictions.append(prediction)

            # Step 4: Register prediction record for future ground truth evaluation
            if self.evaluation_enabled:
                drac_res = associate_drac_pair(
                    target_track_id=tid,
                    target_class_name=prediction.class_name,
                    target_distance=getattr(obj, "distance", None),
                    target_depth_confidence=getattr(obj, "depth_confidence", 1.0),
                    all_objects=objects,
                    motion_dict=motion,
                    lane_info=lane_info,
                    scene_context=scene_context,
                    distance_type=self.distance_type,
                    drac_config=self.drac_config,
                )

                self.prediction_manager.add_prediction(
                    track_id=tid,
                    track_instance_id=instance_id,
                    class_name=prediction.class_name,
                    prediction_timestamp=current_time,
                    horizon_seconds=self.horizon_seconds,
                    current_position=cur_pos,
                    predicted_position=final_pos,
                    predictor=self.model_name,
                    uncertainty=prediction.uncertainty,
                    waypoints=trajectory_points,
                    following_track_id=drac_res.following_track_id,
                    leading_track_id=drac_res.leading_track_id,
                    following_speed_mps=drac_res.following_speed_mps,
                    leading_speed_mps=drac_res.leading_speed_mps,
                    inter_vehicle_gap_m=drac_res.inter_vehicle_gap_m,
                    drac_mps2=drac_res.drac_mps2,
                    drac_status=drac_res.drac_status,
                )

        pred_lat_ms = (time.perf_counter() - pred_start) * 1000.0
        self._prediction_latencies.append(pred_lat_ms)

        return predictions

    # ------------------------------------------------------------------ #
    #  Excel Export
    # ------------------------------------------------------------------ #

    def export_evaluation(self, output_path: str | Path | None = None) -> Path | None:
        """Export trajectory evaluation workbook to Excel."""
        if not self.export_enabled:
            self._logger.info("Excel export disabled in configuration.")
            return None

        if output_path is not None:
            dest_path = Path(output_path)
        else:
            dest_path = Path(self.export_output_dir) / self.export_filename

        records = self.prediction_manager.get_records()

        # Telemetry metrics
        avg_pred_lat = (
            sum(self._prediction_latencies) / len(self._prediction_latencies)
            if self._prediction_latencies
            else 0.0
        )
        avg_eval_lat = (
            sum(self._evaluation_latencies) / len(self._evaluation_latencies)
            if self._evaluation_latencies
            else 0.0
        )
        avg_tracked = (
            sum(self._tracked_counts) / len(self._tracked_counts)
            if self._tracked_counts
            else 0.0
        )

        video_cfg = self.config.get("video", {})
        video_src = video_cfg.get("path", video_cfg.get("type", "camera"))

        metadata = {
            "video_source": video_src,
            "video_resolution": f"{video_cfg.get('width', 1280)} x {video_cfg.get('height', 720)}",
            "model": self.model_name,
            "horizon_seconds": self.horizon_seconds,
            "update_interval": self.update_interval,
            "match_tolerance_seconds": self.match_tolerance_seconds,
            "success_threshold_pixels": self.success_threshold_pixels,
            "history_duration_seconds": self.history_duration_seconds,
            "minimum_history_points": self.minimum_history_points,
            "maximum_track_gap_seconds": self.maximum_track_gap_seconds,
            "stale_track_timeout_seconds": self.stale_track_timeout_seconds,
            "coordinate_system": "image",
            "position_reference": self.position_reference,
            "drac_enabled": self.drac_config.get("enabled", True),
            "drac_require_metric": self.drac_config.get("require_metric_units", True),
            "drac_minimum_gap_meters": self.drac_config.get("minimum_gap_meters", 0.01),
        }


        perf = {
            "avg_prediction_latency_ms": round(avg_pred_lat, 2),
            "avg_evaluation_latency_ms": round(avg_eval_lat, 2),
            "total_predictions_emitted": len(records),
            "avg_tracked_objects": round(avg_tracked, 1),
            "total_frames_processed": self._frames_processed,
        }

        exporter = TrajectoryExcelExporter(
            output_path=str(dest_path),
            success_threshold_pixels=self.success_threshold_pixels,
        )
        return exporter.export(records, metadata=metadata, performance_stats=perf)
