from __future__ import annotations

import math
import numpy as np
import pytest

from src.schemas import BoundingBox, Frame, PerceivedObject, TrackedObject
from src.schemas.trajectory import EvaluationStatus, PredictionRecord, TrajectoryPrediction
from src.trajectory.factory import build_trajectory_predictor
from src.trajectory.history_manager import TrajectoryHistoryManager
from src.trajectory.kalman_filter import ConstantAccelerationKalmanFilter
from src.trajectory.prediction_manager import PredictionManager
from src.trajectory.trajectory_predictor import TrajectoryPredictor


def create_dummy_frame(frame_id: int, timestamp: float) -> Frame:
    """Helper to construct dummy Frame objects for testing."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return Frame(
        image=img,
        frame_id=frame_id,
        timestamp=timestamp,
        source="test_stream",
    )


def create_dummy_object(
    track_id: int,
    bbox: BoundingBox,
    class_name: str = "car",
) -> TrackedObject:
    """Helper to construct dummy TrackedObject."""
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
    )


# ===================================================================== #
# 1. Position Extraction Tests
# ===================================================================== #

class TestPositionExtraction:
    def test_bottom_center_extraction(self) -> None:
        mgr = TrajectoryHistoryManager(position_reference="bottom_center")
        box = BoundingBox(100.0, 200.0, 300.0, 400.0)
        pos = mgr.extract_position(box)
        # x = (100 + 300) / 2 = 200, y = y2 = 400
        assert pos == (200.0, 400.0)

    def test_center_extraction(self) -> None:
        mgr = TrajectoryHistoryManager(position_reference="center")
        box = BoundingBox(100.0, 200.0, 300.0, 400.0)
        pos = mgr.extract_position(box)
        # cx = 200, cy = 300
        assert pos == (200.0, 300.0)


# ===================================================================== #
# 2. Multiple Independent Track Histories
# ===================================================================== #

class TestMultipleTrackHistories:
    def test_independent_tracks(self) -> None:
        mgr = TrajectoryHistoryManager(minimum_history_points=2)
        frame1 = create_dummy_frame(1, 10.0)
        obj1 = create_dummy_object(1, BoundingBox(10.0, 10.0, 30.0, 50.0))
        obj2 = create_dummy_object(2, BoundingBox(100.0, 100.0, 140.0, 160.0))

        mgr.update([obj1, obj2], frame1)

        frame2 = create_dummy_frame(2, 10.1)
        obj1_next = create_dummy_object(1, BoundingBox(15.0, 10.0, 35.0, 50.0))
        obj2_next = create_dummy_object(2, BoundingBox(110.0, 100.0, 150.0, 160.0))
        mgr.update([obj1_next, obj2_next], frame2)

        assert mgr.active_track_ids() == {1, 2}
        h1 = mgr.get_history(1)
        h2 = mgr.get_history(2)
        assert len(h1) == 2
        assert len(h2) == 2
        assert h1[0].position != h2[0].position


# ===================================================================== #
# 3. Variable FPS / Variable Elapsed Time
# ===================================================================== #

class TestVariableFps:
    def test_variable_time_steps(self) -> None:
        kf = ConstantAccelerationKalmanFilter(initial_position=(100.0, 200.0))
        # Variable dt updates
        for dt in [0.033, 0.05, 0.016, 0.1]:
            kf.predict(dt)
            kf.update((100.0 + dt * 10, 200.0 + dt * 5))
            assert np.all(np.isfinite(kf.x))
            assert np.all(np.isfinite(kf.P))


# ===================================================================== #
# 4. Constant Velocity Trajectory Estimation & Prediction
# ===================================================================== #

class TestConstantVelocity:
    def test_linear_trajectory_prediction(self) -> None:
        # Ground truth: starts at (100, 100), moves at vx = 50 px/s, vy = 20 px/s
        vx_true = 50.0
        vy_true = 20.0
        dt = 0.05
        kf = ConstantAccelerationKalmanFilter(
            initial_position=(100.0, 100.0),
            process_noise_std=0.5,
            measurement_noise_std=0.5,
        )

        for step in range(1, 30):
            t = step * dt
            kf.predict(dt)
            meas = (100.0 + vx_true * t, 100.0 + vy_true * t)
            kf.update(meas)

        # Check estimated velocity
        est_vx, est_vy = kf.velocity
        assert abs(est_vx - vx_true) < 3.0
        assert abs(est_vy - vy_true) < 3.0

        # Predict ahead by 1.0 second
        futures = kf.predict_future([1.0])
        pred_x, pred_y, _ = futures[0]
        cur_x, cur_y = kf.position
        assert abs((pred_x - cur_x) - vx_true * 1.0) < 5.0
        assert abs((pred_y - cur_y) - vy_true * 1.0) < 5.0


# ===================================================================== #
# 5. Constant Acceleration Trajectory Estimation & Prediction
# ===================================================================== #

class TestConstantAcceleration:
    def test_accelerating_trajectory_prediction(self) -> None:
        # x(t) = 0.5 * ax * t^2 with ax = 10 px/s^2
        ax_true = 10.0
        dt = 0.05
        kf = ConstantAccelerationKalmanFilter(
            initial_position=(0.0, 0.0),
            process_noise_std=1.0,
            measurement_noise_std=0.2,
        )

        for step in range(1, 40):
            t = step * dt
            kf.predict(dt)
            pos_x = 0.5 * ax_true * (t**2)
            kf.update((pos_x, 0.0))

        # Check acceleration estimation
        est_ax, _ = kf.acceleration
        assert abs(est_ax - ax_true) < 4.0


# ===================================================================== #
# 6. Noisy Detections & Kalman Smoothing
# ===================================================================== #

class TestNoisyDetections:
    def test_noise_reduction(self) -> None:
        np.random.seed(42)
        true_pos = 200.0
        noise_std = 10.0
        kf = ConstantAccelerationKalmanFilter(
            initial_position=(true_pos, true_pos),
            process_noise_std=0.1,
            measurement_noise_std=noise_std,
        )

        raw_errors: list[float] = []
        filtered_errors: list[float] = []

        for _ in range(40):
            kf.predict(0.05)
            noise_x = np.random.normal(0, noise_std)
            noise_y = np.random.normal(0, noise_std)
            meas = (true_pos + noise_x, true_pos + noise_y)
            kf.update(meas)

            raw_errors.append(abs(noise_x))
            filtered_errors.append(abs(kf.position[0] - true_pos))

        # Average filtered error should be significantly lower than raw noise
        assert np.mean(filtered_errors[-20:]) < np.mean(raw_errors)


# ===================================================================== #
# 7. Missing Observations
# ===================================================================== #

class TestMissingObservations:
    def test_predict_without_update(self) -> None:
        kf = ConstantAccelerationKalmanFilter(
            initial_position=(100.0, 100.0),
            initial_velocity=(20.0, 0.0),
        )
        # Advance through multiple frames where detection was missed
        for _ in range(5):
            kf.predict(0.1)

        # Position should propagate with velocity (100 + 20 * 0.5 = 110)
        assert abs(kf.position[0] - 110.0) < 1.0


# ===================================================================== #
# 8. Track ID Changes and Reset on Long Gaps
# ===================================================================== #

class TestTrackIdResets:
    def test_gap_triggers_filter_reset(self) -> None:
        mgr = TrajectoryHistoryManager(maximum_track_gap_seconds=0.5)
        f1 = create_dummy_frame(1, 10.0)
        mgr.update([create_dummy_object(1, BoundingBox(10.0, 10.0, 30.0, 30.0))], f1)

        # Large jump in time (2.0s > 0.5s) with a distant box (ID reused)
        f2 = create_dummy_frame(2, 12.0)
        mgr.update([create_dummy_object(1, BoundingBox(500.0, 500.0, 520.0, 520.0))], f2)

        kf = mgr.get_filter(1)
        assert kf is not None
        # Velocity should not be 500/2 px/s because filter was reset upon gap!
        assert abs(kf.velocity[0]) < 10.0


# ===================================================================== #
# 9. Stale-Track Removal
# ===================================================================== #

class TestStaleTrackRemoval:
    def test_stale_eviction(self) -> None:
        mgr = TrajectoryHistoryManager(stale_track_timeout_seconds=1.0)
        f1 = create_dummy_frame(1, 10.0)
        mgr.update([create_dummy_object(5, BoundingBox(10.0, 10.0, 20.0, 20.0))], f1)
        assert 5 in mgr.active_track_ids()

        # Update at 11.5s (> 1.0s timeout) without track 5
        f2 = create_dummy_frame(2, 11.5)
        mgr.update([], f2)
        assert 5 not in mgr.active_track_ids()
        assert mgr.get_filter(5) is None


# ===================================================================== #
# 10. Configurable Prediction Horizon
# ===================================================================== #

class TestConfigurableHorizon:
    def test_different_horizons(self) -> None:
        cfg1 = {"trajectory": {"prediction": {"horizon_seconds": 1.0, "update_interval": 0.2}}}
        pred1 = TrajectoryPredictor(cfg1)
        assert pred1.horizon_seconds == 1.0
        assert pred1.offsets[-1] == 1.0

        cfg2 = {"trajectory": {"prediction": {"horizon_seconds": 3.5, "update_interval": 0.5}}}
        pred2 = TrajectoryPredictor(cfg2)
        assert pred2.horizon_seconds == 3.5
        assert pred2.offsets[-1] == 3.5


# ===================================================================== #
# 11. Configurable Prediction Interval
# ===================================================================== #

class TestConfigurableInterval:
    def test_dynamic_points_count(self) -> None:
        cfg = {"trajectory": {"prediction": {"horizon_seconds": 2.0, "update_interval": 0.1}}}
        pred = TrajectoryPredictor(cfg)
        # Expected points: 0.1, 0.2, ..., 2.0 -> 20 points
        assert len(pred.offsets) == 20
        assert pred.offsets[0] == 0.1
        assert pred.offsets[-1] == 2.0


# ===================================================================== #
# 12. Correct Final Prediction Timestamp and Non-Divisible Handling
# ===================================================================== #

class TestFinalPredictionTimestamp:
    def test_non_divisible_horizon(self) -> None:
        # Horizon 1.0, interval 0.3 -> [0.3, 0.6, 0.9, 1.0]
        offsets = TrajectoryPredictor._generate_prediction_offsets(1.0, 0.3)
        assert offsets == [0.3, 0.6, 0.9, 1.0]
        assert offsets[-1] == 1.0


# ===================================================================== #
# 13. Prediction Record Creation and Structure
# ===================================================================== #

class TestPredictionRecordCreation:
    def test_record_fields(self) -> None:
        pm = PredictionManager()
        rec = pm.add_prediction(
            track_id=10,
            class_name="truck",
            prediction_timestamp=50.0,
            horizon_seconds=2.0,
            current_position=(100.0, 200.0),
            predicted_position=(150.0, 250.0),
            predictor="constant_acceleration_kalman",
        )
        assert rec.track_id == 10
        assert rec.target_timestamp == 52.0
        assert rec.status == EvaluationStatus.PENDING
        d = rec.to_dict()
        assert d["prediction_id"] == rec.prediction_id
        assert d["target_timestamp"] == 52.0


# ===================================================================== #
# 14. Future-Position Matching Interface (Pending -> Matched / Lost)
# ===================================================================== #

class TestEvaluationMatching:
    def test_successful_match_and_error_calculation(self) -> None:
        hist_mgr = TrajectoryHistoryManager(maximum_track_gap_seconds=0.5)
        pred_mgr = PredictionManager(match_tolerance_seconds=0.05)

        # Prediction made at t=10.0 with horizon=1.0 -> target=11.0, predicted=(200, 200)
        rec = pred_mgr.add_prediction(
            track_id=1,
            class_name="car",
            prediction_timestamp=10.0,
            horizon_seconds=1.0,
            current_position=(100.0, 100.0),
            predicted_position=(200.0, 200.0),
        )

        # Frame arrives at t=10.5 -> target not reached
        f_mid = create_dummy_frame(1, 10.5)
        hist_mgr.update([create_dummy_object(1, BoundingBox(140.0, 140.0, 160.0, 150.0))], f_mid)
        resolved = pred_mgr.evaluate_pending(hist_mgr, current_timestamp=10.5)
        assert len(resolved) == 0
        assert rec.status == EvaluationStatus.PENDING

        # Frame arrives at t=11.0 -> target reached, actual pos=(195, 205)
        f_target = create_dummy_frame(2, 11.0)
        # bottom center: x=(185+205)/2=195, y2=205
        hist_mgr.update([create_dummy_object(1, BoundingBox(185.0, 190.0, 205.0, 205.0))], f_target)
        resolved = pred_mgr.evaluate_pending(hist_mgr, current_timestamp=11.0)
        assert len(resolved) == 1
        assert rec.status == EvaluationStatus.MATCHED
        assert rec.actual_position == (195.0, 205.0)
        # Error: sqrt((200-195)^2 + (200-205)^2) = sqrt(25 + 25) = 7.071
        assert abs(rec.error_distance - math.sqrt(50.0)) < 1e-3

    def test_track_lost_evaluation(self) -> None:
        hist_mgr = TrajectoryHistoryManager(stale_track_timeout_seconds=0.5)
        pred_mgr = PredictionManager(match_tolerance_seconds=0.05)

        rec = pred_mgr.add_prediction(
            track_id=2,
            class_name="car",
            prediction_timestamp=10.0,
            horizon_seconds=1.0,
            current_position=(100.0, 100.0),
            predicted_position=(200.0, 200.0),
        )

        # Track disappeared, clock advances past 11.0 + 0.05
        f_late = create_dummy_frame(10, 11.5)
        hist_mgr.update([], f_late)
        resolved = pred_mgr.evaluate_pending(hist_mgr, current_timestamp=11.5)
        assert len(resolved) == 1
        assert rec.status == EvaluationStatus.TRACK_LOST


# ===================================================================== #
# 15. Invalid / Discontinuous Measurements Handling
# ===================================================================== #

class TestInvalidMeasurements:
    def test_zero_or_negative_dt_safe(self) -> None:
        kf = ConstantAccelerationKalmanFilter(initial_position=(50.0, 50.0))
        kf.predict(0.0)
        kf.predict(-0.5)
        assert kf.position == (50.0, 50.0)

    def test_predictor_disabled(self) -> None:
        cfg = {"trajectory": {"enabled": False}}
        predictor = build_trajectory_predictor(cfg)
        assert predictor is None


# ===================================================================== #
# 16. PerceptionPipeline End-to-End Integration
# ===================================================================== #

class TestPipelineIntegration:
    def test_pipeline_trajectory_execution(self) -> None:
        from src.core.pipeline import PerceptionPipeline

        config = {
            "video": {"type": "file", "path": "test.mp4"},
            "detector": {"type": "yolo"},
            "tracker": {"type": "bytetrack"},
            "depth": {"enabled": False},
            "fusion": {"sampling": "median_center"},
            "distance": {"type": "relative"},
            "memory": {"max_history_per_track": 10},
            "motion": {"enabled": False},
            "ttc": {"type": "simple"},
            "risk": {"type": "score_based"},
            "warning": {"channels": []},
            "lane": {"enabled": False},
            "signs": {"enabled": False},
            "scene": {"enabled": False},
            "pose": {"enabled": False},
            "visualization": {"enabled": False},
            "trajectory": {
                "enabled": True,
                "prediction": {
                    "horizon_seconds": 1.5,
                    "update_interval": 0.1,
                    "minimum_history_points": 2,
                },
            },
        }

        pipeline = PerceptionPipeline(config)
        assert pipeline.trajectory_predictor is not None

        # Feed consecutive frames with dummy tracked objects
        for i in range(5):
            t = 10.0 + i * 0.1
            f = create_dummy_frame(i + 1, t)
            obj = PerceivedObject(
                track_id=42,
                class_name="car",
                bbox=BoundingBox(100.0 + i * 5, 200.0, 140.0 + i * 5, 260.0),
                depth=15.0,
                distance=15.0,
                depth_confidence=0.9,
            )
            # Directly test _predict_trajectories
            preds = pipeline._predict_trajectories([obj], f)
            if i >= 1:  # min 2 points reached
                assert len(preds) == 1
                pred = preds[0]
                assert pred.track_id == 42
                assert pred.horizon_seconds == 1.5
                assert len(pred.predicted_positions) == 15
                assert pred.final_predicted_position == pred.predicted_positions[-1].position
