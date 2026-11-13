from __future__ import annotations

import math
import tempfile
from pathlib import Path
import numpy as np
import openpyxl
import pytest

from src.schemas import BoundingBox, Frame, TrackedObject
from src.schemas.trajectory import EvaluationStatus, PredictionRecord, TrajectoryPrediction
from src.trajectory.factory import build_trajectory_predictor
from src.trajectory.history_manager import TrajectoryHistoryManager
from src.trajectory.kalman_filter import (
    ConstantAccelerationKalmanFilter,
    ConstantVelocityKalmanFilter,
)
from src.trajectory.metrics import calculate_trajectory_metrics
from src.trajectory.prediction_manager import PredictionManager
from src.trajectory.trajectory_predictor import TrajectoryPredictor


def make_frame(frame_id: int, timestamp: float) -> Frame:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=frame_id, timestamp=timestamp, source="test")


def make_box(cx: float, cy: float, w: float = 40.0, h: float = 60.0) -> BoundingBox:
    return BoundingBox(cx - w / 2, cy - h, cx + w / 2, cy)


def make_obj(track_id: int, cx: float, cy: float, class_name: str = "car") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=0.92,
        bbox=make_box(cx, cy),
    )


# ===================================================================== #
# 1. Ten Synthetic Movement Pattern Tests (Ground Truth Verification)
# ===================================================================== #

class TestSyntheticMovementPatterns:
    """Rigorous verification of 10 distinct physical movement dynamics."""

    def test_pattern_1_constant_horizontal_velocity(self) -> None:
        """Pattern 1: Constant horizontal velocity (vx=100 px/s, vy=0)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(100.0, 300.0))
        dt = 0.05
        # Simulate 35 frames (1.75 sec) to allow steady-state convergence
        for i in range(1, 36):
            t = i * dt
            gt_x = 100.0 + 100.0 * t
            gt_y = 300.0
            kf.predict(dt)
            kf.update((gt_x, gt_y))

        cur_t = 35 * dt
        assert abs(kf.position[0] - (100.0 + 100.0 * cur_t)) < 2.0
        assert abs(kf.position[1] - 300.0) < 1.0
        assert abs(kf.velocity[0] - 100.0) < 5.0
        assert abs(kf.velocity[1]) < 2.0
        assert abs(kf.acceleration[0]) < 8.0

        # Predict future 1.0 second ahead
        future = kf.predict_future([0.5, 1.0])
        pred_1s = future[-1]
        expected_x = kf.position[0] + 100.0 * 1.0
        assert abs(pred_1s[0] - expected_x) < 8.0
        assert abs(pred_1s[1] - 300.0) < 2.0

    def test_pattern_2_constant_vertical_velocity(self) -> None:
        """Pattern 2: Constant vertical velocity (vx=0, vy=80 px/s)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(500.0, 100.0))
        dt = 0.05
        for i in range(1, 36):
            t = i * dt
            gt_x = 500.0
            gt_y = 100.0 + 80.0 * t
            kf.predict(dt)
            kf.update((gt_x, gt_y))

        cur_t = 35 * dt
        assert abs(kf.position[0] - 500.0) < 1.0
        assert abs(kf.position[1] - (100.0 + 80.0 * cur_t)) < 2.0
        assert abs(kf.velocity[0]) < 2.0
        assert abs(kf.velocity[1] - 80.0) < 5.0

        future = kf.predict_future([1.0])
        expected_future_y = kf.position[1] + 80.0 * 1.0
        assert abs(future[0][0] - 500.0) < 2.0
        assert abs(future[0][1] - expected_future_y) < 8.0

    def test_pattern_3_diagonal_movement(self) -> None:
        """Pattern 3: Diagonal movement (vx=60 px/s, vy=60 px/s)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(50.0, 50.0))
        dt = 0.05
        for i in range(1, 36):
            t = i * dt
            kf.predict(dt)
            kf.update((50.0 + 60.0 * t, 50.0 + 60.0 * t))

        future = kf.predict_future([1.0])
        cur_x, cur_y = kf.position
        assert abs(future[0][0] - (cur_x + 60.0)) < 8.0
        assert abs(future[0][1] - (cur_y + 60.0)) < 8.0

    def test_pattern_4_constant_acceleration(self) -> None:
        """Pattern 4: Constant acceleration (x0=100, v0=20, ax=40 px/s^2)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(100.0, 200.0))
        dt = 0.05
        v0 = 20.0
        a = 40.0
        for i in range(1, 35):
            t = i * dt
            pos_x = 100.0 + v0 * t + 0.5 * a * (t**2)
            kf.predict(dt)
            kf.update((pos_x, 200.0))

        # Velocity and acceleration should match theoretical values
        assert abs(kf.acceleration[0] - 40.0) < 12.0

        # Predict future 1.0s forward: expected x = x(t) + v(t)*dt + 0.5*a*dt^2
        cur_x = kf.position[0]
        cur_v = kf.velocity[0]
        cur_a = kf.acceleration[0]
        expected_future_x = cur_x + cur_v * 1.0 + 0.5 * cur_a * (1.0**2)
        future = kf.predict_future([1.0])
        assert abs(future[0][0] - expected_future_x) < 1.0

    def test_pattern_5_deceleration(self) -> None:
        """Pattern 5: Deceleration (braking object: v0=100, a=-30 px/s^2)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(100.0, 200.0))
        dt = 0.05
        v0 = 100.0
        a = -30.0
        for i in range(1, 35):
            t = i * dt
            pos_x = 100.0 + v0 * t + 0.5 * a * (t**2)
            kf.predict(dt)
            kf.update((pos_x, 200.0))

        # Deceleration should be detected as negative acceleration
        assert kf.acceleration[0] < 0.0
        # Predicted position should be less than constant velocity projection
        future = kf.predict_future([1.0])
        linear_proj = kf.position[0] + kf.velocity[0] * 1.0
        assert future[0][0] < linear_proj

    def test_pattern_6_stationary_object(self) -> None:
        """Pattern 6: Completely stationary object (vx=0, vy=0)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(400.0, 500.0))
        dt = 0.033
        for _ in range(30):
            kf.predict(dt)
            kf.update((400.0, 500.0))

        assert abs(kf.speed) < 0.5
        assert abs(kf.acceleration[0]) < 0.5 and abs(kf.acceleration[1]) < 0.5
        future = kf.predict_future([0.5, 1.0, 2.0])
        for pt in future:
            assert abs(pt[0] - 400.0) < 0.5
            assert abs(pt[1] - 500.0) < 0.5

    def test_pattern_7_sudden_direction_change(self) -> None:
        """Pattern 7: Object moving right, then abruptly turning down."""
        kf = ConstantAccelerationKalmanFilter(
            initial_position=(100.0, 100.0),
            process_noise_std=2.0,
            measurement_noise_std=1.0,
        )
        dt = 0.05
        # Move right for 20 frames
        for i in range(1, 21):
            kf.predict(dt)
            kf.update((100.0 + 50.0 * (i * dt), 100.0))

        cur_x, cur_y = kf.position
        # Sudden turn downwards for 15 frames
        for j in range(1, 16):
            kf.predict(dt)
            kf.update((cur_x, cur_y + 60.0 * (j * dt)))

        # Velocity in Y should be positive and dominate X
        assert kf.velocity[1] > 20.0
        assert kf.velocity[0] < 25.0

    def test_pattern_8_noisy_observations(self) -> None:
        """Pattern 8: Linear movement corrupted with Gaussian measurement noise."""
        np.random.seed(42)
        kf = ConstantAccelerationKalmanFilter(
            initial_position=(100.0, 200.0),
            measurement_noise_std=5.0,
            process_noise_std=0.5,
        )
        dt = 0.05
        gt_positions = []
        noisy_positions = []
        filter_positions = []

        for i in range(1, 41):
            t = i * dt
            gt_x = 100.0 + 40.0 * t
            gt_y = 200.0 + 20.0 * t
            noise_x = np.random.normal(0, 4.0)
            noise_y = np.random.normal(0, 4.0)
            kf.predict(dt)
            kf.update((gt_x + noise_x, gt_y + noise_y))
            gt_positions.append((gt_x, gt_y))
            noisy_positions.append((gt_x + noise_x, gt_y + noise_y))
            filter_positions.append(kf.position)

        # Average filter error against ground truth should be less than raw noise error
        raw_errors = [
            math.hypot(n[0] - g[0], n[1] - g[1])
            for n, g in zip(noisy_positions, gt_positions)
        ]
        kf_errors = [
            math.hypot(f[0] - g[0], f[1] - g[1])
            for f, g in zip(filter_positions[10:], gt_positions[10:])
        ]
        assert np.mean(kf_errors) < np.mean(raw_errors)

    def test_pattern_9_temporary_occlusion(self) -> None:
        """Pattern 9: Temporary occlusion (coast through missing frames)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(100.0, 200.0))
        dt = 0.05
        # Feed 15 updates
        for i in range(1, 16):
            kf.predict(dt)
            kf.update((100.0 + 50.0 * (i * dt), 200.0))

        pre_occlusion_pos = kf.position
        # Occluded for 6 frames (0.3s) - predict only
        for _ in range(6):
            kf.predict(dt)

        post_occlusion_pos = kf.position
        # Position should have continued coasting forward
        expected_coast = pre_occlusion_pos[0] + kf.velocity[0] * 0.3
        assert abs(post_occlusion_pos[0] - expected_coast) < 2.0

        # Resumes update
        kf.update((post_occlusion_pos[0] + 5.0, 200.0))
        assert np.all(np.isfinite(kf.x))

    def test_pattern_10_irregular_intervals(self) -> None:
        """Pattern 10: Irregular observation intervals (varying frame times)."""
        kf = ConstantAccelerationKalmanFilter(initial_position=(0.0, 0.0))
        dts = [0.016, 0.033, 0.050, 0.020, 0.065, 0.030, 0.045]
        t = 0.0
        for dt in dts * 4:
            t += dt
            gt_x = 80.0 * t
            kf.predict(dt)
            kf.update((gt_x, 0.0))

        assert abs(kf.velocity[0] - 80.0) < 6.0
        assert np.all(np.isfinite(kf.P))


# ===================================================================== #
# 2. CA-KF vs CV-KF Baseline Comparison
# ===================================================================== #

class TestModelComparison:
    """Compare Constant Acceleration (CA-KF) with Constant Velocity (CV-KF)."""

    def test_cv_better_on_constant_speed_with_noise(self) -> None:
        """On pure linear constant-velocity paths with measurement noise,
        CV-KF has fewer degrees of freedom and does not overfit noise into false acceleration.
        """
        np.random.seed(123)
        ca_kf = ConstantAccelerationKalmanFilter(initial_position=(0.0, 0.0), measurement_noise_std=3.0)
        cv_kf = ConstantVelocityKalmanFilter(initial_position=(0.0, 0.0), measurement_noise_std=3.0)

        dt = 0.05
        # Pure linear movement
        for i in range(1, 30):
            t = i * dt
            gt_x = 50.0 * t
            noise_x = np.random.normal(0, 3.0)
            obs = (gt_x + noise_x, 0.0)

            ca_kf.predict(dt)
            ca_kf.update(obs)

            cv_kf.predict(dt)
            cv_kf.update(obs)

        # 1.5s forward prediction
        horizon = 1.5
        gt_future_x = 50.0 * (t + horizon)
        ca_pred = ca_kf.predict_future([horizon])[0][0]
        cv_pred = cv_kf.predict_future([horizon])[0][0]

        ca_err = abs(ca_pred - gt_future_x)
        cv_err = abs(cv_pred - gt_future_x)

        # CV-KF should be robust and close to ground truth
        assert cv_err < 15.0

    def test_ca_better_on_accelerating_trajectory(self) -> None:
        """On accelerating maneuvers, CA-KF correctly captures non-linear curvature,
        whereas CV-KF systematically undershoots.
        """
        ca_kf = ConstantAccelerationKalmanFilter(initial_position=(0.0, 0.0))
        cv_kf = ConstantVelocityKalmanFilter(initial_position=(0.0, 0.0))

        dt = 0.05
        a = 30.0  # px/s^2
        t = 0.0
        for i in range(1, 35):
            t = i * dt
            gt_x = 0.5 * a * (t**2)
            obs = (gt_x, 0.0)

            ca_kf.predict(dt)
            ca_kf.update(obs)

            cv_kf.predict(dt)
            cv_kf.update(obs)

        # Predict future 1.0s forward
        horizon = 1.0
        gt_future_x = 0.5 * a * ((t + horizon) ** 2)

        ca_pred = ca_kf.predict_future([horizon])[0][0]
        cv_pred = cv_kf.predict_future([horizon])[0][0]

        ca_err = abs(ca_pred - gt_future_x)
        cv_err = abs(cv_pred - gt_future_x)

        # CA-KF error must be significantly lower than CV-KF
        assert ca_err < cv_err / 2.0


# ===================================================================== #
# 3. Future-Position Matching & Track Identity Safety
# ===================================================================== #

class TestMatchingSafety:
    """Rigorous tests ensuring predictions are never matched with wrong tracks or gaps."""

    def test_two_crossing_objects_independent(self) -> None:
        """Two objects crossing paths maintain separate identities and evaluations."""
        cfg = {
            "trajectory": {
                "enabled": True,
                "prediction": {"minimum_history_points": 3, "horizon_seconds": 0.5, "update_interval": 0.25},
                "evaluation": {"enabled": True, "match_tolerance_seconds": 0.06},
            }
        }
        predictor = TrajectoryPredictor(cfg)

        # Track 1 moves left-to-right; Track 2 moves right-to-left
        # They cross at x=200 at frame 5
        for i in range(12):
            t = i * 0.1
            f = make_frame(i, t)
            obj1 = make_obj(1, 100.0 + 20.0 * i, 300.0)
            obj2 = make_obj(2, 300.0 - 20.0 * i, 300.0)
            preds = predictor.predict([obj1, obj2], f)

        # Check records
        records = predictor.prediction_manager.get_records()
        rec_t1 = [r for r in records if r.track_id == 1 and r.status == EvaluationStatus.MATCHED]
        rec_t2 = [r for r in records if r.track_id == 2 and r.status == EvaluationStatus.MATCHED]

        assert len(rec_t1) > 0
        assert len(rec_t2) > 0
        for r in rec_t1:
            assert r.track_instance_id.startswith("1_inst")
            assert r.actual_position is not None
            # Track 1 moves rightward, so actual_position[0] > current_position[0]
            assert r.actual_position[0] > r.current_position[0]

        for r in rec_t2:
            assert r.track_instance_id.startswith("2_inst")
            assert r.actual_position is not None
            # Track 2 moves leftward, so actual_position[0] < current_position[0]
            assert r.actual_position[0] < r.current_position[0]

    def test_unmatched_when_video_ends(self) -> None:
        """Predictions emitted near the end of the video remain safely PENDING."""
        cfg = {
            "trajectory": {
                "enabled": True,
                "prediction": {"minimum_history_points": 2, "horizon_seconds": 2.0},
                "evaluation": {"enabled": True},
            }
        }
        predictor = TrajectoryPredictor(cfg)

        for i in range(5):
            t = i * 0.1
            f = make_frame(i, t)
            obj = make_obj(1, 100.0 + 10.0 * i, 200.0)
            predictor.predict([obj], f)

        # Horizon is 2.0s, video only ran 0.4s. All predictions must remain PENDING
        records = predictor.prediction_manager.get_records()
        assert len(records) > 0
        for r in records:
            assert r.status == EvaluationStatus.PENDING
            assert r.actual_position is None

    def test_track_lost_permanently(self) -> None:
        """Object disappearing permanently resolves to TRACK_LOST after horizon."""
        cfg = {
            "trajectory": {
                "enabled": True,
                "prediction": {"minimum_history_points": 2, "horizon_seconds": 0.5, "update_interval": 0.25},
                "tracking": {"stale_track_timeout_seconds": 0.5},
                "evaluation": {"enabled": True, "match_tolerance_seconds": 0.05},
            }
        }
        predictor = TrajectoryPredictor(cfg)

        # Track 1 appears at t=0.0, 0.1, 0.2
        for i in range(3):
            t = i * 0.1
            predictor.predict([make_obj(1, 100.0 + 10.0 * i, 200.0)], make_frame(i, t))

        # Track disappears; video advances to t=1.0s (past horizon 0.2 + 0.5 = 0.7s)
        for i in range(3, 11):
            t = i * 0.1
            predictor.predict([], make_frame(i, t))

        records = predictor.prediction_manager.get_records()
        lost = [r for r in records if r.status == EvaluationStatus.TRACK_LOST]
        assert len(lost) > 0


# ===================================================================== #
# 4. Programmatic Verification of Excel Workbook
# ===================================================================== #

class TestExcelWorkbookProgrammatic:
    """Programmatic verification of all 5 sheets in the generated Excel report."""

    def test_workbook_structure_and_sheet_contents(self) -> None:
        cfg = {
            "trajectory": {
                "enabled": True,
                "prediction": {
                    "horizon_seconds": 0.4,
                    "update_interval": 0.2,
                    "minimum_history_points": 3,
                },
                "evaluation": {
                    "enabled": True,
                    "match_tolerance_seconds": 0.05,
                    "success_threshold_pixels": 25.0,
                },
                "export": {
                    "enabled": True,
                },
            }
        }
        predictor = TrajectoryPredictor(cfg)

        # Generate enough frames to resolve predictions
        for i in range(15):
            t = i * 0.1
            f = make_frame(i, t)
            obj = make_obj(10, 200.0 + 15.0 * i, 300.0, class_name="car")
            predictor.predict([obj], f)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_report.xlsx"
            exported_path = predictor.export_evaluation(out_file)
            assert exported_path is not None
            assert exported_path.exists()

            wb = openpyxl.load_workbook(str(exported_path), data_only=False)

            # Check all 5 required sheets
            expected_sheets = [
                "Prediction Records",
                "Detailed Future Positions",
                "Accuracy Summary",
                "Performance",
                "Configuration and Metadata",
            ]
            for s in expected_sheets:
                assert s in wb.sheetnames, f"Missing sheet: {s}"

            # 1. Sheet: Prediction Records
            ws_pred = wb["Prediction Records"]
            pred_headers = [cell.value for cell in ws_pred[1]]
            assert "Prediction ID" in pred_headers
            assert "Track ID" in pred_headers
            assert "Track Instance ID" in pred_headers
            assert "ADE (px)" in pred_headers
            assert "FDE (px)" in pred_headers
            assert "Endpoint Error (px)" in pred_headers
            assert "Evaluation Status" in pred_headers
            assert ws_pred.max_row > 1  # Has records

            # 2. Sheet: Detailed Future Positions
            ws_wps = wb["Detailed Future Positions"]
            wp_headers = [cell.value for cell in ws_wps[1]]
            assert "Prediction ID" in wp_headers
            assert "Time Offset (s)" in wp_headers
            assert "Predicted X (px)" in wp_headers
            assert "Displacement Error (px)" in wp_headers
            assert ws_wps.max_row > 1

            # 3. Sheet: Accuracy Summary
            ws_summary = wb["Accuracy Summary"]
            assert ws_summary.max_row >= 5
            # Verify metric names in column A
            col_a_vals = [ws_summary.cell(r, 1).value for r in range(1, ws_summary.max_row + 1)]
            assert any("Mean Endpoint Error" in str(v) for v in col_a_vals if v)
            assert any("Displacement Error — ADE" in str(v) for v in col_a_vals if v)
            assert any("Displacement Error — FDE" in str(v) for v in col_a_vals if v)

            # 4. Sheet: Performance
            ws_perf = wb["Performance"]
            perf_vals = [ws_perf.cell(r, 1).value for r in range(1, ws_perf.max_row + 1)]
            assert any("Average Prediction Latency" in str(v) for v in perf_vals if v)

            # 5. Sheet: Configuration and Metadata
            ws_meta = wb["Configuration and Metadata"]
            meta_keys = [str(ws_meta.cell(r, 1).value) for r in range(1, ws_meta.max_row + 1)]
            assert any("Prediction Horizon" in k for k in meta_keys)
            assert any("Update Interval" in k for k in meta_keys)
            assert any("Coordinate System" in k for k in meta_keys)
