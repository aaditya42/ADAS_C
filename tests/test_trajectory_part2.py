from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from src.schemas import BoundingBox, Frame, TrackedObject
from src.schemas.trajectory import (
    EvaluationStatus,
    PredictionRecord,
    TrajectoryPoint,
    TrajectoryPrediction,
    TrajectoryWaypointRecord,
)
from src.trajectory.excel_exporter import TrajectoryExcelExporter
from src.trajectory.history_manager import TrajectoryHistoryManager
from src.trajectory.metrics import calculate_trajectory_metrics
from src.trajectory.prediction_manager import PredictionManager
from src.trajectory.trajectory_predictor import TrajectoryPredictor
from src.visualization.layers.trajectory_layer import TrajectoryLayer


def make_frame(frame_id: int, timestamp: float) -> Frame:
    return Frame(
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=timestamp,
        source="test_cam",
    )


def make_box_obj(track_id: int, x1: float, y1: float, x2: float, y2: float, class_name: str = "car") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(x1, y1, x2, y2),
    )


# ===================================================================== #
# 1. Endpoint Error Calculation
# ===================================================================== #

def test_endpoint_error_calculation() -> None:
    rec = PredictionRecord(
        prediction_id="p1",
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        current_position=(100.0, 100.0),
        horizon_seconds=2.0,
        target_timestamp=12.0,
        predicted_position=(140.0, 130.0),
        actual_position=(143.0, 134.0),
        endpoint_error=math.hypot(140.0 - 143.0, 130.0 - 134.0),
        status=EvaluationStatus.MATCHED,
    )
    # sqrt(3^2 + 4^2) = 5.0
    assert rec.endpoint_error == 5.0


# ===================================================================== #
# 2. ADE and FDE Calculation
# ===================================================================== #

def test_ade_and_fde_calculation() -> None:
    hist = TrajectoryHistoryManager()
    pm = PredictionManager(match_tolerance_seconds=0.05)

    # Waypoints at t=10.1 (offset 0.1) and t=10.2 (offset 0.2)
    wps = [
        TrajectoryPoint(future_timestamp=10.1, time_offset=0.1, position=(110.0, 110.0)),
        TrajectoryPoint(future_timestamp=10.2, time_offset=0.2, position=(120.0, 120.0)),
    ]

    rec = pm.add_prediction(
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        horizon_seconds=0.2,
        current_position=(100.0, 100.0),
        predicted_position=(120.0, 120.0),
        waypoints=wps,
    )

    # Frame 1: t=10.1, actual pos=(113, 114) -> error = 5.0
    f1 = make_frame(1, 10.1)
    # bottom-center: x=(103+123)/2=113, y2=114
    hist.update([make_box_obj(1, 103.0, 90.0, 123.0, 114.0)], f1)
    pm.evaluate_pending(hist, current_timestamp=10.1)

    # Frame 2: t=10.2, actual pos=(126, 128) -> error = 10.0
    f2 = make_frame(2, 10.2)
    # bottom-center: x=(116+136)/2=126, y2=128
    hist.update([make_box_obj(1, 116.0, 100.0, 136.0, 128.0)], f2)
    pm.evaluate_pending(hist, current_timestamp=10.2)

    assert rec.status == EvaluationStatus.MATCHED
    # FDE = final endpoint error = 10.0
    assert abs(rec.fde - 10.0) < 1e-3
    assert abs(rec.endpoint_error - 10.0) < 1e-3
    # ADE = mean(5.0, 10.0) = 7.5
    assert abs(rec.ade - 7.5) < 1e-3


# ===================================================================== #
# 3. Matching by Target Timestamp
# ===================================================================== #

def test_matching_by_target_timestamp() -> None:
    hist = TrajectoryHistoryManager()
    pm = PredictionManager(match_tolerance_seconds=0.05)

    rec = pm.add_prediction(
        track_id=1,
        class_name="pedestrian",
        prediction_timestamp=10.0,
        horizon_seconds=1.0,
        current_position=(50.0, 50.0),
        predicted_position=(60.0, 60.0),
    )
    assert rec.target_timestamp == 11.0

    # At t=10.8 -> target not yet reached
    hist.update([make_box_obj(1, 40.0, 40.0, 60.0, 55.0)], make_frame(1, 10.8))
    pm.evaluate_pending(hist, current_timestamp=10.8)
    assert rec.status == EvaluationStatus.PENDING

    # At t=11.0 -> target reached
    hist.update([make_box_obj(1, 45.0, 45.0, 75.0, 62.0)], make_frame(2, 11.0))
    resolved = pm.evaluate_pending(hist, current_timestamp=11.0)
    assert len(resolved) == 1
    assert rec.status == EvaluationStatus.MATCHED
    assert rec.actual_position == (60.0, 62.0)


# ===================================================================== #
# 4. Matching Within Configurable Tolerance
# ===================================================================== #

def test_matching_within_tolerance() -> None:
    hist = TrajectoryHistoryManager()
    # Tolerance 0.05s
    pm = PredictionManager(match_tolerance_seconds=0.05)

    rec = pm.add_prediction(
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        horizon_seconds=1.0,
        current_position=(10.0, 10.0),
        predicted_position=(20.0, 20.0),
    )

    # Observation arrived at 11.03s (delta 0.03 <= 0.05s tolerance)
    hist.update([make_box_obj(1, 10.0, 10.0, 30.0, 22.0)], make_frame(1, 11.03))
    pm.evaluate_pending(hist, current_timestamp=11.03)
    assert rec.status == EvaluationStatus.MATCHED
    assert rec.actual_position == (20.0, 22.0)


# ===================================================================== #
# 5. Rejection of Observations Outside Tolerance
# ===================================================================== #

def test_rejection_outside_tolerance() -> None:
    hist = TrajectoryHistoryManager()
    pm = PredictionManager(match_tolerance_seconds=0.02)  # Strict 0.02s

    rec = pm.add_prediction(
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        horizon_seconds=1.0,
        current_position=(10.0, 10.0),
        predicted_position=(20.0, 20.0),
    )

    # Observation at 11.08s (delta 0.08 > 0.02s tolerance)
    hist.update([make_box_obj(1, 10.0, 10.0, 30.0, 22.0)], make_frame(1, 11.08))
    pm.evaluate_pending(hist, current_timestamp=11.08)
    # Outside tolerance without closer observation -> MISSING
    assert rec.status == EvaluationStatus.MISSING


# ===================================================================== #
# 6. Handling of Missing Future Observations
# ===================================================================== #

def test_handling_missing_observations() -> None:
    hist = TrajectoryHistoryManager(maximum_track_gap_seconds=0.5, stale_track_timeout_seconds=2.0)
    pm = PredictionManager(match_tolerance_seconds=0.05)

    rec = pm.add_prediction(
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        horizon_seconds=1.0,
        current_position=(10.0, 10.0),
        predicted_position=(20.0, 20.0),
        track_instance_id=hist.get_track_instance_id(1),
    )

    # Observation at 10.5, then tracking gap of 1.0s (next observation at 11.5)
    # Target timestamp 11.0 has no continuous observation -> INVALID (due to gap)
    hist.update([make_box_obj(1, 10.0, 10.0, 30.0, 22.0)], make_frame(1, 10.5))
    hist.update([make_box_obj(1, 10.0, 10.0, 30.0, 22.0)], make_frame(2, 11.5))
    pm.evaluate_pending(hist, current_timestamp=11.5)

    assert rec.status in (EvaluationStatus.MISSING, EvaluationStatus.TRACK_LOST, EvaluationStatus.INVALID)
    assert rec.status == EvaluationStatus.INVALID




# ===================================================================== #
# 7. Track ID Reuse and Track-Instance Separation
# ===================================================================== #

def test_track_instance_separation_on_gap() -> None:
    hist = TrajectoryHistoryManager(maximum_track_gap_seconds=0.5)
    pm = PredictionManager(match_tolerance_seconds=0.05)

    # Initial episode for track 1
    hist.update([make_box_obj(1, 0.0, 0.0, 20.0, 20.0)], make_frame(1, 10.0))
    inst_1 = hist.get_track_instance_id(1)

    rec = pm.add_prediction(
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        horizon_seconds=1.0,
        current_position=(10.0, 20.0),
        predicted_position=(15.0, 25.0),
        track_instance_id=inst_1,
    )

    # Large gap of 2.0s (> 0.5s) triggers new instance (ID reused by another vehicle)
    hist.update([make_box_obj(1, 400.0, 400.0, 420.0, 420.0)], make_frame(2, 12.0))
    inst_2 = hist.get_track_instance_id(1)
    assert inst_1 != inst_2

    # Querying position for inst_1 at target 11.0 returns None, avoiding false match against inst_2!
    pos_match = hist.get_position_at_timestamp(1, 11.0, 0.05, track_instance_id=inst_1)
    assert pos_match is None


# ===================================================================== #
# 8. Interpolation Across Valid Observations
# ===================================================================== #

def test_interpolation_across_valid_observations() -> None:
    hist = TrajectoryHistoryManager(maximum_track_gap_seconds=0.5)

    # Observations at t=10.0 (pos 100, 200) and t=10.2 (pos 120, 220)
    hist.update([make_box_obj(1, 90.0, 180.0, 110.0, 200.0)], make_frame(1, 10.0))
    hist.update([make_box_obj(1, 110.0, 200.0, 130.0, 220.0)], make_frame(2, 10.2))

    # Interpolate at target t=10.1 (midpoint)
    match_res = hist.get_position_at_timestamp(1, 10.1, match_tolerance_seconds=0.05)
    assert match_res is not None
    interp_x, interp_y, interp_t = match_res
    assert abs(interp_x - 110.0) < 1e-3
    assert abs(interp_y - 210.0) < 1e-3
    assert interp_t == 10.1


# ===================================================================== #
# 9. Rejection of Interpolation Across Tracking Gaps
# ===================================================================== #

def test_rejection_of_interpolation_across_gap() -> None:
    hist = TrajectoryHistoryManager(maximum_track_gap_seconds=0.5)

    # Observation at t=10.0 and next at t=11.0 (gap = 1.0s > 0.5s)
    hist.update([make_box_obj(1, 0.0, 0.0, 20.0, 20.0)], make_frame(1, 10.0))
    hist.update([make_box_obj(1, 100.0, 100.0, 120.0, 120.0)], make_frame(2, 11.0))

    # Attempt interpolation at t=10.5 across the gap -> must reject
    match_res = hist.get_position_at_timestamp(1, 10.5, match_tolerance_seconds=0.05)
    assert match_res is None


# ===================================================================== #
# 10. Excel Export with Matched and Unmatched Records
# ===================================================================== #

def test_excel_export_with_records(tmp_path: Path) -> None:
    out_file = tmp_path / "test_eval.xlsx"
    exporter = TrajectoryExcelExporter(output_path=str(out_file), success_threshold_pixels=20.0)

    r1 = PredictionRecord(
        prediction_id="rec-001",
        track_id=1,
        track_instance_id="1_inst1",
        class_name="car",
        prediction_timestamp=10.0,
        current_position=(100.0, 200.0),
        horizon_seconds=2.0,
        target_timestamp=12.0,
        predicted_position=(150.0, 250.0),
        actual_position=(152.0, 248.0),
        actual_timestamp=12.0,
        endpoint_error=math.hypot(2.0, -2.0),
        ade=2.5,
        fde=2.83,
        status=EvaluationStatus.MATCHED,
        waypoint_records=[
            TrajectoryWaypointRecord(11.0, 1.0, (125.0, 225.0), actual_position=(124.0, 224.0), displacement_error=1.41, status=EvaluationStatus.MATCHED),
            TrajectoryWaypointRecord(12.0, 2.0, (150.0, 250.0), actual_position=(152.0, 248.0), displacement_error=2.83, status=EvaluationStatus.MATCHED),
        ],
    )
    r2 = PredictionRecord(
        prediction_id="rec-002",
        track_id=2,
        track_instance_id="2_inst1",
        class_name="pedestrian",
        prediction_timestamp=10.0,
        current_position=(50.0, 50.0),
        horizon_seconds=1.5,
        target_timestamp=11.5,
        predicted_position=(70.0, 70.0),
        status=EvaluationStatus.TRACK_LOST,
    )

    path = exporter.export([r1, r2])
    assert path.is_file()

    import openpyxl
    wb = openpyxl.load_workbook(str(path))
    assert "Prediction Records" in wb.sheetnames
    assert "Detailed Future Positions" in wb.sheetnames
    assert "Accuracy Summary" in wb.sheetnames
    assert "Performance" in wb.sheetnames
    assert "Configuration and Metadata" in wb.sheetnames
    wb.close()


# ===================================================================== #
# 11. Excel Export with Empty Dataset
# ===================================================================== #

def test_excel_export_empty_dataset(tmp_path: Path) -> None:
    out_file = tmp_path / "test_empty.xlsx"
    exporter = TrajectoryExcelExporter(output_path=str(out_file))
    path = exporter.export([])
    assert path.is_file()

    import openpyxl
    wb = openpyxl.load_workbook(str(path))
    assert len(wb.sheetnames) == 5
    wb.close()


# ===================================================================== #
# 12. Configurable Prediction Horizon and Update Interval
# ===================================================================== #

def test_configurable_horizon_and_interval() -> None:
    cfg = {
        "trajectory": {
            "prediction": {
                "horizon_seconds": 2.5,
                "update_interval": 0.25,
            }
        }
    }
    predictor = TrajectoryPredictor(cfg)
    assert predictor.horizon_seconds == 2.5
    assert predictor.update_interval == 0.25
    assert len(predictor.offsets) == 10
    assert predictor.offsets[-1] == 2.5


# ===================================================================== #
# 13. Visualization Layer with Multiple Simultaneous Tracked Objects
# ===================================================================== #

def test_visualization_multiple_tracks() -> None:
    layer = TrajectoryLayer(
        enabled=True,
        show_predicted_path=True,
        show_predicted_endpoint=True,
        show_actual_endpoint=True,
        show_error_line=True,
        show_error_label=True,
    )

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    pred1 = TrajectoryPrediction(
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        horizon_seconds=2.0,
        current_position=(100.0, 100.0),
        estimated_velocity=(10.0, 0.0),
        estimated_acceleration=(0.0, 0.0),
        predicted_positions=[
            TrajectoryPoint(11.0, 1.0, (110.0, 100.0)),
            TrajectoryPoint(12.0, 2.0, (120.0, 100.0)),
        ],
        final_predicted_position=(120.0, 100.0),
    )
    pred2 = TrajectoryPrediction(
        track_id=2,
        class_name="truck",
        prediction_timestamp=10.0,
        horizon_seconds=2.0,
        current_position=(300.0, 300.0),
        estimated_velocity=(0.0, 15.0),
        estimated_acceleration=(0.0, 0.0),
        predicted_positions=[
            TrajectoryPoint(11.0, 1.0, (300.0, 315.0)),
            TrajectoryPoint(12.0, 2.0, (300.0, 330.0)),
        ],
        final_predicted_position=(300.0, 330.0),
    )

    ctx = {"trajectories": [pred1, pred2], "trajectory_evaluations": []}
    out_canvas = layer(canvas, ctx)
    assert out_canvas.shape == (720, 1280, 3)
    # The canvas should have drawn annotations (non-zero pixels)
    assert np.count_nonzero(out_canvas) > 0


# ===================================================================== #
# 14. Distinction Between Predicted and Actual Endpoints in Visualization
# ===================================================================== #

def test_visualization_error_overlay() -> None:
    layer = TrajectoryLayer(
        enabled=True,
        show_actual_endpoint=True,
        show_error_line=True,
        show_error_label=True,
    )
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    rec = PredictionRecord(
        prediction_id="rec-01",
        track_id=5,
        class_name="car",
        prediction_timestamp=10.0,
        current_position=(200.0, 200.0),
        horizon_seconds=1.0,
        target_timestamp=11.0,
        predicted_position=(250.0, 250.0),
        actual_position=(255.0, 252.0),
        actual_timestamp=11.0,
        endpoint_error=math.hypot(5.0, 2.0),
        status=EvaluationStatus.MATCHED,
    )

    ctx = {"trajectories": [], "trajectory_evaluations": [rec]}
    out = layer(canvas, ctx)
    assert np.count_nonzero(out) > 0
