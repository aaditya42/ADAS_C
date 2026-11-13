from __future__ import annotations

from pathlib import Path
import pytest
import openpyxl

from src.schemas.trajectory import EvaluationStatus, PredictionRecord
from src.trajectory.drac import (
    DRACResult,
    associate_drac_pair,
    calculate_drac,
    cm_to_m,
    kmh_to_mps,
    mps_to_kmh,
)
from src.trajectory.excel_exporter import TrajectoryExcelExporter


def test_1_following_speed_greater_than_leading_speed() -> None:
    """1. Following speed greater than leading speed: DRAC = (vf - vl)^2 / (2 * d)."""
    # vf = 20 m/s, vl = 10 m/s, d = 25 m -> DRAC = (20-10)^2 / (2*25) = 100/50 = 2.0 m/s^2
    drac, status = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        gap_m=25.0,
        is_metric=True,
        require_metric=True,
    )
    assert status == "VALID"
    assert drac is not None
    assert pytest.approx(drac, 1e-4) == 2.0


def test_2_following_speed_equal_to_leading_speed() -> None:
    """2. Following speed equal to leading speed: non-closing case -> 0 m/s^2."""
    drac, status = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=20.0,
        gap_m=25.0,
        is_metric=True,
        require_metric=True,
    )
    assert status == "ZERO_CLOSING_SPEED"
    assert drac == 0.0


def test_3_following_speed_lower_than_leading_speed() -> None:
    """3. Following speed lower than leading speed: non-closing case -> 0 m/s^2."""
    drac, status = calculate_drac(
        following_speed_mps=10.0,
        leading_speed_mps=20.0,
        gap_m=25.0,
        is_metric=True,
        require_metric=True,
    )
    assert status == "ZERO_CLOSING_SPEED"
    assert drac == 0.0


def test_4_positive_inter_vehicle_gap() -> None:
    """4. Positive inter-vehicle gap: valid positive calculation."""
    drac, status = calculate_drac(
        following_speed_mps=15.0,
        leading_speed_mps=5.0,
        gap_m=10.0,
        is_metric=True,
        require_metric=True,
    )
    assert status == "VALID"
    # (15 - 5)^2 / (2 * 10) = 100 / 20 = 5.0 m/s^2
    assert drac is not None
    assert pytest.approx(drac, 1e-4) == 5.0


def test_5_zero_or_negative_gap() -> None:
    """5. Zero or negative gap: must return UNAVAILABLE."""
    # Zero gap
    drac_zero, status_zero = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        gap_m=0.0,
        is_metric=True,
        require_metric=True,
    )
    assert status_zero == "UNAVAILABLE"
    assert drac_zero is None

    # Negative gap
    drac_neg, status_neg = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        gap_m=-5.0,
        is_metric=True,
        require_metric=True,
    )
    assert status_neg == "UNAVAILABLE"
    assert drac_neg is None

    # Below minimum threshold (0.005 < 0.01)
    drac_min, status_min = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        gap_m=0.005,
        minimum_gap_meters=0.01,
        is_metric=True,
        require_metric=True,
    )
    assert status_min == "UNAVAILABLE"
    assert drac_min is None


def test_6_missing_speed() -> None:
    """6. Missing speed: follower or leader speed None -> UNAVAILABLE."""
    drac1, status1 = calculate_drac(
        following_speed_mps=None,
        leading_speed_mps=10.0,
        gap_m=25.0,
        is_metric=True,
        require_metric=True,
    )
    assert status1 == "UNAVAILABLE"
    assert drac1 is None

    drac2, status2 = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=None,
        gap_m=25.0,
        is_metric=True,
        require_metric=True,
    )
    assert status2 == "UNAVAILABLE"
    assert drac2 is None


def test_7_missing_distance() -> None:
    """7. Missing distance: gap None -> UNAVAILABLE."""
    drac, status = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        gap_m=None,
        is_metric=True,
        require_metric=True,
    )
    assert status == "UNAVAILABLE"
    assert drac is None


def test_8_invalid_or_relative_depth_only_distance() -> None:
    """8. Invalid or relative-depth-only distance: must mark UNAVAILABLE."""
    # When metric is required and is_metric is False (e.g. MiDaS relative depth)
    drac, status = calculate_drac(
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        gap_m=25.0,
        is_metric=False,
        require_metric=True,
    )
    assert status == "UNAVAILABLE"
    assert drac is None

    # associate_drac_pair with distance_type='relative'
    res = associate_drac_pair(
        target_track_id=1,
        target_class_name="car",
        target_distance=25.0,
        target_depth_confidence=1.0,
        all_objects=[],
        motion_dict={},
        distance_type="relative",
        drac_config={"enabled": True, "require_metric_units": True},
    )
    assert res.drac_status == "UNAVAILABLE"
    assert res.drac_mps2 is None


def test_9_speed_conversion_from_kmh_to_mps() -> None:
    """9. Speed conversion from km/h to m/s: 72 km/h -> 20 m/s, 36 km/h -> 10 m/s."""
    vf_mps = kmh_to_mps(72.0)
    vl_mps = kmh_to_mps(36.0)
    assert pytest.approx(vf_mps, 1e-4) == 20.0
    assert pytest.approx(vl_mps, 1e-4) == 10.0

    drac, status = calculate_drac(
        following_speed_mps=vf_mps,
        leading_speed_mps=vl_mps,
        gap_m=25.0,
        is_metric=True,
        require_metric=True,
    )
    assert status == "VALID"
    assert pytest.approx(drac, 1e-4) == 2.0


def test_10_correct_excel_formula_generation(tmp_path: Path) -> None:
    """10. Correct Excel formula generation and syntax."""
    out_xlsx = tmp_path / "drac_formula_test.xlsx"
    exporter = TrajectoryExcelExporter(output_path=str(out_xlsx))

    rec = PredictionRecord(
        prediction_id="p1",
        track_id=1,
        class_name="car",
        prediction_timestamp=10.0,
        current_position=(100.0, 200.0),
        horizon_seconds=2.0,
        target_timestamp=12.0,
        predicted_position=(110.0, 210.0),
        following_track_id="EGO",
        leading_track_id=1,
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        inter_vehicle_gap_m=25.0,
        drac_mps2=2.0,
        drac_status="VALID",
    )

    exporter.export([rec])
    wb = openpyxl.load_workbook(out_xlsx, data_only=False)
    ws = wb["Prediction Records"]

    # Check headers
    assert ws.cell(row=1, column=19).value == "Following Track ID"
    assert ws.cell(row=1, column=20).value == "Leading Track ID"
    assert ws.cell(row=1, column=21).value == "Following Speed (m/s)"
    assert ws.cell(row=1, column=22).value == "Leading Speed (m/s)"
    assert ws.cell(row=1, column=23).value == "Inter-Vehicle Gap (m)"
    assert ws.cell(row=1, column=24).value == "DRAC (m/s²)"
    assert ws.cell(row=1, column=25).value == "DRAC Status"

    # Row 2 values
    assert ws.cell(row=2, column=19).value == "EGO"
    assert ws.cell(row=2, column=20).value == 1
    assert ws.cell(row=2, column=21).value == 20.0
    assert ws.cell(row=2, column=22).value == 10.0
    assert ws.cell(row=2, column=23).value == 25.0

    # Formula in Column 24 (X2)
    formula = ws.cell(row=2, column=24).value
    expected_formula = '=IF(OR(U2="",V2="",W2="",W2<=0),"",IF(U2<=V2,0,(U2-V2)^2/(2*W2)))'
    assert formula == expected_formula
    assert ws.cell(row=2, column=25).value == "VALID"


def test_11_correct_formulas_across_multiple_rows(tmp_path: Path) -> None:
    """11. Correct formulas across multiple rows with varied conditions."""
    out_xlsx = tmp_path / "drac_multi_row_test.xlsx"
    exporter = TrajectoryExcelExporter(output_path=str(out_xlsx))

    records = [
        # Row 2: valid closing
        PredictionRecord(
            prediction_id="p1",
            track_id=1,
            class_name="car",
            prediction_timestamp=10.0,
            current_position=(100.0, 200.0),
            horizon_seconds=2.0,
            target_timestamp=12.0,
            predicted_position=(110.0, 210.0),
            following_track_id="EGO",
            leading_track_id=1,
            following_speed_mps=20.0,
            leading_speed_mps=10.0,
            inter_vehicle_gap_m=25.0,
            drac_mps2=2.0,
            drac_status="VALID",
        ),
        # Row 3: non-closing (vf < vl)
        PredictionRecord(
            prediction_id="p2",
            track_id=2,
            class_name="truck",
            prediction_timestamp=10.0,
            current_position=(200.0, 300.0),
            horizon_seconds=2.0,
            target_timestamp=12.0,
            predicted_position=(210.0, 310.0),
            following_track_id="EGO",
            leading_track_id=2,
            following_speed_mps=15.0,
            leading_speed_mps=25.0,
            inter_vehicle_gap_m=30.0,
            drac_mps2=0.0,
            drac_status="ZERO_CLOSING_SPEED",
        ),
        # Row 4: unavailable (uncalibrated distance / missing gap)
        PredictionRecord(
            prediction_id="p3",
            track_id=3,
            class_name="car",
            prediction_timestamp=10.0,
            current_position=(300.0, 400.0),
            horizon_seconds=2.0,
            target_timestamp=12.0,
            predicted_position=(310.0, 410.0),
            following_track_id=None,
            leading_track_id=None,
            following_speed_mps=None,
            leading_speed_mps=None,
            inter_vehicle_gap_m=None,
            drac_mps2=None,
            drac_status="UNAVAILABLE",
        ),
    ]

    exporter.export(records)
    wb = openpyxl.load_workbook(out_xlsx, data_only=False)
    ws = wb["Prediction Records"]

    assert ws.cell(row=2, column=24).value == '=IF(OR(U2="",V2="",W2="",W2<=0),"",IF(U2<=V2,0,(U2-V2)^2/(2*W2)))'
    assert ws.cell(row=3, column=24).value == '=IF(OR(U3="",V3="",W3="",W3<=0),"",IF(U3<=V3,0,(U3-V3)^2/(2*W3)))'
    assert ws.cell(row=4, column=24).value == '=IF(OR(U4="",V4="",W4="",W4<=0),"",IF(U4<=V4,0,(U4-V4)^2/(2*W4)))'

    # Check status column values
    assert ws.cell(row=2, column=25).value == "VALID"
    assert ws.cell(row=3, column=25).value == "ZERO_CLOSING_SPEED"
    assert ws.cell(row=4, column=25).value == "UNAVAILABLE"


def test_12_preservation_of_all_previous_sheets_and_columns(tmp_path: Path) -> None:
    """12. Preservation of all previous workbook sheets and columns."""
    out_xlsx = tmp_path / "drac_preservation_test.xlsx"
    exporter = TrajectoryExcelExporter(output_path=str(out_xlsx))

    rec = PredictionRecord(
        prediction_id="p1",
        track_id=1,
        class_name="car",
        prediction_timestamp=1.0,
        current_position=(100.0, 100.0),
        horizon_seconds=2.0,
        target_timestamp=3.0,
        predicted_position=(120.0, 120.0),
        actual_position=(125.0, 122.0),
        actual_timestamp=3.0,
        endpoint_error=5.385,
        ade=3.2,
        fde=5.385,
        status=EvaluationStatus.MATCHED,
        following_track_id="EGO",
        leading_track_id=1,
        following_speed_mps=20.0,
        leading_speed_mps=10.0,
        inter_vehicle_gap_m=25.0,
        drac_mps2=2.0,
        drac_status="VALID",
    )

    exporter.export([rec])
    wb = openpyxl.load_workbook(out_xlsx, data_only=False)

    sheet_names = wb.sheetnames
    expected_sheets = [
        "Prediction Records",
        "Detailed Future Positions",
        "Accuracy Summary",
        "Performance",
        "Configuration and Metadata",
    ]
    for s in expected_sheets:
        assert s in sheet_names, f"Expected sheet '{s}' missing from workbook"

    ws1 = wb["Prediction Records"]
    # Check original first 18 columns
    original_18 = [
        "Prediction ID",
        "Track Instance ID",
        "Track ID",
        "Object Class",
        "Prediction Timestamp (s)",
        "Current X (px)",
        "Current Y (px)",
        "Horizon (s)",
        "Target Timestamp (s)",
        "Predicted X (px)",
        "Predicted Y (px)",
        "Actual X (px)",
        "Actual Y (px)",
        "Endpoint Error (px)",
        "ADE (px)",
        "FDE (px)",
        "Evaluation Status",
        "Prediction Model",
    ]
    for i, col_name in enumerate(original_18, start=1):
        assert ws1.cell(row=1, column=i).value == col_name

    # Check 7 DRAC appended columns
    drac_7 = [
        "Following Track ID",
        "Leading Track ID",
        "Following Speed (m/s)",
        "Leading Speed (m/s)",
        "Inter-Vehicle Gap (m)",
        "DRAC (m/s²)",
        "DRAC Status",
    ]
    for i, col_name in enumerate(drac_7, start=19):
        assert ws1.cell(row=1, column=i).value == col_name

    # Check total column count
    assert ws1.max_column == 25

    # Check Sheet 5 contains DRAC metadata
    ws5 = wb["Configuration and Metadata"]
    param_cells = [ws5.cell(row=r, column=1).value for r in range(4, ws5.max_row + 1)]
    assert "DRAC Evaluation Enabled" in param_cells
    assert "DRAC Metric Calibration Required" in param_cells
    assert "DRAC Minimum Gap Threshold (m)" in param_cells

    # Check calcMode settings
    assert wb.calculation.fullCalcOnLoad is True
    assert wb.calculation.calcMode == "auto"
