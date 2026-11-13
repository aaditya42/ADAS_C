from __future__ import annotations

from pathlib import Path
import numpy as np
import openpyxl
import pytest

from src.overtaking.excel_exporter import OvertakingExcelExporter
from src.overtaking.schemas import (
    CorridorPosition,
    IdentityContinuityStatus,
    OvertakingCycleResult,
    OvertakingEvent,
    OvertakingEventStatus,
    OvertakingState,
)
from src.schemas import BoundingBox, Frame
from src.visualization.layers.overtaking_layer import OvertakingLayer


def make_test_event(
    event_id: str,
    track_id: int,
    status: OvertakingEventStatus,
    state: OvertakingState,
    start_t: float | None = 1.0,
    end_t: float | None = 3.5,
    passing_side: str = "LEFT",
) -> OvertakingEvent:
    ev = OvertakingEvent(
        event_id=event_id,
        track_id=track_id,
        track_instance_id=f"{track_id}_inst1",
        vehicle_class="car",
        state=state,
        status=status,
        passing_side=passing_side,
        candidate_start_timestamp=start_t,
        start_timestamp=start_t,
        start_confirmation_timestamp=start_t + 0.25 if start_t else None,
        start_confirmation_delay_seconds=0.25 if start_t else None,
        initial_position=(600.0, 300.0),
        latest_position=(200.0, 500.0),
        initial_bbox=BoundingBox(570.0, 270.0, 630.0, 330.0),
        latest_bbox=BoundingBox(150.0, 450.0, 250.0, 550.0),
        initial_depth=25.0,
        latest_depth=12.0,
        initial_corridor_position=CorridorPosition.IN_EGO_CORRIDOR,
        latest_corridor_position=CorridorPosition.LEFT_OF_CORRIDOR,
        valid_observation_count=20,
    )
    if status == OvertakingEventStatus.COMPLETED and end_t is not None:
        ev.finalize_completion(end_time=end_t, confirmation_time=end_t + 0.40)
    elif status == OvertakingEventStatus.INCOMPLETE:
        ev.finalize_incomplete("TRACK_LOST_IN_MID_FIELD")
    elif status == OvertakingEventStatus.UNCERTAIN:
        ev.identity_continuity_status = IdentityContinuityStatus.SUSPECTED_ID_SWITCH
        ev.finalize_incomplete("TRACK_LOST_AMBIGUOUS", status=OvertakingEventStatus.UNCERTAIN)
    return ev


# ===================================================================== #
# Visualization Tests
# ===================================================================== #

def test_overtaking_layer_disabled_returns_unchanged_canvas() -> None:
    layer = OvertakingLayer(enabled=False)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = Frame(image=canvas, frame_id=1, timestamp=1.5, source="test")

    ev = make_test_event("evt1", 1, OvertakingEventStatus.IN_PROGRESS, OvertakingState.PASSING)
    ctx = {
        "frame": frame,
        "overtaking": OvertakingCycleResult(active_events=[ev], newly_started=[], newly_completed=[], newly_incomplete=[]),
    }

    out = layer(canvas, ctx)
    # Output must be identical to input when layer is disabled
    assert np.array_equal(out, canvas)


def test_overtaking_layer_renders_active_event_and_target_markers() -> None:
    layer = OvertakingLayer(enabled=True, show_target_marker=True, show_elapsed_time=True)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    original = canvas.copy()
    frame = Frame(image=canvas, frame_id=10, timestamp=2.5, source="test")

    ev = make_test_event("evt1", 4, OvertakingEventStatus.IN_PROGRESS, OvertakingState.PASSING, start_t=1.0)
    ctx = {
        "frame": frame,
        "overtaking": OvertakingCycleResult(active_events=[ev], newly_started=[], newly_completed=[], newly_incomplete=[]),
    }

    out = layer(canvas, ctx)
    assert not np.array_equal(out, original)
    # Check that pixels were drawn (non-zero)
    assert np.count_nonzero(out) > 0


def test_overtaking_layer_displays_completion_notification_and_expires() -> None:
    layer = OvertakingLayer(enabled=True, show_completion_notification=True, notification_duration=1.0)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Frame 1: Event completes at t=3.0
    frame1 = Frame(image=canvas, frame_id=30, timestamp=3.0, source="test")
    ev = make_test_event("evt2", 7, OvertakingEventStatus.COMPLETED, OvertakingState.OVERTAKING_COMPLETED, start_t=1.0, end_t=3.0)
    ctx1 = {
        "frame": frame1,
        "overtaking": OvertakingCycleResult(active_events=[], newly_started=[], newly_completed=[ev], newly_incomplete=[]),
    }
    out1 = layer(canvas.copy(), ctx1)
    assert len(layer._notification_buffer) == 1
    assert np.count_nonzero(out1) > 0

    # Frame 2: Current time t=3.5 (still within 1.0s duration) -> still in buffer
    frame2 = Frame(image=canvas, frame_id=35, timestamp=3.5, source="test")
    ctx2 = {
        "frame": frame2,
        "overtaking": OvertakingCycleResult(active_events=[], newly_started=[], newly_completed=[], newly_incomplete=[]),
    }
    out2 = layer(canvas.copy(), ctx2)
    assert len(layer._notification_buffer) == 1

    # Frame 3: Current time t=4.2 (past 1.0s duration) -> notification expired and purged
    frame3 = Frame(image=canvas, frame_id=42, timestamp=4.2, source="test")
    ctx3 = {
        "frame": frame3,
        "overtaking": OvertakingCycleResult(active_events=[], newly_started=[], newly_completed=[], newly_incomplete=[]),
    }
    out3 = layer(canvas.copy(), ctx3)
    assert len(layer._notification_buffer) == 0
    assert np.array_equal(out3, canvas)


# ===================================================================== #
# Excel Reporting Tests
# ===================================================================== #

def test_excel_export_creates_four_valid_sheets(tmp_path: Path) -> None:
    exporter = OvertakingExcelExporter(output_dir=str(tmp_path), filename="test_overtaking.xlsx")

    ev_completed = make_test_event("evt_comp", 1, OvertakingEventStatus.COMPLETED, OvertakingState.OVERTAKING_COMPLETED, start_t=1.0, end_t=3.5)
    ev_incomplete = make_test_event("evt_inc", 2, OvertakingEventStatus.INCOMPLETE, OvertakingState.INCOMPLETE, start_t=4.0, end_t=None)
    ev_uncertain = make_test_event("evt_unc", 3, OvertakingEventStatus.UNCERTAIN, OvertakingState.INCOMPLETE, start_t=6.0, end_t=None)

    metadata = {
        "video_path": "data/test1.mp4",
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "frame_count": 300,
        "duration_seconds": 10.0,
    }

    out_file = exporter.export([ev_completed, ev_incomplete, ev_uncertain], metadata=metadata, config={"overtaking": {}})
    assert out_file.exists()

    wb = openpyxl.load_workbook(out_file)
    assert "Overtaking Events" in wb.sheetnames
    assert "Summary" in wb.sheetnames
    assert "Event Diagnostics" in wb.sheetnames
    assert "Configuration & Metadata" in wb.sheetnames

    # Check Sheet 1: Overtaking Events
    ws1 = wb["Overtaking Events"]
    assert ws1.cell(row=1, column=1).value == "Event ID"
    assert ws1.cell(row=1, column=7).value == "Overtaking Duration (s)"
    assert ws1.max_row == 4  # Header + 3 events

    # Row 2 (completed): Duration must be 2.500
    assert ws1.cell(row=2, column=1).value == "evt_comp"
    assert ws1.cell(row=2, column=7).value == 2.5
    assert ws1.cell(row=2, column=24).value == "COMPLETED"

    # Row 3 (incomplete): Duration and End Timestamp must be None (blank)
    assert ws1.cell(row=3, column=1).value == "evt_inc"
    assert ws1.cell(row=3, column=6).value is None
    assert ws1.cell(row=3, column=7).value is None
    assert ws1.cell(row=3, column=24).value == "INCOMPLETE"

    # Check Sheet 2: Summary metrics
    ws2 = wb["Summary"]
    summary_dict = {}
    for r in range(2, ws2.max_row + 1):
        metric = ws2.cell(row=r, column=1).value
        val = ws2.cell(row=r, column=2).value
        if metric:
            summary_dict[metric] = val

    assert summary_dict.get("Total Candidate Vehicles Ahead") == 3
    assert summary_dict.get("Completed Overtaking Events") == 1
    assert summary_dict.get("Incomplete Events") == 1
    assert summary_dict.get("Uncertain Events") == 1
    assert summary_dict.get("Mean Duration (s)") == "2.500"

    # Check Sheet 3: Event Diagnostics
    ws3 = wb["Event Diagnostics"]
    assert ws3.max_row == 4
    assert ws3.cell(row=2, column=9).value == "COMPLETED"
    assert ws3.cell(row=3, column=9).value == "INCOMPLETE"

    # Check Sheet 4: Configuration & Metadata
    ws4 = wb["Configuration & Metadata"]
    assert ws4.max_row > 10


def test_excel_export_empty_event_list_handles_gracefully(tmp_path: Path) -> None:
    """Zero events should produce valid summary with N/A rather than crashing."""
    exporter = OvertakingExcelExporter(output_dir=str(tmp_path), filename="empty_test.xlsx")
    out_file = exporter.export([], metadata={"fps": 30.0}, config={})
    assert out_file.exists()

    wb = openpyxl.load_workbook(out_file)
    ws2 = wb["Summary"]

    summary_dict = {
        ws2.cell(row=r, column=1).value: ws2.cell(row=r, column=2).value
        for r in range(2, ws2.max_row + 1)
        if ws2.cell(row=r, column=1).value
    }

    assert summary_dict.get("Total Candidate Vehicles Ahead") == 0
    assert summary_dict.get("Completed Overtaking Events") == 0
    assert summary_dict.get("Completion Rate (%)") == "N/A"
    assert summary_dict.get("Mean Duration (s)") == "N/A"


def test_excel_export_does_not_overwrite_existing_report(tmp_path: Path) -> None:
    exporter = OvertakingExcelExporter(output_dir=str(tmp_path), filename="report.xlsx")
    file1 = exporter.export([], metadata={}, config={})
    assert file1.exists()

    file2 = exporter.export([], metadata={}, config={})
    assert file2.exists()
    # File 2 must have a distinct non-overwriting timestamped filename
    assert file1 != file2
