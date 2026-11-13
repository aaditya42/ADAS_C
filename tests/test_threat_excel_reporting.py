"""Unit tests for ThreatExcelExporter and Excel workbook generation."""

import tempfile
from pathlib import Path
import openpyxl
import pytest

from src.threat.excel_exporter import ThreatExcelExporter
from src.threat.schemas import (
    CameraDirection,
    DirectionalWarningLevel,
    WarningEvent,
    WarningEventState,
)


class TestThreatExcelReporting:
    """Verifies Excel exporter behavior, sheet structure, and styling integrity."""

    @pytest.fixture
    def sample_events(self) -> list[WarningEvent]:
        ev1 = WarningEvent(
            event_id="warn_car_following_1_0001",
            track_id=1,
            object_class="car",
            scenario="CAR_FOLLOWING",
            direction=CameraDirection.FRONT,
            current_level=DirectionalWarningLevel.HIGH,
            previous_level=DirectionalWarningLevel.MEDIUM,
            state=WarningEventState.ACTIVE_HIGH,
            first_detected_timestamp=10.500,
            confirmation_timestamp=10.600,
            activation_timestamp=10.600,
            last_updated_timestamp=12.200,
            risk_score=0.825,
            risk_model="OPTION_A_ADRS",
            reliability="HIGH",
            ttc=1.45,
            collision_urgency=True,
            priority_rank=1,
            escalation_timestamps=[(11.000, "MEDIUM", "HIGH")],
        )
        ev2 = WarningEvent(
            event_id="warn_cut_in_2_0002",
            track_id=2,
            object_class="truck",
            scenario="CUT_IN",
            direction=CameraDirection.LEFT,
            current_level=DirectionalWarningLevel.LOW,
            previous_level=DirectionalWarningLevel.UNASSESSED,
            state=WarningEventState.RESOLVED,
            first_detected_timestamp=15.000,
            confirmation_timestamp=15.200,
            activation_timestamp=15.200,
            resolution_timestamp=17.500,
            last_updated_timestamp=17.500,
            risk_score=0.420,
            risk_model="OPTION_B_LINEAR",
            reliability="MEDIUM",
            ttc=4.10,
            collision_urgency=False,
            priority_rank=2,
            resolution_reason="THREAT_SUBSIDED_BELOW_ADVISORY",
        )
        return [ev1, ev2]

    @pytest.fixture
    def sample_history(self) -> list[dict]:
        return [
            {
                "timestamp": 10.600,
                "event_id": "warn_car_following_1_0001",
                "track_id": 1,
                "direction": "CENTER",
                "scenario": "CAR_FOLLOWING",
                "transition_type": "ACTIVATION",
                "previous_level": "UNASSESSED",
                "new_level": "MEDIUM",
                "event_state": "ACTIVE_MEDIUM",
                "risk_score": 0.650,
                "risk_model": "OPTION_A_ADRS",
                "reliability": "HIGH",
                "ttc": 2.10,
                "triggering_evidence": "Developing potential conflict",
            },
            {
                "timestamp": 11.000,
                "event_id": "warn_car_following_1_0001",
                "track_id": 1,
                "direction": "CENTER",
                "scenario": "CAR_FOLLOWING",
                "transition_type": "ESCALATION",
                "previous_level": "MEDIUM",
                "new_level": "HIGH",
                "event_state": "ACTIVE_HIGH",
                "risk_score": 0.825,
                "risk_model": "OPTION_A_ADRS",
                "reliability": "HIGH",
                "ttc": 1.45,
                "triggering_evidence": "Urgent TTC=1.45s",
            },
        ]

    @pytest.fixture
    def sample_details(self) -> list[dict]:
        return [
            {
                "timestamp": 11.000,
                "track_id": 1,
                "object_class": "car",
                "scenario": "CAR_FOLLOWING",
                "risk_model": "OPTION_A_ADRS",
                "model_status": "SUCCESS",
                "reliability": "HIGH",
                "raw_ttc": 1.45,
                "norm_ttc": 0.793,
                "raw_pet": 3.0,
                "norm_pet": 0.500,
                "raw_drac": 3.6,
                "norm_drac": 0.600,
                "raw_dist": 20.0,
                "norm_dist": 0.600,
                "raw_rel_speed": 6.5,
                "norm_rel_speed": 0.433,
                "raw_encroachment": 0.70,
                "norm_encroachment": 0.700,
                "published_coefficients": "ttc:0.25, pet:0.24, drac:0.21, gap:0.16, speed_diff:0.14",
                "weighted_contributions": "ttc:0.198, pet:0.120, drac:0.126, gap:0.096, speed_diff:0.061",
                "risk_score": 0.825,
                "priority_rank": 1,
            }
        ]

    def test_workbook_creation_and_all_five_sheets(
        self, sample_events, sample_history, sample_details
    ):
        """Exporter must produce exactly the 5 specified sheets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            exporter = ThreatExcelExporter(output_dir=tmp_dir, filename="test_threat.xlsx")
            file_path = exporter.export(
                events=sample_events,
                history=sample_history,
                risk_details=sample_details,
                suppressed_duplicates=15,
                metadata={"duration_s": 38.87, "frame_count": 1165, "fps": 29.97},
            )

            assert file_path.exists()
            wb = openpyxl.load_workbook(file_path)

            expected_sheets = [
                "Threat Events",
                "Risk Assessment Details",
                "Warning History",
                "Summary & Statistics",
                "Configuration & Metadata",
            ]
            assert wb.sheetnames == expected_sheets

            # Verify Threat Events sheet data
            ws1 = wb["Threat Events"]
            assert ws1.cell(row=1, column=1).value == "Event ID"
            assert ws1.cell(row=2, column=1).value == "warn_car_following_1_0001"
            assert ws1.cell(row=2, column=7).value == "HIGH"  # Peak Level
            assert ws1.cell(row=3, column=1).value == "warn_cut_in_2_0002"

            # Verify Risk Assessment Details sheet data
            ws2 = wb["Risk Assessment Details"]
            assert ws2.cell(row=1, column=1).value == "Media Time (s)"
            assert ws2.cell(row=2, column=8).value == 1.45  # Raw TTC
            assert ws2.cell(row=2, column=9).value == 0.793  # Norm TTC

            # Verify Warning History sheet data
            ws3 = wb["Warning History"]
            assert ws3.cell(row=2, column=6).value == "ACTIVATION"
            assert ws3.cell(row=3, column=6).value == "ESCALATION"

            # Verify Summary & Statistics sheet data
            ws4 = wb["Summary & Statistics"]
            assert ws4.cell(row=1, column=1).value == "THREAT ASSESSMENT & WARNING SYSTEM PERFORMANCE SUMMARY"

            # Verify Configuration & Metadata
            ws5 = wb["Configuration & Metadata"]
            assert ws5.cell(row=1, column=1).value == "ADAS THREAT ASSESSMENT & WARNING CONFIGURATION METADATA"

    def test_non_overwriting_mechanism(self, sample_events):
        """If file exists, exporter must generate a timestamped file rather than overwriting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            exporter = ThreatExcelExporter(output_dir=tmp_dir, filename="test_threat.xlsx")
            path1 = exporter.export(events=sample_events)
            assert path1.name == "test_threat.xlsx"

            path2 = exporter.export(events=sample_events)
            assert path2.name != "test_threat.xlsx"
            assert path2.name.startswith("test_threat_")
            assert path2.name.endswith(".xlsx")
            assert path1.exists()
            assert path2.exists()
