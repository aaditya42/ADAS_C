"""Excel exporter for threat assessment, risk scoring, and directional warning events.

Generates a structured, auditable 5-sheet workbook:
- Sheet 1: Threat Events (unique warning events, lifecycle, duration, peak levels, and reasons)
- Sheet 2: Risk Assessment Details (raw vs normalized indicators, published coefficients, and weighted contributions)
- Sheet 3: Warning History (chronological state transitions: ACTIVATION, ESCALATION, DEESCALATION, RESOLUTION, EXPIRATION)
- Sheet 4: Summary & Statistics (aggregate counts, scenario/class/direction distributions, TTC stats)
- Sheet 5: Configuration & Metadata (pipeline setup, published models, threshold provenance, operational limitations)
"""

from __future__ import annotations

import datetime
from pathlib import Path
import statistics
from typing import Any

from src.threat.schemas import WarningEvent, WarningEventState, DirectionalWarningLevel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreatExcelExporter:
    """Exports comprehensive threat assessments and warning lifecycle logs to an Excel workbook."""

    def __init__(
        self,
        output_dir: str = "./reports",
        filename: str = "Threat_Assessment_Evaluation.xlsx",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.filename = filename

    def export(
        self,
        events: list[WarningEvent],
        history: list[dict[str, Any]] | None = None,
        risk_details: list[dict[str, Any]] | None = None,
        suppressed_duplicates: int = 0,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Generate and save the 5-sheet Excel threat evaluation workbook.

        Args:
            events: All recorded WarningEvent instances (active, resolved, expired).
            history: Chronological warning state transitions.
            risk_details: Detailed per-indicator diagnostics and contributions.
            suppressed_duplicates: Total count of duplicate warnings suppressed.
            metadata: Video metadata (fps, resolution, duration, video path).
            config: Full ADAS / threat warning configuration dictionary.

        Returns:
            Resolved Path to the saved Excel file.
        """
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter

        self.output_dir.mkdir(parents=True, exist_ok=True)
        meta = metadata or {}
        cfg = config or {}
        hist = history or []
        details = risk_details or []

        # Safe non-overwriting destination path
        target_path = self.output_dir / self.filename
        if target_path.exists():
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = target_path.stem
            target_path = self.output_dir / f"{stem}_{timestamp_str}.xlsx"

        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # Remove default blank sheet

        # ------------------------------------------------------------- #
        # Professional Style Tokens
        # ------------------------------------------------------------- #
        header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        sub_header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
        card_fill = PatternFill(start_color="F2F4F4", end_color="F2F4F4", fill_type="solid")

        # Warning level fills
        high_fill = PatternFill(start_color="FADBD8", end_color="FADBD8", fill_type="solid")  # soft red
        med_fill = PatternFill(start_color="FCF3CF", end_color="FCF3CF", fill_type="solid")   # soft amber
        low_fill = PatternFill(start_color="D4EFDF", end_color="D4EFDF", fill_type="solid")   # soft green
        gray_fill = PatternFill(start_color="EAECEE", end_color="EAECEE", fill_type="solid")  # neutral gray

        header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        sub_header_font = Font(name="Segoe UI", size=9, bold=True, color="FFFFFF")
        bold_font = Font(name="Segoe UI", size=9, bold=True)
        body_font = Font(name="Segoe UI", size=9)

        thin_border = Border(
            left=Side(style="thin", color="D5D8DC"),
            right=Side(style="thin", color="D5D8DC"),
            top=Side(style="thin", color="D5D8DC"),
            bottom=Side(style="thin", color="D5D8DC"),
        )
        double_bottom = Border(
            left=Side(style="thin", color="D5D8DC"),
            right=Side(style="thin", color="D5D8DC"),
            top=Side(style="thin", color="D5D8DC"),
            bottom=Side(style="double", color="1B365D"),
        )

        align_center = Alignment(horizontal="center", vertical="center")
        align_left = Alignment(horizontal="left", vertical="center")
        align_right = Alignment(horizontal="right", vertical="center")

        def _level_fill(level_str: str) -> PatternFill | None:
            if "HIGH" in level_str.upper():
                return high_fill
            if "MEDIUM" in level_str.upper() or "MED" in level_str.upper():
                return med_fill
            if "LOW" in level_str.upper():
                return low_fill
            if "UNASSESSED" in level_str.upper() or "RESOLVED" in level_str.upper():
                return gray_fill
            return None

        # ============================================================= #
        # SHEET 1: Threat Events
        # ============================================================= #
        ws1 = wb.create_sheet(title="Threat Events")
        ws1.views.sheetView[0].showGridLines = True

        headers1 = [
            "Event ID",
            "Track ID",
            "Object Class",
            "Scenario",
            "Direction",
            "Initial Level",
            "Peak Level",
            "Final Level",
            "State",
            "Detected Media Time (s)",
            "Confirmation Time (s)",
            "Activation Time (s)",
            "Resolution Time (s)",
            "Warning Duration (s)",
            "Primary Risk Model",
            "Peak Risk Score",
            "Min TTC (s)",
            "Collision Urgency",
            "Priority Rank",
            "Escalation Count",
            "De-escalation Count",
            "Resolution Reason",
        ]
        ws1.append(headers1)
        for col_idx in range(1, len(headers1) + 1):
            cell = ws1.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
        ws1.row_dimensions[1].height = 24

        for row_idx, ev in enumerate(events, start=2):
            dur = ev.duration
            init_lvl = "LOW" if ev.activation_timestamp is not None else "UNASSESSED"
            # Peak level calculation
            levels_seen = [ev.current_level.value]
            for esc in ev.escalation_timestamps:
                levels_seen.append(esc[2])
            peak_lvl = "HIGH" if "HIGH" in levels_seen else ("MEDIUM" if "MEDIUM" in levels_seen else "LOW")

            row_data = [
                ev.event_id,
                ev.track_id,
                ev.object_class,
                ev.scenario,
                ev.direction.value if hasattr(ev.direction, "value") else str(ev.direction),
                init_lvl,
                peak_lvl,
                ev.current_level.value,
                ev.state.value,
                round(ev.first_detected_timestamp, 3),
                round(ev.confirmation_timestamp, 3) if ev.confirmation_timestamp else "",
                round(ev.activation_timestamp, 3) if ev.activation_timestamp else "",
                round(ev.resolution_timestamp, 3) if ev.resolution_timestamp else "",
                round(dur, 3) if dur is not None else "",
                ev.risk_model,
                round(ev.risk_score, 3) if ev.risk_score is not None else "",
                round(ev.ttc, 2) if ev.ttc is not None else "",
                "YES" if ev.collision_urgency else "NO",
                ev.priority_rank if ev.priority_rank is not None else "",
                len(ev.escalation_timestamps),
                len(ev.deescalation_timestamps),
                ev.resolution_reason or "ACTIVE",
            ]
            ws1.append(row_data)
            ws1.row_dimensions[row_idx].height = 18

            for col_idx in range(1, len(headers1) + 1):
                cell = ws1.cell(row=row_idx, column=col_idx)
                cell.font = body_font
                cell.border = thin_border

                # Alignment
                if col_idx in (1, 3, 4, 15, 22):
                    cell.alignment = align_left
                elif col_idx in (2, 5, 6, 7, 8, 9, 18, 19, 20, 21):
                    cell.alignment = align_center
                else:
                    cell.alignment = align_right

                # Coloring for Warning Level columns
                if col_idx in (7, 8):  # Peak Level & Final Level
                    fill_c = _level_fill(str(cell.value))
                    if fill_c:
                        cell.fill = fill_c
                        cell.font = bold_font

        ws1.freeze_panes = "A2"
        if len(events) > 0:
            ws1.auto_filter.ref = ws1.dimensions

        # ============================================================= #
        # SHEET 2: Risk Assessment Details
        # ============================================================= #
        ws2 = wb.create_sheet(title="Risk Assessment Details")
        ws2.views.sheetView[0].showGridLines = True

        headers2 = [
            "Media Time (s)",
            "Track ID",
            "Object Class",
            "Scenario",
            "Risk Model",
            "Model Status",
            "Reliability",
            "Raw TTC (s)",
            "Norm TTC",
            "Raw PET (s)",
            "Norm PET",
            "Raw DRAC (m/s²)",
            "Norm DRAC",
            "Raw Dist (m)",
            "Norm Dist",
            "Raw Rel Speed (m/s)",
            "Norm Rel Speed",
            "Raw Lane Encroach",
            "Norm Lane Encroach",
            "Published Coefficients",
            "Weighted Contributions",
            "Calculated Risk Score",
            "Priority Rank",
        ]
        ws2.append(headers2)
        for col_idx in range(1, len(headers2) + 1):
            cell = ws2.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
        ws2.row_dimensions[1].height = 24

        for row_idx, det in enumerate(details, start=2):
            raw_ttc = det.get("raw_ttc")
            norm_ttc = det.get("norm_ttc")
            raw_pet = det.get("raw_pet")
            norm_pet = det.get("norm_pet")
            raw_drac = det.get("raw_drac")
            norm_drac = det.get("norm_drac")
            raw_dist = det.get("raw_dist")
            norm_dist = det.get("norm_dist")
            raw_rel_spd = det.get("raw_rel_speed")
            norm_rel_spd = det.get("norm_rel_speed")
            raw_encroach = det.get("raw_encroachment")
            norm_encroach = det.get("norm_encroachment")
            score = det.get("risk_score")

            row_data = [
                round(det.get("timestamp", 0.0), 3),
                det.get("track_id", ""),
                det.get("object_class", ""),
                det.get("scenario", ""),
                det.get("risk_model", ""),
                det.get("model_status", ""),
                det.get("reliability", ""),
                round(raw_ttc, 2) if raw_ttc is not None else "",
                round(norm_ttc, 3) if norm_ttc is not None else "",
                round(raw_pet, 2) if raw_pet is not None else "",
                round(norm_pet, 3) if norm_pet is not None else "",
                round(raw_drac, 2) if raw_drac is not None else "",
                round(norm_drac, 3) if norm_drac is not None else "",
                round(raw_dist, 2) if raw_dist is not None else "",
                round(norm_dist, 3) if norm_dist is not None else "",
                round(raw_rel_spd, 2) if raw_rel_spd is not None else "",
                round(norm_rel_spd, 3) if norm_rel_spd is not None else "",
                round(raw_encroach, 2) if raw_encroach is not None else "",
                round(norm_encroach, 3) if norm_encroach is not None else "",
                det.get("published_coefficients", ""),
                det.get("weighted_contributions", ""),
                round(score, 3) if score is not None else "",
                det.get("priority_rank", ""),
            ]
            ws2.append(row_data)
            ws2.row_dimensions[row_idx].height = 18

            for col_idx in range(1, len(headers2) + 1):
                cell = ws2.cell(row=row_idx, column=col_idx)
                cell.font = body_font
                cell.border = thin_border
                if col_idx in (3, 4, 5, 6, 7, 20, 21):
                    cell.alignment = align_left
                elif col_idx in (2, 23):
                    cell.alignment = align_center
                else:
                    cell.alignment = align_right

        ws2.freeze_panes = "A2"
        if len(details) > 0:
            ws2.auto_filter.ref = ws2.dimensions

        # ============================================================= #
        # SHEET 3: Warning History
        # ============================================================= #
        ws3 = wb.create_sheet(title="Warning History")
        ws3.views.sheetView[0].showGridLines = True

        headers3 = [
            "Media Time (s)",
            "Event ID",
            "Track ID",
            "Direction",
            "Scenario",
            "Transition Type",
            "Previous Level",
            "New Level",
            "Event State",
            "Risk Score",
            "Risk Model",
            "Reliability",
            "TTC (s)",
            "Triggering Evidence / Reason",
        ]
        ws3.append(headers3)
        for col_idx in range(1, len(headers3) + 1):
            cell = ws3.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
        ws3.row_dimensions[1].height = 24

        for row_idx, h in enumerate(hist, start=2):
            ttc_val = h.get("ttc")
            score_val = h.get("risk_score")
            row_data = [
                round(h.get("timestamp", 0.0), 3),
                h.get("event_id", ""),
                h.get("track_id", ""),
                h.get("direction", ""),
                h.get("scenario", ""),
                h.get("transition_type", ""),
                h.get("previous_level", ""),
                h.get("new_level", ""),
                h.get("event_state", ""),
                round(score_val, 3) if score_val is not None else "",
                h.get("risk_model", ""),
                h.get("reliability", ""),
                round(ttc_val, 2) if ttc_val is not None else "",
                h.get("triggering_evidence", ""),
            ]
            ws3.append(row_data)
            ws3.row_dimensions[row_idx].height = 18

            for col_idx in range(1, len(headers3) + 1):
                cell = ws3.cell(row=row_idx, column=col_idx)
                cell.font = body_font
                cell.border = thin_border
                if col_idx in (2, 5, 11, 14):
                    cell.alignment = align_left
                elif col_idx in (3, 4, 6, 7, 8, 9, 12):
                    cell.alignment = align_center
                else:
                    cell.alignment = align_right

                # Color transition level
                if col_idx == 8:  # New Level
                    fill_c = _level_fill(str(cell.value))
                    if fill_c:
                        cell.fill = fill_c
                        cell.font = bold_font

        ws3.freeze_panes = "A2"
        if len(hist) > 0:
            ws3.auto_filter.ref = ws3.dimensions

        # ============================================================= #
        # SHEET 4: Summary & Statistics
        # ============================================================= #
        ws4 = wb.create_sheet(title="Summary & Statistics")
        ws4.views.sheetView[0].showGridLines = True

        # Header Title Banner
        ws4.merge_cells("A1:D1")
        banner = ws4["A1"]
        banner.value = "THREAT ASSESSMENT & WARNING SYSTEM PERFORMANCE SUMMARY"
        banner.fill = header_fill
        banner.font = header_font
        banner.alignment = align_center
        ws4.row_dimensions[1].height = 26

        # Sub-header
        ws4["A2"] = "Metric Category"
        ws4["B2"] = "Performance Indicator"
        ws4["C2"] = "Observed Value"
        ws4["D2"] = "Operational Benchmark / Defensibility"
        for col in ("A", "B", "C", "D"):
            c = ws4[f"{col}2"]
            c.fill = sub_header_fill
            c.font = sub_header_font
            c.alignment = align_center
            c.border = thin_border
        ws4.row_dimensions[2].height = 20

        # Calculations
        durations = [ev.duration for ev in events if ev.duration is not None]
        mean_dur = statistics.mean(durations) if durations else 0.0
        min_dur = min(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0

        ttcs = [ev.ttc for ev in events if ev.ttc is not None]
        mean_ttc = statistics.mean(ttcs) if ttcs else 0.0
        min_ttc = min(ttcs) if ttcs else 0.0

        high_events = sum(1 for ev in events if ev.current_level == DirectionalWarningLevel.HIGH)
        med_events = sum(1 for ev in events if ev.current_level == DirectionalWarningLevel.MEDIUM)
        low_events = sum(1 for ev in events if ev.current_level == DirectionalWarningLevel.LOW)

        peak_high = sum(1 for ev in events if any(esc[2] == "HIGH" for esc in ev.escalation_timestamps) or ev.current_level == DirectionalWarningLevel.HIGH)
        peak_med = sum(1 for ev in events if (not any(esc[2] == "HIGH" for esc in ev.escalation_timestamps) and ev.current_level != DirectionalWarningLevel.HIGH) and (any(esc[2] == "MEDIUM" for esc in ev.escalation_timestamps) or ev.current_level == DirectionalWarningLevel.MEDIUM))
        peak_low = len(events) - peak_high - peak_med

        escalations = sum(len(ev.escalation_timestamps) for ev in events)
        deescalations = sum(len(ev.deescalation_timestamps) for ev in events)

        # Scenarios distribution
        scenarios_dist: dict[str, int] = {}
        for ev in events:
            scenarios_dist[ev.scenario] = scenarios_dist.get(ev.scenario, 0) + 1

        # Directions distribution
        dir_dist: dict[str, int] = {}
        for ev in events:
            d_str = ev.direction.value if hasattr(ev.direction, "value") else str(ev.direction)
            dir_dist[d_str] = dir_dist.get(d_str, 0) + 1

        # Class distribution
        cls_dist: dict[str, int] = {}
        for ev in events:
            cls_dist[ev.object_class] = cls_dist.get(ev.object_class, 0) + 1

        summary_rows = [
            ("Video Context", "Total Processed Media Duration", f"{meta.get('duration_s', 0.0):.2f} s", "Video stream media time"),
            ("Video Context", "Processed Frame Count", f"{meta.get('frame_count', 0)} frames", "Total frames ingested"),
            ("Video Context", "Effective Processing FPS", f"{meta.get('processing_fps', 0.0):.1f} FPS", "Throughput without frame loss"),
            ("Warning Events", "Total Unique Warning Events", len(events), "Unique track-scenario instances"),
            ("Warning Events", "Peak HIGH Warnings", peak_high, "Events that reached HIGH tier"),
            ("Warning Events", "Peak MEDIUM Warnings", peak_med, "Events peaking at MEDIUM tier"),
            ("Warning Events", "Peak LOW Warnings", peak_low, "Events remaining at LOW tier"),
            ("Warning Lifecycle", "Total Warning Escalations", escalations, "Confirmed risk elevation transitions"),
            ("Warning Lifecycle", "Total Warning De-escalations", deescalations, "Hysteresis-sustained de-escalations"),
            ("Warning Lifecycle", "Duplicate Warnings Suppressed", suppressed_duplicates, "Same-level redundant alerts filtered"),
            ("Warning Temporal", "Mean Warning Duration", f"{mean_dur:.2f} s", "Average duration of active warnings"),
            ("Warning Temporal", "Min Warning Duration", f"{min_dur:.2f} s", "Minimum active event persistence"),
            ("Warning Temporal", "Max Warning Duration", f"{max_dur:.2f} s", "Longest continuous warning interaction"),
            ("Conflict Diagnostics", "Valid TTC Measurements Recorded", len(ttcs), "Kinematically closing objects with valid depth"),
            ("Conflict Diagnostics", "Mean Valid TTC", f"{mean_ttc:.2f} s", "Average time-to-collision for valid tracks"),
            ("Conflict Diagnostics", "Minimum Valid TTC", f"{min_ttc:.2f} s", "Most acute closing conflict observed"),
        ]

        curr_r = 3
        for cat, ind, val, desc in summary_rows:
            ws4[f"A{curr_r}"] = cat
            ws4[f"B{curr_r}"] = ind
            ws4[f"C{curr_r}"] = val
            ws4[f"D{curr_r}"] = desc
            for col in ("A", "B", "C", "D"):
                c = ws4[f"{col}{curr_r}"]
                c.font = body_font
                c.border = thin_border
            ws4[f"A{curr_r}"].alignment = align_left
            ws4[f"B{curr_r}"].alignment = align_left
            ws4[f"C{curr_r}"].alignment = align_center
            ws4[f"C{curr_r}"].font = bold_font
            ws4[f"D{curr_r}"].alignment = align_left
            ws4.row_dimensions[curr_r].height = 18
            curr_r += 1

        # Section: Distribution Tables
        curr_r += 1
        ws4.merge_cells(f"A{curr_r}:D{curr_r}")
        sec2 = ws4[f"A{curr_r}"]
        sec2.value = "THREAT SCENARIO & DIRECTIONAL DISTRIBUTION"
        sec2.fill = sub_header_fill
        sec2.font = sub_header_font
        sec2.alignment = align_center
        ws4.row_dimensions[curr_r].height = 20
        curr_r += 1

        ws4[f"A{curr_r}"] = "Dimension"
        ws4[f"B{curr_r}"] = "Category / Zone"
        ws4[f"C{curr_r}"] = "Event Count"
        ws4[f"D{curr_r}"] = "Proportion (%)"
        for col in ("A", "B", "C", "D"):
            c = ws4[f"{col}{curr_r}"]
            c.fill = card_fill
            c.font = bold_font
            c.alignment = align_center
            c.border = thin_border
        curr_r += 1

        total_ev = max(len(events), 1)
        # Scenarios
        for sc_name, sc_count in scenarios_dist.items():
            ws4[f"A{curr_r}"] = "Scenario"
            ws4[f"B{curr_r}"] = sc_name
            ws4[f"C{curr_r}"] = sc_count
            ws4[f"D{curr_r}"] = f"{(sc_count / total_ev) * 100:.1f} %"
            for col in ("A", "B", "C", "D"):
                ws4[f"{col}{curr_r}"].font = body_font
                ws4[f"{col}{curr_r}"].border = thin_border
            ws4[f"A{curr_r}"].alignment = align_left
            ws4[f"B{curr_r}"].alignment = align_left
            ws4[f"C{curr_r}"].alignment = align_center
            ws4[f"D{curr_r}"].alignment = align_center
            curr_r += 1

        # Directions
        for d_name, d_count in dir_dist.items():
            ws4[f"A{curr_r}"] = "Direction Zone"
            ws4[f"B{curr_r}"] = d_name
            ws4[f"C{curr_r}"] = d_count
            ws4[f"D{curr_r}"] = f"{(d_count / total_ev) * 100:.1f} %"
            for col in ("A", "B", "C", "D"):
                ws4[f"{col}{curr_r}"].font = body_font
                ws4[f"{col}{curr_r}"].border = thin_border
            ws4[f"A{curr_r}"].alignment = align_left
            ws4[f"B{curr_r}"].alignment = align_left
            ws4[f"C{curr_r}"].alignment = align_center
            ws4[f"D{curr_r}"].alignment = align_center
            curr_r += 1

        # Object Classes
        for cl_name, cl_count in cls_dist.items():
            ws4[f"A{curr_r}"] = "Object Class"
            ws4[f"B{curr_r}"] = cl_name
            ws4[f"C{curr_r}"] = cl_count
            ws4[f"D{curr_r}"] = f"{(cl_count / total_ev) * 100:.1f} %"
            for col in ("A", "B", "C", "D"):
                ws4[f"{col}{curr_r}"].font = body_font
                ws4[f"{col}{curr_r}"].border = thin_border
            ws4[f"A{curr_r}"].alignment = align_left
            ws4[f"B{curr_r}"].alignment = align_left
            ws4[f"C{curr_r}"].alignment = align_center
            ws4[f"D{curr_r}"].alignment = align_center
            curr_r += 1

        # ============================================================= #
        # SHEET 5: Configuration & Metadata
        # ============================================================= #
        ws5 = wb.create_sheet(title="Configuration & Metadata")
        ws5.views.sheetView[0].showGridLines = True

        ws5.merge_cells("A1:C1")
        title5 = ws5["A1"]
        title5.value = "ADAS THREAT ASSESSMENT & WARNING CONFIGURATION METADATA"
        title5.fill = header_fill
        title5.font = header_font
        title5.alignment = align_center
        ws5.row_dimensions[1].height = 26

        ws5["A2"] = "Parameter Group"
        ws5["B2"] = "Parameter Name"
        ws5["C2"] = "Configured Value / Provenance"
        for col in ("A", "B", "C"):
            c = ws5[f"{col}2"]
            c.fill = sub_header_fill
            c.font = sub_header_font
            c.alignment = align_center
            c.border = thin_border
        ws5.row_dimensions[2].height = 20

        meta_rows = [
            ("Video Source", "Input File", str(meta.get("video_path", "data/test1.mp4"))),
            ("Video Source", "Resolution", f"{meta.get('width', 1280)} x {meta.get('height', 720)}"),
            ("Video Source", "Source FPS", f"{meta.get('fps', 30.0):.2f}"),
            ("Video Source", "Total Frames", str(meta.get("frame_count", 0))),
            ("Video Source", "Media Duration", f"{meta.get('duration_s', 0.0):.2f} s"),
            ("Perception Architecture", "Object Detector", "YOLOv8x (COCO Pretrained)"),
            ("Perception Architecture", "Multi-Object Tracker", "ByteTrack (Kalman Filter + Hungarian Matching)"),
            ("Perception Architecture", "Depth Estimator", "MiDaS v2.1 Small (Relative Inverse Depth)"),
            ("Perception Architecture", "Optical Flow", "Farneback Dense Flow / Ego-Motion Compensation"),
            ("Risk Models", "Option A Model (2025 Comprehensive)", "Weights: TTC=0.25, PET=0.24, DRAC=0.21, Gap=0.16, SpeedDiff=0.14 (Sum=1.00) [Published Exact]"),
            ("Risk Models", "Option B Model (2025 Direct Linear)", "Weights: TTC=0.30, Dist=0.30, RelSpd=0.20, Overlap=0.10, Head=0.05, TTZ=0.05 (Sum=1.00) [Published Exact]"),
            ("Risk Models", "SSAM Diagnostic Model", "Surrogate Safety Assessment Model: TTC <= 1.5s, PET <= 5.0s, DRAC >= 3.4 m/s² [Published Exact]"),
            ("Risk Models", "Rule-Based Fallback", "Deterministic geometric collision check for uncalibrated or single-indicator situations"),
            ("Warning Generation", "LOW Advisory Threshold", "Risk Score >= 0.35 or Lane Encroachment [Calibrated ADAS Baseline]"),
            ("Warning Generation", "MEDIUM Warning Threshold", "Risk Score >= 0.55 or Developing Conflict [Calibrated ADAS Baseline]"),
            ("Warning Generation", "HIGH Urgent Warning Threshold", "Risk Score >= 0.75 or Critical TTC <= 2.0s Closing [Calibrated ADAS Baseline]"),
            ("Warning Dynamics", "Activation Confirmation Window", "3 frames AND 0.10 s duration filter"),
            ("Warning Dynamics", "Fast-Track Urgency Bypass", "TTC <= 1.5s OR Overlap > 40% (0 confirmation frames)"),
            ("Warning Dynamics", "De-escalation Hysteresis Hold", "0.80 s sustained lower risk required"),
            ("Warning Dynamics", "Resolution Grace Period", "0.50 s below advisory before termination"),
            ("Warning Dynamics", "Track Loss Grace Timeout", "0.50 s dropout tolerance before expiry"),
            ("Warning Dynamics", "Duplicate Warning Suppression", "Active (Re-alerts suppressed while level maintained)"),
            ("Directional Estimation", "Camera Mounting & Field of View", "Forward-facing monocular camera: Center [-0.3, 0.3], Left [-1.0, -0.3], Right [0.3, 1.0] normalized"),
            ("Directional Estimation", "Blind Spot & Rear Coverage", "STRICT LIMITATION: Monocular forward camera cannot assess side blind spots or rear approaching vehicles"),
            ("Safety Disclaimer", "Regulatory Status", "RESEARCH PROTOTYPE ONLY — NOT FOR PRODUCTION SAFETY-CRITICAL AUTOMOTIVE DEPLOYMENT"),
            ("Safety Disclaimer", "Certification Level", "Non-certified experimental ADAS system for educational and research benchmarking"),
        ]

        curr_r = 3
        for grp, p_name, p_val in meta_rows:
            ws5[f"A{curr_r}"] = grp
            ws5[f"B{curr_r}"] = p_name
            ws5[f"C{curr_r}"] = p_val
            for col in ("A", "B", "C"):
                c = ws5[f"{col}{curr_r}"]
                c.font = body_font
                c.border = thin_border
            ws5[f"A{curr_r}"].alignment = align_left
            ws5[f"B{curr_r}"].alignment = align_left
            ws5[f"C{curr_r}"].alignment = align_left
            ws5.row_dimensions[curr_r].height = 18
            curr_r += 1

        # ------------------------------------------------------------- #
        # Auto-adjust Column Widths for All Sheets
        # ------------------------------------------------------------- #
        for sheet in (ws1, ws2, ws3, ws4, ws5):
            for col in sheet.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    if cell.value is not None:
                        val_str = str(cell.value)
                        # Skip long banner strings for width calculation
                        if len(val_str) < 80:
                            max_len = max(max_len, len(val_str))
                sheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

        wb.save(target_path)
        logger.info("Threat Assessment Evaluation Excel exported to %s", target_path)
        return target_path

