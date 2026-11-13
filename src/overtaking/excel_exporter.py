from __future__ import annotations

import datetime
from pathlib import Path
import statistics
from typing import Any

from src.overtaking.schemas import OvertakingEvent, OvertakingEventStatus, OvertakingState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OvertakingExcelExporter:
    """Exports ego-vehicle overtaking detection events and evaluation data to an Excel workbook.

    Generates a structured 4-sheet workbook:
    - Sheet 1: Overtaking Events (all event details, timestamps, positions, quality metrics)
    - Sheet 2: Summary (session-level statistics, completion rate, duration metrics, delay analysis)
    - Sheet 3: Event Diagnostics (tracking health, dropouts, termination reasons, continuity)
    - Sheet 4: Configuration & Metadata (model parameters, detection thresholds, video source metadata)
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        filename: str = "Overtaking_Evaluation.xlsx",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.filename = filename

    def export(
        self,
        events: list[OvertakingEvent],
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Generate and save the Excel evaluation workbook.

        Args:
            events: All recorded OvertakingEvent records (completed, incomplete, uncertain).
            metadata: Video metadata (fps, resolution, frame count, duration, video path).
            config: Full ADAS / overtaking configuration dictionary.

        Returns:
            Resolved Path to the saved Excel file.
        """
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter

        self.output_dir.mkdir(parents=True, exist_ok=True)
        meta = metadata or {}
        cfg = config or {}

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

        # Status fills
        completed_fill = PatternFill(start_color="E8F8F5", end_color="E8F8F5", fill_type="solid")
        incomplete_fill = PatternFill(start_color="FDEDEC", end_color="FDEDEC", fill_type="solid")
        uncertain_fill = PatternFill(start_color="FEF9E7", end_color="FEF9E7", fill_type="solid")

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

        # ============================================================= #
        # Sheet 1: Overtaking Events
        # ============================================================= #
        ws1 = wb.create_sheet(title="Overtaking Events")
        ws1.views.sheetView[0].showGridLines = True
        ws1.freeze_panes = "A2"

        headers_s1 = [
            "Event ID",
            "Track ID",
            "Track Instance ID",
            "Vehicle Class",
            "Overtaking Start Timestamp (s)",
            "Overtaking End Timestamp (s)",
            "Overtaking Duration (s)",
            "Start Confirmation Timestamp (s)",
            "End Confirmation Timestamp (s)",
            "Start Confirmation Delay (s)",
            "Completion Confirmation Delay (s)",
            "Initial Bounding Box Centre X (px)",
            "Initial Bounding Box Centre Y (px)",
            "Final Bounding Box Centre X (px)",
            "Final Bounding Box Centre Y (px)",
            "Initial Bounding Box Width (px)",
            "Initial Bounding Box Height (px)",
            "Final Bounding Box Width (px)",
            "Final Bounding Box Height (px)",
            "Initial Relative Depth",
            "Final Relative Depth",
            "Initial Ego-Corridor Relationship",
            "Final Ego-Corridor Relationship",
            "Event Status",
            "Completion Confidence",
            "Valid Observation Count",
            "Missing Observation Duration (s)",
            "Identity Continuity Status",
            "Lane Data Validity",
            "Depth Data Validity",
            "Uncertainty / Termination Reason",
            "Event Notes",
        ]

        ws1.append(headers_s1)
        for col_idx in range(1, len(headers_s1) + 1):
            cell = ws1.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
        ws1.row_dimensions[1].height = 28

        for row_idx, ev in enumerate(events, start=2):
            is_comp = ev.status == OvertakingEventStatus.COMPLETED

            # Coordinates
            init_cx = round(ev.initial_position[0], 1) if ev.initial_position else None
            init_cy = round(ev.initial_position[1], 1) if ev.initial_position else None
            final_cx = round(ev.latest_position[0], 1) if ev.latest_position else None
            final_cy = round(ev.latest_position[1], 1) if ev.latest_position else None

            init_w = round(ev.initial_bbox.width, 1) if ev.initial_bbox else None
            init_h = round(ev.initial_bbox.height, 1) if ev.initial_bbox else None
            final_w = round(ev.latest_bbox.width, 1) if ev.latest_bbox else None
            final_h = round(ev.latest_bbox.height, 1) if ev.latest_bbox else None

            # Notes
            notes_parts = [f"Side: {ev.passing_side}"]
            if ev.reassociated_track_ids:
                notes_parts.append(f"Reassociated IDs: {ev.reassociated_track_ids}")
            notes_str = "; ".join(notes_parts)

            row_data = [
                ev.event_id,
                ev.track_id,
                ev.track_instance_id,
                ev.vehicle_class,
                round(ev.start_timestamp, 3) if ev.start_timestamp is not None else None,
                round(ev.end_timestamp, 3) if is_comp and ev.end_timestamp is not None else None,
                round(ev.duration_seconds, 3) if is_comp and ev.duration_seconds is not None else None,
                round(ev.start_confirmation_timestamp, 3) if ev.start_confirmation_timestamp is not None else None,
                round(ev.end_confirmation_timestamp, 3) if is_comp and ev.end_confirmation_timestamp is not None else None,
                round(ev.start_confirmation_delay_seconds, 3) if ev.start_confirmation_delay_seconds is not None else None,
                round(ev.completion_confirmation_delay_seconds, 3) if is_comp and ev.completion_confirmation_delay_seconds is not None else None,
                init_cx,
                init_cy,
                final_cx,
                final_cy,
                init_w,
                init_h,
                final_w,
                final_h,
                round(ev.initial_depth, 2) if ev.initial_depth is not None else "N/A",
                round(ev.latest_depth, 2) if ev.latest_depth is not None else "N/A",
                getattr(ev.initial_corridor_position, "value", str(ev.initial_corridor_position)),
                getattr(ev.latest_corridor_position, "value", str(ev.latest_corridor_position)),
                ev.status.value,
                round(ev.completion_confidence, 2),
                ev.valid_observation_count,
                round(ev.missing_observation_duration, 3),
                ev.identity_continuity_status.value,
                "VALID" if ev.lane_data_validity else "INVALID",
                "VALID" if ev.depth_data_validity else "INVALID",
                ev.uncertainty_reason or ("NONE" if is_comp else "UNKNOWN"),
                notes_str,
            ]
            ws1.append(row_data)

            # Row styling
            row_fill = completed_fill if is_comp else (uncertain_fill if ev.status == OvertakingEventStatus.UNCERTAIN else incomplete_fill)
            for c_idx in range(1, len(row_data) + 1):
                c = ws1.cell(row=row_idx, column=c_idx)
                c.font = body_font
                c.border = thin_border
                c.fill = row_fill
                if c_idx in (1, 2, 3, 4, 22, 23, 24, 28, 29, 30):
                    c.alignment = align_center
                elif c_idx in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27):
                    c.alignment = align_right
                else:
                    c.alignment = align_left
            ws1.row_dimensions[row_idx].height = 19

        # Auto-fit column widths
        for col in ws1.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws1.column_dimensions[col_letter].width = max(max_len + 3, 12)

        # ============================================================= #
        # Sheet 2: Summary
        # ============================================================= #
        ws2 = wb.create_sheet(title="Summary")
        ws2.views.sheetView[0].showGridLines = True

        total_candidates = len(events)
        confirmed_starts = len([e for e in events if e.start_timestamp is not None])
        completed_events = len([e for e in events if e.status == OvertakingEventStatus.COMPLETED])
        incomplete_events = len([e for e in events if e.status == OvertakingEventStatus.INCOMPLETE])
        uncertain_events = len([e for e in events if e.status == OvertakingEventStatus.UNCERTAIN])
        invalidated_events = len([e for e in events if e.status == OvertakingEventStatus.INVALIDATED])

        completion_rate = (
            f"{(completed_events / confirmed_starts * 100.0):.1f}%"
            if confirmed_starts > 0
            else "N/A"
        )

        completed_durations = [
            e.duration_seconds
            for e in events
            if e.status == OvertakingEventStatus.COMPLETED and e.duration_seconds is not None
        ]

        if completed_durations:
            mean_dur = f"{statistics.mean(completed_durations):.3f}"
            median_dur = f"{statistics.median(completed_durations):.3f}"
            min_dur = f"{min(completed_durations):.3f}"
            max_dur = f"{max(completed_durations):.3f}"
            std_dur = (
                f"{statistics.stdev(completed_durations):.3f}"
                if len(completed_durations) > 1
                else "0.000"
            )
        else:
            mean_dur = median_dur = min_dur = max_dur = std_dur = "N/A"

        start_delays = [
            e.start_confirmation_delay_seconds
            for e in events
            if e.start_confirmation_delay_seconds is not None
        ]
        comp_delays = [
            e.completion_confirmation_delay_seconds
            for e in events
            if e.completion_confirmation_delay_seconds is not None
        ]
        mean_start_delay = f"{statistics.mean(start_delays):.3f}" if start_delays else "N/A"
        mean_comp_delay = f"{statistics.mean(comp_delays):.3f}" if comp_delays else "N/A"

        unique_instances = len(set(e.track_instance_id for e in events))

        summary_rows = [
            ("Session & Video Information", "", ""),
            ("Source Video Path", str(meta.get("video_path", "N/A")), "Input video file"),
            ("Processing Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Session execution time"),
            ("Total Processed Frames", meta.get("frame_count", 0), "Frames ingested"),
            ("Source Video FPS", meta.get("fps", 30.0), "Frame rate"),
            ("Video Resolution", f"{meta.get('width', 1280)} x {meta.get('height', 720)}", "Width x Height"),
            ("Total Video Duration (s)", round(meta.get("duration_seconds", 0.0), 3), "Source elapsed seconds"),
            ("Unique Target Track Episodes", unique_instances, "Continuous track episodes"),
            ("", "", ""),
            ("Overtaking Event Counts", "", ""),
            ("Total Candidate Vehicles Ahead", total_candidates, "Candidates evaluated"),
            ("Confirmed Overtaking Starts", confirmed_starts, "Events exceeding start duration threshold"),
            ("Completed Overtaking Events", completed_events, "Manoeuvres completed through exit boundary"),
            ("Incomplete Events", incomplete_events, "Lost or aborted manoeuvres"),
            ("Uncertain Events", uncertain_events, "Ambiguous exits or unverified continuity"),
            ("Invalidated Events", invalidated_events, "Spurious or corrupted detections"),
            ("Completion Rate (%)", completion_rate, "Completed / Confirmed Starts * 100"),
            ("", "", ""),
            ("Manoeuvre Duration Statistics (Completed Events Only)", "", ""),
            ("Mean Duration (s)", mean_dur, "Average completed duration"),
            ("Median Duration (s)", median_dur, "Median completed duration"),
            ("Minimum Duration (s)", min_dur, "Fastest completed pass"),
            ("Maximum Duration (s)", max_dur, "Longest completed pass"),
            ("Standard Deviation (s)", std_dur, "Duration dispersion"),
            ("", "", ""),
            ("Detector Confirmation Latency Analysis", "", ""),
            ("Mean Start Confirmation Delay (s)", mean_start_delay, "Delay between onset and start confirmation"),
            ("Mean Completion Confirmation Delay (s)", mean_comp_delay, "Delay between exit boundary and confirmation"),
        ]

        ws2.append(["Metric / Parameter", "Value", "Notes & Documentation"])
        ws2.cell(row=1, column=1).fill = header_fill
        ws2.cell(row=1, column=1).font = header_font
        ws2.cell(row=1, column=2).fill = header_fill
        ws2.cell(row=1, column=2).font = header_font
        ws2.cell(row=1, column=3).fill = header_fill
        ws2.cell(row=1, column=3).font = header_font
        ws2.row_dimensions[1].height = 26

        for r_idx, (metric, val, note) in enumerate(summary_rows, start=2):
            ws2.append([metric, val, note])
            c1 = ws2.cell(row=r_idx, column=1)
            c2 = ws2.cell(row=r_idx, column=2)
            c3 = ws2.cell(row=r_idx, column=3)
            ws2.row_dimensions[r_idx].height = 20

            if not val and not note:
                # Section header row
                c1.fill = sub_header_fill
                c1.font = sub_header_font
                c2.fill = sub_header_fill
                c3.fill = sub_header_fill
            else:
                c1.font = bold_font if "Rate" in metric or "Completed" in metric else body_font
                c2.font = bold_font
                c3.font = body_font
                c1.border = thin_border
                c2.border = thin_border
                c3.border = thin_border
                c2.alignment = align_center

        ws2.column_dimensions["A"].width = 46
        ws2.column_dimensions["B"].width = 28
        ws2.column_dimensions["C"].width = 42

        # ============================================================= #
        # Sheet 3: Event Diagnostics
        # ============================================================= #
        ws3 = wb.create_sheet(title="Event Diagnostics")
        ws3.views.sheetView[0].showGridLines = True
        ws3.freeze_panes = "A2"

        headers_s3 = [
            "Event ID",
            "Track ID",
            "Track Instance ID",
            "Vehicle Class",
            "Candidate Start Timestamp (s)",
            "Start Confirmation Timestamp (s)",
            "Candidate End Timestamp (s)",
            "End Confirmation Timestamp (s)",
            "Final Event Status",
            "Termination Reason",
            "Missing Observation Duration (s)",
            "Valid Observation Count",
            "Lane Data Availability",
            "Depth Data Availability",
            "Identity Continuity Status",
            "Diagnostic Detail",
        ]

        ws3.append(headers_s3)
        for col_idx in range(1, len(headers_s3) + 1):
            cell = ws3.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
        ws3.row_dimensions[1].height = 28

        for row_idx, ev in enumerate(events, start=2):
            is_comp = ev.status == OvertakingEventStatus.COMPLETED

            # Detail summary
            if is_comp:
                diag = f"Completed successfully on {ev.passing_side} side. Valid observations: {ev.valid_observation_count}."
            elif ev.status == OvertakingEventStatus.INCOMPLETE:
                diag = f"Incomplete: {ev.uncertainty_reason}. Last seen at t={ev.last_observed_timestamp:.3f}s with missing dur={ev.missing_observation_duration:.3f}s."
            else:
                diag = f"Uncertain: {ev.uncertainty_reason}. Identity continuity: {ev.identity_continuity_status.value}."

            row_data_s3 = [
                ev.event_id,
                ev.track_id,
                ev.track_instance_id,
                ev.vehicle_class,
                round(ev.candidate_start_timestamp, 3) if ev.candidate_start_timestamp is not None else "N/A",
                round(ev.start_confirmation_timestamp, 3) if ev.start_confirmation_timestamp is not None else "N/A",
                round(ev.candidate_end_timestamp, 3) if ev.candidate_end_timestamp is not None else "N/A",
                round(ev.end_confirmation_timestamp, 3) if ev.end_confirmation_timestamp is not None else "N/A",
                ev.status.value,
                ev.uncertainty_reason or ("NONE" if is_comp else "UNKNOWN"),
                round(ev.missing_observation_duration, 3),
                ev.valid_observation_count,
                "AVAILABLE" if ev.lane_data_validity else "UNAVAILABLE",
                "AVAILABLE" if ev.depth_data_validity else "UNAVAILABLE",
                ev.identity_continuity_status.value,
                diag,
            ]
            ws3.append(row_data_s3)

            row_fill = completed_fill if is_comp else (uncertain_fill if ev.status == OvertakingEventStatus.UNCERTAIN else incomplete_fill)
            for c_idx in range(1, len(row_data_s3) + 1):
                c = ws3.cell(row=row_idx, column=c_idx)
                c.font = body_font
                c.border = thin_border
                c.fill = row_fill
                if c_idx in (1, 2, 3, 4, 9, 13, 14, 15):
                    c.alignment = align_center
                elif c_idx in (5, 6, 7, 8, 11, 12):
                    c.alignment = align_right
                else:
                    c.alignment = align_left
            ws3.row_dimensions[row_idx].height = 19

        for col in ws3.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws3.column_dimensions[col_letter].width = max(max_len + 3, 12)

        # ============================================================= #
        # Sheet 4: Configuration & Metadata
        # ============================================================= #
        ws4 = wb.create_sheet(title="Configuration & Metadata")
        ws4.views.sheetView[0].showGridLines = True

        ot_cfg = cfg.get("overtaking", {})
        det_cfg = ot_cfg.get("detection", {})
        track_cfg = ot_cfg.get("tracking", {})
        vis_cfg = ot_cfg.get("visualization", {})

        config_rows = [
            ("Parameter / Field", "Configured Value", "Description & Rationale"),
            ("Source Video Path", str(meta.get("video_path", "data/test1.mp4")), "Processed input video path"),
            ("Execution Date / Time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Timestamp when report was generated"),
            ("Source FPS", meta.get("fps", 30.0), "Nominal frame rate"),
            ("Video Resolution", f"{meta.get('width', 1280)} x {meta.get('height', 720)}", "Source dimensions"),
            ("Total Video Duration (s)", round(meta.get("duration_seconds", 0.0), 3), "Stream presentation duration"),
            ("Total Frames Processed", meta.get("frame_count", 0), "Ingested frames"),
            ("", "", ""),
            ("Overtaking Detection Parameters", "", ""),
            ("Overtaking Feature Enabled", ot_cfg.get("enabled", True), "Master feature flag"),
            ("Target Vehicle Classes", ", ".join(ot_cfg.get("target_classes", ["car", "truck", "bus", "van"])), "Monitored semantic categories"),
            ("Min Initial Observations", det_cfg.get("min_initial_observations", 3), "Baseline observations before start check"),
            ("Scale Growth Start Threshold", det_cfg.get("scale_growth_start_threshold", 1.20), "BBox area ratio indicating approach"),
            ("Vertical Drop Ratio", det_cfg.get("vertical_drop_ratio", 0.05), "Normalized y2 descent indicating closure"),
            ("Ahead Max Y Ratio", det_cfg.get("ahead_max_y_ratio", 0.65), "Upper boundary for initial vehicle ahead detection"),
            ("Scale Growth Passing Threshold", det_cfg.get("scale_growth_passing_threshold", 1.45), "BBox area ratio confirming passing alongside"),
            ("Passing Y Ratio", det_cfg.get("passing_y_ratio", 0.55), "Bottom edge ratio confirming lateral overlap"),
            ("Exit Margin Pixels", det_cfg.get("exit_margin_pixels", 60.0), "Boundary pixel distance for optical exit"),
            ("Start Confirmation Duration (s)", det_cfg.get("start_confirmation_duration", 0.25), "Required persistence before confirming start"),
            ("Completion Confirmation Duration (s)", det_cfg.get("completion_confirmation_duration", 0.20), "Disappearance delay before confirming completion"),
            ("Max Event Duration (s)", det_cfg.get("max_event_duration_seconds", 15.0), "Timeout limit for active manoeuvres"),
            ("", "", ""),
            ("Tracking & Identity Continuity", "", ""),
            ("Maximum Track Gap (s)", track_cfg.get("maximum_track_gap_seconds", 0.5), "Gap duration before new track episode"),
            ("Track Loss Grace Duration (s)", track_cfg.get("track_loss_grace_seconds", 0.4), "Tolerated dropout window without dropping candidate"),
            ("Identity Reassociation Enabled", track_cfg.get("identity_reassociation_enabled", True), "Reassociate candidate upon ID switch"),
            ("Identity Reassociation Max Gap (s)", track_cfg.get("identity_reassociation_max_gap", 0.4), "Max temporal gap for ID reassociation"),
            ("Timestamp Fallback Enabled", track_cfg.get("timestamp_fallback_enabled", True), "Fallback to FPS-derived timestamps"),
            ("Fallback FPS", track_cfg.get("fallback_fps", 30.0), "Synthetic timestamp rate"),
            ("", "", ""),
            ("Visualization Configuration", "", ""),
            ("Overlay Visualization Enabled", vis_cfg.get("enabled", True), "Live display layer state"),
            ("Show Target Marker", vis_cfg.get("show_target_marker", True), "Corner brackets and tag above target"),
            ("Show Elapsed Time", vis_cfg.get("show_elapsed_time", True), "Live timer on active badge"),
            ("Show Event State", vis_cfg.get("show_event_state", True), "Display state name"),
            ("Show Completion Notification", vis_cfg.get("show_completion_notification", True), "Temporary banner upon completion"),
            ("Notification Duration (s)", vis_cfg.get("notification_duration", 2.5), "Banner display lifetime"),
            ("Maximum Displayed Events", vis_cfg.get("maximum_displayed_events", 5), "Simultaneous HUD cards limit"),
        ]

        ws4.append(["Configuration Parameter", "Value", "Description"])
        ws4.cell(row=1, column=1).fill = header_fill
        ws4.cell(row=1, column=1).font = header_font
        ws4.cell(row=1, column=2).fill = header_fill
        ws4.cell(row=1, column=2).font = header_font
        ws4.cell(row=1, column=3).fill = header_fill
        ws4.cell(row=1, column=3).font = header_font
        ws4.row_dimensions[1].height = 26

        for r_idx, (p_name, p_val, p_desc) in enumerate(config_rows, start=2):
            ws4.append([p_name, str(p_val), p_desc])
            c1 = ws4.cell(row=r_idx, column=1)
            c2 = ws4.cell(row=r_idx, column=2)
            c3 = ws4.cell(row=r_idx, column=3)
            ws4.row_dimensions[r_idx].height = 20

            if not str(p_val) and not p_desc:
                c1.fill = sub_header_fill
                c1.font = sub_header_font
                c2.fill = sub_header_fill
                c3.fill = sub_header_fill
            else:
                c1.font = body_font
                c2.font = bold_font
                c3.font = body_font
                c1.border = thin_border
                c2.border = thin_border
                c3.border = thin_border
                c2.alignment = align_center

        ws4.column_dimensions["A"].width = 44
        ws4.column_dimensions["B"].width = 30
        ws4.column_dimensions["C"].width = 48

        wb.save(target_path)
        logger.info("Saved overtaking evaluation report to %s (%d events exported)", target_path, len(events))
        return target_path

