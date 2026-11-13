from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from src.schemas.trajectory import EvaluationStatus, PredictionRecord
from src.trajectory.metrics import calculate_trajectory_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryExcelExporter:
    """Exports trajectory forecasting and evaluation data to an Excel workbook.

    Generates a structured multi-sheet workbook:
    - Sheet 1: Prediction Records (event-level predictions and endpoints)
    - Sheet 2: Detailed Future Positions (waypoint-level step predictions and ground truth)
    - Sheet 3: Accuracy Summary (ADE, FDE, endpoint errors, class/horizon breakdowns)
    - Sheet 4: Performance (timing, throughput, and scale metrics)
    - Sheet 5: Configuration and Metadata (system, model, and parameter metadata)
    """

    def __init__(
        self,
        output_path: str = "./reports/NAYAN_Trajectory_Evaluation.xlsx",
        success_threshold_pixels: float = 20.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.success_threshold_pixels = success_threshold_pixels

    def export(
        self,
        records: list[PredictionRecord],
        metadata: dict[str, Any] | None = None,
        performance_stats: dict[str, Any] | None = None,
    ) -> Path:
        """Generate and save the Excel evaluation workbook.

        Args:
            records: All PredictionRecord instances.
            metadata: Run configuration and video source parameters.
            performance_stats: Runtime latency and throughput statistics.

        Returns:
            Resolved Path to the saved Excel file.
        """
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()
        # Remove default sheet
        wb.remove(wb.active)

        # Style tokens
        header_fill = PatternFill(
            start_color="1B365D", end_color="1B365D", fill_type="solid"
        )
        sub_header_fill = PatternFill(
            start_color="2C3E50", end_color="2C3E50", fill_type="solid"
        )
        accent_fill = PatternFill(
            start_color="EAECEE", end_color="EAECEE", fill_type="solid"
        )
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        sub_header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        title_font = Font(name="Segoe UI", size=13, bold=True, color="1B365D")
        body_font = Font(name="Segoe UI", size=10)
        bold_font = Font(name="Segoe UI", size=10, bold=True)
        thin_border = Border(
            left=Side(style="thin", color="D5D8DC"),
            right=Side(style="thin", color="D5D8DC"),
            top=Side(style="thin", color="D5D8DC"),
            bottom=Side(style="thin", color="D5D8DC"),
        )

        metrics = calculate_trajectory_metrics(
            records, success_threshold_pixels=self.success_threshold_pixels
        )

        # ------------------------------------------------------------- #
        # Sheet 1: Prediction Records
        # ------------------------------------------------------------- #
        ws1 = wb.create_sheet(title="Prediction Records")
        ws1.views.sheetView[0].showGridLines = True
        ws1.freeze_panes = "A2"

        headers_s1 = [
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
            "Following Track ID",
            "Leading Track ID",
            "Following Speed (m/s)",
            "Leading Speed (m/s)",
            "Inter-Vehicle Gap (m)",
            "DRAC (m/s²)",
            "DRAC Status",
        ]

        ws1.append(headers_s1)
        for col_idx in range(1, len(headers_s1) + 1):
            cell = ws1.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for r_idx, rec in enumerate(records, start=2):
            drac_formula = (
                f'=IF(OR(U{r_idx}="",V{r_idx}="",W{r_idx}="",W{r_idx}<=0),"",'
                f'IF(U{r_idx}<=V{r_idx},0,(U{r_idx}-V{r_idx})^2/(2*W{r_idx})))'
            )
            row_data = [
                rec.prediction_id,
                rec.track_instance_id,
                rec.track_id,
                rec.class_name,
                rec.prediction_timestamp,
                rec.current_position[0],
                rec.current_position[1],
                rec.horizon_seconds,
                rec.target_timestamp,
                rec.predicted_position[0],
                rec.predicted_position[1],
                rec.actual_position[0] if rec.actual_position is not None else None,
                rec.actual_position[1] if rec.actual_position is not None else None,
                rec.endpoint_error,
                rec.ade,
                rec.fde,
                rec.status.value,
                rec.predictor,
                rec.following_track_id if rec.following_track_id is not None else "",
                rec.leading_track_id if rec.leading_track_id is not None else "",
                rec.following_speed_mps if rec.following_speed_mps is not None else "",
                rec.leading_speed_mps if rec.leading_speed_mps is not None else "",
                rec.inter_vehicle_gap_m if rec.inter_vehicle_gap_m is not None else "",
                drac_formula,
                rec.drac_status,
            ]
            ws1.append(row_data)

            # Apply numeric formatting for numeric DRAC columns
            for c_idx in (21, 22, 23, 24):
                ws1.cell(row=r_idx, column=c_idx).number_format = "0.00"


        # ------------------------------------------------------------- #
        # Sheet 2: Detailed Future Positions (Waypoints)
        # ------------------------------------------------------------- #
        ws2 = wb.create_sheet(title="Detailed Future Positions")
        ws2.views.sheetView[0].showGridLines = True
        ws2.freeze_panes = "A2"

        headers_s2 = [
            "Prediction ID",
            "Track Instance ID",
            "Track ID",
            "Object Class",
            "Prediction Timestamp (s)",
            "Future Timestamp (s)",
            "Time Offset (s)",
            "Predicted X (px)",
            "Predicted Y (px)",
            "Actual X (px)",
            "Actual Y (px)",
            "Displacement Error (px)",
            "Matching Status",
            "Variance X",
            "Variance Y",
        ]

        ws2.append(headers_s2)
        for col_idx in range(1, len(headers_s2) + 1):
            cell = ws2.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for rec in records:
            for wp in rec.waypoint_records:
                wp_row = [
                    rec.prediction_id,
                    rec.track_instance_id,
                    rec.track_id,
                    rec.class_name,
                    rec.prediction_timestamp,
                    wp.future_timestamp,
                    wp.time_offset,
                    wp.predicted_position[0],
                    wp.predicted_position[1],
                    wp.actual_position[0] if wp.actual_position is not None else None,
                    wp.actual_position[1] if wp.actual_position is not None else None,
                    wp.displacement_error,
                    wp.status.value,
                    wp.variance[0] if wp.variance is not None else None,
                    wp.variance[1] if wp.variance is not None else None,
                ]
                ws2.append(wp_row)

        # ------------------------------------------------------------- #
        # Sheet 3: Accuracy Summary
        # ------------------------------------------------------------- #
        ws3 = wb.create_sheet(title="Accuracy Summary")
        ws3.views.sheetView[0].showGridLines = True

        ws3["A1"] = "NAYAN ADAS - Trajectory Prediction Evaluation Summary"
        ws3["A1"].font = title_font

        # Overview Table
        ws3.append([])
        ws3.append(["1. Prediction Counts & Matching Rates", ""])
        ws3.cell(row=3, column=1).font = bold_font
        ws3.cell(row=3, column=1).fill = accent_fill
        ws3.cell(row=3, column=2).fill = accent_fill

        counts_data = [
            ("Total Predictions Generated", metrics["total_predictions"]),
            ("Matched Predictions", metrics["matched_predictions"]),
            ("Unmatched Predictions", metrics["unmatched_predictions"]),
            ("Track Lost Predictions", metrics["track_lost_predictions"]),
            ("Missing Detections at Target", metrics["missing_predictions"]),
            ("Invalid Predictions (Gap/Reset)", metrics["invalid_predictions"]),
            ("Pending Predictions", metrics["pending_predictions"]),
            (
                "Matching Success Rate",
                f"{metrics['matching_success_rate_pct']:.1f}%"
                if metrics["matching_success_rate_pct"] is not None
                else "N/A",
            ),
            (
                f"Threshold Success Rate (<= {metrics['success_threshold_pixels']:.1f} px)",
                f"{metrics['threshold_success_rate_pct']:.1f}%"
                if metrics["threshold_success_rate_pct"] is not None
                else "N/A",
            ),
        ]
        for label, val in counts_data:
            ws3.append([label, val])

        # Distance Error Table
        ws3.append([])
        err_header_row = ws3.max_row + 1
        ws3.append(["2. Trajectory Forecasting Error Metrics (Pixels)", ""])
        ws3.cell(row=err_header_row, column=1).font = bold_font
        ws3.cell(row=err_header_row, column=1).fill = accent_fill
        ws3.cell(row=err_header_row, column=2).fill = accent_fill

        errors_data = [
            ("Mean Endpoint Error (px)", metrics["mean_endpoint_error_px"] or "N/A"),
            ("Median Endpoint Error (px)", metrics["median_endpoint_error_px"] or "N/A"),
            ("Minimum Endpoint Error (px)", metrics["min_endpoint_error_px"] or "N/A"),
            ("Maximum Endpoint Error (px)", metrics["max_endpoint_error_px"] or "N/A"),
            ("Standard Deviation of Error (px)", metrics["std_endpoint_error_px"] or "N/A"),
            ("Mean Average Displacement Error — ADE (px)", metrics["mean_ade_px"] or "N/A"),
            ("Mean Final Displacement Error — FDE (px)", metrics["mean_fde_px"] or "N/A"),
        ]
        for label, val in errors_data:
            ws3.append([label, val])

        # Breakdown by Class Table
        ws3.append([])
        ws3.append(["3. Accuracy Breakdown by Object Class"])
        ws3.cell(row=ws3.max_row, column=1).font = bold_font

        class_headers = [
            "Class",
            "Total",
            "Matched",
            "Match Rate (%)",
            "Mean Error (px)",
            "Median Error (px)",
            "Mean ADE (px)",
        ]
        ws3.append(class_headers)
        hdr_row = ws3.max_row
        for col_idx in range(1, len(class_headers) + 1):
            cell = ws3.cell(row=hdr_row, column=col_idx)
            cell.fill = sub_header_fill
            cell.font = sub_header_font

        for cls_name, cls_info in metrics.get("by_class", {}).items():
            ws3.append([
                cls_name,
                cls_info["total"],
                cls_info["matched"],
                round(cls_info["match_rate_pct"], 1),
                round(cls_info["mean_endpoint_error_px"], 2) if cls_info["mean_endpoint_error_px"] is not None else "N/A",
                round(cls_info["median_endpoint_error_px"], 2) if cls_info["median_endpoint_error_px"] is not None else "N/A",
                round(cls_info["mean_ade_px"], 2) if cls_info["mean_ade_px"] is not None else "N/A",
            ])

        # Breakdown by Horizon Table
        ws3.append([])
        ws3.append(["4. Accuracy Breakdown by Prediction Horizon"])
        ws3.cell(row=ws3.max_row, column=1).font = bold_font

        horizon_headers = [
            "Horizon (s)",
            "Total",
            "Matched",
            "Match Rate (%)",
            "Mean Error (px)",
            "Median Error (px)",
            "Mean ADE (px)",
        ]
        ws3.append(horizon_headers)
        hdr_row_h = ws3.max_row
        for col_idx in range(1, len(horizon_headers) + 1):
            cell = ws3.cell(row=hdr_row_h, column=col_idx)
            cell.fill = sub_header_fill
            cell.font = sub_header_font

        for horizon_val, h_info in metrics.get("by_horizon", {}).items():
            ws3.append([
                f"{horizon_val:.2f}s",
                h_info["total"],
                h_info["matched"],
                round(h_info["match_rate_pct"], 1),
                round(h_info["mean_endpoint_error_px"], 2) if h_info["mean_endpoint_error_px"] is not None else "N/A",
                round(h_info["median_endpoint_error_px"], 2) if h_info["median_endpoint_error_px"] is not None else "N/A",
                round(h_info["mean_ade_px"], 2) if h_info["mean_ade_px"] is not None else "N/A",
            ])

        # ------------------------------------------------------------- #
        # Sheet 4: Performance
        # ------------------------------------------------------------- #
        ws4 = wb.create_sheet(title="Performance")
        ws4.views.sheetView[0].showGridLines = True
        ws4["A1"] = "NAYAN ADAS - Trajectory Prediction Runtime Performance"
        ws4["A1"].font = title_font
        ws4.append([])

        perf = performance_stats or {}
        perf_rows = [
            ("Average Prediction Latency (ms)", perf.get("avg_prediction_latency_ms", "N/A")),
            ("Average Evaluation Latency (ms)", perf.get("avg_evaluation_latency_ms", "N/A")),
            ("Total Prediction Events Emitted", perf.get("total_predictions_emitted", len(records))),
            ("Average Tracked Objects per Frame", perf.get("avg_tracked_objects", "N/A")),
            ("Average Pipeline FPS", perf.get("pipeline_fps", "N/A")),
            ("Total Frames Processed", perf.get("total_frames_processed", "N/A")),
        ]

        ws4.append(["Metric", "Value"])
        ws4.cell(row=3, column=1).fill = sub_header_fill
        ws4.cell(row=3, column=1).font = sub_header_font
        ws4.cell(row=3, column=2).fill = sub_header_fill
        ws4.cell(row=3, column=2).font = sub_header_font

        for label, val in perf_rows:
            ws4.append([label, val])

        # ------------------------------------------------------------- #
        # Sheet 5: Configuration and Metadata
        # ------------------------------------------------------------- #
        ws5 = wb.create_sheet(title="Configuration and Metadata")
        ws5.views.sheetView[0].showGridLines = True
        ws5["A1"] = "NAYAN ADAS - System Configuration & Session Metadata"
        ws5["A1"].font = title_font
        ws5.append([])

        meta = metadata or {}
        meta_rows = [
            ("Video Source Identifier", meta.get("video_source", "data/test2.mp4")),
            ("Video Frame Rate (FPS)", meta.get("video_fps", "N/A")),
            ("Video Resolution", meta.get("video_resolution", "1280 x 720")),
            ("Prediction Model", meta.get("model", "constant_acceleration")),
            ("Configured Prediction Horizon (s)", meta.get("horizon_seconds", 2.0)),
            ("Configured Update Interval (s)", meta.get("update_interval", 0.1)),
            ("Ground Truth Match Tolerance (s)", meta.get("match_tolerance_seconds", 0.05)),
            ("Success Threshold Distance (px)", meta.get("success_threshold_pixels", self.success_threshold_pixels)),
            ("History Retention Duration (s)", meta.get("history_duration_seconds", 2.0)),
            ("Minimum History Points for Prediction", meta.get("minimum_history_points", 10)),
            ("Maximum Track Gap Timeout (s)", meta.get("maximum_track_gap_seconds", 0.5)),
            ("Stale Track Timeout (s)", meta.get("stale_track_timeout_seconds", 1.0)),
            ("Coordinate System", meta.get("coordinate_system", "image")),
            ("Position Reference Point", meta.get("position_reference", "bottom_center")),
            ("DRAC Evaluation Enabled", str(meta.get("drac_enabled", True))),
            ("DRAC Metric Calibration Required", str(meta.get("drac_require_metric", True))),
            ("DRAC Minimum Gap Threshold (m)", meta.get("drac_minimum_gap_meters", 0.01)),
            ("Export Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]

        ws5.append(["Parameter", "Configured Setting / Value"])
        ws5.cell(row=3, column=1).fill = sub_header_fill
        ws5.cell(row=3, column=1).font = sub_header_font
        ws5.cell(row=3, column=2).fill = sub_header_fill
        ws5.cell(row=3, column=2).font = sub_header_font

        for label, val in meta_rows:
            ws5.append([label, val])

        # ------------------------------------------------------------- #
        # Auto-adjust column widths across all sheets
        # ------------------------------------------------------------- #
        for sheet in wb.worksheets:
            for col in sheet.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    if cell.value is not None:
                        val_str = str(cell.value)
                        # Avoid huge column widths from multiline titles
                        if len(val_str) > max_len and len(val_str) < 60:
                            max_len = len(val_str)
                sheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

        # Configure workbook to automatically calculate all formulas upon opening
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.calcMode = "auto"

        # Save to disk
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(self.output_path))
        logger.info("Successfully exported trajectory evaluation workbook to %s", self.output_path)
        return self.output_path.resolve()
