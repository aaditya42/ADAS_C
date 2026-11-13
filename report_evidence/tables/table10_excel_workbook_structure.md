### Table: Excel Workbook Structure

| Workbook Filename | Sheet Name | Row Count | Key Fields Recorded |
| --- | --- | --- | --- |
| Threat_Assessment_Evaluation.xlsx | Threat Events | 63 | Event ID, Track ID, Class, Scenario, Direction, Peak Level, Activation Time, Duration, Min TTC, Reason |
| Threat_Assessment_Evaluation.xlsx | Risk Assessment Details | 3884 | Media Time, Track ID, Raw vs Norm TTC/PET/DRAC/Speed/Encroach, Published Coeffs, Weighted Contribs, Score |
| Threat_Assessment_Evaluation.xlsx | Warning History | 134 | Timestamp, Event ID, Transition Type (ACTIVATION, ESCALATION, RESOLUTION), Prev Level, New Level, Evidence |
| Threat_Assessment_Evaluation.xlsx | Summary & Statistics | 34 | Macro KPIs, Scenario distribution, Directional distribution, Class breakdown, TTC statistics, Duplicate counts |
| Threat_Assessment_Evaluation.xlsx | Configuration & Metadata | 28 | Video source parameters, Perception model setup, Published weights, Calibrated thresholds, Research disclaimer |
| Overtaking_Evaluation.xlsx | Overtaking Events | 9 | Event ID, Track ID, Start/End Media Times, Duration, Lateral Side, Scale Growth, Completion Confidence, Notes |
| Overtaking_Evaluation.xlsx | Summary | 25 | Completed vs Incomplete rates, Mean duration, Confirmation delays, ID switch reassociation summary |
| Overtaking_Evaluation.xlsx | Event Diagnostics | 9 | Tracking health, Lost observations, Exit boundary verification, Continuity status |
| NAYAN_Trajectory_Evaluation.xlsx | Prediction Records | 3039 | Prediction ID, Track ID, Timestamp, Horizon (2.0s), Start/End Positions, Status (MATCHED, TRACK_LOST) |
| NAYAN_Trajectory_Evaluation.xlsx | Accuracy Summary | 35 | ADE, FDE, Endpoint error distributions, Matching rate (43.0%), Class accuracy breakdown |
