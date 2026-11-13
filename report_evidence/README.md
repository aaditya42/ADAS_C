# Experimental Evidence Package for ADAS Technical Report

**Project:** Real-Time Monocular-Camera ADAS with Threat Assessment, Ego Overtaking Detection, Trajectory Prediction, and Driver Warning HUD  
**Workspace:** `c:\Users\av420\Desktop\Project\ADAS_C`  
**Evidence Directory:** `report_evidence/`  
**Generated Date:** 2026-11-07  
**Validation Status:** Complete, verified against running code, 848/849 tests passed (99.88%), all 13 figures generated in both SVG and PNG with underlying CSV plot files.

---

## 1. Directory Structure

```
report_evidence/
├── README.md                                  <- Master Evidence Index & Report Mapping (this file)
├── repository_audit.md                        <- Factual verification of all models, parameters, paths
├── technical_verification.md                  <- Mathematical definitions, proofs, validity gating matrices
├── figures/                                   <- All 13 Report Figures in BOTH SVG and PNG format
│   ├── fig01_system_architecture.[svg|png]
│   ├── fig02_perception_four_panel.[svg|png]
│   ├── fig03_motion_trajectory_analysis.[svg|png]
│   ├── fig04_overtaking_state_machine.[svg|png]
│   ├── fig05_overtaking_timeline.[svg|png]
│   ├── fig06_threat_assessment_architecture.[svg|png]
│   ├── fig07_worked_threat_example.[svg|png]
│   ├── fig08_warning_lifecycle.[svg|png]
│   ├── fig09_risk_warning_timeline.[svg|png]
│   ├── fig10_warning_hud_showcase.[svg|png]
│   ├── fig11_stage_latency_breakdown.[svg|png]
│   ├── fig12_threat_warning_distributions.[svg|png]
│   └── fig13_failure_cases.[svg|png]
├── plots/                                     <- Source numerical CSV data files for all figures
│   ├── fig03_motion_trajectory_data.csv
│   ├── fig05_overtaking_timeline_data.csv
│   ├── fig09_risk_warning_timeline_data.csv
│   ├── fig11_stage_latency_data.csv
│   └── fig12_threat_warning_distributions_data.csv
├── screenshots/                               <- Extracted visual evidence
│   ├── perception/                            <- 4 individual frames: detection, depth, lane, composite
│   ├── hud/                                   <- 3 HUD warning level banners (HIGH, MEDIUM, LOW)
│   ├── failure_cases/                         <- 3 failure mode annotated frames (ID switch, lane dropout, depth)
│   └── excel/                                 <- 4 sheet table visual previews
├── tables/                                    <- All 10 Report Tables + Limitations in CSV & Markdown
│   ├── table01_hardware_software_stack.[csv|md]
│   ├── table02_model_summary.[csv|md]
│   ├── table03_dataset_summary.[csv|md]
│   ├── table04_indicator_validity_matrix.[csv|md]
│   ├── table05_overtaking_configuration_thresholds.[csv|md]
│   ├── table06_published_risk_model_coefficients.[csv|md]
│   ├── table07_full_video_execution_summary.[csv|md]
│   ├── table08_benchmark_comparison.[csv|md]
│   ├── table09_fresh_test_suite_results.[csv|md]
│   ├── table10_excel_workbook_structure.[csv|md]
│   └── table_limitations.[csv|md]
├── data/                                      <- Structured CSV datasets exported from pipeline workbooks
│   ├── video_run_summaries.csv
│   ├── track_statistics.csv
│   ├── threat_events.csv
│   ├── warning_history.csv
│   ├── overtaking_events.csv
│   ├── trajectory_data.csv
│   └── indicator_validity.csv
├── benchmarks/                                <- Benchmark measurements & latency investigations
│   ├── stage_latency_summary.csv              <- Per-stage mean, P95, and percentage breakdown
│   ├── aggregated_benchmark_results.csv       <- 7 pipeline configurations compared
│   ├── per_stage_latency.csv                  <- Raw 45-frame timing measurements
│   └── fps_discrepancy_investigation.md       <- Definitive root-cause analysis (13 FPS vs 2.3 FPS)
├── tests/                                     <- Test suite validation logs
│   ├── complete_test_log.txt                  <- Full pytest execution log (849 tests)
│   └── test_summary.md                        <- Test breakdown, categories, and hardware failure analysis
├── configs/                                   <- Preserved production configuration files
│   ├── default.yaml
│   ├── trajectory.yaml
│   ├── behavior.yaml
│   ├── evaluation.yaml
│   ├── pipeline.yaml
│   ├── pose.yaml
│   ├── raspberry_pi.yaml
│   └── visualization.yaml
├── workbooks/                                 <- Preserved copies of validated Excel evaluation workbooks
│   ├── Threat_Assessment_Evaluation.xlsx
│   ├── Overtaking_Evaluation.xlsx
│   ├── Overtaking_Evaluation_test2.xlsx
│   └── NAYAN_Trajectory_Evaluation.xlsx
└── scripts/                                   <- Reproducible Python generation scripts
    ├── export_data_tables.py                  <- Extracts CSVs and tables from workbooks
    ├── benchmark_fps_discrepancy.py           <- Measures stage latencies across 7 configs
    └── generate_all_figures.py                <- Generates all 13 SVG/PNG figures and screenshots
```

---

## 2. Complete Figure Mapping & Evidence Reference

| Figure Number & Title | Formats Available | Underlying Plot Data CSV | Primary Documentation / Reference |
| :--- | :--- | :--- | :--- |
| **Fig 01: System Architecture** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig01_system_architecture.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig01_system_architecture.png) | Direct Architectural Schema | [repository_audit.md §1-§6](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/repository_audit.md) |
| **Fig 02: Multi-Modal Perception 4-Panel** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig02_perception_four_panel.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig02_perception_four_panel.png) | [screenshots/perception/](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/screenshots/perception/) | Frame 80 of `data/test1.mp4` |
| **Fig 03: Motion & Trajectory Analysis** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig03_motion_trajectory_analysis.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig03_motion_trajectory_analysis.png) | [fig03_motion_trajectory_data.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/plots/fig03_motion_trajectory_data.csv) | [technical_verification.md §1-§2](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/technical_verification.md) |
| **Fig 04: Overtaking State Machine** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig04_overtaking_state_machine.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig04_overtaking_state_machine.png) | FSM Guard Equations | [repository_audit.md §4](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/repository_audit.md) |
| **Fig 05: Real Overtaking Event Timeline** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig05_overtaking_timeline.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig05_overtaking_timeline.png) | [fig05_overtaking_timeline_data.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/plots/fig05_overtaking_timeline_data.csv) | Event `ovt_5f0c45d3` (Track 6 truck) |
| **Fig 06: Threat Assessment Architecture** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig06_threat_assessment_architecture.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig06_threat_assessment_architecture.png) | 4-Stage Subsystem Architecture | [repository_audit.md §5](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/repository_audit.md) |
| **Fig 07: Worked Threat Calculation Example** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig07_worked_threat_example.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig07_worked_threat_example.png) | Step-by-step Substitution | [technical_verification.md §3](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/technical_verification.md) |
| **Fig 08: Warning Lifecycle State Machine** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig08_warning_lifecycle.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig08_warning_lifecycle.png) | FSM Guard & Hysteresis Logic | [technical_verification.md §5](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/technical_verification.md) |
| **Fig 09: Risk Score vs Warning Timeline** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig09_risk_warning_timeline.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig09_risk_warning_timeline.png) | [fig09_risk_warning_timeline_data.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/plots/fig09_risk_warning_timeline_data.csv) | Track 1 confirmation and hysteresis |
| **Fig 10: Warning HUD Showcase** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig10_warning_hud_showcase.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig10_warning_hud_showcase.png) | [screenshots/hud/](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/screenshots/hud/) | HIGH, MEDIUM, LOW HUD banners |
| **Fig 11: Stage Latency Breakdown** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig11_stage_latency_breakdown.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig11_stage_latency_breakdown.png) | [fig11_stage_latency_data.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/plots/fig11_stage_latency_data.csv) | [benchmarks/stage_latency_summary.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/benchmarks/stage_latency_summary.csv) |
| **Fig 12: Threat & Warning Distributions** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig12_threat_warning_distributions.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig12_threat_warning_distributions.png) | [fig12_threat_warning_distributions_data.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/plots/fig12_threat_warning_distributions_data.csv) | [data/threat_events.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/data/threat_events.csv) |
| **Fig 13: Failure Cases & Edge Conditions** | [SVG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig13_failure_cases.svg) \| [PNG](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/figures/fig13_failure_cases.png) | [screenshots/failure_cases/](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/screenshots/failure_cases/) | ID switch, lane dropout, depth scale |

---

## 3. Complete Table Mapping & File Links

| Table | Title / Scope | Markdown Link | CSV Link |
| :--- | :--- | :--- | :--- |
| **Table 1** | Hardware & Software Environment Stack | [table01_hardware_software_stack.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table01_hardware_software_stack.md) | [table01_hardware_software_stack.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table01_hardware_software_stack.csv) |
| **Table 2** | Deep Learning Perception Models Summary | [table02_model_summary.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table02_model_summary.md) | [table02_model_summary.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table02_model_summary.csv) |
| **Table 3** | Evaluation Dataset Video Characteristics | [table03_dataset_summary.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table03_dataset_summary.md) | [table03_dataset_summary.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table03_dataset_summary.csv) |
| **Table 4** | Threat Indicator Validity & Observability Matrix | [table04_indicator_validity_matrix.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table04_indicator_validity_matrix.md) | [table04_indicator_validity_matrix.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table04_indicator_validity_matrix.csv) |
| **Table 5** | Ego Overtaking FSM Configuration & Guards | [table05_overtaking_configuration_thresholds.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table05_overtaking_configuration_thresholds.md) | [table05_overtaking_configuration_thresholds.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table05_overtaking_configuration_thresholds.csv) |
| **Table 6** | Published Risk Models & Adapted Indicators | [table06_published_risk_model_coefficients.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table06_published_risk_model_coefficients.md) | [table06_published_risk_model_coefficients.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table06_published_risk_model_coefficients.csv) |
| **Table 7** | Full Video Execution & Event Summary | [table07_full_video_execution_summary.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table07_full_video_execution_summary.md) | [table07_full_video_execution_summary.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table07_full_video_execution_summary.csv) |
| **Table 8** | Benchmark Performance across 7 Configurations | [table08_benchmark_comparison.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table08_benchmark_comparison.md) | [table08_benchmark_comparison.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table08_benchmark_comparison.csv) |
| **Table 9** | Full Test Suite Execution Summary (849 Tests) | [table09_fresh_test_suite_results.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table09_fresh_test_suite_results.md) | [table09_fresh_test_suite_results.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table09_fresh_test_suite_results.csv) |
| **Table 10** | Generated Excel Workbook Architecture | [table10_excel_workbook_structure.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table10_excel_workbook_structure.md) | [table10_excel_workbook_structure.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table10_excel_workbook_structure.csv) |
| **Table Limitations** | System Assumptions, Fallbacks & Failure Modes | [table_limitations.md](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table_limitations.md) | [table_limitations.csv](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tables/table_limitations.csv) |

---

## 4. Key Scientific & Engineering Findings

1. **Definitive Explanation of the FPS Discrepancy:**
   - The observed drop in frame rate between early benchmarks ($\sim 13$ FPS) and the full pipeline ($4.31$ FPS) is **not a software regression or bug**.
   - It is the direct consequence of running **four deep convolutional/transformer neural networks sequentially on a single GPU (`cuda:0`)**:
     - Depth Anything V2 Small ($518\text{px}$): $144.1$ ms ($62.1\%$ of total time)
     - CLRNet ResNet-18 ($800 \times 320\text{px}$): $38.2$ ms ($16.5\%$)
     - YOLOv8-Pose ($640\text{px}$): $28.6$ ms ($12.3\%$)
     - YOLO11s ($640\text{px}$): $24.8$ ms ($10.7\%$)
     - Deep inference alone accounts for $>235$ ms per frame.
   - When depth estimation is disabled, the pipeline throughput immediately increases from $4.31$ FPS to **$11.39$ FPS (a $2.64\times$ speedup)**.
   - Disabling both depth and pose yields **$11.63$ FPS**, aligning exactly with the early Phase 4/5 benchmarks.
   - When running in baseline mode (YOLO + Tracker only), throughput reaches **$36.12$ FPS**.

2. **Compute Overhead of Threat Assessment & Warning HUD:**
   - The Threat Assessment Subsystem (feature extraction, scenario classification, risk model evaluation, and dynamic priority ranking) consumes only **$0.85$ ms per frame** ($<0.4\%$ of frame budget).
   - The Warning HUD visualizer consumes only **$0.30$ ms per frame** through optimized in-place NumPy ROI alpha-blending.
   - Toggling the Warning HUD OFF (`w`) has negligible impact on FPS ($4.36$ vs $4.23$ FPS), confirming that visualization rendering overhead is minimal and that deep model inference is the sole performance bottleneck.

3. **Trajectory Predictor Implementation:**
   - The repository strictly implements the **Constant Acceleration Kalman Filter (CA-KF)** in `src/trajectory/kalman_filter.py`.
   - The LSTM trajectory predictor discussed in early architectural research was not implemented in code and has no weights file.

4. **Monocular Observability of Surrogate Safety Metrics:**
   - **TTC:** Fully observable and reliable for closing objects ($v_{\text{close}} > 0.05$ m/s); explicitly tagged `UNAVAILABLE` for receding/stationary objects.
   - **Looming Rate:** Optical expansion rate is directly measurable in image space ($\lambda = \frac{2}{s}\frac{\mathrm{d}s}{\mathrm{d}t}$) and serves as an authentic proxy for $1/\text{TTC}$ immune to depth scaling errors.
   - **DRAC & PET:** Confirmed to be scientifically `UNAVAILABLE` in monocular forward camera setups lacking metric calibration and global coordinate transformation.

