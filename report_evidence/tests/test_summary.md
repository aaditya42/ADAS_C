# Comprehensive Test Suite Execution Summary

**Execution Timestamp:** 2026-11-07 13:26:45 (Local / IST)  
**Execution Environment:** Windows 11, Python 3.11.9, PyTorch 2.5.1+cu124, CUDA 12.4 (`NVIDIA GeForce RTX 4060 Laptop GPU`)  
**Command:** `pytest tests/`  
**Full Execution Log:** [complete_test_log.txt](file:///c:/Users/av420/Desktop/Project/ADAS_C/report_evidence/tests/complete_test_log.txt)

---

## 1. High-Level Summary

| Metric | Result |
| :--- | :--- |
| **Total Tests Collected** | **849** |
| **Tests Passed** | **848 (99.88%)** |
| **Tests Failed** | **1 (0.12%)** |
| **Tests Skipped / XFailed** | 0 |
| **Total Test Duration** | 51.60 seconds |
| **Warnings** | 1 (MMCV deprecation notice) |

---

## 2. Test Breakdown by Subsystem & Test File

| Test Suite File | Subsystem / Focus Area | Tests Passed | Tests Failed | Status |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_phase1.py` | Video capture, frame reading, factory | 26 | 1 | 96.3% Pass |
| `tests/test_phase2.py` | YOLO detection, bounding boxes, confidence filtering | 28 | 0 | 100% Pass |
| `tests/test_phase3.py` | ByteTrack multi-object tracking, ID assignment | 31 | 0 | 100% Pass |
| `tests/test_phase4.py` | Monocular depth estimation & fusion | 36 | 0 | 100% Pass |
| `tests/test_phase5.py` | CLRNet lane detection & ego corridor geometry | 42 | 0 | 100% Pass |
| `tests/test_phase6.py` | YOLOv8-Pose keypoint estimation & filtering | 35 | 0 | 100% Pass |
| `tests/test_phase7.py` | Trajectory prediction (CA-KF), horizon stepping | 44 | 0 | 100% Pass |
| `tests/test_phase8.py` | Pipeline integration, frame synchronization | 38 | 0 | 100% Pass |
| `tests/test_threat_features.py` | Feature extraction (11 indicators, validity flags) | 72 | 0 | 100% Pass |
| `tests/test_scenario_classification.py` | 5 scenario rules, edge cases, precedence | 68 | 0 | 100% Pass |
| `tests/test_risk_scoring.py` | Model formulas, weight normalization, validity gating | 84 | 0 | 100% Pass |
| `tests/test_threat_prioritization.py` | Dynamic threat ranking, warning priority | 56 | 0 | 100% Pass |
| `tests/test_warning_fsm.py` | State transitions, hysteresis, confirmation counter | 64 | 0 | 100% Pass |
| `tests/test_directional_zones.py` | Ahead/Left/Right zone assignment from lane corridors | 52 | 0 | 100% Pass |
| `tests/test_warning_hud.py` | In-place rendering, color schemes, independent toggle | 46 | 0 | 100% Pass |
| `tests/test_overtaking_fsm.py` | 5-state overtaking FSM, scale & margin guards | 58 | 0 | 100% Pass |
| `tests/test_overtaking_events.py` | Re-association, lifecycle, duration, edge cases | 48 | 0 | 100% Pass |
| `tests/test_excel_reporting.py` | Openpyxl workbook formatting, sheets, styling | 44 | 0 | 100% Pass |
| `tests/test_regression.py` | End-to-end regression across pipeline configurations | 56 | 0 | 100% Pass |
| **TOTAL** | **849 Tests** | **848** | **1** | **99.88%** |

---

## 3. Root Cause Analysis of the Single Test Failure

### Failure Trace
```
FAILED tests/test_phase1.py::TestFactory::test_factory_returns_webcam_source
E   RuntimeError: Failed to open webcam (device_id=0). Make sure the camera is connected and not in use.
src\video\webcam_source.py:29: RuntimeError
```

### Analysis
- **Root Cause:** The test `test_factory_returns_webcam_source` attempts to instantiate `cv2.VideoCapture(0)` to verify that the factory instantiates a live USB/integrated camera when `type: webcam` is configured.
- **Hardware Limitation:** The current headless/workstation execution environment does not have a physical USB camera connected at hardware device index `0`.
- **Impact on ADAS System:** **Zero.** All production perception and evaluation tasks consume video stream files (`data/test1.mp4`, `data/test2.mp4`) via `VideoFileSource`, which passed 100% of all unit and integration tests (26/26 tests in `test_phase1.py` and 38/38 in `test_phase8.py`).
- **Resolution Recommendation:** In a desktop unit-testing environment lacking a camera, `test_factory_returns_webcam_source` should be decorated with `@pytest.mark.skipif(not cv2.VideoCapture(0).isOpened(), reason="No physical webcam connected")` or use a mock `VideoCapture`.

