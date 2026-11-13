# NAYAN ADAS — Trajectory Prediction & Evaluation Subsystem

## 1. Overview & Architecture

The **Trajectory Prediction & Evaluation Subsystem** provides monocular 2D forward trajectory forecasting, future-position ground truth matching, quantitative accuracy evaluation (ADE, FDE, Endpoint Error), and multi-sheet Excel reporting for tracked traffic participants.

It seamlessly integrates into the NAYAN perception pipeline without introducing secondary neural network models or modifying existing YOLO detection, ByteTrack tracking, MiDaS depth estimation, TTC estimation, or collision warning logic.

```
                    Video Input (1080p / 720p)
                               ↓
                   YOLOv8 Object Detection
                               ↓
                   ByteTrack Multi-Object Tracking
                               ↓
                Position Stabilization & Extraction
                    (Bottom-Center / Center)
                               ↓
                 Trajectory History Management
           (Time-indexed ring buffer per track instance)
                               ↓
                 Continuous State Estimation
            (CA-KF: 6-state / CV-KF: 4-state Baseline)
                               ↓
            Future Spatial Trajectory Forecasting
          (Configurable Horizon T_h & Update Interval dt)
                               ↓
                 Trajectory Visualization Layer
        (Predicted polyline, endpoint crosshair, error line)
                               ↓
              Ground Truth Future-Position Matching
       (Time-indexed matching with strict instance identity)
                               ↓
            Evaluation Metrics: ADE, FDE, Endpoint Error
                               ↓
             Multi-Sheet OpenPyXL Excel Workbook Export
```

---

## 2. State Representation & Prediction Algorithm

### Primary Predictor: Constant Acceleration Kalman Filter (CA-KF)

State vector $\mathbf{x} \in \mathbb{R}^6$:
$$\mathbf{x} = \begin{bmatrix} x & y & v_x & v_y & a_x & a_y \end{bmatrix}^T$$

- Position: $(x, y)$ in image pixel space.
- Velocity: $(v_x, v_y)$ in pixels per second.
- Acceleration: $(a_x, a_y)$ in pixels per second squared.

Continuous-time state transition matrix $F(\Delta t)$:
$$F(\Delta t) = \begin{bmatrix}
1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

Process noise $Q(\Delta t)$ is derived using the Continuous White Noise Acceleration (CWPA) model with spectral density $q = \sigma_a^2$. The measurement update incorporates observations $\mathbf{z} = [x, y]^T$ with measurement noise covariance $R = \sigma_z^2 I_2$ using the **Joseph stabilized form**:
$$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T$$
guaranteeing positive semi-definiteness and numerical stability across variable $\Delta t$.

### Baseline Predictor: Constant Velocity Kalman Filter (CV-KF)

State vector $\mathbf{x} \in \mathbb{R}^4$:
$$\mathbf{x} = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^T$$

- Assumes acceleration is implicitly zero.
- Uses Continuous White Noise Velocity (CWPV) model.
- **Trade-off Analysis:**
  - *When CV-KF performs better:* Steady-speed cruising on straight highway lanes where detection noise could otherwise lead CA-KF to fit spurious acceleration spikes.
  - *When CA-KF performs better:* Braking, accelerating from stops, or curved maneuvers where constant velocity systematically undershoots ground truth positions.

---

## 3. Coordinate System & Position Reference

- **Image Coordinate Space:** Origin $(0, 0)$ is at the top-left corner; $+X$ extends to the right; $+Y$ extends downwards toward the bottom of the frame.
- **Reference Point:**
  - `bottom_center` (Default): $\left(\frac{x_1 + x_2}{2}, y_2\right)$, aligning with the tire-road contact point in driving scenarios, which minimizes perspective distortion and bounding-box height breathing.
  - `center`: $\left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$, useful for airborne or pedestrian tracking.

---

## 4. Track Identity & Safety Mechanisms

1. **Continuous Track Instance ID (`{track_id}_inst{counter}`):**
   - When a track is continuous, it retains its instance identity (e.g. `12_inst1`).
   - If ByteTrack loses detection for longer than `maximum_track_gap_seconds` (default: 0.5s), the subsystem increments the counter to `12_inst2` and resets the Kalman filter.
2. **Prevention of Cross-Contamination & Track Re-use:**
   - Predictions are permanently tagged with `track_instance_id`.
   - Matching will **only** compare predictions against actual observations sharing the **exact** same `track_instance_id`.
   - If a track ID is recycled by ByteTrack for a new physical object, pending predictions from the prior instance are marked `INVALID` instead of calculating false errors.
3. **Safe Linear Interpolation:**
   - Interpolation between observed timestamps is permitted only between adjacent continuous frames within the same continuous episode ($\Delta t \le \text{maximum\_track\_gap\_seconds}$).
4. **Frozen Predictions:**
   - Once generated at frame timestamp $t_0$, prediction records are strictly immutable. They are never updated or retroactively altered by future observations.

---

## 5. Quantitative Evaluation Metrics

For each prediction emitted at $t_0$ for target timestamp $t_{\text{target}} = t_0 + T_h$:

1. **Endpoint Error (Pixel Displacement at Horizon):**
   $$\text{Endpoint Error} = \sqrt{(x_{\text{pred}}(T_h) - x_{\text{actual}}(t_{\text{target}}))^2 + (y_{\text{pred}}(T_h) - y_{\text{actual}}(t_{\text{target}}))^2}$$
2. **Final Displacement Error (FDE):**
   $$\text{FDE} = \text{Endpoint Error}$$
3. **Average Displacement Error (ADE):**
   Calculated across all $N$ discrete intermediate waypoints $\tau_i \in (0, T_h]$:
   $$\text{ADE} = \frac{1}{N} \sum_{i=1}^N \sqrt{(x_{\text{pred}}(\tau_i) - x_{\text{actual}}(t_0 + \tau_i))^2 + (y_{\text{pred}}(\tau_i) - y_{\text{actual}}(t_0 + \tau_i))^2}$$
4. **Matching Success Rate:** Percentage of generated predictions that successfully matched valid ground truth observations.
5. **Threshold Success Rate:** Percentage of matched predictions whose final endpoint error is within `success_threshold_pixels` (default: 20.0 px).

---

## 6. Multi-Sheet Excel Workbook Structure

Exported to `reports/NAYAN_Trajectory_Evaluation.xlsx`:

| Sheet # | Sheet Title | Description |
|---|---|---|
| **Sheet 1** | `Prediction Records` | Complete record of every forecast event: Prediction ID, Track Instance ID, Track ID, Class, Timestamps, Predicted $(X, Y)$, Actual $(X, Y)$, Endpoint Error, ADE, FDE, and Evaluation Status. |
| **Sheet 2** | `Detailed Future Positions` | Granular waypoint records: predicted vs. actual positions, time offsets, displacement errors, and coordinate variances for every intermediate step. |
| **Sheet 3** | `Accuracy Summary` | High-level metrics: total predictions, matched count, match rate %, threshold success rate %, mean/median/min/max/std Endpoint Error, ADE, FDE, and breakdowns by class and horizon. |
| **Sheet 4** | `Performance` | Runtime benchmarking stats: average prediction latency, average evaluation latency, average tracked objects, and total frames processed. |
| **Sheet 5** | `Configuration and Metadata` | Session provenance: video source, resolution, FPS, horizon, update interval, match tolerance, noise parameters, and coordinate reference. |

---

## 7. Configuration & Control

In `configs/trajectory.yaml` or `configs/default.yaml`:

```yaml
trajectory:
  enabled: true                     # Master toggle

  prediction:
    horizon_seconds: 2.0            # Forward prediction horizon
    update_interval: 0.1            # Interval between intermediate waypoints
    model: constant_acceleration    # 'constant_acceleration' or 'constant_velocity'
    history_duration_seconds: 2.0   # Sliding window history buffer
    minimum_history_points: 10      # Warm-up observation threshold
    process_noise_std: 1.0          # Q matrix scaling
    measurement_noise_std: 2.0      # R matrix scaling

  tracking:
    maximum_track_gap_seconds: 0.5  # Detection gap triggering instance increment
    stale_track_timeout_seconds: 1.0# Eviction timeout for departed objects

  coordinates:
    system: image
    position_reference: bottom_center # 'bottom_center' or 'center'

  evaluation:
    enabled: true                   # Ground truth matching and ADE/FDE calculation
    match_tolerance_seconds: 0.05   # Temporal matching window tolerance
    max_records: 10000              # Bounded in-memory record cap
    success_threshold_pixels: 20.0  # Pixel threshold for accuracy classification

  visualization:
    enabled: true                   # Live visual overlay
    show_predicted_path: true
    show_predicted_endpoint: true
    show_actual_endpoint: true
    show_error_line: true
    show_error_label: true

  export:
    enabled: true                   # Export Excel report
    format: excel
    filename: NAYAN_Trajectory_Evaluation.xlsx
    output_dir: ./reports
    export_on_session_end: true
```

### Keyboard Shortcuts (Runtime Interactive Mode)
- Press `t` to toggle the Trajectory overlay layer on/off in real-time.
- Press `q` to terminate the session and trigger automatic Excel report generation.

---

## 8. Critical Limitations & Physical Interpretations

> [!WARNING]
> **Image-Space Error vs. Physical-World Error:**
> The errors reported by this subsystem (ADE, FDE, Endpoint Error) are measured strictly in **2D image pixel space**.
>
> 1. **Perspective Distortion:** An error of 15 pixels for an object 60 meters away corresponds to a physical displacement of several meters, whereas an error of 15 pixels for an object 5 meters in front of the vehicle represents only a few centimeters.
> 2. **Camera Ego-Motion:** In a moving ego-vehicle, apparent motion in image space is the vector sum of object world motion and ego-camera pitch/yaw/forward translation.
> 3. **Bounding-Box Jitter:** Detection bounding boxes fluctuate naturally from frame to frame due to detector confidence and partial occlusions.
> 4. **No Calibrated Collision Claim:** This subsystem provides 2D image-space spatial forecasting. It **does not** constitute a calibrated 3D world-coordinate trajectory or guarantee crash avoidance.
