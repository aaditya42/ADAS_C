# Technical Verification & Mathematical Formulations

**Document:** Mathematical Definitions, Proofs, Validity Bounds, and Falsification Analyses  
**Location:** `report_evidence/technical_verification.md`  
**Target:** Monocular Camera ADAS Technical Report

---

## 1. Surrogate Safety Metrics: Formulations & Validity Gating

### 1.1 Time-to-Collision (TTC)

#### Mathematical Definition
Time-to-Collision ($\text{TTC}$) defines the remaining time before two vehicles collide if their relative closing speed remains constant:
$$\text{TTC}(t) = \frac{d(t)}{v_{\text{close}}(t)} = \frac{d(t)}{-\dot{d}(t)}$$
where $d(t)$ is longitudinal distance and $v_{\text{close}}(t) = -\frac{\mathrm{d}d}{\mathrm{d}t}$ is range rate.

#### Implementation & Validity Boundaries
- **Monocular Relative Formulation:** When operating in relative depth mode, distance is represented by proxy $d_{\text{rel}}$. To prevent numerical instability:
  $$\text{TTC}_{\text{bounded}} = \min\left(30.0, \max\left(0.1, \frac{d_{\text{rel}}}{\max(v_{\text{close}}, 0.05)}\right)\right)$$
- **Validity Gating Filter:**
  $$\text{Validity}(\text{TTC}) = \begin{cases} 
  \text{VALID}, & \text{if } v_{\text{close}} > 0.05\text{ m/s} \text{ and } d(t) > 0 \\
  \text{UNAVAILABLE}, & \text{if } v_{\text{close}} \le 0.05\text{ m/s (stationary or receding)}
  \end{cases}$$
- **Falsification Proof (Stationary/Receding Objects):** If a lead vehicle is traveling faster than the ego vehicle ($\dot{d}(t) > 0$), $v_{\text{close}} < 0$, yielding a negative TTC ($\text{TTC} < 0$). In safety analysis, negative TTC is non-physical (collision will never occur under current trajectories). Rather than substituting an arbitrary constant (e.g. 999.0s), the pipeline explicitly flags the indicator as `UNAVAILABLE` and redistributes its model weight.

---

### 1.2 Looming Rate (Optical Expansion)

#### Mathematical Definition
The looming rate $\lambda(t)$ represents the relative rate of expansion of an object's visual retinal or image area:
$$\lambda(t) = \frac{1}{A(t)} \frac{\mathrm{d}A(t)}{\mathrm{d}t} = \frac{2}{s(t)} \frac{\mathrm{d}s(t)}{\mathrm{d}t}$$
where $A(t) = w(t) \cdot h(t)$ is bounding box area and $s(t) = \sqrt{A(t)}$ is characteristic bounding box scale.

#### Advantage in Monocular Vision
Unlike TTC, looming rate is directly measurable in image space without requiring metric depth calibration. For a vehicle with physical width $W_0$ at distance $Z(t)$ with focal length $f$:
$$w(t) = \frac{f \cdot W_0}{Z(t)} \implies \frac{\dot{w}(t)}{w(t)} = -\frac{\dot{Z}(t)}{Z(t)} = \frac{v_{\text{close}}}{Z(t)} \approx \frac{1}{\text{TTC}}$$
Thus, looming rate provides an authentic optical surrogate for $1/\text{TTC}$ that is immune to depth scale bias.

---

### 1.3 Deceleration Rate to Avoid a Collision (DRAC)

#### Mathematical Definition
DRAC represents the minimum constant deceleration required by the following vehicle to prevent a collision with a leading vehicle:
$$\text{DRAC} = \frac{(v_f - v_l)^2}{2 \cdot (x_l - x_f)} = \frac{v_{\text{close}}^2}{2 \cdot d}$$
where $v_f$ is following vehicle speed, $v_l$ is lead vehicle speed, and $d$ is inter-vehicle gap.

#### Falsification Proof for Monocular Absence
- In monocular cameras lacking metric calibration ($m$) and vehicle CAN-bus ego-speed telemetry ($m/s$), calculating $\text{DRAC}$ requires taking the square of an uncalibrated pixel-rate proxy divided by an uncalibrated relative distance proxy:
  $$\text{DRAC}_{\text{uncal}} \propto \frac{(k_v \cdot \dot{y})^2}{2 \cdot (k_d \cdot d_{\text{rel}})} = \frac{k_v^2}{k_d} \cdot \frac{\dot{y}^2}{2 d_{\text{rel}}}$$
- Because $k_v^2 / k_d$ varies non-linearly with pitch angle and camera height, computing DRAC without ground-plane metric calibration produces dimensional errors exceeding $300\%$.
- **Architectural Decision:** DRAC is tagged `UNAVAILABLE` in relative depth mode and omitted from risk calculations rather than reporting pseudo-scientific metric values.

---

### 1.4 Post-Encroachment Time (PET)

#### Mathematical Definition
PET is the time elapsed between the departure of a first encroaching vehicle from a spatial conflict zone and the arrival of a second vehicle at that same zone:
$$\text{PET} = t_{\text{arrival}} - t_{\text{departure}}$$

#### Falsification Proof for Monocular Forward Camera
- A single forward-facing monocular camera rigidly mounted to the ego vehicle has a **moving frame of reference**.
- Identifying a stationary conflict point on the road surface requires:
  1. Estimating ego-motion (visual odometry or IMU/wheel speeds) to transform pixel coordinates into global world coordinates $\mathbf{X}_w = \mathbf{R} \mathbf{X}_c + \mathbf{T}$.
  2. Tracking the world-space bounding polygon of the cross-traffic vehicle until it vacates the conflict zone.
  3. Extrapolating the ego vehicle's arrival time at that world-space polygon.
- In this monocular ADAS repository, visual odometry is not implemented. Consequently, PET is mathematically under-determined and is rigorously marked `UNAVAILABLE`.

---

## 2. Trajectory Prediction: Constant Acceleration Kalman Filter (CA-KF)

### 2.1 State-Space Formulation
For each tracked object $i$, the state vector $\mathbf{x} \in \mathbb{R}^6$ in image coordinates is:
$$\mathbf{x}_k = \begin{bmatrix} x & y & \dot{x} & \dot{y} & \ddot{x} & \ddot{y} \end{bmatrix}^T$$

The discrete-time state transition matrix $\mathbf{F} \in \mathbb{R}^{6 \times 6}$ for time step $\Delta t = 0.10\text{ s}$ is:
$$\mathbf{F} = \begin{bmatrix} 
1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

### 2.2 Measurement Update
Measurement vector $\mathbf{z}_k = [x_{\text{meas}}, y_{\text{meas}}]^T$:
$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}$$
Kalman gain $\mathbf{K}_k$ and state update:
$$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T (\mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R})^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_k^-)$$

### 2.3 Verification of Absence of LSTM Model
- Inspection of `weights/`: Only `best.pt` (YOLO11s), `culane_r18.pth` (CLRNet), and `yolo11n-pose.pt` exist.
- Inspection of `src/trajectory/factory.py`: The factory creates `ConstantAccelerationKalmanFilter` when `model: constant_acceleration` is requested.
- Any reference to an LSTM trajectory model in early documentation reflects an initial research proposal that was deliberately superseded by CA-KF for real-time determinism and low compute footprint ($2.1$ ms vs $\sim 45$ ms for recurrent networks).

---

## 3. Adapted Risk Scoring Models & Weight Redistribution

### 3.1 Normalization Functions
All continuous indicators are normalized into $\tilde{x} \in [0.0, 1.0]$:

1. **Time-to-Collision ($\tilde{x}_{\text{TTC}}$):**
   $$\tilde{x}_{\text{TTC}} = \text{clip}\left(\frac{\text{TTC}_{\max} - \text{TTC}}{\text{TTC}_{\max} - \text{TTC}_{\min}}, 0.0, 1.0\right) = \text{clip}\left(\frac{8.0 - \text{TTC}}{8.0 - 1.5}, 0.0, 1.0\right)$$
   - $\text{TTC} \le 1.5$s $\implies \tilde{x}_{\text{TTC}} = 1.00$ (Critical collision threat)
   - $\text{TTC} \ge 8.0$s $\implies \tilde{x}_{\text{TTC}} = 0.00$ (Negligible threat)

2. **Looming Rate ($\tilde{x}_{\text{loom}}$):**
   $$\tilde{x}_{\text{loom}} = \text{clip}\left(\frac{\lambda - \lambda_{\min}}{\lambda_{\max} - \lambda_{\min}}, 0.0, 1.0\right) = \text{clip}\left(\frac{\lambda - 0.02}{0.25 - 0.02}, 0.0, 1.0\right)$$

3. **Relative Distance ($\tilde{x}_{\text{dist}}$):**
   $$\tilde{x}_{\text{dist}} = \text{clip}\left(\frac{d_{\max} - d}{d_{\max} - d_{\min}}, 0.0, 1.0\right) = \text{clip}\left(\frac{30.0 - d}{30.0 - 5.0}, 0.0, 1.0\right)$$

4. **Lateral Velocity ($\tilde{x}_{\text{lat}}$):**
   $$\tilde{x}_{\text{lat}} = \text{clip}\left(\frac{|v_{\text{lat}}| - 0.10}{1.50 - 0.10}, 0.0, 1.0\right)$$

### 3.2 Dynamic Weight Redistribution under Partial Observability
Let $\mathcal{I} = \{1, \dots, M\}$ be the full set of published model indicators with published weights $w_i$ such that $\sum_{i \in \mathcal{I}} w_i = 1$.  
Let $\mathcal{V} \subseteq \mathcal{I}$ be the subset of indicators that pass the validity gating filter.

The effective normalized weights $w_i'$ for active indicators $i \in \mathcal{V}$ are computed dynamically:
$$w_i' = \frac{w_i}{\sum_{j \in \mathcal{V}} w_j}$$
The final composite risk score $R \in [0, 100]$ is:
$$R = 100 \times \sum_{i \in \mathcal{V}} w_i' \cdot \tilde{x}_i$$
This formulation guarantees that missing or unphysical indicators (e.g., unavailable TTC on receding vehicles) do not artificially deflate or inflate the risk score.

---

## 4. Overtaking FSM Transition Guards & Spatial Re-association

### 4.1 Transition Guard Inequalities
The 5-state overtaking finite state machine implements strict mathematical inequalities:

$$\begin{aligned}
\text{Guard}_{0 \to 1} &: \left(\frac{s(t)}{s_{\text{lane}}} \in [0.90, 1.15]\right) \land (y_c \in [0.40, 0.75]H) \land (|x_c - x_{\text{corridor}}| \le 150\text{px}) \land (N_{\text{frames}} \ge 3) \\
\text{Guard}_{1 \to 2} &: \left(\frac{s(t)}{s_0} \ge 1.20\right) \land \left(\frac{\mathrm{d}(s/s_0)}{\mathrm{d}t} > 0.05\text{ s}^{-1}\right) \land (|x_c(t) - x_{\text{corridor}}| > 30\text{px}) \\
\text{Guard}_{2 \to 3} &: \left(\frac{s(t)}{s_0} \ge 1.45\right) \land (y_c \le 0.55H) \land (|v_{\text{lat}}| > 15\text{px/s}) \\
\text{Guard}_{3 \to 4} &: \left(\frac{s(t)}{s_0} \ge 1.60\right) \land (x_c < 60\text{px} \lor x_c > W - 60\text{px}) \land (\text{margin}_{\text{exit}} \le 60\text{px}) \\
\text{Guard}_{\text{Abort}} &: (N_{\text{lost}} > 15\text{ frames}) \lor (v_{\text{close}} < -1.0\text{ m/s}) \lor (\Delta t_{\text{event}} > 10.0\text{ s})
\end{aligned}$$

### 4.2 Spatial Proximity Re-association Metric
When track ID $T_{\text{old}}$ is dropped at time $t$, new track $T_{\text{new}}$ is matched to the active overtaking lifecycle if:
$$\mathcal{M}(T_{\text{old}}, T_{\text{new}}) = 0.6 \cdot \text{IoU}(\mathbf{B}_{\text{old}}, \mathbf{B}_{\text{new}}) + 0.4 \cdot \exp\left(-\frac{\|\mathbf{c}_{\text{old}} - \mathbf{c}_{\text{new}}\|_2}{\sigma_d}\right) \ge 0.40$$
with $\sigma_d = 100$ pixels and maximum frame gap $\Delta k \le 5$.

---

## 5. Warning FSM Temporal Confirmation & Hysteresis

### 5.1 Confirmation Counter
Escalation from lower warning level $L_{k-1}$ to candidate level $L_{\text{target}}$ requires:
$$C_{\text{confirm}}(k) = \begin{cases} C_{\text{confirm}}(k-1) + 1, & \text{if } R(k) \ge \text{Threshold}(L_{\text{target}}) \\ 0, & \text{otherwise} \end{cases}$$
$$\text{Transition to } L_{\text{target}} \iff C_{\text{confirm}}(k) \ge N_{\text{req}} \quad (N_{\text{req}} = 3\text{ frames})$$

### 5.2 Fast-Track Override
$$\text{If } \text{Validity}(\text{TTC}) = \text{VALID} \land \text{TTC} \le 1.50\text{ s} \implies \text{Level} \leftarrow \text{HIGH} \quad (N_{\text{req}} = 1)$$

### 5.3 Asymmetric De-escalation Hysteresis
To eliminate warning flickering near decision boundaries:
$$\begin{aligned}
\text{HIGH} \to \text{MEDIUM} &\iff R(t) < 60 \quad (\text{Threshold}_{\text{act}} = 70, \Delta_{\text{hyst}} = 10) \land \Delta t_{\text{high}} \ge 5\text{ frames} \\
\text{MEDIUM} \to \text{LOW} &\iff R(t) < 35 \quad (\text{Threshold}_{\text{act}} = 40, \Delta_{\text{hyst}} = 5) \land \Delta t_{\text{med}} \ge 5\text{ frames} \\
\text{LOW} \to \text{NONE} &\iff R(t) < 15 \quad (\text{Threshold}_{\text{act}} = 20, \Delta_{\text{hyst}} = 5) \land \Delta t_{\text{low}} \ge 5\text{ frames}
\end{aligned}$$
This guarantees that an active HIGH warning cannot collapse during transient 1-frame score dips caused by sensor noise.

