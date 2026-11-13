# Directional Warning Generation & Event Management

## 1. Architectural Overview

The **Directional Warning Generation and Event Management Subsystem** represents Stage 4 of the Priority-Based Warning System in the Monocular ADAS System. It serves as the bridge between objective scenario risk scoring (Stage 3) and situational driver alerts / reporting (Stages 5 & 6).

```
                      [Stage 2: Threat Features & Scenarios]
                                        │
                                        ▼
                      [Stage 3: Scientific Risk Scoring & Prioritizer]
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│              STAGE 4: DIRECTIONAL WARNING & EVENT MANAGEMENT                    │
│                                                                                 │
│   ┌───────────────────────────┐         ┌───────────────────────────────────┐   │
│   │    DirectionEstimator     │         │   DirectionalWarningGenerator     │   │
│   │                           │         │                                   │   │
│   │ • Lane corridor bounds    │         │ • Level Decision (HIGH/MED/LOW)   │   │
│   │ • Optical periphery       │         │ • Fast-track TTC <= 1.5s bypass   │   │
│   │ • Center FOV fallback     │────────▶│ • Escalation Confirmation (2 fr)  │   │
│   │ • Temporal filter (2 fr)  │         │ • De-escalation Hysteresis (0.5s) │   │
│   │ • Rejection of rear zones │         │ • Track loss grace period (0.4s)  │   │
│   │                           │         │ • Duplicate Event Suppression     │   │
│   └───────────────────────────┘         └───────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 [Structured Warnings & Persistent Warning Events]
                     ├── WarningOutput (Frame-level alerts)
                     └── WarningEvent  (Lifecycle-tracked entities)
```

The system operates under three guiding principles:
1. **Physical Grounding**: A forward-facing monocular camera cannot sense rear or blind-spot obstacles; directional warnings strictly reflect the physical camera field of view.
2. **Measurement Fidelity**: Unassessed threats or invalid measurements are never fabricated into risk scores or defaulted to LOW.
3. **Driver Alert Stability**: The system applies temporal confirmation to reject single-frame false alarms and asymmetric hysteresis (rapid escalation, delayed de-escalation) to prevent warning flicker.

---

## 2. Camera-Relative Coordinate Frame and Direction Definitions

All directional assignments are established with respect to the forward-facing camera installed along the ego vehicle's longitudinal centerline:

$$\begin{aligned}
+X &: \text{Lateral axis pointing towards the right side of the image} \\
+Y &: \text{Vertical axis pointing downwards in the image} \\
+Z &: \text{Optical depth axis extending forward along the vehicle heading}
\end{aligned}$$

### Supported Directional Zones

| Direction | Image Region & Geometric Condition | Operational Meaning |
| :--- | :--- | :--- |
| `FRONT` | Bounding box center within ego lane corridor $[x_{\text{left}} - 0.1W_{\text{lane}}, x_{\text{right}} + 0.1W_{\text{lane}}]$ | Directly ahead in ego vehicle travel path |
| `FRONT_LEFT` | Bounding box center to the left of the left ego lane boundary ($x < x_{\text{left}}$) | Ahead and laterally positioned in adjacent left lane or shoulder |
| `FRONT_RIGHT` | Bounding box center to the right of the right ego lane boundary ($x > x_{\text{right}}$) | Ahead and laterally positioned in adjacent right lane or shoulder |
| `LEFT` | Extreme lateral boundary ($x \le 0.15W_{\text{img}}$) and lower image ($y_{\text{bottom}} > 0.50H_{\text{img}}$) | Flanking / alongside ego vehicle on the driver's left side |
| `RIGHT` | Extreme lateral boundary ($x \ge 0.85W_{\text{img}}$) and lower image ($y_{\text{bottom}} > 0.50H_{\text{img}}$) | Flanking / alongside ego vehicle on the passenger's right side |
| `UNKNOWN` | Invalid bounding box, missing dimensions, or out-of-frame coordinates | Spatial direction indeterminate |

---

## 3. Direction Assignment Algorithm

The `DirectionEstimator` executes a three-tier spatial assignment with temporal hysteresis:

```
                  Bounding Box (cx, by) + Lane Info
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Lateral Periphery?]             [Lane Geometry Valid?]
     cx <= 0.15W or cx >= 0.85W?         rx > lx and conf >= 0.2?
                 │                               │
         YES ┌───┴───┐ NO                YES ┌───┴───┐ NO
             │       │                       │       │
             ▼       ▼                       ▼       ▼
        LEFT / RIGHT  (Continue)      Ego Corridor?  Center FOV?
                                      [lx-tol, rx+tol] [0.40W, 0.60W]
                                             │               │
                                             ▼               ▼
                                         FRONT / FL / FR  FRONT / FL / FR
                                             │               │
                                             └───────┬───────┘
                                                     │
                                                     ▼
                                          [Temporal Filter]
                                        Candidate == Confirmed OR
                                        Occurrences >= 2 in history?
                                                     │
                                             ┌───────┴───────┐
                                             ▼               ▼
                                         Confirmed        Hold Prior
```

1. **Periphery Check**: Evaluates whether an obstacle is entering or departing the immediate vehicle flanks ($x \le 0.15W$ or $x \ge 0.85W$ with $y_{\text{bottom}} > 0.50H$).
2. **Lane-Aware Corridor Check**: When polynomial lane boundaries are available and confident ($\ge 0.20$), the left and right lane boundaries ($x_{\text{left}}, x_{\text{right}}$) are evaluated at the obstacle's base contact point ($y_{\text{bottom}}$). An adaptive tolerance margin (10% of lane width) accounts for perspective foreshortening.
3. **Geometric Fallback**: When lane detection is unavailable, obscured, or of low confidence, the estimator falls back to the camera optical center corridor ($[0.40W, 0.60W]$).
4. **Temporal Stability Buffer**: Raw candidate directions are passed through a sliding history buffer of length 5. A direction change requires at least 2 consecutive confirming observations (`min_confirmation_frames = 2`). Transient single-frame bounding box noise is filtered out.

---

## 4. Absence of Rear / Blind-Spot Warnings and Technical Justification

> [!IMPORTANT]
> The monocular ADAS system strictly refrains from generating `REAR`, `REAR_LEFT`, `REAR_RIGHT`, or blind-spot warnings.

### Technical & Safety Justification
- **Physical Monocular Inability**: A forward-facing windshield-mounted camera has a forward field of view (typically $50^\circ$ to $70^\circ$ horizontal FOV). Obstacles directly behind the vehicle or in adjacent blind spots are physically occluded by vehicle pillars and body panels.
- **Safety Standard Compliance (ISO 17387 / ISO 22839)**: Blind Spot Detection (BSD) and Lane Change Decision Aid Systems (LCDAS) require verified rear-quarter sensors (radar, ultrasonic, or dedicated rear/side cameras). Fabricating blind-spot warnings from dead-reckoning trajectory extrapolations from a forward camera violates safety standards.
- **Explicit Invalidation**: If an object completely exits the camera frame lateral boundaries, its direction is not guessed; it is retained during a brief grace period ($0.4\text{ s}$) and subsequently expired if not re-observed.

---

## 5. Warning Level Decision Criteria

Warning levels are assigned per-object based on scientific risk scores, scenario states, physical indicators, and measurement validity:

```
                                  Threat Assessment
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
       [Invalid or Receding?]                           [Risk Score & Urgency]
   INSUFFICIENT / Receding Diverging?                             │
                  │                                               │
          YES ┌───┴───┐ NO                                        │
              ▼       │                                           │
         UNASSESSED   └───────────────────────────┬───────────────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       ▼
                   [HIGH Level]            [MEDIUM Level]            [LOW Level]
               Score >= 0.70 OR         Score in [0.40, 0.70) OR   Score in [0.20, 0.40) OR
               Critical TTC <= 1.8s OR  Developing Conflict OR     Adjacent Lane Encroach
               Conflict State           Moderate TTC <= 3.0s
```

### Decision Matrix

| Warning Level | Primary Risk Score | Physical / Scenario Triggers | Validity Requirement |
| :--- | :--- | :--- | :--- |
| `HIGH` | $\text{Score} \ge 0.70$ | $TTC \le 1.8\text{ s}$ with closing speed and lane overlap/conflict, OR active conflict states (`REAR_END_CONFLICT`, `PEDESTRIAN_CROSSING_CONFLICT`, `CONFIRMED_CUT_IN`, `CYCLIST_CROSSING_CONFLICT`) | `Reliability != UNRELIABLE`, indicators calibrated |
| `MEDIUM` | $0.40 \le \text{Score} < 0.70$ | Developing conflict states (`POTENTIAL_CUT_IN`, `PEDESTRIAN_APPROACHING_PATH`, `CYCLIST_ENTERING_LANE`, `CYCLIST_LEAD_CONFLICT`, `REAR_END_POTENTIAL`) with closing speed, OR moderate urgency ($TTC \le 3.0\text{ s}$ and closing) | `Reliability >= MODERATE` |
| `LOW` | $0.20 \le \text{Score} < 0.40$ | Advisory threats, `ADJACENT_LANE_ENCROACHING`, or lateral drift toward ego lane | Basic tracking confidence |
| `UNASSESSED` | None or $< 0.20$ | `INSUFFICIENT_INDICATORS`, `UNSUPPORTED_SCENARIO`, or safe receding vehicle | Measurements absent or uncalibrated |

---

## 6. Urgency Override Mechanism

In safety-critical scenarios (e.g., sudden lead-vehicle brake slam or sharp cut-in), an obstacle may rapidly decelerate before statistical multi-frame indicators fully reflect high risk.

The **Urgency Override** guarantees immediate driver warning:
1. **Critical TTC Override ($TTC \le 1.8\text{ s}$)**: When $TTC \le 1.8\text{ s}$, relative closing speed exceeds $0.2\text{ m/s}$, and there is trajectory conflict or lane overlap $> 15\%$, the candidate warning level is forced to `HIGH` regardless of baseline model score.
2. **Fast-Track Activation ($TTC \le 1.5\text{ s}$)**: Bypasses the 2-frame candidate confirmation queue to issue a `HIGH` warning on the very first frame of critical detection.

---

## 7. Temporal Confirmation and Flicker Rejection

Monocular perception pipelines frequently suffer from single-frame detection false positives (e.g., shadows, roadside signs misclassified as pedestrians).

### Multi-Frame Confirmation Protocol
- **Candidate Buffer**: When a track first triggers a warning threshold (LOW, MEDIUM, or HIGH), it enters `_candidate_buffer[(track_id, scenario)]`.
- **Confirmation Requirement**: The warning must persist for at least `confirmation_frames = 2` and span `confirmation_duration_s = 0.15s` in video presentation time before an active warning is dispatched.
- **Accurate Timestamps**: Once confirmed, the event records both its true origin timestamp (`first_detected_timestamp`) and the confirmation timestamp (`activation_timestamp`).

---

## 8. Escalation vs. De-escalation Asymmetry

Human factors research indicates that drivers require immediate alerting during escalating hazards, but suffer severe annoyance and cognitive fatigue from flapping/flickering warnings that oscillate rapidly between severity tiers.

The warning engine enforces asymmetric state transitions:

```
                            [INACTIVE / CANDIDATE]
                                       │
                         Fast-track OR │ 2 frames
                                       ▼
                              ┌────────────────┐
                ┌────────────▶│   ACTIVE_LOW   │
                │             └────────────────┘
                │               │            ▲
                │      2 frames │            │ Sustained 0.5s lower
                │               ▼            │
                │             ┌────────────────┐
                │             │ ACTIVE_MEDIUM  │
                │             └────────────────┘
                │               │            ▲
                │      2 frames │            │ Sustained 0.5s lower
                │   (or TTC<=1.5s)           │
                │               ▼            │
                │             ┌────────────────┐
                │             │  ACTIVE_HIGH   │
                │             └────────────────┘
                │                      │
                │                      │ Sustained 0.8s low/zero
                │                      ▼
                └──────────────── [ RESOLVED ]
```

### 1. Rapid Escalation
- Progressing from `LOW` $\to$ `MEDIUM` or `MEDIUM` $\to$ `HIGH` requires only 2 confirming frames.
- **Fast-Track Escalation**: If $TTC \le 1.5\text{ s}$, escalation to `HIGH` is instantaneous, avoiding unnecessary driver reaction delay.

### 2. Delayed De-escalation (Hysteresis)
- Stepping down from `HIGH` $\to$ `MEDIUM` or `MEDIUM` $\to$ `LOW` is prohibited until the lower threat state is sustained for at least `deescalation_duration_s = 0.50 s` ($500\text{ ms}$).
- Transient risk dips (e.g., momentary velocity oscillation or tracking jitter) are suppressed, holding the warning at its higher severity tier until safety is established.

---

## 9. Duplicate Warning Suppression

To prevent multiple overlapping alerts or duplicate entries for the same physical threat:
- Each warning event is keyed on a unique tuple: `(track_id, scenario_type)`.
- If an active warning event exists for that key, its existing state is updated in place:
  - `last_updated_timestamp` is advanced.
  - Direction, TTC, risk score, and collision urgency are refreshed.
  - `event_id` (e.g., `warn_rear_end_10_0001`) remains stable throughout the life of the manoeuvre.
- Downstream visualizers receive a clean, deduplicated stream of active warnings.

---

## 10. Warning Event Data Structures and Lifecycle State Transitions

### Lifecycle States (`WarningEventState`)

| State | Definition | Next Transitions |
| :--- | :--- | :--- |
| `INACTIVE` | Threat score below advisory threshold ($< 0.20$) or safe | `CANDIDATE` |
| `CANDIDATE` | Temporal confirmation buffer accumulating frames ($< 2$ frames) | `ACTIVE_*`, `INACTIVE` |
| `ACTIVE_LOW` | Confirmed advisory warning active | `ACTIVE_MEDIUM`, `ACTIVE_HIGH`, `RESOLVED` |
| `ACTIVE_MEDIUM`| Confirmed developing warning active | `ACTIVE_HIGH`, `ACTIVE_LOW`, `RESOLVED` |
| `ACTIVE_HIGH` | Confirmed critical collision warning active | `ACTIVE_MEDIUM`, `RESOLVED` |
| `RESOLVED` | Threat resolved (risk $< 0.20$ sustained for $0.8\text{ s}$ or target departed) | Terminal |
| `EXPIRED` | Track lost beyond stale timeout ($> 2.0\text{ s}$) | Terminal |

### Data Structures

#### `WarningEvent` (Persistent Lifecycle Record)
```python
@dataclass
class WarningEvent:
    event_id: str
    track_id: int
    object_class: str
    scenario: str
    direction: CameraDirection
    current_level: DirectionalWarningLevel
    previous_level: DirectionalWarningLevel
    state: WarningEventState

    first_detected_timestamp: float
    confirmation_timestamp: float | None = None
    activation_timestamp: float | None = None
    last_updated_timestamp: float = 0.0

    escalation_timestamps: list[tuple[float, str, str]]
    deescalation_timestamps: list[tuple[float, str, str]]
    resolution_timestamp: float | None = None
    resolution_reason: str = ""

    risk_score: float | None = None
    risk_model: str = ""
    reliability: str = "PROVISIONAL"
    ttc: float | None = None
    collision_urgency: float | None = None
    is_primary_threat: bool = False
    priority_rank: int | None = None
    evidence: list[str] = field(default_factory=list)
```

#### `WarningOutput` (Frame-Level Dispatch)
```python
@dataclass
class WarningOutput:
    event_id: str
    track_id: int
    object_class: str
    scenario: str
    direction: CameraDirection
    level: DirectionalWarningLevel
    state: WarningEventState
    risk_score: float | None
    risk_model: str
    reliability: str
    ttc: float | None
    collision_urgency: float | None
    is_primary_threat: bool
    priority_rank: int | None
    timestamp: float
    reason: str
    escalated: bool = False
    deescalated: bool = False
```

---

## 11. Multi-Threat Handling and Non-Interference

The system supports multiple simultaneous threats across distinct directions, classes, and scenarios without cross-track interference:
- **Independent State Machines**: Each `(track_id, scenario)` maintains its own candidate buffer, confirmation timestamps, hysteresis timers, and direction estimators.
- **No Shared Flapping**: A sudden de-escalation of a lead car in `FRONT` does not alter the active `HIGH` warning of a crossing pedestrian in `FRONT_LEFT`.
- **Primary Threat Flagging**: Downstream warning renderers can highlight the primary threat (`is_primary_threat=True`, `priority_rank=1`) while rendering secondary advisory warnings with lower visual prominence.

---

## 12. Tracker ID Reuse and Track-Loss Handling

### Temporary Track Loss Grace Period ($0.40\text{ s}$)
- When an object undergoes brief occlusion (e.g. passing behind a pillar or missing 1-2 video frames), deleting the warning immediately causes alert blinking.
- If $0.0 < \Delta t \le 0.40\text{ s}$, the warning event remains active with reason `TEMPORARY_TRACK_LOSS_GRACE`.
- If the track reappears within $0.40\text{ s}$, it continues seamlessly under the existing `event_id`.

### Tracker ID Reuse Detection
- In monocular tracking (ByteTrack), track IDs may be recycled after an object leaves.
- The warning manager tracks `_track_classes[track_id]`. If track ID 20 previously belonged to a `car` and reappears as a `pedestrian`, the previous event is immediately resolved with `resolution_reason = "TRACKER_ID_REUSE_CLASS_CHANGED"`, and a fresh event is instantiated for the pedestrian.
- If a track ID is silent for $> 2.0\text{ s}$ (`id_reuse_gap_seconds`), its state is completely flushed.

---

## 13. Configuration Parameters and Tuning Guide

All operational thresholds are configurable in `configs/default.yaml` under `threat_assessment.directional_warnings`:

```yaml
threat_assessment:
  directional_warnings:
    enabled: true
    direction:
      enabled: true
      min_confirmation_frames: 2
      lateral_band_ratio: 0.15
      ego_corridor_tolerance_ratio: 0.10
    thresholds:
      high_risk_score: 0.70
      medium_risk_score: 0.40
      low_risk_score: 0.20
      critical_ttc_override_s: 1.8
      critical_ttc_fast_track_s: 1.5
    timing:
      confirmation_frames: 2
      confirmation_duration_s: 0.15
      escalation_frames: 2
      deescalation_duration_s: 0.50
      resolution_duration_s: 0.80
      track_loss_grace_s: 0.40
      stale_event_timeout_s: 2.0
```

### Tuning Recommendations
- **Urban Driving (Stop & Go)**: Decrease `critical_ttc_override_s` to $1.5\text{ s}$ and increase `deescalation_duration_s` to $0.60\text{ s}$ to reduce advisory warnings during tight stop-and-go maneuvers.
- **Highway Cruising**: Increase `critical_ttc_override_s` to $2.2\text{ s}$ and `medium_risk_score` to $0.45$ for earlier alerts at high closing speeds.
- **Rough / Bumpy Roads**: Increase `direction.min_confirmation_frames` to 3 to prevent ego-pitch vibration from flapping lateral lane boundaries.

---

## 14. Downstream Integration Touchpoints

1. **Pipeline Execution Flow**:
   - `PerceptionPipeline.process()` calls `_generate_directional_warnings(...)` immediately following `_prioritize_threats(...)`.
   - Populates `"warnings"`, `"directional_warnings"`, and `"warning_events"` in the pipeline frame result.
2. **Stage 5 (Visualizer & HUD)**:
   - Directional warnings consume `CameraDirection` to position visual banners, arrows, and bounding-box color codings:
     - `HIGH`: Crimson red (`#E53935`)
     - `MEDIUM`: Amber yellow (`#FB8C00`)
     - `LOW`: Cyan blue (`#039BE5`)
     - Direction indicators point toward `FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `LEFT`, or `RIGHT`.
3. **Stage 6 (Excel Reporting & Metrics)**:
   - `WarningEvent.to_dict()` provides serialized data for exporting full incident audit logs, including activation times, duration, peak severity, escalation history, and resolution reasons.

