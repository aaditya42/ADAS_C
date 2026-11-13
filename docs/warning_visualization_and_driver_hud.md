# Driver Warning HUD & Independent Visualization Layer

## 1. Architectural Overview

The **Driver Warning HUD** represents Stage 5 of the Priority-Based Warning System in the Monocular ADAS System. It translates upstream multi-threat feature extractions, scenario classifications, scientific risk assessments, and directional warning events into an ergonomic, real-time driver alert interface.

```
                           [Perception Pipeline]
                                     │
                                     ▼
                      [Threat Feature Extraction]
                                     │
                                     ▼
                      [Scenario Classification]
                                     │
                                     ▼
                      [Risk Scoring & Prioritization]
                                     │
                                     ▼
                     [Directional Warning Generation]
                                     │
                                     ▼
                        [Warning Event Manager]
                                     │
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
            [Driver Warning HUD]           [Event Logging & Excel]
            (Stage 5 - Toggleable)          (Stage 6 - Background)
                       │
                       ▼
              [Rendered Display]
```

### Critical Independence Rule
The warning HUD is strictly a **presentation-only visualization layer**. Toggling the HUD OFF hides the on-screen visual banners and object highlights, but **never stops or pauses** background threat feature extraction, scientific risk calculation, dynamic threat prioritization, directional warning generation, event lifecycle state machines, hysteresis timers, or event logging.

---

## 2. Visual Hierarchy and HUD Layout

The visual display follows a multi-tier hierarchy designed to communicate hazard urgency immediately without obstructing the driver's forward road view:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ [TOP STRIP: Speed Limit, Traffic Light, In-Lane Count, Nearest Distance]        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                        ┌─────────────────────────────────┐                      │
│                        │       PRIMARY WARNING PANEL     │                      │
│                        │ [! HIGH RISK !]  FRONT [^]  CAR │  OTHER THREATS (2)   │
│                        │ ID: 101 | REAR-END | TTC: 1.2s  │  2. HIGH FL PED TTC:1│
│                        │ Imminent collision risk         │  3. MED  FR CYC TTC:2│
│                        └─────────────────────────────────┘                      │
│                                                                                 │
│                                                                                 │
│                                  ┌─────────────┐                                │
│                                  │ [HIGH FRONT]│                                │
│                                  │┌───     ───┐│                                │
│                                  ││ Lead Car  ││                                │
│                                  │└───     ───┘│                                │
│                                  └─────────────┘                                │
│                                                                                 │
│                                                     [FPS: 58.2 | Frame 1240]    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Visual Tiers

| Warning Tier | Visual Characteristics | Color Scheme | Display Height | Fields Shown |
| :--- | :--- | :--- | :--- | :--- |
| **`HIGH`** | Prominent warning panel with bold crimson border and 6px left accent bar | Crimson (`#E53935` / BGR `(40, 40, 225)`) | $74\text{ px}$ | Critical Header, Direction Symbol & Label, Object Class, Track ID, Scenario, TTC, Reliability, Short Threat Reason |
| **`MEDIUM`** | Moderately prominent amber panel with 6px left accent bar | Amber (`#FB8C00` / BGR `(0, 140, 255)`) | $70\text{ px}$ | Caution Header, Direction, Object Class, Track ID, Scenario, TTC (if valid), Reliability, Short Threat Reason |
| **`LOW`** | Compact cyan advisory panel | Cyan (`#039BE5` / BGR `(220, 160, 0)`) | $56\text{ px}$ | Advisory Header, Direction, Object Class, Track ID, Scenario, Contextual Note |
| **`NEUTRAL`** | Slim unobtrusive status badge | Charcoal Grey (`#757575` / BGR `(140, 140, 140)`) | $22\text{ px}$ | Active Monitoring Status (`0 THREATS`) |
| **`UNASSESSED`** | Excluded from driver warnings | None | $0\text{ px}$ | Never rendered as LOW; omitted from active driver warnings |

---

## 3. Highest-Priority Threat Selection

The HUD selects the primary active threat deterministically using the structured outputs from Stages 3 and 4:

$$\text{SortKey}(w) = \Big( -\text{LevelOrder}(w.\text{level}),\; \mathbb{I}_{[\neg w.\text{is\_primary}]},\; w.\text{rank},\; w.\text{ttc},\; \mathbb{I}_{[w.\text{id} \neq \text{prev\_id}]},\; w.\text{track\_id} \Big)$$

1. **Warning Severity**: Active `HIGH` warnings strictly precede `MEDIUM`, which precede `LOW`.
2. **Prioritizer Identity**: Honors `is_primary_threat` and `priority_rank` established by Stage 3.
3. **Collision Urgency**: For threats with identical level and rank, the lower valid TTC takes precedence.
4. **Persistence Tie-Breaking**: When two threats share equal priority and urgency, the threat currently occupying the primary panel is retained, eliminating frame-to-frame visual flapping.
5. **No Recalculation**: The visualizer never recalculates risk scores, TTC, or directions.

---

## 4. Secondary Threats Stack

When multiple concurrent warnings are active:
- Up to `max_secondary_warnings` (default: 3) secondary threats are rendered in a compact vertical stack along the upper-right canvas (`x = w - 272`).
- Each row displays: Rank index, 3-letter level abbreviation (`HIG`, `MED`, `LOW`), direction, uppercase object class, and valid TTC (`TTC: 2.8s` or `TTC: N/A`).
- Rows utilize in-place ROI alpha blending ($0.80$) with a thin left accent bar matching the threat's severity color.

---

## 5. Object-to-HUD Visual Association

When `highlight_threat_object` is enabled:
- The target object's bounding box is resolved from `ctx["threat_features"]` or `ctx["perceived"]`.
- Distinct stylized corner brackets ($c_{\text{len}} \in [8, 24]\text{ px}$, thickness $2\text{ px}$ for HIGH, $1\text{ px}$ for MEDIUM/LOW) frame the vehicle or obstacle without occluding its interior.
- A compact directional threat pill is anchored above the top-left corner of the box: e.g. `[! HIGH FRONT !]`.

---

## 6. Directional Visual Indicators

All directions are rendered with unambiguous paired text and geometric symbols:

| Direction | Paired Visual Indicator | Camera Orientation |
| :--- | :--- | :--- |
| `FRONT` | `FRONT [^]` | Directly ahead along vehicle heading |
| `FRONT_LEFT` | `FRONT-LEFT [<^]` | Ahead and offset toward driver's left lane/shoulder |
| `FRONT_RIGHT`| `FRONT-RIGHT [>^]` | Ahead and offset toward passenger's right lane/shoulder |
| `LEFT` | `LEFT [<-]` | Entering or flanking vehicle's left lateral periphery |
| `RIGHT` | `RIGHT [->]` | Entering or flanking vehicle's right lateral periphery |
| `UNKNOWN` | `UNKNOWN [?]` | Direction indeterminate |

> [!NOTE]
> Rear direction indicators (`REAR`, `REAR_LEFT`, `REAR_RIGHT`) are physically unsupported by a forward-facing camera and are strictly omitted.

---

## 7. Keyboard Shortcut & Runtime Toggle

### Dedicated Keybinding: `W` / `w`
- **Key**: Pressing `w` (or `W`) toggles the Warning HUD layer between `ON` and `OFF`.
- **Feedback Banner**: When toggled, a high-contrast notification badge is displayed for $1.5\text{ s}$ near the bottom of the canvas:
  - `[WARNING HUD: ON]` (Bright green border)
  - `[WARNING HUD: OFF]` (Amber/orange border)

### Keyboard Shortcut Reference

| Key | Layer Toggled | Independent Operational Effect |
| :---: | :--- | :--- |
| **`W` / `w`** | **Driver Warning HUD** | **Hides/shows warning panel and threat brackets. Background risk calculation continues normally.** |
| `O` / `o` | Overtaking Layer | Hides/shows overtaking cards and banners. Background cycle detection continues. |
| `D` / `d` | Depth Layer | Toggles Depth-Anything false-color overlay. |
| `L` / `l` | Lane Layer | Toggles polynomial lane boundary overlay. |
| `P` / `p` | Perceived Objects Layer | Toggles 3D distance and speed labels. |
| `R` / `r` | Baseline Risk Layer | Toggles baseline risk badges. |
| `S` / `s` | Sign Layer | Toggles traffic sign detection bounding boxes. |
| `T` / `t` | Trajectory Layer | Toggles future trajectory prediction polylines. |
| `H` / `h` | Top HUD Strip | Toggles scene context status bar. |
| `F` / `f` | FPS Layer | Toggles frame rate counter. |
| `Q` / `q` | Quit | Gracefully closes video source and exports reports. |

---

## 8. Configuration Specification

Configurable under `visualization.warning` in `configs/default.yaml`:

```yaml
visualization:
  enabled: true
  layers:
    - depth
    - lane
    - perceived
    - risk
    - sign
    - trajectory
    - overtaking
    - warning      # Stage 5 Warning HUD Layer
    - hud
    - fps

  warning:
    enabled: true                     # Initial startup state (overridable via 'w')
    show_primary_warning: true        # Show prominent center warning panel
    show_secondary_warnings: true      # Show compact secondary threat list
    max_secondary_warnings: 3         # Maximum secondary rows to display
    show_direction_indicator: true    # Show [^], [<^], [>^] symbols
    highlight_threat_object: true     # Draw target brackets and badge on vehicle
    show_ttc: true                    # Show TTC when valid (omits or displays N/A)
    show_risk_score: false            # Hide raw model score to prevent driver distraction
    show_track_id: true               # Display track ID
    show_scenario: true               # Display scenario name
    show_reliability: true            # Display indicator reliability grade
    neutral_status_indicator: true    # Show slim status pill when 0 threats active
    toggle_notification_duration: 1.5 # Duration (seconds) of toggle feedback banner
```

### Precedence Between Configuration & Runtime Toggle
1. **Startup**: Configuration `visualization.warning.enabled` determines the initial visualizer state.
2. **Runtime**: Pressing `w` flips `warning_layer.enabled` directly inside the `Renderer` name map.
3. **Session Persistence**: Runtime toggling is stateful and persists until manually toggled again or until session termination.

---

## 9. Performance Overhead & Optimization

The layer implements optimized **in-place ROI alpha-blending** (`_alpha_blend_rect`), completely avoiding expensive full-frame `canvas.copy()` operations:

- **HUD ON Latency**: **1.82 ms/frame** ($548\text{ FPS}$)
- **HUD OFF Latency**: **1.52 ms/frame** ($656\text{ FPS}$) *(baseline frame clone)*
- **Net Drawing Overhead**: **0.30 ms**
- **HUD OFF Drawing Overhead**: **0.00 ms** (immediate return without drawing)

---

## 10. Verification & Test Coverage

### Test Results (`tests/test_warning_hud.py`)
- **HUD Rendering (12/12 passed)**:
  - `test_primary_threat_selection_hierarchy`: Verified HIGH > MEDIUM > LOW and UNASSESSED exclusion.
  - `test_primary_threat_persistence_tie_breaker`: Verified stable primary threat retention without frame flapping.
  - `test_render_high_warning_panel`: Verified prominent panel rendering with crimson accent.
  - `test_render_medium_and_low_warning_panels`: Verified caution and advisory panel themes.
  - `test_render_neutral_status_when_no_active_warnings`: Verified neutral pill rendering and clean frame when disabled.
  - `test_missing_and_invalid_information_handling`: Verified TTC=None renders as N/A, long text truncates gracefully.
  - `test_secondary_threats_stack_limit`: Verified secondary rows truncate to configured maximum.
  - `test_target_object_highlight`: Verified corner bracket and tag placement on target bounding box.
  - `test_renderer_toggle_warning_layer`: Verified keyboard toggle via `"warning"` and `"warning_hud"`.
  - `test_hud_disabled_renders_nothing`: Verified exact pixel equality when HUD is disabled.
  - `test_critical_requirement_background_processing_independence`: Verified background threat assessments, warning generator, and event log continue updating when HUD is OFF.
  - `test_layer_independence_with_overtaking`: Verified non-interference with overtaking layer.

### Full Test Suite
- **Threat & Warning Subsystem**: **54 tests passed** across all 5 stages.
- **Repository Regression Suite**: **828 tests passed** with zero regressions.

---

## 11. Known Limitations

1. **Daytime High-Glare Scenarios**: Bright windshield reflections may slightly reduce contrast on semi-transparent backgrounds; text uses high-contrast opaque borders to maintain legibility.
2. **Extreme Occlusion**: If a primary target vehicle is completely obscured by another large vehicle, corner brackets are suppressed, while the top HUD panel continues alerting based on the vehicle's last known track trajectory.

