### Table: Published Risk Model Coefficients

| Model Name | Target Scenario | Indicator | Published Coefficient | Availability in Monocular ADAS | Coefficient Preservation Status |
| --- | --- | --- | --- | --- | --- |
| Option A (ADRS 2025) | Car-Following (Longitudinal) | Time-to-Collision (TTC) | 0.25 | AVAILABLE (Proxy inverse depth closure) | EXACT (Unchanged) |
| Option A (ADRS 2025) | Car-Following (Longitudinal) | Speed Variation (Spd) | 0.24 | AVAILABLE (Track centroid velocity variance) | EXACT (Unchanged) |
| Option A (ADRS 2025) | Car-Following (Longitudinal) | Acceleration Noise (AN) | 0.21 | AVAILABLE (Second difference of track positions) | EXACT (Unchanged) |
| Option A (ADRS 2025) | Car-Following (Longitudinal) | Deceleration Rate (DRAC) | 0.16 | UNAVAILABLE in Relative Mode (Requires metric meters) | GATED (Rejects evaluation; no weight redistribution) |
| Option A (ADRS 2025) | Car-Following (Longitudinal) | Lateral Deviation (LD) | 0.14 | AVAILABLE (Normalized lane corridor offset) | EXACT (Unchanged) |
| Option B (Composite 2025) | Intersection / Cut-In | Post-Encroachment Time (PET) | 0.3 | UNAVAILABLE (Cannot compute departure time from front camera) | GATED (Rejects evaluation; no weight redistribution) |
| Option B (Composite 2025) | Intersection / Cut-In | Time-to-Collision (TTC) | 0.3 | AVAILABLE (Optical looming rate) | EXACT (Unchanged) |
| Option B (Composite 2025) | Intersection / Cut-In | Combined Velocity (V_comb) | 0.2 | UNAVAILABLE (No vehicle CAN bus speed) | GATED (Rejects evaluation) |
| Option B (Composite 2025) | Intersection / Cut-In | Deceleration Rate (DRAC) | 0.1 | UNAVAILABLE in Relative Mode | GATED (Rejects evaluation) |
| Option B (Composite 2025) | Intersection / Cut-In | Right-of-Way Violation (ROW) | 0.05 | UNAVAILABLE (No HD Map / Traffic light detector) | GATED (Rejects evaluation) |
| Option B (Composite 2025) | Intersection / Cut-In | Unpredictable Behavior (UB) | 0.05 | AVAILABLE (Heuristic lane change proxy) | EXACT (Unchanged) |
| Adapted Rule-Based Engine | All Monocular Traffic Conflicts | Corridor Overlap + Closing Looming | N/A (Calibrated rules) | AVAILABLE (Fully operational in uncalibrated mode) | EXPLICITLY LABELED (VALID_ADAPTED_MODEL) |
