### Table: Overtaking Configuration Thresholds

| Parameter Name | Configured Value | Units | Physical / Algorithmic Meaning |
| --- | --- | --- | --- |
| min_initial_observations | 3.0 | frames | Minimum history observations before a track is registered as candidate ahead |
| scale_growth_start_threshold | 1.2 | ratio (A_curr / A_init) | 20% optical scale expansion triggers transition to OVERTAKING_STARTED |
| scale_growth_passing_threshold | 1.45 | ratio (A_curr / A_init) | 45% optical scale expansion triggers transition to PASSING |
| vertical_drop_ratio | 0.05 | ratio of img_h | Downward centroid displacement confirming ego vehicle is closing |
| ahead_max_y_ratio | 0.65 | ratio of img_h | Upper vertical boundary for initial detection in forward road plane |
| passing_y_ratio | 0.55 | ratio of img_h | Vertical threshold below which vehicle is alongside ego camera |
| exit_margin_pixels | 60.0 | pixels | Boundary buffer for left/right lateral screen exit confirmation |
| min_event_duration_seconds | 0.4 | seconds | Rejection floor for transient artifacts or spurious false triggers |
| start_confirmation_duration | 0.25 | seconds | Temporal persistence hold before locking start timestamp |
| stale_timeout_seconds | 2.0 | seconds | Dropout grace duration before terminating active overtaking attempt |
