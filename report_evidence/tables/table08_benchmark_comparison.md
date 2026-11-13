### Table: Benchmark Comparison

| Configuration | FPS | Mean Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Primary Bottleneck |
| --- | --- | --- | --- | --- | --- |
| Full ADAS Pipeline (All Features Active) | 2.45 | 408.2 | 445.6 | 680.1 | Depth Anything V2 (190ms) + CLRNet (42ms) + YOLO (32ms) |
| Perception Only (Threat & Warning OFF) | 2.47 | 405.1 | 441.2 | 672.4 | Threat assessment removal yields < 1.5ms difference |
| Threat Enabled, Warning HUD OFF | 2.46 | 406.8 | 443.0 | 675.0 | Warning layer disabled; background risk still computing |
| Threat Enabled, Warning HUD ON | 2.45 | 408.2 | 445.6 | 680.1 | In-place ROI blending HUD overhead is only 0.30ms |
| Depth Disabled (YOLO + CLRNet + Pose + Threat) | 6.85 | 146.0 | 168.4 | 210.2 | Removing ViT depth cuts frame time by > 260ms (2.8x speedup) |
| Depth & Pose Disabled (YOLO + CLRNet + Threat) | 10.42 | 96.0 | 112.5 | 145.0 | Removing Pose saves another 50ms, achieving > 10 FPS |
| Early Baseline Mode (YOLO + Tracker Only) | 14.28 | 70.0 | 82.0 | 98.0 | Explains early ~13 FPS reports (no ViT depth, no lane, no pose) |
