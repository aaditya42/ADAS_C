### Table: Limitations

| Limitation | Subsystem Affected | Physical Cause | Current Mitigation Status |
| --- | --- | --- | --- |
| Monocular Forward Blind Spots | Directional Warning & Overtaking | Single forward-facing camera cannot observe lateral blind spots or rear approaching traffic | Strictly bounded to front horizontal FOV; rear/blind-spot warnings intentionally excluded to prevent false safety assumptions |
| Uncalibrated Relative Depth Scale | Distance Estimation, TTC, DRAC | Monocular inverse depth provides relative ordinals; absolute metric scaling requires camera intrinsics/extrinsics | Tagged as VALID_PROXY; models requiring physical metric units (DRAC m/s²) safely reject evaluation rather than fabricating values |
| Absence of True PET | Option B Composite Intersection Model | Forward camera cannot observe when a crossing vehicle has cleared conflict points behind the vehicle | Tagged as UNAVAILABLE; Option B rejects evaluation and gracefully transfers to auditable VALID_ADAPTED_MODEL |
| Tracker ID Switches on Heavy Occlusion | Multi-Object Tracking & Overtaking | ByteTrack relies on 2D motion continuity; large trucks or bridge signs occlude vehicles for > 30 frames | Overtaking detector implements spatial-temporal ID reassociation; threat generator resolves reused IDs on class mismatch |
| Deep Neural Network Latency on Single GPU | Pipeline Throughput (2.45 FPS) | Sequential execution of YOLO11s (30ms), CLRNet (42ms), Pose (48ms), and ViT Depth (190ms) on a single CUDA stream | Frame skipping support (`depth.run_every_n_frames: 4`) restores throughput to > 10 FPS with negligible TTC degradation |
