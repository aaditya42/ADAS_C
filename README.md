# ADAS Perception System

A modular, pipeline-based Advanced Driver-Assistance perception system built in Python.

## Architecture

- **Modules never call each other directly.** They consume and produce well-defined data structures (schemas).
- **The pipeline orchestrates everything** — each module is a pure function / class with a clear interface.
- **No drawing inside modules.** Only the Visualization layer draws.
- **Abstract base classes** (Detector, Tracker, DepthEstimator, etc.) make implementations swappable via config.
- **Schemas are plain frozen dataclasses** with no business logic.

## Folder Structure

```
src/
  core/           # Pipeline orchestrator, base interfaces
  config/         # YAML/dict configs
  schemas/        # Detection, TrackedObject, PerceivedObject, RiskObject, LaneInfo, TrafficSignInfo, Frame
  utils/          # Logging, timing, drawing helpers (used only by visualization)
  video/          # Frame providers
  detection/      # YOLO + future detectors
  tracking/       # ByteTrack + future trackers
  depth/          # Depth Anything V2 + future depth models
  fusion/         # Object fusion
  memory/         # TrackHistoryManager
  distance/       # Relative + calibrated distance
  motion/         # Relative speed/acceleration
  ttc/            # Time-to-collision
  risk/           # Risk engine
  warning/        # Warning manager
  lane/           # Lane detection
  signs/          # Traffic sign recognition
  scene/          # Scene understanding
  visualization/  # Drawing, display
  evaluation/     # Metrics, ground-truth comparison
  optimization/   # ONNX / TensorRT
main.py
configs/default.yaml
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Traffic Sign Recognition (Phase 13)

Traffic sign detection is **disabled by default**. To enable it:

1. Train or download a YOLO traffic-sign model (e.g. from
   [Roboflow GTSRB](https://universe.roboflow.com/roboflow-58fyf/gtsrb) or
   [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign)).
2. Place the weights file at `weights/traffic_signs_yolo.pt` (or update the
   path in `configs/default.yaml`).
3. Set `signs.enabled: true` in the config.

Class names from the model are mapped to `SignType` using `DEFAULT_MAP`
in `src/signs/yolo_sign_detector.py`.  A numeric suffix (e.g.
`"speed_limit_50"`) is parsed as the sign's `value` field.

## Pose Estimation & Behaviour Classification

Human and animal pose estimation (YOLOv8-Pose, RTMPose, AP-10K HRNet) and
rule-based behaviour classification (fallen, distracted, crossing road,
grazing, lying, etc.) are integrated as pipeline stages.  The
[pose/behaviour subsystem docs](docs/pose_behavior_subsystem.md) cover
architecture, configuration reference, extension guides, benchmark
methodology, and known limitations.
```
