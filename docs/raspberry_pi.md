# ADAS on Raspberry Pi

## Prerequisites

- Raspberry Pi 4 or 5 (2 GB+ RAM recommended)
- Raspberry Pi OS Bookworm (64-bit)
- Camera Module v2 or v3 (or HQ Camera)
- At least 2 GB free disk space

## 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera
sudo raspi-config nonint do_camera 0
# Or: sudo raspi-config → Interface Options → Camera → Enable

# Install system dependencies
sudo apt install -y \
    libopenblas-dev \
    libcap-dev \
    libatlas-base-dev \
    python3-pip \
    python3-dev \
    git

# Reboot to apply camera changes
sudo reboot
```

## 2. Clone & Install

```bash
git clone <repo-url> ~/adas_c
cd ~/adas_c

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Pi-optimized dependencies
pip install --upgrade pip
pip install -r requirements-pi.txt
```

## 3. Export ONNX Models

On a desktop machine (or directly on Pi, though it is slow):

### YOLO

```bash
pip install ultralytics torch
python -m src.optimization.export_onnx yolo \
    --weights yolo11n.pt \
    --out weights/yolo11n.onnx \
    --imgsz 640
```

### Depth Anything V2

```bash
pip install transformers torch
python -m src.optimization.export_onnx depth_anything \
    --weights depth-anything/Depth-Anything-V2-Small-hf \
    --out weights/depth_anything_v2_small.onnx \
    --imgsz 224
```

Copy the exported `.onnx` files to the Pi's `weights/` directory.

## 4. Run

```bash
# With PiCamera
python main.py --config configs/raspberry_pi.yaml

# With recorded video file for testing
python main.py --config configs/raspberry_pi.yaml --video data/test.mp4
```

**Keyboard controls** (same as desktop):
- `q` quit
- `d` toggle depth layer
- `l` toggle lane layer
- `r` toggle risk layer
- `p` toggle perceived layer
- `h` toggle HUD
- `f` toggle FPS counter

## 5. Benchmark

```bash
python -m src.optimization.benchmark \
    --config configs/raspberry_pi.yaml \
    --frames 300
```

Expected performance (Pi 5, 640×480, depth every 3rd frame):
- YOLO ONNX: ~15-20 FPS
- Depth Anything ONNX (224px): ~5-8 FPS (skip factor 3 → effective 15-24 FPS)
- Full pipeline: ~10-15 FPS

## 6. Configuration Tips

| Setting | Recommended | Rationale |
|---------|-------------|-----------|
| `depth.run_every_n_frames` | 2-4 | Depth is expensive; skip frames |
| `detector.conf_threshold` | 0.3-0.4 | Fewer false positives → less NMS work |
| `tracker.frame_rate` | 15 | Match expected pipeline throughput |
| `visualization.layers` | minimal | Skip `depth` layer overlay on display |

See `configs/raspberry_pi.yaml` for the full reference config.

## 7. TensorRT (Future Path)

For maximum performance on Pi 5:

```bash
# Install NVIDIA TensorRT (if using Jetson) or onnxruntime-ext
pip install onnxruntime-openvino

# Update config providers
# providers: ["OpenVINOExecutionProvider"]
```

No code changes required — ONNX Runtime provider swap is handled via config.

## Troubleshooting

| Symptom | Likely Fix |
|---------|------------|
| `picamera2` import error | `pip install picamera2` |
| Camera not detected | Check `sudo raspi-config` → camera enabled; reboot |
| Low FPS | Reduce `depth.run_every_n_frames` to 4+ |
| Memory error | Reduce `visualization.layers`; disable depth entirely |
| ONNX Runtime crash | Try `pip install onnxruntime-silicon` (macOS) or `onnxruntime-openvino` |
