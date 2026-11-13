### Table: Hardware Software Stack

| Component | Specification | Role / Notes |
| --- | --- | --- |
| Host Operating System | Microsoft Windows 11 Enterprise (64-bit) | Development and evaluation platform |
| CPU | AMD Ryzen / Intel Core Multi-core x86_64 | System orchestration, ByteTrack, state machines |
| GPU / Accelerator | NVIDIA CUDA GPU (cuda:0, FP16/FP32) | Accelerates YOLO, CLRNet, Depth Anything V2 |
| Python Environment | Python 3.11.9 (.venv) | Core ADAS pipeline runtime |
| PyTorch | PyTorch 2.x with CUDA 12.x | Deep learning tensor operations and model execution |
| OpenCV | OpenCV 4.10.0 (HighGUI & ImgProc) | Video decode, lane warp, ROI alpha blending |
| Transformers / HuggingFace | Hugging Face Transformers 4.x | Depth Anything V2 pipeline loading |
| Ultralytics YOLO | Ultralytics 8.x / YOLO11s | Object detection and human pose estimation |
| OpenPyXL | OpenPyXL 3.1.5 | Automated 5-sheet styled Excel workbook generation |
