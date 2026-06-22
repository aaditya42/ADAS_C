#!/usr/bin/env python3
"""Export trained models to ONNX format.

Supports:
  - ``yolo``: exports an Ultralytics YOLO model (uses ``ultralytics``).
  - ``depth_anything``: exports a Depth Anything V2 HuggingFace model
    (uses ``torch.onnx.export``).

Usage::

    python -m src.optimization.export_onnx yolo \\
        --weights yolo11n.pt --out weights/yolo11n.onnx --imgsz 640

    python -m src.optimization.export_onnx depth_anything \\
        --weights depth-anything/Depth-Anything-V2-Small-hf \\
        --out weights/depth_anything_v2_small.onnx --imgsz 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _export_yolo(args: argparse.Namespace) -> None:
    print(f"[export] Exporting YOLO: {args.weights}")

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        print("[export] ultralytics not installed. Install it with: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=13,
        dynamic=True,
    )

    expected = args.out or str(Path(args.weights).with_suffix(".onnx"))
    if not Path(expected).exists():
        onnx_path = Path(args.weights).with_suffix(".onnx")
        if onnx_path.exists():
            expected = str(onnx_path)
        else:
            print(f"[export] ONNX file not found at expected paths")
            sys.exit(1)

    print(f"[export] YOLO ONNX saved to: {expected}")
    _verify(expected)


def _export_depth_anything(args: argparse.Namespace) -> None:
    print(f"[export] Exporting Depth Anything: {args.weights}")
    import torch  # noqa: PLC0415

    out_path = args.out or "weights/depth_anything_v2_small.onnx"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor  # noqa: PLC0415
    except ImportError:
        print("[export] transformers not installed. Install with: pip install transformers")
        sys.exit(1)

    device = "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(args.weights).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)

    print(f"[export] Exporting to ONNX: {out_path}")

    input_names = ["pixel_values"]
    output_names = ["predicted_depth"]

    dynamic_axes: dict[str, dict[int, str]] = {
        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        "predicted_depth": {0: "batch_size", 2: "height", 3: "width"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
        )

    print(f"[export] Depth Anything ONNX saved to: {out_path}")
    _verify(out_path)


def _verify(path: str) -> None:
    try:
        import onnx  # noqa: PLC0415
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"[export] ONNX verification passed  |  inputs={len(model.graph.input)}"
              f"  outputs={len(model.graph.output)}")
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  input: {inp.name}  shape={shape}")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  output: {out.name}  shape={shape}")
    except Exception as e:
        print(f"[export] ONNX verification FAILED: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    subparsers = parser.add_subparsers(dest="command", required=True)

    yolo_parser = subparsers.add_parser("yolo", help="Export YOLO model")
    yolo_parser.add_argument("--weights", required=True, help="Path to .pt weights")
    yolo_parser.add_argument("--out", default=None, help="Output .onnx path")
    yolo_parser.add_argument("--imgsz", type=int, default=640, help="Input image size")

    depth_parser = subparsers.add_parser("depth_anything", help="Export Depth Anything model")
    depth_parser.add_argument("--weights", required=True, help="HF model ID or local path")
    depth_parser.add_argument("--out", default=None, help="Output .onnx path")
    depth_parser.add_argument("--imgsz", type=int, default=224, help="Input image size")

    args = parser.parse_args()

    if args.command == "yolo":
        _export_yolo(args)
    elif args.command == "depth_anything":
        _export_depth_anything(args)


if __name__ == "__main__":
    main()
