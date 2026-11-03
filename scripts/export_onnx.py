#!/usr/bin/env python3
"""Export the student detector (YOLOv8n family, Ultralytics) to a fixed-shape ONNX.

The same ONNX feeds both backends: TensorRT on the Jetson and the Hailo DFC on
an x86 host with the DFC. Fixed batch 1 and a fixed input size keep both compilers happy. Prints the
sha256 that measurement records use as model_sha256.

Usage:  .venv-train/bin/python scripts/export_onnx.py WEIGHTS.pt OUT.onnx [--imgsz 640]
        (WEIGHTS can be the public yolov8n.pt to test the toolchains before training)
"""
import argparse
import shutil
import sys
from pathlib import Path

import onnx
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgegate.gate import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    a = ap.parse_args()

    path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True,
                                  dynamic=False, batch=1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, out)
    m = onnx.load(str(out))
    shape = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    print("inputs:", [(i.name, shape(i)) for i in m.graph.input])
    print("outputs:", [(o.name, shape(o)) for o in m.graph.output])
    print("sha256:", sha256_file(out))


if __name__ == "__main__":
    main()
