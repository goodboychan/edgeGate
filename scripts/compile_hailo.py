#!/usr/bin/env python3
"""Compile the student ONNX to a Hailo-8 HEF with the Hailo Dataflow Compiler.

Runs on an x86 host with the DFC installed (an x86 host with the DFC). The network ends at the six
YOLOv8 head convolutions (box and class per scale); DFL decoding and NMS run on
the host CPU, as is usual for YOLOv8 on Hailo. The HEF takes uint8 RGB 640x640;
normalization to [0, 1] is folded into the network.

Calibration images (--calib) are a .npy of shape (N, H, W, 3), uint8 RGB, taken
from calibration sessions only, never from eval sessions.

Build-time faults, for fault-injected candidates only (judged by their measured
metrics like any build): --calib-limit N uses only the first N calibration
images, --precision a8_w4 quantizes every conv weight to 4 bits, and
--no-normalization leaves out the folded /255 so the network sees 0..255.

Usage:  python compile_hailo.py MODEL.onnx CALIB.npy OUT.hef [--opt-level 1]
            [--calib-limit 10] [--precision a8_w4] [--no-normalization]
"""
import argparse
import hashlib

import numpy as np
from hailo_sdk_client import ClientRunner

END_NODES = [f"/model.22/{head}.{i}/{head}.{i}.2/Conv" for i in range(3) for head in ("cv2", "cv3")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx")
    ap.add_argument("calib")
    ap.add_argument("out")
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--calib-limit", type=int, help="fault: use only the first N calibration images")
    ap.add_argument("--precision", choices=["a8_w8", "a8_w4"], default="a8_w8", help="a8_w4 is a fault")
    ap.add_argument("--no-normalization", action="store_true", help="fault: skip the folded /255")
    a = ap.parse_args()

    calib = np.load(a.calib)[:a.calib_limit]
    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(a.onnx, "student", start_node_names=["images"], end_node_names=END_NODES,
                                net_input_shapes={"images": [1, 3, calib.shape[1], calib.shape[2]]})
    script = "" if a.no_normalization else "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
    script += f"model_optimization_flavor(optimization_level={a.opt_level}, compression_level=0)\n"
    if a.precision != "a8_w8":
        script += f"quantization_param({{conv*}}, precision_mode={a.precision})\n"
    runner.load_model_script(script)
    runner.optimize(calib.astype(np.float32))
    hef = runner.compile()
    with open(a.out, "wb") as f:
        f.write(hef)
    print(f"wrote {a.out} ({len(hef) / 1e6:.1f} MB) sha256 {hashlib.sha256(hef).hexdigest()}, "
          f"{len(calib)} calibration images, optimization level {a.opt_level}, {a.precision}"
          + (", no normalization" if a.no_normalization else ""))


if __name__ == "__main__":
    main()
