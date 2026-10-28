#!/usr/bin/env python3
"""Make a fault-injected copy of the student ONNX, to build a bad candidate on purpose.

The fault sits in front of the network as a fixed 1x1 convolution, which every
toolchain compiles, so the fault survives into the TensorRT engine or HEF:
  bgr      swaps the red and blue input channels (a BGR/RGB mix-up in the build)
  no_norm  multiplies the input by 255 (a build that drops the 0..1 normalization)

A fault-injected build is only a candidate. Whether it is bad is decided by its
measured task metrics like any other build, never by its name. Record it with
the original ONNX as --onnx (model_sha256): the model is the same, the build is wrong.
Build-time faults on the Hailo side (few calibration images, 4-bit weights,
no normalization) are flags of compile_hailo.py instead.

Usage:  .venv-train/bin/python scripts/inject_fault.py build/student.onnx bgr build/student_bgr.onnx
"""
import argparse
import hashlib

import numpy as np
import onnx
from onnx import helper, numpy_helper


def weights(fault):
    if fault == "bgr":
        w = np.zeros((3, 3), np.float32)
        w[0, 2] = w[1, 1] = w[2, 0] = 1.0
    else:
        w = np.eye(3, dtype=np.float32) * 255.0
    return w.reshape(3, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx")
    ap.add_argument("fault", choices=["bgr", "no_norm"])
    ap.add_argument("out")
    a = ap.parse_args()
    m = onnx.load(a.onnx)
    g = m.graph
    inp = g.input[0].name
    moved = f"{inp}_fault_{a.fault}"
    for node in g.node:  # everything that read the input now reads the faulted tensor
        node.input[:] = [moved if x == inp else x for x in node.input]
    g.initializer.append(numpy_helper.from_array(weights(a.fault), f"fault_{a.fault}_w"))
    g.node.insert(0, helper.make_node("Conv", [inp, f"fault_{a.fault}_w"], [moved],
                                      name=f"fault_{a.fault}", kernel_shape=[1, 1]))
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"wrote {a.out} with fault {a.fault}, sha256 {sha}")


if __name__ == "__main__":
    main()
