#!/usr/bin/env python3
"""Run the student on one backend over a folder of frames; write detections and timing.

Runs on the device itself (copied there with edgegate/detect.py next to it):
  onnx   onnxruntime on the GX10, the reference
  trt    TensorRT engine on the Jetson (cuda-python for memory)
  hailo  HEF on the Hailo-8 through HailoRT; decoding and NMS on the host CPU

For each frame the timed path is what a live camera frame goes through: letterbox,
inference, decoding, NMS. Reading the JPEG is not timed. Writes:
  OUT.jsonl   {"name": "000120.jpg", "dets": [[x1, y1, x2, y2, score, cls], ...]} (normalized)
  OUT.json    backend, model file sha256, frames, latency p50/p95 (total and inference), peak RSS

Usage:  python3 run_backend.py {onnx,trt,hailo} MODEL FRAMES_DIR OUT [--warmup 20] [--conf 0.25]
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import detect
except ImportError:
    from edgegate import detect


class Onnx:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def __call__(self, rgb):
        x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return detect.decode_full(self.s.run(None, {self.name: x})[0], self.conf)


class Trt:
    def __init__(self, path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.cudart = cudart
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = engine.create_execution_context()
        self.bufs = {}
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            host = np.empty(tuple(engine.get_tensor_shape(n)), trt.nptype(engine.get_tensor_dtype(n)))
            err, dev = cudart.cudaMalloc(host.nbytes)
            assert err == cudart.cudaError_t.cudaSuccess, err
            self.ctx.set_tensor_address(n, dev)
            self.bufs[n] = (host, dev, engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.stream = cudart.cudaStreamCreate()[1]

    def __call__(self, rgb):
        c, k = self.cudart, self.cudart.cudaMemcpyKind
        for host, dev, is_in in self.bufs.values():
            if is_in:
                np.multiply(rgb.transpose(2, 0, 1), np.float32(1 / 255), out=host[0])  # no temporaries
                c.cudaMemcpyAsync(dev, host.ctypes.data, host.nbytes, k.cudaMemcpyHostToDevice, self.stream)
        self.ctx.execute_async_v3(self.stream)
        for host, dev, is_in in self.bufs.values():
            if not is_in:
                c.cudaMemcpyAsync(host.ctypes.data, dev, host.nbytes, k.cudaMemcpyDeviceToHost, self.stream)
        c.cudaStreamSynchronize(self.stream)
        out = next(h for h, _, is_in in self.bufs.values() if not is_in)
        return detect.decode_full(out, self.conf)


class Hailo:
    def __init__(self, path):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, VDevice)
        hef = HEF(path)
        self.dev = VDevice()
        ng = self.dev.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        self.input = hef.get_input_vstream_infos()[0].name
        self.pipe = InferVStreams(ng, InputVStreamParams.make(ng, format_type=FormatType.UINT8),
                                  OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32))
        self.pipe.__enter__()
        self.act = ng.activate(ng.create_params())
        self.act.__enter__()

    def __call__(self, rgb):
        res = self.pipe.infer({self.input: np.ascontiguousarray(rgb)[None]})
        return detect.decode_heads(list(res.values()), self.conf)

    def close(self):
        self.act.__exit__(None, None, None)
        self.pipe.__exit__(None, None, None)
        self.dev.release()


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("backend", choices=["onnx", "trt", "hailo"])
    ap.add_argument("model")
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    model = {"onnx": Onnx, "trt": Trt, "hailo": Hailo}[a.backend](a.model)
    model.conf = a.conf
    frames = sorted(Path(a.frames).glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no .jpg frames in {a.frames}")
    first = detect.letterbox(cv2.imread(str(frames[0])))[0]
    for _ in range(a.warmup):
        model(first)

    total, infer = [], []
    with open(a.out + ".jsonl", "w") as f:
        for p in frames:
            bgr = cv2.imread(str(p))
            t0 = time.perf_counter()
            rgb, lb = detect.letterbox(bgr)
            t1 = time.perf_counter()
            dets = model(rgb)
            t2 = time.perf_counter()
            total.append((t2 - t0) * 1e3)
            infer.append((t2 - t1) * 1e3)
            f.write(json.dumps({"name": p.name, "dets": detect.to_normalized(dets, lb, bgr.shape[:2])}) + "\n")
    if hasattr(model, "close"):
        model.close()
    summary = {"backend": a.backend, "model_file": Path(a.model).name,
               "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
               "frames": len(frames), "conf": a.conf,
               "latency_ms_p50": pct(total, 50), "latency_ms_p95": pct(total, 95),
               "infer_post_ms_p50": pct(infer, 50), "infer_post_ms_p95": pct(infer, 95),
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    Path(a.out + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
