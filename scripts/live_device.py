#!/usr/bin/env python3
"""Run the skill live on a device: frames in, arrival alerts out.

The camera host streams frames (edgegate/stream.py, from capture_oak.py --stream or
replay.py). Each frame goes through the path the recorded-session measurement timed
(run_backend.py: letterbox, inference, decoding, NMS) and the zone rule it scored
(edgegate/events.py). An arrival sends an alert by HTTP POST, to an ntfy topic URL
for example, without holding up the frames.

It starts only with an approval record from scripts/deploy.py that names this device,
its runtime and this exact model file, which the gate passed, and it runs the rule
settings in that record, the ones the measurement used. --measurement runs without an
approval, to measure alert delivery before any exists: its alerts say TEST, and the
rule settings come from --events.

Each alert is one JSON line in --log: the deciding frame's seq, session time and
capture wall time (camera host clock), when this device decided and sent it, and the
HTTP status. A new stream starts the zone state over as unknown, like a camera that
comes back, so packages already there are not alerted.

Usage (on the device, in the folder deploy.py fills):
  python3 live_device.py trt student_fp16.engine --approval approval.json --notify URL
  python3 live_device.py hailo student.hef --measurement --events events.json --notify URL
"""
import argparse
import hashlib
import json
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_backend import Hailo, Onnx, Trt, device_info, pct  # noqa: E402

try:  # copied flat to the device
    import detect
    import events
    import stream
except ImportError:  # in the repo
    from edgegate import detect, events, stream


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rule(a):
    """(events settings, measure settings) from the approval, or from --events for a measurement."""
    if a.measurement:
        cfg = json.loads(Path(a.events).read_text())
        return cfg["events"], cfg["measure"]
    ap = json.loads(Path(a.approval).read_text())
    dev = device_info(a.backend)
    problems = []
    if ap.get("verdict") != "PASS":
        problems.append(f"the approval says {ap.get('verdict')!r}")
    if ap.get("artifact_sha256") != sha256(a.model):
        problems.append("this model file is not the approved one")
    if ap.get("device_id") != dev["device_id"]:
        problems.append("the approval is for another device")
    if ap.get("runtime") != dev["runtime"]:
        problems.append(f"approved for {ap.get('runtime')!r}, this device runs {dev['runtime']!r}")
    if problems:
        raise SystemExit("not starting: " + "; ".join(problems))
    return ap["events"], ap["measure"]


class Notifier:
    """POSTs each alert in its own thread and logs when it was decided, sent and answered."""

    def __init__(self, a):
        self.url, self.test, self.backend = a.notify, a.measurement, a.backend
        self.title = ("EdgeGate TEST" if a.measurement else "EdgeGate") + (f" ({a.label})" if a.label else "")
        self.log = open(a.log, "a")
        self.lock = threading.Lock()

    def send(self, seq, t, wall):
        threading.Thread(target=self._post, args=(seq, t, wall, time.time()), daemon=True).start()

    def _post(self, seq, t, wall, decided):
        at = datetime.fromtimestamp(wall).strftime("%H:%M:%S")
        text = (f"TEST: package seen at the entrance at {at}. Delivery measurement, not an approved deployment."
                if self.test else f"A package was left at the entrance at {at}.")
        req = urllib.request.Request(self.url, data=text.encode(),
                                     headers={"Title": self.title, "Priority": "high", "Tags": "package"})
        sent = time.time()
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                status = r.status
        except Exception as e:  # noqa: BLE001 - logged; a failed alert must not stop the stream
            status = f"{type(e).__name__}: {e}"[:200]
        done = time.time()
        entry = {"backend": self.backend, "mode": "measurement" if self.test else "approved", "seq": seq,
                 "t": round(t, 4), "capture_wall": wall, "decided_wall": decided, "sent_wall": sent,
                 "done_wall": done, "post_ms": round((done - sent) * 1e3, 1), "status": status}
        with self.lock:
            self.log.write(json.dumps(entry) + "\n")
            self.log.flush()
        print(f"ALERT seq {seq} t {t:.2f}: {status}, POST {entry['post_ms']} ms", flush=True)


def serve(conn, peer, model, classes, rule, notifier):
    """One stream from the camera host, until it closes."""
    zone = events.ArrivalDetector(rule)
    n, missing, last_seq, lat, report = 0, 0, None, [], time.monotonic()
    print(f"stream from {peer[0]}", flush=True)
    for seq, t, wall, jpeg in stream.receive(conn):
        if last_seq is not None and seq > last_seq + 1:
            missing += seq - last_seq - 1  # dropped on the camera host: this device fell behind
        last_seq = seq
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        rgb, lb = detect.letterbox(bgr)
        dets = model(rgb)
        lat.append((time.perf_counter() - t0) * 1e3)
        norm = detect.to_normalized(dets, lb, bgr.shape[:2])
        if zone.update(t, [d[:5] + [classes[d[5]]] for d in norm if d[5] < len(classes)]):
            notifier.send(seq, t, wall)
        n += 1
        if time.monotonic() - report >= 10:
            print(f"{n} frames, p50 {pct(lat, 50):.1f} ms, {missing} missing, zone {zone.status(t)}", flush=True)
            lat, report = [], time.monotonic()
    print(f"stream from {peer[0]} ended after {n} frames, {missing} missing", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("backend", choices=["onnx", "trt", "hailo"])
    ap.add_argument("model")
    ap.add_argument("--approval", help="approval record written by deploy.py")
    ap.add_argument("--measurement", action="store_true", help="no approval; alerts say TEST")
    ap.add_argument("--events", help="with --measurement: {events, measure} settings from the policy")
    ap.add_argument("--notify", required=True, help="URL each alert is POSTed to")
    ap.add_argument("--label", default="", help="device name in the alert title")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--log", default="alerts.jsonl")
    a = ap.parse_args()
    if a.measurement == bool(a.approval) or (a.measurement and not a.events):
        ap.error("give --approval, or --measurement with --events")
    rule, measure = load_rule(a)
    model = {"onnx": Onnx, "trt": Trt, "hailo": Hailo}[a.backend](a.model)
    model.conf = measure["det_conf"]
    warm = np.full((detect.IMGSZ, detect.IMGSZ, 3), 114, np.uint8)
    for _ in range(10):
        model(warm)
    notifier = Notifier(a)
    srv = socket.create_server(("", a.port))
    print(f"listening on :{a.port}, {'MEASUREMENT (alerts say TEST)' if a.measurement else 'approved'}", flush=True)
    while True:
        conn, peer = srv.accept()
        with conn:
            serve(conn, peer, model, measure["classes"], rule, notifier)


if __name__ == "__main__":
    main()
