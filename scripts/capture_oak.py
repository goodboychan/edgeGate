#!/usr/bin/env python3
"""Record one session from the OAK-D Pro as timestamped JPEG frames.

Layout: data/sessions/<session>/
  frames/000000.jpg ...   RGB frames
  frames.csv              seq,t  (t = seconds since session start, host clock)
  meta.json               split, lighting, notes, device, settings, camera outages
  latest.jpg              refreshed about once a second while recording, to check framing

Times come from the host-synced frame timestamps, so they stay on one clock when
the camera is unplugged and comes back; the gap is logged as an outage. The link
is held at USB 2 with NV12 frames (about 21 MB/s at 720p15): on this desk's cable
and hub the camera boots but never comes back at USB 3, and every session should
record the same way. Focus is fixed at the zone's distance (no hunting when someone
walks past), and auto exposure meters on the policy's zone, so a bright hallway or
a dark monitor does not set the exposure for the drop zone. Train,
calibration and eval are separate sessions, never mixed within one.

Usage:  .venv-oak/bin/python scripts/capture_oak.py SESSION --split eval --light day \
            [--minutes 10] [--note "two boxes, person stands over the box at ~5 min"]
Stop early with Ctrl-C; the frames written so far are kept.
"""
import argparse
import csv
import json
import shutil
import signal
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import depthai as dai

ROOT = Path(__file__).resolve().parent.parent
SIZE, FPS = (1280, 720), 15
# Manual lens position for the drop zone, about 2-3 m away. Sweep on 10/29: zone
# sharpness peaked at 120 (Laplacian variance 165 vs 101 at 110 and 71 at 130),
# and a one-shot autofocus on the zone landed on 119-120.
LENS_POS = 120
stop = False


def on_signal(*_):
    global stop
    stop = True


def zone_box(policy):
    """Normalized bounding box (x0, y0, x1, y1) of the policy's zone polygon."""
    zone = tomllib.loads(Path(policy).read_text())["events"]["zone"]
    xs, ys = [x for x, _ in zone], [y for _, y in zone]
    return min(xs), min(ys), max(xs), max(ys)


def sensor_rect(frame, box):
    """Normalized output box -> pixel rect on the sensor mode image, which camera
    controls (exposure and focus regions) use, before the ISP scales and crops."""
    inv = frame.getTransformation().getMatrixInv()
    w, h = frame.getWidth(), frame.getHeight()
    pts = [(inv[0][0] * x * w + inv[0][1] * y * h + inv[0][2], inv[1][0] * x * w + inv[1][1] * y * h + inv[1][2])
           for x in box[0::2] for y in box[1::2]]
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    return int(x0), int(y0), int(max(p[0] for p in pts) - x0), int(max(p[1] for p in pts) - y0)


def record(dev, q, writer, sess, state):
    """Pull frames until stop, the time limit, or the device drops."""
    last_preview = 0.0
    while not stop and time.monotonic() < state["deadline"]:
        f = q.tryGet()
        if f is None:
            if dev.isClosed():
                raise RuntimeError("device closed")
            time.sleep(0.005)
            continue
        t = (f.getTimestamp() - state["t0"]).total_seconds()
        img = f.getCvFrame()
        seq = state["n"]
        cv2.imwrite(str(sess / "frames" / f"{seq:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        writer.writerow([seq, f"{t:.4f}"])
        state["n"] += 1
        state["last_t"] = t
        if t - last_preview >= 1.0:
            cv2.imwrite(str(sess / "latest.jpg"), img)
            last_preview = t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--split", required=True, choices=["train", "calib", "eval"])
    ap.add_argument("--light", required=True, choices=["day", "evening", "night"])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(ROOT / "data/sessions"))
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"),
                    help="exposure meters on this policy's zone")
    a = ap.parse_args()

    sess = Path(a.out) / a.session
    if sess.exists():
        raise SystemExit(f"{sess} exists; pick a new session name")
    (sess / "frames").mkdir(parents=True)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    meta = {"session": a.session, "split": a.split, "lighting": a.light, "note": a.note,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resolution": list(SIZE), "fps": FPS, "camera": "CAM_A (RGB, NV12 over USB 2)", "depthai": dai.__version__,
            "lens_position": LENS_POS, "exposure_box": zone_box(a.policy),
            "devices": [], "outages": []}
    state = {"t0": dai.Clock.now(), "n": 0, "last_t": None,
             "deadline": time.monotonic() + a.minutes * 60}
    with open(sess / "frames.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["seq", "t"])
        while not stop and time.monotonic() < state["deadline"]:
            try:
                dev = dai.Device(dai.UsbSpeed.HIGH)
                with dai.Pipeline(dev) as p:
                    cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                    cam.initialControl.setManualFocus(LENS_POS)
                    ctrl = cam.inputControl.createInputQueue()
                    q = cam.requestOutput(SIZE, dai.ImgFrame.Type.NV12, fps=FPS).createOutputQueue(
                        maxSize=30, blocking=False)
                    p.start()
                    # The region needs the sensor-side mapping, known from the first frame.
                    c = dai.CameraControl()
                    c.setAutoExposureRegion(*sensor_rect(q.get(), meta["exposure_box"]))
                    ctrl.send(c)
                    info = {"product": dev.getProductName(), "id": dev.getDeviceId(),
                            "usb": str(dev.getUsbSpeed()).split(".")[-1]}
                    if info not in meta["devices"]:
                        meta["devices"].append(info)
                    if meta["outages"] and meta["outages"][-1]["end"] is None:
                        meta["outages"][-1]["end"] = round((dai.Clock.now() - state["t0"]).total_seconds(), 3)
                    print(f"recording {a.session} from {info['product']} ({info['usb']})", flush=True)
                    record(dev, q, writer, sess, state)
            except RuntimeError as e:
                # Camera unplugged or crashed: log the outage and keep trying to reconnect.
                if not meta["outages"] or meta["outages"][-1]["end"] is not None:
                    meta["outages"].append({"start": state["last_t"], "end": None, "error": str(e)[:200]})
                    print(f"camera lost at t={state['last_t']}: {e}; retrying", flush=True)
                fcsv.flush()
                time.sleep(1.0)
    meta["frames"] = state["n"]
    meta["duration_s"] = state["last_t"]
    (sess / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {state['n']} frames, {state['last_t'] or 0:.1f} s, "
          f"{len(meta['outages'])} outages -> {sess}  ({shutil.disk_usage(sess).free / 1e9:.0f} GB free)")


if __name__ == "__main__":
    main()
