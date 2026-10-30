#!/usr/bin/env python3
"""Measure alert delivery on the live path, for the policy's notify_s.

Delivery is the time from the frame that decides an arrival to the alert showing on
the phone. Both ends are read off one clock, the camera's: the phone lies face up in
view next to the zone, and its screen lighting up in the recorded frames is the
alert received. No device clocks need to agree, and the phone's own push delay is in.

  start   run a device's live runner in measurement mode (alerts say TEST, no
          approval: nothing is deployed) with the policy's rule settings
  report  pair each alert in the runner's log with the first frame after it where
          the phone area brightens, and print the delays and their maximum

Steps: start on one device, record with capture_oak.py SESSION --split live
--stream HOST:8790 while placing packages, copy the device's alerts.jsonl next to
the session, then report with the phone's box in normalized coordinates.

Usage:  python3 scripts/measure_delivery.py start trt build/s1/student_fp16.engine --notify URL
        .venv-train/bin/python scripts/measure_delivery.py report data/sessions/m01 m01_trt_alerts.jsonl \
            --phone 0.70,0.80,0.78,0.95
Hosts: JETSON (default jetson) and HAILO_PI (default chans@192.168.50.151).
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from deploy import FILES, HOSTS, LIVE, RUNTIME, SSH, STOP  # noqa: E402
from edgegate.gate import _parse_time, load_policy  # noqa: E402


def start(a):
    policy = load_policy(a.policy)
    host = HOSTS[a.backend]
    events = ROOT / "runs" / "delivery_events.json"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(json.dumps({"events": policy["events"], "measure": policy["measure"]}, indent=2) + "\n")
    subprocess.run(SSH + [host, f"{STOP}; mkdir -p {LIVE}"], check=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), *[str(ROOT / f) for f in FILES], a.model,
                    str(events), f"{host}:{LIVE}/"], check=True)
    run = (f"cd {LIVE} && {{ nohup python3 live_device.py {RUNTIME[a.backend]} {Path(a.model).name} --measurement "
           f"--events {events.name} --notify '{a.notify}' --label {a.backend} --port {a.port} "
           f"> live.log 2>&1 < /dev/null & echo $! > live.pid; }}")
    subprocess.run(SSH + [host, run], check=True)
    print(f"{a.backend}: measurement runner started on {host}:{a.port} (alerts say TEST); "
          f"log in {LIVE}/live.log, stop with: python3 scripts/deploy.py --stop {a.backend}")


def phone_level(path, box):
    """Mean gray level of the phone's box in one frame."""
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    x1, y1, x2, y2 = box
    return float(g[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)].mean())


def report(a):
    import numpy as np  # report runs in .venv-train (numpy, OpenCV); start needs neither
    sess = Path(a.session)
    box = [float(v) for v in a.phone.split(",")]
    rows = [(int(r["seq"]), float(r["t"])) for r in csv.DictReader(open(sess / "frames.csv"))]
    ts = np.array([t for _, t in rows])
    # The runner's log keeps every stream; this session's alerts are the ones whose
    # capture clock started when the session did (capture_wall - t = session start).
    t0 = _parse_time(json.loads((sess / "meta.json").read_text())["started_utc"]).timestamp()
    out = []
    for al in map(json.loads, open(a.alerts)):
        if abs(al["capture_wall"] - al["t"] - t0) > 5 or al["t"] > ts[-1]:
            continue
        before = [i for i in range(len(rows)) if al["t"] - 1.0 <= ts[i] < al["t"]]
        after = [i for i in range(len(rows)) if al["t"] <= ts[i] <= al["t"] + a.window]
        base = np.median([phone_level(sess / "frames" / f"{rows[i][0]:06d}.jpg", box) for i in before])
        lit = next((i for i in after
                    if phone_level(sess / "frames" / f"{rows[i][0]:06d}.jpg", box) > base + a.rise), None)
        delay = None if lit is None else round(ts[lit] - al["t"], 3)
        out.append({"seq": al["seq"], "t": al["t"], "status": al["status"], "phone_lit_t":
                    None if lit is None else float(ts[lit]), "delay_s": delay})
        print(f"alert t {al['t']:.2f} ({al['status']}): " + (f"phone lit {delay:.2f} s later" if delay is not None
                                                             else f"phone not seen lit within {a.window} s"))
    seen = [o["delay_s"] for o in out if o["delay_s"] is not None]
    print(f"{len(seen)} of {len(out)} alerts seen on the phone; max delay "
          + (f"{max(seen):.2f} s" if seen else "-"))
    if a.out:
        Path(a.out).write_text(json.dumps({"session": sess.name, "alerts": a.alerts, "phone_box": box,
                                           "rise": a.rise, "results": out}, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("backend", choices=list(RUNTIME))
    s.add_argument("model")
    s.add_argument("--notify", required=True)
    s.add_argument("--port", type=int, default=8790)
    s.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    r = sub.add_parser("report")
    r.add_argument("session")
    r.add_argument("alerts", help="the runner's alerts.jsonl for this session")
    r.add_argument("--phone", required=True, help="x1,y1,x2,y2 of the phone, normalized")
    r.add_argument("--rise", type=float, default=25.0, help="gray levels above the second before = lit")
    r.add_argument("--window", type=float, default=15.0, help="seconds to look for the phone after an alert")
    r.add_argument("--out", help="write the results here (JSON)")
    a = ap.parse_args()
    start(a) if a.cmd == "start" else report(a)


if __name__ == "__main__":
    main()
