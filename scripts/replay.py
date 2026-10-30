#!/usr/bin/env python3
"""Stream a recorded session to live runners, paced to its recorded frame times.

Each frame goes out as its recorded JPEG bytes with its session time, the way
capture_oak.py --stream sends a live one. For testing the live path without the
camera, and for showing what a held build would have done, labeled as a replay.

Usage:  python3 scripts/replay.py data/sessions/e01 --stream jetson:8790 [--stream ...] \
            [--speed 2] [--start 120] [--end 300]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgegate import stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--stream", action="append", required=True, metavar="HOST:PORT")
    ap.add_argument("--speed", type=float, default=1.0, help="2 plays twice as fast")
    ap.add_argument("--start", type=float, default=0.0, help="session time to start from (s)")
    ap.add_argument("--end", type=float, default=float("inf"), help="session time to stop at (s)")
    a = ap.parse_args()
    sess = Path(a.session)
    rows = [(int(r["seq"]), float(r["t"])) for r in csv.DictReader(open(sess / "frames.csv"))]
    rows = [(seq, t) for seq, t in rows if a.start <= t <= a.end]
    senders = [stream.Sender(*stream.target(x)) for x in a.stream]
    deadline = time.monotonic() + 15
    while not all(s.connected for s in senders):
        if time.monotonic() > deadline:
            raise SystemExit("not connected: " + ", ".join(x for x, s in zip(a.stream, senders) if not s.connected))
        time.sleep(0.1)
    print(f"replaying {sess.name} t {rows[0][1]:.1f}-{rows[-1][1]:.1f} s at {a.speed}x to {', '.join(a.stream)}",
          flush=True)
    t_first, start = rows[0][1], time.monotonic()
    for seq, t in rows:
        wait = start + (t - t_first) / a.speed - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        jpeg = (sess / "frames" / f"{seq:06d}.jpg").read_bytes()
        wall = time.time()
        for s in senders:
            s.put(seq, t, wall, jpeg)
    for x, s in zip(a.stream, senders):
        s.flush()
        print(f"{x}: sent {s.sent}, dropped {s.dropped}")


if __name__ == "__main__":
    main()
