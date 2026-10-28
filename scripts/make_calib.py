#!/usr/bin/env python3
"""Pack calibration images for the Hailo DFC from a recorded calibration session.

Takes --count frames spread evenly over the session, letterboxed to 640x640 the
way every backend sees them (edgegate/detect.py), as uint8 RGB in one .npy of
shape (N, 640, 640, 3), the input compile_hailo.py expects. Eval sessions are
refused: quantization must not see the frames the gate judges on.

Usage:  .venv-train/bin/python scripts/make_calib.py data/sessions/c01 build/calib.npy [--count 64]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgegate import detect  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir")
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=64)
    a = ap.parse_args()
    sess = Path(a.session_dir)
    split = json.loads((sess / "meta.json").read_text()).get("split")
    if split == "eval":
        raise SystemExit(f"{sess} is an eval session; use a calib session")
    rows = list(csv.DictReader(open(sess / "frames.csv")))
    picks = np.linspace(0, len(rows) - 1, min(a.count, len(rows))).round().astype(int)
    imgs = [detect.letterbox(cv2.imread(str(sess / "frames" / f"{int(rows[i]['seq']):06d}.jpg")))[0]
            for i in picks]
    np.save(a.out, np.stack(imgs))
    print(f"wrote {len(imgs)} images from the {split} session {sess.name} -> {a.out}")


if __name__ == "__main__":
    main()
