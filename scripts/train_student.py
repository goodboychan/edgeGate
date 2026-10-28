#!/usr/bin/env python3
"""Train the student detector (YOLOv8n, Ultralytics) on SAM 3.1 teacher labels.

Only sessions recorded with --split train are accepted: calibration and eval
sessions never reach training, so the eval set stays unseen. Each session's last
--val-frac of labeled frames (in time) is held out as validation for watching the
training only; the gate judges the student on eval sessions, never on this split.

Classes are fixed in this order, which is what measure.py --classes expects:
  0 package, 1 person
Frames with no teacher boxes stay in as negatives.

Writes OUT/: dataset/ (symlinked frames, YOLO label files), data.yaml, the
Ultralytics run, student.pt (best weights) and provenance.json (sessions, label
file hashes, base weights, arguments).

Usage:  .venv-train/bin/python scripts/train_student.py data/sessions/t01 data/sessions/t02 \
            --out runs/train/s1 [--epochs 50] [--base build/probe/yolov8n.pt]
The base is the public COCO yolov8n.pt (pretrained, AGPL-3.0); training on our
labels is what was built during the hackathon.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["package", "person"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yolo_line(det):
    x1, y1, x2, y2, _, name = det
    return f"{CLASSES.index(name)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def add_session(sess, ds, val_frac):
    meta = json.loads((sess / "meta.json").read_text())
    if meta.get("split") != "train":
        raise SystemExit(f"{sess} is a {meta.get('split')!r} session; only train sessions are used")
    rows = [json.loads(line) for line in open(sess / "labels_sam.jsonl")]
    cut = int(len(rows) * (1 - val_frac))
    counts = {"train": 0, "val": 0, "boxes": {c: 0 for c in CLASSES}}
    for i, r in enumerate(rows):
        part = "train" if i < cut else "val"
        name = f"{sess.name}_{r['seq']:06d}"
        (ds / "images" / part / f"{name}.jpg").symlink_to((sess / "frames" / f"{r['seq']:06d}.jpg").resolve())
        dets = [d for d in r["dets"] if d[5] in CLASSES]
        (ds / "labels" / part / f"{name}.txt").write_text("".join(yolo_line(d) + "\n" for d in dets))
        counts[part] += 1
        for d in dets:
            counts["boxes"][d[5]] += 1
    return {"session": sess.name, "lighting": meta.get("lighting"),
            "labels_sha256": sha256(sess / "labels_sam.jsonl"), **counts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(ROOT / "build/probe/yolov8n.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new run name")
    ds = out / "dataset"
    for kind in ("images", "labels"):
        for part in ("train", "val"):
            (ds / kind / part).mkdir(parents=True)
    sessions = [add_session(Path(s).resolve(), ds, a.val_frac) for s in a.sessions]
    (out / "data.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    for s in sessions:
        print(f"{s['session']}: {s['train']} train / {s['val']} val frames, boxes {s['boxes']}")

    model = YOLO(a.base)
    model.train(data=str(out / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                seed=a.seed, deterministic=True, project=str(out), name="ultralytics",
                flipud=0.0, plots=False, verbose=False)
    best = out / "ultralytics/weights/best.pt"
    shutil.copy(best, out / "student.pt")
    (out / "provenance.json").write_text(json.dumps({
        "classes": CLASSES, "sessions": sessions, "base": Path(a.base).name, "base_sha256": sha256(a.base),
        "args": {k: v for k, v in vars(a).items() if k != "sessions"}, "ultralytics": ultralytics.__version__,
        "student_sha256": sha256(out / "student.pt"), "python": sys.version.split()[0]}, indent=2) + "\n")
    print(f"student -> {out / 'student.pt'}")


if __name__ == "__main__":
    main()
