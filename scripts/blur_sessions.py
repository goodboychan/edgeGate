"""Public copy of recorded sessions with people blurred: person boxes from the student's detections,
widened and carried ±1 s to cover missed frames, plus the monitor and the hallway always.

Usage: .venv-train/bin/python scripts/blur_sessions.py OUT_DIR SESSION DETS_JSONL [SESSION DETS_JSONL ...]
DETS_JSONL is either the events form ({"t", "dets": [[x1,y1,x2,y2,score,"person"], ...]}, one row
per frame in order) or run_backend's raw form ({"name": "000120.jpg", "dets": [[..., score, cls]]}).
"""
import csv
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path.home() / "edgegate"
CLASSES = ["package", "person"]
FIXED = [(0.70, 0.0, 1.0, 0.45), (0.30, 0.0, 0.53, 0.52)]  # monitor, hallway
CARRY = 15      # frames (1 s at 15 fps) a person box is carried to each side
MARGIN = 0.12   # of the box's own width/height
RADIUS = 22


def load_boxes(dets_path, n_frames):
    per = [[] for _ in range(n_frames)]
    rows = [json.loads(l) for l in open(dets_path) if l.strip()]
    for k, r in enumerate(rows):
        i = int(r["name"][:6]) if "name" in r else k
        if i >= n_frames:
            continue
        for d in r["dets"]:
            cls = d[5] if isinstance(d[5], str) else (CLASSES[int(d[5])] if int(d[5]) < len(CLASSES) else "package")
            if cls == "person" and d[4] >= 0.25:
                per[i].append(d[:4])
    # carry each box to the neighbouring frames so a missed frame stays covered
    out = [[] for _ in range(n_frames)]
    for i, boxes in enumerate(per):
        for j in range(max(0, i - CARRY), min(n_frames, i + CARRY + 1)):
            out[j].extend(boxes)
    return out


def blur_one(args):
    src, dst, boxes = args
    im = Image.open(src).convert("RGB")
    W, H = im.size
    regions = list(FIXED)
    for x1, y1, x2, y2 in boxes:
        mw, mh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        regions.append((max(0, x1 - mw), max(0, y1 - mh), min(1, x2 + mw), min(1, y2 + mh)))
    for x1, y1, x2, y2 in regions:
        box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        im.paste(im.crop(box).filter(ImageFilter.GaussianBlur(RADIUS)), box)
    im.save(dst, "JPEG", quality=85, optimize=True)


def main():
    out = Path(sys.argv[1])
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    for sess, dets in pairs:
        sdir = ROOT / "data/sessions" / sess
        n = sum(1 for _ in csv.DictReader(open(sdir / "frames.csv")))
        boxes = load_boxes(dets, n)
        covered = sum(1 for b in boxes if b)
        odir = out / "sessions" / sess
        # a folder on the Hub holds at most 10,000 files: long sessions go 1,000 frames per folder
        def dst(i):
            return odir / "frames" / (f"{i:06d}"[:3] if n > 10000 else "") / f"{i:06d}.jpg"
        for i in range(0, n, 1000):
            dst(i).parent.mkdir(parents=True, exist_ok=True)
        jobs = [(sdir / "frames" / f"{i:06d}.jpg", dst(i), boxes[i]) for i in range(n)]
        with Pool(8) as pool:
            for k, _ in enumerate(pool.imap_unordered(blur_one, jobs, chunksize=64), 1):
                if k % 5000 == 0:
                    print(f"{sess}: {k}/{n}", flush=True)
        for f in ("frames.csv", "meta.json"):
            shutil.copy(sdir / f, odir / f)
        gt = ROOT / "data/gt" / f"{sess}.json"
        if gt.exists():
            (out / "gt").mkdir(exist_ok=True)
            shutil.copy(gt, out / "gt" / f"{sess}.json")
        print(f"{sess}: {n} frames written, person blur on {covered} frames ({100 * covered / n:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
