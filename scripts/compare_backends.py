#!/usr/bin/env python3
"""Compare detections from backends against a reference backend, frame by frame.

A reference box counts as matched when the other backend has a box of the same
class with IoU >= --iou (greedy, highest score first). This is a diagnostic only:
two wrong models can agree, so the gate decides on task metrics, not on this.

Usage:  python3 scripts/compare_backends.py REF.jsonl OTHER.jsonl [OTHER.jsonl ...] [--iou 0.5]
"""
import argparse
import json


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load(path):
    return {r["name"]: r["dets"] for r in map(json.loads, open(path))}


def compare(ref, other, thr):
    n_ref = n_other = matched = 0
    ious, dscore = [], []
    for name, rd in ref.items():
        od = sorted(other.get(name, []), key=lambda d: -d[4])
        used = set()
        n_ref += len(rd)
        n_other += len(od)
        for r in sorted(rd, key=lambda d: -d[4]):
            best, bj = 0.0, None
            for j, o in enumerate(od):
                if j not in used and o[5] == r[5] and iou(r, o) > best:
                    best, bj = iou(r, o), j
            if bj is not None and best >= thr:
                used.add(bj)
                matched += 1
                ious.append(best)
                dscore.append(abs(od[bj][4] - r[4]))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {"ref_boxes": n_ref, "other_boxes": n_other, "matched": matched,
            "recall_vs_ref": matched / n_ref if n_ref else float("nan"),
            "extra_boxes": n_other - matched, "mean_iou": mean(ious), "mean_abs_score_diff": mean(dscore),
            "missing_frames": len(set(ref) - set(other))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    ref = load(a.ref)
    for p in a.others:
        r = compare(ref, load(p), a.iou)
        print(f"{p}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
