#!/usr/bin/env python3
"""Label a recorded session with SAM 3.1 from text prompts (teacher boxes).

Reads data/sessions/<session>/frames.csv + frames/, keeps every --stride-th frame,
and runs the SAM 3.1 video predictor over chunks of --chunk kept frames, once per
class prompt. Writes, next to the frames:
  labels_sam.jsonl       {"seq": 120, "t": 8.0, "dets": [[x1, y1, x2, y2, score, "package"], ...]}
  labels_sam.meta.json   prompts, stride, chunk, threshold, checkpoint, timing
Coordinates are normalized to [0, 1], the format edgegate/events.py reads, so the
teacher's own alerts can be scored against the hand-marked events too.

Each box is the bounding box of the main piece of SAM's mask (pieces within 2% of
the frame width joined, since a label across a box splits its mask), not SAM's own
box: on the first recording, a mask on one box of the fixed stack also
held a stray patch on the desk boxes, and its box spanned 70% of the frame.
The default package prompt is "box": with "package", the fixed shopping bag was
labeled too and the spanning boxes were more frequent (140 vs 80 of 171 frames,
before the mask fix). "cardboard box" missed a black rigid gift box set in the
zone on 10/29; "box" found it and the same six cardboard boxes, and still left
the bag out.

Long sessions are chunked because the predictor keeps every frame of a session
in memory. Objects are re-detected at each chunk start, so track ids are not kept.

Usage:  .venv-sam3/bin/python scripts/label_sam.py SESSION_DIR [--stride 5] [--chunk 150]
            [--prompt package=box] [--prompt person=person]   (these are the defaults)
"""
import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_predictor

CKPT = Path.home() / "models/sam3.1/sam3.1_multiplex.pt"


def load_predictor():
    # FlashAttention 3 is not available for the GB10 (sm_121).
    pred = build_sam3_predictor(version="sam3.1", checkpoint_path=str(CKPT), use_fa3=False)
    # Upstream bug: start_session always passes offload_state_to_cpu, which the
    # multiplex model's init_state does not accept. Drop it here; third_party
    # stays unmodified.
    init = pred.model.init_state
    pred.model.init_state = lambda offload_state_to_cpu=False, **kw: init(**kw)
    return pred


def mask_box(mask, gap=0.02):
    """Normalized [x1, y1, x2, y2] of the main piece of a binary mask, or None.

    Pieces closer than gap (fraction of the frame width) count as one: a label or
    tape across a box splits its mask, and the box is the whole of it. A stray
    patch further away is left out.
    """
    m = np.asarray(mask, dtype=np.uint8).squeeze()
    if not m.any():
        return None
    k = max(1, round(gap * m.shape[1]))
    joined = cv2.dilate(m, np.ones((k, k), np.uint8))
    n, labels = cv2.connectedComponents(joined, connectivity=8)
    piece = labels == 1 + int(np.argmax([np.count_nonzero(m[labels == i]) for i in range(1, n)]))
    ys, xs = np.nonzero(m & piece)
    H, W = m.shape
    return [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]


def label_chunk(pred, frame_paths, prompt, thresh):
    """Boxes per chunk index for one text prompt."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(frame_paths):
            os.symlink(p, os.path.join(tmp, f"{i}.jpg"))
        sid = pred.handle_request({"type": "start_session", "resource_path": tmp,
                                   "offload_video_to_cpu": True})["session_id"]
        try:
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                 "text": prompt, "output_prob_thresh": thresh})
            out = {}
            for r in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                 "propagation_direction": "forward",
                                                 "output_prob_thresh": thresh}):
                o = r["outputs"]
                boxes = [mask_box(m) for m in o["out_binary_masks"]]
                out[r["frame_index"]] = [[*map(float, b), float(s)]
                                         for b, s in zip(boxes, o["out_probs"]) if b is not None]
            return out
        finally:
            pred.handle_request({"type": "close_session", "session_id": sid})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir")
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth frame (15 fps / 5 = 3 fps)")
    ap.add_argument("--chunk", type=int, default=150, help="kept frames per predictor session")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--prompt", action="append", default=[], metavar="CLASS=TEXT")
    a = ap.parse_args()
    prompts = dict(p.split("=", 1) for p in a.prompt) or {"package": "box", "person": "person"}

    sess = Path(a.session_dir)
    rows = list(csv.DictReader(open(sess / "frames.csv")))[::a.stride]
    if not rows:
        raise SystemExit(f"no frames in {sess}")
    paths = [str((sess / "frames" / f"{int(r['seq']):06d}.jpg").resolve()) for r in rows]

    t0 = time.time()
    pred = load_predictor()
    load_s = time.time() - t0
    dets = [[] for _ in rows]
    t1 = time.time()
    for start in range(0, len(rows), a.chunk):
        chunk = paths[start:start + a.chunk]
        for cls, text in prompts.items():
            for i, boxes in label_chunk(pred, chunk, text, a.thresh).items():
                dets[start + i] += [b + [cls] for b in boxes]
        done = min(start + a.chunk, len(rows))
        print(f"{done}/{len(rows)} frames, {(time.time() - t1) / done:.2f} s/frame", flush=True)
    label_s = time.time() - t1

    with open(sess / "labels_sam.jsonl", "w") as f:
        for r, d in zip(rows, dets):
            f.write(json.dumps({"seq": int(r["seq"]), "t": float(r["t"]), "dets": d}) + "\n")
    meta = {"model": "SAM 3.1 multiplex", "checkpoint": str(CKPT), "prompts": prompts,
            "stride": a.stride, "chunk": a.chunk, "thresh": a.thresh, "frames": len(rows),
            "box": "main piece of the mask, pieces within 2% of the frame width joined",
            "load_s": round(load_s, 1), "label_s": round(label_s, 1),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (sess / "labels_sam.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    counts = {c: sum(1 for d in dets for b in d if b[5] == c) for c in prompts}
    print(f"wrote {len(rows)} frames to {sess / 'labels_sam.jsonl'}: boxes {counts}, "
          f"{label_s / len(rows):.2f} s/frame, peak {meta['gpu_peak_gb']} GB")


if __name__ == "__main__":
    main()
