#!/usr/bin/env python3
"""Label a recorded session with SAM 3.1 from text prompts (teacher boxes).

Reads data/sessions/<session>/frames.csv + frames/, keeps every --stride-th frame,
and runs the SAM 3.1 video predictor over chunks of --chunk kept frames, once per
class prompt. Writes, next to the frames:
  labels_sam.jsonl       {"seq": 120, "t": 8.0, "dets": [[x1, y1, x2, y2, score, "package"], ...]}
  labels_sam.meta.json   prompts, stride, chunk, threshold, checkpoint, timing
Coordinates are normalized to [0, 1], the format edgegate/events.py reads, so the
teacher's own alerts can be scored against the hand-marked events too.

Long sessions are chunked because the predictor keeps every frame of a session
in memory. Objects are re-detected at each chunk start, so track ids are not kept.

Usage:  .venv-sam3/bin/python scripts/label_sam.py SESSION_DIR [--stride 5] [--chunk 150]
            [--prompt package="cardboard box"] [--prompt person=person]
"""
import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

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
                out[r["frame_index"]] = [
                    [float(x), float(y), float(x + w), float(y + h), float(s)]
                    for (x, y, w, h), s in zip(o["out_boxes_xywh"], o["out_probs"])]
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
    prompts = dict(p.split("=", 1) for p in a.prompt) or {"package": "package", "person": "person"}

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
            "load_s": round(load_s, 1), "label_s": round(label_s, 1),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (sess / "labels_sam.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    counts = {c: sum(1 for d in dets for b in d if b[5] == c) for c in prompts}
    print(f"wrote {len(rows)} frames to {sess / 'labels_sam.jsonl'}: boxes {counts}, "
          f"{label_s / len(rows):.2f} s/frame, peak {meta['gpu_peak_gb']} GB")


if __name__ == "__main__":
    main()
