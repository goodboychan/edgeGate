#!/usr/bin/env python3
"""Replay clip for the demo video: the eval frames around each arrival of one session,
with what each named build did there according to its measurement record (per_event).
Nothing is re-run; the outcomes and delays are read from the records in runs/records
(or results/records). Every frame carries a REPLAY badge, and the frames come from the
blurred public copy (data/public_hf/sessions/<session>/frames), so nobody's face is shown.

Usage:
  python3 scripts/video_overlay.py e05 \
      --build "s4 w4 · Hailo-8=runs/records/s4/w4/hailo8_*.json" \
      --build "s3 clean · Jetson=runs/records/s3/clean/trt_*.json" \
      --speed 2 --before 1 --after 6 --hold 1.2 --out clip_e05.mp4
Writes the clip and <out>.json (session, records used, the arrivals shown).
"""
import argparse
import csv
import glob
import json
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BOOK = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WHITE, GRAY, GREEN, ORANGE, RED, YELLOW = ((255, 255, 255), (190, 190, 190), (90, 220, 120), (255, 170, 60),
                                           (255, 90, 90), (255, 220, 0))


def latest(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        sys.exit(f"no record matches {pattern}")
    return json.loads(Path(paths[-1]).read_text()), str(Path(paths[-1]).relative_to(ROOT))


def status(pe, dt, stamp):
    """(text, color) of one build at dt seconds after the package was let go."""
    if dt < 0:
        return "watching", GRAY
    out = pe["outcome"]
    if out in ("on_time", "late"):
        if dt >= pe["delay"]:
            return f"ALERT  +{pe['delay']:.1f} s" + ("  (late)" if out == "late" else ""), GREEN if out == "on_time" else ORANGE
        return "no alert yet", GRAY
    if out == "missed":
        return ("MISSED  (no alert within 10 s)" if stamp else "no alert"), (RED if stamp else GRAY)
    return out, ORANGE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--build", action="append", required=True, metavar="LABEL=RECORD_GLOB",
                    help="a line in the clip: its label and the record glob (the newest match is used)")
    ap.add_argument("--events", default="all", help="0-based indexes of the in-scope arrivals to show, or all")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed; 2 = twice real time")
    ap.add_argument("--before", type=float, default=1.0, help="seconds of recording before each release")
    ap.add_argument("--after", type=float, default=6.0, help="seconds of recording after each release")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds of video the outcome stamp stays")
    ap.add_argument("--frames", help="frames dir; default is the blurred public copy of the session")
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = a.session
    pub = ROOT / "data/public_hf/sessions" / s
    fdir = Path(a.frames) if a.frames else pub / "frames"
    times = [(int(r["seq"]), float(r["t"])) for r in csv.DictReader(open(pub / "frames.csv"))]
    meta = json.loads((pub / "meta.json").read_text())
    gt = [e for e in json.loads((ROOT / "data/gt" / f"{s}.json").read_text())["events"]]
    zone = tomllib.loads(Path(a.policy).read_text())["events"]["zone"]

    builds = []
    for spec in a.build:
        label, _, pat = spec.partition("=")
        rec, path = latest(pat)
        ps = next((x for x in rec["per_session"] if x["session"] == s), None)
        if ps is None:
            sys.exit(f"{path} has no session {s}")
        by_arrive = {round(e["arrive"], 3): e for e in ps["per_event"]}
        builds.append({"label": label, "metrics": rec["metrics"], "by_arrive": by_arrive, "record": path})
    # in-scope arrivals (second packages are scored apart and not shown)
    first = next(iter(builds))["by_arrive"]
    arrivals = [e for e in gt if not first.get(round(e["arrive"], 3), {}).get("second")]
    idx = list(range(len(arrivals))) if a.events == "all" else [int(i) for i in a.events.split(",")]

    fps_in = meta.get("fps", 15)
    fps_out = fps_in * a.speed
    W, H = meta["resolution"]
    f_badge, f_label, f_small, f_status = (ImageFont.truetype(BOLD, 26), ImageFont.truetype(BOLD, 28),
                                           ImageFont.truetype(BOOK, 19), ImageFont.truetype(BOLD, 30))
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, H))
    if not out.isOpened():
        sys.exit(f"cannot write {a.out}")
    shown = []

    def render(seq, t, ev, stamp):
        img = Image.open(fdir / f"{seq:06d}.jpg").convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H))
        d = ImageDraw.Draw(img, "RGBA")
        d.polygon([(x * W, y * H) for x, y in zone], outline=YELLOW + (230,), width=3)
        badge = f"REPLAY · eval set v2 · {s} ({meta.get('lighting', '')}) · {a.speed:g}× speed"
        tw = d.textlength(badge, font=f_badge)
        d.rectangle([16, 16, 16 + tw + 28, 16 + 44], fill=(0, 0, 0, 170))
        d.text((30, 24), badge, font=f_badge, fill=WHITE)
        clock = f"t = {t:6.1f} s"
        tw = d.textlength(clock, font=f_badge)
        d.rectangle([W - 16 - tw - 28, 16, W - 16, 60], fill=(0, 0, 0, 170))
        d.text((W - 16 - tw - 14, 24), clock, font=f_badge, fill=WHITE)
        dt = t - ev["arrive"]
        if dt >= 0:
            x1, y1, x2, y2 = ev["box"]
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=WHITE + (255,), width=3)
            tag = f"package let go · +{dt:.1f} s"
            tw = d.textlength(tag, font=f_small)
            ty = max(70, y1 * H - 30)
            d.rectangle([x1 * W, ty, x1 * W + tw + 16, ty + 26], fill=(0, 0, 0, 170))
            d.text((x1 * W + 8, ty + 3), tag, font=f_small, fill=WHITE)
        for i, b in enumerate(builds):
            y = H - 20 - (len(builds) - i) * 74
            d.rectangle([16, y, W - 16, y + 66], fill=(0, 0, 0, 175))
            d.text((32, y + 8), b["label"], font=f_label, fill=WHITE)
            m = b["metrics"]
            d.text((32, y + 41), f"recall {m['event_recall']:.2f} · {m['false_alerts_per_h']:.2f} false alerts/h · "
                   f"verdict HOLD", font=f_small, fill=GRAY)
            pe = b["by_arrive"].get(round(ev["arrive"], 3))
            text, color = status(pe, dt, stamp) if pe else ("not scored", GRAY)
            tw = d.textlength(text, font=f_status)
            d.text((W - 36 - tw, y + 17), text, font=f_status, fill=color)
        out.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))

    for i in idx:
        ev = arrivals[i]
        window = [(seq, t) for seq, t in times if ev["arrive"] - a.before <= t <= ev["arrive"] + a.after]
        for seq, t in window:
            render(seq, t, ev, stamp=False)
        for _ in range(int(round(a.hold * fps_out))):
            render(window[-1][0], window[-1][1], ev, stamp=True)
        shown.append({"index": i, "arrive": ev["arrive"], "frames": len(window),
                      **{b["label"]: b["by_arrive"].get(round(ev["arrive"], 3), {}).get("outcome") for b in builds}})
        print(f"arrival {i} at {ev['arrive']:.1f} s: {len(window)} frames; "
              + "; ".join(f"{b['label']}: {shown[-1][b['label']]}" for b in builds))
    out.release()
    n = sum(x["frames"] for x in shown) + len(shown) * int(round(a.hold * fps_out))
    Path(a.out + ".json").write_text(json.dumps({
        "session": s, "frames": str(fdir.relative_to(ROOT)) if fdir.is_relative_to(ROOT) else str(fdir),
        "speed": a.speed, "before_s": a.before, "after_s": a.after, "hold_s": a.hold,
        "builds": [{"label": b["label"], "record": b["record"]} for b in builds], "arrivals": shown}, indent=2) + "\n")
    print(f"{a.out}: {n} frames at {fps_out:g} fps = {n / fps_out:.1f} s")


if __name__ == "__main__":
    main()
