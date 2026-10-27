"""Turn per-frame detections into arrival alerts and score them against human marks.

Detections (one JSONL file per session, coordinates normalized to [0, 1]):
  {"t": 12.40, "dets": [[x1, y1, x2, y2, score], ...]}
Ground truth (one JSON file per session, marked by hand):
  {"session": "s03", "duration_s": 600.0,
   "events": [{"arrive": 41.2, "remove": 95.0}, {"arrive": 300.5, "remove": null}]}
`remove` is null when the package is still there at the end of the session.
Packages already there when a session starts are not events.

Arrival rule: the number of packages in the zone rises and stays up for dwell_s.
No new alert while it stays; a removed package placed again, or a second package
set down next to the first, is a new arrival. After the camera starts or comes
back (no frames for reconnect_s), packages already there are taken as the
baseline and not alerted.

Alert outcomes:
  matched    first alert within match_window_s after an arrival
  late       alert while an unmatched package is there, after its window
  duplicate  another alert while only already-matched packages are there
  false      alert with no package there

Usage:  python3 -m edgegate.events POLICY GT_DIR DETS_DIR   (files paired by session name)
"""
import argparse
import json
import math
import sys
from pathlib import Path

from edgegate.gate import load_policy


def inside(pt, poly):
    """Ray-casting point-in-polygon test."""
    x, y = pt
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def count(dets, zone, score_min):
    """Number of confident boxes with their bottom-center in the zone."""
    return sum(s >= score_min and inside(((x1 + x2) / 2, y2), zone) for x1, y1, x2, y2, s in dets)


def alerts(frames, p):
    """Alert times from [(t, dets)] sorted by t, using the [events] policy section."""
    out = []
    stable = None        # accepted package count; None until it settles after (re)connect
    level = since = None  # settling: count seen and since when
    up_since = up_last = None  # run of frames above stable; dips shorter than gap_s are occlusion
    down_since = None
    last_t = None
    for t, dets in frames:
        if last_t is not None and t - last_t > p["reconnect_s"]:
            stable = level = up_since = down_since = None
        last_t = t
        n = count(dets, p["zone"], p["score_min"])
        if stable is None:
            if n != level:
                level, since = n, t
            elif t - since >= p["dwell_s"]:
                stable = n
            continue
        if n > stable:
            if up_since is None or t - up_last > p["gap_s"]:
                up_since = t
            up_last = t
            down_since = None
            if t - up_since >= p["dwell_s"]:
                out.append(t)
                stable, up_since = n, None
        elif n < stable:
            down_since = t if down_since is None else down_since
            if t - down_since >= p["clear_s"]:
                stable, down_since = n, None
        else:
            down_since = None
    return out


def score_session(alert_times, events, window):
    """Classify alerts against the marked events of one session."""
    matched = {}  # event index -> alert time
    dup = late = false = 0

    def there(e, a):
        return e["arrive"] <= a <= (math.inf if e.get("remove") is None else e["remove"])

    for a in alert_times:
        cand = [i for i, e in enumerate(events) if i not in matched and e["arrive"] <= a <= e["arrive"] + window]
        if cand:
            matched[min(cand, key=lambda i: events[i]["arrive"])] = a
        elif any(there(e, a) for i, e in enumerate(events) if i not in matched):
            late += 1
        elif any(there(e, a) for e in events):
            dup += 1
        else:
            false += 1
    delays = [a - events[i]["arrive"] for i, a in matched.items()]
    return {"events": len(events), "matched": len(matched), "duplicate": dup,
            "late": late, "false": false, "delays": delays}


def percentile(xs, q):
    """Nearest-rank percentile, or None for an empty list (the gate then holds)."""
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)]


def summarize(per_session, duration_s):
    """Pool session counts into the metric names the policy thresholds use."""
    n = sum(s["events"] for s in per_session)
    delays = [d for s in per_session for d in s["delays"]]
    hours = duration_s / 3600
    return {
        "event_recall": sum(s["matched"] for s in per_session) / n if n else None,
        "false_alerts_per_h": sum(s["false"] for s in per_session) / hours if hours else None,
        "duplicate_alerts": sum(s["duplicate"] for s in per_session),
        "late_alerts": sum(s["late"] for s in per_session),
        "detection_delay_s_p50": percentile(delays, 50),
        "detection_delay_s_p95": percentile(delays, 95),
    }


def read_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                frames.append((r["t"], r["dets"]))
    return sorted(frames, key=lambda fr: fr[0])


def evaluate(policy, gt_dir, dets_dir):
    p = policy["events"]
    per_session, duration = [], 0.0
    for gt_path in sorted(Path(gt_dir).glob("*.json")):
        gt = json.loads(gt_path.read_text())
        dets_path = Path(dets_dir) / f"{gt['session']}.jsonl"
        if not dets_path.exists():
            # A missing session would silently raise recall, so refuse to score.
            raise FileNotFoundError(f"no detections for session {gt['session']}: {dets_path}")
        s = score_session(alerts(read_frames(dets_path), p), gt["events"], p["match_window_s"])
        s["session"] = gt["session"]
        per_session.append(s)
        duration += gt["duration_s"]
    if not per_session:
        raise FileNotFoundError(f"no ground-truth sessions in {gt_dir}")
    return summarize(per_session, duration), per_session


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy")
    ap.add_argument("gt_dir")
    ap.add_argument("dets_dir")
    a = ap.parse_args()
    try:
        metrics, per_session = evaluate(load_policy(a.policy), a.gt_dir, a.dets_dir)
    except FileNotFoundError as e:
        sys.exit(str(e))
    for s in per_session:
        print(f"{s['session']}: {s['matched']}/{s['events']} matched, "
              f"{s['false']} false, {s['duplicate']} duplicate, {s['late']} late", file=sys.stderr)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
