"""Turn per-frame detections into arrival alerts and score them against human marks.

Detections (one JSONL file per session, coordinates normalized to [0, 1]):
  {"t": 12.40, "dets": [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score, "person"], ...]}
Rows without a class are packages. Person boxes tell "hidden" apart from "empty".
Ground truth (one JSON file per session, marked by hand):
  {"session": "s03", "duration_s": 600.0,
   "events": [{"arrive": 41.2, "remove": 95.0}, {"arrive": 300.5, "remove": null}]}
`arrive` is when the hand lets go of the package; `remove` is null if it is still
there at the end. Packages already there when a session starts are not events.

Arrival rule (MVP: a first package into an empty zone; times, not frame counts):
  - empty zone, then a package seen for dwell_s -> one alert
  - no further alert while it stays; it is removed once the zone has been empty
    (not hidden) for clear_s, and a package placed after that is a new arrival
  - hidden (a person box covers the package) is not empty
  - a second package next to the first is not detected (out of MVP scope)
  - no frames for reconnect_s = camera away: the zone state is unknown, not empty.
    After it comes back, packages already there are the baseline and not alerted,
    so a package that arrived while the camera was away is never alerted.

Scoring: an alert is on time if alert time + notify_s (measured delivery delay) is
within match_window_s of the release; dwell_s is part of that budget.
  on time    first alert within the window after an arrival (counts toward recall)
  late       alert for a package that is there but past its window (not recall)
  duplicate  another alert while only already-alerted packages are there
  false      alert with no package there
Arrivals during a camera outage are misses, also counted as outage_misses.

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


def _overlap(a, b):
    """Fraction of box b covered by box a."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (b[2] - b[0]) * (b[3] - b[1])
    return max(w, 0) * max(h, 0) / area if area > 0 else 0.0


class ArrivalDetector:
    """Zone state machine for one camera: unknown | empty | occupied."""

    def __init__(self, p):
        self.p = p
        self.state = "unknown"
        self.last_t = self.last_box = None
        self.present_since = self.last_present = self.empty_since = None

    def status(self, now):
        """What the zone looks like now; unknown while the camera is away."""
        if self.last_t is None or now - self.last_t > self.p["reconnect_s"]:
            return "unknown"
        return self.state

    def _view(self, dets):
        p = self.p
        for d in dets:
            if (len(d) < 6 or d[5] == "package") and d[4] >= p["score_min"] \
                    and inside(((d[0] + d[2]) / 2, d[3]), p["zone"]):
                self.last_box = d[:4]
                return "present"
        if self.last_box and any(len(d) > 5 and d[5] == "person" and d[4] >= p["score_min"]
                                 and _overlap(d, self.last_box) >= p["hidden_overlap"] for d in dets):
            return "hidden"
        return "empty"

    def update(self, t, dets):
        """Feed one frame; True if it raises an arrival alert."""
        p = self.p
        if self.last_t is None or t - self.last_t > p["reconnect_s"]:
            self.state = "unknown"
            self.present_since = self.empty_since = None
        self.last_t = t
        view = self._view(dets)
        if view == "empty":
            self.empty_since = t if self.empty_since is None else self.empty_since
            if self.present_since is not None and t - self.last_present > p["gap_s"]:
                self.present_since = None
        else:
            self.empty_since = None
            if view == "present" and (self.present_since is None or t - self.last_present > p["gap_s"]):
                self.present_since = t
            if self.present_since is not None:
                self.last_present = t
        held = view != "empty" and self.present_since is not None and t - self.present_since >= p["dwell_s"]
        cleared = self.empty_since is not None and t - self.empty_since >= p["clear_s"]
        if cleared:
            self.present_since = None
        if self.state == "unknown":
            self.state = "occupied" if held else "empty" if cleared else "unknown"
        elif self.state == "empty" and held:
            self.state = "occupied"
            return True
        elif self.state == "occupied" and cleared:
            self.state = "empty"
        return False


def alerts(frames, p):
    """(alert times, camera outages as (start, end)) from [(t, dets)] sorted by t."""
    det = ArrivalDetector(p)
    out, outages, last_t = [], [], None
    for t, dets in frames:
        if last_t is not None and t - last_t > p["reconnect_s"]:
            outages.append((last_t, t))
        last_t = t
        if det.update(t, dets):
            out.append(t)
    return out, outages


def score_session(alert_times, events, window, notify_s=0.0, outages=()):
    """Classify alerts against the marked events of one session."""
    matched = {}  # event index -> effective alert time
    dup = late = false = 0

    def there(e, a):
        return e["arrive"] <= a <= (math.inf if e.get("remove") is None else e["remove"])

    for a in (a + notify_s for a in alert_times):
        cand = [i for i, e in enumerate(events) if i not in matched and e["arrive"] <= a <= e["arrive"] + window]
        if cand:
            matched[min(cand, key=lambda i: events[i]["arrive"])] = a
        elif any(there(e, a) for i, e in enumerate(events) if i not in matched):
            late += 1
        elif any(there(e, a) for e in events):
            dup += 1
        else:
            false += 1
    outage_misses = sum(1 for i, e in enumerate(events) if i not in matched
                        and any(s <= e["arrive"] <= t for s, t in outages))
    delays = [a - events[i]["arrive"] for i, a in matched.items()]
    return {"events": len(events), "matched": len(matched), "duplicate": dup, "late": late,
            "false": false, "outage_misses": outage_misses, "delays": delays}


def percentile(xs, q):
    """Nearest-rank percentile, or None for an empty list (the gate then holds)."""
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)]


def summarize(per_session, duration_s):
    """Pool session counts into the metric names the policy thresholds use.

    event_recall counts on-time alerts only; late alerts are reported apart.
    """
    n = sum(s["events"] for s in per_session)
    delays = [d for s in per_session for d in s["delays"]]
    hours = duration_s / 3600
    return {
        "event_recall": sum(s["matched"] for s in per_session) / n if n else None,
        "false_alerts_per_h": sum(s["false"] for s in per_session) / hours if hours else None,
        "duplicate_alerts": sum(s["duplicate"] for s in per_session),
        "late_alerts": sum(s["late"] for s in per_session),
        "outage_misses": sum(s.get("outage_misses", 0) for s in per_session),
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
        times, outages = alerts(read_frames(dets_path), p)
        s = score_session(times, gt["events"], p["match_window_s"], p["notify_s"], outages)
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
              f"{s['false']} false, {s['duplicate']} duplicate, {s['late']} late, "
              f"{s['outage_misses']} missed during camera outages", file=sys.stderr)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
