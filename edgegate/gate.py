"""Rule engine: decide PASS/HOLD per backend from a policy and measurement records.

The rules alone decide. Glimmer can only veto (turn a PASS into a HOLD) and
explain; it can never approve a backend the rules held.

A measurement record (one JSON file per model x backend):
  {"schema": 1, "model_sha256": "...", "backend": "hailo8", "device": "pi5-hailo",
   "eval_set_sha256": "...", "measured_at": "2026-10-31T12:00:00Z",
   "tool": {"ok": true, "error": null},
   "metrics": {"event_recall": 0.93, "latency_ms_p95": 21.4, ...}}

Usage:  python3 -m edgegate.gate POLICY --model-sha SHA RECORD.json... [--backends a,b] [--now ISO]
"""
import argparse
import hashlib
import json
import math
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

PASS, HOLD = "PASS", "HOLD"


@dataclass
class Decision:
    backend: str
    verdict: str = PASS
    reasons: list = field(default_factory=list)  # (check, detail); empty when PASS
    vetoed_by: str = None

    def hold(self, check, detail):
        self.verdict = HOLD
        self.reasons.append((check, detail))


def load_policy(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_records(paths):
    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))
    return records


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _number(v):
    """The value as a finite float, or None (missing, wrong type, NaN, inf)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _parse_time(s):
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return t if t.tzinfo else None  # naive timestamps are ambiguous


def _valid_record(rec, model_sha, policy, now):
    """Problems that make a record unusable, as (check, detail) pairs."""
    task = policy["task"]
    problems = []
    if rec.get("schema") != 1:
        problems.append(("schema", f"unknown schema {rec.get('schema')!r}"))
    tool = rec.get("tool") or {}
    if tool.get("ok") is not True:
        problems.append(("tool", f"measurement tool failed: {tool.get('error')!r}"))
    if rec.get("model_sha256") != model_sha:
        problems.append(("model", "measured a different model"))
    if rec.get("eval_set_sha256") != task["eval_set_sha256"]:
        problems.append(("eval_set", "measured on a different eval set"))
    t = _parse_time(rec.get("measured_at"))
    if t is None:
        problems.append(("time", f"bad measured_at {rec.get('measured_at')!r}"))
    elif t > now + timedelta(minutes=5):
        problems.append(("time", f"measured_at {rec['measured_at']} is in the future"))
    elif now - t > timedelta(hours=task["max_age_hours"]):
        problems.append(("stale", f"measured_at {rec['measured_at']} is older than {task['max_age_hours']} h"))
    return problems


def _latest(records, backend):
    """Most recent record for a backend; unparseable times sort first."""
    mine = [r for r in records if r.get("backend") == backend]
    if not mine:
        return None
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return max(mine, key=lambda r: _parse_time(r.get("measured_at")) or epoch)


def decide(policy, records, model_sha, backends, now=None):
    """One Decision per backend in `backends`."""
    now = now or datetime.now(timezone.utc)
    ref_name = policy["task"].get("reference_backend")
    ref = _latest(records, ref_name) if ref_name else None
    ref_ok = ref is not None and not _valid_record(ref, model_sha, policy, now)
    decisions = []
    for backend in backends:
        d = Decision(backend)
        decisions.append(d)
        if policy["task"].get("eval_set_sha256") in (None, "", "unset"):
            d.hold("policy", "policy names no eval set")
            continue
        rec = _latest(records, backend)
        if rec is None:
            d.hold("missing", "no measurement record")
            continue
        for check, detail in _valid_record(rec, model_sha, policy, now):
            d.hold(check, detail)
        if d.verdict == HOLD:
            continue
        limits = {**policy.get("thresholds", {}), **policy.get("backends", {}).get(backend, {})}
        metrics = rec.get("metrics") or {}
        for name, lim in limits.items():
            v = _number(metrics.get(name))
            if v is None:
                d.hold(name, f"missing or not a finite number: {metrics.get(name)!r}")
                continue
            if "min" in lim and v < lim["min"]:
                d.hold(name, f"{v:g} < min {lim['min']:g}")
            if "max" in lim and v > lim["max"]:
                d.hold(name, f"{v:g} > max {lim['max']:g}")
            if "max_drop" in lim and backend != ref_name:
                r = _number((ref.get("metrics") or {}).get(name)) if ref_ok else None
                if r is None:
                    d.hold(name, f"no valid {ref_name} reference to compare against")
                elif r - v > lim["max_drop"]:
                    d.hold(name, f"{v:g} is {r - v:g} below reference {r:g} (max_drop {lim['max_drop']:g})")
    return decisions


def apply_veto(decisions, vetoes, by="glimmer"):
    """Apply audit vetoes {backend: reason}. A veto can only turn PASS into HOLD."""
    for d in decisions:
        if d.backend in vetoes and d.verdict == PASS:
            d.hold("veto", vetoes[d.backend])
            d.vetoed_by = by
    return decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy")
    ap.add_argument("records", nargs="*")
    ap.add_argument("--model-sha", required=True)
    ap.add_argument("--backends", help="comma-separated; default: every backend in the records")
    ap.add_argument("--now", help="ISO time to judge staleness against (default: now)")
    ap.add_argument("--json", help="also write the decisions to this file")
    a = ap.parse_args()
    policy = load_policy(a.policy)
    records = load_records(a.records)
    backends = a.backends.split(",") if a.backends else sorted({r.get("backend") for r in records} - {None})
    now = _parse_time(a.now) if a.now else None
    if a.now and now is None:
        sys.exit(f"bad --now {a.now!r} (need ISO time with timezone)")
    out = []
    for d in decide(policy, records, a.model_sha, backends, now):
        print(f"{d.backend:12s} {d.verdict}" + "".join(f"\n    {c}: {t}" for c, t in d.reasons))
        out.append({"backend": d.backend, "verdict": d.verdict, "reasons": d.reasons})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
