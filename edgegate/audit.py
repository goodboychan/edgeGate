"""Audit tools: the extra checks Glimmer can ask for after the rules have decided.

Glimmer never reads files directly. It calls these tools (served over MCP by
edgegate/audit_mcp.py), and every result gets an evidence id (E1, E2, ...). Its
only power is propose_hold, which is accepted only when
  - the rules passed that backend (a veto cannot touch a rule HOLD), and
  - it cites evidence ids from this audit, at least one of them about that
    backend or about the eval set as a whole (eval_coverage).
Whether the evidence really shows a problem is Glimmer's judgment; wrong HOLDs on
good candidates are counted in the comparison, so a veto has a cost.

Slices come from the per-event and per-session rows in the measurement record
(edgegate/events.py): lighting from the session's meta.json, size from the marked
package box, tags from the marks. They never change a rule metric.
"""
import math
from collections import defaultdict

from edgegate.gate import PASS, _latest, decide, sha256_file

SLICES = ("lighting", "size", "tag", "session")


def artifact_sha(value):
    """BACKEND=VALUE spec value: a 64-hex sha as given (replays, tests), else the file's sha256."""
    if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return value
    return sha256_file(value)


def _size(box, edges):
    if not box:
        return "unknown"
    side = math.sqrt(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))
    return "small" if side < edges[0] else "medium" if side < edges[1] else "large"


def _ratio(a, b):
    return round(a / b, 3) if b else None


def slice_rows(record, by, size_edges):
    """{slice: counts} for one record. Events count once per slice; tags can overlap.
    A second package (an arrival while another is still there) is out of scope and not in
    the gate's recall (edgegate/events.py), so it counts under out_of_scope, not in events."""
    out = defaultdict(lambda: {"events": 0, "on_time": 0, "late": 0, "missed": 0, "out_of_scope": 0})
    hours = defaultdict(float)
    false = defaultdict(int)
    for s in record.get("per_session") or []:
        if by in ("lighting", "session"):
            key = s.get("lighting", "unknown") if by == "lighting" else s["session"]
            hours[key] += s.get("duration_s", 0) / 3600
            false[key] += s.get("false", 0)
        for e in s.get("per_event") or []:
            if by == "size":
                keys = [_size(e.get("box"), size_edges)]
            elif by == "tag":
                keys = e.get("tags") or ["untagged"]
            else:
                keys = [s.get("lighting", "unknown") if by == "lighting" else s["session"]]
            for k in keys:
                if e.get("second"):
                    out[k]["out_of_scope"] += 1
                    continue
                out[k]["events"] += 1
                out[k][e["outcome"]] += 1
    for k in hours:
        out[k]  # sessions with no events still show their hours and false alerts
    rows = {}
    for k, c in sorted(out.items()):
        rows[k] = {**c, "recall": _ratio(c["on_time"], c["events"])}
        if k in hours:
            rows[k].update(hours=round(hours[k], 3), false_alerts=false[k],
                           false_alerts_per_h=_ratio(false[k], hours[k]))
    return rows


class Audit:
    """One audit of one candidate: rule decisions, tool results, accepted vetoes."""

    def __init__(self, policy, records, model_sha, artifacts, now=None):
        self.policy = policy
        self.ref_name = policy["task"].get("reference_backend")
        self.decisions = {d.backend: d for d in decide(policy, records, model_sha, artifacts, now)}
        self.records = {b: _latest(records, b) for b in list(self.decisions) + [self.ref_name]}
        self.evidence = {}  # id -> {"tool", "backend", "result"}
        self.vetoes = {}  # backend -> {"reason", "evidence"}
        self.rejected = []

    def _cite(self, tool, backend, result):
        eid = f"E{len(self.evidence) + 1}"
        self.evidence[eid] = {"tool": tool, "backend": backend, "result": result}
        return {"evidence_id": eid, **result}

    def gate_decisions(self):
        """Rule verdicts, reasons and summary metrics per backend, and the reference."""
        out = {}
        for b, d in self.decisions.items():
            rec = self.records.get(b) or {}
            out[b] = {"verdict": d.verdict, "reasons": [f"{c}: {t}" for c, t in d.reasons],
                      "device": rec.get("device"), "metrics": rec.get("metrics")}
        ref = self.records.get(self.ref_name) or {}
        return self._cite("gate_decisions", None, {
            "backends": out, "reference": {"backend": self.ref_name, "metrics": ref.get("metrics")},
            "thresholds": self.policy.get("thresholds"), "backend_overrides": self.policy.get("backends", {})})

    def eval_coverage(self):
        """What the eval set covers: sessions, hours and events by lighting, size and tag."""
        rec = next((r for r in self.records.values() if r and r.get("per_session")), None)
        if rec is None:
            return self._cite("eval_coverage", None, {"error": "no record with per-session results"})
        edges = self.policy.get("slices", {}).get("size_edges", [0.08, 0.15])
        light = slice_rows(rec, "lighting", edges)
        return self._cite("eval_coverage", None, {
            "sessions": len(rec["per_session"]),
            "by_lighting": {k: {"sessions": sum(1 for s in rec["per_session"] if s.get("lighting", "unknown") == k),
                                "hours": v["hours"], "events": v["events"]} for k, v in light.items()},
            "events_by_size": {k: v["events"] for k, v in slice_rows(rec, "size", edges).items()},
            "events_by_tag": {k: v["events"] for k, v in slice_rows(rec, "tag", edges).items()},
            "size_edges": edges})

    def slice_metrics(self, backend, by):
        """Recall (on time) per slice for a backend, next to the reference on the same slices."""
        if by not in SLICES:
            return self._cite("slice_metrics", backend, {"error": f"by must be one of {list(SLICES)}"})
        rec = self.records.get(backend)
        if backend not in self.records or rec is None:
            known = [b for b, r in self.records.items() if r]
            return self._cite("slice_metrics", backend, {
                "error": f"no record for backend {backend!r}; use a backend name, not a device label: {known}"})
        edges = self.policy.get("slices", {}).get("size_edges", [0.08, 0.15])
        ref = self.records.get(self.ref_name)
        rows = slice_rows(rec, by, edges)
        ref_rows = slice_rows(ref, by, edges) if ref else {}
        for k, r in rows.items():
            # A count, not a ratio: one event in a small slice is one event.
            if k in ref_rows:
                r["on_time_minus_reference"] = r["on_time"] - ref_rows[k]["on_time"]
        return self._cite("slice_metrics", backend, {
            "backend": backend, "by": by, "slices": rows,
            "reference": {"backend": self.ref_name, "slices": ref_rows or None}})

    def propose_hold(self, backend, reason, evidence):
        """Veto a backend the rules passed; accepted only with evidence from this audit."""
        d = self.decisions.get(backend)
        problem = None
        if d is None:
            problem = f"unknown backend {backend!r}; backends: {list(self.decisions)}"
        elif d.verdict != PASS:
            problem = "the rules already hold this backend; a veto changes nothing"
        elif not reason or not reason.strip():
            problem = "give a reason"
        elif not evidence or any(e not in self.evidence for e in evidence):
            problem = f"cite evidence ids returned by tools in this audit; known: {list(self.evidence)}"
        elif not any(self.evidence[e]["tool"] == "eval_coverage" or self.evidence[e]["backend"] == backend
                     for e in evidence):
            problem = "cite at least one result about this backend (slice_metrics) or eval_coverage"
        if problem:
            self.rejected.append({"backend": backend, "reason": reason, "evidence": evidence, "problem": problem})
            return {"accepted": False, "problem": problem}
        self.vetoes[backend] = {"reason": reason.strip(), "evidence": list(evidence)}
        return {"accepted": True, "backend": backend, "verdict_after_audit": "HOLD"}
