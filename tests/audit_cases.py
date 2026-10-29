"""Synthetic audit cases for tests and Glimmer wiring checks. Not evidence.

Every number here is made up so that a known problem sits in one slice while
the summary metrics pass. They check that the tools surface the problem and that
the veto path works; the real comparison uses measured candidates on real eval
sessions. Metrics are computed from the per-event rows with the same code the
measurement harness uses, so a record is self-consistent.

  control      all slices fine on both devices; any HOLD is a wrong HOLD
  night_gap    no night sessions in the eval set, but the deployment runs at night
  small_miss   hailo8 misses most small packages; overall recall still passes
  occluded     trt is late or misses in occluded scenes; overall recall still passes
  rule_hold    hailo8 over the latency limit: the rules hold it, no veto applies

Write the cases to disk (policy.toml, records/*.json, case.json per case):
  python3 tests/audit_cases.py OUT_DIR
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgegate.events import summarize  # noqa: E402
from edgegate.gate import eval_config_sha256  # noqa: E402

MODEL = "a" * 64
EVAL = "b" * 64
ARTIFACTS = {"trt": "c" * 64, "hailo8": "d" * 64}
MEASURED_AT = "2026-10-31T10:00:00Z"
NOW = "2026-10-31T12:00:00+00:00"
BOXES = {"small": [0.48, 0.70, 0.53, 0.75], "medium": [0.45, 0.65, 0.55, 0.78], "large": [0.40, 0.60, 0.62, 0.90]}
EXPECTED = {"control": {"trt": "PASS", "hailo8": "PASS"}, "night_gap": {"trt": "HOLD", "hailo8": "HOLD"},
            "small_miss": {"trt": "PASS", "hailo8": "HOLD"}, "occluded": {"trt": "HOLD", "hailo8": "PASS"},
            "rule_hold": {"trt": "PASS", "hailo8": "HOLD"}}


def policy_text():
    """The real policy, with the synthetic eval set and alert delivery marked measured."""
    text = (ROOT / "policies/package_entrance.toml").read_text()
    for old, new in (('eval_set_sha256 = "unset"', f'eval_set_sha256 = "{EVAL}"'),
                     ("notify_measured = false", "notify_measured = true")):
        assert old in text, old
        text = text.replace(old, new)
    return text


def policy():
    import tomllib
    return tomllib.loads(policy_text())


def device(backend):
    """A device block that matches the policy's target for the backend."""
    t = policy()["targets"][backend]
    return {"label": {"onnx-gx10": "gx10", "trt": "jetson-orin", "hailo8": "pi5-hailo"}[backend],
            "device_id": t["device_id"], "runtime": t["runtime"]}


def sessions(case):
    """(name, lighting, events) with 12 marked events per half-hour session: one small
    package and one occluded scene each, so a problem there moves overall recall only
    a little (3 of 72 events is a 0.042 drop, inside max_drop 0.05)."""
    lights = ["day", "day", "evening", "evening"] + (["day", "evening"] if case == "night_gap" else ["night", "night"])
    out = []
    for i, light in enumerate(lights):
        evs = []
        for j in range(12):
            size = "small" if j == 0 else ["medium", "large"][j % 2]
            tags = ["occluded"] if j == 6 else []
            evs.append({"arrive": 60.0 + 140 * j, "box": BOXES[size], "tags": tags, "size": size})
        out.append((f"e{i + 1:02d}", light, evs))
    return out


def outcome(case, backend, si, j, ev):
    if case == "small_miss" and backend == "hailo8" and ev["size"] == "small" and si in (1, 2, 4):
        return "missed"
    if case == "occluded" and backend == "trt" and "occluded" in ev["tags"] and si in (0, 2, 5):
        return "late" if si == 0 else "missed"
    # one ordinary miss on every backend, so recall is not a perfect 1.0
    return "missed" if (si, j) == (1, 5) else "on_time"


def record(case, backend):
    per_session = []
    for si, (name, light, evs) in enumerate(sessions(case)):
        rows = []
        for j, ev in enumerate(evs):
            o = outcome(case, backend, si, j, ev)
            rows.append({"arrive": ev["arrive"], "outcome": o, "delay": {"on_time": 3.1, "late": 14.0}.get(o),
                         "outage": False, "box": ev["box"], "tags": ev["tags"]})
        on_time = [r for r in rows if r["outcome"] == "on_time"]
        per_session.append({"session": name, "lighting": light, "duration_s": 1800.0, "events": len(rows),
                            "matched": len(on_time), "late": sum(r["outcome"] == "late" for r in rows),
                            "duplicate": 0, "false": 0, "outage_misses": 0,
                            "delays": [r["delay"] for r in on_time], "per_event": rows})
    metrics = summarize(per_session, sum(s["duration_s"] for s in per_session))
    lat = {"onnx-gx10": 32.0, "trt": 23.0, "hailo8": 60.0 if case == "rule_hold" else 14.0}[backend]
    metrics.update(latency_ms_p50=lat, latency_ms_p95=lat * 1.1, peak_mem_mb=400.0)
    return {"schema": 2, "backend": backend, "runtime": backend, "device": device(backend),
            "model_sha256": MODEL, "artifact_sha256": ARTIFACTS.get(backend, "e" * 64), "eval_set_sha256": EVAL,
            "eval_config_sha256": eval_config_sha256(policy()),
            "sessions": [s["session"] for s in per_session], "measured_at": MEASURED_AT,
            "tool": {"ok": True, "error": None}, "metrics": metrics, "per_session": per_session, "synthetic": True}


def write(out):
    text = policy_text()
    for case, expected in EXPECTED.items():
        d = Path(out) / case
        (d / "records").mkdir(parents=True, exist_ok=True)
        (d / "policy.toml").write_text(text)
        for b in ("onnx-gx10", "trt", "hailo8"):
            (d / "records" / f"{b}.json").write_text(json.dumps(record(case, b), indent=2) + "\n")
        (d / "case.json").write_text(json.dumps({"case": case, "synthetic": True, "expected": expected,
                                                 "model_sha256": MODEL, "artifacts": ARTIFACTS, "now": NOW}) + "\n")


if __name__ == "__main__":
    write(sys.argv[1])
