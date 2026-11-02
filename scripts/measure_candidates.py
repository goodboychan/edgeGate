#!/usr/bin/env python3
"""Measure every candidate of one build on its device, then judge each with the rules.

Reads build/RUN/ as scripts/build_candidates.sh writes it. The GX10 reference (ONNX)
and the two devices run at the same time, each device one candidate at a time so no
two runs share its timing. Records go to runs/records/RUN/<candidate>/ and the
reference to runs/records/RUN/reference/: the gate takes the latest record per
backend, so each candidate must be judged on its own records. All candidates are
recorded against student.onnx, the model they were built from; a fault is in the build.

Then prints one row per candidate and device: the rule verdict, the checks that held
it and the metrics. runs/records/RUN/table.json has the same with the full reasons.

Usage:  python3 scripts/measure_candidates.py s1 --sessions e01,e02,... [--only reference,clean]
        python3 scripts/measure_candidates.py s1 --table-only     # judge the records already there
Hosts: JETSON (default jetson) and HAILO_PI (default chans@192.168.50.151).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edgegate.gate import _latest, decide, load_policy, load_records, sha256_file  # noqa: E402
from measure import GATE_NAME  # noqa: E402

# candidate -> {runtime: file in build/RUN}, as build_candidates.sh names them
CANDIDATES = {
    "clean": {"trt": "student_fp16.engine", "hailo": "student.hef"},
    "bgr": {"trt": "student_bgr_fp16.engine", "hailo": "student_bgr.hef"},
    "no_norm": {"trt": "student_no_norm_fp16.engine", "hailo": "student_no_norm.hef"},
    "w4": {"hailo": "student_w4.hef"},
    "calib10": {"hailo": "student_calib10.hef"},
}
COLUMNS = {"event_recall": "recall", "false_alerts_per_h": "false/h", "duplicate_alerts": "dup",
           "detection_delay_s_p95": "delay p95 s", "latency_ms_p95": "latency p95 ms", "peak_mem_mb": "mem MB"}


def run_queue(jobs, logs):
    """One device: its measure.py runs one after another."""
    for name, cmd in jobs:
        with open(logs / f"{name}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: {'done' if r.returncode == 0 else f'exit {r.returncode}'} (logs/{name}.log)", flush=True)


def measure(a, names, build, out):
    hosts = {"onnx": "local", "trt": os.environ.get("JETSON", "jetson"),
             "hailo": os.environ.get("HAILO_PI", "chans@192.168.50.151")}
    onnx = build / "student.onnx"

    def cmd(runtime, model, dest):
        return [sys.executable, str(ROOT / "scripts/measure.py"), runtime, str(model), "--onnx", str(onnx),
                "--host", hosts[runtime], "--sessions", a.sessions, "--policy", a.policy, "--data", a.data,
                "--out", str(dest)]

    queues = {"onnx": [], "trt": [], "hailo": []}
    if "reference" in names:
        queues["onnx"].append(("reference_onnx", cmd("onnx", onnx, out / "reference")))
    for c in names:
        for runtime, f in CANDIDATES.get(c, {}).items():
            queues[runtime].append((f"{c}_{runtime}", cmd(runtime, build / f, out / c)))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    threads = [threading.Thread(target=run_queue, args=(q, out / "logs")) for q in queues.values() if q]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def judge(policy, names, build, out, now=None):
    """Rule verdicts per candidate, each on its own records plus the reference (`now`: replay a date)."""
    ref = load_records(sorted((out / "reference").glob("*.json")))
    model_sha = sha256_file(build / "student.onnx")
    metrics = lambda rec: {k: ((rec or {}).get("metrics") or {}).get(k) for k in COLUMNS}  # noqa: E731
    rows = [{"candidate": "reference", "backend": GATE_NAME["onnx"], "verdict": "-", "reasons": [],
             "metrics": metrics(_latest(ref, GATE_NAME["onnx"])), "model_sha256": model_sha}]
    for c in names:
        if c not in CANDIDATES:
            continue
        recs = load_records(sorted((out / c).glob("*.json")))
        artifacts = {GATE_NAME[r]: sha256_file(build / f) if (build / f).exists() else None
                     for r, f in CANDIDATES[c].items()}
        for d in decide(policy, ref + recs, model_sha, artifacts, now):
            rec = _latest(recs, d.backend)
            rows.append({"candidate": c, "backend": d.backend, "verdict": d.verdict,
                         "reasons": [f"{k}: {t}" for k, t in d.reasons], "metrics": metrics(rec),
                         "model_sha256": model_sha, "artifact_sha256": artifacts[d.backend],
                         "measured": rec is not None, "measured_at": (rec or {}).get("measured_at")})
    return rows


def show(rows):
    fmt = lambda v: "-" if v is None else f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    print("| candidate | backend | verdict | " + " | ".join(COLUMNS.values()) + " | held for |")
    print("|" + " --- |" * (len(COLUMNS) + 4))
    for r in rows:
        held = ", ".join(dict.fromkeys(x.split(":")[0] for x in r["reasons"]))
        print(f"| {r['candidate']} | {r['backend']} | {r['verdict']} | "
              + " | ".join(fmt(r["metrics"][k]) for k in COLUMNS) + f" | {held} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--sessions", help="comma-separated eval session names")
    ap.add_argument("--only", help=f"comma-separated, from reference,{','.join(CANDIDATES)} (default: all)")
    ap.add_argument("--table-only", action="store_true", help="only judge the records already there")
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args()
    names = a.only.split(",") if a.only else ["reference", *CANDIDATES]
    unknown = set(names) - {"reference", *CANDIDATES}
    if unknown:
        ap.error(f"unknown candidates {sorted(unknown)}")
    build, out = ROOT / "build" / a.run, ROOT / "runs/records" / a.run
    if not a.table_only:
        if not a.sessions:
            ap.error("--sessions is required unless --table-only")
        measure(a, names, build, out)
    rows = judge(load_policy(a.policy), names if a.only else list(CANDIDATES), build, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(rows, indent=2) + "\n")
    show(rows)


if __name__ == "__main__":
    main()
