#!/usr/bin/env python3
"""Measure one compiled model on one device and write the record the gate reads.

Steps: copy the eval sessions' frames and the runner to the device, run the model
there (run_backend.py), pull back detections and timing, turn detections into
arrival alerts and score them against the hand-marked events (edgegate/events.py),
and write a measurement record. Any failure still writes a record, with
tool.ok = false, so the gate holds with the reason instead of finding nothing.

Eval sessions live in data/sessions/<name>/ (frames/, frames.csv, meta.json) and their
marks in data/gt/<name>.json. The eval set hash covers all three (meta.json holds the
lighting the audit slices by), so changing a mark or a session changes the hash and
older records stop counting.

Usage:  python3 scripts/measure.py trt build/student_fp16.engine --onnx build/student.onnx \
            --host jetson --sessions e01,e02 --classes package,person
        python3 scripts/measure.py hailo build/student.hef --onnx build/student.onnx \
            --host chans@192.168.50.151 --sessions e01,e02 --classes package,person
        python3 scripts/measure.py onnx build/student.onnx --onnx build/student.onnx --host local ...
        python3 scripts/measure.py --eval-set-hash --sessions e01,e02   # value for the policy
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgegate.events import evaluate  # noqa: E402
from edgegate.gate import load_policy, sha256_file  # noqa: E402

REMOTE = "edgegate_hackathon"
# Runtime -> backend name used by the policy and the gate.
GATE_NAME = {"onnx": "onnx-gx10", "trt": "trt", "hailo": "hailo8"}


def eval_set_hash(sessions, data):
    h = hashlib.sha256()
    for s in sorted(sessions):
        h.update(s.encode())
        h.update(sha256_file(data / "gt" / f"{s}.json").encode())
        h.update(sha256_file(data / "sessions" / s / "frames.csv").encode())
        meta = data / "sessions" / s / "meta.json"
        h.update(sha256_file(meta).encode() if meta.exists() else b"no meta")
    return h.hexdigest()


# One shared ssh connection per host: the Pi resets bursts of new connections.
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", "ControlPersist=120",
       "-o", "ControlPath=/tmp/edgegate-ssh-%r@%h:%p"]
LOCAL_PY = ROOT / ".venv-train/bin/python"  # has onnxruntime for the GX10 reference


def sh(cmd, timeout=3600, tries=3):
    for i in range(tries):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
        # 255 is ssh's own failure (connection dropped), worth one more try
        if r.returncode != 255 or i == tries - 1:
            raise RuntimeError(f"{' '.join(cmd[:3])}... exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
        time.sleep(3)


def run_on_device(a, session_dir, out_dir):
    """Run the backend over one session's frames; returns the summary dict."""
    name = session_dir.name
    if a.host == "local":
        py = str(LOCAL_PY) if LOCAL_PY.exists() else sys.executable
        sh([py, str(ROOT / "scripts/run_backend.py"), a.backend, a.model,
            str(session_dir / "frames"), str(out_dir / name)])
    else:
        rdir = f"{REMOTE}/measure"
        rsync = ["rsync", "-a", "-e", " ".join(SSH)]
        sh(SSH + [a.host, f"mkdir -p {rdir}/frames/{name} {REMOTE}/run"])
        sh(rsync + ["--delete", f"{session_dir}/frames/", f"{a.host}:{rdir}/frames/{name}/"])
        sh(rsync + [str(ROOT / "edgegate/detect.py"), str(ROOT / "scripts/run_backend.py"),
                    a.model, f"{a.host}:{REMOTE}/run/"])
        sh(SSH + [a.host,
            f"cd {REMOTE} && python3 run/run_backend.py {a.backend} run/{Path(a.model).name} "
            f"measure/frames/{name} measure/{name}"])
        sh(rsync + [f"{a.host}:{rdir}/{name}.jsonl", f"{a.host}:{rdir}/{name}.json", f"{out_dir}/"])
    return json.loads((out_dir / f"{name}.json").read_text())


def to_event_dets(out_dir, session_dir, classes, dets_dir):
    """Detections keyed by frame file -> events format keyed by session time."""
    t_of = {f"{int(r['seq']):06d}.jpg": float(r["t"]) for r in csv.DictReader(open(session_dir / "frames.csv"))}
    with open(dets_dir / f"{session_dir.name}.jsonl", "w") as f:
        for r in map(json.loads, open(out_dir / f"{session_dir.name}.jsonl")):
            dets = [d[:5] + [classes[d[5]]] for d in r["dets"] if d[5] < len(classes)]
            f.write(json.dumps({"t": t_of[r["name"]], "dets": dets}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("backend", nargs="?", choices=["onnx", "trt", "hailo"])
    ap.add_argument("model", nargs="?", help="the compiled file that would be deployed")
    ap.add_argument("--onnx", help="the ONNX it was built from (model_sha256)")
    ap.add_argument("--host", default="local", help="ssh host of the device, or local")
    ap.add_argument("--device", help="device label in the record (default: host)")
    ap.add_argument("--name", help="backend name for the gate (default: onnx-gx10, trt, hailo8)")
    ap.add_argument("--sessions", required=True, help="comma-separated eval session names")
    ap.add_argument("--classes", default="package,person", help="class names in model order")
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "runs/records"))
    ap.add_argument("--eval-set-hash", action="store_true", help="only print the eval set hash")
    a = ap.parse_args()
    data, sessions = Path(a.data), a.sessions.split(",")
    if a.eval_set_hash:
        print(eval_set_hash(sessions, data))
        return
    if not (a.backend and a.model and a.onnx):
        ap.error("backend, model and --onnx are required")

    record = {"schema": 1, "backend": a.name or GATE_NAME[a.backend], "runtime": a.backend,
              "device": a.device or a.host,
              "model_sha256": sha256_file(a.onnx), "artifact_sha256": sha256_file(a.model),
              "eval_set_sha256": None, "sessions": sessions,
              "measured_at": None, "tool": {"ok": False, "error": None}, "metrics": {}}
    try:
        record["eval_set_sha256"] = eval_set_hash(sessions, data)
        policy = load_policy(a.policy)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, dets_dir = Path(tmp, "out"), Path(tmp, "dets")
            out_dir.mkdir()
            dets_dir.mkdir()
            gt_dir = Path(tmp, "gt")
            gt_dir.mkdir()
            p50, p95, rss = [], [], []
            for s in sessions:
                sdir = data / "sessions" / s
                summary = run_on_device(a, sdir, out_dir)
                if summary["model_sha256"] != record["artifact_sha256"]:
                    raise RuntimeError(f"device ran {summary['model_sha256'][:12]}, not the file given")
                p50.append(summary["latency_ms_p50"])
                p95.append(summary["latency_ms_p95"])
                rss.append(summary["peak_rss_mb"])
                to_event_dets(out_dir, sdir, a.classes.split(","), dets_dir)
                shutil.copy(data / "gt" / f"{s}.json", gt_dir)
            metrics, per_session = evaluate(policy, gt_dir, dets_dir)
        for ps in per_session:  # for the audit's lighting slices
            meta = data / "sessions" / ps["session"] / "meta.json"
            ps["lighting"] = json.loads(meta.read_text()).get("lighting", "unknown") if meta.exists() else "unknown"
        # Worst session, not the mean: a device that is slow on one session is slow.
        metrics.update({"latency_ms_p50": max(p50), "latency_ms_p95": max(p95), "peak_mem_mb": max(rss)})
        record["metrics"] = metrics
        record["per_session"] = per_session
        record["tool"]["ok"] = True
    except Exception as e:  # noqa: BLE001 - every failure must become a HOLD reason
        record["tool"]["error"] = f"{type(e).__name__}: {e}"[:500]
    record["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['backend']}_{record['artifact_sha256'][:12]}_{record['measured_at'].replace(':', '')}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{path}: tool ok={record['tool']['ok']} {record['tool']['error'] or ''}")
    print(json.dumps(record["metrics"], indent=2))


if __name__ == "__main__":
    main()
