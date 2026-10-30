#!/usr/bin/env python3
"""Measure one compiled model on one device and write the record the gate reads.

Steps: copy the eval sessions' frames and the runner to the device, run the model
there (run_backend.py), pull back detections and timing, turn detections into
arrival alerts and score them against the hand-marked events (edgegate/events.py),
and write a measurement record. Any failure still writes a record, with
tool.ok = false, so the gate holds with the reason instead of finding nothing.

Eval sessions live in data/sessions/<name>/ (frames/, frames.csv, meta.json) and their
marks in data/gt/<name>.json. The eval set hash is the hash of a manifest that lists,
per session, the marks, frames.csv, meta.json (the lighting the audit slices by) and
a digest over every frame's own sha256. --eval-set-hash writes the full manifest,
one line per frame, to data/manifests/<hash>.json, so a changed image can be found.
Changing any frame, mark or session changes the hash and older records stop counting.
The device hashes the frames it actually read (run_backend.py); a digest that differs
from the manifest fails the measurement.

The record also carries which device and runtime ran (checked against the policy's
[targets]) and eval_config_sha256 over the policy's [events] and [measure] tables.
Its metrics are a recorded-session evaluation: alert times come from the recorded
frame times, device latency is a separate metric, and alert delivery is only in
them through a measured notify_s.

Usage:  python3 scripts/measure.py trt build/student_fp16.engine --onnx build/student.onnx \
            --host jetson --sessions e01,e02
        python3 scripts/measure.py hailo build/student.hef --onnx build/student.onnx \
            --host chans@192.168.50.151 --sessions e01,e02
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
from edgegate.gate import eval_config_sha256, load_policy, sha256_file  # noqa: E402

REMOTE = "edgegate_hackathon"
# Runtime -> backend name used by the policy and the gate.
GATE_NAME = {"onnx": "onnx-gx10", "trt": "trt", "hailo": "hailo8"}


def frame_lines(frames_dir):
    """One "NAME SHA256" line per frame in name order (same formula as run_backend.py)."""
    return [f"{p.name} {sha256_file(p)}" for p in sorted(Path(frames_dir).glob("*.jpg"))]


def eval_manifest(sessions, data):
    """{session: {gt, frames_csv, meta, frames, frames_sha256}} and the per-frame lines."""
    manifest, lines = {}, {}
    for s in sorted(sessions):
        sdir = data / "sessions" / s
        lines[s] = frame_lines(sdir / "frames")
        meta = sdir / "meta.json"
        split = json.loads(meta.read_text()).get("split") if meta.exists() else None
        if split != "eval":
            raise ValueError(f"session {s} is a {split!r} session, not eval")
        manifest[s] = {"gt": sha256_file(data / "gt" / f"{s}.json"),
                       "frames_csv": sha256_file(sdir / "frames.csv"),
                       "meta": sha256_file(meta) if meta.exists() else None,
                       "frames": len(lines[s]),
                       "frames_sha256": hashlib.sha256("".join(x + "\n" for x in lines[s]).encode()).hexdigest()}
    return manifest, lines


def eval_set_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


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
    # A 2-hour session is ~110k frames, over an hour at ~50 ms a frame (the slowest
    # builds, or the GX10 under load), so the time limit grows with the session.
    limit = 600 + 0.5 * sum(1 for _ in (session_dir / "frames").glob("*.jpg"))
    if a.host == "local":
        py = str(LOCAL_PY) if LOCAL_PY.exists() else sys.executable
        sh([py, str(ROOT / "scripts/run_backend.py"), a.backend, a.model,
            str(session_dir / "frames"), str(out_dir / name), "--conf", str(a.conf)], timeout=limit)
    else:
        rdir = f"{REMOTE}/measure"
        rsync = ["rsync", "-a", "-e", " ".join(SSH)]
        sh(SSH + [a.host, f"mkdir -p {rdir}/frames/{name} {REMOTE}/run"])
        sh(rsync + ["--delete", f"{session_dir}/frames/", f"{a.host}:{rdir}/frames/{name}/"])
        sh(rsync + [str(ROOT / "edgegate/detect.py"), str(ROOT / "scripts/run_backend.py"),
                    a.model, f"{a.host}:{REMOTE}/run/"])
        sh(SSH + [a.host,
            f"cd {REMOTE} && python3 run/run_backend.py {a.backend} run/{Path(a.model).name} "
            f"measure/frames/{name} measure/{name} --conf {a.conf}"], timeout=limit)
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
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "runs/records"))
    ap.add_argument("--eval-set-hash", action="store_true", help="only print the eval set hash")
    a = ap.parse_args()
    data, sessions = Path(a.data), a.sessions.split(",")
    if a.eval_set_hash:
        manifest, lines = eval_manifest(sessions, data)
        h = eval_set_hash(manifest)
        (data / "manifests").mkdir(exist_ok=True)
        (data / "manifests" / f"{h}.json").write_text(json.dumps(
            {"eval_set_sha256": h, "sessions": manifest, "frames": lines}, indent=1) + "\n")
        print(h)
        return
    if not (a.backend and a.model and a.onnx):
        ap.error("backend, model and --onnx are required")

    policy = load_policy(a.policy)
    classes, a.conf = policy["measure"]["classes"], policy["measure"]["det_conf"]
    record = {"schema": 2, "backend": a.name or GATE_NAME[a.backend], "runtime": a.backend,
              "device": {"label": a.device or a.host},
              "model_sha256": sha256_file(a.onnx), "artifact_sha256": sha256_file(a.model),
              "eval_set_sha256": None, "eval_config_sha256": eval_config_sha256(policy),
              "evaluation": "recorded sessions; alert delivery only through notify_s",
              "sessions": sessions, "measured_at": None, "tool": {"ok": False, "error": None}, "metrics": {}}
    try:
        manifest, _ = eval_manifest(sessions, data)
        record["eval_set_sha256"] = eval_set_hash(manifest)
        record["eval_manifest"] = manifest
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
                if summary["frames_sha256"] != manifest[s]["frames_sha256"]:
                    raise RuntimeError(f"device read different frames for {s} than the eval manifest")
                dev = record["device"]
                if "device_id" in dev and summary["device"] != {k: dev[k] for k in summary["device"]}:
                    raise RuntimeError(f"session {s} ran on a different device or runtime")
                dev.update(summary["device"])
                p50.append(summary["latency_ms_p50"])
                p95.append(summary["latency_ms_p95"])
                rss.append(summary["peak_rss_mb"])
                to_event_dets(out_dir, sdir, classes, dets_dir)
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
