#!/usr/bin/env python3
"""Deploy a candidate only to the devices the rules, and the audit if given, passed.

The rule engine decides PASS or HOLD per backend from the measurement records
(edgegate/gate.py). An audit record from scripts/audit.py adds its accepted vetoes,
which can only turn a PASS into a HOLD. Then per backend:
  HOLD  prints the reasons and deploys nothing. There is no override.
  PASS  writes an approval record to runs/approvals/, copies the model file, the
        approval and the live runner to the device and starts it there
        (scripts/live_device.py), which checks the approval names that device,
        its runtime and that exact file before it runs.

The approval record is the proof the demo shows: model, artifact, device, runtime,
eval set and rule settings, the metrics that passed, the record they came from and
the audit that checked them.

Usage:  python3 scripts/deploy.py POLICY RECORD.json... --model-sha SHA \
            --artifact trt=build/s1/student_fp16.engine --artifact hailo8=build/s1/student.hef \
            [--audit runs/audits/X.json] --notify https://ntfy.sh/TOPIC
        python3 scripts/deploy.py --stop trt,hailo8
Then stream the camera to each started device: capture_oak.py ... --stream HOST:8790
Hosts: JETSON (default jetson) and HAILO_PI (default hailo, an ssh alias).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgegate.gate import (PASS, _latest, _parse_time, apply_veto, decide, eval_config_sha256,  # noqa: E402
                           load_policy, sha256_file)

RUNTIME = {"trt": "trt", "hailo8": "hailo"}
HOSTS = {"trt": os.environ.get("JETSON", "jetson"), "hailo8": os.environ.get("HAILO_PI", "hailo")}
LIVE = "edgegate_hackathon/live"
SSH = ["ssh", "-o", "BatchMode=yes"]
FILES = ["scripts/live_device.py", "scripts/run_backend.py", "edgegate/detect.py", "edgegate/events.py",
         "edgegate/stream.py"]
# Stop the runner this folder started, by its PID; nothing else.
STOP = (f"if [ -f {LIVE}/live.pid ] && ps -p $(cat {LIVE}/live.pid) -o args= | grep -q live_device.py; "
        f"then kill $(cat {LIVE}/live.pid); sleep 1; fi; rm -f {LIVE}/live.pid")


def audit_vetoes(path, model_sha, artifacts):
    """{backend: (reason, by)} the audit held; it must be about this model and these files."""
    audit = json.loads(Path(path).read_text())
    if audit.get("model_sha256") != model_sha or audit.get("artifacts") != artifacts:
        raise SystemExit(f"{path} audited another model or other files")
    return audit, {f["backend"]: (f["reasons"][-1][1], f["vetoed_by"])
                   for f in audit["final"] if f.get("vetoed_by")}


def approval(policy, a, backend, rec, path, artifact_sha, audit):
    task = policy["task"]
    limits = {**policy.get("thresholds", {}), **policy.get("backends", {}).get(backend, {})}
    return {
        "schema": 1, "task": task["name"], "instruction": task["instruction"], "backend": backend,
        "verdict": PASS, "model_sha256": a.model_sha, "artifact_sha256": artifact_sha,
        "artifact_file": Path(a.files[backend]).name, "device_id": rec["device"]["device_id"],
        "runtime": rec["device"]["runtime"], "device": rec["device"],
        "eval_set_sha256": rec["eval_set_sha256"], "eval_config_sha256": eval_config_sha256(policy),
        "events": policy["events"], "measure": policy["measure"], "metrics": rec["metrics"], "thresholds": limits,
        "record": {"path": str(path), "sha256": sha256_file(path), "measured_at": rec["measured_at"]},
        "audit": None if audit is None else {"path": str(a.audit), "sha256": sha256_file(a.audit),
                                             "label": audit.get("label"), "audited_at": audit.get("audited_at"),
                                             "vetoes": audit.get("vetoes"), "error": audit.get("error")},
        "policy_sha256": sha256_file(a.policy), "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def start(backend, model, approval_path, notify, port):
    host = HOSTS[backend]
    subprocess.run(SSH + [host, f"{STOP}; mkdir -p {LIVE}"], check=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), *[str(ROOT / f) for f in FILES], str(model),
                    f"{host}:{LIVE}/"], check=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), str(approval_path), f"{host}:{LIVE}/approval.json"], check=True)
    # The braces keep nohup itself in the background, so live.pid is the runner's own PID.
    run = (f"cd {LIVE} && {{ nohup python3 live_device.py {RUNTIME[backend]} {Path(model).name} "
           f"--approval approval.json --notify '{notify}' --label {backend} --port {port} "
           f"> live.log 2>&1 < /dev/null & echo $! > live.pid; }}")
    subprocess.run(SSH + [host, run], check=True)
    for _ in range(60):  # the runner loads the model, then says it listens or why it refused
        time.sleep(1)
        log = subprocess.run(SSH + [host, f"cat {LIVE}/live.log"], capture_output=True, text=True).stdout
        if "listening on" in log or "not starting" in log or "Error" in log:
            return log.strip().splitlines()[-1]
    return "no answer from the runner within 60 s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy", nargs="?")
    ap.add_argument("records", nargs="*")
    ap.add_argument("--model-sha")
    ap.add_argument("--artifact", action="append", default=[], metavar="BACKEND=PATH")
    ap.add_argument("--audit", help="audit record from scripts/audit.py for these files")
    ap.add_argument("--notify", help="URL the devices POST alerts to (an ntfy topic, for example)")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--now", help="ISO time to judge staleness against; only shows decisions, deploys nothing")
    ap.add_argument("--stop", help="comma-separated backends whose live runner to stop")
    a = ap.parse_args()
    if a.stop:
        for b in a.stop.split(","):
            subprocess.run(SSH + [HOSTS[b], STOP], check=True)
            print(f"{b}: stopped")
        return
    if not (a.policy and a.model_sha and a.artifact and a.notify):
        ap.error("policy, records, --model-sha, --artifact and --notify are required")
    policy = load_policy(a.policy)
    a.files = dict(spec.partition("=")[::2] for spec in a.artifact)
    unknown = set(a.files) - set(RUNTIME)
    if unknown:
        ap.error(f"deploy targets are {sorted(RUNTIME)}, not {sorted(unknown)}")
    artifacts = {b: sha256_file(p) for b, p in a.files.items()}
    paths = {}
    records = []
    for p in a.records:
        rec = json.loads(Path(p).read_text())
        paths[id(rec)] = p
        records.append(rec)
    now = _parse_time(a.now) if a.now else None
    decisions = decide(policy, records, a.model_sha, artifacts, now)
    audit = None
    if a.audit:
        audit, vetoes = audit_vetoes(a.audit, a.model_sha, artifacts)
        for b, (reason, by) in vetoes.items():
            apply_veto(decisions, {b: reason}, by=by)
    out = ROOT / "runs/approvals"
    out.mkdir(parents=True, exist_ok=True)
    for d in decisions:
        if d.verdict != PASS:
            print(f"{d.backend}: HOLD, not deployed" + "".join(f"\n    {c}: {t}" for c, t in d.reasons))
            continue
        if now is not None:
            print(f"{d.backend}: PASS as of {a.now}; a replayed decision deploys nothing")
            continue
        rec = _latest(records, d.backend)
        ap_rec = approval(policy, a, d.backend, rec, paths[id(rec)], artifacts[d.backend], audit)
        ap_path = out / f"{d.backend}_{artifacts[d.backend][:12]}_{ap_rec['approved_at'].replace(':', '')}.json"
        ap_path.write_text(json.dumps(ap_rec, indent=2) + "\n")
        print(f"{d.backend}: PASS, approval {ap_path.relative_to(ROOT)}")
        print(f"    {HOSTS[d.backend]}: {start(d.backend, a.files[d.backend], ap_path, a.notify, a.port)}")


if __name__ == "__main__":
    main()
