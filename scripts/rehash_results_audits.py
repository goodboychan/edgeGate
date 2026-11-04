#!/usr/bin/env python3
"""After results/ is scrubbed (addresses and home paths replaced), the record files there no
longer hash like the originals under runs/, so an audit's record_sha256 map would not name
the copies beside it. This rewrites each audit's record_sha256 values in results/audits/ to
the hashes of the scrubbed copies, for every record path that exists under results/, so the
binding check (edgegate.gate.audit_binding_problem) holds in results/ as it does in runs/.
Run by scripts/snapshot_results.sh after the scrub."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    changed = 0
    for ap in sorted((ROOT / "results/audits").glob("*.json")):
        a = json.loads(ap.read_text())
        m = a.get("record_sha256") or {}
        new = {}
        for path, h in m.items():
            rel = path.replace("runs/records/", "results/records/", 1)
            copy = ROOT / rel
            new[path] = sha(copy) if copy.exists() else h
        if new != m:
            a["record_sha256"] = new
            ap.write_text(json.dumps(a, indent=2) + "\n")
            changed += 1
    print(f"results/audits: {changed} audit(s) rehashed to the scrubbed record copies")


if __name__ == "__main__":
    sys.exit(main())
