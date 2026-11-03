#!/usr/bin/env bash
# Copy what the README's numbers rest on into results/, which is in git: every
# measurement record, every audit, the approval records (none while everything is
# held), the verdict table per run, the gate comparison and the dashboard page.
# Recordings, models, detections and training outputs stay out of git.
# Usage:  scripts/snapshot_results.sh      (run again after measuring; it replaces results/)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-train/bin/python}
mkdir -p results/records results/audits results/approvals
rsync -a --delete --prune-empty-dirs --include='*/' --include='*.json' --exclude='*' runs/records/ results/records/
rsync -a --delete runs/audits/ results/audits/
[ -d runs/approvals ] && rsync -a --delete runs/approvals/ results/approvals/
[ -d runs/teach ] && rsync -a --delete runs/teach/ results/teach/
[ -d runs/audits_synthetic ] && rsync -a --delete runs/audits_synthetic/ results/audits_synthetic/
for r in runs/records/*/; do
    run=$(basename "$r")
    [ -f "build/$run/student.onnx" ] || continue
    "$PY" scripts/measure_candidates.py "$run" --table-only > "results/table_$run.md"
done
"$PY" scripts/compare_gates.py > results/compare_gates.md
"$PY" scripts/dashboard.py --out results/dashboard.html > /dev/null
# Addresses used to reach the devices (user@host or a private IPv4) are not part of
# the evidence; older records carry the Pi's address as its label.
# (four octets, so a version such as "tensorrt 10.3.0" is left alone)
{ grep -rlE '([a-z0-9_]+@)?\b(192\.168|10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+\b' results || true; } \
    | xargs -r sed -i -E 's/([a-z0-9_]+@)?\b(192\.168|10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+\b/device/g'
# (grep exits 1 when nothing is left to replace; that is the normal case, not a failure)
# Paths inside records and audits are made relative to the repository.
{ grep -rl '/home/[a-z0-9_]*/edgegate/' results || true; } | xargs -r sed -i -E 's#/home/[a-z0-9_]+/edgegate/##g'
# deploy.py's refusal on s3, captured 11/3, is evidence too (logs/ is not in git)
[ -f logs/deploy_s3_hold.txt ] && cp logs/deploy_s3_hold.txt results/deploy_s3_hold.txt
date -u +%Y-%m-%dT%H:%M:%SZ > results/snapshot_at.txt
echo "results/ written at $(cat results/snapshot_at.txt): $(find results -type f | wc -l) files, $(du -sh results | cut -f1)"
