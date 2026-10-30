#!/usr/bin/env bash
# Label the train sessions with SAM 3.1 (those not labeled yet) and train one student.
#
#   scripts/make_student.sh RUN data/sessions/t01 data/sessions/t02 ...
#
# Writes labels_sam.jsonl next to each session's frames (scripts/label_sam.py, about
# 1.4 s per kept frame at stride 5, so ~40 min per 10 min of recording), then
# runs/train/RUN/ (scripts/train_student.py, which refuses non-train sessions).
# Sessions already labeled are not labeled again, so this can run while later
# sessions are still being recorded and be run again with all of them.
# Then: scripts/build_candidates.sh RUN runs/train/RUN/student.pt build/calib.npy
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 2 ] || { sed -n '2,11p' "$0"; exit 2; }
RUN=$1
shift
mkdir -p logs

for s in "$@"; do
    split=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["split"])' "$s/meta.json")
    [ "$split" = train ] || { echo "$s is a $split session; only train sessions are labeled for training" >&2; exit 1; }
done

for s in "$@"; do
    if [ -s "$s/labels_sam.jsonl" ] && [ -s "$s/labels_sam.meta.json" ]; then
        echo "$(date +%T) $s: already labeled"
        continue
    fi
    echo "$(date +%T) $s: labeling (log logs/label_$(basename "$s").log)"
    .venv-sam3/bin/python scripts/label_sam.py "$s" > "logs/label_$(basename "$s").log" 2>&1
done

echo "$(date +%T) training runs/train/$RUN (log logs/train_$RUN.log)"
.venv-train/bin/python scripts/train_student.py "$@" --out "runs/train/$RUN" > "logs/train_$RUN.log" 2>&1
echo "$(date +%T) done: runs/train/$RUN/student.pt"
python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); print("student sha256", p["student_sha256"]); [print(" ", s["session"], s["train"], "train /", s["val"], "val frames", s["boxes"]) for s in p["sessions"]]' "runs/train/$RUN/provenance.json"
