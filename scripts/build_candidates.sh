#!/usr/bin/env bash
# Build every deployment candidate of one trained student, clean and fault-injected.
#
#   scripts/build_candidates.sh RUN STUDENT_PT CALIB_NPY
#
# Writes build/RUN/: student.onnx (every candidate is built from it; its sha256 is the
# model the gate approves), the fault ONNX files, three TensorRT FP16 engines built on
# the Jetson and five HEFs compiled on the x86 Hailo DFC host. The two device builds
# run in parallel (about 25 and 35 minutes for YOLOv8n); logs in build/RUN/*.log.
#
# Faults, each a candidate the gate judges only by its measured metrics:
#   bgr      red and blue swapped at the input (both devices)
#   no_norm  the /255 left out (TensorRT: in the ONNX; Hailo: in the compile)
#   w4       4-bit weights (Hailo)
#   calib10  10 calibration images instead of all of them (Hailo)
#
# Hosts, from the environment: JETSON (default jetson); DFC_HOST (default kepler),
# DFC_JUMP (a host to hop through, if the DFC host is not reachable directly) and
# DFC_PY (the DFC's python, default python3). Files go to and from the DFC host as tar
# streams, so nothing is left on a hop.
set -euo pipefail

if [ $# -ne 3 ]; then sed -n '4p' "$0"; exit 2; fi
RUN=$1 PT=$2 CALIB=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/build/$RUN
PY=$ROOT/.venv-train/bin/python
JETSON=${JETSON:-jetson}
DFC_HOST=${DFC_HOST:-kepler}
DFC_PY=${DFC_PY:-python3}
REMOTE=edgegate_hackathon/build/$RUN
SSH=(ssh -o BatchMode=yes)
HEFS=(student student_bgr student_no_norm student_w4 student_calib10)

if [ -e "$OUT" ]; then echo "$OUT exists; pick a new run name" >&2; exit 1; fi
mkdir -p "$OUT"
log() { echo "$(date +%T) $*"; }

"$PY" "$ROOT/scripts/export_onnx.py" "$PT" "$OUT/student.onnx"
"$PY" "$ROOT/scripts/inject_fault.py" "$OUT/student.onnx" bgr "$OUT/student_bgr.onnx"
"$PY" "$ROOT/scripts/inject_fault.py" "$OUT/student.onnx" no_norm "$OUT/student_no_norm.onnx"
cp "$CALIB" "$OUT/calib.npy"

jetson_build() {
  "${SSH[@]}" "$JETSON" "mkdir -p $REMOTE"
  rsync -a -e "${SSH[*]}" "$OUT/student.onnx" "$OUT/student_bgr.onnx" "$OUT/student_no_norm.onnx" "$JETSON:$REMOTE/"
  for m in student student_bgr student_no_norm; do
    log "trtexec $m"
    "${SSH[@]}" "$JETSON" "cd $REMOTE && /usr/src/tensorrt/bin/trtexec --onnx=$m.onnx \
      --saveEngine=${m}_fp16.engine --fp16 > trt_$m.log 2>&1"
  done
  rsync -a -e "${SSH[*]}" "$JETSON:$REMOTE/*_fp16.engine" "$OUT/"
  log "engines done"
}

dfc() {  # run one shell command on the DFC host
  if [ -n "${DFC_JUMP:-}" ]; then
    "${SSH[@]}" "$DFC_JUMP" "ssh -o BatchMode=yes $DFC_HOST $(printf %q "$1")"
  else
    "${SSH[@]}" "$DFC_HOST" "$1"
  fi
}

hailo_build() {
  tar -C "$OUT" -c student.onnx student_bgr.onnx calib.npy -C "$ROOT/scripts" compile_hailo.py \
    | dfc "mkdir -p $REMOTE && tar -C $REMOTE -x"
  local spec name onnx flags
  for spec in "student student.onnx" "student_bgr student_bgr.onnx" \
      "student_no_norm student.onnx --no-normalization" "student_w4 student.onnx --precision a8_w4" \
      "student_calib10 student.onnx --calib-limit 10"; do
    read -r name onnx flags <<< "$spec"
    log "compile_hailo $name"
    dfc "cd $REMOTE && $DFC_PY compile_hailo.py $onnx calib.npy $name.hef $flags > hef_$name.log 2>&1"
  done
  dfc "tar -C $REMOTE -c ${HEFS[*]/%/.hef}" | tar -C "$OUT" -x
  log "HEFs done"
}

jetson_build > "$OUT/jetson.log" 2>&1 & jetson=$!
hailo_build > "$OUT/hailo.log" 2>&1 & hailo=$!
failed=""
wait $jetson || failed="$failed jetson (build/$RUN/jetson.log)"
wait $hailo || failed="$failed hailo (build/$RUN/hailo.log)"
if [ -n "$failed" ]; then echo "failed:$failed" >&2; exit 1; fi
cd "$OUT" && sha256sum student*.onnx student*.engine student*.hef
