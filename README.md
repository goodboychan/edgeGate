# EdgeGate

Teach a new perception skill to the mixed edge devices in a home with one sentence (spoken or typed), and let an audit agent decide, chip by chip, whether it is safe to deploy.
Meta Global AI Developer Hackathon, Indie track, "Push the Frontier".

**Flow:** "Tell me when a package is left at the entrance" → Muse Voice Transcribe → Muse Spark plans → SAM 3.1 auto-labels → student detector is distilled → compiled per backend → Muse Glimmer audits → deployed only to approved backends → Home Assistant notifies, and a Creoqode Nova head turns to look at the target.

## Built during the hackathon

All code in this repository was written during the hackathon (2026-10-26 to 11-04); the commit history shows when. No pre-existing codebase is included.

- **Pre-existing:** home-lab hardware (GX10, edge boards, sensors, Nova), the open-source tools below, public model weights
- **Built this week:** all code in this repository. Status as of 10/29:

| Part | Status |
|---|---|
| GX10 setup, Glimmer serving (FP8, tool-call parser) | Implemented |
| Meta Model API check (Spark text/image, Voice Transcribe), Kokoro TTS | Implemented |
| Glimmer tool calling | Wiring test only: `toolcheck` uses a stub tool with fixed numbers |
| OAK-D Pro session capture (timestamped frames, outage log) | Implemented; mounted at the drop zone, USB 2 with NV12 at 15 fps (no gaps in a 60 s soak), focus fixed at the zone, exposure metered on the zone. Reconnect path not yet tested by unplugging |
| SAM 3.1 text-prompt labeling (box, person) | Implemented; on a 1-minute recording of the scene, 0.7 s/frame. Boxes come from the largest piece of each mask (a stray mask patch made boxes span 70% of the frame) |
| Rule engine (`policies/*.toml` → PASS/HOLD per backend) and event evaluator | Implemented, unit-tested on synthetic data only; no real measurements yet |
| Toolchain path to both devices (public YOLOv8n, not yet trained on our data) | Compile and run on both; on 33 public sample frames TensorRT FP16 matches ONNX 116/116 boxes, Hailo-8 INT8 111/116 (mean IoU 0.977); end to end per frame (letterbox, inference, decoding, NMS) p50 23 ms Jetson (15W mode), 14 ms Hailo Pi |
| Measurement harness (device run → event metrics → record the gate reads) | Implemented; the device hashes every frame it reads and reports its identity and runtime. Dry run on a 1-minute recording with no marked events on all three backends (2-epoch student, a toolchain check, not accuracy): each device matched its policy target, and p50 per frame was 32.7 ms GX10 ONNX, 19.6 ms Jetson, 12.4 ms Hailo Pi. No real eval data yet |
| Glimmer audit tools (MCP: rule verdicts, eval coverage, per-slice recall; veto-only `propose_hold` with evidence ids) | Implemented; wiring check on 5 synthetic cases (made-up numbers, one run each, not evidence): Glimmer FP8 gave the expected verdict in all 5, no HOLD on the control, 3.5 to 6 min per audit. Its HOLD reasons name the planted slice but also cite one-event lighting gaps. Not yet run on measured candidates, no repeat runs yet |
| GT marking tool (browser) | Implemented; save and validation tested, page not yet used on a real session |
| Fault-injected builds (BGR swap, no normalization, 4-bit weights, 10 calibration images) | Built for both devices from the public YOLOv8n; on 33 public frames vs ONNX: no normalization 0 boxes matched, 4-bit weights 65/116, BGR 100/116 (Hailo) and 104/116 (TRT), 10 calibration images 112/116, about the same as the clean build |
| Student training (YOLOv8n on SAM labels, train sessions only) | Script dry-run only: 2 epochs on 1 minute of video, exported and compiled for both devices. No real training yet |
| Deployment, Nova, Home Assistant | Planned |

The external tools below are used as-is, without modification.

| External tool | Used for |
|---|---|
| vLLM | Serving Muse Glimmer |
| facebookresearch/sam3 | SAM 3.1 inference |
| Qualcomm QNN SDK | VENTUNO Q Hexagon compile |
| HailoRT / Dataflow Compiler | Hailo-8 compile |
| edgetpu_compiler | Coral Edge TPU compile |
| TensorRT | Jetson compile |
| Ultralytics YOLOv8 (AGPL-3.0) and its public COCO weights | Student detector architecture; the COCO weights only for toolchain tests |
| DepthAI | OAK-D Pro camera capture |
| Kokoro-82M | TTS for testing spoken instructions without a mic |

## Meta models used

Muse Glimmer 30B (local audit agent), Muse Spark / Muse Spark Open (planning and verification), Muse Voice Transcribe (spoken instructions), SAM 3.1 (teacher labels), Muse Code (development).

## GX10 setup

```bash
bash scripts/setup_gx10.sh               # model downloads + SAM 3.1, TTS and OAK envs + checks
bash scripts/start_glimmer.sh            # Glimmer vLLM server (:8001, vLLM >= 0.32, FP8 by default)
bash scripts/start_glimmer.sh test       # plain chat check
bash scripts/start_glimmer.sh toolcheck  # one tool-call round trip
python3 scripts/check_meta_api.py        # Meta Model API check (META_API_KEY in .env)
.venv-tts/bin/python scripts/tts.py "Tell me when a package is left at the entrance." data/instruction.wav
python3 scripts/check_meta_api.py data/instruction.wav   # speech → Voice Transcribe
python3 scripts/check_baseten.py         # Muse Spark Open on Baseten (BASETEN_API_KEY in .env)
.venv-modal/bin/modal run scripts/check_modal.py   # Modal GPU check (after `modal setup`)
.venv-oak/bin/python scripts/capture_oak.py s01 --split eval --light day --minutes 10   # record a session (OAK-D Pro)
.venv-sam3/bin/python scripts/label_sam.py data/sessions/s01   # SAM 3.1 teacher boxes for package and person
python3 scripts/mark_gt.py data/sessions/e01   # mark arrivals by hand in the browser (127.0.0.1:8765) -> data/gt/e01.json
.venv-train/bin/python scripts/train_student.py data/sessions/t01 data/sessions/t02 --out runs/train/s1   # student
.venv-train/bin/python scripts/make_calib.py data/sessions/c01 build/calib.npy   # Hailo calibration images
python3 scripts/measure.py --eval-set-hash --sessions e01,e02   # eval set hash for the policy; manifest in data/manifests/
```

## Gate

```bash
python3 -m unittest discover -s tests                     # rule engine + evaluator tests
.venv-train/bin/python -m unittest discover -s tests      # also the detector post-processing (numpy, OpenCV)
python3 -m edgegate.events policies/package_entrance.toml GT_DIR DETS_DIR   # detections → event metrics
python3 scripts/measure.py hailo build/student.hef --onnx build/student.onnx --host chans@192.168.50.151 \
    --sessions e01,e02          # device run -> runs/records/*.json (any failure is recorded as a tool failure)
python3 -m edgegate.gate policies/package_entrance.toml runs/records/*.json --model-sha SHA \
    --artifact trt=build/student.engine --artifact hailo8=build/student.hef
```

Audit (after the rules): Glimmer gets the MCP tools in `edgegate/audit_mcp.py` (rule verdicts, eval coverage, per-slice recall by lighting, package size, scene tag or session) and can only propose a HOLD for a backend the rules passed, citing the evidence ids of tool results from that audit.

```bash
bash scripts/setup_gx10.sh auditenv       # MCP SDK env for the audit
.venv-audit/bin/python scripts/audit.py policies/package_entrance.toml runs/records/*.json --model-sha SHA \
    --artifact trt=build/student.engine --artifact hailo8=build/student.hef --repeats 5 --label fp8
python3 tests/audit_cases.py /tmp/cases   # synthetic wiring cases (made-up numbers, not evidence)
```

Fault-injected candidates (judged only by their measured metrics): `scripts/inject_fault.py` (BGR swap or missing normalization, built into the ONNX given to the compiler) and the `compile_hailo.py` flags `--calib-limit`, `--precision a8_w4`, `--no-normalization`.

The rules decide; Glimmer can only veto a PASS and explain why. Missing, stale, NaN or failed measurements, a different model or eval set, a deployed file that is not the one measured, or a policy without an eval set all give HOLD. A record also counts only for:

- **its device:** the policy's `[targets]` names each backend's device (a hash of its machine id) and runtime version. A record from another device, a reinstalled OS or another runtime version holds.
- **its eval set:** the eval set hash is a manifest of every frame's sha256 plus the marks, `frames.csv` and `meta.json`. A changed image changes the hash, and the device's digest of the frames it read must match it.
- **its evaluation settings:** `eval_config_sha256` covers the policy's `[events]` (zone, dwell, windows) and `[measure]` (class order, detection threshold). Changing them voids older records; thresholds are applied at decision time and are not in it.

**What the metrics are:** a recorded-session evaluation. Alert times come from the recorded frame times, and device latency is a separate metric. Alert delivery is not in them: `notify_s` stays 0 and `notify_measured = false`, so the gate holds every backend until the live path is measured. That measurement will log release (marked) → frame → inference done → event → alert received by Home Assistant → alert shown on the phone, each as its own timestamp.

Arrival rule (first version, fixed before evaluation, not tuned): a package seen for 2 s in an empty zone is an arrival, and the alert must be received within 10 s of the package being let go, dwell included; later alerts are recorded as late, not as successes. One alert while it stays; a new arrival needs the zone empty, not just hidden by a person, for 2 s. Scope: a first package into an empty zone; a second package next to the first is not detected. Known limit: when the camera drops out the zone state is unknown, not empty, and packages already there on reconnect are not alerted, so a package that arrived during the outage is never alerted.
