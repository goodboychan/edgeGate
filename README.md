# EdgeGate

Teach a new perception skill to the mixed edge devices in a home with one sentence (spoken or typed), and let an audit agent decide, chip by chip, whether it is safe to deploy.
Meta Global AI Developer Hackathon, Indie track, "Push the Frontier".

**Flow:** "Tell me when a package is left at the entrance" → Muse Voice Transcribe → Muse Spark plans → SAM 3.1 auto-labels → student detector is distilled → compiled per backend → Muse Glimmer audits → deployed only to approved backends → Home Assistant notifies, and a Creoqode Nova head turns to look at the target.

## Built during the hackathon

All code in this repository was written during the hackathon (2026-10-26 to 11-04); the commit history shows when. No pre-existing codebase is included.

- **Pre-existing:** home-lab hardware (GX10, edge boards, sensors, Nova), the open-source tools below, public model weights
- **Built this week:** all code in this repository. Status as of 10/27:

| Part | Status |
|---|---|
| GX10 setup, Glimmer serving (FP8, tool-call parser) | Implemented |
| Meta Model API check (Spark text/image, Voice Transcribe), Kokoro TTS | Implemented |
| Glimmer tool calling | Wiring test only: `toolcheck` uses a stub tool with fixed numbers |
| SAM 3.1 text-prompt labeling | Feasibility probe on sample images; no labeling script yet |
| Student detector, per-backend compile and measurement | Planned |
| Deployment gate (rules + audit agent), deployment, Nova, Home Assistant | Planned |

The external tools below are used as-is, without modification.

| External tool | Used for |
|---|---|
| vLLM | Serving Muse Glimmer |
| facebookresearch/sam3 | SAM 3.1 inference |
| Qualcomm QNN SDK | VENTUNO Q Hexagon compile |
| HailoRT / Dataflow Compiler | Hailo-8 compile |
| edgetpu_compiler | Coral Edge TPU compile |
| TensorRT | Jetson compile |
| DepthAI | OAK-D Lite (Myriad X) |
| Kokoro-82M | TTS for testing spoken instructions without a mic |

## Meta models used

Muse Glimmer 30B (local audit agent), Muse Spark / Muse Spark Open (planning and verification), Muse Voice Transcribe (spoken instructions), SAM 3.1 (teacher labels), Muse Code (development).

## GX10 setup

```bash
bash scripts/setup_gx10.sh               # model downloads + SAM 3.1 env + TTS env + checks
bash scripts/start_glimmer.sh            # Glimmer vLLM server (:8001, vLLM >= 0.32, FP8 by default)
bash scripts/start_glimmer.sh test       # plain chat check
bash scripts/start_glimmer.sh toolcheck  # one tool-call round trip
python3 scripts/check_meta_api.py        # Meta Model API check (META_API_KEY in .env)
.venv-tts/bin/python scripts/tts.py "Tell me when a package is left at the entrance." data/instruction.wav
python3 scripts/check_meta_api.py data/instruction.wav   # speech → Voice Transcribe
```
