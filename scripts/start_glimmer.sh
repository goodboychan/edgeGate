#!/usr/bin/env bash
# Serve Muse Glimmer 30B on the GX10 with vLLM (OpenAI-compatible API).
#
# Needs vLLM >= 0.32 (MuseGlimmerForConditionalGeneration and the muse_glimmer
# tool/reasoning parsers). The shared ~/llm-serving venv is 0.23 and cannot load
# it, so this uses a separate venv at ~/edgegate/.venv-vllm by default.
#
# Memory: BF16 weights are ~60 GB. Stop other vLLM servers (Qwen on 8002,
# Gemma on 8000) first unless you have checked that both fit in the 128 GB
# unified memory.
#
# Port 8001 avoids clashing with the existing vLLM (8000/8002), the AREL RAG
# proxy (8010) and its dashboard (3001).
#
# Usage:  bash start_glimmer.sh             # serve
#         bash start_glimmer.sh test        # plain chat check
#         bash start_glimmer.sh toolcheck   # one tool call round trip

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Muse-Glimmer-30B}"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-65536}"
GPU_UTIL="${GPU_UTIL:-0.65}"
VLLM_BIN="${VLLM_BIN:-$HOME/edgegate/.venv-vllm/bin/vllm}"
URL="http://localhost:$PORT/v1/chat/completions"

if [ "${1:-}" = "test" ]; then
  # Model card recommends temperature 1.0, top_p 0.95, top_k 64, and
  # "Reasoning strength: <low|medium|high|xhigh>" in the system prompt.
  curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "muse-glimmer",
      "messages": [
        {"role": "system", "content": "Reasoning strength: low"},
        {"role": "user", "content": "Reply with exactly: EdgeGate online"}
      ],
      "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 256
    }' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'
  exit 0
fi

if [ "${1:-}" = "toolcheck" ]; then
  # Day-1 done criterion: Glimmer calls a dummy tool, gets the result, answers.
  python3 - "$URL" <<'PY'
import json, sys, urllib.request

url = sys.argv[1]
tools = [{"type": "function", "function": {
    "name": "measure_backend",
    "description": "Run the student detector on one edge backend and return its open-loop error.",
    "parameters": {"type": "object", "properties": {
        "backend": {"type": "string", "enum": ["qnn", "edgetpu", "hailo8", "trt"]}},
        "required": ["backend"]}}}]
messages = [
    {"role": "system", "content": "Reasoning strength: low"},
    {"role": "user", "content": "Measure the hailo8 backend and tell me its box error."},
]

def chat():
    body = {"model": "muse-glimmer", "messages": messages, "tools": tools,
            "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 1024}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["choices"][0]["message"]

msg = chat()
calls = msg.get("tool_calls") or []
if not calls:
    sys.exit(f"FAIL: no tool call. content={msg.get('content')!r}")
call = calls[0]
print("tool call:", call["function"]["name"], call["function"]["arguments"])
messages.append(msg)
messages.append({"role": "tool", "tool_call_id": call["id"],
                 "content": json.dumps({"backend": "hailo8", "box_l1_px": 3.7, "latency_ms": 11.2})})
print("answer:", chat().get("content"))
PY
  exit 0
fi

# FlashInfer JIT-builds kernels at startup and looks up ninja (and nvcc) on PATH.
export PATH="$(dirname "$VLLM_BIN"):/usr/local/cuda/bin:$PATH"

echo "Serving $MODEL_DIR on :$PORT (max_len=$MAX_LEN, gpu_util=$GPU_UTIL)"
exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name muse-glimmer \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer
