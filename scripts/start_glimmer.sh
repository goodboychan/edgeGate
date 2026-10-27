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
#         bash start_glimmer.sh toolcheck   # one tool call round trip (stub tool)

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Muse-Glimmer-30B}"
# The audit agent runs on the GX10 itself, so the server only listens locally.
# Set HOST=0.0.0.0 together with API_KEY to reach it from other machines.
HOST="${HOST:-127.0.0.1}"
API_KEY="${API_KEY:-}"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-65536}"
# vLLM reserves this fraction of the 121 GiB up front whatever the weights need.
# 0.45 = ~54 GiB: FP8 weights 32 GiB + ~20 GiB KV (>1M tokens), leaving room for SAM 3.1.
GPU_UTIL="${GPU_UTIL:-0.45}"
VLLM_BIN="${VLLM_BIN:-$HOME/edgegate/.venv-vllm/bin/vllm}"
# Online quantization of the BF16 weights. Decode on the GX10 is memory-bandwidth
# bound: in the 10/27 server logs BF16 ran ~4.3 tok/s with 55.8 GiB of weights and
# FP8 ~7.9 tok/s with 32.4 GiB. Not a controlled comparison, and FP8 tool-call
# quality is only checked by toolcheck so far. QUANT= (empty) serves plain BF16;
# pair it with GPU_UTIL=0.65.
QUANT="${QUANT-fp8}"
URL="http://localhost:$PORT/v1/chat/completions"

if [ "${1:-}" = "test" ]; then
  # Model card recommends temperature 1.0, top_p 0.95, top_k 64, and
  # "Reasoning strength: <low|medium|high|xhigh>" in the system prompt.
  curl -s "$URL" \
    -H "Content-Type: application/json" ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
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
  # Wiring test only: measure_backend is a stub that returns fixed numbers, so a
  # pass says nothing about real measurements or audit quality. It checks that
  # Glimmer picks the right tool with the right argument and reports the result.
  API_KEY="$API_KEY" python3 - "$URL" <<'PY'
import json, os, sys, urllib.request

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
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    return json.load(urllib.request.urlopen(req))["choices"][0]["message"]

msg = chat()
calls = msg.get("tool_calls") or []
if len(calls) != 1:
    sys.exit(f"FAIL: expected 1 tool call, got {len(calls)}. content={msg.get('content')!r}")
call = calls[0]
name, args = call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")
print("tool call:", name, args)
if name != "measure_backend" or args != {"backend": "hailo8"}:
    sys.exit("FAIL: wrong tool or arguments")
messages.append(msg)
messages.append({"role": "tool", "tool_call_id": call["id"],  # stub values, not a measurement
                 "content": json.dumps({"backend": "hailo8", "box_l1_px": 3.7, "latency_ms": 11.2})})
answer = chat().get("content") or ""
print("answer:", answer)
if "3.7" not in answer:
    sys.exit("FAIL: answer does not report the tool result")
print("PASS")
PY
  exit 0
fi

# FlashInfer JIT-builds kernels at startup and looks up ninja (and nvcc) on PATH.
export PATH="$(dirname "$VLLM_BIN"):/usr/local/cuda/bin:$PATH"

if [ "$HOST" != "127.0.0.1" ] && [ -z "$API_KEY" ]; then
  echo "Refusing to listen on $HOST without API_KEY" >&2
  exit 1
fi

echo "Serving $MODEL_DIR on $HOST:$PORT (max_len=$MAX_LEN, gpu_util=$GPU_UTIL, quant=${QUANT:-none})"
exec "$VLLM_BIN" serve "$MODEL_DIR" \
  ${QUANT:+--quantization "$QUANT"} \
  ${API_KEY:+--api-key "$API_KEY"} \
  --served-model-name muse-glimmer \
  --host "$HOST" --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer
