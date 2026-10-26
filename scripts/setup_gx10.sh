#!/usr/bin/env bash
# EdgeGate - GX10 environment setup (Meta Global AI Developer Hackathon)
#
# What this does
#   1. Checks GPU, disk, Python tooling
#   2. Logs in to Hugging Face (once)
#   3. Downloads Muse Glimmer 30B (BF16, ~60 GB) and SAM 3.1 (~3.5 GB)
#   4. Creates a separate Python env for SAM 3.1 (PyTorch + sam3)
#   5. Runs quick smoke checks
#
# Muse Spark Open (816B MoE, ~700 GB even in NVFP4) does NOT fit on the GX10.
# Use it through Baseten / Modal instead.
#
# Before running:
#   - Accept the SAM license on https://huggingface.co/facebook/sam3.1
#     (gated repo; the download fails until you accept it)
#
# Usage:  bash setup_gx10.sh            # everything
#         bash setup_gx10.sh download   # only model downloads
#         bash setup_gx10.sh sam3env    # only the SAM 3.1 Python env
#         bash setup_gx10.sh check      # only smoke checks

set -euo pipefail

EDGEGATE_HOME="${EDGEGATE_HOME:-$HOME/edgegate}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
GLIMMER_REPO="meta-models/Muse-Glimmer-30B"
SAM_REPO="facebook/sam3.1"
SAM_ENV="$EDGEGATE_HOME/.venv-sam3"
TOOLS_ENV="$EDGEGATE_HOME/.venv-tools"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

log()  { printf '\n\033[1;36m[edgegate]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }

preflight() {
  log "Preflight checks"
  mkdir -p "$EDGEGATE_HOME" "$MODELS_DIR"
  uname -m
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || warn "nvidia-smi failed"
  df -h "$MODELS_DIR" | tail -1
  avail_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc '0-9')
  if [ "${avail_gb:-0}" -lt 90 ]; then
    warn "Only ${avail_gb} GB free under $MODELS_DIR. Need ~70 GB for models + headroom."
  fi
  if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

tools_env() {
  if [ ! -x "$TOOLS_ENV/bin/hf" ]; then
    log "Creating tools env (huggingface_hub CLI)"
    uv venv "$TOOLS_ENV" --python 3.12
    uv pip install --python "$TOOLS_ENV/bin/python" -U "huggingface_hub[cli,hf_xet]"
  fi
  export PATH="$TOOLS_ENV/bin:$PATH"
  if ! hf auth whoami >/dev/null 2>&1; then
    log "Hugging Face login (paste a read token from https://huggingface.co/settings/tokens)"
    hf auth login
  fi
  hf auth whoami
}

download_models() {
  tools_env
  log "Downloading $GLIMMER_REPO (BF16, ~60 GB). Resumable; rerun if interrupted."
  hf download "$GLIMMER_REPO" --local-dir "$MODELS_DIR/Muse-Glimmer-30B"

  log "Downloading $SAM_REPO (~3.5 GB, gated: accept the license on the HF page first)"
  if ! hf download "$SAM_REPO" --local-dir "$MODELS_DIR/sam3.1"; then
    warn "SAM 3.1 download failed. Open https://huggingface.co/facebook/sam3.1, accept the license, then rerun: bash setup_gx10.sh download"
  fi
  du -sh "$MODELS_DIR"/* || true
}

sam3_env() {
  log "Creating SAM 3.1 env at $SAM_ENV"
  [ -d "$SAM_ENV" ] || uv venv "$SAM_ENV" --python 3.12
  PY="$SAM_ENV/bin/python"

  log "Installing PyTorch (aarch64 CUDA wheels from $TORCH_INDEX)"
  uv pip install --python "$PY" torch torchvision --index-url "$TORCH_INDEX"

  log "Installing sam3 from source"
  if [ ! -d "$EDGEGATE_HOME/third_party/sam3" ]; then
    mkdir -p "$EDGEGATE_HOME/third_party"
    git clone https://github.com/facebookresearch/sam3.git "$EDGEGATE_HOME/third_party/sam3"
  fi
  uv pip install --python "$PY" -e "$EDGEGATE_HOME/third_party/sam3"
  uv pip install --python "$PY" einops pycocotools opencv-python-headless pillow numpy psutil
}

check() {
  log "Smoke check: SAM 3.1 env"
  if [ -x "$SAM_ENV/bin/python" ]; then
    "$SAM_ENV/bin/python" - <<'PY' || true
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", float((x @ x).sum()) != 0.0)
import sam3; print("sam3 import ok")
PY
  else
    warn "SAM env missing. Run: bash setup_gx10.sh sam3env"
  fi
  ls -lh "$MODELS_DIR/sam3.1" 2>/dev/null | grep -E '\.(pt|safetensors)$' || warn "SAM 3.1 checkpoint not found"

  log "Smoke check: Glimmer weights"
  ls "$MODELS_DIR/Muse-Glimmer-30B"/*.safetensors >/dev/null 2>&1 \
    && du -sh "$MODELS_DIR/Muse-Glimmer-30B" \
    || warn "Glimmer weights not found"

  cat <<EOF

Next: start Glimmer with  bash start_glimmer.sh
If 'cuda available: False' above, the PyTorch wheel does not support this GPU yet.
Fallback: run SAM 3.1 inside the NGC PyTorch container instead:
  docker run --gpus all -it --rm -v $HOME:$HOME nvcr.io/nvidia/pytorch:<latest>-py3
EOF
}

case "${1:-all}" in
  all)      preflight; download_models; sam3_env; check ;;
  download) preflight; download_models ;;
  sam3env)  preflight; sam3_env; check ;;
  check)    check ;;
  *) echo "usage: $0 [all|download|sam3env|check]"; exit 1 ;;
esac
