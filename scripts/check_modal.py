"""Check that Modal (hackathon credits) can run a GPU job.

Runs nvidia-smi and a small bf16 matmul on one GPU (MODAL_GPU, default H100)
and prints the GPU, the matmul rate and the wall time. Needs `modal setup` first.
The usage page on modal.com shows what it cost.

Usage:  [MODAL_GPU=L40S] .venv-modal/bin/modal run scripts/check_modal.py
"""
import os
import time

import modal

app = modal.App("edgegate-check")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.9.1")


@app.function(image=image, gpu=os.environ.get("MODAL_GPU", "H100").split(","), timeout=600)
def probe():
    import subprocess

    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 8192**3 / (time.time() - t) / 1e12
    return smi, round(tflops, 1)


@app.local_entrypoint()
def main():
    t = time.time()
    smi, tflops = probe.remote()
    print(f"GPU: {smi} | bf16 matmul {tflops} TFLOPS | wall {time.time() - t:.0f} s (includes cold start)")
