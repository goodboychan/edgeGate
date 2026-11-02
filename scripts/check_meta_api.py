#!/usr/bin/env python3
"""Check that the Meta Model API is reachable from here with our key.

Runs four cheap calls: list models, a Muse Spark text chat, a Muse Spark image
question, and a Muse Voice Transcribe file upload. Reads META_API_KEY (or
MODEL_API_KEY) from the environment or from ../.env. Stdlib only.

Usage:  python3 scripts/check_meta_api.py [speech.wav] [image.jpg]
"""
import base64
import io
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

BASE = "https://api.meta.ai/v1"
SPARK = os.environ.get("SPARK_MODEL", "muse-spark-1.3")
TRANSCRIBE = "muse-voice-transcribe-1.0"
ROOT = Path(__file__).resolve().parent.parent


def load_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("META_API_KEY not set (env or .env)")
    return key


def call(method, path, key, body=None, content_type="application/json", base=BASE):
    req = urllib.request.Request(base + path, data=body, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    if body is not None:
        req.add_header("Content-Type", content_type)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400], time.time() - t


def chat(key, content):
    body = {"model": SPARK, "messages": [{"role": "user", "content": content}], "max_tokens": 2048}  # Spark reasons first; 200 ran out before any content
    return call("POST", "/chat/completions", key, json.dumps(body).encode())


def tone_wav():
    # 1 s, 440 Hz, mono 16-bit 16 kHz: proves the endpoint works, transcript will be empty.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000)))
                               for i in range(16000)))
    return buf.getvalue()


def transcribe(key, audio):
    boundary = uuid.uuid4().hex
    settings = json.dumps({"mode": "PUSH_TO_TALK", "model": TRANSCRIBE, "audioEncoding": "WAV"})
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
            f"Content-Type: application/json\r\n\r\n{settings}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    return call("POST", "/asr/transcribe", key, body, f"multipart/form-data; boundary={boundary}")


def report(name, result, show):
    status, data, dt = result
    ok = status == 200
    detail = show(data) if ok else data
    print(f"[{'OK ' if ok else 'ERR'}] {name}: HTTP {status}, {dt:.1f}s -> {detail}")
    return ok


def main():
    key = load_key()
    wav = Path(sys.argv[1]).read_bytes() if len(sys.argv) > 1 else tone_wav()
    img = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "third_party/sam3/assets/images/truck.jpg"

    results = [
        report("list models", call("GET", "/models", key),
               lambda d: sorted(m["id"] for m in d.get("data", []))),
        report(f"{SPARK} text", chat(key, "Reply with exactly: EdgeGate online"),
               lambda d: d["choices"][0]["message"]["content"]),
    ]
    if img.exists():
        url = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
        results.append(report(f"{SPARK} image", chat(key, [
            {"type": "text", "text": "In one short sentence, what is the main object?"},
            {"type": "image_url", "image_url": {"url": url}}]),
            lambda d: d["choices"][0]["message"]["content"]))
    results.append(report(TRANSCRIBE, transcribe(key, wav),
                          lambda d: {"transcript": d.get("transcript"), "ms": d.get("audioDurationMs")}))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
