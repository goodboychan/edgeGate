#!/usr/bin/env python3
"""Check Muse Spark Open on Baseten (hackathon partner credits).

Runs a text chat and an image question against the hackathon deployment.
Reads BASETEN_API_KEY from the environment or ../.env. Stdlib only.

The hackathon key only works with meta-models/muse-spark-open-1026 (Baseten's
docs say meta-models/muse-spark-open; that name does not work for this key).

Usage:  python3 scripts/check_baseten.py [image.jpg]
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://inference.baseten.co/v1"
MODEL = "meta-models/muse-spark-open-1026"
ROOT = Path(__file__).resolve().parent.parent


def load_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        sys.exit("BASETEN_API_KEY not set (env or .env)")
    return key


def chat(key, content):
    # Muse Spark Open always reasons before answering, so leave room for it.
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 2048}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400], time.time() - t


def report(name, result):
    status, data, dt = result
    if status != 200:
        print(f"[ERR] {name}: HTTP {status}, {dt:.1f}s -> {data}")
        return False
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    print(f"[OK ] {name}: HTTP 200, {dt:.1f}s, finish={choice.get('finish_reason')}, "
          f"tokens={usage.get('completion_tokens')} -> {choice['message'].get('content')!r}")
    return bool(choice["message"].get("content"))


def main():
    key = load_key()
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "third_party/sam3/assets/images/truck.jpg"
    results = [report(f"{MODEL} text", chat(key, "Reply with exactly: EdgeGate online"))]
    if img.exists():
        url = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
        results.append(report(f"{MODEL} image", chat(key, [
            {"type": "text", "text": "In one short sentence, what is the main object?"},
            {"type": "image_url", "image_url": {"url": url}}])))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
