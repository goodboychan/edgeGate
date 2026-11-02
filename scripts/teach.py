#!/usr/bin/env python3
"""One sentence in, a task spec out: the front end of the pipeline, checked against the policy.

The sentence comes typed or spoken (a WAV, transcribed by Muse Voice Transcribe). Muse
Spark turns it into a task spec: a name, the object classes to detect (text prompts for
SAM 3.1, the target first), what makes an alert and what must not, and the conditions
to watch. The spec is written to runs/teach/ with the transcript, the raw answer, the
models and the timings, and compared with the policy: the instruction and the class
list must match. Nothing here writes the policy. The policy, its zone, rule and
thresholds were written by hand and frozen before any evaluation (see the README);
this script shows that the same spec comes out of the sentence, and it was added on
11/3 after the candidates had been measured.

Usage:  python3 scripts/teach.py "Tell me when a package is left at the entrance."
        python3 scripts/teach.py --wav data/instruction.wav
        python3 scripts/teach.py --wav data/instruction.wav --base https://inference.baseten.co/v1 \
            --model meta-models/muse-spark-open-1026 --key-env BASETEN_API_KEY   # Muse Spark Open instead
Keys come from the environment or ../.env (META_API_KEY, BASETEN_API_KEY).
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from check_meta_api import BASE, SPARK, TRANSCRIBE, call, load_key, transcribe  # noqa: E402
from edgegate.gate import load_policy  # noqa: E402

SYSTEM = """You turn one sentence from a person at home into a task spec for a camera skill on a small edge device.
Answer with one JSON object and nothing else, with exactly these keys:
  "task": a short kebab-case name
  "instruction": the sentence as given
  "detect": the object classes to detect, as short singular nouns usable as text prompts for a segmentation model; the target object first, then any class needed to tell the target's state apart (for example the person handling it)
  "event": one sentence saying what makes one alert
  "not_event": one sentence saying what must not alert
  "watch": up to three conditions in this home under which detection could fail, each a few words
Use the camera context for "watch". Do not add keys."""


def spec_from(text):
    """The first JSON object in a model answer, or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def compare(spec, policy):
    """Where the spec and the policy disagree, as human-readable lines (empty = they match)."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", str(s).lower()).split()  # noqa: E731
    out = []
    if norm(spec.get("instruction")) != norm(policy["task"]["instruction"]):
        out.append(f"instruction: spec {spec.get('instruction')!r}, policy {policy['task']['instruction']!r}")
    classes = [str(c).lower() for c in (spec.get("detect") or [])]
    if classes != policy["measure"]["classes"]:
        out.append(f"classes: spec {classes}, policy {policy['measure']['classes']}")
    for key in ("task", "event", "not_event", "watch"):
        if not spec.get(key):
            out.append(f"{key}: missing from the spec")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("instruction", nargs="?", help="the sentence, typed")
    ap.add_argument("--wav", help="the sentence, spoken (mono 16-bit WAV); transcribed first")
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--base", default=BASE, help="OpenAI-compatible API base")
    ap.add_argument("--model", default=SPARK)
    ap.add_argument("--key-env", default="META_API_KEY", help="environment variable (or .env entry) with the key")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="both Spark models reason before answering (Spark Open used 1.6k tokens of it), so leave room")
    ap.add_argument("--out", default=str(ROOT / "runs/teach"))
    a = ap.parse_args()
    if not a.instruction and not a.wav:
        ap.error("give the sentence or --wav")
    load_key()  # fills the environment from .env
    key = os.environ.get(a.key_env) or sys.exit(f"{a.key_env} not set (env or .env)")
    policy = load_policy(a.policy)
    rec = {"schema": 1, "taught_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base": a.base,
           "model": a.model, "policy": a.policy, "wav": a.wav, "typed": a.instruction}

    text = a.instruction
    if a.wav:
        meta_key = os.environ.get("META_API_KEY") or sys.exit("META_API_KEY not set: Voice Transcribe needs it")
        status, data, dt = transcribe(meta_key, Path(a.wav).read_bytes())
        if status != 200:
            sys.exit(f"transcribe: HTTP {status} {data}")
        text = (data.get("transcript") or "").strip()
        rec["transcribe"] = {"model": TRANSCRIBE, "transcript": text, "audio_ms": data.get("audioDurationMs"),
                             "s": round(dt, 2)}
        print(f"{TRANSCRIBE}: {text!r} ({data.get('audioDurationMs')} ms of audio, {dt:.1f} s)")
        if not text:
            sys.exit("empty transcript")
    rec["instruction"] = text

    body = {"model": a.model, "max_tokens": a.max_tokens, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Instruction: {text}\nCamera context: {policy['task'].get('context', '')}"}]}
    t = time.time()
    status, data, dt = call("POST", "/chat/completions", key, json.dumps(body).encode(), base=a.base)
    if status != 200:
        sys.exit(f"{a.model}: HTTP {status} {data}")
    msg = data["choices"][0]["message"]
    rec["spark"] = {"answer": msg.get("content"), "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:4000],
                    "usage": data.get("usage"), "s": round(time.time() - t, 2)}
    spec = spec_from(msg.get("content"))
    if spec is None:
        rec["spec"] = None
        print(f"{a.model} gave no JSON object:\n{msg.get('content')}")
    else:
        rec["spec"] = spec
        rec["differs"] = compare(spec, policy)
        print(f"{a.model} ({dt:.1f} s):\n{json.dumps(spec, indent=2)}")
        print("matches the policy's instruction and classes" if not rec["differs"] else
              "differs from the policy:\n  " + "\n  ".join(rec["differs"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rec['taught_at'].replace(':', '')}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"written to {path.relative_to(ROOT)}; the policy was not changed")
    sys.exit(0 if spec is not None and not rec["differs"] else 1)


if __name__ == "__main__":
    main()
