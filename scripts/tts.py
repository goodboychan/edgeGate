#!/usr/bin/env python3
"""Synthesize a spoken instruction to a WAV that Muse Voice Transcribe accepts.

Kokoro-82M (open source, Apache 2.0) outputs 24 kHz; we write mono 16-bit PCM,
which the /asr/transcribe file endpoint takes as-is. Used to test the voice
path without a microphone.

Usage:  .venv-tts/bin/python scripts/tts.py "Tell me when a package lands at the front door." out.wav [voice]
"""
import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    text, out = sys.argv[1], sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else "af_heart"
    # CPU is fast enough for an 82M model and leaves the GPU memory to Glimmer and SAM.
    pipe = KPipeline(lang_code=voice[0], device="cpu")  # 'a' American / 'b' British English
    audio = np.concatenate([np.asarray(chunk.audio) for chunk in pipe(text, voice=voice)])
    sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"{out}: {len(audio) / SAMPLE_RATE:.1f}s, voice={voice}")


if __name__ == "__main__":
    main()
