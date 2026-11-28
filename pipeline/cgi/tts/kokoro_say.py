"""Speak one line with Kokoro-82M (Apache 2.0) and write a 24 kHz mono 16-bit WAV.

    python kokoro_say.py --text "..." --voice bf_emma --lang b --speed 0.9 --out line.wav

Needs: pip install kokoro "misaki[ja]" soundfile   (the model downloads from Hugging Face once)
Runs in its own process so the pipeline itself needs no PyTorch.
"""
import argparse
import struct
import sys
import wave


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--lang", default="a")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from kokoro import KPipeline  # noqa: PLC0415 - heavy: only here
    pipe = KPipeline(lang_code=a.lang, repo_id="hexgrad/Kokoro-82M")
    samples = []
    for _graphemes, _phonemes, audio in pipe(a.text, voice=a.voice, speed=a.speed):
        if audio is None:
            continue
        samples.extend(float(x) for x in (audio.tolist() if hasattr(audio, "tolist") else audio))
        samples.extend([0.0] * int(24000 * 0.12))           # a breath between sentences
    if not samples:
        print("kokoro produced no audio", file=sys.stderr)
        return 3
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack(f"<{len(samples)}h", *(max(-32767, min(32767, int(s * 32767))) for s in samples)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
