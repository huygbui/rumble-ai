# Quick perf benchmark for the Fish Speech S2 Pro Modal endpoint.
#
# S2 Pro has no built-in voice (the old voice:"default" now 400s), so the bench
# registers one reusable voice up front and synthesizes by name. Set a reference:
#   export REF_AUDIO=./reference.wav   # LOCAL file (registration uploads its bytes)
#   export REF_TEXT="exact transcript of the reference clip"
import concurrent.futures as cf
import io
import mimetypes
import os
import statistics
import time
import wave

import requests

BASE = os.environ["TTS_URL"].rstrip("/")
URL = f"{BASE}/v1/audio/speech"
VOICES_URL = f"{BASE}/v1/audio/voices"
VOICE = "bench"  # registered below before any synthesis


def register_bench_voice() -> None:
    ref_audio = os.environ.get("REF_AUDIO")
    ref_text = os.environ.get("REF_TEXT")
    if not ref_audio or not os.path.isfile(ref_audio) or not ref_text:
        raise SystemExit(
            "bench needs a voice: set REF_AUDIO (a LOCAL wav) + REF_TEXT "
            "(S2 Pro has no built-in voice; default was removed in vllm-omni 0.22.x)"
        )
    mime = mimetypes.guess_type(ref_audio)[0] or "audio/wav"
    with open(ref_audio, "rb") as f:
        r = requests.post(
            VOICES_URL,
            data={"name": VOICE, "consent": "I consent.", "ref_text": ref_text},
            files={"audio_sample": (os.path.basename(ref_audio), f, mime)},
            timeout=120,
        )
    r.raise_for_status()

SHORT = "Hello, this is a short test sentence."
MEDIUM = (
    "The quick brown fox jumps over the lazy dog while the morning sun rises slowly "
    "over the quiet hills, casting long shadows across the dew covered meadow below."
)
LONG = (MEDIUM + " ") * 3


def wav_dur(b: bytes) -> float:
    try:
        w = wave.open(io.BytesIO(b))
        return w.getnframes() / w.getframerate()
    except Exception:
        return max(0.0, (len(b) - 44) / (44100 * 2))  # assume 44.1k mono s16


def synth(text, stream=False, timeout=600):
    payload = {"input": text, "voice": VOICE, "response_format": "wav", "seed": 58842}
    t0 = time.time()
    if stream:
        payload["stream"] = True
        r = requests.post(URL, json=payload, stream=True, timeout=timeout)
        r.raise_for_status()
        ttfb, buf = None, bytearray()
        for c in r.iter_content(8192):
            if c:
                if ttfb is None:
                    ttfb = time.time() - t0
                buf += c
        return time.time() - t0, ttfb, bytes(buf)
    r = requests.post(URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return time.time() - t0, None, r.content


def line(label, total, audio, extra=""):
    rtf = total / audio if audio else float("nan")
    print(f"{label:<22} latency={total:6.2f}s  audio={audio:5.2f}s  RTF={rtf:4.2f}  {extra}")


register_bench_voice()
print(f"Endpoint: {URL}  (voice={VOICE!r})\n")

# 1) Cold start (or warm if already up) — first request
print("== 1. First request (includes cold start if scaled to zero) ==")
t, _, b = synth(SHORT)
line("cold/first(short)", t, wav_dur(b))

# 2) Warm latency by input length (3 runs each, report median)
print("\n== 2. Warm latency by input length (median of 3) ==")
for name, text in [("short(~7w)", SHORT), ("medium(~28w)", MEDIUM), ("long(~84w)", LONG)]:
    runs = [synth(text) for _ in range(3)]
    tot = statistics.median(r[0] for r in runs)
    aud = statistics.median(wav_dur(r[2]) for r in runs)
    line(name, tot, aud, extra=f"chars={len(text)}")

# 3) Streaming time-to-first-audio
print("\n== 3. Streaming (time-to-first-audio) ==")
t, ttfb, b = synth(MEDIUM, stream=True)
print(f"stream(medium)         total={t:6.2f}s  TTFA={ttfb:5.2f}s  audio={wav_dur(b):5.2f}s")

# 4) Concurrency: 8 simultaneous (matches @modal.concurrent max_inputs=8)
print("\n== 4. Concurrency: 8 simultaneous short requests (1 warm container) ==")
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(lambda _: synth(SHORT), range(8)))
wall = time.time() - t0
lat = sorted(r[0] for r in res)
print(f"8 concurrent: wall={wall:.2f}s  per-req min={lat[0]:.2f}s "
      f"median={statistics.median(lat):.2f}s max={lat[-1]:.2f}s")
print(f"throughput ~= {8/wall:.2f} req/s vs sequential ~= {1/statistics.median(lat):.2f} req/s")
