import concurrent.futures as cf
import io
import os
import statistics
import time
import wave

import httpx

BASE = os.environ["TTS_URL"].rstrip("/")
URL = f"{BASE}/v1/audio/speech"
MODEL = os.environ.get("TTS_MODEL", "omnivoice").lower()
OUT_DIR = os.environ.get("TTS_OUT_DIR", "out")
RUN_PERF = os.environ.get("BENCH_PERF", "1") not in ("", "0")
RUN_QUALITY = os.environ.get("BENCH_QUALITY") not in (None, "", "0")
TAG = os.environ.get("BENCH_TAG", MODEL)

OMNI_INSTRUCTIONS = os.environ.get("OMNI_INSTRUCTIONS", "child, australian accent, high pitch")

SHORT = "Hello, this is a short test sentence."
MEDIUM = (
    "The quick brown fox jumps over the lazy dog while the morning sun rises slowly "
    "over the quiet hills, casting long shadows across the dew covered meadow below."
)
LONG = (MEDIUM + " ") * 3

SAMPLE_TEXTS = [
    ("01_short_greeting",
     "Hello! Welcome to the voice demo. How are you doing today?"),
    ("02_medium_narrative",
     "The old lighthouse stood at the edge of the cliff, its beam sweeping slowly across "
     "the dark water. For nearly a century it had guided ships safely home through fog and storm."),
    ("03_long_form",
     "Artificial speech has come a long way in just a few short years. What once sounded "
     "robotic and flat can now carry warmth, rhythm, and genuine emotion. As these models "
     "grow more capable, they open up new possibilities for storytelling, accessibility, and "
     "everyday communication. The challenge ahead is not only making voices that sound human, "
     "but making them expressive, reliable, and fast enough to use in real time, across many "
     "languages."),
    ("04_emotion_tags",
     "I can't believe we actually pulled this off! [laughing] This is the best day ever. "
     "[whispering] But let's keep it a secret for now. [angry] And tell no one who dares to ask."),
    ("05_numbers_prosody",
     "On March 3rd, 2026, the team shipped version 0.22.1, cutting latency by 37 percent. "
     "Isn't that incredible? You can reach me at 555-0192 any time."),
]


def make_payload(text: str, stream: bool = False) -> dict:
    if MODEL == "omnivoice":
        p = {"input": text, "instructions": OMNI_INSTRUCTIONS,
             "language": "English", "response_format": "wav"}
    else:
        raise SystemExit(f"bench supports TTS_MODEL=omnivoice; got {MODEL!r}")
    if stream:
        p["stream"] = True
    return p


def wav_dur(b: bytes) -> float:
    try:
        w = wave.open(io.BytesIO(b))
        return w.getnframes() / w.getframerate()
    except Exception:
        return max(0.0, (len(b) - 44) / (24000 * 2))


def synth(text, stream=False, timeout=600):
    payload = make_payload(text, stream)
    t0 = time.time()
    if stream:
        with httpx.stream("POST", URL, json=payload, timeout=timeout) as r:
            r.raise_for_status()
            ttfb, buf = None, bytearray()
            for c in r.iter_bytes(8192):
                if c:
                    if ttfb is None:
                        ttfb = time.time() - t0
                    buf += c
        return time.time() - t0, ttfb, bytes(buf)
    r = httpx.post(URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return time.time() - t0, None, r.content


def audio_stats(b: bytes) -> str:
    # Cheap signal for clipping/artifact failures; not a quality score.
    try:
        w = wave.open(io.BytesIO(b))
        sr, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    except Exception:
        return "non-WAV body"
    import array
    a = array.array("h")
    a.frombytes(raw[: len(raw) // 2 * 2])
    if not a:
        return "empty audio"
    peak = max(abs(x) for x in a) / 32768.0
    clip = sum(1 for x in a if abs(x) >= 32760) / len(a)
    rms = (sum(x * x for x in a) / len(a)) ** 0.5 / 32768.0
    thr = 327
    lead = next((i for i, x in enumerate(a) if abs(x) > thr), len(a)) / sr
    trail = next((i for i, x in enumerate(reversed(a)) if abs(x) > thr), len(a)) / sr
    flag = " ARTIFACT?" if (clip > 0.02 or rms > 0.45) else ""
    return (f"{sr}Hz {n/sr:5.2f}s  peak={peak:.2f} rms={rms:.2f} clip={clip*100:4.1f}% "
            f"sil(lead/trail)={lead:.2f}/{trail:.2f}s{flag}")


def line(label, total, audio, extra=""):
    rtf = total / audio if audio else float("nan")
    print(f"{label:<22} latency={total:6.2f}s  audio={audio:5.2f}s  RTF={rtf:4.2f}  {extra}")


def run_perf() -> None:
    print("== 1. First request (includes cold start if scaled to zero) ==")
    t, _, b = synth(SHORT)
    line("cold/first(short)", t, wav_dur(b))

    print("\n== 2. Warm latency by input length (median of 3) ==")
    for name, text in [("short(~7w)", SHORT), ("medium(~28w)", MEDIUM), ("long(~84w)", LONG)]:
        runs = [synth(text) for _ in range(3)]
        tot = statistics.median(r[0] for r in runs)
        aud = statistics.median(wav_dur(r[2]) for r in runs)
        line(name, tot, aud, extra=f"chars={len(text)}")

    print("\n== 3. Streaming (time-to-first-audio) ==")
    try:
        t, ttfb, b = synth(MEDIUM, stream=True)
        ttfa = "n/a" if ttfb is None else f"{ttfb:.2f}s"
        print(f"stream(medium)         total={t:6.2f}s  TTFA={ttfa}  audio={wav_dur(b):5.2f}s")
    except Exception as e:
        print(f"streaming errored/unsupported: {e}")

    print("\n== 4. Concurrency: 8 simultaneous short requests (1 warm container) ==")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda _: synth(SHORT), range(8)))
    wall = time.time() - t0
    lat = sorted(r[0] for r in res)
    print(f"8 concurrent: wall={wall:.2f}s  per-req min={lat[0]:.2f}s "
          f"median={statistics.median(lat):.2f}s max={lat[-1]:.2f}s")
    print(f"throughput ~= {8/wall:.2f} req/s vs sequential ~= {1/statistics.median(lat):.2f} req/s")


def run_quality() -> None:
    print(f"\n== 5. Quality samples -> {OUT_DIR}/ ==")
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, text in SAMPLE_TEXTS:
        _, _, b = synth(text)
        path = os.path.join(OUT_DIR, f"{TAG}_{name}.wav")
        with open(path, "wb") as f:
            f.write(b)
        print(f"  {TAG}_{name}.wav  {audio_stats(b)}")


print(f"Endpoint: {URL}\nModel: {MODEL}"
      + f"  instructions={OMNI_INSTRUCTIONS!r}"
      + "\n")
if RUN_PERF:
    run_perf()
if RUN_QUALITY:
    run_quality()
