import array
import io
import os
import queue
import re
import subprocess
import threading
import time
import wave

import httpx

from app.core.config import settings

TTS_URL = settings.tts_url
SPEECH_URL = settings.tts_speech_url
MODEL = settings.tts_model
OUT_DIR = settings.tts_out_dir
PLAY = settings.play
COMPARE = settings.compare
OMNI_INSTRUCTIONS = settings.omni_instructions
OMNI_SEED = settings.omni_seed
VOICE = settings.tts_voice
MAX_LEN = settings.say_max_len
FIRST_MAX = settings.say_first_max
MIN_LEN = settings.say_min_len
GAP_MS = settings.say_gap_ms
FADE_MS = settings.say_fade_ms
SILENCE_THR = 400

CLIENT = httpx.Client(timeout=600, limits=httpx.Limits(max_keepalive_connections=8))


def make_payload(text: str) -> dict:
    if MODEL == "omnivoice":
        return {
            "input": text,
            "instructions": OMNI_INSTRUCTIONS,
            "language": "English",
            "response_format": "wav",
            "seed": OMNI_SEED,
        }
    if MODEL == "fish":
        return {"input": text, "voice": VOICE, "response_format": "wav", "seed": 58842}
    raise SystemExit(f"app.cli.say supports TTS_MODEL=omnivoice|fish; got {MODEL!r}")


def synth(text: str):
    if not TTS_URL:
        raise SystemExit("set TTS_URL to synthesize (only STITCH_ONLY works without it)")
    t0 = time.time()
    r = CLIENT.post(SPEECH_URL, json=make_payload(text))
    r.raise_for_status()
    return time.time() - t0, r.content


def wav_dur(b: bytes) -> float:
    try:
        w = wave.open(io.BytesIO(b))
        return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def _read_pcm(b: bytes):
    w = wave.open(io.BytesIO(b))
    a = array.array("h")
    a.frombytes(w.readframes(w.getnframes()))
    return w.getframerate(), a


def _write_pcm(sr: int, a: array.array) -> bytes:
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(a.tobytes())
    w.close()
    return buf.getvalue()


def _trim_and_fade(a: array.array, sr: int) -> array.array:
    n = len(a)
    if not n:
        return a
    start = next((i for i, x in enumerate(a) if abs(x) > SILENCE_THR), 0)
    end = n - next((i for i, x in enumerate(reversed(a)) if abs(x) > SILENCE_THR), 0)
    margin = int(sr * 0.015)
    a = array.array("h", a[max(0, start - margin):min(n, end + margin)])
    f = max(1, int(sr * FADE_MS / 1000))
    for i in range(min(f, len(a))):
        a[i] = int(a[i] * i / f)
        a[-1 - i] = int(a[-1 - i] * i / f)
    return a


def stitch(byte_list: list[bytes]) -> bytes:
    sr = 24000
    out = array.array("h")
    for j, b in enumerate(byte_list):
        sr, a = _read_pcm(b)
        if j and GAP_MS:
            out.extend(array.array("h", bytes(2 * int(sr * GAP_MS / 1000))))
        out.extend(_trim_and_fade(a, sr))
    return _write_pcm(sr, out)


ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "mt",
    "vs",
    "jr",
    "sr",
    "prof",
    "sgt",
    "gen",
    "rev",
    "hon",
    "co",
    "inc",
    "ltd",
    "etc",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for p in re.split(r"(?<=[.!?])\s+", text.strip()):
        p = p.strip()
        if not p:
            continue
        if out:
            toks = out[-1].split()
            tail = toks[-1].lower().rstrip(".") if toks else ""
            if tail in ABBREV:
                out[-1] = out[-1] + " " + p
                continue
        out.append(p)
    return out


def _merge_tiny(chunks: list[str], min_len: int = MIN_LEN) -> list[str]:
    out: list[str] = []
    for c in chunks:
        if out and len(out[-1]) < min_len:
            out[-1] = (out[-1] + " " + c).strip()
        else:
            out.append(c)
    if len(out) >= 2 and len(out[-1]) < min_len:
        out[-2] = (out[-2] + " " + out.pop()).strip()
    return out


def _shorten_first(chunks: list[str], first_max: int = FIRST_MAX) -> list[str]:
    if not chunks or len(chunks[0]) <= first_max:
        return chunks
    head = chunks[0]
    m = next((mm for mm in re.finditer(r"[,;:]\s+", head) if mm.start() >= 20), None)
    if not m:
        return chunks
    first, rest = head[: m.start() + 1].strip(), head[m.end():].strip()
    return [first, rest] + chunks[1:]


def split_clauses(text: str, max_len: int = MAX_LEN) -> list[str]:
    chunks: list[str] = []
    for sent in _sentences(text):
        if len(sent) <= max_len:
            chunks.append(sent)
            continue
        buf = ""
        for part in re.split(r"(?<=[,;:])\s+", sent):
            if len(buf) + len(part) + 1 <= max_len:
                buf = (buf + " " + part).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
    chunks = _merge_tiny(chunks)
    chunks = _shorten_first(chunks)
    return chunks


def stream(chunks: list[str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    q: queue.Queue = queue.Queue()
    t0 = time.time()

    def producer():
        for i, c in enumerate(chunks):
            try:
                tt, b = synth(c)
                q.put((i, c, b, tt))
            except Exception as e:
                q.put((i, c, None, e))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    first = None
    total_audio = 0.0
    collected: list = []
    while True:
        item = q.get()
        if item is None:
            break
        i, c, b, tt = item
        if b is None:
            print(f"  [{i}] FAILED: {tt}")
            continue
        if first is None:
            first = time.time() - t0
            print(f"  >> TTFA (first clause audio ready): {first:.2f}s")
        path = os.path.join(OUT_DIR, f"say_{i:02d}.wav")
        with open(path, "wb") as f:
            f.write(b)
        collected.append((i, b))
        dur = wav_dur(b)
        total_audio += dur
        print(f"  [{i}] synth {tt:5.2f}s  audio {dur:5.2f}s  | {c[:55]!r}")
        if PLAY:
            subprocess.run(["afplay", path], check=False)
    wall = time.time() - t0
    print(f"  -- chunked total wall {wall:.2f}s for {total_audio:.2f}s audio ({len(chunks)} clauses)")
    if collected:
        stitched = stitch([b for _, b in sorted(collected)])
        sp = os.path.join(OUT_DIR, "say.wav")
        with open(sp, "wb") as f:
            f.write(stitched)
        print(f"  -- stitched -> {sp} ({wav_dur(stitched):.2f}s, gap={GAP_MS}ms fade={FADE_MS}ms)")


__all__ = [
    "ABBREV",
    "CLIENT",
    "COMPARE",
    "FADE_MS",
    "FIRST_MAX",
    "GAP_MS",
    "MAX_LEN",
    "MIN_LEN",
    "MODEL",
    "OMNI_INSTRUCTIONS",
    "OMNI_SEED",
    "OUT_DIR",
    "PLAY",
    "SPEECH_URL",
    "SILENCE_THR",
    "TTS_URL",
    "VOICE",
    "make_payload",
    "split_clauses",
    "stitch",
    "stream",
    "synth",
    "wav_dur",
]
