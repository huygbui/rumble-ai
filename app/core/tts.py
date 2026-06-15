import array
import io
import time
import wave

import httpx

from app.core.config import settings

SILENCE_THR = 400

CLIENT = httpx.Client(timeout=600, limits=httpx.Limits(max_keepalive_connections=8))


def make_speech_payload(text: str) -> dict:
    if settings.tts_model == "omnivoice":
        return {
            "input": text,
            "instructions": settings.omni_instructions,
            "language": "English",
            "response_format": "wav",
            "seed": settings.omni_seed,
        }
    if settings.tts_model == "fish":
        return {"input": text, "voice": settings.tts_voice, "response_format": "wav", "seed": 58842}
    raise SystemExit(f"TTS_MODEL must be omnivoice or fish; got {settings.tts_model!r}")


def synthesize(text: str) -> tuple[float, bytes]:
    if not settings.tts_url:
        raise SystemExit("set TTS_URL to synthesize (only STITCH_ONLY works without it)")
    t0 = time.time()
    r = CLIENT.post(settings.tts_speech_url, json=make_speech_payload(text))
    r.raise_for_status()
    return time.time() - t0, r.content


def wav_duration(b: bytes) -> float:
    try:
        with wave.open(io.BytesIO(b)) as w:
            return w.getnframes() / w.getframerate()
    except (EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def _read_pcm(b: bytes):
    with wave.open(io.BytesIO(b)) as w:
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
        return w.getframerate(), a


def _write_pcm(sr: int, a: array.array) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.tobytes())
    return buf.getvalue()


def _trim_and_fade(a: array.array, sr: int) -> array.array:
    n = len(a)
    if not n:
        return a
    start = next((i for i, x in enumerate(a) if abs(x) > SILENCE_THR), 0)
    end = n - next((i for i, x in enumerate(reversed(a)) if abs(x) > SILENCE_THR), 0)
    margin = int(sr * 0.015)
    a = array.array("h", a[max(0, start - margin):min(n, end + margin)])
    f = max(1, int(sr * settings.say_fade_ms / 1000))
    for i in range(min(f, len(a))):
        a[i] = int(a[i] * i / f)
        a[-1 - i] = int(a[-1 - i] * i / f)
    return a


def stitch(byte_list: list[bytes]) -> bytes:
    sr = 24000
    out = array.array("h")
    for j, b in enumerate(byte_list):
        sr, a = _read_pcm(b)
        if j and settings.say_gap_ms:
            out.extend(array.array("h", bytes(2 * int(sr * settings.say_gap_ms / 1000))))
        out.extend(_trim_and_fade(a, sr))
    return _write_pcm(sr, out)
