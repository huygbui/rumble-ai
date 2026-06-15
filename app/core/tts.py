import io
import wave

from app.core.config import settings

def make_speech_payload(text: str) -> dict:
    if settings.tts_model == "omnivoice":
        return {
            "input": text,
            "instructions": settings.omni_instructions,
            "language": "English",
            "response_format": "wav",
            "seed": settings.omni_seed,
        }
    raise SystemExit(f"TTS_MODEL must be omnivoice; got {settings.tts_model!r}")


def wav_duration(b: bytes) -> float:
    try:
        with wave.open(io.BytesIO(b)) as w:
            return w.getnframes() / w.getframerate()
    except (EOFError, wave.Error, ZeroDivisionError):
        return 0.0
