import httpx

from app.core import client
from app.core.config import settings

AUDIO_EXT_BY_TYPE = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


async def transcribe(http: httpx.AsyncClient, audio: bytes, content_type: str) -> str:
    filename = f"speech.{AUDIO_EXT_BY_TYPE.get(content_type, 'webm')}"
    response = await client.post(
        http,
        "STT",
        settings.stt_transcriptions_url,
        files={"file": (filename, audio, content_type)},
        data={"model": settings.stt_model},
    )
    return (response.json().get("text") or "").strip()
