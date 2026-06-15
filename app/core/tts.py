import httpx

from app.core import client
from app.core.config import settings


async def synthesize(http: httpx.AsyncClient, text: str) -> bytes:
    body = {
        "input": text,
        "instructions": settings.omni_instructions,
        "language": "English",
        "response_format": "wav",
        "seed": settings.omni_seed,
    }
    response = await client.post(http, "TTS", settings.tts_speech_url, json=body)
    return response.content
