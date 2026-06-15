import httpx

from app.core import client
from app.core.config import settings

SPEECH_PATH = "/v1/audio/speech"


async def synthesize(http: httpx.AsyncClient, text: str) -> bytes:
    url = settings.tts_url + SPEECH_PATH
    body = {
        "input": text,
        "instructions": settings.omni_instructions,
        "language": "English",
        "response_format": "wav",
        "seed": settings.omni_seed,
    }
    response = await client.post(http, "TTS", url, json=body)
    return response.content
