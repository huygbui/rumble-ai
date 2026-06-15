import base64
import mimetypes
from functools import cache
from pathlib import Path

import httpx

from app.core import client
from app.core.config import settings


@cache
def _ref_audio() -> str | None:
    """Resolve the clone anchor to a form OmniVoice accepts (public URL or data URI).

    Returns None when no usable reference is configured, so synthesize() falls back
    to the voice-design path. Resolved once (settings are frozen, the file is static).
    """
    ref = settings.omni_ref_audio
    if not ref or not settings.omni_ref_text:
        return None
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref)
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _payload(text: str) -> dict:
    body = {
        "input": text,
        "language": settings.language,
        "response_format": "wav",
        "seed": settings.omni_seed,
    }
    ref = _ref_audio()
    if ref is not None:
        # Clone one fixed voice so timbre stays constant across clauses.
        body["ref_audio"] = ref
        body["ref_text"] = settings.omni_ref_text
    else:
        # No anchor: design a voice from attributes (drifts clause-to-clause).
        body["instructions"] = settings.omni_instructions
    return body


async def synthesize(http: httpx.AsyncClient, text: str) -> bytes:
    response = await client.post(http, settings.tts_speech_url, json=_payload(text))
    return response.content
