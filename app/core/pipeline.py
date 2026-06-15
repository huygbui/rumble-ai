import asyncio
import base64
import io
import json
import time
import wave
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

import httpx

from app.core import clauses
from app.core.config import settings

STT_EXT = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


@dataclass(frozen=True, slots=True)
class Event:
    event: str
    data: dict[str, object]


QueueItem = Event | str | None


def meta_payload() -> dict[str, object]:
    return {
        "llm": settings.llm_url or None,
        "model": settings.llm_model,
        "tts": settings.tts_url or None,
        "tts_on": settings.tts_on,
        "stt": settings.stt_url or None,
        "stt_model": settings.stt_model,
        "stt_on": settings.stt_on,
    }


async def run_turn(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[Event]:
    out: asyncio.Queue[QueueItem] = asyncio.Queue()
    t0 = time.time()

    async def produce():
        buffer = clauses.ClauseBuffer()
        ttft_sent = False

        try:
            async with client.stream(
                "POST",
                settings.llm_chat_url,
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "max_tokens": settings.chat_max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or None
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if not delta:
                        continue
                    if not ttft_sent and delta.strip():
                        ttft_sent = True
                        await out.put(Event("ttft", {"t": time.time() - t0}))
                    for clause in buffer.feed(delta):
                        await out.put(clause)
            for clause in buffer.flush():
                await out.put(clause)
        except Exception as e:
            await out.put(Event("error", {"message": f"{type(e).__name__}: {e}"}))
        await out.put(None)

    task = asyncio.create_task(produce())
    try:
        async for event in _emit_turn(client, out, t0, audio=settings.tts_on):
            yield event
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _emit_turn(
    client: httpx.AsyncClient,
    items: asyncio.Queue[QueueItem],
    t0: float,
    audio: bool,
) -> AsyncIterator[Event]:
    parts, total_audio, i = [], 0.0, 0
    while (item := await items.get()) is not None:
        if isinstance(item, Event):
            yield item
            continue

        parts.append(item)
        payload: dict[str, object] = {"i": i, "text": item, "t_ready": time.time() - t0}
        if audio:
            try:
                synth_s, wav = await synthesize_clause(client, item)
            except Exception as e:
                yield Event("clause", payload)
                yield Event("error", {"message": f"{type(e).__name__}: {e}"})
                i += 1
                continue
            try:
                with wave.open(io.BytesIO(wav)) as audio_file:
                    dur = audio_file.getnframes() / audio_file.getframerate()
            except (EOFError, wave.Error, ZeroDivisionError):
                dur = 0.0
            total_audio += dur
            payload |= {
                "synth_s": synth_s,
                "audio_s": dur,
                "wav_b64": base64.b64encode(wav).decode(),
            }
        yield Event("clause", payload)
        i += 1
    yield Event(
        "done",
        {
            "wall": time.time() - t0,
            "total_audio": total_audio,
            "n": len(parts),
            "full_reply": " ".join(parts),
        },
    )


async def synthesize_clause(client: httpx.AsyncClient, text: str) -> tuple[float, bytes]:
    t0 = time.time()
    r = await client.post(
        settings.tts_speech_url,
        json={
            "input": text,
            "instructions": settings.omni_instructions,
            "language": "English",
            "response_format": "wav",
            "seed": settings.omni_seed,
        },
    )
    r.raise_for_status()
    return time.time() - t0, r.content


def _stages() -> list[tuple[str, str]]:
    stages = []
    if settings.stt_on:
        stages.append(("stt", settings.stt_url))
    if settings.llm_url:
        stages.append(("llm", settings.llm_url))
    if settings.tts_on:
        stages.append(("tts", settings.tts_url))
    return stages


async def _health_ok(client: httpx.AsyncClient, base: str) -> bool:
    try:
        r = await client.get(f"{base}/health", timeout=30)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def warm(client: httpx.AsyncClient) -> dict[str, object]:
    stages = _stages()
    if not stages:
        return {"ready": False, "stages": []}
    t0 = time.time()

    async def warm_up(name, base):
        while time.time() - t0 < settings.warm_budget:
            if await _health_ok(client, base):
                return {"name": name, "status": "ready", "t": time.time() - t0}
            await asyncio.sleep(2)
        return {"name": name, "status": "failed", "t": time.time() - t0}

    results = await asyncio.gather(*(warm_up(name, base) for name, base in stages))
    return {"ready": all(result["status"] == "ready" for result in results), "stages": results}


async def transcribe(client: httpx.AsyncClient, audio: bytes, ctype: str, attempts: int = 4) -> str:
    ext = STT_EXT.get(ctype, "webm")
    last = ""
    for _ in range(attempts):
        r = await client.post(
            f"{settings.stt_url}/v1/audio/transcriptions",
            files={"file": (f"speech.{ext}", audio, ctype)},
            data={"model": settings.stt_model},
        )
        if r.status_code == 200:
            return (r.json().get("text") or "").strip()
        last = f"{r.status_code} {r.text[:160]}"
        if r.status_code in (302, 303, 307, 308, 502, 503):
            await asyncio.sleep(1.5)
            continue
        r.raise_for_status()
    raise RuntimeError(f"ASR not ready after {attempts} tries: {last}")
