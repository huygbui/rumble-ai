import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

import httpx

from app.core import clauses, llm, tts
from app.core.config import settings


@dataclass(frozen=True, slots=True)
class Event:
    event: str
    data: dict[str, object]


Item = str | Exception | None


def meta_payload() -> dict[str, object]:
    return {
        "llm": settings.llm_url or None,
        "model": settings.llm_model,
        "tts": settings.tts_url or None,
        "tts_on": bool(settings.tts_url),
        "stt": settings.stt_url or None,
        "stt_model": settings.stt_model,
        "stt_on": bool(settings.stt_url),
    }


async def warm(client: httpx.AsyncClient) -> dict[str, object]:
    stages = [
        (name, url)
        for name, url in (
            ("stt", settings.stt_url),
            ("llm", settings.llm_url),
            ("tts", settings.tts_url),
        )
        if url
    ]
    if not stages:
        return {"ready": False, "stages": []}

    async def warm_stage(name: str, base: str) -> dict[str, object]:
        try:
            async with asyncio.timeout(settings.warm_budget):
                while True:
                    if await _health_ok(client, base):
                        return {"name": name, "status": "ready"}
                    await asyncio.sleep(2)
        except TimeoutError:
            return {"name": name, "status": "failed"}

    results = await asyncio.gather(*(warm_stage(name, base) for name, base in stages))
    return {"ready": all(result["status"] == "ready" for result in results), "stages": results}


async def chat_events(
    client: httpx.AsyncClient,
    messages: list[dict],
) -> AsyncIterator[Event]:
    queue: asyncio.Queue[Item] = asyncio.Queue()
    source = clauses.stream(llm.stream(client, messages))
    events = _speak(client, queue) if settings.tts_url else _text(queue)

    # Keep clause reading independent from event emission; this lets TTS overlap when enabled.
    reader = asyncio.create_task(_read(source, queue))
    try:
        async for event in events:
            yield event
    finally:
        reader.cancel()
        with suppress(asyncio.CancelledError):
            await reader


async def _read(source: AsyncIterator[str], queue: asyncio.Queue[Item]) -> None:
    try:
        async for clause in source:
            await queue.put(clause)
    except Exception as e:
        await queue.put(e)
    await queue.put(None)


async def _text(queue: asyncio.Queue[Item]) -> AsyncIterator[Event]:
    reply: list[str] = []
    failed = False
    while (item := await queue.get()) is not None:
        if isinstance(item, Exception):
            failed = True
            yield Event("error", {"message": str(item)})
            continue

        reply.append(item)
        yield Event("clause", {"text": item})
    if failed:
        return
    yield Event(
        "done",
        {"full_reply": " ".join(reply)},
    )


async def _speak(client: httpx.AsyncClient, queue: asyncio.Queue[Item]) -> AsyncIterator[Event]:
    reply: list[str] = []
    failed = False
    while (item := await queue.get()) is not None:
        if isinstance(item, Exception):
            failed = True
            yield Event("error", {"message": str(item)})
            continue

        reply.append(item)
        payload: dict[str, object] = {"text": item}
        try:
            wav = await tts.synthesize(client, item)
        except Exception as e:
            yield Event("clause", payload)
            yield Event("error", {"message": str(e)})
            continue
        payload["wav_b64"] = base64.b64encode(wav).decode()
        yield Event("clause", payload)
    if failed:
        return
    yield Event(
        "done",
        {"full_reply": " ".join(reply)},
    )

async def _health_ok(client: httpx.AsyncClient, base: str) -> bool:
    try:
        r = await client.get(f"{base}/health", timeout=30)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
