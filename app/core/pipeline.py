import asyncio
import base64
import time
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


Part = Event | str | None


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

    started_at = time.monotonic()
    deadline = started_at + settings.warm_budget

    async def warm_stage(name: str, base: str) -> dict[str, object]:
        while (remaining := deadline - time.monotonic()) > 0:
            if await _health_ok(client, base, timeout=min(30, remaining)):
                return {"name": name, "status": "ready", "t": time.monotonic() - started_at}
            await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
        return {"name": name, "status": "failed", "t": time.monotonic() - started_at}

    results = await asyncio.gather(*(warm_stage(name, base) for name, base in stages))
    return {"ready": all(result["status"] == "ready" for result in results), "stages": results}


async def chat_events(
    client: httpx.AsyncClient,
    messages: list[dict],
) -> AsyncIterator[Event]:
    parts: asyncio.Queue[Part] = asyncio.Queue()
    started_at = time.monotonic()

    task = asyncio.create_task(_pipe(_llm(client, messages, started_at), parts))
    try:
        async for event in _events(
            client,
            parts,
            started_at,
            audio=bool(settings.tts_url),
        ):
            yield event
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _llm(
    client: httpx.AsyncClient,
    messages: list[dict],
    started_at: float,
) -> AsyncIterator[Event | str]:
    buffer = clauses.ClauseBuffer()
    ttft_sent = False

    async for delta in llm.stream(client, messages):
        if delta.strip() and not ttft_sent:
            ttft_sent = True
            yield Event("ttft", {"t": time.monotonic() - started_at})
        for clause in buffer.feed(delta):
            yield clause

    for clause in buffer.flush():
        yield clause


async def _pipe(source: AsyncIterator[Event | str], parts: asyncio.Queue[Part]) -> None:
    try:
        async for part in source:
            await parts.put(part)
    except Exception as e:
        await parts.put(Event("error", {"message": f"{type(e).__name__}: {e}"}))
    await parts.put(None)


async def _events(
    client: httpx.AsyncClient,
    parts: asyncio.Queue[Part],
    started_at: float,
    audio: bool,
) -> AsyncIterator[Event]:
    text_parts: list[str] = []
    failed = False
    i = 0
    while (part := await parts.get()) is not None:
        if isinstance(part, Event):
            failed = failed or part.event == "error"
            yield part
            continue

        text_parts.append(part)
        payload: dict[str, object] = {"i": i, "text": part, "t_ready": time.monotonic() - started_at}
        if audio:
            try:
                wav = await tts.synthesize(client, part)
            except Exception as e:
                yield Event("clause", payload)
                yield Event("error", {"message": f"{type(e).__name__}: {e}"})
                i += 1
                continue
            payload["wav_b64"] = base64.b64encode(wav).decode()
        yield Event("clause", payload)
        i += 1
    if failed:
        return
    yield Event(
        "done",
        {
            "wall": time.monotonic() - started_at,
            "full_reply": " ".join(text_parts),
        },
    )


async def _health_ok(client: httpx.AsyncClient, base: str, timeout: float) -> bool:
    try:
        r = await client.get(f"{base}/health", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
