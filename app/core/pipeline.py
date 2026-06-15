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


QueueItem = Event | str | None


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


async def run_turn(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[Event]:
    out: asyncio.Queue[QueueItem] = asyncio.Queue()
    t0 = time.time()

    async def produce():
        buffer = clauses.ClauseBuffer()
        ttft_sent = False

        try:
            async for delta in llm.stream(client, messages):
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
        async for event in _emit_turn(client, out, t0, audio=bool(settings.tts_url)):
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
    parts, i = [], 0
    while (item := await items.get()) is not None:
        if isinstance(item, Event):
            yield item
            continue

        parts.append(item)
        payload: dict[str, object] = {"i": i, "text": item, "t_ready": time.time() - t0}
        if audio:
            try:
                wav = await tts.synthesize(client, item)
            except Exception as e:
                yield Event("clause", payload)
                yield Event("error", {"message": f"{type(e).__name__}: {e}"})
                i += 1
                continue
            payload["wav_b64"] = base64.b64encode(wav).decode()
        yield Event("clause", payload)
        i += 1
    yield Event(
        "done",
        {
            "wall": time.time() - t0,
            "n": len(parts),
            "full_reply": " ".join(parts),
        },
    )


def _stages() -> list[tuple[str, str]]:
    return [
        (name, url)
        for name, url in (
            ("stt", settings.stt_url),
            ("llm", settings.llm_url),
            ("tts", settings.tts_url),
        )
        if url
    ]


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
