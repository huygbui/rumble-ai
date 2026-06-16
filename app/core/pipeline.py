import asyncio
import base64
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

import httpx

from app.core import clauses, llm, tts
from app.core.config import settings


EventName = Literal["clause", "done", "error"]


@dataclass(frozen=True, slots=True)
class Event:
    event: EventName
    data: dict[str, object]


def _clause_event(text: str, wav_b64: str | None = None) -> Event:
    data: dict[str, object] = {"text": text}
    if wav_b64 is not None:
        data["wav_b64"] = wav_b64
    return Event("clause", data)


def _done_event(full_reply: str) -> Event:
    return Event("done", {"full_reply": full_reply})


def _error_event(message: object) -> Event:
    return Event("error", {"message": str(message)})


Item = str | Exception | None


def meta_payload() -> dict[str, object]:
    return {
        "language": settings.language,
        "stt_language": settings.stt_language,
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
        ready = await _poll_health(client, base, settings.warm_budget)
        return {"name": name, "status": "ready" if ready else "failed"}

    results = await asyncio.gather(*(warm_stage(name, base) for name, base in stages))
    return {"ready": all(result["status"] == "ready" for result in results), "stages": results}


# Readiness cache: a /health 200 stays trusted for ready_cache_ttl, so warm requests skip
# the pre-flight poll (one us-east RTT) entirely. A stage that scales to zero in the meantime
# is caught by the retry layer in app.core.client, which waits out the cold-start 503 on the
# real call — so a stale-True here costs at most a retry, never a failure.
_ready_seen: dict[str, float] = {}


def _mark_ready(base: str) -> None:
    _ready_seen[base] = time.monotonic()


def _recently_ready(base: str) -> bool:
    seen = _ready_seen.get(base)
    return seen is not None and time.monotonic() - seen < settings.ready_cache_ttl


async def ensure_ready(client: httpx.AsyncClient, *bases: str) -> bool:
    """Wait out a cold start: poll /health per stage, bounded by cold_start_budget.

    Skips stages confirmed ready within ready_cache_ttl so warm requests pay no pre-flight
    RTT; a stage gone cold meanwhile is absorbed by the retry layer on the real call. Empty
    URLs are skipped; returns False if a stage never comes up.
    """
    targets = [base for base in bases if base and not _recently_ready(base)]
    if not targets:
        return True
    results = await asyncio.gather(
        *(_poll_health(client, base, settings.cold_start_budget) for base in targets)
    )
    return all(results)


def _with_language(messages: list[dict]) -> list[dict]:
    """Append a "reply in <language>" directive for non-English languages (no-op for English)."""
    directive = f" Always reply in {settings.language}."
    if settings.language == "English" or not messages:
        return messages
    head, *rest = messages
    if head.get("role") == "system":
        if directive.strip() in head.get("content", ""):
            return messages
        return [{**head, "content": head.get("content", "") + directive}, *rest]
    system = {"role": "system", "content": settings.chat_system_prompt}
    return [system, *messages]


async def chat_events(
    client: httpx.AsyncClient,
    messages: list[dict],
) -> AsyncIterator[Event]:
    queue: asyncio.Queue[Item] = asyncio.Queue()
    source = clauses.stream(llm.stream(client, _with_language(messages)))
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
            yield _error_event(item)
            continue

        reply.append(item)
        yield _clause_event(item)
    if failed:
        return
    yield _done_event(" ".join(reply))


async def _speak(client: httpx.AsyncClient, queue: asyncio.Queue[Item]) -> AsyncIterator[Event]:
    reply: list[str] = []
    failed = False
    while (item := await queue.get()) is not None:
        if isinstance(item, Exception):
            failed = True
            yield _error_event(item)
            continue

        reply.append(item)
        try:
            wav = await tts.synthesize(client, item)
        except Exception as e:
            yield _clause_event(item)
            yield _error_event(e)
            continue
        yield _clause_event(item, wav_b64=base64.b64encode(wav).decode())
    if failed:
        return
    yield _done_event(" ".join(reply))


async def _health_ok(client: httpx.AsyncClient, base: str) -> bool:
    try:
        r = await client.get(f"{base}/health", timeout=30)
    except httpx.HTTPError:
        return False
    if r.status_code == 200:
        _mark_ready(base)
        return True
    return False


async def _poll_health(client: httpx.AsyncClient, base: str, budget: float) -> bool:
    """Poll /health until it answers 200 or the budget elapses."""
    try:
        async with asyncio.timeout(budget):
            while not await _health_ok(client, base):
                await asyncio.sleep(2)
            return True
    except TimeoutError:
        return False
