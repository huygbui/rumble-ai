import asyncio
import base64
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

import httpx

from app.core import llm, text_chunks, tts
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
class StreamEvent:
    event: str
    data: dict[str, object]


def meta_payload() -> dict[str, object]:
    return {
        "llm": settings.llm_url or None,
        "model": settings.llm_model,
        "tts": settings.tts_url or None,
        "tts_model": settings.tts_model,
        "tts_on": settings.tts_on,
        "stt": settings.stt_url or None,
        "stt_model": settings.stt_model,
        "stt_on": settings.stt_on,
    }


async def run_turn(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[StreamEvent]:
    clauses: asyncio.Queue = asyncio.Queue()
    t0 = time.time()

    async def produce():
        splitter = text_chunks.StreamingClauseSplitter()
        ttft_sent = False
        try:
            async with client.stream("POST", settings.llm_chat_url, json=llm.make_chat_payload(messages)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not (delta := llm.parse_sse_delta(line)):
                        continue
                    if not ttft_sent and delta.strip():
                        ttft_sent = True
                        await clauses.put(StreamEvent("ttft", {"t": time.time() - t0}))
                    for clause in splitter.feed(delta):
                        await clauses.put(clause)
            for clause in splitter.flush():
                await clauses.put(clause)
        except Exception as e:
            await clauses.put(StreamEvent("error", {"message": f"{type(e).__name__}: {e}"}))
        await clauses.put(None)

    task = asyncio.create_task(produce())
    try:
        gen = _voice_turn(client, clauses, t0) if settings.tts_on else _text_turn(clauses, t0)
        async for event in gen:
            yield event
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _text_turn(clauses, t0) -> AsyncIterator[StreamEvent]:
    parts: list[str] = []
    while (item := await clauses.get()) is not None:
        if isinstance(item, StreamEvent):
            yield item
            continue
        parts.append(item)
        yield StreamEvent("clause", {"i": len(parts) - 1, "text": item, "t_ready": time.time() - t0})
    yield StreamEvent(
        "done",
        {"wall": time.time() - t0, "total_audio": 0.0, "n": len(parts), "full_reply": " ".join(parts)},
    )


async def _voice_turn(client, clauses, t0) -> AsyncIterator[StreamEvent]:
    parts, total_audio, i = [], 0.0, 0
    while (item := await clauses.get()) is not None:
        if isinstance(item, StreamEvent):
            yield item
            continue
        try:
            synth_s, wav = await synthesize_clause(client, item)
        except Exception as e:
            yield StreamEvent("error", {"message": f"{type(e).__name__}: {e}"})
            continue
        dur = tts.wav_duration(wav)
        total_audio += dur
        parts.append(item)
        yield StreamEvent(
            "clause",
            {
                "i": i,
                "text": item,
                "t_ready": time.time() - t0,
                "synth_s": synth_s,
                "audio_s": dur,
                "wav_b64": base64.b64encode(wav).decode(),
            },
        )
        i += 1
    yield StreamEvent(
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
    r = await client.post(settings.tts_speech_url, json=tts.make_speech_payload(text))
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


async def warm_stream(client: httpx.AsyncClient) -> AsyncIterator[StreamEvent]:
    stages = _stages()
    if not stages:
        yield StreamEvent("done", {"ready": False})
        return
    t0 = time.time()

    async def warm_up(name, base):
        while time.time() - t0 < settings.warm_budget:
            if await _health_ok(client, base):
                return name, True, time.time() - t0
            await asyncio.sleep(2)
        return name, False, time.time() - t0

    yield StreamEvent("warming", {"stages": [n for n, _ in stages]})
    tasks = [asyncio.create_task(warm_up(n, b)) for n, b in stages]
    all_ok = True
    for done in asyncio.as_completed(tasks):
        name, ok, t = await done
        all_ok &= ok
        yield StreamEvent("stage", {"name": name, "status": "ready" if ok else "failed", "t": t})
    yield StreamEvent("done", {"ready": all_ok})


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
