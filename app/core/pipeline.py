import asyncio
import base64
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import field_validator

from app.core import dialogue, speech
from app.core.settings import AppSettings, strip_url


class PipelineSettings(AppSettings):
    stt_url: str = ""
    stt_model: str = "Qwen/Qwen3-ASR-0.6B"
    warm_budget: int = 300

    @field_validator("stt_url", mode="before")
    @classmethod
    def _strip_url(cls, value: str | None) -> str:
        return strip_url(value)


pipeline_settings = PipelineSettings()

STT_URL = pipeline_settings.stt_url
STT_MODEL = pipeline_settings.stt_model
WARM_BUDGET = pipeline_settings.warm_budget
TTS_ON = bool(speech.TTS_URL)
STT_ON = bool(STT_URL)

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
    data: dict[str, Any]


def meta_payload() -> dict[str, Any]:
    return {
        "llm": dialogue.LLM_URL or None,
        "model": dialogue.LLM_MODEL,
        "tts": speech.TTS_URL or None,
        "tts_model": speech.MODEL,
        "tts_on": TTS_ON,
        "stt": STT_URL or None,
        "stt_model": STT_MODEL,
        "stt_on": STT_ON,
    }


async def run_turn(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[StreamEvent]:
    clauses: asyncio.Queue = asyncio.Queue()
    t0 = time.time()

    async def produce():
        streamer = dialogue.ClauseStreamer()
        ttft_sent = False
        try:
            async with client.stream("POST", dialogue.LLM_CHAT_URL, json=dialogue.llm_payload(messages)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not (delta := dialogue.parse_sse_delta(line)):
                        continue
                    if not ttft_sent and delta.strip():
                        ttft_sent = True
                        await clauses.put(StreamEvent("ttft", {"t": time.time() - t0}))
                    for clause in streamer.feed(delta):
                        await clauses.put(clause)
            for clause in streamer.flush():
                await clauses.put(clause)
        except Exception as e:
            await clauses.put(StreamEvent("error", {"message": f"{type(e).__name__}: {e}"}))
        await clauses.put(None)

    task = asyncio.create_task(produce())
    try:
        gen = _voice_turn(client, clauses, t0) if TTS_ON else _text_turn(clauses, t0)
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
    parts, wavs, total_audio, i = [], [], 0.0, 0
    while (item := await clauses.get()) is not None:
        if isinstance(item, StreamEvent):
            yield item
            continue
        try:
            synth_s, wav = await synth(client, item)
        except Exception as e:
            yield StreamEvent("error", {"message": f"{type(e).__name__}: {e}"})
            continue
        dur = speech.wav_dur(wav)
        total_audio += dur
        parts.append(item)
        wavs.append((i, wav))
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
    await asyncio.to_thread(_save_stitched, wavs)
    yield StreamEvent(
        "done",
        {
            "wall": time.time() - t0,
            "total_audio": total_audio,
            "n": len(parts),
            "full_reply": " ".join(parts),
        },
    )


async def synth(client: httpx.AsyncClient, text: str) -> tuple[float, bytes]:
    t0 = time.time()
    r = await client.post(speech.SPEECH_URL, json=speech.make_payload(text))
    r.raise_for_status()
    return time.time() - t0, r.content


def _save_stitched(wavs: list[tuple[int, bytes]]) -> None:
    if not wavs:
        return
    try:
        out = Path(speech.OUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        (out / "web.wav").write_bytes(speech.stitch([wav for _, wav in sorted(wavs)]))
    except Exception:
        pass


def _stages() -> list[tuple[str, str]]:
    stages = []
    if STT_ON:
        stages.append(("stt", STT_URL))
    if dialogue.LLM_URL:
        stages.append(("llm", dialogue.LLM_URL))
    if TTS_ON:
        stages.append(("tts", speech.TTS_URL))
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
        while time.time() - t0 < WARM_BUDGET:
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
        print(f"  warm: {name} {'ready' if ok else 'FAILED'} in {t:.1f}s", flush=True)
        yield StreamEvent("stage", {"name": name, "status": "ready" if ok else "failed", "t": t})
    yield StreamEvent("done", {"ready": all_ok})


async def transcribe(client: httpx.AsyncClient, audio: bytes, ctype: str, attempts: int = 4) -> str:
    ext = STT_EXT.get(ctype, "webm")
    last = ""
    for _ in range(attempts):
        r = await client.post(
            f"{STT_URL}/v1/audio/transcriptions",
            files={"file": (f"speech.{ext}", audio, ctype)},
            data={"model": STT_MODEL},
        )
        if r.status_code == 200:
            return (r.json().get("text") or "").strip()
        last = f"{r.status_code} {r.text[:160]}"
        if r.status_code in (302, 303, 307, 308, 502, 503):
            await asyncio.sleep(1.5)
            continue
        r.raise_for_status()
    raise RuntimeError(f"ASR not ready after {attempts} tries: {last}")
