# web.py
# Browser front-end for the chat.py voice loop -- so you can HEAR the clause-overlap win
# instead of reading TTFA numbers off a terminal. Each clause's WAV is pushed to the page the
# instant it's synthesized and played in order (Web Audio) while later clauses are still
# generating: you hear clause 0 while clause 2 is mid-flight. Fully async (httpx + asyncio):
# one task streams the LLM into a clause queue while this request synthesizes + emits in order.
#
#   export LLM_URL="<flash url from `modal deploy llm/qwen3_5_4b.py`>"
#   export TTS_URL="<flash url from `modal deploy tts/omnivoice.py`>"   # omit -> text-only
#   export STT_URL="<flash url from `modal deploy stt/qwen3_asr.py`>"   # omit -> mic disabled
#   uvicorn web:app            # or `python web.py`  ->  http://127.0.0.1:8000
#
# /api/chat streams an SSE turn (ttft/clause/done/error); history lives in the browser.
import asyncio
import base64
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ClassVar, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

import chat  # ClauseStreamer, llm_payload/parse_sse_delta, LLM_BASE/LLM_CHAT_URL/LLM_MODEL
import say   # make_payload, wav_dur, stitch, URL/BASE/MODEL/OUT_DIR

INDEX = Path(__file__).parent / "web" / "index.html"

STT_BASE = os.environ.get("STT_URL", "").rstrip("/")
STT_MODEL = os.environ.get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
WARM_BUDGET = int(os.environ.get("WARM_BUDGET", "300"))  # per-stage warm-up deadline (s)

TTS_ON = bool(say.BASE)
STT_ON = bool(STT_BASE)

# Browser MediaRecorder hands us webm/ogg/mp4; vLLM[audio] decodes via soundfile + PyAV.
STT_EXT = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a",
           "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav"}

# --- API schema (Pydantic) ------------------------------------------------------------
# One source of truth for every wire shape the browser sees: the POST body it sends, the JSON
# the meta/stt endpoints return, and each SSE frame streamed during a turn or warm-up. Each
# Event subclass owns its SSE `event:` name and renders itself as a native fastapi.sse
# ServerSentEvent, so the streaming endpoints just yield typed objects -- no dict literals,
# stringly-typed names, or hand-rolled wire framing.
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class MetaResponse(BaseModel):
    llm: str | None
    model: str
    tts: str | None
    tts_model: str
    tts_on: bool
    stt: str | None
    stt_model: str
    stt_on: bool


class SttResponse(BaseModel):  # one of these is set; the route drops the null one (exclude_none)
    text: str | None = None
    error: str | None = None


class Event(BaseModel):
    """Base for an SSE frame: a subclass sets `event` (the SSE event name) + its data fields.
    .to_sse() wraps it as a native ServerSentEvent -- FastAPI does the wire framing, keep-alive
    pings, and disconnect handling, so the streaming endpoints just yield these."""
    event: ClassVar[str]

    def to_sse(self) -> ServerSentEvent:
        return ServerSentEvent(data=self, event=self.event)


class Ttft(Event):       # first token landed -- the felt-latency clock
    event = "ttft"
    t: float


class Clause(Event):     # one ready clause (synth_s/audio_s/wav_b64 set only in voice mode)
    event = "clause"
    i: int
    text: str
    t_ready: float
    synth_s: float | None = None
    audio_s: float | None = None
    wav_b64: str | None = None


class Done(Event):       # end of a turn
    event = "done"
    wall: float
    total_audio: float
    n: int
    full_reply: str


class Error(Event):      # transport/LLM/TTS failure, surfaced mid-stream
    event = "error"
    message: str


class Warming(Event):    # warm-up began for these stages
    event = "warming"
    stages: list[str]


class Stage(Event):      # one stage finished warming
    event = "stage"
    name: str
    status: Literal["ready", "failed"]
    t: float


class WarmDone(Event):   # warm-up finished (shares the SSE name "done" with a turn's Done)
    event = "done"
    ready: bool


# --- One conversational turn, as a stream of typed SSE Events --------------------------
# A producer task streams the LLM into a clause queue; this generator pulls clauses and (in
# voice mode) synthesizes them. Synthesis runs while the producer keeps reading the LLM -- the
# same overlap as chat.converse(), so first audio tracks the FIRST clause, not the whole reply.
# The queue carries either an Event to pass straight through (Ttft/Error) or a raw clause
# string the consumer enriches into a Clause once it's (optionally) synthesized.
async def run_turn(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[Event]:
    clauses: asyncio.Queue = asyncio.Queue()
    t0 = time.time()

    async def produce():
        streamer = chat.ClauseStreamer()
        ttft_sent = False
        try:
            async with client.stream("POST", chat.LLM_CHAT_URL, json=chat.llm_payload(messages)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not (delta := chat.parse_sse_delta(line)):
                        continue
                    if not ttft_sent and delta.strip():
                        ttft_sent = True
                        await clauses.put(Ttft(t=time.time() - t0))
                    for clause in streamer.feed(delta):
                        await clauses.put(clause)
            for clause in streamer.flush():
                await clauses.put(clause)
        except Exception as e:  # surface transport/LLM errors instead of hanging the stream
            await clauses.put(Error(message=f"{type(e).__name__}: {e}"))
        await clauses.put(None)

    task = asyncio.create_task(produce())
    try:
        gen = _voice_turn(client, clauses, t0) if TTS_ON else _text_turn(clauses, t0)
        async for event in gen:
            yield event
    finally:
        task.cancel()  # clean up the producer if the client disconnects mid-turn


# Text-only (no TTS_URL): stream clauses as text, no audio -- dev without the GPU.
async def _text_turn(clauses, t0) -> AsyncIterator[Event]:
    parts: list[str] = []
    while (item := await clauses.get()) is not None:
        if isinstance(item, Event):
            yield item  # ttft / error, straight through
            continue
        parts.append(item)
        yield Clause(i=len(parts) - 1, text=item, t_ready=time.time() - t0)
    yield Done(wall=time.time() - t0, total_audio=0.0, n=len(parts), full_reply=" ".join(parts))


# Voice: synthesize each clause as it arrives and emit ready audio in order.
async def _voice_turn(client, clauses, t0) -> AsyncIterator[Event]:
    parts, wavs, total_audio, i = [], [], 0.0, 0
    while (item := await clauses.get()) is not None:
        if isinstance(item, Event):
            yield item  # ttft / error
            continue
        try:
            synth_s, wav = await synth(client, item)
        except Exception as e:
            yield Error(message=f"{type(e).__name__}: {e}")
            continue
        dur = say.wav_dur(wav)
        total_audio += dur
        parts.append(item)
        wavs.append((i, wav))
        yield Clause(i=i, text=item, t_ready=time.time() - t0, synth_s=synth_s,
                     audio_s=dur, wav_b64=base64.b64encode(wav).decode())
        i += 1
    await asyncio.to_thread(_save_stitched, wavs)
    yield Done(wall=time.time() - t0, total_audio=total_audio, n=len(parts), full_reply=" ".join(parts))


async def synth(client: httpx.AsyncClient, text: str) -> tuple[float, bytes]:
    t0 = time.time()
    r = await client.post(say.URL, json=say.make_payload(text))
    r.raise_for_status()
    return time.time() - t0, r.content


def _save_stitched(wavs: list[tuple[int, bytes]]) -> None:
    # Persist the seamless stitched reply too (parity with chat.py), best-effort.
    if not wavs:
        return
    try:
        out = Path(say.OUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        (out / "web.wav").write_bytes(say.stitch([wav for _, wav in sorted(wavs)]))
    except Exception:
        pass


# --- Warm-up: every stage scales to zero, so the first hit cold-starts (GPU snapshots make
# that ~seconds, but it's jarring on a button tap). /api/warm waits each stage up, streaming a
# `stage` event as each lands so the page can light its status dots. ----------------------
def _stages() -> list[tuple[str, str]]:
    # Configured stages in pipeline order: mic -> brain -> voice.
    stages = []
    if STT_ON:
        stages.append(("stt", STT_BASE))
    if chat.LLM_BASE:
        stages.append(("llm", chat.LLM_BASE))
    if TTS_ON:
        stages.append(("tts", say.BASE))
    return stages


async def _health_ok(client: httpx.AsyncClient, base: str) -> bool:
    # 200 == ready. A scaled-to-zero endpoint returns 503/303 while it wakes, so one probe both
    # reports readiness AND nudges the container up; warm_up loops it to actually wait one up.
    try:
        r = await client.get(f"{base}/health", timeout=30)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def warm_stream(client: httpx.AsyncClient) -> AsyncIterator[Event]:
    stages = _stages()
    if not stages:
        yield WarmDone(ready=False)
        return
    t0 = time.time()

    async def warm_up(name, base):
        while time.time() - t0 < WARM_BUDGET:
            if await _health_ok(client, base):
                return name, True, time.time() - t0
            await asyncio.sleep(2)  # cold start in progress -> re-probe, keep nudging
        return name, False, time.time() - t0

    yield Warming(stages=[n for n, _ in stages])
    tasks = [asyncio.create_task(warm_up(n, b)) for n, b in stages]
    all_ok = True
    for done in asyncio.as_completed(tasks):
        name, ok, t = await done
        all_ok &= ok
        print(f"  warm: {name} {'ready' if ok else 'FAILED'} in {t:.1f}s", flush=True)
        yield Stage(name=name, status="ready" if ok else "failed", t=t)
    yield WarmDone(ready=all_ok)


# Mic -> text. Forward the recorded audio to the OpenAI-compatible ASR endpoint as multipart.
# Cold start: while the scaled-to-zero container wakes, Modal proxies a 303 (which httpx would
# follow as a body-less GET) instead of holding the request -- so with follow_redirects off we
# SEE the 303 and re-POST the same multipart until it serves a 200. (Verified 2026-06-15.)
async def transcribe(client: httpx.AsyncClient, audio: bytes, ctype: str, attempts: int = 4) -> str:
    ext = STT_EXT.get(ctype, "webm")
    last = ""
    for _ in range(attempts):
        r = await client.post(
            f"{STT_BASE}/v1/audio/transcriptions",
            files={"file": (f"speech.{ext}", audio, ctype)},
            data={"model": STT_MODEL},  # language auto-detected; add "language":"en" to force
        )
        if r.status_code == 200:
            return (r.json().get("text") or "").strip()
        last = f"{r.status_code} {r.text[:160]}"
        if r.status_code in (302, 303, 307, 308, 502, 503):  # cold-start handoff -> re-POST
            await asyncio.sleep(1.5)
            continue
        r.raise_for_status()
    raise RuntimeError(f"ASR not ready after {attempts} tries: {last}")


# --- HTTP API -------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled async client for every upstream host (follow_redirects off for the STT 303 dance).
    async with httpx.AsyncClient(timeout=600, follow_redirects=False) as client:
        app.state.http = client
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/api/meta")
def meta() -> MetaResponse:
    return MetaResponse(llm=chat.LLM_BASE or None, model=chat.LLM_MODEL,
                        tts=say.BASE or None, tts_model=say.MODEL, tts_on=TTS_ON,
                        stt=STT_BASE or None, stt_model=STT_MODEL, stt_on=STT_ON)


@app.get("/api/warm", response_class=EventSourceResponse)
async def warm(request: Request) -> AsyncIterator[ServerSentEvent]:
    async for ev in warm_stream(request.app.state.http):
        yield ev.to_sse()


@app.post("/api/chat", response_class=EventSourceResponse)
async def post_chat(req: ChatRequest, request: Request) -> AsyncIterator[ServerSentEvent]:
    client = request.app.state.http
    messages = [m.model_dump() for m in req.messages]
    if not chat.LLM_BASE:
        yield Error(message="LLM_URL is not set -- export it before starting web.py").to_sse()
        return
    last = None
    async for event in run_turn(client, messages):
        yield event.to_sse()
        if isinstance(event, Done):
            last = event
    if last:
        print(f"  turn: {last.n} clauses, {last.wall:.2f}s wall, "
              f"{last.total_audio:.2f}s audio", flush=True)


@app.post("/api/stt", response_model_exclude_none=True)
async def post_stt(request: Request) -> SttResponse:
    # Always 200 with {text} or {error} so the page handles both uniformly.
    if not STT_ON:
        return SttResponse(error="STT_URL is not set")
    audio = await request.body()
    ctype = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip()
    try:
        text = await transcribe(request.app.state.http, audio, ctype)
        print(f"  stt: {len(audio)} bytes -> {text[:60]!r}", flush=True)
        return SttResponse(text=text)
    except Exception as e:
        return SttResponse(error=f"{type(e).__name__}: {e}")


def main():
    import uvicorn

    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    print(f"STT = {STT_BASE + '  model=' + STT_MODEL if STT_ON else 'OFF (mic disabled; export STT_URL)'}")
    print(f"LLM = {chat.LLM_BASE or '(unset -- export LLM_URL)'}  model={chat.LLM_MODEL}")
    print(f"TTS = {say.BASE + '  model=' + say.MODEL if TTS_ON else 'OFF (text-only; export TTS_URL)'}")
    print(f"\n  open  ->  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
