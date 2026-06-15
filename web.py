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
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- One conversational turn, as a stream of (event, data) SSE pairs -------------------
# A producer task streams the LLM into a clause queue; this generator pulls clauses and (in
# voice mode) synthesizes them. Synthesis runs while the producer keeps reading the LLM -- the
# same overlap as chat.converse(), so first audio tracks the FIRST clause, not the whole reply.
async def run_turn(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[tuple[str, dict]]:
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
                        await clauses.put(("ttft", {"t": time.time() - t0}))
                    for clause in streamer.feed(delta):
                        await clauses.put(("clause", clause))
            for clause in streamer.flush():
                await clauses.put(("clause", clause))
        except Exception as e:  # surface transport/LLM errors instead of hanging the stream
            await clauses.put(("error", {"message": f"{type(e).__name__}: {e}"}))
        await clauses.put(None)

    task = asyncio.create_task(produce())
    try:
        gen = _voice_turn(client, clauses, t0) if TTS_ON else _text_turn(clauses, t0)
        async for event in gen:
            yield event
    finally:
        task.cancel()  # clean up the producer if the client disconnects mid-turn


def _clause_event(i, text, t0, *, synth_s=None, audio_s=None, wav=None) -> dict:
    return {"i": i, "text": text, "t_ready": time.time() - t0,
            "synth_s": synth_s, "audio_s": audio_s,
            "wav_b64": base64.b64encode(wav).decode() if wav else None}


def _done_event(parts, total_audio, t0) -> dict:
    return {"wall": time.time() - t0, "total_audio": total_audio,
            "n": len(parts), "full_reply": " ".join(parts)}


# Text-only (no TTS_URL): stream clauses as text, no audio -- dev without the GPU.
async def _text_turn(clauses, t0) -> AsyncIterator[tuple[str, dict]]:
    parts: list[str] = []
    while (item := await clauses.get()) is not None:
        kind, data = item
        if kind == "clause":
            parts.append(data)
            yield "clause", _clause_event(len(parts) - 1, data, t0)
        else:
            yield kind, data  # ttft / error, straight through
    yield "done", _done_event(parts, 0.0, t0)


# Voice: synthesize each clause as it arrives and emit ready audio in order.
async def _voice_turn(client, clauses, t0) -> AsyncIterator[tuple[str, dict]]:
    parts, wavs, total_audio, i = [], [], 0.0, 0
    while (item := await clauses.get()) is not None:
        kind, clause = item
        if kind != "clause":
            yield kind, clause  # ttft / error
            continue
        try:
            synth_s, wav = await synth(client, clause)
        except Exception as e:
            yield "error", {"message": f"{type(e).__name__}: {e}"}
            continue
        dur = say.wav_dur(wav)
        total_audio += dur
        parts.append(clause)
        wavs.append((i, wav))
        yield "clause", _clause_event(i, clause, t0, synth_s=synth_s, audio_s=dur, wav=wav)
        i += 1
    await asyncio.to_thread(_save_stitched, wavs)
    yield "done", _done_event(parts, total_audio, t0)


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


async def warm_stream(client: httpx.AsyncClient) -> AsyncIterator[str]:
    stages = _stages()
    if not stages:
        yield sse("done", {"ready": False})
        return
    t0 = time.time()

    async def warm_up(name, base):
        while time.time() - t0 < WARM_BUDGET:
            if await _health_ok(client, base):
                return name, True, time.time() - t0
            await asyncio.sleep(2)  # cold start in progress -> re-probe, keep nudging
        return name, False, time.time() - t0

    yield sse("warming", {"stages": [n for n, _ in stages]})
    tasks = [asyncio.create_task(warm_up(n, b)) for n, b in stages]
    all_ok = True
    for done in asyncio.as_completed(tasks):
        name, ok, t = await done
        all_ok &= ok
        print(f"  warm: {name} {'ready' if ok else 'FAILED'} in {t:.1f}s", flush=True)
        yield sse("stage", {"name": name, "status": "ready" if ok else "failed", "t": t})
    yield sse("done", {"ready": all_ok})


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
class ChatRequest(BaseModel):
    class Message(BaseModel):
        role: str
        content: str

    messages: list[Message]


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
def meta():
    return {"llm": chat.LLM_BASE or None, "model": chat.LLM_MODEL,
            "tts": say.BASE or None, "tts_model": say.MODEL, "tts_on": TTS_ON,
            "stt": STT_BASE or None, "stt_model": STT_MODEL, "stt_on": STT_ON}


@app.get("/api/warm")
async def warm(request: Request):
    return StreamingResponse(warm_stream(request.app.state.http),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/chat")
async def post_chat(req: ChatRequest, request: Request):
    client = request.app.state.http
    messages = [m.model_dump() for m in req.messages]

    async def stream() -> AsyncIterator[str]:
        if not chat.LLM_BASE:
            yield sse("error", {"message": "LLM_URL is not set -- export it before starting web.py"})
            return
        last = None
        async for event, data in run_turn(client, messages):
            yield sse(event, data)
            if event == "done":
                last = data
        if last:
            print(f"  turn: {last['n']} clauses, {last['wall']:.2f}s wall, "
                  f"{last['total_audio']:.2f}s audio", flush=True)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/stt")
async def post_stt(request: Request):
    # Always 200 with {text} or {error} so the page handles both uniformly.
    if not STT_ON:
        return JSONResponse({"error": "STT_URL is not set"})
    audio = await request.body()
    ctype = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip()
    try:
        text = await transcribe(request.app.state.http, audio, ctype)
        print(f"  stt: {len(audio)} bytes -> {text[:60]!r}", flush=True)
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"})


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
