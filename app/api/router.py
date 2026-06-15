from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.schemas import ChatRequest, MetaResponse, TranscriptionResponse
from app.core import dialogue, pipeline

INDEX = Path(__file__).resolve().parents[2] / "web" / "index.html"

router = APIRouter()


def _sse(event: pipeline.StreamEvent) -> ServerSentEvent:
    return ServerSentEvent(event=event.event, data=event.data)


@router.get("/")
def index():
    return FileResponse(INDEX)


@router.get("/api/meta", response_model=MetaResponse)
def meta() -> dict:
    return pipeline.meta_payload()


@router.get("/api/warm", response_class=EventSourceResponse)
async def warm(request: Request) -> AsyncIterator[ServerSentEvent]:
    async for event in pipeline.warm_stream(request.app.state.http):
        yield _sse(event)


@router.post("/api/chat", response_class=EventSourceResponse)
async def post_chat(req: ChatRequest, request: Request) -> AsyncIterator[ServerSentEvent]:
    client = request.app.state.http
    messages = [m.model_dump() for m in req.messages]
    if not dialogue.LLM_URL:
        yield ServerSentEvent(
            event="error",
            data={"message": "LLM_URL is not set -- export it before starting the web app"},
        )
        return
    last = None
    async for event in pipeline.run_turn(client, messages):
        yield _sse(event)
        if event.event == "done":
            last = event.data
    if last:
        print(
            f"  turn: {last['n']} clauses, {last['wall']:.2f}s wall, "
            f"{last['total_audio']:.2f}s audio",
            flush=True,
        )


@router.post("/api/stt", response_model=TranscriptionResponse)
async def post_stt(request: Request) -> dict:
    if not pipeline.STT_ON:
        return {"error": "STT_URL is not set"}
    audio = await request.body()
    ctype = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip()
    try:
        text = await pipeline.transcribe(request.app.state.http, audio, ctype)
        print(f"  stt: {len(audio)} bytes -> {text[:60]!r}", flush=True)
        return {"text": text}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
