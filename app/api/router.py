from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.schemas import ChatRequest, MetaResponse, TranscriptionResponse, WarmResponse
from app.core import pipeline, stt

INDEX = Path(__file__).resolve().parents[2] / "web" / "index.html"

router = APIRouter()


@router.get("/")
def index():
    return FileResponse(INDEX)


@router.get("/api/meta", response_model=MetaResponse)
def meta() -> dict:
    return pipeline.meta_payload()


@router.post("/api/warm", response_model=WarmResponse)
async def warm(request: Request) -> dict:
    return await pipeline.warm(request.app.state.http)


@router.post("/api/chat", response_class=EventSourceResponse)
async def post_chat(req: ChatRequest, request: Request) -> AsyncIterator[ServerSentEvent]:
    client = request.app.state.http
    messages = [m.model_dump() for m in req.messages]
    async for event in pipeline.run_turn(client, messages):
        yield ServerSentEvent(event=event.event, data=event.data)


@router.post("/api/stt", response_model=TranscriptionResponse)
async def post_stt(request: Request) -> dict:
    audio = await request.body()
    content_type = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip()
    try:
        return {"text": await stt.transcribe(request.app.state.http, audio, content_type)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
