from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.schemas import ChatRequest, MetaResponse, SpeechRequest, TranscriptionResponse, WarmResponse
from app.core import pipeline, stt, tts
from app.core.config import settings

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
    # SSE has no status code to surface, so report not-ready as an error event.
    if not await pipeline.ensure_ready(client, settings.llm_url, settings.tts_url):
        yield ServerSentEvent(event="error", data={"message": "Service not ready"})
        return
    messages = [m.model_dump() for m in req.messages]
    async for event in pipeline.chat_events(client, messages):
        yield ServerSentEvent(event=event.event, data=event.data)


@router.post("/api/stt", response_model=TranscriptionResponse)
async def post_stt(request: Request) -> dict:
    audio = await request.body()
    content_type = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip()
    http = request.app.state.http
    if not await pipeline.ensure_ready(http, settings.stt_url):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="STT not ready")
    try:
        return {"text": await stt.transcribe(http, audio, content_type)}
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e


@router.post("/api/tts")
async def post_tts(req: SpeechRequest, request: Request) -> Response:
    http = request.app.state.http
    if not await pipeline.ensure_ready(http, settings.tts_url):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTS not ready")
    try:
        wav = await tts.synthesize(http, req.text)
        return Response(wav, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
