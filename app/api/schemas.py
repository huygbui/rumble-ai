from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class MetaResponse(BaseModel):
    llm: str | None
    model: str
    tts: str | None
    tts_on: bool
    stt: str | None
    stt_model: str
    stt_on: bool


class WarmStage(BaseModel):
    name: str
    status: str
    t: float


class WarmResponse(BaseModel):
    ready: bool
    stages: list[WarmStage]


class TranscriptionResponse(BaseModel):
    text: str
