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
    tts_model: str
    tts_on: bool
    stt: str | None
    stt_model: str
    stt_on: bool


class TranscriptionResponse(BaseModel):
    text: str
