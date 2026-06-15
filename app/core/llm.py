import json

from app.core.config import settings


def make_chat_payload(messages: list[dict]) -> dict:
    return {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": settings.chat_max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }


def parse_sse_delta(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)["choices"][0].get("delta", {}).get("content") or None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
