import json
from collections.abc import AsyncIterator

import httpx

from app.core import client
from app.core.config import settings


async def stream(http: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[str]:
    body = {
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
    async for line in client.stream_lines(http, settings.llm_chat_url, json=body):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or None
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if delta:
            yield delta
