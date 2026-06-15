import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.router import router
from app.core.config import settings


def validate_startup_settings() -> None:
    if not settings.llm_url:
        raise RuntimeError("LLM_URL is not set -- export it before starting the web app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_startup_settings()
    async with httpx.AsyncClient(timeout=600, follow_redirects=False) as client:
        app.state.http = client
        yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    try:
        validate_startup_settings()
    except RuntimeError as e:
        raise SystemExit(str(e)) from e

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(
        "STT = "
        + (
            settings.stt_url + "  model=" + settings.stt_model
            if settings.stt_on
            else "OFF (mic disabled; export STT_URL)"
        )
    )
    print(f"LLM = {settings.llm_url or '(unset -- export LLM_URL)'}  model={settings.llm_model}")
    print(
        "TTS = "
        + (
            settings.tts_url + "  model=" + settings.tts_model
            if settings.tts_on
            else "OFF (text-only; export TTS_URL)"
        )
    )
    print(f"\n  open  ->  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
