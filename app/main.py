import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.router import router
from app.core import dialogue, pipeline, speech


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(
        "STT = "
        + (
            pipeline.STT_BASE + "  model=" + pipeline.STT_MODEL
            if pipeline.STT_ON
            else "OFF (mic disabled; export STT_URL)"
        )
    )
    print(f"LLM = {dialogue.LLM_BASE or '(unset -- export LLM_URL)'}  model={dialogue.LLM_MODEL}")
    print(
        "TTS = "
        + (
            speech.BASE + "  model=" + speech.MODEL
            if pipeline.TTS_ON
            else "OFF (text-only; export TTS_URL)"
        )
    )
    print(f"\n  open  ->  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
