from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.router import router


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
