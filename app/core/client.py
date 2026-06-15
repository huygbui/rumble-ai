import asyncio
from collections.abc import AsyncIterator

import httpx

ATTEMPTS = 4
RETRY_STATUS = {502, 503, 504}
RETRY_DELAY = 1.5


def _error(response: httpx.Response) -> str:
    return f"{response.status_code} {response.text[:160]}"


async def post(http: httpx.AsyncClient, stage: str, url: str, **kwargs) -> httpx.Response:
    last_error = ""
    for attempt in range(ATTEMPTS):
        response = await http.post(url, follow_redirects=True, **kwargs)
        if response.status_code not in RETRY_STATUS:
            response.raise_for_status()
            return response
        last_error = _error(response)
        if attempt < ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY)
    raise RuntimeError(f"{stage} not ready after {ATTEMPTS} tries: {last_error}")


async def stream_lines(http: httpx.AsyncClient, stage: str, url: str, **kwargs) -> AsyncIterator[str]:
    last_error = ""
    for attempt in range(ATTEMPTS):
        async with http.stream("POST", url, follow_redirects=True, **kwargs) as response:
            if response.status_code in RETRY_STATUS:
                await response.aread()
                last_error = _error(response)
            else:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line
                return
        if attempt < ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY)
    raise RuntimeError(f"{stage} not ready after {ATTEMPTS} tries: {last_error}")
