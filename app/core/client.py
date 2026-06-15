import asyncio
import time
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings

RETRY_STATUS = {502, 503, 504}
# 503 = Flash stage cold-starting: retry patiently up to cold_start_budget.
# 502/504 = a live container erred: a few quick retries, then give up.
TRANSIENT_STATUS = {502, 504}
COLD_START_DELAY = 2.0
TRANSIENT_ATTEMPTS = 3
TRANSIENT_DELAY = 1.5


def _error(response: httpx.Response) -> str:
    return f"{response.status_code} {response.text[:160]}"


class _Retry:
    """Per-request retry policy: patient on cold-start 503s, brief on 502/504."""

    def __init__(self) -> None:
        self._deadline = time.monotonic() + settings.cold_start_budget
        self._transient_left = TRANSIENT_ATTEMPTS

    def next_delay(self, status_code: int) -> float | None:
        """Seconds to wait before the next try, or None to stop retrying."""
        if status_code == 503:
            return COLD_START_DELAY if time.monotonic() < self._deadline else None
        if status_code in TRANSIENT_STATUS and self._transient_left > 0:
            self._transient_left -= 1
            return TRANSIENT_DELAY
        return None


async def post(http: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    retry = _Retry()
    last_error = ""
    while True:
        response = await http.post(url, follow_redirects=True, **kwargs)
        if response.status_code not in RETRY_STATUS:
            response.raise_for_status()
            return response
        last_error = _error(response)
        delay = retry.next_delay(response.status_code)
        if delay is None:
            raise RuntimeError(f"Service not ready: {last_error}")
        await asyncio.sleep(delay)


async def stream_lines(http: httpx.AsyncClient, url: str, **kwargs) -> AsyncIterator[str]:
    retry = _Retry()
    last_error = ""
    while True:
        async with http.stream("POST", url, follow_redirects=True, **kwargs) as response:
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line
                return
            await response.aread()
            last_error = _error(response)
        delay = retry.next_delay(response.status_code)
        if delay is None:
            raise RuntimeError(f"Service not ready: {last_error}")
        await asyncio.sleep(delay)
