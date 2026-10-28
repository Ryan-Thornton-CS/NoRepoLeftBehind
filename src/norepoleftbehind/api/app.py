"""Expose the local HTTP API and cancel upstream work when a client disconnects."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from norepoleftbehind import __version__
from norepoleftbehind.api.upstream import ClientFactory, Upstream
from norepoleftbehind.configuration.models import Settings
from norepoleftbehind.protocol.chat import compact
from norepoleftbehind.protocol.errors import BridgeError
from norepoleftbehind.runtime.service import Bridge
from norepoleftbehind.translation.parsing import decode_json

from .streaming import sse


async def _watch_disconnect(request: Request) -> None:
    """Wait for an ASGI disconnect after the request body has been consumed."""
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


def create_app(
    settings: Settings | None = None, client_factory: ClientFactory | None = None
) -> FastAPI:
    """Construct the local HTTP service without contacting the provider.

    Args:
        settings: Validated configuration; defaults are used when omitted.
        client_factory: Optional SDK-client factory for tests or embedding.

    Returns:
        A FastAPI app exposing /healthz and /v1/chat/completions. Its
        state.bridge attribute owns request orchestration and session state.
    """
    config = settings or Settings()
    bridge = Bridge(config, Upstream(config, client_factory))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Close diagnostic handlers when the application shuts down."""
        yield
        bridge.diagnostics.close()

    app = FastAPI(title="NoRepoLeftBehind", version=__version__, lifespan=lifespan)
    app.state.bridge = bridge

    @app.exception_handler(BridgeError)
    async def bridge_error(request: Request, error: BridgeError) -> JSONResponse:
        """Translate a sanitized bridge failure into the OpenAI error envelope."""
        bridge.diagnostics.record("error", status=error.status, code=error.code)
        return JSONResponse(status_code=error.status, content=error.payload())

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        """Report process health without checking provider availability."""
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> Any:
        """Read a bounded request, monitor disconnects, and return validated JSON or SSE."""
        raw = bytearray()
        try:
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > config.limits.max_request_bytes:
                    raise BridgeError(413, "request_too_large", "Request exceeded byte budget")
        except ClientDisconnect as exc:
            raise BridgeError(499, "client_disconnected", "Client disconnected") from exc
        try:
            body = decode_json(raw.decode("utf-8"), broad=False)
            compact(body).encode("utf-8")
        except (ValueError, RecursionError) as exc:
            raise BridgeError(400, "invalid_json", "Request body must be valid JSON") from exc
        work = asyncio.create_task(bridge.handle(body, dict(request.headers)))
        disconnect = asyncio.create_task(_watch_disconnect(request))
        try:
            done, _ = await asyncio.wait({work, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            if work not in done:
                raise BridgeError(499, "client_disconnected", "Client disconnected")
            response = await work
        finally:
            for task in (work, disconnect):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    if task is disconnect or not task.done():
                        await task
        if isinstance(body, dict) and body.get("stream"):
            options = body.get("stream_options") or {}
            return StreamingResponse(
                sse(response, bool(options.get("include_usage"))),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(response)

    return app
