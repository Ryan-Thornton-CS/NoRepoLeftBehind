import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from . import __version__
from .config import Settings
from .errors import BridgeError
from .parsing import decode_json
from .protocol import compact
from .service import Bridge
from .upstream import ClientFactory, Upstream


async def sse(response: dict[str, Any], include_usage: bool) -> AsyncIterator[str]:
    base = {key: response[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"

    def event(delta: dict[str, Any], finish: str | None = None) -> str:
        return (
            "data: "
            + compact({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            + "\n\n"
        )

    yield event({"role": "assistant"})
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    for offset in range(0, len(content), 1024):
        yield event({"content": content[offset : offset + 1024]})
    for index, call in enumerate(message.get("tool_calls", [])):
        yield event(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }
                ]
            }
        )
        arguments = call["function"]["arguments"]
        for offset in range(0, len(arguments), 1024):
            yield event(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "function": {"arguments": arguments[offset : offset + 1024]},
                        }
                    ]
                }
            )
    yield event({}, response["choices"][0]["finish_reason"])
    if include_usage and "usage" in response:
        yield "data: " + compact({**base, "choices": [], "usage": response["usage"]}) + "\n\n"
    if "bridge" in response:
        # Extension metadata stays outside delta/tool arguments, preserving standard clients.
        yield "data: " + compact({**base, "choices": [], "bridge": response["bridge"]}) + "\n\n"
    yield "data: [DONE]\n\n"


async def _watch_disconnect(request: Request) -> None:
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


def create_app(
    settings: Settings | None = None, client_factory: ClientFactory | None = None
) -> FastAPI:
    config = settings or Settings()
    bridge = Bridge(config, Upstream(config, client_factory))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        bridge.diagnostics.close()

    app = FastAPI(title="LLM Tool Bridge", version=__version__, lifespan=lifespan)
    app.state.bridge = bridge

    @app.exception_handler(BridgeError)
    async def bridge_error(request: Request, error: BridgeError) -> JSONResponse:
        bridge.diagnostics.record("error", status=error.status, code=error.code)
        return JSONResponse(status_code=error.status, content=error.payload())

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> Any:
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
