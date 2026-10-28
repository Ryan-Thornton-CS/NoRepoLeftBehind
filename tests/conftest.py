import inspect
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI

from norepoleftbehind.api.app import create_app
from norepoleftbehind.configuration.models import Settings

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Request a file from the harness",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
]


def envelope(path: str = "README.md") -> str:
    return json.dumps(
        {"content": None, "tool_calls": [{"name": "read_file", "arguments": {"path": path}}]}
    )


def completion(content: str | None = "hello", **kwargs: Any) -> dict[str, Any]:
    return {
        "id": "upstream-response",
        "object": "chat.completion",
        "created": 1,
        "model": "gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        **kwargs,
    }


@dataclass
class Gateway:
    received: list[tuple[dict[str, Any], dict[str, str]]] = field(default_factory=list)
    responder: Any = None

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def respond(request: Request) -> Any:
            body = await request.json()
            headers = dict(request.headers)
            self.received.append((body, headers))
            result = self.responder(body, headers) if self.responder else completion(envelope())
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                return result
            if body.get("stream"):

                async def events() -> Any:
                    choice = result["choices"][0]
                    text = choice["message"].get("content") or ""
                    for start in range(0, len(text), 7):
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": result["id"],
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text[start : start + 7]},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            + "\n\n"
                        )
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                **{key: value for key, value in result.items() if key != "choices"},
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": choice["finish_reason"],
                                    }
                                ],
                            }
                        )
                        + "\n\n"
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(events(), media_type="text/event-stream")
            return JSONResponse(result)

        return app

    def factory(self, headers: dict[str, str]) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="unused",
            base_url="http://gateway/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app())),
        )


@pytest.fixture
def gateway() -> Gateway:
    return Gateway()


@pytest.fixture
def request_body() -> dict[str, Any]:
    return {
        "model": "gemini-3.8-flash",
        "messages": [
            {"role": "system", "content": "Use the file tool when needed."},
            {"role": "user", "content": "Read README.md"},
        ],
        "tools": deepcopy(TOOLS),
    }


@pytest.fixture
def api(gateway: Gateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://bridge",
        transport=httpx.ASGITransport(app=create_app(Settings(), gateway.factory)),
    )
