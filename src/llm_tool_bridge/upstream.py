from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, Omit, Timeout

from .config import Settings
from .errors import BridgeError
from .parsing import decode_json
from .protocol import UpstreamResult

ClientFactory = Callable[[dict[str, str]], AsyncOpenAI]


def _merge_chunk(document: dict[str, Any], chunk: dict[str, Any]) -> None:
    for key, value in chunk.items():
        if key not in {"choices", "usage", "object"}:
            document[key] = value
    if chunk.get("usage") is not None:
        document["usage"] = chunk["usage"]
    for choice in chunk.get("choices", []):
        if choice.get("index", 0) != 0:
            raise BridgeError(
                502, "invalid_upstream_response", "Multiple upstream choices unsupported"
            )
        target = document["choices"][0]
        message = target["message"]
        delta = choice.get("delta", choice.get("message", {}))
        if delta.get("content"):
            if not isinstance(delta["content"], str):
                raise BridgeError(
                    502, "invalid_upstream_response", "Expected textual upstream output"
                )
            message["content"] = (message.get("content") or "") + delta["content"]
        if delta.get("refusal"):
            message["refusal"] = (message.get("refusal") or "") + delta["refusal"]
        for call in delta.get("tool_calls", []):
            index = call.get("index", 0)
            if not isinstance(index, int) or index < 0 or index >= 256:
                raise BridgeError(502, "invalid_upstream_response", "Invalid upstream tool index")
            calls = message.setdefault("tool_calls", [])
            while len(calls) <= index:
                calls.append(
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
            item = calls[index]
            if call.get("id"):
                item["id"] = call["id"]
            for key in ("name", "arguments"):
                if call.get("function", {}).get(key):
                    item["function"][key] += call["function"][key]
        if choice.get("finish_reason"):
            target["finish_reason"] = choice["finish_reason"]


def decode_sse(raw: bytes) -> dict[str, Any]:
    document: dict[str, Any] = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": None}
        ]
    }
    try:
        source = raw.decode("utf-8").replace("\r\n", "\n")
        done = False
        for event in source.split("\n\n"):
            data = "\n".join(
                line[5:].lstrip(" ") for line in event.splitlines() if line.startswith("data:")
            )
            if not data:
                continue
            if data.strip() == "[DONE]":
                done = True
                break
            chunk = decode_json(data, broad=False)
            if "error" in chunk:
                raise BridgeError(502, "upstream_stream_error", "Upstream returned an SSE error")
            _merge_chunk(document, chunk)
        if not done or not document["choices"][0]["finish_reason"]:
            raise BridgeError(
                502, "upstream_stream_incomplete", "Upstream stream ended before completion"
            )
        return document
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise BridgeError(
            502, "invalid_upstream_response", "Malformed upstream event stream"
        ) from exc


class Upstream:
    def __init__(self, settings: Settings, factory: ClientFactory | None = None) -> None:
        self.settings = settings
        self.factory = factory

    @asynccontextmanager
    async def client(self, headers: dict[str, str]) -> AsyncIterator[AsyncOpenAI]:
        # A logical request owns its SDK client and cookie jar. Credentials, session cookies,
        # and auth defaults cannot leak across chats. Corrections use a separate client.
        client = (
            self.factory(headers)
            if self.factory
            else AsyncOpenAI(
                api_key="not-used",
                base_url=self.settings.upstream_base_url,
                max_retries=0,
                organization="",
                project="",
                http_client=DefaultAsyncHttpxClient(follow_redirects=False),
                timeout=Timeout(
                    self.settings.limits.request_timeout_seconds,
                    connect=self.settings.limits.connect_timeout_seconds,
                ),
            )
        )
        async with client:
            yield client

    async def generate(
        self,
        client: AsyncOpenAI,
        body: dict[str, Any],
        headers: dict[str, str],
        *,
        correction: bool = False,
    ) -> UpstreamResult:
        allowed = {key.lower() for key in self.settings.forward_headers}
        blocked = {
            "host",
            "content-length",
            "content-type",
            "connection",
            "transfer-encoding",
            "cookie",
            "set-cookie",
            "accept-encoding",
            "proxy-authorization",
            "authorization",
            "keep-alive",
            "te",
            "trailer",
            "upgrade",
            "proxy-authenticate",
            "proxy-connection",
        }
        blocked.update(name.strip().lower() for name in headers.get("connection", "").split(","))
        blocked.update(
            {
                self.settings.cloud_context.session_id_header.lower()
                if self.settings.cloud_context.session_id_header
                else ""
            }
        )
        forwarded: dict[str, Any] = {
            key: value
            for key, value in headers.items()
            if key.lower() in allowed and key.lower() not in blocked
        }
        # Override the SDK's placeholder bearer header, or explicitly omit it for API-key auth.
        forwarded["Authorization"] = headers.get("authorization", Omit())
        if correction:
            forwarded.pop("x-request-id", None)
        extra = {
            key: value for key, value in body.items() if key not in {"model", "messages", "stream"}
        }
        async with client.chat.completions.with_streaming_response.create(
            model=body["model"],
            messages=body["messages"],
            stream=body.get("stream", False),
            extra_body=extra,
            extra_headers=forwarded,
        ) as response:
            data = bytearray()
            try:
                async for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > self.settings.limits.max_response_bytes:
                        raise BridgeError(
                            502, "response_too_large", "Upstream response exceeded byte budget"
                        )
            except BridgeError:
                raise
            except Exception as exc:
                raise BridgeError(
                    502, "upstream_stream_error", "Upstream response stream failed"
                ) from exc
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            try:
                if "text/event-stream" in response_headers.get("content-type", ""):
                    document = decode_sse(bytes(data))
                else:
                    document = decode_json(data.decode("utf-8"), broad=False)
                if not isinstance(document, dict):
                    raise ValueError("Expected object")
            except (ValueError, RecursionError) as exc:
                raise BridgeError(
                    502, "invalid_upstream_response", "Invalid upstream JSON"
                ) from exc
            return UpstreamResult(document, response_headers)
