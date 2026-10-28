"""Serialize validated Chat Completions responses as buffered server-sent events."""

from collections.abc import AsyncIterator
from typing import Any

from norepoleftbehind.protocol.chat import compact


async def sse(response: dict[str, Any], include_usage: bool) -> AsyncIterator[str]:
    """Yield OpenAI SSE frames after the complete response has been validated.

    Args:
        response: A complete Chat Completions response from the runtime.
        include_usage: Whether to emit reported original-generation usage.

    Yields:
        Framed role, content, call, completion, optional metadata, and DONE events.
        Tool arguments are fragmented without executing or interpreting them.
    """
    base = {key: response[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"

    def event(delta: dict[str, Any], finish: str | None = None) -> str:
        """Encode one indexed choice delta and its optional finish reason."""
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
