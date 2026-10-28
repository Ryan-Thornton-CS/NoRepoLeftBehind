"""Render supplied conversation history into the configured upstream request format."""

import math
from copy import deepcopy
from typing import Any

from norepoleftbehind.configuration.models import Settings
from norepoleftbehind.protocol.chat import ToolPolicy, compact, text_content
from norepoleftbehind.protocol.errors import BridgeError
from norepoleftbehind.protocol.pointers import remove
from norepoleftbehind.translation.prompts import tool_prompt


def budget(body: dict[str, Any], settings: Settings) -> None:
    """Enforce transformed prompt bytes and an optional explicitly approximate token preflight."""
    prompt_bytes = len(
        compact(
            {
                key: value
                for key, value in body.items()
                if key in {"messages", "tools", "response_format"}
            }
        ).encode()
    )
    limits = settings.limits
    estimated = math.ceil(
        prompt_bytes / limits.estimated_bytes_per_token * limits.token_safety_margin
    )
    if prompt_bytes > limits.max_prompt_bytes or (
        limits.estimated_input_token_limit and estimated > limits.estimated_input_token_limit
    ):
        raise BridgeError(
            400,
            "context_budget_exceeded",
            f"Transformed prompt is {prompt_bytes} bytes (estimated {estimated} input tokens, "
            "not a provider count), including schemas. Let the harness compact it.",
        )


def transform(
    original: dict[str, Any],
    settings: Settings,
    policy: ToolPolicy,
    delta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy a validated request and render history, instructions, and tools for the provider.

    Args:
        original: Full request supplied by the harness; never modified in place.
        settings: Effective profile controlling injection and preservation.
        policy: Current tool definitions and calling restrictions.
        delta: Optional unseen turns for a verified provider continuation.

    Returns:
        A budget-checked upstream payload with bridge-only fields removed.
        Full instructions and tools can be reinjected even when history is a delta.
    """
    body = deepcopy(original)
    remove(body, settings.cloud_context.request_id_pointer)
    remove(body, settings.cloud_context.session_id_body_pointer)
    for key in settings.drop_body_fields:
        body.pop(key, None)
    if not all(key in body for key in ("model", "messages")):
        raise BridgeError(400, "invalid_config", "drop_body_fields cannot remove model or messages")
    if settings.tools_mode != "preserve":
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            body.pop(key, None)
    if (
        settings.tools_mode == "prompt_and_translate"
        and policy.schemas
        and body.get("response_format")
    ):
        raise BridgeError(
            400, "unsupported_parameter", "response_format conflicts with tool translation"
        )
    all_messages = original["messages"]
    selected = delta if delta is not None else all_messages
    instructions = [
        {"role": message["role"], "content": text_content(message.get("content"))}
        for message in all_messages
        if message["role"] in {"system", "developer"}
    ]
    rendered: list[dict[str, Any]] = (
        deepcopy(instructions) if settings.instructions == "preserve" else []
    )
    names = {
        call["id"]: call["function"]["name"]
        for message in all_messages
        for call in message.get("tool_calls") or []
    }
    for message in selected:
        role = message["role"]
        if role in {"system", "developer"}:
            continue
        if settings.tools_mode == "preserve":
            rendered.append(deepcopy(message))
            continue
        record = {"role": role, "content": text_content(message.get("content"))}
        if message.get("tool_calls"):
            record["tool_calls"] = message["tool_calls"]
        if role == "tool":
            record["tool_call_id"] = message["tool_call_id"]
            record["name"] = names.get(message["tool_call_id"], message.get("name", "unknown"))
        rendered.append(record)
    blocks = []
    if settings.tools_mode == "prompt_and_translate" and policy.schemas:
        blocks.append(tool_prompt(policy, original["model"], settings))
    if settings.skill_prompt:
        blocks.append("AVAILABLE_SKILLS_GUIDANCE\n" + settings.skill_prompt)
    if settings.instructions == "inject" and instructions:
        blocks.append("SYSTEM_AND_DEVELOPER_INSTRUCTIONS\n" + compact(instructions))
    if settings.history_mode == "flatten":
        blocks.append("CONVERSATION_DATA_IN_CHRONOLOGICAL_ORDER\n" + compact(rendered))
        blocks.append("CURRENT_CONTINUATION\nRespond to the latest request or tool results above.")
        body["messages"] = [{"role": "user", "content": "\n\n".join(blocks)}]
    else:
        messages: list[dict[str, Any]] = []
        for record in rendered:
            if settings.tools_mode == "preserve" or record["role"] in {"system", "developer"}:
                messages.append(record)
            elif record["role"] == "tool":
                messages.append({"role": "user", "content": "TOOL_RESULT_DATA\n" + compact(record)})
            elif "tool_calls" in record:
                messages.append(
                    {"role": "assistant", "content": "ASSISTANT_CALL_DATA\n" + compact(record)}
                )
            else:
                messages.append(record)
        # Put injection after the most recent result, never in an earlier user message.
        if not messages or selected[-1]["role"] != "user":
            messages.append({"role": "user", "content": "Continue from the preceding results."})
        if blocks:
            messages[-1]["content"] = "\n\n".join(
                blocks + ["CURRENT_USER_MESSAGE_DATA\n" + compact(messages[-1]["content"])]
            )
        body["messages"] = messages
    if "max_tokens" not in body and "max_completion_tokens" not in body:
        body["max_tokens"] = settings.limits.max_output_tokens
    budget(body, settings)
    return body
