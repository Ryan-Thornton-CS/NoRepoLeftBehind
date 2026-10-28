"""Recognize supported tool envelopes and validate every call before returning any action."""

import json
import math
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from norepoleftbehind.configuration.models import Parsing
from norepoleftbehind.protocol.chat import Parsed, ToolPolicy
from norepoleftbehind.protocol.errors import ParseFailure


def _reject_constant(value: str) -> None:
    """Reject non-JSON numeric spellings such as NaN and Infinity."""
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    """Decode a numeric token only when its floating-point representation is finite."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite floating-point range")
    return number


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys instead of choosing between ambiguous argument values."""
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON keys are ambiguous")
        value[key] = item
    return value


def decode_json(text: str, broad: bool) -> Any:
    """Decode strict JSON or apply only the explicitly supported local syntax repairs.

    Broad mode removes trailing commas outside strings and closes complete,
    otherwise balanced wrappers. It never closes strings, invents values, or
    evaluates expressions. Invalid JSON raises ValueError or RecursionError.
    """

    def load(source: str) -> Any:
        """Decode one candidate while rejecting duplicate keys and non-finite numbers."""
        return json.loads(
            source,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )

    try:
        return load(text)
    except (ValueError, RecursionError):
        if not broad:
            raise
    # Remove trailing commas only outside strings, then close balanced, incomplete wrappers.
    output: list[str] = []
    stack: list[str] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack.pop() != char:
                raise ValueError("Mismatched JSON brackets")
        elif char == ",":
            following = index + 1
            while following < len(text) and text[following].isspace():
                following += 1
            if following < len(text) and text[following] in "}]":
                continue
        output.append(char)
    if in_string:
        raise ValueError("Unterminated JSON string cannot be repaired locally")
    return load("".join(output) + "".join(reversed(stack)))


def _unwrap(text: str, broad: bool) -> str:
    """Strip a whole-response JSON fence in broad mode; never scan prose for embedded calls."""
    text = text.strip()
    if broad:
        match = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
        if match:
            return match[1].strip()
    return text


def _call(value: Any, broad: bool) -> dict[str, Any]:
    """Normalize one supported call wrapper into an exact tool name and argument object."""
    if not isinstance(value, dict):
        raise ParseFailure("Tool calls must be objects")
    if broad and "functionCall" in value:
        if len(value) != 1:
            raise ParseFailure("Ambiguous functionCall wrapper")
        value = value["functionCall"]
    if broad and isinstance(value, dict) and "function" in value:
        if set(value) - {"id", "type", "index", "function"}:
            raise ParseFailure("Ambiguous function wrapper")
        value = value["function"]
    if not isinstance(value, dict) or set(value) - {"name", "arguments", "args"}:
        raise ParseFailure("Unsupported tool-call fields")
    if "args" in value and "arguments" in value:
        raise ParseFailure("Conflicting argument fields")
    name = value.get("name")
    arguments = value.get("arguments", value.get("args") if broad else None)
    if broad and isinstance(arguments, str):
        try:
            arguments = decode_json(arguments, broad=True)
        except (ValueError, RecursionError) as exc:
            raise ParseFailure("Invalid argument JSON") from exc
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ParseFailure("Tool calls require a name and an arguments object")
    return {"name": name, "arguments": arguments}


def validate_parsed(parsed: Parsed, policy: ToolPolicy, config: Parsing) -> Parsed:
    """Validate Unicode, calling restrictions, all tool names, and all argument schemas
    atomically.
    """
    try:
        if parsed.content is not None:
            parsed.content.encode("utf-8")
        for call in parsed.calls:
            json.dumps(call, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ParseFailure("Output contains invalid Unicode or non-finite numbers") from exc
    if len(parsed.calls) > config.max_tool_calls:
        raise ParseFailure("Too many tool calls")
    if parsed.calls and policy.choice == "none":
        raise ParseFailure("Tool calls are prohibited by tool_choice")
    if not parsed.calls and policy.choice in {"required", "named"}:
        raise ParseFailure("A tool call is required")
    if len(parsed.calls) > 1 and not policy.parallel:
        raise ParseFailure("Parallel tool calls are disabled")
    for call in parsed.calls:
        name = call["name"]
        if name not in policy.schemas or (policy.named and name != policy.named):
            raise ParseFailure("Tool is not allowed")
        schema = policy.schemas[name].get("parameters", {"type": "object"})
        try:
            Draft202012Validator(schema).validate(call["arguments"])
        except (ValidationError, RecursionError) as exc:
            raise ParseFailure("Tool arguments do not match their schema") from exc
        except Exception as exc:
            # Reference-resolution failures must never escape as executable calls.
            raise ParseFailure("Tool schema could not validate the response") from exc
    return parsed


def parse_output(text: str, policy: ToolPolicy, config: Parsing) -> Parsed:
    """Classify a whole response as an answer or a recognizable attempted tool call.

    Only complete supported envelopes are considered actionable; arbitrary
    prose and quoted examples are not searched for call-shaped substrings.
    An invalid or ambiguous attempt raises ParseFailure for model correction.
    """
    broad = config.tolerance == "broad"
    source = _unwrap(text, broad)
    # Never search arbitrary prose, code, or quoted examples for tool-shaped substrings.
    attempted = source.startswith(("{", "[")) and bool(
        re.search(r'"(?:tool_calls|functionCall|function|arguments|args)"\s*:', source)
    )
    try:
        value = decode_json(source, broad)
    except (ValueError, RecursionError) as exc:
        if attempted:
            raise ParseFailure("Malformed attempted tool call") from exc
        return validate_parsed(Parsed(text), policy, config)
    if isinstance(value, dict) and "tool_calls" in value:
        if set(value) - {"content", "tool_calls"} or (not broad and "content" not in value):
            raise ParseFailure("Invalid canonical output envelope")
        calls = value["tool_calls"]
        content = value.get("content")
        if not isinstance(calls, list) or (content is not None and not isinstance(content, str)):
            raise ParseFailure("Invalid output content or tool_calls")
        parsed = Parsed(content, [_call(call, broad) for call in calls])
    elif broad and attempted and isinstance(value, (dict, list)):
        calls = value if isinstance(value, list) else [value]
        parsed = Parsed(None, [_call(call, broad) for call in calls])
    elif attempted:
        raise ParseFailure("Output does not match the configured tool envelope")
    else:
        parsed = Parsed(text)
    return validate_parsed(parsed, policy, config)
