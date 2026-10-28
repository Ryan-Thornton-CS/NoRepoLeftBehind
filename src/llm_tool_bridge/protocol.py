import json
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import BridgeError


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for part in value:
            if (
                not isinstance(part, dict)
                or part.get("type") != "text"
                or not isinstance(part.get("text"), str)
            ):
                raise BridgeError(
                    400, "unsupported_content", "Only text content parts are supported"
                )
            chunks.append(part["text"])
        return "".join(chunks)
    raise BridgeError(400, "unsupported_content", "Content must be text, text parts, or null")


@dataclass
class ToolPolicy:
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    choice: str = "auto"
    named: str | None = None
    parallel: bool = True

    @classmethod
    def from_request(cls, body: dict[str, Any]) -> "ToolPolicy":
        schemas: dict[str, dict[str, Any]] = {}
        tools = body.get("tools") or []
        if not isinstance(tools, list):
            raise BridgeError(400, "invalid_tools", "tools must be an array")
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise BridgeError(400, "unsupported_tool", "Only function tools are supported")
            function = tool.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise BridgeError(400, "invalid_tools", "Each function requires a name")
            name = function["name"]
            if not name or len(name) > 128 or name in schemas:
                raise BridgeError(400, "invalid_tools", "Tool names must be nonempty and unique")
            schema = function.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(schema, dict):
                raise BridgeError(
                    400, "invalid_schema", "Function parameters must be a JSON schema"
                )
            # No network or filesystem schema resolution. Local recursive schemas are supported.
            cls._check_refs(schema)
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise BridgeError(
                    400, "invalid_schema", "Invalid function parameter schema"
                ) from exc
            schemas[name] = function
        selection = body.get("tool_choice", "auto" if schemas else "none")
        named = None
        if isinstance(selection, dict):
            function_choice = selection.get("function")
            named = function_choice.get("name") if isinstance(function_choice, dict) else None
            if (
                selection.get("type") != "function"
                or not isinstance(named, str)
                or named not in schemas
            ):
                raise BridgeError(400, "invalid_tool_choice", "Selected tool must be declared")
            choice = "named"
        elif selection in ("auto", "none", "required"):
            choice = selection
        else:
            raise BridgeError(400, "invalid_tool_choice", "Unsupported tool_choice")
        if choice == "required" and not schemas:
            raise BridgeError(400, "invalid_tool_choice", "Required tool choice needs tools")
        parallel = body.get("parallel_tool_calls", True)
        if not isinstance(parallel, bool):
            raise BridgeError(400, "invalid_tool_choice", "parallel_tool_calls must be boolean")
        return cls(schemas, choice, named, parallel)

    @staticmethod
    def _check_refs(node: Any, depth: int = 0) -> None:
        if depth > 64:
            raise BridgeError(400, "invalid_schema", "Schema nesting exceeds 64 levels")
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise BridgeError(
                        400, "external_schema_ref", "Only local schema references are allowed"
                    )
                ToolPolicy._check_refs(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                ToolPolicy._check_refs(value, depth + 1)

    def constraints(self) -> dict[str, Any]:
        return {"tool_choice": self.named or self.choice, "parallel_tool_calls": self.parallel}


@dataclass
class Parsed:
    content: str | None
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UpstreamResult:
    document: dict[str, Any]
    headers: dict[str, str]


def validate_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise BridgeError(400, "invalid_request", "Request body must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise BridgeError(400, "invalid_request", "model is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BridgeError(400, "invalid_request", "messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise BridgeError(400, "invalid_message", "Unsupported message role")
        text_content(message.get("content"))
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise BridgeError(400, "invalid_message", "tool_calls must be an array")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise BridgeError(400, "invalid_message", "Malformed historical tool call")
            if (
                not isinstance(call.get("id"), str)
                or not isinstance(call["function"].get("name"), str)
                or not isinstance(call["function"].get("arguments"), str)
            ):
                raise BridgeError(
                    400, "invalid_message", "Historical calls require id, name, and arguments"
                )
        if message["role"] == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise BridgeError(400, "invalid_message", "Tool results require tool_call_id")
        if any(key in message for key in ("audio", "function_call")):
            raise BridgeError(
                400, "unsupported_content", "Legacy function calls and audio are unsupported"
            )
    if body.get("n", 1) != 1:
        raise BridgeError(400, "unsupported_parameter", "Only n=1 is supported")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise BridgeError(400, "invalid_request", "stream must be boolean")
    if body.get("stream_options") is not None and not isinstance(body["stream_options"], dict):
        raise BridgeError(400, "invalid_request", "stream_options must be an object")
    if body.get("modalities", ["text"]) != ["text"] or body.get("audio"):
        raise BridgeError(400, "unsupported_content", "Only text output is supported")
    if "functions" in body or "function_call" in body:
        raise BridgeError(
            400, "unsupported_parameter", "Use tools and tool_choice, not legacy functions"
        )
    return body
