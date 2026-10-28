"""Run offline synthetic translation checks without network requests or tool execution."""

from typing import Any

from norepoleftbehind.configuration.models import Settings
from norepoleftbehind.protocol.chat import ToolPolicy
from norepoleftbehind.translation.parsing import parse_output
from norepoleftbehind.translation.transform import transform


def diagnostics(settings: Settings) -> dict[str, Any]:
    """Check both render modes and canonical parsing using static fixtures; perform no I/O
    actions.
    """
    request = {
        "model": "gemini-3.8-flash",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Read the synthetic fixture."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_fixture",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    policy = ToolPolicy.from_request(request)
    checks: dict[str, bool] = {}
    for mode in ("flatten", "messages"):
        profile = settings.model_copy(
            update={
                "history_mode": mode,
                "instructions": "inject",
                "tools_mode": "prompt_and_translate",
            }
        )
        outgoing = transform(request, profile, policy)
        checks[mode] = outgoing["messages"][-1]["role"] == "user" and "tools" not in outgoing
    parsed = parse_output(
        '{"content":null,"tool_calls":[{"name":"read_fixture","arguments":{"path":"fixture"}}]}',
        policy,
        settings.parsing,
    )
    checks["tool_translation"] = parsed.calls[0]["name"] == "read_fixture"
    return {
        "ok": all(checks.values()),
        "network_requests": 0,
        "tools_executed": 0,
        "checks": checks,
    }
