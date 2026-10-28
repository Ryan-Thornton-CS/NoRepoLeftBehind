import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn
from pydantic import ValidationError

from . import __version__
from .app import create_app
from .config import Settings, load_settings
from .parsing import parse_output
from .protocol import ToolPolicy
from .transform import transform


def diagnostics(settings: Settings) -> dict[str, Any]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI tool prompt and translation bridge; never executes tools"
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("serve", "validate", "diagnostics"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path)
        sub.add_argument("--profile")
        sub.add_argument("--upstream-base-url")
        sub.add_argument("--host")
        sub.add_argument("--port", type=int)
        sub.add_argument("--max-concurrency", type=int)
    args = parser.parse_args(argv)
    overrides = {
        key: getattr(args, key)
        for key in ("upstream_base_url", "host", "port")
        if getattr(args, key) is not None
    }
    if args.max_concurrency is not None:
        overrides["limits"] = {"max_concurrency": args.max_concurrency}
    try:
        settings = load_settings(args.config, args.profile, overrides)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "profile": args.profile,
                        "cloud_context": settings.cloud_context.enabled,
                    }
                )
            )
        elif args.command == "diagnostics":
            result = diagnostics(settings)
            print(json.dumps(result))
            return 0 if result["ok"] else 1
        else:
            if ".invalid" in settings.upstream_base_url:
                parser.error(
                    "Configure --upstream-base-url or upstream_base_url in YAML before serving"
                )
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            for name in ("openai", "httpx", "httpcore"):
                logging.getLogger(name).setLevel(logging.WARNING)
            uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    except ValidationError as exc:
        # Do not echo input_value: configuration may contain credential-bearing values.
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": [
                        {"location": list(error["loc"]), "type": error["type"]}
                        for error in exc.errors(include_input=False, include_context=False)
                    ],
                }
            ),
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
