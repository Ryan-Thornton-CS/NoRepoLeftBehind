"""Provide the installed serve, validate, and offline diagnostics commands."""

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from norepoleftbehind import __version__
from norepoleftbehind.api.app import create_app
from norepoleftbehind.configuration import load_settings

from .runtime.checks import diagnostics


def main(argv: list[str] | None = None) -> int:
    """Run a command using YAML, environment, and explicit command-line settings.

    Args:
        argv: Arguments without the executable name; None uses sys.argv.

    Returns:
        Zero on success, one for failed diagnostics, or two for invalid settings.
        argparse handles help and argument errors using its normal exit behavior.
    """
    parser = argparse.ArgumentParser(
        description="NoRepoLeftBehind: OpenAI-compatible tool prompt and translation middleware"
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
