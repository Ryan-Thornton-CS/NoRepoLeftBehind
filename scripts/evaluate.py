"""Opt-in live model evaluation via an already running middleware endpoint.

Acts as a synthetic harness, supplying predefined tool results; never executes a tool.
Credentials are read from NOREPOLEFTBEHIND_EVAL_TOKEN, not command-line arguments.
"""

import argparse
import asyncio
import json
import os
import time
from typing import Any

from openai import AsyncOpenAI

from norepoleftbehind.protocol.chat import compact

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_fixture",
            "description": "Read a named synthetic fixture through the harness.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "enum": ["alpha", "beta"]}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
]


async def evaluate(base_url: str, models: list[str], repeats: int) -> dict[str, Any]:
    token = os.environ.get("NOREPOLEFTBEHIND_EVAL_TOKEN")
    if not token:
        raise SystemExit("Set NOREPOLEFTBEHIND_EVAL_TOKEN; this command makes live model requests")
    reports: dict[str, Any] = {}
    for model in models:
        counters = {
            "requests": 0,
            "valid_responses": 0,
            "first_pass_valid": 0,
            "corrected_successes": 0,
            "tool_checks": 0,
            "correct_tool_checks": 0,
            "false_tool_detections": 0,
            "completed_loops": 0,
            "errors": 0,
        }
        started = time.perf_counter()
        async with AsyncOpenAI(api_key=token, base_url=base_url, max_retries=0) as client:
            for _ in range(repeats):
                for kind in ("answer", "single", "multiple"):
                    prompt = {
                        "answer": "Say hello. Do not request any tool.",
                        "single": "Read fixture alpha using read_fixture, then quote its result.",
                        "multiple": (
                            "Read independent fixtures alpha and beta in one batch, "
                            "then quote both results."
                        ),
                    }[kind]
                    history: list[Any] = [{"role": "user", "content": prompt}]
                    expected = (
                        set()
                        if kind == "answer"
                        else {"alpha", "beta"}
                        if kind == "multiple"
                        else {"alpha"}
                    )
                    seen: set[str] = set()
                    for _turn in range(4):
                        counters["requests"] += 1
                        try:
                            response = await client.chat.completions.create(
                                model=model,
                                messages=history,
                                tools=TOOLS,
                                parallel_tool_calls=True,
                                tool_choice="auto",
                            )
                            counters["valid_responses"] += 1
                            metadata = (
                                response.model_extra.get("bridge", {})
                                if response.model_extra
                                else {}
                            )
                            if metadata.get("correction_attempts", 0):
                                counters["corrected_successes"] += 1
                            else:
                                counters["first_pass_valid"] += 1
                            message = response.choices[0].message
                            if not message.tool_calls:
                                # Completion must quote the predefined results, not just stop.
                                if seen == expected and all(
                                    f"FIXTURE_{name.upper()}" in (message.content or "")
                                    for name in expected
                                ):
                                    counters["completed_loops"] += 1
                                break
                            if kind == "answer":
                                counters["false_tool_detections"] += len(message.tool_calls)
                            history.append(message.model_dump(exclude_none=True))
                            for call in message.tool_calls:
                                counters["tool_checks"] += 1
                                args = json.loads(call.function.arguments)
                                path = args.get("path")
                                correct = call.function.name == "read_fixture" and path in expected
                                if correct:
                                    counters["correct_tool_checks"] += 1
                                    seen.add(path)
                                # Static fixture lookup only: no tool dispatcher, I/O, or execution.
                                fixture = {"alpha": "FIXTURE_ALPHA", "beta": "FIXTURE_BETA"}.get(
                                    path, "INVALID_FIXTURE"
                                )
                                history.append(
                                    {"role": "tool", "tool_call_id": call.id, "content": fixture}
                                )
                        except Exception as exc:
                            counters["errors"] += 1
                            # SDK exception strings may contain sensitive content.
                            counters["last_error_type"] = type(exc).__name__
                            break
                    else:
                        counters["errors"] += 1
        reports[model] = {
            **counters,
            "first_pass_valid_rate": counters["first_pass_valid"] / max(counters["requests"], 1),
            "corrected_success_rate": counters["corrected_successes"]
            / max(counters["requests"], 1),
            "tool_correctness_rate": counters["correct_tool_checks"]
            / max(counters["tool_checks"], 1),
            "loop_completion_rate": counters["completed_loops"] / (3 * repeats),
            "seconds": round(time.perf_counter() - started, 2),
        }
    return {
        "models": reports,
        "tools_executed": 0,
        "fixture_results_supplied_by_evaluation_harness": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787/v1")
    parser.add_argument("--models", nargs="+", default=["gemini-3.8-flash", "gemini-4-argon"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(compact(asyncio.run(evaluate(args.base_url, args.models, args.repeats))))
