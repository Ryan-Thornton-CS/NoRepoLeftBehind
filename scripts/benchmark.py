"""In-process, synthetic upstream benchmark; no network or tool execution."""

import argparse
import asyncio
import json
import logging
import statistics
import time
import tracemalloc

import httpx
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

from llm_tool_bridge.app import create_app
from llm_tool_bridge.config import Settings
from llm_tool_bridge.protocol import ToolPolicy
from llm_tool_bridge.transform import transform


async def benchmark(count: int, concurrency: int, prompt_bytes: int, delay: float) -> dict:
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def generate(request: Request) -> dict:
        await request.json()
        await asyncio.sleep(delay)
        return {
            "id": "synthetic",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"content": None, "tool_calls": [{"name": "fixture", "arguments": {}}]}
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    def factory(headers: dict) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="synthetic",
            base_url="http://synthetic/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream)),
        )

    config = Settings.model_validate({"limits": {"max_concurrency": concurrency}})
    app = create_app(config, factory)
    body = {
        "model": "gemini-3.8-flash",
        "messages": [{"role": "user", "content": "x" * prompt_bytes}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "fixture",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    policy = ToolPolicy.from_request(body)
    transformed_start = time.perf_counter()
    for _ in range(100):
        transform(body, config, policy)
    transform_ms = (time.perf_counter() - transformed_start) * 10
    pool = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as client:
        # Warm imports and client initialization before measuring steady-state allocations.
        (await client.post("/v1/chat/completions", json=body)).raise_for_status()
        latencies = []

        async def one() -> None:
            async with pool:
                started = time.perf_counter()
                response = await client.post("/v1/chat/completions", json=body)
                response.raise_for_status()
                assert response.json()["choices"][0]["message"]["tool_calls"]
                latencies.append((time.perf_counter() - started) * 1000)

        tracemalloc.start()
        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(count)))
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    ordered = sorted(latencies)
    return {
        "kind": "in_process_synthetic_not_provider_or_TLS_throughput",
        "requests": count,
        "concurrency": concurrency,
        "prompt_bytes": prompt_bytes,
        "mock_delay_ms": delay * 1000,
        "requests_per_second": round(count / elapsed, 2),
        "latency_p50_ms": round(statistics.median(ordered), 2),
        "latency_p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
        "transform_mean_ms": round(transform_ms, 3),
        "traced_peak_bytes": peak,
        "tools_executed": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-bytes", type=int, default=4096)
    parser.add_argument("--mock-delay", type=float, default=0.01)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.prompt_bytes) < 1 or args.mock_delay < 0:
        parser.error("Counts must be positive and delay nonnegative")
    logging.disable(logging.CRITICAL)
    print(
        json.dumps(
            asyncio.run(
                benchmark(args.requests, args.concurrency, args.prompt_bytes, args.mock_delay)
            ),
            indent=2,
        )
    )
