import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import AsyncOpenAI

from norepoleftbehind.api.app import create_app
from norepoleftbehind.configuration.models import Settings

from .conftest import completion, envelope


async def test_sdk_nonstreaming_and_header_passthrough(api, gateway, request_body):
    async with AsyncOpenAI(
        api_key="harness-token", base_url="http://bridge/v1", max_retries=0, http_client=api
    ) as client:
        response = await client.chat.completions.create(**request_body)
    call = response.choices[0].message.tool_calls[0]
    assert call.type == "function"
    assert call.function.name == "read_file"
    assert json.loads(call.function.arguments) == {"path": "README.md"}
    assert response.choices[0].finish_reason == "tool_calls"
    body, headers = gateway.received[0]
    assert body["model"] == "gemini-3.8-flash"
    assert len(body["messages"]) == 1
    assert "tools" not in body
    assert headers["authorization"] == "Bearer harness-token"


async def test_sdk_and_raw_streaming_reconstruction(gateway, request_body):
    app = create_app(Settings(), gateway.factory)
    async with AsyncOpenAI(
        api_key="secret",
        base_url="http://bridge/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    ) as client:
        stream = await client.chat.completions.create(**request_body, stream=True)
        chunks = [chunk async for chunk in stream]
    calls = [
        call
        for chunk in chunks
        for choice in chunk.choices
        for call in choice.delta.tool_calls or []
    ]
    assert calls[0].id.startswith("call_")
    assert calls[0].function.name == "read_file"
    assert json.loads("".join(call.function.arguments or "" for call in calls)) == {
        "path": "README.md"
    }
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as raw:
        response = await raw.post("/v1/chat/completions", json={**request_body, "stream": True})
    assert response.text.endswith("data: [DONE]\n\n")
    assert response.headers["content-type"].startswith("text/event-stream")


async def test_harness_executes_simulated_tool_loop(api, gateway, request_body, tmp_path):
    forbidden_file = tmp_path / "must-not-be-read.txt"
    forbidden_file.write_text("PRIVATE_CONTENT_NOT_FROM_TOOL")
    gateway.responder = lambda body, headers: completion(
        '{"content":"The harness supplied the result.","tool_calls":[]}'
        if "FIXTURE_TOOL_RESULT" in str(body["messages"])
        else envelope(str(forbidden_file))
    )
    first = (await api.post("/v1/chat/completions", json=request_body)).json()
    call = first["choices"][0]["message"]["tool_calls"][0]
    # The test acts as the harness: it supplies a static fixture, never opens the requested file.
    followup = deepcopy(request_body)
    followup["messages"] += [
        first["choices"][0]["message"],
        {"role": "tool", "tool_call_id": call["id"], "content": "FIXTURE_TOOL_RESULT"},
    ]
    second = await api.post("/v1/chat/completions", json=followup)
    assert second.json()["choices"][0]["message"]["content"] == "The harness supplied the result."
    assert "PRIVATE_CONTENT_NOT_FROM_TOOL" not in str(gateway.received)
    assert len(gateway.received) == 2


async def test_correction_is_isolated_and_usage_is_separate(api, gateway, request_body):
    def respond(body, headers):
        if len(gateway.received) == 1:
            return completion(
                '{"tool_calls":[{"name":"read_file","arguments":{}}]}',
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        payload = json.loads(body["messages"][0]["content"])
        assert set(payload) == {"schema", "failed_response", "example"}
        assert "Read README.md" not in json.dumps(payload)
        assert "Use the file tool" not in json.dumps(payload)
        assert "tools" not in body and "previous_interaction_id" not in body
        return completion(
            envelope(), usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        )

    gateway.responder = respond
    response = await api.post(
        "/v1/chat/completions", json=request_body, headers={"Authorization": "Bearer same-token"}
    )
    result = response.json()
    assert response.status_code == 200
    assert result["bridge"]["correction_attempts"] == 1
    assert result["usage"]["total_tokens"] == 15
    assert result["bridge"]["correction_usage"][0]["total_tokens"] == 5
    assert {headers["authorization"] for _, headers in gateway.received} == {"Bearer same-token"}


async def test_exhaustion_is_http_error_even_when_streaming(api, gateway, request_body):
    gateway.responder = lambda body, headers: completion(
        '{"tool_calls":[{"name":"missing","arguments":{}}]}'
    )
    response = await api.post("/v1/chat/completions", json={**request_body, "stream": True})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_translation_failed"
    assert "text/event-stream" not in response.headers["content-type"]
    assert len(gateway.received) == 3


async def test_eight_concurrent_chats_keep_auth_history_and_corrections_isolated(
    api, gateway, request_body
):
    async def respond(body, headers):
        await asyncio.sleep(0.005)
        token = headers["authorization"].split()[-1]
        source = body["messages"][0]["content"]
        assert f"user-{token}" in source or '"failed_response"' in source
        if '"failed_response"' not in source:
            return completion('{"tool_calls":[{"name":"read_file","arguments":{"path":null}}]}')
        return completion(envelope(token))

    gateway.responder = respond

    async def one(index):
        request = deepcopy(request_body)
        request["messages"][-1]["content"] = f"user-{index}"
        result = await api.post(
            "/v1/chat/completions", json=request, headers={"Authorization": f"Bearer {index}"}
        )
        assert result.status_code == 200
        args = result.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(args)["path"] == str(index)

    await asyncio.gather(*(one(index) for index in range(8)))
    assert len(gateway.received) == 16


@pytest.mark.parametrize(
    "change,code",
    [
        (
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
                ]
            },
            "unsupported_content",
        ),
        ({"n": 2}, "unsupported_parameter"),
        ({"messages": []}, "invalid_request"),
        ({"tool_choice": "nope"}, "invalid_tool_choice"),
    ],
)
async def test_invalid_requests_fail_before_upstream(api, gateway, request_body, change, code):
    response = await api.post("/v1/chat/completions", json={**request_body, **change})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert not gateway.received


async def test_body_limit_and_invalid_json(gateway, request_body):
    app = create_app(
        Settings.model_validate({"limits": {"max_request_bytes": 1024}}), gateway.factory
    )
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as client:
        response = await client.post("/v1/chat/completions", content="{" + "a" * 2000)
        assert response.status_code == 413
        response = await client.post("/v1/chat/completions", content="not-json")
        assert response.status_code == 400
    assert not gateway.received


async def test_timeout_releases_capacity_and_cancels_upstream(gateway, request_body):
    cancelled = asyncio.Event()

    async def respond(body, headers):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    gateway.responder = respond
    app = create_app(
        Settings.model_validate({"limits": {"request_timeout_seconds": 0.03}}), gateway.factory
    )
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as client:
        response = await client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 504
    assert cancelled.is_set()
    assert app.state.bridge.capacity._value == 8


async def test_upstream_errors_are_sanitized_and_not_retried(api, gateway, request_body, caplog):
    gateway.responder = lambda body, headers: JSONResponse(
        {"error": {"message": "secret-token"}}, status_code=429
    )
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 429
    assert "secret-token" not in response.text + caplog.text
    assert len(gateway.received) == 1


async def test_no_fabricated_usage_and_custom_auth_without_bearer(api, gateway, request_body):
    response = await api.post(
        "/v1/chat/completions", json=request_body, headers={"api-key": "gateway-token"}
    )
    assert response.status_code == 200
    assert "usage" not in response.json()
    assert gateway.received[0][1]["api-key"] == "gateway-token"
    assert "authorization" not in gateway.received[0][1]


async def test_native_passthrough_validates_and_preserves_content(gateway, request_body):
    settings = Settings(history_mode="messages", tools_mode="preserve", instructions="preserve")
    app = create_app(settings, gateway.factory)
    result = completion("I will read it.")
    result["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "provider-call",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"x"}'},
        }
    ]
    result["choices"][0]["finish_reason"] = "tool_calls"
    gateway.responder = lambda body, headers: result
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as client:
        response = await client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert "tools" in gateway.received[0][0]
    assert response.json()["choices"][0]["message"]["content"] == "I will read it."
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "provider-call"


async def test_normal_answer_with_tools(api, gateway, request_body):
    gateway.responder = lambda body, headers: completion('{"content":"Done.","tool_calls":[]}')
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.json()["choices"][0]["message"]["content"] == "Done."
    assert response.json()["choices"][0]["finish_reason"] == "stop"


async def test_streaming_usage_and_correction_metadata(api, gateway, request_body):
    gateway.responder = lambda body, headers: completion(
        envelope(), usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    )
    response = await api.post(
        "/v1/chat/completions",
        json={**request_body, "stream": True, "stream_options": {"include_usage": True}},
    )
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert frames[-1]["usage"]["total_tokens"] == 5
    assert frames[-1]["choices"] == []


async def test_invalid_and_oversize_upstream(api, gateway, request_body):
    gateway.responder = lambda body, headers: Response(
        content="not-json", media_type="application/json"
    )
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 502
    gateway.responder = lambda body, headers: StreamingResponse(
        iter([b'data: {"choices":[]}\n\n']), media_type="text/event-stream"
    )
    response = await api.post("/v1/chat/completions", json={**request_body, "stream": True})
    assert response.json()["error"]["code"] == "upstream_stream_incomplete"
