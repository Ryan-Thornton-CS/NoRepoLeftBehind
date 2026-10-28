import asyncio
import json
import logging
from copy import deepcopy

import httpx
from fastapi.responses import JSONResponse

from llm_tool_bridge.app import create_app
from llm_tool_bridge.config import Settings
from llm_tool_bridge.context import Sessions
from llm_tool_bridge.errors import BridgeError
from llm_tool_bridge.upstream import Upstream

from .conftest import completion, envelope


async def test_content_traces_opt_in_redact_authentication_and_rotate(
    tmp_path, gateway, request_body
):
    settings = Settings.model_validate(
        {
            "diagnostics": {
                "content_traces": True,
                "trace_directory": str(tmp_path / "traces"),
                "trace_max_bytes": 1024,
                "trace_record_bytes": 1024,
            }
        }
    )
    app = create_app(settings, gateway.factory)
    gateway.responder = lambda body, headers: completion("token-never-log " + "x" * 500)
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as api:
        for _ in range(12):
            response = await api.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"Authorization": "Bearer token-never-log"},
            )
            assert response.status_code == 200
    app.state.bridge.diagnostics.close()
    files = list((tmp_path / "traces").iterdir())
    assert 1 <= len(files) <= 3
    assert all(file.stat().st_size < 2048 for file in files)
    assert all("token-never-log" not in file.read_text() for file in files)
    assert all("authorization" not in file.read_text().lower() for file in files)


async def test_default_metadata_does_not_log_prompts_or_tokens(api, request_body, caplog):
    caplog.set_level(logging.INFO, logger="llm_tool_bridge")
    response = await api.post(
        "/v1/chat/completions",
        json=request_body,
        headers={"Authorization": "Bearer confidential-token"},
    )
    assert response.status_code == 200
    assert "confidential-token" not in caplog.text
    assert "Read README.md" not in caplog.text
    assert "parse" in caplog.text


async def test_corrections_do_not_reuse_provider_cookie_jar(api, gateway, request_body):
    def respond(body, headers):
        if len(gateway.received) == 1:
            return JSONResponse(
                completion('{"tool_calls":[{"name":"read_file","arguments":{}}]}'),
                headers={"Set-Cookie": "provider-chat=private; Path=/"},
            )
        assert "cookie" not in headers
        return completion(envelope())

    gateway.responder = respond
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 200
    assert len(gateway.received) == 2


async def test_disconnected_client_cancels_work_and_releases_slots(gateway, request_body):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def respond(body, headers):
        entered.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    gateway.responder = respond
    app = create_app(Settings(), gateway.factory)
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {
                "type": "http.request",
                "body": json.dumps(request_body).encode(),
                "more_body": False,
            }
        await entered.wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [],
            "server": ("bridge", 80),
            "client": ("local", 1),
        },
        receive,
        send,
    )
    assert cancelled.is_set()
    assert app.state.bridge.capacity._value == 8
    assert sent[0]["status"] == 499


async def test_response_limit(gateway, request_body):
    settings = Settings.model_validate({"limits": {"max_response_bytes": 256}})
    app = create_app(settings, gateway.factory)
    gateway.responder = lambda body, headers: completion("x" * 2000)
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as api:
        result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 502
    assert result.json()["error"]["code"] == "response_too_large"


async def test_native_invalid_arguments_are_corrected(api, gateway, request_body):
    def respond(body, headers):
        if len(gateway.received) == 1:
            response = completion(None)
            response["choices"][0]["message"]["tool_calls"] = [
                {
                    "id": "call_native",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":'},
                }
            ]
            response["choices"][0]["finish_reason"] = "tool_calls"
            return response
        return completion(envelope())

    gateway.responder = respond
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 200
    assert result.json()["bridge"]["correction_attempts"] == 1


async def test_all_calls_validate_before_any_are_emitted(api, gateway, request_body):
    valid = json.loads(envelope())["tool_calls"][0]
    invalid = {"name": "missing", "arguments": {}}
    gateway.responder = lambda body, headers: completion(
        json.dumps({"content": None, "tool_calls": [valid, invalid]})
    )
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 502
    assert "tool_calls" not in result.json()


async def test_session_capacity_never_evicts_active_session():
    sessions = Sessions(Settings.model_validate({"cloud_context": {"max_sessions": 1}}))
    async with sessions.acquire("first"):
        try:
            async with sessions.acquire("second"):
                raise AssertionError("Busy session must not be evicted")
        except BridgeError as exc:
            assert exc.status == 429
    async with sessions.acquire("second"):
        assert "first" not in sessions.entries


async def test_multiple_calls_and_forced_choice(api, gateway, request_body):
    calls = json.loads(envelope("one"))["tool_calls"] + json.loads(envelope("two"))["tool_calls"]
    gateway.responder = lambda body, headers: completion(
        json.dumps({"content": "Reading both.", "tool_calls": calls})
    )
    result = await api.post(
        "/v1/chat/completions",
        json={
            **request_body,
            "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        },
    )
    message = result.json()["choices"][0]["message"]
    assert len(message["tool_calls"]) == 2
    assert message["content"] == "Reading both."
    assert message["tool_calls"][0]["id"] != message["tool_calls"][1]["id"]


async def test_request_concurrency_is_bounded(gateway, request_body):
    active = peak = 0

    async def respond(body, headers):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return completion(envelope())

    gateway.responder = respond
    app = create_app(Settings.model_validate({"limits": {"max_concurrency": 2}}), gateway.factory)
    async with httpx.AsyncClient(
        base_url="http://bridge", transport=httpx.ASGITransport(app=app)
    ) as api:
        responses = await asyncio.gather(
            *(api.post("/v1/chat/completions", json=deepcopy(request_body)) for _ in range(8))
        )
    assert all(response.status_code == 200 for response in responses)
    assert peak == 2


async def test_malformed_named_choice_is_a_client_error(api, request_body):
    result = await api.post(
        "/v1/chat/completions",
        json={**request_body, "tool_choice": {"type": "function", "function": {"name": []}}},
    )
    assert result.status_code == 400


async def test_nullable_historical_tool_calls(api, request_body):
    request_body["messages"].insert(
        1, {"role": "assistant", "content": "previous", "tool_calls": None}
    )
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 200


async def test_nonfinite_and_duplicate_request_fields_are_rejected(api):
    for body in (
        '{"model":"m","messages":[],"temperature":NaN}',
        '{"model":"m","model":"other","messages":[]}',
    ):
        result = await api.post("/v1/chat/completions", content=body)
        assert result.status_code == 400


async def test_sdk_transport_does_not_redirect_credentials_or_retry():
    settings = Settings.model_validate({"limits": {"connect_timeout_seconds": 7}})
    async with Upstream(settings).client({"authorization": "Bearer private"}) as sdk:
        assert sdk.max_retries == 0
        assert sdk.timeout.connect == 7
        assert sdk._client.follow_redirects is False
