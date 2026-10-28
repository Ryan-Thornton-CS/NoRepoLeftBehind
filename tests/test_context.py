import asyncio
from copy import deepcopy

import httpx
import pytest
from fastapi.responses import JSONResponse

from llm_tool_bridge.app import create_app
from llm_tool_bridge.config import Settings

from .conftest import completion, envelope


def cloud_settings(**kwargs):
    return Settings.model_validate(
        {
            "cloud_context": {
                "enabled": True,
                "contract_verified": True,
                **kwargs,
            }
        }
    )


def client(settings, gateway):
    app = create_app(settings, gateway.factory)
    return app, httpx.AsyncClient(base_url="http://bridge", transport=httpx.ASGITransport(app=app))


async def first_turn(api, body, session="chat-a", token="token-a"):
    return await api.post(
        "/v1/chat/completions",
        json=body,
        headers={"X-LLM-Bridge-Session-ID": session, "Authorization": "Bearer " + token},
    )


def followup(body, response):
    result = deepcopy(body)
    assistant = response.json()["choices"][0]["message"]
    result["messages"] += [assistant, {"role": "user", "content": "NEXT_TURN_ONLY"}]
    return result


async def test_cloud_context_requires_session_id(gateway, request_body):
    _, api = client(cloud_settings(), gateway)
    async with api:
        response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "session_id_required"
    assert not gateway.received


async def test_continuation_sends_only_unseen_turns_and_reinjects_instructions(
    gateway, request_body
):
    gateway.responder = lambda body, headers: completion(
        '{"content":"first answer","tool_calls":[]}', id=f"context-{len(gateway.received)}"
    )
    app, api = client(cloud_settings(), gateway)
    async with api:
        first = await first_turn(api, request_body)
        assert first.status_code == 200
        second = await first_turn(api, followup(request_body, first))
    assert second.status_code == 200
    body, headers = gateway.received[-1]
    assert body["previous_interaction_id"] == "context-1"
    assert "NEXT_TURN_ONLY" in body["messages"][0]["content"]
    assert "Read README.md" not in body["messages"][0]["content"]
    assert "Use the file tool" in body["messages"][0]["content"]
    assert "TOOL_DEFINITIONS" in body["messages"][0]["content"]
    assert "x-llm-bridge-session-id" not in headers
    state = next(iter(app.state.bridge.sessions.entries.values()))
    assert "first answer" not in str(state.history)  # only digests are stored


async def test_eight_cloud_sessions_and_credential_scopes_do_not_mix(gateway, request_body):
    async def respond(body, headers):
        await asyncio.sleep(0.002)
        token = headers["authorization"].split()[-1]
        prior = body.get("previous_interaction_id")
        if prior:
            assert prior == f"context-{token}"
        return completion('{"content":"answer","tool_calls":[]}', id=f"context-{token}")

    gateway.responder = respond
    app, api = client(cloud_settings(), gateway)
    async with api:

        async def one(index):
            # Same explicit ID, different credentials: must still be isolated.
            first = await first_turn(api, request_body, "same-id", str(index))
            second = await first_turn(api, followup(request_body, first), "same-id", str(index))
            assert second.status_code == 200

        await asyncio.gather(*(one(index) for index in range(8)))
    assert len(app.state.bridge.sessions.entries) == 8


@pytest.mark.parametrize("change", ["history", "model", "credentials", "expiry"])
async def test_incompatible_continuation_resets(gateway, request_body, change):
    gateway.responder = lambda body, headers: completion('{"content":"answer","tool_calls":[]}')
    app, api = client(cloud_settings(), gateway)
    async with api:
        first = await first_turn(api, request_body)
        next_body = followup(request_body, first)
        token = "token-a"
        if change == "history":
            next_body["messages"][1]["content"] = "EDITED_FIRST_TURN"
        elif change == "model":
            next_body["model"] = "gemini-4-argon"
        elif change == "credentials":
            token = "token-b"
        else:
            next(iter(app.state.bridge.sessions.entries.values())).last_used -= 2000
        second = await first_turn(api, next_body, token=token)
    assert second.status_code == 200
    assert "previous_interaction_id" not in gateway.received[-1][0]


async def test_rejected_context_restarts_with_full_history(gateway, request_body):
    def respond(body, headers):
        if body.get("previous_interaction_id"):
            return JSONResponse({"error": {"code": "context_not_found"}}, status_code=400)
        return completion('{"content":"answer","tool_calls":[]}')

    gateway.responder = respond
    _, api = client(cloud_settings(), gateway)
    async with api:
        first = await first_turn(api, request_body)
        second = await first_turn(api, followup(request_body, first))
    assert second.status_code == 200
    assert len(gateway.received) == 3
    assert "Read README.md" in str(gateway.received[-1][0])
    assert "previous_interaction_id" not in gateway.received[-1][0]


async def test_custom_nested_json_mapping_and_body_session(gateway, request_body):
    settings = cloud_settings(
        preset="custom",
        response_id_pointer="/vertex/context",
        request_id_pointer="/metadata/context",
        session_id_header=None,
        session_id_body_pointer="/bridge_session",
    )
    gateway.responder = lambda body, headers: completion(
        '{"content":"answer","tool_calls":[]}', vertex={"context": "custom-id"}
    )
    _, api = client(settings, gateway)
    body = {**request_body, "bridge_session": "chat-a"}
    async with api:
        first = await api.post("/v1/chat/completions", json=body)
        second = await api.post("/v1/chat/completions", json=followup(body, first))
    assert second.status_code == 200
    assert gateway.received[-1][0]["metadata"]["context"] == "custom-id"
    assert "bridge_session" not in gateway.received[-1][0]


async def test_header_context_mapping(gateway, request_body):
    settings = cloud_settings(response_id_pointer=None, response_id_header="X-Vertex-Context")
    gateway.responder = lambda body, headers: JSONResponse(
        completion('{"content":"answer","tool_calls":[]}'),
        headers={"X-Vertex-Context": "header-id"},
    )
    _, api = client(settings, gateway)
    async with api:
        first = await first_turn(api, request_body)
        second = await first_turn(api, followup(request_body, first))
    assert second.status_code == 200
    assert gateway.received[-1][0]["previous_interaction_id"] == "header-id"


async def test_missing_id_does_not_commit_state(gateway, request_body):
    _, api = client(cloud_settings(response_id_pointer="/absent"), gateway)
    async with api:
        response = await first_turn(api, request_body)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "context_id_missing"


async def test_repair_never_reuses_or_commits_cloud_context(gateway, request_body):
    def respond(body, headers):
        if "failed_response" in str(body):
            assert "previous_interaction_id" not in body
            return completion(envelope())
        return completion('{"tool_calls":[{"name":"read_file","arguments":{}}]}')

    gateway.responder = respond
    app, api = client(cloud_settings(), gateway)
    async with api:
        first = await first_turn(api, request_body)
        assert first.status_code == 200
        second = await first_turn(api, followup(request_body, first))
    assert second.status_code == 200
    assert all("previous_interaction_id" not in body for body, _ in gateway.received)
    assert next(iter(app.state.bridge.sessions.entries.values())).upstream_id is None


async def test_same_session_requests_are_serialized(gateway, request_body):
    active = 0
    max_active = 0

    async def respond(body, headers):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return completion('{"content":"answer","tool_calls":[]}')

    gateway.responder = respond
    _, api = client(cloud_settings(), gateway)
    async with api:
        responses = await asyncio.gather(
            first_turn(api, request_body), first_turn(api, request_body)
        )
    assert all(response.status_code == 200 for response in responses)
    assert max_active == 1


async def test_expired_context_with_incomplete_history_fails(gateway, request_body):
    _, api = client(cloud_settings(), gateway)
    request_body["messages"] = [{"role": "tool", "tool_call_id": "unknown", "content": "result"}]
    async with api:
        response = await first_turn(api, request_body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "full_history_required"
    assert not gateway.received
