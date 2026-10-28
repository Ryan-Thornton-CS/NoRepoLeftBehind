from copy import deepcopy

import pytest

from llm_tool_bridge.config import Settings
from llm_tool_bridge.errors import BridgeError
from llm_tool_bridge.protocol import ToolPolicy
from llm_tool_bridge.transform import transform


def test_flatten_everything_without_mutating_input(request_body):
    original = deepcopy(request_body)
    output = transform(request_body, Settings(), ToolPolicy.from_request(request_body))
    assert request_body == original
    assert len(output["messages"]) == 1
    assert output["messages"][0]["role"] == "user"
    content = output["messages"][0]["content"]
    assert "SYSTEM_AND_DEVELOPER_INSTRUCTIONS" in content
    assert "TOOL_DEFINITIONS" in content
    assert "Read README.md" in content
    assert "tools" not in output


def test_results_remain_after_the_calls(request_body):
    request_body["messages"] += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "FIXTURE_RESULT"},
    ]
    for mode in ("flatten", "messages"):
        settings = Settings(history_mode=mode)
        output = transform(request_body, settings, ToolPolicy.from_request(request_body))
        text = str(output["messages"])
        assert text.index("call_1") < text.index("FIXTURE_RESULT")
        assert output["messages"][-1]["role"] == "user"
        assert "TOOL_DEFINITIONS" in output["messages"][-1]["content"]


@pytest.mark.parametrize("instruction", ["inject", "preserve", "omit"])
@pytest.mark.parametrize("tools", ["prompt_and_translate", "preserve", "omit"])
def test_independent_modes(request_body, instruction, tools):
    settings = Settings(history_mode="messages", instructions=instruction, tools_mode=tools)
    output = transform(request_body, settings, ToolPolicy.from_request(request_body))
    assert ("tools" in output) == (tools == "preserve")
    assert any(m["role"] == "system" for m in output["messages"]) == (instruction == "preserve")
    assert ("TOOL_DEFINITIONS" in str(output)) == (tools == "prompt_and_translate")
    assert ("Use the file tool" in str(output)) == (instruction != "omit")


def test_transformed_budget_counts_injected_schemas(request_body):
    settings = Settings.model_validate({"limits": {"max_prompt_bytes": 300}})
    with pytest.raises(BridgeError, match="including schemas"):
        transform(request_body, settings, ToolPolicy.from_request(request_body))


def test_disabled_cloud_context_is_removed(request_body):
    request_body["previous_interaction_id"] = "must-not-be-forwarded"
    output = transform(request_body, Settings(), ToolPolicy.from_request(request_body))
    assert "previous_interaction_id" not in output


def test_preserved_history_is_not_flattened(request_body):
    request_body["messages"] += [
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    output = transform(
        request_body, Settings(history_mode="messages"), ToolPolicy.from_request(request_body)
    )
    assert [m["role"] for m in output["messages"]] == ["user", "assistant", "user"]
    assert output["messages"][1]["content"] == "first answer"
