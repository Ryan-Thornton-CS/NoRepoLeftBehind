import json

import pytest

from norepoleftbehind.configuration.models import Settings
from norepoleftbehind.protocol.chat import ToolPolicy
from norepoleftbehind.translation.prompts import correction_body, tool_prompt
from norepoleftbehind.translation.transform import transform


def test_tool_prompt_describes_output_without_caller_implementation(request_body):
    prompt = tool_prompt(ToolPolicy.from_request(request_body), request_body["model"], Settings())
    assert "The harness, not you or this middleware, executes them." not in prompt
    protocol_instructions = prompt.split("\n\nOUTPUT_SCHEMA", 1)[0].lower()
    assert "harness" not in protocol_instructions
    assert "middleware" not in protocol_instructions
    assert "OUTPUT_SCHEMA" in prompt


def test_injected_schemas_are_exactly_the_tools_advertised_by_the_request(request_body):
    request_body["tools"].append(
        {
            "type": "function",
            "function": {
                "name": "mcp_catalog_search",
                "description": "Search the connected catalog",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    )
    policy = ToolPolicy.from_request(request_body)
    outgoing = transform(request_body, Settings(), policy)
    prompt = outgoing["messages"][-1]["content"]
    tools = json.loads(prompt.split("TOOL_DEFINITIONS\n", 1)[1].split("\n\n", 1)[0])
    assert tools == [item["function"] for item in request_body["tools"]]
    assert "tools" not in outgoing


@pytest.mark.parametrize("mode", ["flatten", "messages"])
def test_skill_catalog_stays_in_context_without_becoming_a_tool(request_body, mode):
    request_body["messages"][0]["content"] += (
        " Available skill: review_changes; description: review a patch; "
        "instructions: /workspace/skills/review_changes/SKILL.md"
    )
    policy = ToolPolicy.from_request(request_body)
    outgoing = transform(request_body, Settings(history_mode=mode), policy)
    text = outgoing["messages"][-1]["content"]
    assert "AVAILABLE_SKILLS_GUIDANCE" in text
    assert "review_changes" in text
    assert "/workspace/skills/review_changes/SKILL.md" in text
    assert "wait for its result" in text
    assert set(policy.schemas) == {"read_file"}
    assert '"name":"review_changes"' not in text


def test_skill_guidance_can_be_disabled_or_overridden(request_body):
    policy = ToolPolicy.from_request(request_body)
    disabled = transform(request_body, Settings(skill_prompt=None), policy)
    assert "AVAILABLE_SKILLS_GUIDANCE" not in str(disabled)
    custom = transform(
        request_body, Settings(skill_prompt="Use only explicitly invoked skills."), policy
    )
    assert (
        "AVAILABLE_SKILLS_GUIDANCE\nUse only explicitly invoked skills."
        in custom["messages"][-1]["content"]
    )


def test_skills_do_not_require_an_advertised_tool_or_correction_context(request_body):
    policy = ToolPolicy.from_request(request_body)
    settings = Settings(skill_prompt="PRIVATE_SKILL_CATALOG")
    repair = correction_body("invalid", policy, request_body["model"], settings)
    assert "PRIVATE_SKILL_CATALOG" not in json.dumps(repair)
    request_body["tools"] = []
    outgoing = transform(request_body, settings, ToolPolicy.from_request(request_body))
    assert "PRIVATE_SKILL_CATALOG" in outgoing["messages"][-1]["content"]
    assert "TOOL_DEFINITIONS" not in outgoing["messages"][-1]["content"]


def test_omitted_instructions_do_not_reappear_as_discovered_skills(request_body):
    request_body["messages"][0]["content"] = "PRIVATE_UNAVAILABLE_SKILL"
    outgoing = transform(
        request_body, Settings(instructions="omit"), ToolPolicy.from_request(request_body)
    )
    assert "PRIVATE_UNAVAILABLE_SKILL" not in str(outgoing)
