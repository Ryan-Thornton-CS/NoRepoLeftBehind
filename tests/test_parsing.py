import json

import pytest

from norepoleftbehind.configuration.models import Parsing, Settings
from norepoleftbehind.protocol.chat import ToolPolicy
from norepoleftbehind.protocol.errors import BridgeError, ParseFailure
from norepoleftbehind.translation.parsing import parse_output
from norepoleftbehind.translation.prompts import correction_body

from .conftest import TOOLS, envelope


def policy(**kwargs):
    return ToolPolicy.from_request({"tools": TOOLS, **kwargs})


@pytest.mark.parametrize(
    "source",
    [
        envelope(),
        "```json\n" + envelope() + "\n```",
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"a",},},],}',
        '{"function":{"name":"read_file","arguments":"{\\"path\\":\\"a\\"}"}}',
        '{"functionCall":{"name":"read_file","args":{"path":"a"}}}',
        '[{"name":"read_file","arguments":{"path":"a"}}]',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"a"}}',
    ],
)
def test_supported_variants(source):
    parsed = parse_output(source, policy(), Parsing())
    assert len(parsed.calls) == 1
    assert parsed.calls[0]["name"] == "read_file"


@pytest.mark.parametrize(
    "source",
    [
        '{"tool_calls":[{"name":"unknown","arguments":{"path":"x"}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":1}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"x","extra":true}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"x"},"args":{}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"unfinished',
        '{"tool_calls":[],"tool_calls":[{"name":"read_file","arguments":{"path":"x"}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":NaN}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"x"}},null]}',
    ],
)
def test_invalid_or_ambiguous_calls(source):
    with pytest.raises(ParseFailure):
        parse_output(source, policy(), Parsing())


@pytest.mark.parametrize(
    "source",
    [
        "Here is an example:\n```json\n" + envelope() + "\n```",
        "The tool_calls field describes requests. " + envelope(),
        json.dumps(envelope()),
        '{"name":"A person"}',
        "def example():\n    return " + envelope(),
    ],
)
def test_quoted_examples_and_prose_are_answers(source):
    result = parse_output(source, policy(), Parsing())
    assert not result.calls
    assert result.content == source


def test_choice_and_parallel_controls():
    with pytest.raises(ParseFailure):
        parse_output(envelope(), policy(tool_choice="none"), Parsing())
    with pytest.raises(ParseFailure):
        parse_output("answer", policy(tool_choice="required"), Parsing())
    with pytest.raises(ParseFailure):
        parse_output(
            json.dumps({"content": None, "tool_calls": json.loads(envelope())["tool_calls"] * 2}),
            policy(parallel_tool_calls=False),
            Parsing(),
        )


def test_canonical_mode_does_not_repair():
    assert parse_output(envelope(), policy(), Parsing(tolerance="canonical")).calls
    with pytest.raises(ParseFailure):
        parse_output(
            '{"functionCall":{"name":"read_file","args":{"path":"x"}}}',
            policy(),
            Parsing(tolerance="canonical"),
        )


def test_correction_has_only_schema_failure_and_valid_example():
    request = correction_body("failed", policy(), "gemini-4-argon", Settings())
    payload = json.loads(request["messages"][0]["content"])
    assert set(payload) == {"schema", "failed_response", "example"}
    assert parse_output(json.dumps(payload["example"]), policy(), Parsing()).calls
    assert request["model"] == "gemini-4-argon"


def test_complex_example_override():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "code",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "pattern": "^ABC$"}},
                    "required": ["id"],
                },
            },
        }
    ]
    selected = ToolPolicy.from_request({"tools": tools})
    with pytest.raises(BridgeError, match="example_arguments"):
        correction_body("bad", selected, "m", Settings())
    settings = Settings.model_validate({"parsing": {"example_arguments": {"code": {"id": "ABC"}}}})
    assert correction_body("bad", selected, "m", settings)


def test_remote_schema_refs_are_rejected():
    tools = [
        {
            "type": "function",
            "function": {"name": "bad", "parameters": {"$ref": "https://example.com/schema.json"}},
        }
    ]
    with pytest.raises(BridgeError):
        ToolPolicy.from_request({"tools": tools})


@pytest.mark.parametrize(
    "source",
    [
        '{"tool_calls":[{"name":"read_file","arguments":{"path":1e999}}]}',
        '{"tool_calls":[{"name":"read_file","arguments":{"path":"\\ud800"}}]}',
    ],
)
def test_nonrepresentable_json_values_are_not_returned_as_calls(source):
    with pytest.raises(ParseFailure):
        parse_output(source, policy(), Parsing())
