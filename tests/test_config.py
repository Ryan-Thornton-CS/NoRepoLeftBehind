import json

import pytest
from pydantic import ValidationError

from norepoleftbehind.cli import main
from norepoleftbehind.configuration import Settings, load_settings
from norepoleftbehind.protocol.pointers import get, insert, remove


def test_yaml_profile_env_cli_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("port: 8000\nprofiles:\n  work:\n    port: 8001\n    history_mode: messages\n")
    monkeypatch.setenv("NOREPOLEFTBEHIND__PORT", "8002")
    monkeypatch.setenv("NOREPOLEFTBEHIND__LIMITS__MAX_CONCURRENCY", "4")
    config = load_settings(path, "work", {"port": 8003})
    assert config.port == 8003
    assert config.history_mode == "messages"
    assert config.limits.max_concurrency == 4


@pytest.mark.parametrize(
    "patch",
    [
        {"unexpected": True},
        {"upstream_base_url": "ftp://host"},
        {"upstream_base_url": "https://token:secret@host/v1"},
        {"upstream_base_url": "https://host/v1/chat/completions"},
        {"instructions": "preserve"},
        {"tools_mode": "preserve"},
        {"cloud_context": {"enabled": True}},
        {"cloud_context": {"response_id_header": "x-context"}},
        {"cloud_context": {"request_id_pointer": "/messages/0"}},
        {"limits": {"max_concurrency": 0}},
    ],
)
def test_bad_config(patch):
    with pytest.raises(ValidationError):
        Settings.model_validate(patch)


def test_native_mode_and_header_context_config():
    settings = Settings.model_validate(
        {
            "history_mode": "messages",
            "instructions": "preserve",
            "tools_mode": "preserve",
            "cloud_context": {
                "enabled": True,
                "contract_verified": True,
                "response_id_pointer": None,
                "response_id_header": "x-context",
            },
        }
    )
    assert settings.cloud_context.enabled


def test_config_is_not_shared():
    left, right = Settings(), Settings()
    left.forward_headers.append("custom")
    assert "custom" not in right.forward_headers


def test_pointer_escaping_and_no_original_mutation():
    original = {"a/b": {"~key": "old"}}
    result = insert(original, "/a~1b/~0key", "new")
    assert get(result, "/a~1b/~0key") == "new"
    assert original["a/b"]["~key"] == "old"
    remove(result, "/a~1b/~0key")
    assert get(result, "/a~1b/~0key") is None


def test_cli_diagnostics_and_validate(capsys):
    assert main(["diagnostics"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network_requests"] == result["tools_executed"] == 0
    assert main(["validate", "--config", "examples/bridge.yaml"]) == 0


def test_invalid_cli_config_does_not_echo_credentials(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("upstream_base_url: https://user:super-secret@example.com/v1")
    assert main(["validate", "--config", str(path)]) == 2
    assert "super-secret" not in capsys.readouterr().err
