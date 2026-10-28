import json
import os
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .pointers import parts


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CloudContext(StrictModel):
    enabled: bool = False
    contract_verified: bool = False
    preset: Literal["google_interactions", "custom"] = "google_interactions"
    response_id_pointer: str | None = "/id"
    response_id_header: str | None = None
    request_id_pointer: str = "/previous_interaction_id"
    session_id_header: str | None = "X-LLM-Bridge-Session-ID"
    session_id_body_pointer: str | None = None
    idle_ttl_seconds: float = Field(default=1800, gt=0)
    max_sessions: int = Field(default=128, ge=1, le=10000)
    max_history_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    rejected_context_codes: list[str] = [
        "invalid_context",
        "context_not_found",
        "invalid_previous_interaction_id",
    ]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        for pointer in [
            self.response_id_pointer,
            self.request_id_pointer,
            self.session_id_body_pointer,
        ]:
            if pointer is not None:
                parts(pointer)
        if bool(self.response_id_pointer) == bool(self.response_id_header):
            raise ValueError("Select exactly one response ID source: body pointer or header")
        if self.enabled and not self.contract_verified:
            raise ValueError(
                "Cloud context requires contract_verified: true after gateway verification"
            )
        if self.enabled and bool(self.session_id_header) == bool(self.session_id_body_pointer):
            raise ValueError(
                "Enabled cloud context requires exactly one verified session ID source"
            )
        # Never overwrite the conversation, identity, or tool registry with an opaque ID.
        if parts(self.request_id_pointer)[0] in {"messages", "model", "tools", "stream"}:
            raise ValueError(
                "Context ID destination cannot overwrite a core Chat Completions field"
            )
        return self


class Limits(StrictModel):
    max_concurrency: int = Field(default=8, ge=1, le=512)
    request_timeout_seconds: float = Field(default=180, gt=0)
    connect_timeout_seconds: float = Field(default=10, gt=0)
    max_request_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_prompt_bytes: int = Field(default=8 * 1024 * 1024, ge=256)
    max_response_bytes: int = Field(default=8 * 1024 * 1024, ge=256)
    max_output_tokens: int = Field(default=16384, ge=1)
    estimated_input_token_limit: int | None = Field(default=None, ge=1)
    estimated_bytes_per_token: float = Field(default=3, gt=0)
    token_safety_margin: float = Field(default=1.2, ge=1)


class Parsing(StrictModel):
    tolerance: Literal["broad", "canonical"] = "broad"
    correction_attempts: int = Field(default=2, ge=0, le=5)
    max_tool_calls: int = Field(default=32, ge=1, le=256)
    example_arguments: dict[str, dict[str, Any]] = {}


class Diagnostics(StrictModel):
    content_traces: bool = False
    trace_directory: Path = Path("traces")
    trace_max_bytes: int = Field(default=1024 * 1024, ge=1024)
    trace_backup_count: int = Field(default=2, ge=0, le=10)
    trace_record_bytes: int = Field(default=65536, ge=256)


class Settings(StrictModel):
    upstream_base_url: str = "https://gateway.example.invalid/v1"
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)
    history_mode: Literal["flatten", "messages"] = "flatten"
    instructions: Literal["inject", "preserve", "omit"] = "inject"
    tools_mode: Literal["prompt_and_translate", "preserve", "omit"] = "prompt_and_translate"
    prompt_template: Literal["auto", "generic_v1", "gemini_flash_v1", "gemini_argon_v1"] = "auto"
    forward_headers: list[str] = [
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "x-request-id",
        "ocp-apim-subscription-key",
        "openai-organization",
        "openai-project",
    ]
    drop_body_fields: list[str] = []
    limits: Limits = Field(default_factory=Limits)
    parsing: Parsing = Field(default_factory=Parsing)
    cloud_context: CloudContext = Field(default_factory=CloudContext)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    @field_validator("upstream_base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("upstream_base_url must be an HTTP(S) base URL without credentials")
        if url.query or url.fragment or url.path.rstrip("/").endswith("/chat/completions"):
            raise ValueError("Use the upstream base URL ending in /v1, not /chat/completions")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_modes(self) -> Self:
        if self.history_mode == "flatten" and self.instructions == "preserve":
            raise ValueError("Native instruction roles require history_mode: messages")
        if self.history_mode == "flatten" and self.tools_mode == "preserve":
            raise ValueError("Native tool history requires history_mode: messages")
        return self


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_settings(
    path: Path | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    document: dict[str, Any] = {}
    if path:
        loaded = yaml.safe_load(path.read_text())
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("Configuration must be a YAML object")
        document = loaded or {}
    profiles = document.pop("profiles", {})
    if profile:
        if not isinstance(profiles, dict) or profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}")
        document = merge(document, profiles[profile])
    for name, value in os.environ.items():
        if not name.startswith("LLM_BRIDGE__"):
            continue
        keys = name[len("LLM_BRIDGE__") :].lower().split("__")
        # JSON decoding accepts arrays/bools/numbers without treating URLs as YAML.
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        nested: Any = parsed
        for key in reversed(keys):
            nested = {key: nested}
        document = merge(document, nested)
    return Settings.model_validate(merge(document, overrides or {}))
