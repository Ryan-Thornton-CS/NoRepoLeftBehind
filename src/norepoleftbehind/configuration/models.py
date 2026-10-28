"""Define validated configuration schemas, defaults, and incompatible-mode checks."""

from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from norepoleftbehind.protocol.pointers import parts
from norepoleftbehind.translation.skills import DEFAULT_SKILL_PROMPT


class StrictModel(BaseModel):
    """Reject unknown configuration keys instead of silently ignoring misspellings."""

    model_config = ConfigDict(extra="forbid")


class CloudContext(StrictModel):
    """Configure optional provider continuation, independent of prompt placement.

    Attributes:
        enabled: Retain and resend opaque provider IDs; off by default.
        contract_verified: Explicit declaration that gateway continuation semantics were
            tested.
        preset: Mapping label, google_interactions or custom; explicit pointers remain
            authoritative.
        response_id_pointer: JSON Pointer for a returned ID, mutually exclusive with its
            header source.
        response_id_header: Alternative response-header source, or None.
        request_id_pointer: JSON Pointer at which to insert the saved ID on continuation.
        session_id_header: Header carrying a verified per-conversation identity.
        session_id_body_pointer: Alternative identity source, removed before forwarding.
        idle_ttl_seconds: Idle lifetime before state is discarded on registry access.
        max_sessions: Registry capacity; idle entries may be evicted to make room.
        max_history_bytes: Maximum supplied history size for a cloud-context request.
        rejected_context_codes: Provider error codes that permit one full-history retry.
    """

    enabled: bool = False
    contract_verified: bool = False
    preset: Literal["google_interactions", "custom"] = "google_interactions"
    response_id_pointer: str | None = "/id"
    response_id_header: str | None = None
    request_id_pointer: str = "/previous_interaction_id"
    session_id_header: str | None = "X-NoRepoLeftBehind-Session-ID"
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
        """Reject ambiguous ID sources, unverified enablement, and core-field destinations."""
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
    """Bound request resource use and optional estimated-token preflight.

    Attributes:
        max_concurrency: Maximum active logical requests, including their corrections.
        request_timeout_seconds: Overall deadline, including queueing and session locks.
        connect_timeout_seconds: SDK connection-establishment timeout.
        max_request_bytes: Maximum incoming HTTP body size.
        max_prompt_bytes: Maximum serialized prompt after transformation and schema injection.
        max_response_bytes: Maximum buffered body for each upstream generation.
        max_output_tokens: Fallback output cap when the harness supplies none; also used for
            repairs.
        estimated_input_token_limit: Optional estimate-based input cap, not an exact provider
            count.
        estimated_bytes_per_token: UTF-8 bytes divided by this value estimate the token count.
        token_safety_margin: Multiplier applied to the optional token estimate.
    """

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
    """Control tool syntax tolerance, repair attempts, and correction examples.

    Attributes:
        tolerance: broad accepts defined wrappers/repairs; canonical requires the output
            envelope.
        correction_attempts: Maximum isolated model corrections, separate from HTTP retries.
        max_tool_calls: Maximum calls accepted in one fully validated response.
        example_arguments: Schema-valid argument examples keyed by exact tool name.
    """

    tolerance: Literal["broad", "canonical"] = "broad"
    correction_attempts: int = Field(default=2, ge=0, le=5)
    max_tool_calls: int = Field(default=32, ge=1, le=256)
    example_arguments: dict[str, dict[str, Any]] = {}


class Diagnostics(StrictModel):
    """Configure optional local content tracing; metadata logging stays enabled.

    Attributes:
        content_traces: Explicit opt-in for sensitive prompt/output traces.
        trace_directory: Trace directory, relative to the process working directory when not
            absolute.
        trace_max_bytes: Rotation threshold per content log file.
        trace_backup_count: Number of rotated backups to retain.
        trace_record_bytes: Encoded record cap after content truncation and redaction.
    """

    content_traces: bool = False
    trace_directory: Path = Path("traces")
    trace_max_bytes: int = Field(default=1024 * 1024, ge=1024)
    trace_backup_count: int = Field(default=2, ge=0, le=10)
    trace_record_bytes: int = Field(default=65536, ge=256)


class Settings(StrictModel):
    """Describe one effective middleware profile; authentication comes from the harness.

    Attributes:
        upstream_base_url: Provider API base URL, without the /chat/completions suffix.
        host: Local HTTP listener address.
        port: Local HTTP listener port.
        history_mode: flatten sends one user message; messages retains conversation turns.
        instructions: inject moves system/developer text, preserve retains roles, omit removes
            it.
        tools_mode: prompt_and_translate injects schemas, preserve uses native fields, omit
            removes them.
        prompt_template: auto selects by model ID; named v1 templates override selection.
        skill_prompt: Guidance for supplied skill context; None disables only this added
            guidance.
        forward_headers: Additional incoming headers allowed upstream; transport exclusions
            still apply.
        drop_body_fields: Gateway-specific top-level fields to remove before forwarding.
        limits: Resource and preflight budgets.
        parsing: Tool validation and correction policy.
        cloud_context: Optional provider continuation configuration.
        diagnostics: Local logging and trace policy.
    """

    upstream_base_url: str = "https://gateway.example.invalid/v1"
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)
    history_mode: Literal["flatten", "messages"] = "flatten"
    instructions: Literal["inject", "preserve", "omit"] = "inject"
    tools_mode: Literal["prompt_and_translate", "preserve", "omit"] = "prompt_and_translate"
    prompt_template: Literal["auto", "generic_v1", "gemini_flash_v1", "gemini_argon_v1"] = "auto"
    skill_prompt: str | None = DEFAULT_SKILL_PROMPT
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
        """Require an HTTP(S) API base URL without embedded credentials, query, or fragment."""
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("upstream_base_url must be an HTTP(S) base URL without credentials")
        if url.query or url.fragment or url.path.rstrip("/").endswith("/chat/completions"):
            raise ValueError("Use the upstream base URL ending in /v1, not /chat/completions")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_modes(self) -> Self:
        """Reject native role/tool preservation combined with single-user-message flattening."""
        if self.history_mode == "flatten" and self.instructions == "preserve":
            raise ValueError("Native instruction roles require history_mode: messages")
        if self.history_mode == "flatten" and self.tools_mode == "preserve":
            raise ValueError("Native tool history requires history_mode: messages")
        return self
