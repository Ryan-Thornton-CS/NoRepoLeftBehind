import asyncio
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .errors import BridgeError
from .parsing import decode_json
from .pointers import get
from .protocol import compact, text_content


def normalized_history(messages: list[dict[str, Any]]) -> list[str]:
    normalized = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message["role"],
            "content": text_content(message.get("content")),
        }
        if "tool_call_id" in message:
            item["tool_call_id"] = message["tool_call_id"]
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                try:
                    function["arguments"] = json_arguments(function["arguments"])
                except ValueError:
                    pass
                calls.append({"id": call["id"], "function": function})
            item["tool_calls"] = calls
        # Store only digests; full transcripts need not live in the session registry.
        normalized.append(
            hashlib.sha256(
                json.dumps(item, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
            ).hexdigest()
        )
    return normalized


def json_arguments(value: str) -> Any:
    return decode_json(value, broad=False)


@dataclass
class Session:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)
    upstream_id: str | None = None
    history: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.upstream_id = None
        self.history = []


class Sessions:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.secret = secrets.token_bytes(32)
        self.entries: dict[str, Session] = {}
        self.lock = asyncio.Lock()

    def key(self, body: dict[str, Any], headers: dict[str, str]) -> str:
        config = self.settings.cloud_context
        identity = (
            headers.get(config.session_id_header.lower())
            if config.session_id_header
            else get(body, config.session_id_body_pointer or "")
        )
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 512:
            raise BridgeError(
                400, "session_id_required", "Cloud context requires a verified conversation ID"
            )
        # Include every forwarded non-observability header: custom gateway credentials may
        # be arbitrary header names. HMAC keeps both credentials and IDs out of registry keys.
        credential_scope = sorted(
            (key.lower(), value)
            for key, value in headers.items()
            if key.lower()
            in {name.lower() for name in self.settings.forward_headers} | {"authorization"}
            and key.lower() not in {"x-request-id", "traceparent", "tracestate"}
        )
        payload = compact(
            [self.settings.upstream_base_url, body["model"], identity, credential_scope]
        )
        return hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[Session]:
        config = self.settings.cloud_context
        async with self.lock:
            now = time.monotonic()
            expired = [
                key
                for key, item in self.entries.items()
                if item.refs == 0 and now - item.last_used > config.idle_ttl_seconds
            ]
            for expired_key in expired:
                del self.entries[expired_key]
            if key not in self.entries:
                if len(self.entries) >= config.max_sessions:
                    idle = [(key, item) for key, item in self.entries.items() if item.refs == 0]
                    if not idle:
                        raise BridgeError(
                            429, "session_capacity", "All cloud session slots are busy"
                        )
                    del self.entries[min(idle, key=lambda pair: pair[1].last_used)[0]]
                self.entries[key] = Session()
            session = self.entries[key]
            session.refs += 1
        try:
            async with session.lock:
                if time.monotonic() - session.last_used > config.idle_ttl_seconds:
                    session.reset()
                yield session
        finally:
            async with self.lock:
                session.refs -= 1
                session.last_used = time.monotonic()


def require_full_history(messages: list[dict[str, Any]]) -> None:
    meaningful = [message for message in messages if message["role"] not in {"system", "developer"}]
    if not meaningful or meaningful[0]["role"] != "user":
        raise BridgeError(
            400,
            "full_history_required",
            "A fresh cloud context requires history beginning with a user turn",
        )
