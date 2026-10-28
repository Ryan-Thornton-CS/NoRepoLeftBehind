import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from .config import Settings
from .context import Session, Sessions, normalized_history, require_full_history
from .diagnostics import Diagnostics
from .errors import BridgeError, ParseFailure
from .parsing import decode_json, parse_output, validate_parsed
from .pointers import get, insert
from .prompts import correction_body
from .protocol import Parsed, ToolPolicy, UpstreamResult, compact, validate_request
from .transform import budget, transform
from .upstream import Upstream


class Bridge:
    def __init__(self, settings: Settings, upstream: Upstream | None = None) -> None:
        self.settings = settings
        self.upstream = upstream or Upstream(settings)
        self.sessions = Sessions(settings)
        self.capacity = asyncio.Semaphore(settings.limits.max_concurrency)
        self.diagnostics = Diagnostics(settings)

    async def handle(self, body: Any, headers: dict[str, str]) -> dict[str, Any]:
        started = time.monotonic()
        headers = {key.lower(): value for key, value in headers.items()}
        request = validate_request(body)
        try:
            async with asyncio.timeout(self.settings.limits.request_timeout_seconds):
                async with self.capacity:
                    async with self._session(request, headers) as session:
                        async with self.upstream.client(headers) as client:
                            response = await self._complete(client, request, headers, session)
            self.diagnostics.record(
                "completion",
                status=200,
                milliseconds=round((time.monotonic() - started) * 1000),
                calls=len(response["choices"][0]["message"].get("tool_calls", [])),
            )
            return response
        except TimeoutError as exc:
            raise BridgeError(504, "request_timeout", "Overall request deadline exceeded") from exc
        except APITimeoutError as exc:
            raise BridgeError(504, "upstream_timeout", "Upstream request timed out") from exc
        except APIStatusError as exc:
            status = exc.status_code if 400 <= exc.status_code <= 599 else 502
            # Upstream error messages may echo credentials, prompt content, or URLs.
            raise BridgeError(status, "upstream_error", f"Upstream returned HTTP {status}") from exc
        except APIConnectionError as exc:
            raise BridgeError(502, "upstream_connection_error", "Could not reach upstream") from exc

    @asynccontextmanager
    async def _session(self, body: dict[str, Any], headers: dict[str, str]) -> Any:
        if self.settings.cloud_context.enabled:
            async with self.sessions.acquire(self.sessions.key(body, headers)) as session:
                try:
                    yield session
                except BaseException:
                    # A request may have advanced the provider even if its response was lost.
                    session.reset()
                    raise
        else:
            yield None

    async def _complete(
        self,
        client: AsyncOpenAI,
        original: dict[str, Any],
        headers: dict[str, str],
        session: Session | None,
    ) -> dict[str, Any]:
        settings = self.settings
        policy = ToolPolicy.from_request(original)
        if settings.tools_mode == "omit":
            if policy.choice in {"required", "named"}:
                raise BridgeError(
                    400, "tools_disabled", "Profile omits tools but request requires a call"
                )
            policy = ToolPolicy()
        original_history = original["messages"]
        history = normalized_history(original_history)
        if (
            session
            and len(compact(original_history).encode()) > settings.cloud_context.max_history_bytes
        ):
            raise BridgeError(
                400, "session_history_too_large", "Cloud session history exceeds byte budget"
            )
        delta = None
        if session:
            if (
                session.upstream_id
                and len(history) > len(session.history)
                and (history[: len(session.history)] == session.history)
            ):
                delta = original_history[len(session.history) :]
            else:
                session.reset()
                require_full_history(original_history)
        request = transform(original, settings, policy, delta)
        if session and session.upstream_id:
            try:
                request = insert(
                    request, settings.cloud_context.request_id_pointer, session.upstream_id
                )
            except ValueError as exc:
                raise BridgeError(
                    400, "context_field_conflict", "Context ID field collides with request data"
                ) from exc
        try:
            upstream = await self.upstream.generate(client, request, headers)
        except APIStatusError as exc:
            code = self._error_code(exc)
            if (
                not session
                or not session.upstream_id
                or code not in settings.cloud_context.rejected_context_codes
            ):
                raise
            session.reset()
            require_full_history(original_history)
            request = transform(original, settings, policy)
            upstream = await self.upstream.generate(client, request, headers)

        first = upstream
        attempts = 0
        correction_usage: list[dict[str, Any]] = []
        while True:
            try:
                parsed, finish_reason = self._parse(upstream, policy)
                break
            except ParseFailure as exc:
                self.diagnostics.record("parse_failure", correction_attempt=attempts)
                if attempts >= settings.parsing.correction_attempts:
                    raise BridgeError(
                        502, "tool_translation_failed", "Tool output failed parsing or validation"
                    ) from exc
                failed = self._failed_output(upstream)
                correction = correction_body(failed, policy, original["model"], settings)
                budget(correction, settings)
                # A fresh SDK client also isolates provider cookies, not just JSON IDs.
                async with self.upstream.client(headers) as correction_client:
                    upstream = await self.upstream.generate(
                        correction_client, correction, headers, correction=True
                    )
                if isinstance(upstream.document.get("usage"), dict):
                    correction_usage.append(upstream.document["usage"])
                attempts += 1

        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if parsed.calls:
            message["tool_calls"] = [
                {
                    "id": call.get("id") or "call_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": compact(call["arguments"])},
                }
                for call in parsed.calls
            ]
        response: dict[str, Any] = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": original["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if parsed.calls else finish_reason,
                }
            ],
        }
        if isinstance(first.document.get("usage"), dict):
            response["usage"] = first.document["usage"]
        if attempts:
            response["bridge"] = {
                "correction_attempts": attempts,
                "correction_usage": correction_usage,
            }
        if session:
            if attempts or finish_reason != "stop":
                # A repaired answer differs from what the provider stored. Start fresh next turn.
                session.reset()
            else:
                context_id = (
                    first.headers.get(settings.cloud_context.response_id_header.lower())
                    if settings.cloud_context.response_id_header
                    else get(first.document, settings.cloud_context.response_id_pointer or "")
                )
                if not isinstance(context_id, str) or not context_id or len(context_id) > 4096:
                    raise BridgeError(
                        502,
                        "context_id_missing",
                        "Configured upstream context ID was absent or invalid",
                    )
                session.upstream_id = context_id
                session.history = normalized_history(original_history + [message])
        self.diagnostics.content(
            redact_values=list(headers.values()),
            transformed_prompt=compact(request["messages"]),
            model_output=self._failed_output(first),
        )
        self.diagnostics.record("parsed", corrections=attempts, parser=settings.parsing.tolerance)
        return response

    def _parse(self, result: UpstreamResult, policy: ToolPolicy) -> tuple[Parsed, str]:
        try:
            choices = result.document["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Expected one choice")
            choice = choices[0]
            message = choice["message"]
            finish = choice.get("finish_reason")
            if finish not in {"stop", "tool_calls", "length", "content_filter"}:
                raise ValueError("Missing or unsupported finish reason")
            if message.get("refusal"):
                return Parsed(message["refusal"]), "content_filter"
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Expected text")
            if content is not None:
                content.encode("utf-8")
            if finish == "content_filter":
                return Parsed(content), finish
            if finish == "length" and (policy.schemas or message.get("tool_calls")):
                raise ParseFailure("Tool output was truncated by the upstream output budget")
            if message.get("tool_calls"):
                calls = []
                try:
                    ids: set[str] = set()
                    for call in message["tool_calls"]:
                        function = call["function"]
                        args = decode_json(function["arguments"], broad=False)
                        if not isinstance(args, dict) or not isinstance(function["name"], str):
                            raise ValueError("Invalid native arguments")
                        item = {"name": function["name"], "arguments": args}
                        if self.settings.tools_mode == "preserve":
                            identifier = call.get("id")
                            if (
                                not isinstance(identifier, str)
                                or not identifier
                                or identifier in ids
                            ):
                                raise ValueError("Missing or duplicate native call ID")
                            ids.add(identifier)
                            item["id"] = identifier
                        calls.append(item)
                except (ValueError, KeyError, TypeError, RecursionError) as exc:
                    raise ParseFailure("Malformed native tool requests") from exc
                return validate_parsed(
                    Parsed(content, calls), policy, self.settings.parsing
                ), "stop"
            if finish == "content_filter":
                return Parsed(content), finish
            if finish == "length":
                # Never turn truncated output into an action, even if bracket repair could parse it.
                if policy.schemas:
                    raise ParseFailure("Tool output was truncated by the upstream output budget")
                return Parsed(content), finish
            if finish == "tool_calls":
                raise ParseFailure("Upstream indicated calls without any tool requests")
            if policy.schemas and self.settings.tools_mode == "prompt_and_translate":
                return parse_output(content or "", policy, self.settings.parsing), "stop"
            return validate_parsed(Parsed(content), policy, self.settings.parsing), "stop"
        except ParseFailure:
            raise
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, RecursionError) as exc:
            raise BridgeError(
                502, "invalid_upstream_response", "Malformed upstream Chat Completions response"
            ) from exc

    @staticmethod
    def _failed_output(result: UpstreamResult) -> str:
        try:
            message = result.document["choices"][0]["message"]
            if message.get("tool_calls"):
                return compact(message)
            return str(message.get("content") or "")
        except (KeyError, TypeError, IndexError):
            return ""

    @staticmethod
    def _error_code(exc: APIStatusError) -> str | None:
        value = exc.body
        if isinstance(value, dict):
            error = value.get("error", value)
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                return error["code"]
        return None
