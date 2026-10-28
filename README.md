# LLM Tool Bridge

Local OpenAI-compatible middleware for VS Code Copilot. It translates tool definitions
into prompts and generated tool requests back into Chat Completions `tool_calls`.
**It never executes tools.** Your harness approves and executes every returned call.

Requires Python 3.12+ on Linux/WSL. See the example YAML for configuration.

For a release wheel, install with `python -m pip install /path/to/llm_tool_bridge-0.1.0-py3-none-any.whl`,
then run `llm-tool-bridge serve --upstream-base-url https://your-gateway.example/v1`.
The following commands are for a source checkout with uv installed:

```bash
uv sync --locked --dev
uv run llm-tool-bridge validate --config examples/bridge.yaml
uv run llm-tool-bridge diagnostics
uv run llm-tool-bridge serve --config examples/bridge.yaml --upstream-base-url https://your-gateway.example/v1
```

The gateway URL must be a base URL, not the complete `/chat/completions` route.
Keep credentials in the harness. They are forwarded per request, never stored in config.

## VS Code Copilot

In **Chat: Manage Language Models**, add a **Custom Endpoint** using the full URL
`http://127.0.0.1:8787/v1/chat/completions`. Select Chat Completions, enter your actual
gateway model ID, and store your gateway credential in VS Code's secret storage.
Set `toolCalling: true`, `vision: false`, and the appropriate input/output limits.
The proxy translates tool requests; Copilot retains its tool execution and approval flow.

Representative entry in `chatLanguageModels.json` (follow your VS Code version's
generated provider structure):

```json
{
  "name": "Work gateway through bridge",
  "vendor": "customendpoint",
  "apiKey": "${input:workGatewayKey}",
  "models": [{
    "id": "gemini-3.8-flash",
    "name": "Work Gemini Flash",
    "url": "http://127.0.0.1:8787/v1/chat/completions",
    "apiType": "chat-completions",
    "toolCalling": true,
    "vision": false,
    "streaming": true,
    "maxInputTokens": 120000,
    "maxOutputTokens": 16000
  }]
}
```

These token values are examples, not verified gateway/model limits. Injected schemas
and instructions consume additional input budget. Check buffering latency against
your Copilot version's timeout before regular use. In WSL or VS Code Remote, ensure
the extension host can reach the selected localhost address; use port forwarding as needed.

For API-key header authentication, configure the harness's `requestHeaders` with
`api-key: ${apiKey}` (or the header your gateway requires). The bridge preserves that
header without adding a bearer credential. Extend `forward_headers` for other gateway
headers. Hop-by-hop headers, cookies, and the bridge session header are not forwarded.
The bridge does not provide `/v1/models` discovery: configure exact model IDs explicitly.

## Profiles and limits

`examples/bridge.yaml` documents the available settings and two optional profiles:

```bash
uv run llm-tool-bridge serve --config examples/bridge.yaml --profile preserves_messages --upstream-base-url https://your-gateway.example/v1
```

Precedence is defaults → YAML base → selected YAML profile → environment → CLI flags.
Environment names use `LLM_BRIDGE__` and double underscores for nested settings:

```bash
export LLM_BRIDGE__LIMITS__MAX_CONCURRENCY=8
export LLM_BRIDGE__LIMITS__REQUEST_TIMEOUT_SECONDS=180
export LLM_BRIDGE__HISTORY_MODE=messages
```

Arrays/objects, booleans, and numbers use JSON values in environment variables.
Unknown configuration fields fail validation. Restart after changes. Native tool or
instruction preservation requires `history_mode: messages`; incompatible flattening
combinations fail validation instead of silently losing roles.

Flattening serializes **all history supplied by the harness**, not history previously
discarded by that harness. Message mode preserves user/assistant turns, renders tool
exchanges as text, and injects schemas/instructions into the final user continuation.
Text-only content arrays are accepted; image/audio/file parts are rejected. `n` must be
1. Legacy `functions` are unsupported. Function schemas may use local `$ref` references;
external references are rejected without network access.

Native `response_format` conflicts with prompt-based tool translation and is rejected.
Use `drop_body_fields` only for gateway-specific unsupported parameters; the bridge does
not assume that native Gemini parameter restrictions apply unchanged through your gateway.

The hard prompt limit measures bytes **after** transformation. Optional token preflight
uses `ceil(UTF-8 bytes / estimated_bytes_per_token * token_safety_margin)` and is explicitly
an estimate; it cannot guarantee an exact model context fit. Upstream budget errors are
propagated with sanitized messages. No automatic trimming or summarization is performed.

## Parsing and correction

The model is instructed to produce an object with `content` and `tool_calls`. Broad mode
accepts entire JSON fences, OpenAI/Gemini function wrappers, root call arrays, trailing
commas, and missing closing brackets when strings and values are otherwise complete.
It never searches arbitrary prose for calls, evaluates expressions, fills missing values,
or accepts unknown tools. Duplicate keys, unterminated strings, and invalid arguments
are sent for correction. Canonical mode requires the canonical shape.

`tool_choice` (including `none`, `required`, and named tools), `parallel_tool_calls`, and
`parsing.max_tool_calls` are enforced on the whole batch. A partial valid batch is not
returned. Printed or quoted examples cannot trigger a tool through substring matching.
Text protocols remain ambiguous in principle: a whole response intentionally reproducing
an exact call envelope is indistinguishable from a requested call. Harness approval remains
the execution boundary. Injected instructions also cannot recreate native system-role priority.

Correction requests contain only `schema`, `failed_response`, and `example`. They use a
fresh SDK client, the same model/authentication, and no conversation or provider cookies.
The default is two attempts within the original deadline. For complex parameter schemas,
supply a valid `parsing.example_arguments.<tool_name>` if the bounded example generator
cannot produce one; otherwise correction fails clearly rather than fabricating arguments.

The middleware buffers JSON or SSE upstream responses, validates/corrects them, and only
then opens downstream SSE. Errors therefore remain HTTP errors. Missing usage stays
missing; original generation usage and correction usage are kept separate using a `bridge`
extension. Cancellation stops upstream work. SDK automatic retries are disabled. Only a
recognized rejected continuation code permits one retry with a fresh full context.

## Concurrent chats and optional cloud context

Eight concurrent requests are allowed by default. With cloud context disabled there is
no history registry and no session ID requirement. SDK clients, authentication, cookies,
buffers, and corrections are isolated per logical request. This sacrifices connection
reuse across independent requests in favor of isolation; benchmark actual TLS latency
against your gateway before optimizing pooling.

Cloud context is **off by default**. The Google Interactions preset (`/id` →
`/previous_interaction_id`) is a configurable mapping, not evidence that your gateway's
Chat Completions route supports it. Vertex `/responseId` is not assumed to be a continuation
token. Verify field semantics and stateless corrections against your actual gateway first.
Enable only with `cloud_context.enabled: true` and `contract_verified: true`.

To select a response header instead, set `response_id_pointer: null` and
`response_id_header: X-Your-Context-ID`. A custom mapping may use nested JSON Pointers,
including escaped object keys. `request_id_pointer` selects the extra request field.

Enabled cloud state also requires a **verified, stable per-conversation ID** from the harness.
The default source is `X-LLM-Bridge-Session-ID`; alternatively set `session_id_header: null`
and a `session_id_body_pointer`. Do not configure a fixed provider-wide header and call
it a chat ID. Current Copilot custom endpoints may not supply a suitable dynamic ID.
Use the default request-local mode in that case; do not enable cloud continuation.

The registry holds opaque provider IDs and history digests, not transcripts or raw credentials.
Keys are HMAC-scoped by upstream/model/conversation/forwarded credentials. Sessions expire
after 30 minutes idle by default, have a bounded count, and are never persisted. Same-session
requests serialize. Unseen turns continue from the saved ID; current instructions and tools
are reinjected. Model/auth changes, edits, expiry, or rejected IDs start fresh with supplied
full history. A fresh context cannot begin with an orphaned tool result. Repaired responses
reset cloud state because the provider may have stored the malformed original response.

## Diagnostics and development

Metadata logs contain statuses, timings, call counts, and correction outcomes. They omit
headers, prompts, and arguments. `diagnostics.content_traces: true` explicitly enables
bounded rotating local content traces; protect that directory because it can contain work
content. Incoming header values are redacted from traces. No cloud session transcript is
persisted. `/healthz` reports process health, not upstream availability.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
uv run python scripts/benchmark.py --requests 64 --concurrency 8
```

The benchmark uses a synthetic in-process gateway and reports transformation overhead,
latency, and traced allocations. It does not measure provider inference, TLS performance,
or justify a Python-versus-Rust claim by itself.

Live evaluation is opt-in, uses your configured gateway through a running bridge, and incurs
model requests. Provide the token through the environment, never a command-line argument:

```bash
# Set LLM_BRIDGE_EVAL_TOKEN using your preferred secret-loading mechanism first.
uv run python scripts/evaluate.py --models gemini-3.8-flash gemini-4-argon --repeats 3
```

The evaluator acts as a synthetic harness supplying predefined fixture results. It reports
first-pass validity, corrected responses, tool-name/argument correctness, false tool detections,
and completed loops; it never dispatches generated actions. Exact work-gateway aliases may differ.

## CI and release

GitHub Actions runs lint/format/type checks, fake-upstream and SDK tests, package builds,
fresh wheel/source installs, and an installed HTTP-server health smoke test on Python
3.12, 3.13, and 3.14. Normal CI needs no work credentials or live model calls.

A version tag such as `v0.1.0` reuses the full CI matrix, checks tag/package/module version
agreement, and creates a **draft** GitHub Release with wheel, source archive, checksums,
README, and example YAML. PyPI publishing is not configured. Workflows become active
when this project is pushed to a GitHub repository; no repository or release is created
by local installation. `/v1/responses` is a future adapter, not implemented in v0.1.
