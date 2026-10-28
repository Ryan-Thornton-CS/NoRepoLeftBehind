# NoRepoLeftBehind developer README

For installing and using the service, read the [user README](../README.md).
For setup and contribution checks, read [CONTRIBUTING.md](../CONTRIBUTING.md).

## Source layout

```text
src/norepoleftbehind/
  __init__.py              Package version
  __main__.py              python -m norepoleftbehind
  cli.py                   Installed CLI entry point
  api/
    app.py                 FastAPI routes, bounded bodies, disconnect handling
    streaming.py           Downstream OpenAI SSE serializer
    upstream.py            AsyncOpenAI transport and upstream SSE decoding
  configuration/
    models.py              Validated settings and defaults
    loader.py              YAML profiles, environment, and CLI precedence
  protocol/
    chat.py                Chat request validation, tool policy, response records
    errors.py              Sanitized HTTP failures and recoverable parser errors
    pointers.py            Configurable JSON field extraction and insertion
  translation/
    prompts.py             Tool-call protocol, examples, isolated corrections
    skills.py              Guidance for skill information already in context
    parsing.py             Syntax recognition and full-batch schema validation
    transform.py           History/instruction/schema rendering and budgets
  runtime/
    service.py             Request orchestration and response conversion
    sessions.py            Locked, bounded cloud continuation registry
    diagnostics.py         Metadata events and opt-in redacted content traces
    checks.py              Offline synthetic CLI diagnostics
```

Modules inside these groups are internal organization, not a promise of stable
import paths. The installed command, module entry point, YAML settings, and HTTP
endpoint are the supported interfaces. Embedding tests use
`norepoleftbehind.api.app.create_app` and `norepoleftbehind.configuration.Settings`.

FastAPI defines the application API and builds on Starlette's HTTP/ASGI layer.
Pydantic validates settings; JSON Schema validates tool arguments. `AsyncOpenAI`
handles upstream requests, with automatic retries and redirects disabled.

## Request and response flow

1. Validate a bounded text-only Chat Completions request and its current tool policy.
2. Render supplied history and selected instructions/tool schemas. Only the tools
   in this request's `tools` array enter `TOOL_DEFINITIONS`; there is no MCP discovery.
3. Send the transformed body using a request-owned SDK client and credentials.
4. Buffer JSON or SSE output, recognize the response, and validate the entire call batch.
5. If needed, make isolated correction requests containing only `schema`,
   `failed_response`, and a valid `example`.
6. Return ordinary Chat Completions JSON or SSE for the client to consume and act on.

The canonical model output is:

```json
{"content":null,"tool_calls":[{"name":"read_file","arguments":{"path":"README.md"}}]}
```

Answers put text in `content` and use an empty `tool_calls` array. Broad parsing
supports defined whole-response wrappers and unambiguous punctuation repairs;
canonical parsing requires the envelope. Arbitrary prose and quoted examples are
not searched for calls. Duplicate keys, invented arguments, unknown tools,
unterminated strings, invalid schemas, and partial valid batches are not accepted.
An entire exact call envelope is inherently indistinguishable from a model printing
that envelope as an example; client approval remains the execution boundary.

Honor `tool_choice`, named tools, and `parallel_tool_calls`. Generated argument
objects are serialized back to standard JSON-string `function.arguments`. Native
passthrough retains provider call IDs; translated calls receive new IDs. Missing
usage stays missing. Original usage and correction usage remain separate in the
optional `bridge` response extension. SSE is opened only after validation succeeds,
so repair exhaustion remains an HTTP error rather than an already-open stream.

Skill metadata and loaded instructions remain in their original supplied context,
subject to the history/instruction settings. `skill_prompt` adds procedural guidance
but creates neither a second skill catalog nor synthetic callable functions. It is
excluded from isolated corrections, along with all other original context.

## State and failure handling

The overall request deadline includes concurrency queueing, per-session locks,
generation, and correction. Disconnects cancel work and release capacity. A fresh
SDK client/cookie jar isolates each logical request and each correction. HTTP
failures are sanitized because provider errors may echo work content or credentials.

Cloud state requires an explicitly verified provider contract and a client-supplied
conversation ID. The registry stores provider IDs and normalized history digests,
not transcripts. Its HMAC keys scope identities by provider, model, and credential
headers. Active and queued entries cannot be evicted. Changed or incomplete history,
expiry, provider rejection, or model/auth changes require a fresh context with full
supplied history. Repaired output clears state because it differs from what the
provider may have stored. Disabling cloud state removes configured continuation
fields without disabling prompt injection.

The Google `/id` → `/previous_interaction_id` preset labels a known Interactions
contract; it does not implement a native Google endpoint or prove that a work
Chat Completions gateway honors those fields. Exact mappings remain configurable.

Byte budgets are hard limits. Optional token preflight uses
`ceil(UTF-8 bytes / estimated_bytes_per_token * token_safety_margin)` and is not an
exact tokenizer. Never label these estimates as provider usage. There is no automatic
history trimming or summarization.

## Why `scripts/` exists

These are developer/release entry points, not runtime application modules. Wheels
install `src/` and the CLI; source distributions and the complete ZIP retain these
scripts so contributors can reproduce checks and releases.

| Script | Purpose | Network / credentials |
|---|---|---|
| `benchmark.py` | Measures an eight-request synthetic gateway workload, rendering overhead, latency, and traced allocations. | No external requests or credentials. |
| `evaluate.py` | Opt-in evaluation of actual work models, using static fixture results as a synthetic client. | Calls a running bridge; reads `NOREPOLEFTBEHIND_EVAL_TOKEN`. |
| `check_release.py` | Verifies tag, package, and module versions agree. | None. |
| `smoke_install.py` | Starts an installed CLI, checks local `/healthz`, and stops its child process. | Loopback only; no model requests. |
| `package_release.py` | Verifies source/wheel completeness, assembles a full source ZIP and release assets, and writes checksums. | None; run after `uv build`. |

```bash
uv run python scripts/benchmark.py --requests 64 --concurrency 8
# Load NOREPOLEFTBEHIND_EVAL_TOKEN through your preferred secret mechanism first.
uv run python scripts/evaluate.py --models gemini-3.8-flash gemini-4-argon --repeats 3
```

The benchmark excludes real TLS/provider inference. Live evaluation reports validity,
correction success, tool correctness, false tool detections, and completed fixture
loops. Neither script dispatches generated actions.

## Python API documentation

Source modules, classes, methods, and functions contain Python docstrings. Read them
with the IDE or Python's standard documentation viewer:

```bash
uv run python -m pydoc norepoleftbehind.configuration.models.Settings
uv run python -m pydoc norepoleftbehind.runtime.service.Bridge
uv run python -m pydoc norepoleftbehind.translation.prompts
```

Docstring checks apply to `src/`; test fixtures and maintenance scripts are exempt
from public-API docstring lint. The commented configuration example covers every
settings field, default, and enumerated option. Add behavioral tests for interface
changes; avoid implementation-mirroring tests for prose edits.

## CI and release packaging

CI runs on Python 3.12, 3.13, and 3.14 with locked development dependencies. It checks
formatting, lint/docstrings, typing, fake-provider and SDK integration tests, builds,
clean wheel/source installs, CLI diagnostics, installed-server startup, and release
archive completeness. Normal CI uses no work API credentials.

To prepare a release from the repository root:

```bash
uv sync --locked --dev
uv run python scripts/check_release.py v0.1.0
uv build
uv run python scripts/package_release.py
```

Assets are written to `dist/release/v0.1.0/`:

- Installable wheel and standard source `.tar.gz`.
- `NoRepoLeftBehind-0.1.0.zip`: the complete source tree, wheel, ready-to-edit
  `bridge.yaml`, and internal file checksums.
- User README, contributor guide, changelog, configuration example, source manifest,
  and `SHA256SUMS` covering all other release assets.

The source archive/ZIP include all Python source and package initializers, project
metadata, dependency lock, license, configuration, documentation, tests, maintenance
scripts, and GitHub workflows. Build checks fail on missing or stale source. Third-party
dependencies are installed by pip/uv; they are not vendored for offline installation.

After reviewing and committing changes, push a version tag such as `v0.1.0`. The
release workflow reruns CI, validates the version, builds and smoke-tests packages,
and creates a **draft** GitHub Release. Inspect its assets before publishing through
GitHub. PyPI publication is not configured.
