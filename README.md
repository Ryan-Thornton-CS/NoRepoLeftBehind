# NoRepoLeftBehind

[![CI](https://github.com/Ryan-Thornton-CS/NoRepoLeftBehind/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryan-Thornton-CS/NoRepoLeftBehind/actions/workflows/ci.yml)

Use your work LLM API with Copilot, even when the API drops system instructions,
ignores conversation history, or does not support native tool calling.

NoRepoLeftBehind runs locally between Copilot and your OpenAI-compatible API. It
puts the instructions, conversation, and available tool definitions into a format
the model can use, then returns ordinary OpenAI responses to Copilot.
**Copilot continues to approve and execute tools.**

Supports Linux/WSL, Python 3.12+, and `/v1/chat/completions`. Model IDs and API
credentials stay in your Copilot configuration. The installed command is
`norepoleftbehind`.

## Install and start

1. Open [Releases](https://github.com/Ryan-Thornton-CS/NoRepoLeftBehind/releases)
   and download `NoRepoLeftBehind-0.1.0.zip`. It includes the installable wheel,
   complete project source, and a commented `bridge.yaml`. Alternatively, download
   the wheel and `bridge.yaml` separately.
2. Extract the ZIP and open a terminal in its `NoRepoLeftBehind-0.1.0` directory.
   Create an environment and install the wheel:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install ./norepoleftbehind-0.1.0-py3-none-any.whl
   ```

   pip downloads the required Python dependencies. You do not need the `scripts/`
   directory to run the application.
3. Edit `bridge.yaml`. Set `upstream_base_url` to your work API's base URL, such as
   `https://your-gateway.example/v1`. Leave off `/chat/completions`. Keep tokens
   out of this file.
4. Validate the configuration, run the offline check, and start the service:

   ```bash
   norepoleftbehind validate --config bridge.yaml
   norepoleftbehind diagnostics --config bridge.yaml
   norepoleftbehind serve --config bridge.yaml
   ```

The offline check sends no model requests. Once started, the local endpoint is
`http://127.0.0.1:8787/v1/chat/completions`. Leave that terminal running; press
**Ctrl+C** to stop. For installation from a source checkout, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Connect VS Code Copilot

1. Run **Chat: Manage Language Models** and add a **Custom Endpoint**.
2. Choose **Chat Completions** and use
   `http://127.0.0.1:8787/v1/chat/completions` as the endpoint URL.
3. Enter your gateway's exact model ID and store its API key through VS Code's
   credential prompt. Repeat for other models you want to use.
4. Enable tool calling and disable image inputs for this endpoint. Set input/output
   token limits according to your gateway's actual limits.
5. Select the model in Copilot and try a short request, then a task that uses a tool.

Custom gateway authentication headers can be supplied through VS Code's
`requestHeaders`, for example `api-key: ${apiKey}`. Add custom header names to
`forward_headers` in the bridge configuration when needed. In WSL or Remote
sessions, the extension host must be able to reach the local endpoint.

The [VS Code model configuration guide](https://code.visualstudio.com/docs/agent-customization/language-models)
provides the current UI and configuration reference.

## Choose how context is sent

The default, `history_mode: flatten`, places all history supplied by Copilot into
one user message together with instructions and tool definitions. Use it when your
API reads only the last user message.

If the API keeps user/assistant message history, use the supplied profile:

```bash
norepoleftbehind serve --config bridge.yaml --profile preserves_messages
```

You can independently inject, preserve, or omit system instructions and tool
schemas. Every setting and allowed choice is described in
[examples/bridge.yaml](examples/bridge.yaml). Configuration precedence is YAML base,
selected profile, environment, then CLI flags. Restart after changes.

For example, change a nested setting without editing YAML:

```bash
export NOREPOLEFTBEHIND__LIMITS__MAX_CONCURRENCY=4
```

## Where tools and skills come from

**Tools:** Copilot sends the enabled tools for each request in its `tools` array.
This can include built-in, extension, and MCP tools. The bridge injects their names,
descriptions, and JSON argument schemas into the prompt. It validates generated
calls and returns standard `tool_calls`; it never connects to MCP servers or
executes the tools. Only tools actually advertised in that request are available.

**Skills:** skills provide instructions and supporting resources. Copilot supplies
available skill information and loads relevant `SKILL.md` content into context.
The bridge preserves that supplied context and adds `skill_prompt` guidance for
using it. If instructions still need loading, the model must request an available
loading or reading tool. Skill names do not become invented function names.
[How Copilot uses skills](https://code.visualstudio.com/docs/agent-customization/agent-skills#how-copilot-uses-skills)
explains this distinction.

Set `skill_prompt: null` to disable the added skill guidance, or replace the text
in YAML. This does not remove skill instructions already supplied by Copilot;
those follow the normal history/instruction settings. The bridge does not discover
skills or read local skill files on its own.

## Sessions, privacy, and limits

- Multiple chats can run concurrently. By default, each request uses only its own
  supplied history; no conversation identifier or persistent transcript is needed.
- Cloud context is optional and off by default. Enable it only after verifying your
  gateway's continuation contract and a stable per-chat ID from the client. A static
  provider-wide header is not a separate identity for each chat. The Google mapping
  in the example is a preset, not a guarantee of gateway support.
- Logs contain metadata by default. Content tracing requires explicit opt-in and can
  contain work content. Credentials are forwarded per request, not saved in YAML.
- Responses are buffered for validation and possible correction before streaming
  to Copilot. This delays the first visible output. Check the client timeout with
  your gateway.
- Oversized prompts fail clearly instead of silently losing history. Token estimates
  are approximate, and injected schemas also consume input budget.
- This release handles text and function tools. Images, audio, legacy `functions`,
  multiple choices, `/v1/models` discovery, and `/v1/responses` are not supported.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Service asks for a gateway URL | Replace the `.invalid` placeholder in `bridge.yaml`. |
| Authentication fails | Check the credential in Copilot and the forwarded header name. |
| Tool requests repeatedly fail | Check the configured schema and correction budget. Complex schemas may need `parsing.example_arguments`. |
| Context budget is exceeded | Let Copilot compact its history, or adjust the configured budget to your API's limits. |
| Cloud context asks for a session ID | Use a verified per-conversation ID or leave cloud context disabled. |
| Copilot times out | Check provider latency, buffering, and the configured overall deadline. |

For development, benchmarks, source layout, and release instructions, see the
[developer README](docs/README.md). Report problems in
[GitHub Issues](https://github.com/Ryan-Thornton-CS/NoRepoLeftBehind/issues).
