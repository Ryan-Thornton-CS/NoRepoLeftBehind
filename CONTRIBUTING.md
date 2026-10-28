# Contributing to NoRepoLeftBehind

The [main README](README.md) covers installation and use. The
[developer README](docs/README.md) covers architecture, protocol details, maintenance
scripts, documentation, and releases.

## Development setup

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/Ryan-Thornton-CS/NoRepoLeftBehind.git
cd NoRepoLeftBehind
uv sync --locked --dev
uv run norepoleftbehind diagnostics
uv run norepoleftbehind serve --config examples/bridge.yaml --upstream-base-url https://your-gateway.example/v1
```

## Before submitting a change

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
uv run python scripts/package_release.py
```

Keep module, class, and function docstrings current. Public source docstrings are
checked by Ruff. Explain public interfaces, configuration semantics, failure modes,
and state ownership where they matter. Update `examples/bridge.yaml` whenever a
configuration field or allowed value changes.

Use fake upstreams for normal tests. Live evaluations require an explicit manual
invocation and your own gateway credential. Never commit tokens, personal gateway
configurations, traces, or generated packages.

The middleware translates requests and responses. Tool execution, approval, and
MCP/skill discovery remain the client's responsibility.
