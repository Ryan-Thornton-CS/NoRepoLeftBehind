# Changelog

## 0.1.0

Initial NoRepoLeftBehind release.

- Configurable OpenAI-compatible tool translation for Copilot, including full-history
  flattening and message-preserving profiles.
- Validated tool parsing, isolated correction requests, buffered SSE, concurrent chats,
  and optional provider continuation.
- Skill guidance for metadata and instructions supplied in client context; no invented
  tools, skill discovery, or tool execution in the middleware.
- Source grouped into API, configuration, protocol, translation, and runtime packages,
  with Python docstrings throughout.
- User and contributor documentation, a fully commented configuration example, and
  complete-source release packages with manifests and checksums.
- Python 3.12–3.14 CI covering protocol behavior, isolation, typing, packaging, and startup.
