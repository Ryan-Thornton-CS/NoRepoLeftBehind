import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any

from .config import Settings


class Diagnostics:
    def __init__(self, settings: Settings) -> None:
        self.config = settings.diagnostics
        self.logger = logging.getLogger("llm_tool_bridge.metadata")
        self.trace: logging.Logger | None = None
        if self.config.content_traces:
            directory = self.config.trace_directory
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.trace = logging.Logger(f"llm_tool_bridge.trace.{id(self)}")
            handler = RotatingFileHandler(
                directory / "content.jsonl",
                maxBytes=self.config.trace_max_bytes,
                backupCount=self.config.trace_backup_count,
            )
            (directory / "content.jsonl").chmod(0o600)
            self.trace.addHandler(handler)
            self.trace.setLevel(logging.INFO)

    def record(self, event: str, **metadata: Any) -> None:
        self.logger.info(json.dumps({"event": event, **metadata}))

    def content(self, *, redact_values: list[str] | None = None, **values: Any) -> None:
        if self.trace:
            # Only explicitly selected content strings; never bodies, headers, or credentials.
            record = {
                key: str(value)[: self.config.trace_record_bytes // 8]
                for key, value in values.items()
            }
            for secret in redact_values or []:
                variants = [secret]
                if secret.lower().startswith("bearer "):
                    variants.append(secret[7:])
                for variant in variants:
                    if variant:
                        record = {
                            key: value.replace(variant, "[REDACTED]")
                            for key, value in record.items()
                        }
            encoded = json.dumps(record, ensure_ascii=True)
            if len(encoded.encode()) <= self.config.trace_record_bytes:
                self.trace.info(encoded)

    def close(self) -> None:
        if self.trace:
            for handler in self.trace.handlers[:]:
                handler.close()
                self.trace.removeHandler(handler)
