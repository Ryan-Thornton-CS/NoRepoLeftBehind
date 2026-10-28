"""Represent sanitized HTTP errors and recoverable tool parsing failures."""

from typing import Any


class BridgeError(Exception):
    """Carry a safe HTTP status, machine-readable code, and user-facing error message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        """Store sanitized error details; callers must not include credentials or prompt text."""
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def payload(self) -> dict[str, Any]:
        """Return the standard OpenAI-style error object without internal exception details."""
        return {"error": {"message": self.message, "type": "bridge_error", "code": self.code}}


class ParseFailure(Exception):
    """A recognizable attempted tool call failed validation."""
