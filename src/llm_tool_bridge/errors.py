from typing import Any


class BridgeError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def payload(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": "bridge_error", "code": self.code}}


class ParseFailure(Exception):
    """A recognizable attempted tool call failed validation."""
