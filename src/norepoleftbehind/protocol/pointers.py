"""Read and update configured JSON Pointer locations in provider payloads."""

from copy import deepcopy
from typing import Any


def parts(pointer: str) -> list[str]:
    """Split a non-root JSON Pointer and unescape slash and tilde tokens."""
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must begin with /")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def get(document: Any, pointer: str) -> Any:
    """Read a pointer from mappings or arrays; return None for missing or invalid paths."""
    try:
        for part in parts(pointer):
            document = document[int(part)] if isinstance(document, list) else document[part]
        return document
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def remove(document: dict[str, Any], pointer: str | None) -> None:
    """Remove a configured field in place if present; missing paths are harmless."""
    if not pointer:
        return
    keys = parts(pointer)
    parent = (
        get(document, "/" + "/".join(pointer[1:].split("/")[:-1])) if len(keys) > 1 else document
    )
    if isinstance(parent, dict):
        parent.pop(keys[-1], None)
    elif isinstance(parent, list) and keys[-1].isdigit() and int(keys[-1]) < len(parent):
        parent.pop(int(keys[-1]))


def insert(document: dict[str, Any], pointer: str, value: Any) -> dict[str, Any]:
    """Return a copied payload with an opaque value inserted at an object-field pointer.

    Missing object parents are created. Existing scalar or array parents raise
    ValueError instead of being overwritten. The original payload is unchanged.
    """
    result = deepcopy(document)
    current = result
    keys = parts(pointer)
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise ValueError("Configured context pointer collides with a non-object field")
        current = current[key]
    current[keys[-1]] = value
    return result
