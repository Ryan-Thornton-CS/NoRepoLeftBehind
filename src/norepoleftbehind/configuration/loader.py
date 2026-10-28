"""Load YAML profiles and apply environment and CLI overrides without storing credentials."""

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .models import Settings


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge mapping overrides into a new dict; scalar values and lists replace existing
    values.
    """
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_settings(
    path: Path | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load and validate settings in increasing precedence order.

    Args:
        path: Optional YAML file containing base settings and named profiles.
        profile: Optional entry from the YAML profiles mapping.
        overrides: Explicit CLI or embedding overrides, applied last.

    Returns:
        Settings loaded from defaults, YAML base, profile, NOREPOLEFTBEHIND__ environment
        variables, and explicit overrides. Unknown fields fail validation.

    Raises:
        OSError: If the requested file cannot be read.
        ValueError: If a profile, YAML structure, or settings value is invalid.
    """
    document: dict[str, Any] = {}
    if path:
        loaded = yaml.safe_load(path.read_text())
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("Configuration must be a YAML object")
        document = loaded or {}
    profiles = document.pop("profiles", {})
    if profile:
        if not isinstance(profiles, dict) or profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}")
        document = merge(document, profiles[profile])
    for name, value in os.environ.items():
        if not name.startswith("NOREPOLEFTBEHIND__"):
            continue
        keys = name[len("NOREPOLEFTBEHIND__") :].lower().split("__")
        # JSON decoding accepts arrays/bools/numbers without treating URLs as YAML.
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        nested: Any = parsed
        for key in reversed(keys):
            nested = {key: nested}
        document = merge(document, nested)
    return Settings.model_validate(merge(document, overrides or {}))
