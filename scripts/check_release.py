import sys
import tomllib
from pathlib import Path

from llm_tool_bridge import __version__

version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
if len(sys.argv) != 2 or sys.argv[1] != f"v{version}" or __version__ != version:
    raise SystemExit("Tag, package version, and module version must agree")
print(f"Verified v{version}")
