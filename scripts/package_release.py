"""Verify built archives and assemble installable, complete-source release assets.

Run after ``uv build``. Uses only the standard library and never contacts GitHub.
The source ZIP includes a wheel for installation; third-party dependencies are not vendored.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
)
SOURCE_DIRECTORIES = ("src", "tests", "examples", "scripts", "docs", ".github")
REQUIRED_FILES = (
    "src/norepoleftbehind/__init__.py",
    "src/norepoleftbehind/__main__.py",
    "examples/bridge.yaml",
    "docs/README.md",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
)


def source_files(root: Path) -> dict[str, bytes]:
    """Collect release source and docs, excluding caches, traces, and environment files."""
    result = {name: (root / name).read_bytes() for name in ROOT_FILES}
    excluded = {"__pycache__", ".venv", ".git", ".pytest_cache", "traces"}
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise ValueError(f"Required source directory is missing: {directory}")
        for path in base.rglob("*"):
            if not path.is_file() or excluded.intersection(path.relative_to(root).parts):
                continue
            if (
                path.suffix in {".pyc", ".pyo"}
                or path.name == ".env"
                or path.name.startswith(".env.")
            ):
                continue
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    missing = set(REQUIRED_FILES) - result.keys()
    if missing:
        raise ValueError("Required release files are missing: " + ", ".join(sorted(missing)))
    return result


def verify_sdist(path: Path, expected: dict[str, bytes]) -> None:
    """Reject source archives missing or containing stale required project files."""
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for item in archive.getmembers():
            if not item.isfile():
                continue
            parts = PurePosixPath(item.name).parts
            if len(parts) < 2 or item.name.startswith("/") or ".." in parts:
                raise ValueError("Unsafe source archive member")
            member = archive.extractfile(item)
            if member is not None:
                name = PurePosixPath(*parts[1:]).as_posix()
                if name in files:
                    raise ValueError(f"Duplicate source archive entry: {name}")
                files[name] = member.read()
    for name, data in expected.items():
        if name not in files:
            raise ValueError(f"Source archive is missing {name}; update the build include list")
        if files[name] != data:
            raise ValueError(f"Source archive contains stale {name}; run uv build again")


def verify_wheel(path: Path, expected: dict[str, bytes]) -> None:
    """Ensure every nested Python source module is present and current in the wheel."""
    with zipfile.ZipFile(path) as archive:
        for name, data in expected.items():
            if not name.startswith("src/") or not name.endswith(".py"):
                continue
            target = name.removeprefix("src/")
            if target not in archive.namelist():
                raise ValueError(f"Wheel is missing {target}")
            if archive.read(target) != data:
                raise ValueError(f"Wheel contains stale {target}; run uv build again")


def checksums(files: dict[str, bytes]) -> str:
    """Return sha256sum-compatible lines for all named files, in stable path order."""
    return "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())
    )


def package_release(root: Path, dist: Path | None = None) -> Path:
    """Verify distributions, then write a source ZIP, manifest, and checksummed assets."""
    root = root.resolve()
    dist = dist.resolve() if dist else root / "dist"
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    package = project["name"].replace("-", "_")
    wheels = list(dist.glob(f"{package}-{version}-*.whl"))
    sdist = dist / f"{package}-{version}.tar.gz"
    if len(wheels) != 1 or not sdist.is_file():
        raise ValueError(
            "Run uv build first; exactly one wheel and matching source archive are required"
        )
    expected = source_files(root)
    verify_sdist(sdist, expected)
    verify_wheel(wheels[0], expected)

    output = dist / "release" / f"v{version}"
    output.mkdir(parents=True, exist_ok=True)
    assets: dict[str, bytes] = {}
    for path in [
        wheels[0],
        sdist,
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "CHANGELOG.md",
        root / "examples/bridge.yaml",
    ]:
        target = output / path.name
        shutil.copyfile(path, target)
        assets[path.name] = target.read_bytes()

    bundle_name = f"NoRepoLeftBehind-{version}.zip"
    bundle_files = {
        **expected,
        wheels[0].name: wheels[0].read_bytes(),
        "bridge.yaml": (root / "examples/bridge.yaml").read_bytes(),
    }
    with zipfile.ZipFile(output / bundle_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(bundle_files.items()):
            archive.writestr(f"NoRepoLeftBehind-{version}/{name}", data)
        archive.writestr(f"NoRepoLeftBehind-{version}/SHA256SUMS", checksums(bundle_files))
    assets[bundle_name] = (output / bundle_name).read_bytes()
    manifest = {
        "project": "NoRepoLeftBehind",
        "version": version,
        "source_files": sorted(expected),
        "assets": sorted(assets),
        "third_party_dependencies_vendored": False,
    }
    assets["release-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    (output / "release-manifest.json").write_bytes(assets["release-manifest.json"])
    (output / "SHA256SUMS").write_text(checksums(assets))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist-dir", type=Path)
    args = parser.parse_args()
    try:
        directory = package_release(args.root, args.dist_dir)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Release packaging failed: {exc}\n")
    print(
        json.dumps(
            {
                "release_directory": str(directory),
                "assets": sorted(p.name for p in directory.iterdir()),
            }
        )
    )
