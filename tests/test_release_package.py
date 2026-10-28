import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from scripts.package_release import ROOT_FILES, package_release, source_files


@pytest.fixture
def release_project(tmp_path):
    files = {name: name.encode() for name in ROOT_FILES}
    files.update(
        {
            "pyproject.toml": b'[project]\nname = "norepoleftbehind"\nversion = "0.1.0"\n',
            "src/norepoleftbehind/__init__.py": b'__version__ = "0.1.0"\n',
            "src/norepoleftbehind/__main__.py": b"from .cli import main\n",
            "src/norepoleftbehind/cli.py": b"def main(): return 0\n",
            "src/norepoleftbehind/api/__init__.py": b'"""API."""\n',
            "src/norepoleftbehind/api/app.py": b'"""HTTP routes."""\n',
            "examples/bridge.yaml": b"history_mode: flatten\n",
            "docs/README.md": b"# Developer documentation\n",
            "scripts/package_release.py": b'"""Packaging helper."""\n',
            "tests/test_example.py": b"def test_example(): assert True\n",
            ".github/workflows/ci.yml": b"name: CI\n",
            ".github/workflows/release.yml": b"name: Release\n",
        }
    )
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return tmp_path, files


def build_fixtures(root, files, *, missing_sdist=None, stale_sdist=None, missing_wheel=None):
    dist = root / "dist"
    dist.mkdir()
    with tarfile.open(dist / "norepoleftbehind-0.1.0.tar.gz", "w:gz") as archive:
        for name, data in files.items():
            if name == missing_sdist:
                continue
            data = b"stale" if name == stale_sdist else data
            member = tarfile.TarInfo("norepoleftbehind-0.1.0/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    with zipfile.ZipFile(dist / "norepoleftbehind-0.1.0-py3-none-any.whl", "w") as archive:
        for name, data in files.items():
            if name.startswith("src/") and name.endswith(".py") and name != missing_wheel:
                archive.writestr(name.removeprefix("src/"), data)


def test_complete_source_bundle_contains_every_required_file_and_valid_checksums(release_project):
    root, files = release_project
    build_fixtures(root, files)
    output = package_release(root)
    manifest = json.loads((output / "release-manifest.json").read_text())
    assert set(manifest["source_files"]) == set(files)
    assert manifest["third_party_dependencies_vendored"] is False
    with zipfile.ZipFile(output / "NoRepoLeftBehind-0.1.0.zip") as bundle:
        prefix = "NoRepoLeftBehind-0.1.0/"
        for name, content in files.items():
            assert bundle.read(prefix + name) == content
        assert bundle.read(prefix + "bridge.yaml") == files["examples/bridge.yaml"]
        for line in bundle.read(prefix + "SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(bundle.read(prefix + name)).hexdigest() == digest
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "omitted",
    ["src/norepoleftbehind/api/app.py", "docs/README.md", ".github/workflows/ci.yml", "uv.lock"],
)
def test_release_rejects_missing_required_source(release_project, omitted):
    root, files = release_project
    build_fixtures(root, files, missing_sdist=omitted)
    with pytest.raises(ValueError, match="missing"):
        package_release(root)


def test_release_rejects_stale_source(release_project):
    root, files = release_project
    build_fixtures(root, files, stale_sdist="src/norepoleftbehind/api/app.py")
    with pytest.raises(ValueError, match="stale"):
        package_release(root)


def test_release_rejects_missing_wheel_subpackage(release_project):
    root, files = release_project
    build_fixtures(root, files, missing_wheel="src/norepoleftbehind/api/app.py")
    with pytest.raises(ValueError, match="Wheel is missing"):
        package_release(root)


def test_source_collection_excludes_credentials_and_python_caches(release_project):
    root, _ = release_project
    for name in ["src/.env", "examples/.env.personal", "src/__pycache__/test.pyc"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must not ship")
    assert all(b"must not ship" != value for value in source_files(root).values())


def test_release_requires_built_artifacts(release_project):
    root, _ = release_project
    with pytest.raises(ValueError, match="uv build"):
        package_release(root)
