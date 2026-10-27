"""Dependency isolation and cache identity of the optional CZI reader runtime."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _version(text: str, name: str) -> tuple[int, ...]:
    match = re.search(rf"^{re.escape(name)}==([0-9.]+)$", text, re.MULTILINE)
    assert match is not None, f"{name} is not pinned in the lock"
    return tuple(int(part) for part in match.group(1).split("."))


def test_czi_runtime_is_optional_pinned_and_lock_backed() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = project["dependency-groups"]["czi-runtime"]
    assert "czifile==2026.10.14" in runtime
    assert any(item.startswith("imagecodecs>=2026.8.16") for item in runtime)
    assert not any(
        "czifile" in item.lower() for item in project["project"]["dependencies"]
    )
    for extra in project["project"].get("optional-dependencies", {}).values():
        assert not any("czifile" in item.lower() for item in extra)

    lock_path = ROOT / "requirements" / "requirements.czi.lock"
    lock = lock_path.read_text()
    assert "czifile==2026.10.14" in lock
    assert _version(lock, "imagecodecs") >= (2026, 8, 16)
    assert "squidpy" not in lock

    environment = (ROOT / "envs" / "environment.czi.yml").read_text()
    match = re.search(
        r"^# requirements.lock sha256: ([0-9a-f]{64})$", environment, re.M
    )
    assert match is not None
    assert match.group(1) == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert "-r ../requirements/requirements.czi.lock" in environment
    assert "python=3.12" in environment


def test_czi_runtime_matches_the_base_numpy_and_tifffile() -> None:
    base = (ROOT / "requirements" / "requirements.lock").read_text()
    lock = (ROOT / "requirements" / "requirements.czi.lock").read_text()
    for name in ("numpy", "tifffile"):
        assert _version(lock, name) == _version(base, name)


def test_base_lock_stays_without_czifile() -> None:
    base = (ROOT / "requirements" / "requirements.lock").read_text()
    assert "czifile" not in base


def test_ci_installs_the_czi_runtime() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    script = (ROOT / "scripts" / "run_ci_checks.sh").read_text()
    for text in (workflow, script):
        assert "requirements/requirements.czi.lock" in text
        assert "MERXEN_CZI_PYTHON" in text
