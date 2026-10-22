"""Dependency isolation and cache identity for the optional pciSeq runtime."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path


def test_pciseq_runtime_is_optional_pinned_and_lock_backed() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    runtime = project["dependency-groups"]["pciseq-runtime"]
    assert any("pciSeq_3d.git@5e2c0fd" in item for item in runtime)
    assert "pandas>=2.2,<3" in runtime
    assert not any(
        "pciseq" in item.lower() for item in project["project"]["dependencies"]
    )
    lock = root / "requirements" / "requirements.pciseq.lock"
    environment = (root / "envs" / "environment.pciseq.yml").read_text()
    match = re.search(
        r"^# requirements.lock sha256: ([0-9a-f]{64})$", environment, re.M
    )
    assert match is not None
    assert match.group(1) == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert "-r ../requirements/requirements.pciseq.lock" in environment
    dockerfile = (root / "containers" / "Dockerfile.pciseq").read_text()
    assert "-r requirements/requirements.pciseq.lock" in dockerfile
