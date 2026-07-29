"""Fixtures for workflow configuration smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def dwight_config_text() -> str:
    """Return the workstation execution profile as source text."""
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "workflows" / "conf" / "dwight.config").read_text()


@pytest.fixture
def combined_config_text(dwight_config_text: str) -> str:
    """Return base scientific defaults plus the default workstation profile."""
    repo_root = Path(__file__).resolve().parents[2]
    base_text = (repo_root / "workflows" / "nextflow.config").read_text()
    return f"{base_text}\n{dwight_config_text}"
