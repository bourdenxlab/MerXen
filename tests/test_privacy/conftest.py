"""Fixtures enabling the synthetic private files (``synthetic.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen import local_config, privacy

from .synthetic import write_env, write_table


@pytest.fixture
def fake_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Enable the synthetic alias table for one test."""
    path = write_table(tmp_path / "aliases.tsv")
    monkeypatch.setenv(privacy.ALIAS_FILE_VAR, str(path))
    privacy.clear_cache()
    return path


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Enable a synthetic .env for one test."""
    path = write_env(tmp_path / "env")
    monkeypatch.setenv(local_config.ENV_FILE_VAR, str(path))
    local_config.clear_cache()
    privacy.clear_cache()
    return path
