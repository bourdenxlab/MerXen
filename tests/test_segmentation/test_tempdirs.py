"""Tests for the large-temporary-file location of segmentation steps."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen.segmentation.tempdirs import MERXEN_TMPDIR_ENV, large_temp_root


def test_explicit_root_wins_and_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit directory beats $MERXEN_TMPDIR and the cwd, and is created."""
    monkeypatch.setenv(MERXEN_TMPDIR_ENV, str(tmp_path / "env"))
    monkeypatch.chdir(tmp_path)
    explicit = tmp_path / "explicit" / "nested"
    assert large_temp_root(explicit) == explicit
    assert explicit.is_dir()


def test_env_root_beats_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """$MERXEN_TMPDIR is used when no explicit directory is given."""
    monkeypatch.setenv(MERXEN_TMPDIR_ENV, str(tmp_path / "env"))
    monkeypatch.chdir(tmp_path)
    assert large_temp_root() == tmp_path / "env"
    assert (tmp_path / "env").is_dir()


def test_default_is_the_working_directory_not_the_system_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without overrides, temporary files go to the cwd (a task's work dir)."""
    monkeypatch.delenv(MERXEN_TMPDIR_ENV, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "system_tmp"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert large_temp_root() == work


def test_blank_env_value_falls_back_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty $MERXEN_TMPDIR is treated as unset."""
    monkeypatch.setenv(MERXEN_TMPDIR_ENV, "")
    monkeypatch.chdir(tmp_path)
    assert large_temp_root() == tmp_path
