"""Tests of merxen.local_config: the .env format, discovery and lookups."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pytest

from merxen import local_config
from merxen.local_config import LocalConfigError

from .synthetic import STORE_VALUE, write_env

VALID_ENV = """\
# a comment with $, `backticks` and 'quotes' is free text
MERXEN_STORAGE_ROOT=/labshare/proj
  MERXEN_EVIDENCE_ROOT=/labshare/proj/evidence_2026

MERXEN_PANEL_CSV="/labshare/My Drive/panel.csv"
MERXEN_EMPTY=
ODD_CHARS=a#b,c:d@e+f=g%h
"""


@pytest.mark.parametrize(
    "line",
    [
        "export MERXEN_X=/labshare/x",
        "MERXEN_X=/labshare/with space",
        "MERXEN_X=$HOME/x",
        "MERXEN_X=`pwd`",
        "MERXEN_X=a\\b",
        "MERXEN_X='quoted'",
        'MERXEN_X="has $dollar"',
        "MERXEN_X=\"has 'single'\"",
        'MERXEN_X="unterminated',
        "MERXEN_X=~/x",
        "MERXEN_X=a;b",
        "MERXEN_X=a|b",
        "MERXEN_X",
        "1BAD=x",
        "MERXEN_X=ok\r",
    ],
)
def test_read_env_file_rejects_lines_bash_would_read_differently(
    tmp_path: Path, line: str
) -> None:
    env = tmp_path / "env"
    env.write_text(f"# header\n{line}\n", encoding="utf-8")
    with pytest.raises(LocalConfigError, match=r"env:2"):
        local_config.read_env_file(env)


def test_read_env_file_error_never_shows_the_value(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text("MERXEN_X=/labshare/secret place\n", encoding="utf-8")
    with pytest.raises(LocalConfigError) as error:
        local_config.read_env_file(env)
    assert "secret" not in str(error.value)
    assert "MERXEN_X" in str(error.value)


def test_read_env_file_reads_the_strict_format(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text(VALID_ENV, encoding="utf-8")
    assert local_config.read_env_file(env) == {
        "MERXEN_STORAGE_ROOT": "/labshare/proj",
        "MERXEN_EVIDENCE_ROOT": "/labshare/proj/evidence_2026",
        "MERXEN_PANEL_CSV": "/labshare/My Drive/panel.csv",
        "MERXEN_EMPTY": "",
        "ODD_CHARS": "a#b,c:d@e+f=g%h",
    }


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bash_sourcing_reads_the_same_values(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text(VALID_ENV, encoding="utf-8")
    parsed = local_config.read_env_file(env)
    script = (
        f'. "{env}"; for key in {" ".join(parsed)}; do printf "%s\\0" "${{!key}}"; done'
    )
    out = subprocess.run(
        ["bash", "-c", script], check=True, capture_output=True, text=True
    ).stdout
    assert out.split("\0")[:-1] == list(parsed.values())


def test_get_prefers_a_non_empty_environment_value(
    fake_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert local_config.get("MERXEN_STORAGE_ROOT") == STORE_VALUE
    monkeypatch.setenv("MERXEN_STORAGE_ROOT", "/labshare/other")
    assert local_config.get("MERXEN_STORAGE_ROOT") == "/labshare/other"
    # An empty value (`export X` of an undefined X) counts as unset.
    monkeypatch.setenv("MERXEN_STORAGE_ROOT", "")
    assert local_config.get("MERXEN_STORAGE_ROOT") == STORE_VALUE
    assert local_config.get("MERXEN_EVIDENCE_ROOT") is None


def test_require_path_names_the_variable_and_never_falls_back(fake_env: Path) -> None:
    assert local_config.require_path("MERXEN_STORAGE_ROOT") == Path(STORE_VALUE)
    with pytest.raises(LocalConfigError) as error:
        local_config.require_path("MERXEN_EVIDENCE_ROOT", "reads recorded runs")
    message = str(error.value)
    assert message.startswith("MERXEN_EVIDENCE_ROOT is not set: add it to .env")
    assert "template .env.example" in message
    assert "reads recorded runs" in message


def test_require_path_does_not_touch_the_file_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MERXEN_EVIDENCE_ROOT", "/labshare/does/not/exist")
    assert local_config.require_path("MERXEN_EVIDENCE_ROOT") == Path(
        "/labshare/does/not/exist"
    )


def test_hermetic_tests_see_no_private_file() -> None:
    assert local_config.env_file() is None
    assert local_config.snapshot() == {
        "MERXEN_ALIAS_FILE": "none",
        "MERXEN_SITE_CONFIG": "/dev/null",
        "MERXEN_DWIGHT_SITE_CONFIG": "/dev/null",
    }


@pytest.mark.parametrize("value", ["none", "/dev/null"])
def test_env_file_absent_on_purpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    write_env(tmp_path / ".env")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv(local_config.ENV_FILE_VAR, value)
    assert local_config.env_file(checkout=tmp_path) is None


def test_env_file_empty_value_means_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = write_env(tmp_path / ".env")
    monkeypatch.setenv(local_config.ENV_FILE_VAR, "")
    assert local_config.env_file(checkout=tmp_path) == env


def test_env_file_named_missing_file_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(local_config.ENV_FILE_VAR, str(tmp_path / "missing.env"))
    with pytest.raises(LocalConfigError, match="MERXEN_ENV_FILE names a file"):
        local_config.env_file()


def _linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A main checkout and a linked worktree, laid out as git does."""
    main = tmp_path / "main"
    gitdir = main / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    return main, worktree


def test_linked_worktree_finds_the_main_checkout_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(local_config.ENV_FILE_VAR)
    main, worktree = _linked_worktree(tmp_path)
    assert local_config.main_checkout(worktree) == main.resolve()
    assert local_config.env_file(checkout=worktree) is None
    env = write_env(main / ".env")
    assert local_config.env_file(checkout=worktree) == main.resolve() / ".env"
    # The worktree's own file wins.
    own = write_env(worktree / ".env")
    assert local_config.env_file(checkout=worktree) == own
    assert env.is_file()


def test_main_checkout_is_none_outside_a_linked_worktree(tmp_path: Path) -> None:
    assert local_config.main_checkout(tmp_path) is None
    (tmp_path / ".git").mkdir()
    assert local_config.main_checkout(tmp_path) is None


def test_add_location_argument_defaults_from_the_variable(fake_env: Path) -> None:
    parser = argparse.ArgumentParser()
    local_config.add_location_argument(
        parser, "--store", "MERXEN_STORAGE_ROOT", "storage root"
    )
    assert parser.parse_args([]).store == Path(STORE_VALUE)
    assert parser.parse_args(["--store", "/x"]).store == Path("/x")
    assert "$MERXEN_STORAGE_ROOT" in parser.format_help()


def test_add_location_argument_is_required_when_unset() -> None:
    parser = argparse.ArgumentParser()
    local_config.add_location_argument(
        parser, "--evidence", "MERXEN_EVIDENCE_ROOT", "evidence archive"
    )
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["--evidence", "e"]).evidence == Path("e")


def test_snapshot_merges_environment_and_file(
    fake_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERXEN_EVIDENCE_ROOT", "/labshare/evidence")
    merged = local_config.snapshot()
    assert merged["MERXEN_STORAGE_ROOT"] == STORE_VALUE
    assert merged["MERXEN_EVIDENCE_ROOT"] == "/labshare/evidence"
    assert merged["MERXEN_OUTPUT_ROOT"] == "/path/to/output"


def test_known_has_no_runtime_knob() -> None:
    knobs = {"MERXEN_TMPDIR", "MERXEN_ANNOTATION_MAP_N_PROCESSORS", "MERXEN_CI_VENV"}
    assert not knobs & set(local_config.KNOWN)
    assert len(set(local_config.KNOWN)) == len(local_config.KNOWN)
    assert all(name.startswith("MERXEN_") for name in local_config.KNOWN)
