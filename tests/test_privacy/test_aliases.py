"""Tests of the alias engine of merxen.privacy (synthetic table only)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from merxen import local_config, privacy
from merxen.local_config import LocalConfigError

from .synthetic import (
    BULK_COLUMN,
    DONOR,
    DONOR_IF,
    DONOR_NUMBER,
    INSTRUMENT,
    MOUSE,
    MOUSE2_NAME,
    MOUSE2_PAIR,
    MOUSE_VARIANT,
    RUN,
    TABLE_ROWS,
    USER,
    write_env,
    write_table,
)


@pytest.mark.parametrize(
    ("real", "alias"),
    [
        (DONOR, "HD1"),
        (DONOR.lower() + "_c", "hd1_c"),
        (DONOR + "_MERSCOPE", "HD1_MERSCOPE"),
        ("test_" + DONOR.lower() + "_fixture", "test_hd1_fixture"),
        (DONOR_IF, "HD1_IF"),
        (DONOR_IF + "_MERSCOPE", "HD1_IF_MERSCOPE"),
        (MOUSE, "MS1"),
        (MOUSE + "=labels.csv", "MS1=labels.csv"),
        (MOUSE + "_MERSCOPE", "MS1_MERSCOPE"),
        (MOUSE.upper() + "_SYMBOLS", "MS1_SYMBOLS"),
        ("mouse_" + MOUSE + "_500", "mouse_ms1_500"),
        ("--" + MOUSE + "-labels", "--ms1-labels"),
        (MOUSE_VARIANT, "MS1"),
        (MOUSE2_PAIR, "MS2"),
        ("mouse_qqz5_815", "mouse_ms2_815"),
        ("QQZ5 run", "MS2 run"),
        (RUN, "<HD1-merscope-run>"),
        (BULK_COLUMN, "bulk_HD1"),
        (DONOR + "-2", "HD1"),
    ],
)
def test_to_alias_keeps_the_case_pattern(
    fake_table: Path, real: str, alias: str
) -> None:
    assert privacy.to_alias(real) == alias


@pytest.mark.parametrize(
    "text",
    [
        "archive.qqz5 format",  # the public format name, not the nickname
        "xen_" + DONOR_NUMBER + "_adata",  # bare numbers: never on runtime data
        f"{USER} and {INSTRUMENT}",  # deny rows: hook only
        "zq99 and azq9",  # token boundaries
    ],
)
def test_to_alias_leaves_formats_numbers_and_deny_rows(
    fake_table: Path, text: str
) -> None:
    assert privacy.to_alias(text) == text


@pytest.mark.parametrize(
    ("text", "form", "real"),
    [
        ("HD1", "pair_id", DONOR),
        ("hd1_c", "pair_id", DONOR.lower() + "_c"),
        ("HD1_MERSCOPE", "pair_id", DONOR + "_MERSCOPE"),
        ("HD1_IF", "pair_id", DONOR_IF),
        ("MS1", "pair_id", MOUSE),
        ("MS1_MERSCOPE", "pair_id", MOUSE + "_MERSCOPE"),
        ("mouse_ms1_500", "pair_id", "mouse_" + MOUSE + "_500"),
        ("MS2", "pair_id", MOUSE2_PAIR),
        ("MS2", "name", MOUSE2_NAME),
        ("mouse_ms2_815", "name", "mouse_" + MOUSE2_NAME.lower() + "_815"),
        ("in_ms1,in_ms2", "name", f"in_{MOUSE},in_{MOUSE2_NAME.lower()}"),
        ("HD1", "bulk_column", BULK_COLUMN),
        ("bulk_HD1", "pair_id", BULK_COLUMN),
        ("HD1", "merscope_run", RUN),
        ("HD10 and AHD1", "pair_id", "HD10 and AHD1"),
    ],
)
def test_to_real_returns_the_requested_form(
    fake_table: Path, text: str, form: str, real: str
) -> None:
    assert privacy.to_real(text, form=form) == real


def test_to_real_round_trips_ids_read_from_data(fake_table: Path) -> None:
    for real in [DONOR, DONOR + "_XENIUM", MOUSE, MOUSE + "_MERSCOPE", MOUSE2_PAIR]:
        assert privacy.to_real(privacy.to_alias(real)) == real


def test_to_real_refuses_an_alias_without_the_form(fake_table: Path) -> None:
    with pytest.raises(LocalConfigError, match="HD1 has no name form"):
        privacy.to_real("HD1", form="name")


def test_display_prefers_the_name_then_the_pair_id(fake_table: Path) -> None:
    assert privacy.display("HD1 vs MS2 vs mo2_ms1") == (
        f"{DONOR} vs {MOUSE2_NAME} vs mo2_{MOUSE}"
    )


def test_disabled_table_is_identity() -> None:
    # The hermetic fixture sets MERXEN_ALIAS_FILE=none.
    assert privacy.alias_state() == "disabled"
    assert privacy.to_alias(DONOR) == DONOR
    assert privacy.to_real("HD1") == "HD1"
    assert privacy.display("HD1") == "HD1"
    assert privacy.find_names(DONOR) == []
    assert not privacy.has_aliases()


def test_missing_table_refuses_translation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(privacy.ALIAS_FILE_VAR)
    monkeypatch.setattr(local_config, "checkout_root", lambda: None)
    privacy.clear_cache()
    assert privacy.alias_state() == "missing"
    with pytest.raises(LocalConfigError, match="no alias table"):
        privacy.to_alias(DONOR)
    with pytest.raises(LocalConfigError, match="MERXEN_ALIAS_FILE=none"):
        privacy.to_real("HD1")
    with caplog.at_level(logging.WARNING, logger="merxen.privacy"):
        assert privacy.display("HD1") == "HD1"
        assert privacy.display("HD2") == "HD2"
    assert len([r for r in caplog.records if "no alias table" in r.message]) == 1


def test_named_missing_table_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(privacy.ALIAS_FILE_VAR, str(tmp_path / "missing.tsv"))
    privacy.clear_cache()
    with pytest.raises(LocalConfigError, match="MERXEN_ALIAS_FILE names a file"):
        privacy.alias_file()


def test_table_found_beside_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = write_env(tmp_path / ".env")
    table = write_table(tmp_path / privacy.ALIAS_FILE_NAME)
    monkeypatch.setenv(local_config.ENV_FILE_VAR, str(env))
    monkeypatch.delenv(privacy.ALIAS_FILE_VAR)
    privacy.clear_cache()
    assert privacy.alias_file() == table
    assert privacy.to_alias(DONOR) == "HD1"


def test_table_named_in_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = write_table(tmp_path / "elsewhere.tsv")
    env = write_env(tmp_path / "env", extra=f"MERXEN_ALIAS_FILE={table}\n")
    monkeypatch.setenv(local_config.ENV_FILE_VAR, str(env))
    monkeypatch.delenv(privacy.ALIAS_FILE_VAR)
    privacy.clear_cache()
    assert privacy.alias_file() == table
    # The environment wins over .env.
    monkeypatch.setenv(privacy.ALIAS_FILE_VAR, "none")
    assert privacy.alias_state() == "disabled"


def _rows_with(*extra: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [*TABLE_ROWS, *extra]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            _rows_with(
                ("HD1", "human_paired_donor", "pair_id", "token", "QX1", "", "")
            ),
            "already has a pair_id row",
        ),
        (_rows_with(("HD5", "k", "pair_id", "fuzzy", "QX1", "", "")), "unknown match"),
        (_rows_with(("HD5", "k", "pair_id", "token", "<real pair id>")), "placeholder"),
        (_rows_with(("-", "user_name", "pair_id", "token", "QX1")), "deny rows"),
        (_rows_with(("HD5", "k", "deny", "token", "QX1")), "deny rows"),
        (_rows_with(("HD5", "k", "pair_id")), "5-7 tab-separated"),
        (_rows_with(("HD5", "k", "pair_id", "regex", "([")), "invalid regex"),
    ],
)
def test_load_table_rejects_malformed_rows(
    tmp_path: Path, rows: list[tuple[str, ...]], message: str
) -> None:
    path = write_table(tmp_path / "aliases.tsv", rows)
    with pytest.raises(LocalConfigError, match=message):
        privacy.load_table(path)


def test_load_table_allows_repeated_variants_and_deny_rows(tmp_path: Path) -> None:
    rows = _rows_with(
        ("HD1", "human_paired_donor", "variant", "literal", "QX-1"),
        ("-", "user_name", "deny", "token", "qxuser", "", "<user>"),
    )
    table = privacy.load_table(write_table(tmp_path / "aliases.tsv", rows))
    assert table.aliases[0] == "HD1_IF"  # longest alias first
    kinds = [row.match for row in table.rows]
    assert kinds == sorted(
        kinds, key=["literal", "token", "regex", "ident", "none"].index
    )


def test_find_names_honours_skip_paths(fake_table: Path) -> None:
    assert privacy.find_names("the qqz5 reader", path="fmt/reader.py") == []
    assert [m.group(0) for m in privacy.find_names("the qqz5 reader")] == ["qqz5"]
    assert privacy.find_names("the .qqz5 reader") == []


def test_find_names_prefers_the_longest_literal(fake_table: Path) -> None:
    hits = privacy.name_matches(f"run {RUN} done")
    assert [(match.group(0), row.label) for match, row in hits] == [
        (RUN, "<HD1-merscope-run>")
    ]


def test_find_names_includes_numbers_and_deny_rows(fake_table: Path) -> None:
    labels = [
        row.label
        for _, row in privacy.name_matches(
            f"xen_{DONOR_NUMBER}_x by {USER} 1{DONOR_NUMBER}"
        )
    ]
    assert labels == ["HD1", "<user>"]
