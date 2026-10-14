"""Tests of the docs rewrite (merxen.privacy.rewrite and its script)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from merxen import privacy
from merxen.local_config import LocalConfigError

from .synthetic import (
    DONOR,
    DONOR_NUMBER,
    LEGACY_VALUE,
    MOUSE,
    MOUSE2_PAIR,
    STORE_VALUE,
    USER,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/privacy/rewrite_text.py"


def _rewrite(text: str) -> tuple[str, list[privacy.Substitution]]:
    return privacy.rewrite(text, paths=True, names=True, quote_brackets=True)


def test_paths_become_variables_longest_first(fake_table: Path, fake_env: Path) -> None:
    text, report = _rewrite(f"see {STORE_VALUE}/runs and {LEGACY_VALUE}/old\n")
    assert text == "see $MERXEN_STORAGE_ROOT/runs and $MERXEN_SSD1_ROOT/old\n"
    assert [item.kind for item in report] == ["path", "path"]


def test_the_refused_prefix_is_never_substituted(fake_env: Path) -> None:
    pairs = privacy.location_substitutions()
    assert "$MERXEN_REFUSED_PATH_PREFIX" not in [name for _, name in pairs]
    assert (LEGACY_VALUE, "$MERXEN_SSD1_ROOT") in pairs


def test_names_become_aliases_with_numbers_and_deny_rows(fake_table: Path) -> None:
    text, report = privacy.rewrite(
        f"xen_{DONOR_NUMBER}_adata of {DONOR} by {USER}",
        paths=False,
        names=True,
        quote_brackets=True,
    )
    assert text == "xen_hd1_adata of HD1 by <user>"
    assert [item.original for item in report] == [DONOR_NUMBER, DONOR, USER]


def test_names_of_any_case_become_aliases(fake_table: Path) -> None:
    # What the scanner refuses, the rewrite replaces (literal rows of any case).
    text, _ = privacy.rewrite(
        f"under {MOUSE2_PAIR.lower()}/", paths=False, names=True, quote_brackets=True
    )
    assert text == "under MS2/"


def test_blockquotes_bracket_substitutions(fake_table: Path, fake_env: Path) -> None:
    text, report = _rewrite(
        f"> the {MOUSE} data live in {STORE_VALUE}/x\n"
        f"> [the {MOUSE} data owner] said\n"
        f"> see {DONOR}(1)\n"
    )
    assert text.split("\n")[:3] == [
        "> the [MS1] data live in [$MERXEN_STORAGE_ROOT]/x",
        "> [the MS1 data owner] said",
        "> see [HD1](1)",
    ]
    assert [item.flags for item in report][-1] == ("link-risk",)


def test_inline_quotations_are_flagged(fake_table: Path) -> None:
    _, report = privacy.rewrite(
        f'He said "the {DONOR} run" and {DONOR} again',
        paths=False,
        names=True,
        quote_brackets=True,
    )
    assert [item.flags for item in report] == [("inline-quote",), ()]


def test_rewrite_of_names_needs_a_table() -> None:
    with pytest.raises(LocalConfigError, match="alias table"):
        privacy.rewrite("x", paths=False, names=True, quote_brackets=True)


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_script_rewrites_in_place_and_writes_a_private_report(
    fake_table: Path, fake_env: Path, tmp_path: Path
) -> None:
    doc = tmp_path / "docs" / "page.md"
    doc.parent.mkdir()
    doc.write_text(f"{DONOR} in {STORE_VALUE}/x\n", encoding="utf-8")
    report = tmp_path / "private" / "report.tsv"
    result = _run("--paths", "--names", "--report", str(report), str(doc), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert doc.read_text(encoding="utf-8") == "HD1 in $MERXEN_STORAGE_ROOT/x\n"
    with report.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows[0] == list(
        ("file", "line", "column", "kind", "original", "replacement", "flags")
    )
    assert [row[3:6] for row in rows[1:]] == [
        ["name", DONOR, "HD1"],
        ["path", STORE_VALUE, "$MERXEN_STORAGE_ROOT"],
    ]
    assert DONOR not in result.stdout


def test_script_refuses_a_report_inside_a_work_tree(
    fake_table: Path, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()
    doc = tmp_path / "page.md"
    doc.write_text(f"{DONOR}\n", encoding="utf-8")
    result = _run(
        "--names", "--report", str(tmp_path / "r.tsv"), str(doc), cwd=tmp_path
    )
    assert result.returncode == 2
    assert doc.read_text(encoding="utf-8") == f"{DONOR}\n"
