"""The tracked tree holds no host path and no sample name (CLAUDE.md, privacy).

The checker's ``--tree`` mode runs over every tracked file, as the hooks would
over a commit that added them all:

- generic host paths and donor-like ids (rules 1 and 3) always, so this holds
  on CI, where no private file exists;
- the exact values of the real ``.env`` and site configs (rule 2), and the
  real names of the alias table (rule 4), on a host that has them (skipped
  otherwise, with the reason).

A last check looks for real names glued to a neighbouring word or number,
which the boundaries of rule 4 let through (a nickname directly followed by
a word in a file name, for example).

Failures print the place and the alias or rule, never the matched text (the
checker's contract): pytest output can reach a public CI log.

The scans need the tracked file list, so they skip in a code export (no git
metadata), where test runs add caches beside the exported files.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from merxen import local_config, privacy

from .synthetic import DONOR, DONOR_NUMBER, MOUSE, MOUSE2_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "privacy" / "check_private_text.py"
NOTICE = "privacy: no alias table (.aliases.tsv); real-name check skipped"
SCANNED_RE = re.compile(r"privacy: scanned (\d+) file\(s\)")

# Lines the checker skips (base64 payloads, notebook image outputs), and
# base64 data URIs inside a longer line (an image embedded in an SVG).
BASE64_LINE_RE = re.compile(r'^\s*"?[A-Za-z0-9+/=]{76,}(?:\\n)?"?,?\s*$')
NOTEBOOK_IMAGE_RE = re.compile(r'^\s*"image/[\w.+-]+":')
DATA_URI_RE = re.compile(r"base64,[A-Za-z0-9+/=]+")


def tracked_files() -> list[str]:
    """Tracked files of this checkout; skip the test outside a checkout."""
    try:
        top = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout (a code export): tracked files unknown")
    if Path(top).resolve() != REPO_ROOT:
        pytest.skip("not a git checkout (a code export): tracked files unknown")
    raw = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "-c", "core.quotepath=off", "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [name.decode("utf-8") for name in raw.split(b"\0") if name]


def run_tree(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run ``check_private_text.py --tree`` on this checkout."""
    return subprocess.run(
        [sys.executable, str(CHECKER), "--tree", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def assert_clean(result: subprocess.CompletedProcess[str], files: list[str]) -> None:
    """Exit 0, no hit lines, and every tracked file scanned."""
    scanned = SCANNED_RE.search(result.stderr)
    assert scanned is not None, result.stderr
    assert int(scanned.group(1)) == len(files)
    hits = result.stdout.strip()
    assert result.returncode == 0 and not hits, (
        "host paths or sample names in tracked files (write .env variables and "
        "aliases; docs/local-configuration.md):\n" + hits + "\n" + result.stderr
    )


def test_tracked_files_hold_no_host_paths_or_donor_ids() -> None:
    """Rules 1 and 3, as on CI: no .env, no site configs, no alias table."""
    files = tracked_files()
    result = run_tree(dict(os.environ))
    assert NOTICE in result.stderr
    assert_clean(result, files)


def test_tracked_files_hold_no_real_names(real_aliases: Path) -> None:
    """Rule 4 with the real alias table (host only)."""
    files = tracked_files()
    env = dict(os.environ, **{privacy.ALIAS_FILE_VAR: str(real_aliases)})
    result = run_tree(env)
    assert NOTICE not in result.stderr
    assert_clean(result, files)


def test_tracked_files_hold_no_private_values(
    real_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2 with the real .env and the site configs it names (host only)."""
    files = tracked_files()
    monkeypatch.setenv(local_config.ENV_FILE_VAR, str(real_env_file))
    for variable in privacy.SITE_CONFIGS:
        monkeypatch.delenv(variable, raising=False)
    local_config.clear_cache()
    privacy.clear_cache()
    if not privacy.private_values():
        pytest.skip("the .env holds no private location value")
    result = run_tree(dict(os.environ))
    assert_clean(result, files)


@dataclass(frozen=True)
class GluedForm:
    """A real form searched without word boundaries."""

    needle: str
    pattern: re.Pattern[str]
    label: str
    skip_paths: tuple[str, ...]
    format_after_dot: bool

    def skipped(self, path: str) -> bool:
        """Whether ``path`` is one of the form's skip paths (format files)."""
        return any(fnmatch.fnmatch(path, glob) for glob in self.skip_paths)


def glued_forms(table: privacy.AliasTable) -> list[GluedForm]:
    """Real forms with a letter, from literal, token and display-only rows.

    Bare numbers (``ident`` rows, numeric deny rows) are left out: digits
    inside data are not names. A form that a regex row of the same alias
    matches whole (the MS2 nickname) keeps that row's skip paths and its
    public-format exception (the form directly after a dot).
    """
    regex_rows = [row for row in table.rows if row.match == "regex"]
    forms: list[GluedForm] = []
    for row in table.rows:
        if row.match not in ("literal", "token", "none"):
            continue
        if not re.search(r"[A-Za-z]", row.real):
            continue
        inherited = [
            regex
            for regex in regex_rows
            if regex.alias == row.alias and regex.pattern.fullmatch(row.real)
        ]
        skips = row.skip_paths + tuple(g for r in inherited for g in r.skip_paths)
        forms.append(
            GluedForm(
                needle=row.real.lower(),
                pattern=re.compile(re.escape(row.real), re.IGNORECASE),
                label=row.label,
                skip_paths=skips,
                format_after_dot=bool(inherited),
            )
        )
    return forms


def glued_spans(
    line: str, path: str, forms: list[GluedForm]
) -> list[tuple[int, int, str]]:
    """(start, end, label) of every real form in ``line``, boundaries ignored."""
    if BASE64_LINE_RE.match(line) or NOTEBOOK_IMAGE_RE.match(line):
        return []
    # Blank base64 payloads out (same length, so columns stay right).
    line = DATA_URI_RE.sub(lambda match: " " * len(match.group(0)), line)
    spans: set[tuple[int, int, str]] = set()
    for form in forms:
        if form.skipped(path):
            continue
        for match in form.pattern.finditer(line):
            start = match.start()
            if form.format_after_dot and line[start - 1 : start] == ".":
                continue
            spans.add((start, match.end(), form.label))
    return sorted(spans)


def masked(name: str, forms: list[GluedForm]) -> str:
    """A file name safe to print: names, ids and host roots masked."""
    pieces: list[str] = []
    cursor = 0
    for start, end, label in glued_spans(name, name, forms):
        if start >= cursor:
            pieces += [name[cursor:start], f"[{label}]"]
            cursor = end
    pieces.append(name[cursor:])
    return privacy.safe_text("".join(pieces), name)


def test_tracked_files_hold_no_glued_names(real_aliases: Path) -> None:
    """Real forms glued to other words escape rule 4; find them (host only)."""
    files = tracked_files()
    table = privacy.table()
    assert table is not None
    forms = glued_forms(table)
    assert forms
    found: list[str] = []
    for name in files:
        safe = masked(name, forms)
        if glued_spans(name, name, forms):
            found.append(f"{safe} (file name)")
        target = REPO_ROOT / name
        if target.is_symlink() or not target.is_file():
            continue
        data = target.read_bytes()
        if b"\0" in data[:8000]:
            continue
        text = data.decode("utf-8", errors="replace")
        lower = text.lower()
        present = [form for form in forms if form.needle in lower]
        if not present:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for start, _, label in glued_spans(line, name, present):
                found.append(f"{safe}:{number}:{start + 1}: {label}")
    assert not found, "real names glued to other words:\n" + "\n".join(found)


def test_glued_check_finds_what_rule_4_misses(fake_table: Path) -> None:
    """The glued check on the synthetic table: what it finds and leaves."""
    table = privacy.table()
    assert table is not None
    forms = glued_forms(table)
    glued = f"evidence/REAL_{MOUSE2_NAME.lower()}orig_panel.csv"
    assert privacy.find_names(glued, "docs/plan.md") == []
    assert [label for _, _, label in glued_spans(glued, "docs/plan.md", forms)] == [
        "MS2"
    ]
    assert [
        label for *_, label in glued_spans(f"{MOUSE}x {DONOR}9", "a.py", forms)
    ] == [
        "MS1",
        "HD1",
    ]
    # Public format name after a dot, the format files, bare numbers, payloads.
    assert glued_spans(f"reader.{MOUSE2_NAME.lower()}x", "docs/plan.md", forms) == []
    assert glued_spans(f"{MOUSE2_NAME}orig", "fmt/reader.py", forms) == []
    assert glued_spans(f"x{DONOR_NUMBER}0", "a.py", forms) == []
    payload = "A" * 10 + MOUSE + "B" * 10
    assert (
        glued_spans(f'<image href="data:image/png;base64,{payload}"/>', "a.svg", forms)
        == []
    )
    assert masked(glued, forms) == "evidence/REAL_[MS2]orig_panel.csv"
