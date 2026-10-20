"""Sample aliases and the host-path / sample-name scanner (stdlib only).

Tracked code, data, tests and docs name samples by neutral aliases (HD1-HD4,
HD1_IF, HD3_IF, HM1-HM4, MS1, MS2). The gitignored alias table
(``.aliases.tsv``; template ``.aliases.example.tsv``) maps each alias to the
real forms of the sample. Code that meets a real name in data translates it at
the boundary with :func:`to_alias`; a path to existing data built from an
alias uses :func:`to_real` with the form that path family was named by; an
output label that used to be a hard-coded real name goes through
:func:`display`. See docs/local-configuration.md.

The same table drives the privacy hooks (``scripts/privacy``): together with
the generic host-path and donor-id rules below, it keeps host paths and
sample names out of commits, commit messages, branch names and PR text.

Table discovery: ``$MERXEN_ALIAS_FILE`` (environment, then ``.env``);
``none`` or the null device disable the table on purpose; otherwise the
``.aliases.tsv`` beside the ``.env`` in use, then the checkout's, then the
main checkout's (linked worktree). Translation without a table is an error
unless the table is disabled on purpose, because identity on real data turns
membership tests silently false.
"""

from __future__ import annotations

import bisect
import fnmatch
import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from merxen import local_config
from merxen.local_config import LocalConfigError

logger = logging.getLogger(__name__)

ALIAS_FILE_VAR: Final = "MERXEN_ALIAS_FILE"
ALIAS_FILE_NAME: Final = ".aliases.tsv"
#: Nextflow site configs: variable -> default path in a checkout.
SITE_CONFIGS: Final = {
    "MERXEN_SITE_CONFIG": "workflows/conf/site.config",
    "MERXEN_DWIGHT_SITE_CONFIG": "workflows/conf/dwight.site.config",
}

#: Roots of host data and mounts. The rule is built from these bare words, so
#: this source holds no literal host path.
HOST_ROOTS: Final = (
    "srv",
    "media",
    "home",
    "mnt",
    "Users",
    "nfsdata",
    "scratch",
    "Volumes",
)
_ROOTS_RE: Final = r"/(?:" + "|".join(HOST_ROOTS) + r")(?![\w.-])"
# Not after a word character, '.', '~' or '/' (relative paths, home-relative
# paths), except in a file:// URL. '-' may precede (shell defaults ${X:-...}).
_UNIX_HOST_RE: Final = (
    r"(?<![\w.~/])"
    + _ROOTS_RE
    + r"|(?<=file://)"
    + _ROOTS_RE
    + r"|(?<![\w.~/])/data(?=[/,])"
)
# Session scratch directories of the coding assistant, and a Windows user
# directory; split so that this source does not match its own rule.
_TMP_SESSION_RE: Final = "/tmp/" + "claude-"
_WINDOWS_USER_RE: Final = r"\b[A-Za-z]:\\" + r"Users\\"
HOST_PATH_RE: Final = re.compile(
    "|".join((_UNIX_HOST_RE, _TMP_SESSION_RE, _WINDOWS_USER_RE))
)
#: Synthetic donor-like ids of test fixtures and docs (never real samples).
SYNTHETIC_DONOR_IDS: Final = frozenset({"P0001", "P0002", "P9999"})
GENERIC_DONOR_RE: Final = re.compile(r"(?<![A-Za-z0-9])P\d{4}(?![A-Za-z0-9])")
#: Values that are never private, whatever file holds them (rule 2).
PUBLIC_VALUE_PREFIXES: Final = ("/path/to/", "/usr/", "/proc/", "/dev/")
#: Tracked paths that must never hold a private file, whatever the content.
PRIVATE_FILE_PATTERNS: Final = (
    ".env",
    "*/.env",
    ".aliases.tsv",
    "*/.aliases.tsv",
    "workflows/conf/site.config",
    "workflows/conf/*.site.config",
)

_BASE64_LINE_RE: Final = re.compile(r'^\s*"?[A-Za-z0-9+/=]{76,}(?:\\n)?"?,?\s*$')
_NOTEBOOK_IMAGE_RE: Final = re.compile(r'^\s*"image/[\w.+-]+":')
_PRINTABLE_RUN_RE: Final = re.compile(rb"[\x20-\x7e\t]{4,}")
_QUOTED_STRING_RE: Final = re.compile(r"(\w+)?\s*=?\s*([\"'])(.*?)\2")

MATCH_KINDS: Final = ("literal", "token", "regex", "ident", "none")
#: Forms that may repeat for one alias; every other (alias, form) is unique.
REPEATABLE_FORMS: Final = frozenset({"variant", "deny"})
DENY_ALIAS: Final = "-"
DENY_PLACEHOLDER: Final = "<redacted>"

AliasState = Literal["loaded", "disabled", "missing"]


@dataclass(frozen=True)
class Allowance:
    """One explicit exception to the privacy rules (what may stay tracked)."""

    what: str
    where: str
    why: str


#: Everything the rules let stay in tracked files (docs/local-configuration.md
#: lists the same; the user's answers of 2026-10-14 to Q4-Q6). Every item is
#: outside the rules by construction: none of these needs a code path that
#: lets a hit through.
EXCEPTIONS: Final[tuple[Allowance, ...]] = (
    Allowance(
        "OS tool and system paths: /usr/bin/proseg, /usr/local/bin/proseg, "
        "/usr/bin/time, /proc, /dev/null, /opt/... inside container images",
        "anywhere",
        "tool and OS locations, identical on every machine (not host roots)",
    ),
    Allowance(
        "the generic GPU lock file /tmp/merxen-dwight-gpu.lock",
        "Nextflow config, launch scripts, tests and docs",
        "a temp-root location shared by every launch; names a host, not a path layout",
    ),
    Allowance(
        "the host profile name dwight and its file names "
        "(conf/dwight.config, conf/dwight.annotation.config, ...)",
        "anywhere",
        "a profile name, not a path or a sample",
    ),
    Allowance(
        "the disk nicknames SSD1, SSD2 and SS3 in prose",
        "anywhere",
        "nicknames, not paths (the paths are .env variables)",
    ),
    Allowance(
        "the copyright holder of LICENSE and README, and the project's GitHub "
        "organisation URL and CI badge",
        "LICENSE, README.md, docs",
        "public project identity",
    ),
    Allowance(
        "placeholder paths under /path/to/",
        "templates, tests, docs",
        "the .env.example convention; not a host root, excluded from rule 2",
    ),
    Allowance(
        "the public .vzg2 archive format name (and the format files listed in "
        "the MS2 row's skip_paths)",
        "the alias table's skip_paths",
        "a public file-format name, not the dataset nickname",
    ),
    Allowance(
        "the synthetic donor ids " + ", ".join(sorted(SYNTHETIC_DONOR_IDS)),
        "tests, docs/pipeline.md",
        "fixture ids, never real samples",
    ),
)


@dataclass(frozen=True)
class AliasRow:
    """One row of the alias table.

    Attributes:
        alias: Neutral alias, or ``-`` for a deny-only row.
        kind: What the real string is (``human_paired_donor``, ``user_name``, ...).
        form: Which real form the row holds (``pair_id``, ``name``, ``number``,
            ``bulk_column``, ``merscope_run``, ``xenium_output``, ``variant``,
            ``deny``, ...).
        match: ``literal``, ``token``, ``regex``, ``ident`` or ``none``.
        real: The real string (a regex for ``match=regex``).
        skip_paths: Globs of tracked paths where the row is not checked.
        placeholder: Text written instead of the alias, if any.
    """

    alias: str
    kind: str
    form: str
    match: str
    real: str
    skip_paths: tuple[str, ...] = ()
    placeholder: str = ""
    pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)
    scan_pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        pattern = _row_pattern(self.match, self.real)
        object.__setattr__(self, "pattern", pattern)
        # The scanner and the docs rewrite refuse every spelling of a literal
        # real form (a lower-cased run or pair id); translating data with
        # to_alias stays exact.
        scan = re.compile(pattern.pattern, re.I) if self.match == "literal" else pattern
        object.__setattr__(self, "scan_pattern", scan)

    @property
    def is_deny(self) -> bool:
        """Whether the row only denies a string (no alias)."""
        return self.alias == DENY_ALIAS or self.form == "deny"

    @property
    def label(self) -> str:
        """What reports print for a hit of this row (never the real text)."""
        if self.placeholder:
            return self.placeholder
        return DENY_PLACEHOLDER if self.is_deny else self.alias

    def skipped(self, path: str | None) -> bool:
        """Whether the row is not checked in ``path``."""
        return bool(path) and any(
            fnmatch.fnmatch(str(path), glob) for glob in self.skip_paths
        )


@dataclass(frozen=True)
class AliasTable:
    """A loaded alias table with rows in matching priority order."""

    path: Path
    rows: tuple[AliasRow, ...]

    @property
    def aliases(self) -> tuple[str, ...]:
        """Distinct aliases, longest first (``HD1_IF`` before ``HD1``)."""
        names = {row.alias for row in self.rows if not row.is_deny}
        return tuple(sorted(names, key=lambda name: (-len(name), name)))

    def real_form(self, alias: str, form: str) -> AliasRow | None:
        """Return the row holding ``form`` of ``alias``, if any."""
        for row in self.rows:
            if row.alias == alias and row.form == form:
                return row
        return None


@dataclass(frozen=True)
class Hit:
    """One finding of the scanner.

    Attributes:
        line: 1-based line number (0 for a file name or a branch name).
        column: 1-based column.
        rule: ``host-path``, ``env-value``, ``donor-id``, ``name`` or
            ``private-file``.
        label: Safe description: the root word and variable, the alias or the
            placeholder. Never the real text.
        text: The matched real text (printed only on explicit request).
    """

    line: int
    column: int
    rule: str
    label: str
    text: str


@dataclass(frozen=True)
class Substitution:
    """One replacement made by :func:`rewrite`.

    Attributes:
        line: 1-based line number in the input.
        column: 1-based column in the input line.
        kind: ``path`` or ``name``.
        original: The replaced real text.
        replacement: What was written instead.
        flags: Manual-pass flags (``inline-quote``, ``link-risk``).
    """

    line: int
    column: int
    kind: str
    original: str
    replacement: str
    flags: tuple[str, ...] = ()


_TABLE_CACHE: dict[Path, AliasTable] = {}
_RESOLVED: dict[tuple[str, str], tuple[Path | None, AliasState]] = {}
_STATE: dict[str, object] = {}


def _row_pattern(match: str, real: str) -> re.Pattern[str]:
    if match == "none":
        return re.compile(r"(?!x)x")
    if match == "regex":
        return re.compile(real)
    core = re.escape(real)
    if match == "literal":
        return re.compile(core)
    if match == "token":
        return re.compile(r"(?<![A-Za-z0-9])" + core + r"(?![A-Za-z0-9])", re.I)
    if match == "ident":
        return re.compile(
            r"(?:(?<=[_-])" + core + r"(?![A-Za-z0-9])"
            r"|(?<![A-Za-z0-9])" + core + r"(?=[_-]))"
        )
    raise LocalConfigError(f"unknown match kind {match!r}")


def _priority(row: AliasRow) -> tuple[int, int, str]:
    # Literal rows first, longest real string first (a run id wins over its
    # donor), then tokens, regexes and bare numbers.
    order = {"literal": 0, "token": 1, "regex": 2, "ident": 3, "none": 4}
    return (order[row.match], -len(row.real), row.real)


def load_table(path: Path) -> AliasTable:
    """Read and validate an alias table.

    Args:
        path: The TSV file (7 columns; ``#`` comment lines).

    Returns:
        The table, rows in matching priority order.

    Raises:
        LocalConfigError: On a malformed row, an unknown match kind, an
            unedited template placeholder or a repeated (alias, form). Messages
            name the line, never the real string.
    """
    rows: list[AliasRow] = []
    seen: dict[tuple[str, str], int] = {}
    placeholders: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        where = f"{path.name}:{number}"
        cells = line.split("\t")
        if len(cells) < 5 or len(cells) > 7:
            raise LocalConfigError(f"{where}: expected 5-7 tab-separated columns")
        cells += [""] * (7 - len(cells))
        alias, kind, form, match, real, skip, placeholder = (c.strip() for c in cells)
        if match not in MATCH_KINDS:
            raise LocalConfigError(f"{where}: unknown match kind {match!r}")
        if not alias or not form or not real:
            raise LocalConfigError(f"{where}: alias, form and real are required")
        if real.startswith("<") and real.endswith(">"):
            raise LocalConfigError(
                f"{where}: template placeholder in the real column (copy the "
                "real values, or set MERXEN_ALIAS_FILE=none)"
            )
        if (alias == DENY_ALIAS) != (form == "deny"):
            raise LocalConfigError(f"{where}: deny rows have alias '-' and form deny")
        key = (alias, form)
        if form not in REPEATABLE_FORMS:
            if key in seen:
                raise LocalConfigError(
                    f"{where}: {alias} already has a {form} row (line "
                    f"{seen[key]}); forms are unique per alias"
                )
            seen[key] = number
        if placeholder and alias != DENY_ALIAS:
            if placeholders.get(placeholder, alias) != alias:
                raise LocalConfigError(f"{where}: placeholder used by two aliases")
            placeholders[placeholder] = alias
        try:
            row = AliasRow(
                alias=alias,
                kind=kind,
                form=form,
                match=match,
                real=real,
                skip_paths=tuple(g.strip() for g in skip.split(",") if g.strip()),
                placeholder=placeholder,
            )
        except re.error as error:
            raise LocalConfigError(f"{where}: invalid regex ({error})") from error
        rows.append(row)
    rows.sort(key=_priority)
    return AliasTable(path=path, rows=tuple(rows))


def alias_file() -> Path | None:
    """Return the alias table in use (discovery in the module docstring).

    Returns:
        The table file, or None when it is disabled or not found.

    Raises:
        LocalConfigError: If ``$MERXEN_ALIAS_FILE`` names a missing file, or
            ``.env`` is malformed.
    """
    return _resolve_alias_file()[0]


def _resolve_alias_file() -> tuple[Path | None, AliasState]:
    # Cached per value of the two variables, so a changed environment is
    # seen at once; a changed .env needs clear_cache().
    key = (
        os.environ.get(ALIAS_FILE_VAR, ""),
        os.environ.get(local_config.ENV_FILE_VAR, ""),
    )
    if key not in _RESOLVED:
        _RESOLVED[key] = _discover_alias_file()
    return _RESOLVED[key]


def _discover_alias_file() -> tuple[Path | None, AliasState]:
    value = local_config.get(ALIAS_FILE_VAR)
    if value:
        named = local_config.named_file(ALIAS_FILE_VAR, value)
        return (named, "loaded") if named else (None, "disabled")
    env = local_config.env_file()
    if env is not None and (env.parent / ALIAS_FILE_NAME).is_file():
        return env.parent / ALIAS_FILE_NAME, "loaded"
    found = local_config.discover_file(ALIAS_FILE_NAME)
    return (found, "loaded") if found else (None, "missing")


def alias_state() -> AliasState:
    """Return ``loaded``, ``disabled`` (on purpose) or ``missing``."""
    return _resolve_alias_file()[1]


def table() -> AliasTable | None:
    """Return the alias table in use, or None when disabled or missing."""
    found = alias_file()
    if found is None:
        return None
    if found not in _TABLE_CACHE:
        _TABLE_CACHE[found] = load_table(found)
    return _TABLE_CACHE[found]


def has_aliases() -> bool:
    """Whether an alias table is in use."""
    return table() is not None


def _translation_table() -> AliasTable | None:
    found, state = _resolve_alias_file()
    if state == "missing":
        raise LocalConfigError(
            "no alias table: set MERXEN_ALIAS_FILE (template .aliases.example.tsv) "
            "or MERXEN_ALIAS_FILE=none to run on aliases only"
        )
    return table() if found is not None else None


def _case_like(template: str, value: str) -> str:
    if template.isupper():
        return value.upper()
    if template.islower():
        return value.lower()
    return value


def _context(text: str, start: int, end: int) -> str:
    """The identifier around a match (letters, digits, ``_`` and ``-``)."""
    left = start
    while left > 0 and (text[left - 1].isalnum() or text[left - 1] in "_-"):
        left -= 1
    right = end
    while right < len(text) and (text[right].isalnum() or text[right] in "_-"):
        right += 1
    return text[left:right]


def _recase(text: str, match: re.Match[str], canonical: str, value: str) -> str:
    """Case of a replacement: ``value`` re-cased like the matched text.

    Inside a lower-case identifier the replacement is lower case
    (``mouse_<x>_500``); a match equal to its canonical spelling gets the
    canonical replacement (a standalone pair id, ``<x>_MERSCOPE``); otherwise
    the case pattern of the match (upper, lower, or canonical when mixed).
    """
    matched = match.group(0)
    context = _context(text, match.start(), match.end())
    if context != matched and context.islower():
        return value.lower()
    if matched == canonical:
        return value
    return _case_like(matched, value)


def _replacement(row: AliasRow, text: str, match: re.Match[str]) -> str:
    if row.placeholder:
        return row.placeholder
    if row.is_deny:
        return DENY_PLACEHOLDER
    if row.match in ("token", "regex"):
        return _recase(text, match, row.real, row.alias)
    if row.match == "ident":
        return _case_like(_context(text, match.start(), match.end()), row.alias)
    return row.alias


def _spans(
    text: str, rows: Iterable[AliasRow], path: str | None, *, scan: bool = False
) -> list[tuple[re.Match[str], AliasRow]]:
    """Non-overlapping row matches, chosen greedily in priority order.

    ``scan`` uses the case-insensitive literal patterns (scanner, rewrite).
    """
    starts: list[int] = []
    chosen: list[tuple[int, int, re.Match[str], AliasRow]] = []
    lower = text.lower()
    for row in rows:
        # Cheap prefilter: every non-regex row needs its real string in the text.
        if row.match != "regex" and row.real.lower() not in lower:
            continue
        if row.skipped(path):
            continue
        pattern = row.scan_pattern if scan else row.pattern
        for match in pattern.finditer(text):
            start, end = match.span()
            if start == end:
                continue
            index = bisect.bisect_left(starts, start)
            if index > 0 and chosen[index - 1][1] > start:
                continue
            if index < len(chosen) and chosen[index][0] < end:
                continue
            starts.insert(index, start)
            chosen.insert(index, (start, end, match, row))
    return [(match, row) for _, _, match, row in chosen]


def _substitute(
    text: str, spans: Sequence[tuple[re.Match[str], AliasRow]]
) -> tuple[str, list[tuple[int, int, str, AliasRow]]]:
    pieces: list[str] = []
    done: list[tuple[int, int, str, AliasRow]] = []
    cursor = 0
    for match, row in spans:
        new = _replacement(row, text, match)
        pieces.append(text[cursor : match.start()])
        pieces.append(new)
        done.append((match.start(), match.end(), new, row))
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), done


def _data_rows(table_: AliasTable) -> list[AliasRow]:
    # Runtime data: literal, token and regex rows of aliases. Bare donor
    # numbers (ident) and deny rows are for the scanner and the docs rewrite.
    return [
        row
        for row in table_.rows
        if not row.is_deny and row.match in ("literal", "token", "regex")
    ]


def to_alias(text: str, path: str | None = None) -> str:
    """Translate real sample names in ``text`` to their aliases.

    Applies literal, token and regex rows (longest real string first); never
    bare donor numbers or deny rows, so ids in data are not rewritten by
    accident.

    Args:
        text: A value read from outside the repository (pair id, directory
            name, key of a recorded file).
        path: Tracked path the text comes from (honours ``skip_paths``).

    Returns:
        The aliased text (unchanged when the table is disabled on purpose).

    Raises:
        LocalConfigError: If no table is found and it is not disabled.
    """
    table_ = _translation_table()
    if table_ is None:
        return text
    return _substitute(text, _spans(text, _data_rows(table_), path))[0]


def _alias_regex(aliases: Sequence[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + alternatives + r")(?![A-Za-z0-9])", re.I)


def _realify(table_: AliasTable, text: str, forms: Sequence[str]) -> str:
    placeholders = {
        row.placeholder: row.real
        for row in table_.rows
        if row.placeholder and not row.is_deny
    }
    for placeholder in sorted(placeholders, key=len, reverse=True):
        text = text.replace(placeholder, placeholders[placeholder])
    aliases = table_.aliases
    if not aliases:
        return text
    by_upper = {alias.upper(): alias for alias in aliases}

    def replace(match: re.Match[str]) -> str:
        alias = by_upper[match.group(0).upper()]
        for form in forms:
            row = table_.real_form(alias, form)
            if row is not None:
                break
        else:
            raise LocalConfigError(
                f"alias {alias} has no {'/'.join(forms)} form in the alias table"
            )
        if row.match == "literal":
            return row.real
        return _recase(match.string, match, alias, row.real)

    return _alias_regex(aliases).sub(replace, text)


def to_real(text: str, form: str = "pair_id") -> str:
    """Translate aliases in ``text`` to one real form.

    Literal real forms are returned verbatim; token forms take the case
    pattern of the alias as written (``hd1_c`` -> lower-case real id).

    Args:
        text: Text holding aliases (a path segment, a key, a label).
        form: The real form to return: ``pair_id`` for samplesheet-keyed
            results, ``name`` for the evidence archive's nickname directories,
            ``bulk_column``, ``merscope_run``, ...

    Returns:
        The text with real names (unchanged when the table is disabled).

    Raises:
        LocalConfigError: If no table is found and it is not disabled, or an
            alias in ``text`` has no row of ``form``.
    """
    table_ = _translation_table()
    if table_ is None:
        return text
    return _realify(table_, text, (form,))


def display(text: str) -> str:
    """Return ``text`` with aliases shown by their display name.

    For output labels only: the ``name`` form, falling back to ``pair_id``.
    Without a table it returns ``text`` unchanged (one logged warning when the
    table is missing rather than disabled).

    Args:
        text: A label holding aliases.

    Returns:
        The label as written before the cleanup on a host with the table.
    """
    found, state = _resolve_alias_file()
    if found is None:
        if state == "missing" and not _STATE.get("display_warned"):
            _STATE["display_warned"] = True
            logger.warning(
                "no alias table: output labels carry aliases "
                "(set MERXEN_ALIAS_FILE or MERXEN_ALIAS_FILE=none)"
            )
        return text
    table_ = table()
    assert table_ is not None
    return _realify(table_, text, ("name", "pair_id"))


def find_host_paths(text: str) -> list[re.Match[str]]:
    """Return matches of the generic host-path rule (rule 1)."""
    return list(HOST_PATH_RE.finditer(text))


def find_generic_donor_ids(text: str) -> list[re.Match[str]]:
    """Return ``P`` + 4 digit ids other than the synthetic fixture ids (rule 3)."""
    return [
        match
        for match in GENERIC_DONOR_RE.finditer(text)
        if match.group(0) not in SYNTHETIC_DONOR_IDS
    ]


def name_matches(
    text: str, path: str | None = None
) -> list[tuple[re.Match[str], AliasRow]]:
    """Return every real-name match with its row (rule 4; [] without a table).

    Args:
        text: Text to search.
        path: Tracked path of the text (honours ``skip_paths``).

    Returns:
        Non-overlapping (match, row) pairs in text order.
    """
    table_ = table()
    if table_ is None:
        return []
    return _spans(text, table_.rows, path, scan=True)


def find_names(text: str, path: str | None = None) -> list[re.Match[str]]:
    """Return every real-name match (rule 4; empty without a table)."""
    return [match for match, _ in name_matches(text, path)]


def is_private_file(path: str) -> bool:
    """Whether a tracked path is one of the private files (refused by name)."""
    return any(fnmatch.fnmatch(path, glob) for glob in PRIVATE_FILE_PATTERNS)


def _site_config_files() -> list[Path]:
    # Per variable (section "discovery" of docs/local-configuration.md): an
    # explicit value wins ('none' or the null device = absent on purpose),
    # otherwise the checkout's file, then the main checkout's.
    found: dict[Path, None] = {}
    for variable, relative in SITE_CONFIGS.items():
        value = local_config.get(variable)
        config = (
            local_config.named_file(variable, value)
            if value
            else local_config.discover_file(relative)
        )
        if config is not None:
            found[config.resolve()] = None
    return list(found)


def _is_private_value(value: str) -> bool:
    if not value.startswith("/") or value.startswith(PUBLIC_VALUE_PREFIXES):
        return False
    return len([part for part in value.split("/") if part]) >= 2


def private_values() -> list[tuple[str, str]]:
    """Return the exact private values of rule 2 with a safe label.

    Every absolute value of ``.env`` (environment first) and every quoted
    absolute string of the site configs, except values under
    :data:`PUBLIC_VALUE_PREFIXES` and single-component paths. Longest first.

    Returns:
        (value, label) pairs; the label is ``$NAME`` or ``<file>:<param>``.
    """
    values: dict[str, str] = {}
    for name, value in local_config.snapshot().items():
        if _is_private_value(value):
            values.setdefault(value.rstrip("/") or value, f"${name}")
    for config in _site_config_files():
        text = config.read_text(encoding="utf-8", errors="replace")
        for match in _QUOTED_STRING_RE.finditer(text):
            value = match.group(3)
            if value.startswith("file://"):
                value = value[len("file://") :]
            if _is_private_value(value):
                param = match.group(1) or "value"
                values.setdefault(value.rstrip("/"), f"{config.name}:{param}")
    return sorted(values.items(), key=lambda item: (-len(item[0]), item[0]))


def _value_spans(
    text: str, values: Sequence[tuple[str, str]]
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for value, label in values:
        start = text.find(value)
        while start != -1:
            end = start + len(value)
            after = text[end : end + 1]
            boundary = not after or not (after.isalnum() or after in "_.-")
            overlaps = any(s < end and start < e for s, e, _ in spans)
            if boundary and not overlaps:
                spans.append((start, end, label))
            start = text.find(value, start + 1)
    return sorted(spans)


def _skip_line(line: str) -> bool:
    return bool(_BASE64_LINE_RE.match(line) or _NOTEBOOK_IMAGE_RE.match(line))


def _root_word(matched: str) -> str:
    if ":" in matched:
        return "Users"
    words = re.findall(r"[A-Za-z]+", matched)
    return words[0] if words else "?"


def scan_line(
    line: str,
    *,
    path: str | None = None,
    values: Sequence[tuple[str, str]] = (),
    number: int = 1,
) -> list[Hit]:
    """Apply the four rules to one line of text.

    Args:
        line: The text (one line; base64 and notebook image lines are skipped).
        path: Tracked path of the line (``skip_paths``).
        values: Exact private values (:func:`private_values`).
        number: Line number for the hits.

    Returns:
        Hits in column order; a name hit replaces a donor-id hit on the same
        span, and a host path names the ``.env`` variable whose value matched.
    """
    if _skip_line(line):
        return []
    hits: list[Hit] = []
    value_spans = _value_spans(line, values)
    value_at = {start: (end, label) for start, end, label in value_spans}
    host_starts: set[int] = set()
    for match in find_host_paths(line):
        label = f"root={_root_word(match.group(0))}"
        if match.start() in value_at:
            label += f" {value_at[match.start()][1]}"
        host_starts.add(match.start())
        hits.append(Hit(number, match.start() + 1, "host-path", label, match.group(0)))
    for start, end, label in value_spans:
        if start not in host_starts:
            hits.append(Hit(number, start + 1, "env-value", label, line[start:end]))
    names = name_matches(line, path)
    name_spans = [(match.start(), match.end()) for match, _ in names]
    for match, row in names:
        hits.append(Hit(number, match.start() + 1, "name", row.label, match.group(0)))
    for match in find_generic_donor_ids(line):
        start, end = match.span()
        if not any(s <= start and end <= e for s, e in name_spans):
            hits.append(
                Hit(number, match.start() + 1, "donor-id", "P####", match.group(0))
            )
    return sorted(hits, key=lambda hit: (hit.column, hit.rule))


def scan_text(
    text: str,
    *,
    path: str | None = None,
    values: Sequence[tuple[str, str]] = (),
    first_line: int = 1,
) -> list[Hit]:
    """Apply the four rules to every line of ``text`` (see :func:`scan_line`)."""
    hits: list[Hit] = []
    for offset, line in enumerate(text.splitlines()):
        hits.extend(
            scan_line(line, path=path, values=values, number=first_line + offset)
        )
    return hits


def scan_bytes(
    data: bytes, *, path: str | None = None, values: Sequence[tuple[str, str]] = ()
) -> list[Hit]:
    """Search a binary file for exact private values and names (rules 2, 4).

    Only printable runs of at least 4 characters are searched (as ``strings``
    does), so random bytes do not produce hits; rules 1 and 3 are not applied.
    Hit "lines" are byte offsets.

    Args:
        data: The file content.
        path: Tracked path (``skip_paths``).
        values: Exact private values (:func:`private_values`).

    Returns:
        Hits with ``line`` = 0 and ``column`` = byte offset + 1.
    """
    hits: list[Hit] = []
    for run in _PRINTABLE_RUN_RE.finditer(data):
        text = run.group(0).decode("ascii")
        base = run.start()
        for start, end, label in _value_spans(text, values):
            hits.append(Hit(0, base + start + 1, "env-value", label, text[start:end]))
        for match, row in name_matches(text, path):
            hits.append(
                Hit(0, base + match.start() + 1, "name", row.label, match.group(0))
            )
    return hits


def safe_text(text: str, path: str | None = None) -> str:
    """Return ``text`` with names, donor ids and host roots masked for printing.

    Used for file and branch names in reports, so a report never repeats a
    real name or path.

    Args:
        text: A path or ref name.
        path: Tracked path (``skip_paths``).

    Returns:
        The masked text (``[HD1]``, ``[P####]``, ``[root]``).
    """
    spans: list[tuple[int, int, str]] = []
    for match, row in name_matches(text, path):
        spans.append((match.start(), match.end(), f"[{row.label}]"))
    for match in find_generic_donor_ids(text):
        spans.append((match.start(), match.end(), "[P####]"))
    for match in find_host_paths(text):
        spans.append((match.start(), match.end(), "[host-root]"))
    pieces: list[str] = []
    cursor = 0
    for start, end, label in sorted(spans):
        if start < cursor:
            continue
        pieces += [text[cursor:start], label]
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def location_substitutions(
    snapshot: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Return (value, ``$NAME``) pairs for the docs rewrite, in priority order.

    Every absolute location value of :data:`local_config.KNOWN` except the
    refused prefix (it shares its value with ``MERXEN_SSD1_ROOT``); longest
    value first, ties broken by the order of ``KNOWN``.

    Args:
        snapshot: Merged settings (default: :func:`local_config.snapshot`).

    Returns:
        Pairs of (value without trailing slash, ``$NAME``).
    """
    merged = local_config.snapshot() if snapshot is None else snapshot
    order = {name: index for index, name in enumerate(local_config.KNOWN)}
    pairs: dict[str, tuple[int, str]] = {}
    for name in local_config.KNOWN:
        if name == "MERXEN_REFUSED_PATH_PREFIX":
            continue
        value = (merged.get(name) or "").rstrip("/")
        if not _is_private_value(value):
            continue
        if value not in pairs or order[name] < pairs[value][0]:
            pairs[value] = (order[name], name)
    ranked = sorted(pairs.items(), key=lambda item: (-len(item[0]), item[1][0]))
    return [(value, f"${name}") for value, (_, name) in ranked]


def _bracket_spans(line: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\[[^\[\]]*\]", line)]


def _in_quotes(line: str, position: int) -> bool:
    before = line[:position]
    return before.count('"') % 2 == 1 or before.count("“") > before.count("”")


def rewrite(
    text: str,
    *,
    paths: bool,
    names: bool,
    quote_brackets: bool,
    path: str | None = None,
    locations: Sequence[tuple[str, str]] | None = None,
) -> tuple[str, list[Substitution]]:
    """Rewrite host paths to ``.env`` variables and names to aliases (docs).

    Paths: every location value (:func:`location_substitutions`), longest
    first, followed by a path boundary, becomes ``$NAME``. Names: every table
    row, bare donor numbers and deny rows included (deny rows write their
    placeholder). On a blockquote line (``>``) with ``quote_brackets``, every
    substitution is bracketed (``[HD1]``) unless it already sits inside an
    editorial bracket. Substitutions inside inline double quotes are flagged
    ``inline-quote``; a created ``](`` or ``][`` is flagged ``link-risk``.

    Args:
        text: The document.
        paths: Substitute location values.
        names: Substitute real names.
        quote_brackets: Bracket substitutions on blockquote lines.
        path: Tracked path of the document (``skip_paths``).
        locations: (value, ``$NAME``) pairs (default: from ``.env``).

    Returns:
        The new text and every substitution made (holding the real text: keep
        the report private).

    Raises:
        LocalConfigError: If ``names`` is requested and no table is in use.
    """
    table_ = None
    if names:
        table_ = table()
        if table_ is None:
            raise LocalConfigError(
                "rewrite of names needs the alias table (MERXEN_ALIAS_FILE)"
            )
    pairs = list(locations) if locations is not None else []
    if paths and locations is None:
        pairs = location_substitutions()
    out_lines: list[str] = []
    report: list[Substitution] = []
    for number, line in enumerate(text.split("\n"), 1):
        spans: list[tuple[int, int, str, str]] = []
        if paths:
            for start, end, label in _value_spans(line, pairs):
                spans.append((start, end, label, "path"))
        if table_ is not None:
            taken = [(s, e) for s, e, _, _ in spans]
            for match, row in _spans(line, table_.rows, path, scan=True):
                start, end = match.span()
                if any(s < end and start < e for s, e in taken):
                    continue
                spans.append((start, end, _replacement(row, line, match), "name"))
        spans.sort()
        is_quote = line.lstrip().startswith(">")
        brackets = _bracket_spans(line)
        pieces: list[str] = []
        cursor = 0
        for start, end, new, kind in spans:
            flags: list[str] = []
            inside_bracket = any(s < start and end < e for s, e in brackets)
            if quote_brackets and is_quote and not inside_bracket:
                new = f"[{new}]"
                following = line[end : end + 1]
                if following in ("(", "["):
                    flags.append("link-risk")
            if not is_quote and _in_quotes(line, start):
                flags.append("inline-quote")
            pieces += [line[cursor:start], new]
            cursor = end
            report.append(
                Substitution(
                    number, start + 1, kind, line[start:end], new, tuple(flags)
                )
            )
        pieces.append(line[cursor:])
        out_lines.append("".join(pieces))
    return "\n".join(out_lines), report


def clear_cache() -> None:
    """Forget loaded tables, discovery results and the one-time warning."""
    _TABLE_CACHE.clear()
    _RESOLVED.clear()
    _STATE.clear()
