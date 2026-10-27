"""Machine-specific locations, read from the gitignored ``.env`` file.

Every absolute location of a machine (storage roots, reference stores, the
evidence archive, legacy disks) lives outside the tracked tree, in ``.env``
(template ``.env.example``; docs/local-configuration.md). This module is the
one reader of that file for Python code, tests and notebooks. It is stdlib
only because the commit hooks run it under the system ``python3`` and must
parse the same file.

Rules:

- The process environment wins over ``.env``; an empty value counts as unset.
- A missing value raises :class:`LocalConfigError` naming the variable. There
  is never a built-in host default.
- Nothing here touches the file system of a value (no ``exists()`` on the
  caller's behalf). Only the private files themselves are looked up.

Discovery of ``.env`` (identical in the hooks and the shell scripts):

1. ``$MERXEN_ENV_FILE``: ``none`` or the null device mean "deliberately
   absent", an empty value means unset, any other value must name an existing
   file.
2. The checkout's own ``.env``.
3. When the checkout is a linked git worktree, the main checkout's ``.env``
   (read from the ``.git`` link file; no git call).
4. Otherwise absent.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Final

ENV_FILE_VAR: Final = "MERXEN_ENV_FILE"
ENV_FILE_NAME: Final = ".env"
#: Values of a ``MERXEN_*_FILE`` / site-config variable meaning "absent on
#: purpose". An empty value is not among them: it means unset (discovery goes
#: on), because ``export X`` of an undefined variable exports an empty value.
ABSENT: Final = frozenset({"none", "/dev/null"})

#: Every location variable, in the order of the location section of
#: ``.env.example`` (a test keeps the two equal). Runtime knobs such as
#: ``MERXEN_TMPDIR`` or ``MERXEN_ANNOTATION_MAP_N_PROCESSORS`` are not
#: locations: they stay plain ``os.environ`` reads and must never be added
#: here or to ``.env``.
KNOWN: Final[tuple[str, ...]] = (
    # Pointers to the other private files (code exports and worktrees).
    "MERXEN_ALIAS_FILE",
    "MERXEN_SITE_CONFIG",
    "MERXEN_DWIGHT_SITE_CONFIG",
    # Project storage.
    "MERXEN_STORAGE_ROOT",
    "MERXEN_ANNOTATION_DEV_DIR",
    "MERXEN_EVIDENCE_ROOT",
    "MERXEN_HEAVY_LOCK",
    "MERXEN_RESULTS_ROOT",
    "MERXEN_REFERENCE_STORE",
    "MERXEN_REFERENCE_STORE_LARGE",
    "MERXEN_PUBLIC_VALIDATION_ROOT",
    "MERXEN_REFERENCES_DIR",
    "MERXEN_ABC_ATLAS_DIR",
    "MERXEN_MOUSE_GENE_TABLE",
    "MERXEN_WMB_SELFMAP_TEST_CELLS",
    "MERXEN_LEGACY_SNAPSHOT_DIR",
    "MERXEN_TASK_TMP_DIR",
    "MERXEN_MERSCOPE_RAW_ROOT",
    # Data owned by others.
    "MERXEN_MS1_OWNER_ROOT",
    "MERXEN_MS1_CLUSTERED_H5AD",
    "MERXEN_WMB_MAPPING_STATS",
    "MERXEN_MS2_RAW_ROOT",
    "MERXEN_MS2_PUBLISHED_RESULTS",
    "MERXEN_HM_RAW_ROOT",
    # Legacy disks (MERXEN_SSD1_ROOT before MERXEN_REFUSED_PATH_PREFIX: they
    # hold the same value and the docs rewrite substitutes the first).
    "MERXEN_SSD1_ROOT",
    "MERXEN_SSD2_ROOT",
    "MERXEN_SS3_ROOT",
    "MERXEN_REFUSED_PATH_PREFIX",
    "MERXEN_HUMAN_GENE_TABLE",
    "MERXEN_LEGACY_REFERENCE_STORE",
    "MERXEN_MAPMYCELLS_CACHE_DIR",
    "MERXEN_WHB_EXPRESSION_DIR",
    "MERXEN_WHB_METADATA_DIR",
    "MERXEN_WHB_REGION_PRECOMPUTE_SOURCE",
    "MERXEN_LEGACY_ABC_ATLAS_DIR",
    "MERXEN_LEGACY_WMB_GENE_CSV",
    # Notebooks.
    "MERXEN_NOTEBOOK_RESULTS_ROOT",
    "MERXEN_NOTEBOOK_MAPMYCELLS_CACHE_DIR",
    "MERXEN_PANEL_CSV",
    # Optional inputs of scripts and slow tests.
    "MERXEN_GENE_ID_FALLBACK_CSV",
    "MERXEN_MOUSE_GENE_ID_FALLBACK_CSV",
    "MERXEN_WHB_TAXONOMY_DIR",
    "MERXEN_WMB_TAXONOMY_DIR",
    "MERXEN_SEAAD_TERM_CSV",
    "MERXEN_MANUAL_REGISTRATION_TEST_DIR",
    "MERXEN_CZI_PYTHON",
)

_KEY_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Unquoted values: no whitespace, quotes, shell expansion or control
# characters, so `. .env` in bash and this parser read the same string.
_UNQUOTED_RE: Final = re.compile(r"[^\s\"'`$\\;&|<>()~]*")
_QUOTED_RE: Final = re.compile(r"\"([^\"'`$\\]*)\"")

_FILE_CACHE: dict[Path, dict[str, str]] = {}
_DISCOVERED: dict[str, Path | None] = {}


class LocalConfigError(RuntimeError):
    """A private location is missing, malformed or names a missing file."""


def checkout_root() -> Path | None:
    """Return the repository checkout this package runs from, if any.

    Returns:
        The directory holding ``pyproject.toml`` two levels above
        ``src/merxen``, or None for an installed (non-source) package.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def main_checkout(checkout: Path) -> Path | None:
    """Return the main checkout of a linked git worktree.

    Reads ``<checkout>/.git`` (a ``gitdir:`` link file in a linked worktree),
    then that git dir's ``commondir``; the main checkout is the parent of the
    common git dir. No subprocess and no git call.

    Args:
        checkout: A checkout directory.

    Returns:
        The main checkout, or None when ``checkout`` is not a linked worktree.
    """
    link = checkout / ".git"
    if not link.is_file():
        return None
    match = re.match(r"gitdir:\s*(.+)", link.read_text(encoding="utf-8").strip())
    if match is None:
        return None
    gitdir = Path(match.group(1).strip())
    if not gitdir.is_absolute():
        gitdir = checkout / gitdir
    commondir_file = gitdir / "commondir"
    if not commondir_file.is_file():
        return None
    common = Path(commondir_file.read_text(encoding="utf-8").strip())
    if not common.is_absolute():
        common = gitdir / common
    return common.resolve().parent


def discover_file(relative: str, *, checkout: Path | None = None) -> Path | None:
    """Find a private file in the checkout, else in the main checkout.

    Args:
        relative: Path of the private file relative to a checkout root.
        checkout: Checkout to start from (default: :func:`checkout_root`).

    Returns:
        The first existing candidate, or None.
    """
    start = checkout if checkout is not None else checkout_root()
    if start is None:
        return None
    candidate = start / relative
    if candidate.is_file():
        return candidate
    main = main_checkout(start)
    if main is not None and (main / relative).is_file():
        return main / relative
    return None


def named_file(variable: str, value: str) -> Path | None:
    """Resolve an explicit private-file variable.

    Args:
        variable: Variable name (for the error message).
        value: Its non-empty value.

    Returns:
        The named file, or None when the value means "absent on purpose".

    Raises:
        LocalConfigError: If the value names a file that does not exist.
    """
    if value in ABSENT:
        return None
    path = Path(value)
    if not path.is_file():
        raise LocalConfigError(
            f"{variable} names a file that does not exist; fix it or set "
            f"{variable}=none (docs/local-configuration.md)"
        )
    return path


def env_file(*, checkout: Path | None = None) -> Path | None:
    """Return the ``.env`` file in use (discovery in the module docstring).

    Args:
        checkout: Checkout to discover from (default: :func:`checkout_root`).

    Returns:
        The file, or None when it is absent or disabled.

    Raises:
        LocalConfigError: If ``$MERXEN_ENV_FILE`` names a missing file.
    """
    value = os.environ.get(ENV_FILE_VAR, "")
    if value:
        return named_file(ENV_FILE_VAR, value)
    if checkout is not None:
        return discover_file(ENV_FILE_NAME, checkout=checkout)
    if "default" not in _DISCOVERED:
        _DISCOVERED["default"] = discover_file(ENV_FILE_NAME)
    return _DISCOVERED["default"]


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file in the strict shared format.

    ``KEY=VALUE`` per line; ``#`` comment lines and blank lines are ignored;
    a value with spaces is double-quoted; no ``$``, backticks, backslashes,
    single quotes, ``~`` or ``export``, so bash ``. .env`` reads the same
    values. Error messages name the line and key, never the value.

    Args:
        path: The file to read.

    Returns:
        Mapping of keys to values (an empty value stays an empty string).

    Raises:
        LocalConfigError: On a line that breaks the format.
    """
    values: dict[str, str] = {}
    # Bytes, not read_text(): universal newlines would hide a CR that bash keeps.
    text = path.read_bytes().decode("utf-8")
    for number, raw in enumerate(text.split("\n"), 1):
        where = f"{path.name}:{number}"
        if "\r" in raw:
            raise LocalConfigError(f"{where}: CRLF line ending (use LF)")
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if key.startswith("export ") or key == "export":
            raise LocalConfigError(f"{where}: no 'export' (plain KEY=VALUE)")
        if not separator or not _KEY_RE.fullmatch(key):
            raise LocalConfigError(f"{where}: expected KEY=VALUE")
        if value.startswith('"'):
            quoted = _QUOTED_RE.fullmatch(value)
            if quoted is None:
                raise LocalConfigError(
                    f"{where}: {key}: a quoted value may not contain quotes, "
                    "'$', backticks or backslashes"
                )
            value = quoted.group(1)
        elif not _UNQUOTED_RE.fullmatch(value):
            raise LocalConfigError(
                f"{where}: {key}: quote a value with spaces; no quotes, '$', "
                "backticks, backslashes, '~' or shell operators"
            )
        values[key] = value
    return values


def _file_values() -> dict[str, str]:
    path = env_file()
    if path is None:
        return {}
    if path not in _FILE_CACHE:
        _FILE_CACHE[path] = read_env_file(path)
    return _FILE_CACHE[path]


def get(name: str) -> str | None:
    """Return a setting: process environment (non-empty), else ``.env``.

    Args:
        name: Variable name.

    Returns:
        The value, or None when unset or empty in both.
    """
    value = os.environ.get(name)
    if value:
        return value
    return _file_values().get(name) or None


def path(name: str) -> Path | None:
    """Return a location setting as a path (see :func:`get`).

    Args:
        name: Variable name.

    Returns:
        The path, or None when unset.
    """
    value = get(name)
    return Path(value) if value else None


def require_path(name: str, purpose: str = "") -> Path:
    """Return a location setting, or raise naming the variable.

    Args:
        name: Variable name.
        purpose: Optional text saying what needs it (appended to the error).

    Returns:
        The path (never checked for existence).

    Raises:
        LocalConfigError: If the variable is unset. There is no fallback.
    """
    value = path(name)
    if value is None:
        suffix = f" ({purpose})" if purpose else ""
        raise LocalConfigError(
            f"{name} is not set: add it to .env (template .env.example) or "
            f"export it{suffix}"
        )
    return value


def add_location_argument(
    parser: argparse.ArgumentParser, flag: str, name: str, help: str
) -> None:
    """Add a path option whose default comes from a location variable.

    The option is required when the variable is unset, so a script never runs
    on a guessed location. Call it while building the parser, never at import.

    Args:
        parser: The parser to extend.
        flag: Option string, e.g. ``--evidence-root``.
        name: Location variable that provides the default.
        help: Help text; ``(default: $NAME)`` is appended.
    """
    default = path(name)
    parser.add_argument(
        flag,
        type=Path,
        default=default,
        required=default is None,
        help=f"{help} (default: ${name})",
    )


def snapshot() -> dict[str, str]:
    """Return every set location and ``.env`` key with its merged value.

    Returns:
        Mapping of the names in :data:`KNOWN` and the keys of ``.env`` to
        their values (environment first), unset names left out.
    """
    names = list(KNOWN) + [key for key in _file_values() if key not in KNOWN]
    merged: dict[str, str] = {}
    for name in names:
        value = get(name)
        if value is not None:
            merged[name] = value
    return merged


def clear_cache() -> None:
    """Forget parsed and discovered ``.env`` files (tests, long-lived processes)."""
    _FILE_CACHE.clear()
    _DISCOVERED.clear()
