"""Synthetic private files and invented names for the privacy tests.

Every "real" string here is invented, and the ones that the tracked-tree
rules would flag (donor-like ids, host roots) are built from parts, so this
file itself passes the scanner.
"""

from __future__ import annotations

from pathlib import Path

# Invented real forms (never real samples).
DONOR = "P" + "4321"  # -> HD1
DONOR_NUMBER = "4321"
DONOR_IF = DONOR + "_IF"  # -> HD1_IF
BULK_COLUMN = "S9_" + DONOR_NUMBER  # -> bulk_HD1
RUN = "209901010000_" + DONOR + "-2_INSTR9"  # -> <HD1-merscope-run>
SECOND_DONOR = "P" + "8642"  # -> HD2
MOUSE = "zq9"  # MS1 pair id and nickname (lower case, like a real one)
MOUSE_VARIANT = "Zebra09"
MOUSE2_PAIR = "QqBrain_FF_X_rep2_R1"  # MS2 pair id (mixed case, literal)
MOUSE2_NAME = "QQZ5"  # MS2 nickname; ".qqz5" is a public format name
USER = "jdoe"
INSTRUMENT = "INSTR9"
HOST_ROOT = "/" + "srv"
FORMAT_GLOB = "fmt/*"

TABLE_ROWS = [
    ("HD1", "human_paired_donor", "pair_id", "token", DONOR, "", ""),
    ("HD1", "human_paired_donor", "number", "ident", DONOR_NUMBER, "", ""),
    (
        "HD1",
        "human_paired_donor",
        "bulk_column",
        "literal",
        BULK_COLUMN,
        "",
        "bulk_HD1",
    ),
    (
        "HD1",
        "human_paired_donor",
        "merscope_run",
        "literal",
        RUN,
        "",
        "<HD1-merscope-run>",
    ),
    ("HD1", "human_paired_donor", "variant", "literal", DONOR + "-2", "", ""),
    ("HD2", "human_paired_donor", "pair_id", "token", SECOND_DONOR, "", ""),
    ("HD1_IF", "human_if_sample", "pair_id", "token", DONOR_IF, "", ""),
    ("MS1", "mouse_section", "pair_id", "token", MOUSE, "", ""),
    ("MS1", "mouse_section", "name", "token", MOUSE, "", ""),
    ("MS1", "mouse_section", "variant", "token", MOUSE_VARIANT, "", ""),
    ("MS2", "mouse_section", "pair_id", "literal", MOUSE2_PAIR, "", ""),
    ("MS2", "mouse_section", "name", "none", MOUSE2_NAME, "", ""),
    (
        "MS2",
        "mouse_section",
        "variant",
        "regex",
        r"(?i)(?<![.A-Za-z0-9])qqz5(?![A-Za-z0-9])",
        FORMAT_GLOB,
        "",
    ),
    ("-", "instrument", "deny", "token", INSTRUMENT, "", "<instrument>"),
    ("-", "user_name", "deny", "token", USER, "", "<user>"),
]

# Values outside the host roots, so only the exact-value rule (2) sees them.
STORE_VALUE = "/labshare/proj/store"
LEGACY_VALUE = "/labshare/legacy-disk"


def write_table(path: Path, rows: list[tuple[str, ...]] | None = None) -> Path:
    """Write an alias table with the synthetic rows (or the given ones)."""
    lines = ["# synthetic alias table", ""]
    lines += ["\t".join(row) for row in (TABLE_ROWS if rows is None else rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_env(path: Path, extra: str = "") -> Path:
    """Write a synthetic .env with two location values."""
    path.write_text(
        "# synthetic .env\n"
        f"MERXEN_STORAGE_ROOT={STORE_VALUE}\n"
        f"MERXEN_SSD1_ROOT={LEGACY_VALUE}\n"
        f"MERXEN_REFUSED_PATH_PREFIX={LEGACY_VALUE}\n"
        "MERXEN_OUTPUT_ROOT=/path/to/output\n" + extra,
        encoding="utf-8",
    )
    return path
