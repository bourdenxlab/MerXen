#!/usr/bin/env python3
"""Rewrite host paths to ``.env`` variables and sample names to aliases.

The deterministic part of the one-time docs cleanup (docs/local-configuration.md):

- ``--paths``: every value of a location variable of the resolved ``.env``,
  longest first and followed by a path boundary, becomes ``$NAME``. Ties of
  equal values go to the earlier variable of ``merxen.local_config.KNOWN``;
  the launch guard's refused prefix is never substituted (it shares its value
  with ``$MERXEN_SSD1_ROOT``).
- ``--names``: every row of the alias table (bare donor numbers and deny rows
  included) becomes its alias or placeholder, keeping the case pattern.
- On a blockquote line every substitution is bracketed (``[HD1]``), except
  inside an existing editorial bracket; a substitution inside inline double
  quotes is flagged ``inline-quote``, and a created ``](`` / ``][`` is
  flagged ``link-risk``, for the manual pass.

Files are rewritten in place. The substitution report (TSV: file, line,
column, kind, original, replacement, flags) holds the real text, so it must be
written outside every git working tree; the tool refuses otherwise. Standard
output lists only counts and the flagged places.

Usage:
    python3 scripts/privacy/rewrite_text.py --paths --names \\
        --report "$PRIVATE_DIR/rewrite_report.tsv" docs/some-page.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from merxen import privacy  # noqa: E402
from merxen.local_config import LocalConfigError  # noqa: E402

REPORT_COLUMNS = (
    "file",
    "line",
    "column",
    "kind",
    "original",
    "replacement",
    "flags",
)


def inside_work_tree(path: Path) -> bool:
    """Whether ``path`` lies inside any git working tree."""
    return any((parent / ".git").exists() for parent in [path, *path.parents])


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path, help="files to rewrite")
    parser.add_argument("--paths", action="store_true", help="paths -> $NAME")
    parser.add_argument("--names", action="store_true", help="names -> aliases")
    parser.add_argument(
        "--no-quote-brackets",
        action="store_true",
        help="do not bracket substitutions on blockquote lines",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="substitution report (TSV; refused inside a git working tree)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write the report, not the files"
    )
    args = parser.parse_args(argv)
    if not (args.paths or args.names):
        parser.error("choose --paths and/or --names")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the files and write the private report."""
    args = parse_args(argv)
    report = args.report.resolve()
    if inside_work_tree(report.parent):
        print(
            "rewrite_text: refusing a report inside a git working tree (it holds "
            "the real text); write it to a private location",
            file=sys.stderr,
        )
        return 2
    rows: list[list[str]] = []
    flagged: list[str] = []
    try:
        for file in args.files:
            text = file.read_text(encoding="utf-8")
            new_text, substitutions = privacy.rewrite(
                text,
                paths=args.paths,
                names=args.names,
                quote_brackets=not args.no_quote_brackets,
                path=str(file),
            )
            for item in substitutions:
                rows.append(
                    [
                        str(file),
                        str(item.line),
                        str(item.column),
                        item.kind,
                        item.original,
                        item.replacement,
                        ",".join(item.flags),
                    ]
                )
                if item.flags:
                    where = privacy.safe_text(str(file))
                    flagged.append(f"{where}:{item.line}: {','.join(item.flags)}")
            if new_text != text and not args.dry_run:
                file.write_text(new_text, encoding="utf-8")
            where = privacy.safe_text(str(file))
            print(f"{where}: {len(substitutions)} substitution(s)")
    except LocalConfigError as error:
        print(f"rewrite_text: {error}", file=sys.stderr)
        return 2
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(REPORT_COLUMNS)
        writer.writerows(rows)
    report.chmod(0o600)
    for line in flagged:
        print(f"flag: {line}")
    print(f"rewrite_text: {len(rows)} substitution(s), {len(flagged)} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
