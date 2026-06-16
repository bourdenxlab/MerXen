#!/usr/bin/env python
"""Remove stale per-shape assignment tables and downstream outputs.

The script is intentionally dry-run by default. Pass ``--apply`` after
reviewing the planned removals.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import spatialdata as sd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from merxen.enrichment.enrich import _prune_zarr_consolidated_metadata  # noqa: E402
from merxen.path_utils import remove_path  # noqa: E402

DERIVED_TABLE_PREFIX = "table_"
PRESERVED_TABLES = {"table", "table_original"}

PAIR_DOWNSTREAM_DIRS = (
    "comparison",
    "visualization",
    "clustering_squidpy",
    "mapmycells",
    "reseg",
    "original_seg",
)
ALIGNMENT_OUTPUT_DIRS = ("alignment", "alignment_qc")
PLATFORM_DOWNSTREAM_DIRS = (
    "enrichment",
    "qc",
    "reseg",
    "original_seg",
)


@dataclass(frozen=True)
class Sample:
    """One samplesheet row plus the platforms that row will process."""

    pair_id: str
    platforms: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Delete stale derived assignment tables from enriched zarrs and "
            "remove published outputs from enrichment onward."
        )
    )
    parser.add_argument(
        "--samplesheet",
        type=Path,
        required=True,
        help="Samplesheet CSV that lists the samples to clean.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./results"),
        help="Pipeline output directory. Defaults to ./results.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove files/tables. Omit for a dry run.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only remove derived tables from latest zarrs.",
    )
    parser.add_argument(
        "--outputs-only",
        action="store_true",
        help="Only remove published downstream output directories.",
    )
    parser.add_argument(
        "--delete-alignment-output",
        action="store_true",
        help=(
            "Also remove pair-level alignment/alignment_qc directories. "
            "By default, alignment outputs are preserved."
        ),
    )
    return parser.parse_args()


def active_platforms(row: dict[str, str]) -> tuple[str, ...]:
    """Return active platforms from a samplesheet row."""
    mode = (row.get("analysis_mode") or "paired").strip().lower()
    if mode in {"paired", "pair", "both"}:
        return ("MERSCOPE", "XENIUM")
    if mode in {"merscope", "merfish"}:
        return ("MERSCOPE",)
    if mode == "xenium":
        return ("XENIUM",)
    raise ValueError(f"Unknown analysis_mode={mode!r} for pair {row.get('pair_id')!r}")


def read_samples(samplesheet: Path) -> list[Sample]:
    """Read the samplesheet rows targeted by cleanup."""
    with samplesheet.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples: list[Sample] = []
    for row in rows:
        pair_id = (row.get("pair_id") or "").strip()
        if not pair_id:
            raise ValueError(f"Found samplesheet row with missing pair_id: {row}")
        samples.append(Sample(pair_id=pair_id, platforms=active_platforms(row)))
    return samples


def is_derived_assignment_table(table_key: str) -> bool:
    """Return true for derived per-shape assignment table names."""
    return (
        table_key.startswith(DERIVED_TABLE_PREFIX) and table_key not in PRESERVED_TABLES
    )


def action(label: str, path: Path | str, *, apply: bool) -> None:
    """Print a planned or executed filesystem action."""
    prefix = "REMOVE" if apply else "DRY-RUN remove"
    print(f"{prefix} {label}: {path}")


def remove_derived_tables(zarr_path: Path, *, apply: bool) -> None:
    """Remove derived per-shape tables from an enriched SpatialData zarr."""
    if not zarr_path.exists() and not zarr_path.is_symlink():
        print(f"SKIP missing latest zarr: {zarr_path}")
        return

    sdata = sd.read_zarr(zarr_path)
    table_keys = sorted(
        str(table_key)
        for table_key in sdata.tables
        if is_derived_assignment_table(str(table_key))
    )
    if not table_keys:
        print(f"OK no derived tables found: {zarr_path}")
        return

    for table_key in table_keys:
        action("table", f"{zarr_path}::tables/{table_key}", apply=apply)
        if not apply:
            continue
        try:
            sdata.delete_element_from_disk(table_key)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING delete_element_from_disk({table_key!r}) failed: {exc}; "
                "falling back to filesystem removal."
            )
            remove_path(zarr_path / "tables" / table_key)
        with suppress(Exception):
            del sdata.tables[table_key]

    if apply:
        _prune_zarr_consolidated_metadata(
            zarr_path / "zarr.json",
            {f"tables/{table_key}" for table_key in table_keys},
        )
        _prune_zarr_consolidated_metadata(
            zarr_path / "tables" / "zarr.json",
            set(table_keys),
        )


def downstream_output_paths(
    outdir: Path,
    sample: Sample,
    *,
    delete_alignment_output: bool,
) -> list[Path]:
    """Build published output directories that should be regenerated."""
    paths: list[Path] = []
    pair_dirs = PAIR_DOWNSTREAM_DIRS + (
        ALIGNMENT_OUTPUT_DIRS if delete_alignment_output else ()
    )
    paths.extend(outdir / sample.pair_id / dirname for dirname in pair_dirs)

    for platform in sample.platforms:
        platform_root = outdir / sample.pair_id / platform.lower()
        paths.extend(platform_root / dirname for dirname in PLATFORM_DOWNSTREAM_DIRS)
    return paths


def remove_downstream_outputs(
    outdir: Path,
    sample: Sample,
    *,
    apply: bool,
    delete_alignment_output: bool,
) -> None:
    """Remove published outputs from enrichment onward."""
    for path in downstream_output_paths(
        outdir,
        sample,
        delete_alignment_output=delete_alignment_output,
    ):
        if not path.exists() and not path.is_symlink():
            print(f"SKIP missing output: {path}")
            continue
        action("output", path, apply=apply)
        if apply:
            remove_path(path)


def main() -> None:
    """Run the cleanup."""
    args = parse_args()
    if args.tables_only and args.outputs_only:
        raise SystemExit("Choose at most one of --tables-only or --outputs-only.")

    samplesheet = args.samplesheet.resolve()
    outdir = args.outdir.resolve()
    samples = read_samples(samplesheet)
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: samplesheet={samplesheet}")
    print(f"{mode}: outdir={outdir}")
    print(f"{mode}: samples={len(samples)}")

    for sample in samples:
        print(f"\n== {sample.pair_id} ({', '.join(sample.platforms)}) ==")
        if not args.outputs_only:
            for platform in sample.platforms:
                zarr_path = (
                    outdir
                    / sample.pair_id
                    / platform.lower()
                    / "latest"
                    / "latest_spatialdata.zarr"
                )
                remove_derived_tables(zarr_path, apply=args.apply)
        if not args.tables_only:
            remove_downstream_outputs(
                outdir,
                sample,
                apply=args.apply,
                delete_alignment_output=args.delete_alignment_output,
            )


if __name__ == "__main__":
    main()
