#!/usr/bin/env python
"""Remove the map_first tables and outputs a run added to published results.

A map_first run into a legacy results directory (for example the M5 exit run
on P7513 and P1212) adds, per pair:

- ``table_<layer>_clustering_squidpy_<suffix>`` in each platform's
  ``<outdir>/<pair>/<platform>/latest/latest_spatialdata.zarr`` (plan §4.8),
  and the consolidated-metadata entries of that table and of any
  ``.<table>.merxen-backup-<uuid>`` group an overwrite left behind;
- the output directories ``<outdir>/<pair>/<segmentation>/{clustering_squidpy_<suffix>,
  mender_<suffix>}`` and ``<outdir>/<pair>/<platform>/compute_cortical_depth_<suffix>``;
- the unsuffixed ``<outdir>/<pair>/<segmentation>/{annotation_panel,
  annotation_map, annotation_resolve}``, which every map_first run (and
  ``--annotation_prepare_only``) publishes under the same names.

This script removes those, and nothing else: it never touches a table whose
name does not end in ``_clustering_squidpy_<suffix>`` (``<suffix>`` must be
non-empty), and it moves instead of deleting. Because the unsuffixed
annotation directories do not say which run wrote them (a later run's MAP
reuses a published ``annotation_map``), they are moved only with
``--include-annotation-dirs``, which needs ``--new-paths``: the paths the run
created, relative to ``--outdir`` (for example the M5 exit run's
``compare_before_after.new_paths.txt``). With ``--new-paths`` every table and
directory not on the list is kept and reported, so only what the run created
moves. Each table group and output directory is renamed into ``--quarantine``
(which must be on the same file system, so every move is an atomic rename),
and the zarr root ``zarr.json`` loses only the consolidated entries of the
moved tables and of their stale backup groups; every other entry keeps its
value and order. Delete the quarantine directory once the removal has been
checked.

Dry run by default: it prints the plan. ``--apply`` performs it under the
MerXen SpatialData writer lock of each zarr and writes
``<quarantine>/removal_record.json`` (every move, and the pruned
consolidated entries, so a restore can move the directories back and
re-insert the entries).

Usage::

    python scripts/annotation/remove_mapfirst_outputs.py \\
        --outdir /srv/storage/MerXen/results --pair P7513 --pair P1212 \\
        --segmentation proseg_hybrid --quarantine <dir on the same disk> \\
        [--include-annotation-dirs --new-paths <run's new paths>] [--apply]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from merxen.io.spatialdata_io import spatialdata_write_lock  # noqa: E402

logger = logging.getLogger("remove_mapfirst_outputs")

PLATFORMS = ("merscope", "xenium")
ZARR_RELATIVE = Path("latest") / "latest_spatialdata.zarr"
BACKUP_MARKER = ".merxen-backup-"
CONSOLIDATED_PREFIX = "tables/"
SEGMENTATION_DIRS = ("annotation_panel", "annotation_map", "annotation_resolve")
RECORD_NAME = "removal_record.json"


@dataclass(frozen=True)
class Move:
    """One directory rename into the quarantine."""

    source: str
    destination: str


@dataclass
class ZarrPlan:
    """What the removal changes in one zarr."""

    zarr_path: str
    tables: list[str] = field(default_factory=list)
    consolidated_keys: list[str] = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)


@dataclass
class RemovalPlan:
    """The whole removal: zarr tables and output directories.

    Attributes:
        suffix: Table-key suffix.
        zarrs: Per zarr, the tables to move and the entries to prune.
        output_moves: Output directories to move.
        kept: Candidates left in place, with the reason.
    """

    suffix: str
    zarrs: list[ZarrPlan] = field(default_factory=list)
    output_moves: list[Move] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return whether nothing would be removed."""
        return not self.output_moves and not any(
            zarr.moves or zarr.consolidated_keys for zarr in self.zarrs
        )


def is_suffixed_table(name: str, suffix: str) -> bool:
    """Return whether ``name`` is a map_first clustered table of ``suffix``.

    Args:
        name: Table group name.
        suffix: Table-key suffix (non-empty, e.g. ``mapfirst``).

    Returns:
        True for ``table_<layer>_clustering_squidpy_<suffix>`` only.
    """
    return name.startswith("table_") and name.endswith(f"_clustering_squidpy_{suffix}")


def is_suffixed_backup(name: str, suffix: str) -> bool:
    """Return whether ``name`` is a backup group of a suffixed table.

    Args:
        name: Table group name.
        suffix: Table-key suffix.

    Returns:
        True for ``.<suffixed table>.merxen-backup-<uuid>``.
    """
    if not name.startswith(".") or BACKUP_MARKER not in name:
        return False
    return is_suffixed_table(name[1:].split(BACKUP_MARKER, 1)[0], suffix)


def _consolidated_metadata(zarr_path: Path) -> dict[str, Any]:
    """Return the root zarr.json's consolidated metadata entries (or {})."""
    root = zarr_path / "zarr.json"
    if not root.is_file():
        return {}
    data = json.loads(root.read_text())
    consolidated = data.get("consolidated_metadata")
    metadata = consolidated.get("metadata") if isinstance(consolidated, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _table_of(element: str, suffix: str) -> str | None:
    """Return the suffixed table an element is or backs up, else ``None``."""
    if is_suffixed_table(element, suffix):
        return element
    if is_suffixed_backup(element, suffix):
        return element[1:].split(BACKUP_MARKER, 1)[0]
    return None


def plan_zarr(
    zarr_path: Path,
    suffix: str,
    destination_root: Path,
    *,
    created: set[Path] | None = None,
    kept: list[str] | None = None,
) -> ZarrPlan:
    """Plan the removal of one zarr's suffixed tables.

    Args:
        zarr_path: The latest SpatialData zarr.
        suffix: Table-key suffix.
        destination_root: Quarantine directory that receives its table groups.
        created: Paths the run created (``--new-paths``), or ``None``: a
            table not on the list is kept, with its backups and entries.
        kept: Receives a line per kept table.

    Returns:
        The tables to move and the consolidated keys to prune.
    """
    plan = ZarrPlan(zarr_path=str(zarr_path))
    tables_dir = zarr_path / "tables"
    names = (
        sorted(entry.name for entry in tables_dir.iterdir())
        if tables_dir.is_dir()
        else []
    )
    kept_tables = {
        name
        for name in names
        if is_suffixed_table(name, suffix)
        and created is not None
        and tables_dir / name not in created
    }
    for name in sorted(kept_tables):
        if kept is not None:
            kept.append(f"{tables_dir / name} (not created by the run)")
    for name in names:
        table = _table_of(name, suffix)
        if table is None or table in kept_tables:
            continue
        plan.tables.append(name)
        plan.moves.append(
            Move(
                source=str(tables_dir / name), destination=str(destination_root / name)
            )
        )
    for key in _consolidated_metadata(zarr_path):
        if not key.startswith(CONSOLIDATED_PREFIX):
            continue
        table = _table_of(key[len(CONSOLIDATED_PREFIX) :].split("/", 1)[0], suffix)
        if table is not None and table not in kept_tables:
            plan.consolidated_keys.append(key)
    return plan


def read_new_paths(path: Path, outdir: Path) -> set[Path]:
    """Read the paths a run created.

    Args:
        path: One path per line, relative to ``outdir`` or absolute; a
            tab-separated line (``<type>\t<path>\t...``, as the before/after
            comparison writes it) gives its second field.
        outdir: Results root.

    Returns:
        The absolute paths.
    """
    paths: set[Path] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        text = fields[1] if len(fields) > 1 and len(fields[0]) == 1 else fields[0]
        entry = Path(text.strip())
        paths.add(entry if entry.is_absolute() else outdir / entry)
    return paths


def build_plan(
    outdir: Path,
    pairs: list[str],
    segmentations: list[str],
    platforms: list[str],
    suffix: str,
    quarantine: Path,
    *,
    include_annotation_dirs: bool = False,
    new_paths: set[Path] | None = None,
) -> RemovalPlan:
    """Plan the removal for the given pairs.

    Args:
        outdir: Results root.
        pairs: Pair ids.
        segmentations: Segmentations whose output directories are removed.
        platforms: Platform directory names (``merscope``, ``xenium``).
        suffix: Table-key suffix (non-empty).
        quarantine: Directory that receives every moved directory.
        include_annotation_dirs: Also move the unsuffixed
            ``annotation_{panel,map,resolve}`` directories the run created.
        new_paths: Paths the run created (``read_new_paths``): anything not
            on it is kept. Required with ``include_annotation_dirs``.

    Returns:
        The removal plan.

    Raises:
        ValueError: If the suffix is empty or not a plain name, or
            ``include_annotation_dirs`` is set without ``new_paths``.
    """
    if not suffix or "/" in suffix or suffix != suffix.strip():
        raise ValueError(f"--suffix must be a non-empty plain name, got {suffix!r}")
    if include_annotation_dirs and new_paths is None:
        raise ValueError(
            "--include-annotation-dirs needs --new-paths: the unsuffixed "
            "annotation directories do not record which run wrote them"
        )
    plan = RemovalPlan(suffix=suffix)

    def add_output(path: Path) -> None:
        if not path.exists():
            return
        if new_paths is not None and path not in new_paths:
            plan.kept.append(f"{path} (not created by the run)")
            return
        plan.output_moves.append(_quarantine_move(path, outdir, quarantine))

    for pair in pairs:
        for platform in platforms:
            zarr_path = outdir / pair / platform / ZARR_RELATIVE
            if zarr_path.is_dir():
                destination = (
                    quarantine
                    / "zarr_tables"
                    / zarr_path.relative_to(outdir)
                    / "tables"
                )
                plan.zarrs.append(
                    plan_zarr(
                        zarr_path,
                        suffix,
                        destination,
                        created=new_paths,
                        kept=plan.kept,
                    )
                )
            add_output(outdir / pair / platform / f"compute_cortical_depth_{suffix}")
        for segmentation in segmentations:
            for name in (f"clustering_squidpy_{suffix}", f"mender_{suffix}"):
                add_output(outdir / pair / segmentation / name)
            for name in SEGMENTATION_DIRS:
                path = outdir / pair / segmentation / name
                if not include_annotation_dirs:
                    if path.exists():
                        plan.kept.append(
                            f"{path} (unsuffixed; needs --include-annotation-dirs)"
                        )
                    continue
                add_output(path)
    return plan


def _quarantine_move(path: Path, outdir: Path, quarantine: Path) -> Move:
    """Return the move of an output directory into the quarantine."""
    return Move(
        source=str(path),
        destination=str(quarantine / "outputs" / path.relative_to(outdir)),
    )


def _existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _check_move(move: Move) -> None:
    """Refuse a move whose target exists or lies on another file system.

    Creates nothing, so a refused plan leaves the quarantine as it was.
    """
    source = Path(move.source)
    destination = Path(move.destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Quarantine target already exists: {destination}")
    target_device = os.stat(_existing_ancestor(destination.parent)).st_dev
    if os.stat(source.parent).st_dev != target_device:
        raise OSError(
            f"{destination.parent} is on another file system than {source}; "
            "choose a quarantine on the same disk"
        )


def _rename(move: Move) -> None:
    """Rename one directory into the quarantine (checked by _check_move)."""
    _check_move(move)
    Path(move.destination).parent.mkdir(parents=True, exist_ok=True)
    os.rename(move.source, move.destination)


def prune_consolidated(zarr_path: Path, keys: list[str]) -> dict[str, Any]:
    """Drop the given consolidated entries from the root zarr.json.

    Every other entry keeps its value and position; the file is written as
    zarr writes it (``json.dumps(indent=2)``), through a temporary file and
    an atomic replace.

    Args:
        zarr_path: The zarr root.
        keys: Consolidated keys to drop.

    Returns:
        The dropped entries (for the removal record).
    """
    root = zarr_path / "zarr.json"
    data = json.loads(root.read_text())
    metadata = data["consolidated_metadata"]["metadata"]
    dropped = {key: metadata.pop(key) for key in keys if key in metadata}
    temporary = root.with_name(f".zarr.json.remove-mapfirst-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, root)
    return dropped


def apply_plan(plan: RemovalPlan, quarantine: Path) -> Path:
    """Perform a removal plan and write its record.

    Args:
        plan: The plan from ``build_plan``.
        quarantine: The quarantine directory (created if missing).

    Returns:
        The removal record path.
    """
    quarantine.mkdir(parents=True, exist_ok=True)
    # Check every move before the first one, so a clash never leaves a
    # partial removal behind.
    for move in [m for zarr in plan.zarrs for m in zarr.moves] + plan.output_moves:
        _check_move(move)
    record: dict[str, Any] = {"suffix": plan.suffix, "zarrs": [], "output_moves": []}
    for zarr in plan.zarrs:
        zarr_path = Path(zarr.zarr_path)
        with spatialdata_write_lock(zarr_path):
            for move in zarr.moves:
                _rename(move)
                logger.info("Moved table %s -> %s", move.source, move.destination)
            dropped = (
                prune_consolidated(zarr_path, zarr.consolidated_keys)
                if zarr.consolidated_keys
                else {}
            )
        record["zarrs"].append(
            {**asdict(zarr), "dropped_consolidated_entries": dropped}
        )
        logger.info("Pruned %d consolidated entries from %s", len(dropped), zarr_path)
    for move in plan.output_moves:
        _rename(move)
        record["output_moves"].append(asdict(move))
        logger.info("Moved output %s -> %s", move.source, move.destination)
    record_path = quarantine / RECORD_NAME
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record_path


def describe(plan: RemovalPlan) -> str:
    """Return a human-readable description of a plan."""
    lines = [f"map_first removal plan (suffix {plan.suffix!r})"]
    for zarr in plan.zarrs:
        lines.append(f"zarr {zarr.zarr_path}")
        lines.extend(f"  move table {name}" for name in zarr.tables)
        lines.append(f"  prune {len(zarr.consolidated_keys)} consolidated entries")
    lines.extend(
        f"move output {move.source} -> {move.destination}" for move in plan.output_moves
    )
    lines.extend(f"keep {item}" for item in plan.kept)
    if plan.is_empty():
        lines.append("nothing to remove")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", type=Path, required=True, help="Results root.")
    parser.add_argument(
        "--pair", action="append", required=True, help="Pair id (repeatable)."
    )
    parser.add_argument(
        "--segmentation",
        action="append",
        default=None,
        help="Segmentation (repeatable; default proseg_hybrid).",
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=None,
        choices=PLATFORMS,
        help="Platform directory (repeatable; default both).",
    )
    parser.add_argument(
        "--suffix", default="mapfirst", help="Table-key suffix (default mapfirst)."
    )
    parser.add_argument(
        "--quarantine",
        type=Path,
        required=True,
        help="Directory on the same disk that receives every moved directory.",
    )
    parser.add_argument(
        "--include-annotation-dirs",
        action="store_true",
        help=(
            "Also move the unsuffixed annotation_panel / annotation_map / "
            "annotation_resolve directories the run created (needs --new-paths)."
        ),
    )
    parser.add_argument(
        "--new-paths",
        type=Path,
        default=None,
        help=(
            "Paths the run created, one per line, relative to --outdir (or the "
            "before/after comparison's new_paths.txt); anything else is kept."
        ),
    )
    parser.add_argument(
        "--apply", action="store_true", help="Perform the plan (default: dry run)."
    )
    args = parser.parse_args(argv)
    if args.include_annotation_dirs and args.new_paths is None:
        parser.error("--include-annotation-dirs requires --new-paths")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the removal (dry run unless --apply)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    quarantine = args.quarantine.resolve()
    outdir = args.outdir.resolve()
    plan = build_plan(
        outdir,
        args.pair,
        args.segmentation or ["proseg_hybrid"],
        args.platform or list(PLATFORMS),
        args.suffix,
        quarantine,
        include_annotation_dirs=args.include_annotation_dirs,
        new_paths=(
            None if args.new_paths is None else read_new_paths(args.new_paths, outdir)
        ),
    )
    print(describe(plan))
    if not args.apply or plan.is_empty():
        return 0
    record = apply_plan(plan, quarantine)
    print(f"removal record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
