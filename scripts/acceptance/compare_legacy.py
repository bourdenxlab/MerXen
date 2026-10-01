#!/usr/bin/env python
"""Legacy vs map_first per dataset, and H13's "legacy tables untouched" (M8).

Three subcommands, all read-only on the results they compare:

``manifest``
    Write a manifest of published result trees: every entry (file, symlink,
    directory) with its type, path relative to the results root, size,
    mtime (ns) and symlink target, in the M5 e2e manifest format, plus the
    sha256 of every file whose content H13 and the run's classification
    check (``needs_hash``): each latest SpatialData store's ``tables/``
    group and root metadata files, the legacy clustered and MENDER outputs
    (``<pair>/<seg>/clustering_squidpy*/``, ``mender*/``), the earlier
    map_first outputs (``<pair>/<seg>/annotation_*/``), the cortical-depth
    outputs (``<pair>/<platform>/compute_cortical_depth*/``) and everything
    under ``acceptance/``. Symlinks are never followed.

``untouched``
    H13's "legacy tables untouched before the flip (sha256)": compare two
    manifests (taken before the acceptance run and after it). Every table
    group of every store that existed before (``tables/<name>/``), and every
    file of the legacy clustered and MENDER outputs, must be unchanged: the
    same entries, types, sizes, mtimes and sha256, nothing added or removed
    inside. The earlier map_first and cortical-depth outputs and every other
    pre-existing entry are checked the same way and reported beside the
    verdict. New table groups whose name matches ``--new-table-pattern``
    (the run's suffixed tables) and new paths elsewhere are listed, never
    counted against H13.

``compare``
    Legacy vs map_first per pair x segmentation x platform, on the table
    cells: the legacy ``broad_class`` (published legacy clustered H5AD), the
    map_first ``broad_class`` (the run's clustered H5AD) and RESOLVE's
    confident broad label; their compositions, the legacy x map_first
    crosswalk and the H10 inputs (cells both labellings place in the E1
    referee's 7 classes, and the disputes among them), plus the id sets.

Usage::

    python scripts/acceptance/compare_legacy.py manifest --results-root <R> \\
        --out before.tsv P7513 P1212 P7113 P5011
    python scripts/acceptance/compare_legacy.py untouched --before before.tsv \\
        --after after.tsv --out <dir>
    python scripts/acceptance/compare_legacy.py compare --results-root <R> \\
        --mapfirst-root <R>/acceptance/<date>/runs --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("compare_legacy")

ZARR_SUFFIX = "latest/latest_spatialdata.zarr"
ROOT_METADATA = ("zarr.json", ".zattrs", ".zgroup", ".zmetadata")
SEGMENTATIONS = ("proseg_hybrid", "reseg", "proseg_mask", "original_seg")
PLATFORMS = ("MERSCOPE", "XENIUM")
MANIFEST_COLUMNS = ("type", "path", "size", "mtime_ns", "sha256", "link_target")
# Directory names under <pair>/<seg>/ whose files are hashed.
HASHED_SEGMENTATION_DIRS = ("clustering_squidpy", "mender", "annotation_")
# The run's suffixed clustered tables (clustering_squidpy_table_key_suffix m8,
# and a second MENDER policy's own suffix).
DEFAULT_NEW_TABLE_PATTERN = r"_clustering_squidpy_m8(_[a-z0-9]+)?$"
# Groups whose change fails H13 (the legacy tables and outputs).
H13_GROUPS = ("legacy_table", "legacy_clustering", "legacy_mender")
REFEREE_CLASSES = (
    "Neurons",
    "Astrocytes",
    "Oligodendrocytes",
    "Oligodendrocyte precursors",
    "Microglia",
    "Vascular cells",
    "Fibroblasts",
)

# ---------------------------------------------------------------------------
# manifest


def needs_hash(rel: str) -> bool:
    """Return whether a file's sha256 is recorded (``rel`` from the results root).

    Args:
        rel: Path relative to the results root.

    Returns:
        Whether the manifest hashes it.
    """
    parts = rel.split("/")
    marker = f"/{ZARR_SUFFIX}/"
    if marker in f"/{rel}":
        inner = f"/{rel}".split(marker, 1)[1]
        return inner.startswith("tables/") or inner in ROOT_METADATA
    if parts[0] == "acceptance":
        return True
    if len(parts) >= 3 and parts[1] in SEGMENTATIONS:
        return parts[2].startswith(HASHED_SEGMENTATION_DIRS)
    if len(parts) >= 3 and parts[1].upper() in PLATFORMS:
        return parts[2].startswith("compute_cortical_depth")
    return False


def file_sha256(path: str) -> str:
    """Return the sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(results: Path, root: str) -> list[tuple[str, str, int, int, str]]:
    """Return (type, rel, size, mtime_ns, link) for every entry under a root.

    Args:
        results: The results root.
        root: A path relative to it (a pair, or ``acceptance``).

    Returns:
        The entries; none when the root does not exist.
    """
    top = results / root
    if not os.path.lexists(top):
        return []
    rows: list[tuple[str, str, int, int, str]] = []
    stat = top.lstat()
    kind = "l" if top.is_symlink() else "d" if top.is_dir() else "f"
    rows.append((kind, root, stat.st_size, stat.st_mtime_ns, ""))
    if kind != "d":
        return rows
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        for name in sorted(dirnames) + sorted(filenames):
            full = os.path.join(dirpath, name)
            info = os.lstat(full)
            rel = os.path.relpath(full, results)
            if os.path.islink(full):
                rows.append(
                    ("l", rel, info.st_size, info.st_mtime_ns, os.readlink(full))
                )
            elif os.path.isdir(full):
                rows.append(("d", rel, info.st_size, info.st_mtime_ns, ""))
            else:
                rows.append(("f", rel, info.st_size, info.st_mtime_ns, ""))
    return rows


def write_manifest(
    results: Path, roots: Sequence[str], out: Path, *, workers: int = 4
) -> dict[str, int]:
    """Write the manifest of ``roots`` (M5 e2e format) and return its counts."""
    rows: list[tuple[str, str, int, int, str]] = []
    for root in roots:
        rows.extend(walk(results, root))
    to_hash = [rel for kind, rel, *_ in rows if kind == "f" and needs_hash(rel)]
    paths = [str(results / rel) for rel in to_hash]
    if workers <= 1:
        hashes = [file_sha256(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(file_sha256, paths, chunksize=16))
    digests = dict(zip(to_hash, hashes, strict=True))
    with out.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for kind, rel, size, mtime, link in sorted(rows, key=lambda row: row[1]):
            handle.write(
                f"{kind}\t{rel}\t{size}\t{mtime}\t{digests.get(rel, '-')}\t{link}\n"
            )
    return {"entries": len(rows), "hashed": len(to_hash)}


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Return ``{path: row}`` of a manifest TSV."""
    with path.open(encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle, delimiter="\t")}


# ---------------------------------------------------------------------------
# untouched (H13)


def classify_path(rel: str, new_table: re.Pattern[str]) -> tuple[str, str]:
    """Return (group kind, group key) of a manifest path.

    Kinds: ``legacy_table`` (a store's ``tables/<name>`` group, the name not
    matching ``new_table``), ``new_table`` (a matching table group or its
    backup groups), ``root_metadata``, ``tables_metadata`` (the ``tables/``
    group's own metadata), ``legacy_clustering`` / ``legacy_mender``
    (``<pair>/<seg>/clustering_squidpy/``, ``mender/``),
    ``earlier_mapfirst`` (other ``clustering_squidpy_*``, ``mender_*`` and
    ``annotation_*`` directories), ``cortical_depth``
    (``<pair>/<platform>/compute_cortical_depth*``), ``other``.
    """
    marker = f"/{ZARR_SUFFIX}/"
    if marker in f"/{rel}":
        store, inner = f"/{rel}".split(marker, 1)
        store = store.lstrip("/") + "/" + ZARR_SUFFIX
        if inner in ROOT_METADATA:
            return "root_metadata", f"{store}/{inner}"
        parts = inner.split("/")
        if parts[0] == "tables" and len(parts) >= 2:
            name = parts[1]
            if name in ROOT_METADATA:
                return "tables_metadata", f"{store}/tables/{name}"
            base = (
                name[1:].split(".merxen-backup-", 1)[0]
                if name.startswith(".")
                else name
            )
            kind = "new_table" if new_table.search(base) else "legacy_table"
            return kind, f"{store}/tables/{name}"
        return "other", f"{store}/{parts[0]}"
    parts = rel.split("/")
    if len(parts) >= 3 and parts[1] in SEGMENTATIONS:
        key = "/".join(parts[:3])
        if parts[2] == "clustering_squidpy":
            return "legacy_clustering", key
        if parts[2] == "mender":
            return "legacy_mender", key
        if parts[2].startswith(HASHED_SEGMENTATION_DIRS):
            return "earlier_mapfirst", key
    if (
        len(parts) >= 3
        and parts[1].upper() in PLATFORMS
        and parts[2].startswith("compute_cortical_depth")
    ):
        return "cortical_depth", "/".join(parts[:3])
    return "other", "/".join(parts[:2])


@dataclass
class GroupCheck:
    """The before / after comparison of one group of paths.

    Attributes:
        kind: ``classify_path`` kind.
        key: The group's path.
        n_before: Entries before.
        n_after: Entries after.
        n_hashed: Files compared by sha256.
        changed: Paths whose type, size, mtime, sha256 or link changed.
        added: Paths only after.
        removed: Paths only before.
    """

    kind: str
    key: str
    n_before: int = 0
    n_after: int = 0
    n_hashed: int = 0
    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def untouched(self: GroupCheck) -> bool:
        """Whether nothing changed, appeared or disappeared."""
        return not (self.changed or self.added or self.removed)

    def to_row(self: GroupCheck) -> dict[str, Any]:
        """Return the CSV row."""
        return {
            "kind": self.kind,
            "group": self.key,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "n_hashed": self.n_hashed,
            "n_changed": len(self.changed),
            "n_added": len(self.added),
            "n_removed": len(self.removed),
            "untouched": self.untouched,
            "h13": self.kind in H13_GROUPS,
            "examples": "; ".join((self.changed + self.added + self.removed)[:5]),
        }


def _differs(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    if before["type"] == "d" and after["type"] == "d":
        return []  # a directory's content is checked through its entries
    return [
        key
        for key in ("type", "size", "mtime_ns", "sha256", "link_target")
        if before.get(key) != after.get(key)
    ]


def check_untouched(
    before: Mapping[str, Mapping[str, str]],
    after: Mapping[str, Mapping[str, str]],
    *,
    new_table_pattern: str = DEFAULT_NEW_TABLE_PATTERN,
) -> tuple[list[GroupCheck], dict[str, Any]]:
    """Compare two manifests group by group (H13 "legacy tables untouched").

    A group that existed before is checked entry by entry (a directory by
    its entries, not its mtime); a group that exists only after (a new
    table, a new publish directory) is listed as new. H13 passes when every
    ``legacy_table``, ``legacy_clustering`` and ``legacy_mender`` group is
    untouched and at least one legacy table was compared; a file of those
    groups compared without a sha256 on both sides is reported as unhashed.

    Args:
        before: Manifest rows before the run (``load_manifest``).
        after: Manifest rows after it.
        new_table_pattern: Regex of the run's new table names.

    Returns:
        ``(groups, summary)``.
    """
    pattern = re.compile(new_table_pattern)
    groups: dict[tuple[str, str], GroupCheck] = {}
    existed: set[tuple[str, str]] = set()

    def group(rel: str) -> GroupCheck:
        kind, key = classify_path(rel, pattern)
        found = groups.get((kind, key))
        if found is None:
            found = groups[(kind, key)] = GroupCheck(kind, key)
        return found

    for rel, row in before.items():
        item = group(rel)
        existed.add((item.kind, item.key))
        item.n_before += 1
        new = after.get(rel)
        if new is None:
            item.removed.append(rel)
            continue
        diffs = _differs(row, new)
        if row["type"] == "f" and row.get("sha256", "-") != "-":
            item.n_hashed += 1
        if diffs:
            item.changed.append(f"{rel} ({','.join(diffs)})")
    unhashed: list[str] = []
    for rel, row in after.items():
        item = group(rel)
        item.n_after += 1
        if rel not in before:
            item.added.append(rel)
        elif (
            item.kind in H13_GROUPS
            and row["type"] == "f"
            and "-" in (row.get("sha256", "-"), before[rel].get("sha256", "-"))
        ):
            unhashed.append(rel)
    checked = [groups[key] for key in sorted(groups) if key in existed]
    new_groups = [groups[key] for key in sorted(groups) if key not in existed]
    h13 = [item for item in checked if item.kind in H13_GROUPS]
    failing = [item for item in h13 if not item.untouched]
    n_legacy_tables = sum(1 for item in h13 if item.kind == "legacy_table")
    passes = not failing and not unhashed and n_legacy_tables > 0
    summary = {
        "criterion": "H13",
        "part": "legacy tables untouched before the flip (sha256)",
        "passes": passes,
        "n_legacy_tables": n_legacy_tables,
        "n_legacy_clustering": sum(1 for i in h13 if i.kind == "legacy_clustering"),
        "n_legacy_mender": sum(1 for i in h13 if i.kind == "legacy_mender"),
        "n_hashed_files": sum(item.n_hashed for item in h13),
        "failing_groups": [item.key for item in failing],
        "unhashed_files": unhashed[:50],
        "other_changed_groups": sorted(
            item.key
            for item in checked
            if item.kind not in H13_GROUPS and not item.untouched
        ),
        "new_groups": {
            kind: sorted(item.key for item in new_groups if item.kind == kind)
            for kind in sorted({item.kind for item in new_groups})
        },
    }
    return checked + new_groups, summary


# ---------------------------------------------------------------------------
# compare (legacy vs map_first per dataset)


def _obs_broad(path: Path) -> pd.Series:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        values = adata.obs["broad_class"].astype(object)
        values.index = pd.Index(adata.obs_names.astype(str))
        return values
    finally:
        adata.file.close()


def composition_rows(
    labels: Mapping[str, pd.Series], dataset: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return the class counts and shares of each labelling of one dataset."""
    rows = []
    for source, values in labels.items():
        counts = values.fillna("not labelled").astype(str).value_counts()
        total = int(counts.sum())
        for cls, n in counts.items():
            rows.append(
                {
                    **dataset,
                    "source": source,
                    "class": cls,
                    "n": int(n),
                    "share": n / total if total else float("nan"),
                }
            )
    return rows


def h10_input_row(
    new: pd.Series, legacy: pd.Series, dataset: Mapping[str, str]
) -> dict[str, Any]:
    """Return the H10 inputs of one dataset (the referee's disputed cells).

    Args:
        new: RESOLVE confident broad label per table cell (``None`` when not
            confident), index cell id.
        legacy: Legacy ``broad_class`` per cell, index cell id.
        dataset: The dataset's key columns.

    Returns:
        Cell counts: labelled by each, common, both in the 7 classes, the
        disputes among them and the agreement share.
    """
    common = new.index.intersection(legacy.index)
    a = new.reindex(common).astype(object)
    b = legacy.reindex(common).astype(object)
    in_a = a.isin(REFEREE_CLASSES).to_numpy()
    in_b = b.isin(REFEREE_CLASSES).to_numpy()
    both = in_a & in_b
    disputed = both & (a.to_numpy() != b.to_numpy())
    return {
        **dataset,
        "n_new_table": len(new),
        "n_legacy": len(legacy),
        "n_common": len(common),
        "n_new_only": len(new.index.difference(legacy.index)),
        "n_legacy_only": len(legacy.index.difference(new.index)),
        "n_new_confident": int(a.notna().sum()),
        "n_both_in_classes": int(both.sum()),
        "n_disputes": int(disputed.sum()),
        "dispute_share": float(disputed.sum() / both.sum()) if both.any() else np.nan,
        "agreement": float(1 - disputed.sum() / both.sum()) if both.any() else np.nan,
    }


def crosswalk_rows(
    legacy: pd.Series, mapfirst: pd.Series, dataset: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return the legacy x map_first ``broad_class`` counts on common cells."""
    common = legacy.index.intersection(mapfirst.index)
    table = pd.crosstab(
        legacy.reindex(common).fillna("NA").astype(str).to_numpy(),
        mapfirst.reindex(common).fillna("NA").astype(str).to_numpy(),
    )
    rows = []
    for legacy_class, row in table.iterrows():
        for mapfirst_class, n in row.items():
            if n:
                rows.append(
                    {
                        **dataset,
                        "legacy_broad_class": legacy_class,
                        "mapfirst_broad_class": mapfirst_class,
                        "n": int(n),
                    }
                )
    return rows


def compare_dataset(
    legacy_h5ad: Path,
    mapfirst_h5ad: Path,
    labels_parquet: Path,
    dataset: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Compare one dataset's legacy and map_first labels (obs only)."""
    from merxen.annotation.pipeline import read_label_table

    legacy = _obs_broad(legacy_h5ad)
    mapfirst = _obs_broad(mapfirst_h5ad)
    labels, _ = read_label_table(labels_parquet)
    labels = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id")
    labels.index = labels.index.astype(str)
    confident = labels["ct_broad_status"].astype(str).to_numpy() == "confident"
    new = pd.Series(
        np.where(confident, labels["ct_broad_name"].astype(object).to_numpy(), None),
        index=labels.index,
        dtype=object,
    )
    ids = {
        **dataset,
        "n_legacy": len(legacy),
        "n_mapfirst": len(mapfirst),
        "n_label_table": len(labels),
        "mapfirst_equals_label_table": set(mapfirst.index) == set(labels.index),
        "legacy_equals_mapfirst": set(legacy.index) == set(mapfirst.index),
    }
    return {
        "composition": composition_rows(
            {"legacy": legacy, "mapfirst": mapfirst, "new_confident": new}, dataset
        ),
        "crosswalk": crosswalk_rows(legacy, mapfirst, dataset),
        "h10_inputs": [h10_input_row(new, legacy, dataset)],
        "id_sets": [ids],
    }


# ---------------------------------------------------------------------------
# CLI


def _manifest_command(args: argparse.Namespace) -> int:
    counts = write_manifest(
        args.results_root, args.roots, args.out, workers=args.workers
    )
    print(f"{args.out}: {counts['entries']} entries, {counts['hashed']} hashed")
    return 0


def _untouched_command(args: argparse.Namespace) -> int:
    groups, summary = check_untouched(
        load_manifest(args.before),
        load_manifest(args.after),
        new_table_pattern=args.new_table_pattern,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.to_row() for item in groups]).to_csv(
        args.out / "legacy_untouched.csv", index=False
    )
    summary["before"] = str(args.before)
    summary["after"] = str(args.after)
    (args.out / "legacy_untouched.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("passes", "n_legacy_tables")}))
    return 0 if summary["passes"] else 1


def _iter_datasets(
    pairs: Iterable[str], segmentations: Iterable[str]
) -> Iterable[tuple[str, str, str]]:
    for pair in pairs:
        for segmentation in segmentations:
            for platform in PLATFORMS:
                yield pair, segmentation, platform


def _compare_command(args: argparse.Namespace) -> int:
    outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    for pair, segmentation, platform in _iter_datasets(
        [p for p in args.pairs.split(",") if p],
        [s for s in args.segmentations.split(",") if s],
    ):
        sample = f"{pair}_{platform}"
        legacy = (
            args.results_root
            / pair
            / segmentation
            / "clustering_squidpy"
            / "clustering_squidpy_out"
            / platform.lower()
            / f"{sample}_clustered.h5ad"
        )
        mapfirst = (
            args.mapfirst_root
            / pair
            / segmentation
            / args.clustering_dir
            / "clustering_squidpy_out"
            / platform.lower()
            / f"{sample}_clustered.h5ad"
        )
        labels = (
            args.mapfirst_root
            / pair
            / segmentation
            / "annotation_resolve"
            / "annotation_resolve_out"
            / platform.lower()
            / f"{sample}_celltype_labels.parquet"
        )
        absent = [
            str(path) for path in (legacy, mapfirst, labels) if not path.is_file()
        ]
        if absent:
            missing += absent
            continue
        dataset = {"pair": pair, "segmentation": segmentation, "sample_id": sample}
        for name, rows in compare_dataset(legacy, mapfirst, labels, dataset).items():
            outputs[name] += rows
        logger.info("%s %s compared", sample, segmentation)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("composition", "crosswalk", "h10_inputs", "id_sets"):
        pd.DataFrame(outputs[name]).to_csv(args.out / f"legacy_{name}.csv", index=False)
    record = {
        "results_root": str(args.results_root),
        "mapfirst_root": str(args.mapfirst_root),
        "clustering_dir": args.clustering_dir,
        "datasets": len(outputs["id_sets"]),
        "missing_inputs": missing,
        "sources": dict(Counter(row["source"] for row in outputs["composition"])),
    }
    (args.out / "compare_legacy_run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.out}: {record['datasets']} datasets, {len(missing)} missing inputs")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="write a manifest")
    manifest.add_argument("--results-root", type=Path, required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--workers", type=int, default=4)
    manifest.add_argument("roots", nargs="+", help="paths under the results root")
    untouched = commands.add_parser("untouched", help="H13: legacy tables untouched")
    untouched.add_argument("--before", type=Path, required=True)
    untouched.add_argument("--after", type=Path, required=True)
    untouched.add_argument("--out", type=Path, required=True)
    untouched.add_argument("--new-table-pattern", default=DEFAULT_NEW_TABLE_PATTERN)
    compare = commands.add_parser("compare", help="legacy vs map_first labels")
    compare.add_argument("--results-root", type=Path, required=True)
    compare.add_argument("--mapfirst-root", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--clustering-dir", default="clustering_squidpy_m8")
    compare.add_argument("--pairs", default="P7513,P1212,P7113,P5011")
    compare.add_argument("--segmentations", default=",".join(SEGMENTATIONS))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    handler = {
        "manifest": _manifest_command,
        "untouched": _untouched_command,
        "compare": _compare_command,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
