#!/usr/bin/env python
"""Partition-read benchmark of an Atera store's transcripts (plan §3.3, §7.4).

It times the two read patterns that decide between ``row_range`` and
``tile_blocks`` partitions:

- **window reads**: every transcript in a square window (default 1 mm²). The
  parquet files are pruned with the store's points index (file bounding
  boxes), then read with pyarrow ``x``/``y`` filters, which also skip row
  groups by their statistics. One window is also read without the index
  (every file, the same filters) as the full-scan baseline;
- **per-gene reads**: every transcript of one gene, read from all files with a
  ``feature_name`` filter. Within a ``row_range`` file the rows are gene-sorted,
  so the row-group statistics skip most of the data.

The window centres come from the store's 100 µm density grid (seeded, weighted
by density), and the genes from its ``gene_counts_kept`` quantiles, so two
stores built from the same bundle are read at the same windows and genes.
Every read is checked: a window read must return the rows the full filter
returns, and a gene read must return ``gene_counts_kept`` rows.

The report names the dataset only by its alias and holds no path.

Usage::

    atera_partition_bench.py --store <store> --alias <alias> --report <json>
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

logger = logging.getLogger("atera_partition_bench")

POINTS_INDEX_NAME = "merxen_atera_points_index.json"
POINTS_KEY = "transcripts"
DEFAULT_COLUMNS = ("x", "y", "feature_name", "original_assignment")


def _peak_rss_gb() -> float:
    """Peak resident memory of this process so far (Linux ``ru_maxrss`` in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def load_index(store: Path) -> dict[str, Any]:
    """Return the transcripts entry of the store's points index."""
    index = json.loads((store / POINTS_INDEX_NAME).read_text())
    return index["points"][POINTS_KEY]


def load_transcript_attrs(store: Path) -> dict[str, Any]:
    """Return ``attrs["merxen_atera"]["transcripts"]`` of the store root."""
    root = json.loads((store / "zarr.json").read_text())
    return root["attributes"]["merxen_atera"]["transcripts"]


def window_centres(
    attrs: dict[str, Any], n_windows: int, *, seed: int
) -> list[tuple[float, float]]:
    """Pick window centres from the density grid, weighted by density.

    Args:
        attrs: Transcript attrs (``density``, ``density_origin_um``,
            ``density_bin_um``).
        n_windows: Number of centres.
        seed: Random seed.

    Returns:
        ``(x, y)`` centres in µm, at bin centres.
    """
    density = np.asarray(attrs["density"], dtype=np.float64)
    origin_x, origin_y = (float(v) for v in attrs["density_origin_um"])
    bin_um = float(attrs["density_bin_um"])
    weights = density.ravel()
    if weights.sum() <= 0:
        raise ValueError("the density grid is empty")
    rng = np.random.default_rng(seed)
    picks = rng.choice(
        weights.size, size=n_windows, replace=True, p=weights / weights.sum()
    )
    rows, cols = np.unravel_index(picks, density.shape)
    return [
        (origin_x + (int(c) + 0.5) * bin_um, origin_y + (int(r) + 0.5) * bin_um)
        for r, c in zip(rows, cols, strict=True)
    ]


def benchmark_genes(attrs: dict[str, Any], quantiles: Sequence[float]) -> list[str]:
    """Pick genes at count quantiles of ``gene_counts_kept`` (genes kept > 0)."""
    names = list(attrs["feature_names"])
    counts = np.asarray(attrs["gene_counts_kept"], dtype=np.int64)
    present = np.flatnonzero(counts > 0)
    order = present[np.argsort(counts[present], kind="stable")]
    picks = []
    for q in quantiles:
        position = min(len(order) - 1, max(0, round(q * (len(order) - 1))))
        name = names[int(order[position])]
        if name not in picks:
            picks.append(name)
    return picks


def _files_for_window(
    entries: Sequence[dict[str, Any]], box: tuple[float, float, float, float]
) -> list[dict[str, Any]]:
    x0, x1, y0, y1 = box
    keep = []
    for entry in entries:
        bbox = entry.get("bbox_um")
        if bbox is None:
            keep.append(entry)
            continue
        bx0, bx1, by0, by1 = bbox
        if bx1 >= x0 and bx0 < x1 and by1 >= y0 and by0 < y1:
            keep.append(entry)
    return keep


def _window_filter(box: tuple[float, float, float, float]) -> ds.Expression:
    x0, x1, y0, y1 = box
    x, y = ds.field("x"), ds.field("y")
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


def _read(
    paths: Sequence[Path], columns: Sequence[str], expression: ds.Expression
) -> tuple[int, float]:
    started = time.perf_counter()
    if not paths:
        return 0, time.perf_counter() - started
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    table = dataset.to_table(columns=list(columns), filter=expression, use_threads=True)
    return table.num_rows, time.perf_counter() - started


def run_benchmark(
    store: Path,
    *,
    alias: str,
    window_um: float = 1000.0,
    n_windows: int = 6,
    gene_quantiles: Sequence[float] = (0.5, 0.9, 0.99, 1.0),
    columns: Sequence[str] = DEFAULT_COLUMNS,
    repeats: int = 2,
    seed: int = 0,
    full_scan: bool = True,
) -> dict[str, Any]:
    """Time window and per-gene reads of one store.

    Args:
        store: Atera SpatialData store.
        alias: Dataset alias for the report.
        window_um: Window edge in µm.
        n_windows: Number of windows.
        gene_quantiles: Quantiles of ``gene_counts_kept`` that pick the genes.
        columns: Columns read.
        repeats: Reads per window or gene; the first is the cold-ish read.
        seed: Seed of the window centres.
        full_scan: Also read the first window without the index.

    Returns:
        The report (JSON-safe).

    Raises:
        RuntimeError: If a read returns the wrong number of rows.
    """
    store = Path(store)
    index = load_index(store)
    attrs = load_transcript_attrs(store)
    directory = store / index["directory"]
    entries = index["files"]
    all_paths = [directory / e["file"] for e in entries]
    file_bytes = {e["file"]: (directory / e["file"]).stat().st_size for e in entries}
    total_bytes = sum(file_bytes.values())
    total_rows = sum(int(e["n_rows"]) for e in entries)
    root_attrs = json.loads((store / "zarr.json").read_text())["attributes"]
    options = root_attrs["merxen_atera"].get("build", {}).get("options", {})
    spatial_order = options.get("spatial_order")

    windows = []
    half = window_um / 2.0
    for k, (cx, cy) in enumerate(window_centres(attrs, n_windows, seed=seed)):
        box = (cx - half, cx + half, cy - half, cy + half)
        selected = _files_for_window(entries, box)
        paths = [directory / e["file"] for e in selected]
        runs = []
        rows = None
        for _ in range(repeats):
            n, seconds = _read(paths, columns, _window_filter(box))
            if rows is not None and n != rows:
                raise RuntimeError(
                    f"window {k}: repeated reads returned {rows} and {n} rows"
                )
            rows = n
            runs.append(round(seconds, 3))
        record = {
            "centre_um": [round(cx, 3), round(cy, 3)],
            "rows": rows,
            "files_read": len(selected),
            "files_total": len(entries),
            "rows_in_files_read": sum(int(e["n_rows"]) for e in selected),
            "bytes_in_files_read": sum(file_bytes[e["file"]] for e in selected),
            "seconds": runs,
            "rows_per_second_warm": round(rows / min(runs), 1)
            if min(runs) > 0
            else None,
        }
        if full_scan and k == 0:
            n, seconds = _read(all_paths, columns, _window_filter(box))
            if n != rows:
                raise RuntimeError(
                    f"window {k}: pruned read {rows} rows, full scan {n} rows"
                )
            record["full_scan_seconds"] = round(seconds, 3)
        logger.info(
            "[%s] window %d: %d rows from %d/%d files in %s s",
            alias,
            k,
            rows,
            len(selected),
            len(entries),
            runs,
        )
        windows.append(record)

    names = list(attrs["feature_names"])
    kept = np.asarray(attrs["gene_counts_kept"], dtype=np.int64)
    genes = []
    for gene in benchmark_genes(attrs, gene_quantiles):
        expected = int(kept[names.index(gene)])
        runs = []
        rows = None
        for _ in range(repeats):
            rows, seconds = _read(all_paths, columns, ds.field("feature_name") == gene)
            runs.append(round(seconds, 3))
        if rows != expected:
            raise RuntimeError(
                f"gene read returned {rows} rows, gene_counts_kept {expected}"
            )
        logger.info("[%s] gene with %d rows in %s s", alias, rows, runs)
        # Gene symbols are public panel names, not sample identifiers.
        genes.append({"gene": gene, "rows": rows, "seconds": runs})

    def _median(values: list[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    return {
        "alias": alias,
        "spatial_order": spatial_order,
        "columns": list(columns),
        "n_files": len(entries),
        "n_rows": total_rows,
        "points_bytes": total_bytes,
        "window_um": window_um,
        "windows": windows,
        "genes": genes,
        "summary": {
            "window_median_files_read": _median([w["files_read"] for w in windows]),
            "window_median_seconds_first": _median([w["seconds"][0] for w in windows]),
            "window_median_seconds_warm": _median([min(w["seconds"]) for w in windows]),
            "window_median_rows": _median([w["rows"] for w in windows]),
            "window_median_rows_in_files_read": _median(
                [w["rows_in_files_read"] for w in windows]
            ),
            "window_full_scan_seconds": windows[0].get("full_scan_seconds")
            if windows
            else None,
            "gene_median_seconds_warm": _median([min(g["seconds"]) for g in genes]),
            "gene_max_seconds_warm": max(
                (min(g["seconds"]) for g in genes), default=None
            ),
        },
        "peak_rss_gb": round(_peak_rss_gb(), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--alias", required=True, help="Dataset alias for the report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-um", type=float, default=1000.0)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-full-scan", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.perf_counter()
    report = run_benchmark(
        args.store,
        alias=args.alias,
        window_um=args.window_um,
        n_windows=args.windows,
        repeats=args.repeats,
        seed=args.seed,
        full_scan=not args.no_full_scan,
    )
    report["wall_seconds"] = round(time.perf_counter() - started, 2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    logger.info("[%s] benchmark done: %s", args.alias, json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
