"""The prior of a tiled ATERA ProSeg run: vendor cell labels or Cellpose masks.

A ``native_proseg`` plan (AT-9) has no ``prior`` entry: the vendor
``labels/cell_labels`` of the source store are the prior. A ``cellpose_proseg``
plan (AT-7) records a Cellpose prior: the ``.npy`` label mask MerXen's Cellpose
wrote (DAPI + 18S, uint32, the morphology grid of the store) and its float32
cell-probability logits, with their SHA-256 and their size and modification
time.

The Cellpose masks are on exactly the vendor mask grid (morphology grid =
mask grid), so every consumer treats them as it treats the vendor mask: a
micron coordinate maps to pixel ``floor(x / ps)`` (float32, the corner rule of
:func:`merxen.enrichment.label_lookup.pixel_indices`) and ProSeg gets the
transforms ``[ps, 0, col0 ps]`` / ``[0, ps, row0 ps]``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from merxen.segmentation.atera_proseg import (
    AteraProsegError,
    PriorKind,
    proseg_layer,
)

STRIP_ROWS = 512
"""Mask rows read at a time when the label statistics are streamed."""


def plan_prior_kind(plan: dict[str, Any]) -> PriorKind:
    """Return the prior kind of a tile plan (``"vendor"`` when it records none)."""
    prior = plan.get("prior")
    if not prior:
        return "vendor"
    kind = str(prior.get("kind", ""))
    proseg_layer(kind)
    return "cellpose" if kind == "cellpose" else "vendor"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in 4 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path: Path) -> list[int]:
    """Return ``[size, mtime_ns]`` of a file (the cheap identity tiles check)."""
    stat = os.stat(path)
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def describe_cellpose_prior(mask_path: Path, cellprob_path: Path | None) -> dict:
    """Return the ``prior`` entry of a plan for Cellpose files.

    Args:
        mask_path: The Cellpose label mask (``.npy``, integer).
        cellprob_path: Its cell-probability logits (``.npy``, float32), or
            ``None`` when ProSeg gets no cell probabilities.

    Returns:
        ``kind``, the real paths, their SHA-256 and their ``[size, mtime_ns]``.
    """
    mask = Path(os.path.realpath(mask_path))
    entry: dict[str, Any] = {
        "kind": "cellpose",
        "mask_path": str(mask),
        "mask_sha256": file_sha256(mask),
        "mask_stat": file_stat(mask),
        "cellprob_path": None,
        "cellprob_sha256": None,
        "cellprob_stat": None,
    }
    if cellprob_path is not None:
        cellprob = Path(os.path.realpath(cellprob_path))
        entry.update(
            cellprob_path=str(cellprob),
            cellprob_sha256=file_sha256(cellprob),
            cellprob_stat=file_stat(cellprob),
        )
    entry["fingerprint"] = hashlib.sha256(
        f"{entry['mask_sha256']}|{entry['cellprob_sha256']}".encode()
    ).hexdigest()
    return entry


def check_prior_files(prior: dict[str, Any], context: str) -> None:
    """Refuse prior files that changed since the plan described them.

    The size and modification time are compared (re-hashing a 10 GB mask in
    every tile would cost more than the tile); a re-run Cellpose rewrites the
    file and changes both.

    Args:
        prior: The plan's (or tile's) ``prior`` entry.
        context: Prefix of the error message.

    Raises:
        AteraProsegError: If a file is missing or changed.
    """
    for name in ("mask", "cellprob"):
        path = prior.get(f"{name}_path")
        if path is None:
            continue
        recorded = prior.get(f"{name}_stat")
        try:
            found = file_stat(Path(path))
        except OSError as error:
            raise AteraProsegError(
                f"{context}: the Cellpose {name} file of the plan is missing ({error})"
            ) from error
        if recorded is not None and list(recorded) != found:
            raise AteraProsegError(
                f"{context}: the Cellpose {name} file changed since the tile plan was "
                "made (Cellpose re-ran); re-run ATERA_PROSEG_PLAN"
            )


def open_mask(path: Path) -> np.ndarray:
    """Return a read-only memmap of a 2-D integer ``.npy`` label mask."""
    mask: np.ndarray = np.load(Path(path), mmap_mode="r")
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise AteraProsegError(
            f"{Path(path).name} is not a 2-D integer label mask "
            f"(shape {mask.shape}, dtype {mask.dtype})"
        )
    return mask


def label_stats(
    mask: np.ndarray, pixel_size_um: float, *, strip_rows: int = STRIP_ROWS
) -> dict[str, np.ndarray]:
    """Return the ids, centroids and bounds of every label of a mask.

    The mask is read in row strips, so a full-section memmap is never loaded
    whole. The centroid is the mean pixel centre (``(index + 0.5) ps``, the
    vendor rule) and the bounds cover whole pixels (corner rule).

    Args:
        mask: 2-D non-negative integer labels (0 is background).
        pixel_size_um: Microns per pixel.
        strip_rows: Rows per strip.

    Returns:
        ``instance_id`` (uint64, ascending), ``centroid`` (``[n, 2]`` x, y
        microns), ``bounds`` (``[n, 4]`` x0, y0, x1, y1 microns) and
        ``area_px`` (int64), aligned by row.
    """
    from scipy import ndimage

    n_rows, n_cols = (int(v) for v in mask.shape)
    capacity = 1
    count: np.ndarray = np.zeros(capacity, dtype=np.int64)
    sum_row: np.ndarray = np.zeros(capacity, dtype=np.float64)
    sum_col: np.ndarray = np.zeros(capacity, dtype=np.float64)
    big = np.iinfo(np.int64).max
    row_min: np.ndarray = np.full(capacity, big, dtype=np.int64)
    row_max: np.ndarray = np.full(capacity, -1, dtype=np.int64)
    col_min: np.ndarray = np.full(capacity, big, dtype=np.int64)
    col_max: np.ndarray = np.full(capacity, -1, dtype=np.int64)
    for start in range(0, n_rows, max(int(strip_rows), 1)):
        strip = np.asarray(mask[start : start + strip_rows])
        if strip.size == 0:
            continue
        if strip.min() < 0:
            raise AteraProsegError("The Cellpose mask holds negative labels")
        top = int(strip.max())
        if top == 0:
            continue
        if top + 1 > capacity:
            grow = top + 1 - capacity
            count = np.concatenate([count, np.zeros(grow, np.int64)])
            sum_row = np.concatenate([sum_row, np.zeros(grow)])
            sum_col = np.concatenate([sum_col, np.zeros(grow)])
            row_min = np.concatenate([row_min, np.full(grow, big, np.int64)])
            row_max = np.concatenate([row_max, np.full(grow, -1, np.int64)])
            col_min = np.concatenate([col_min, np.full(grow, big, np.int64)])
            col_max = np.concatenate([col_max, np.full(grow, -1, np.int64)])
            capacity = top + 1
        rr, cc = np.nonzero(strip)
        values = strip[rr, cc].astype(np.int64)
        count[: top + 1] += np.bincount(values, minlength=top + 1)
        sum_row[: top + 1] += np.bincount(
            values, weights=(rr + start).astype(np.float64), minlength=top + 1
        )
        sum_col[: top + 1] += np.bincount(
            values, weights=cc.astype(np.float64), minlength=top + 1
        )
        objects = ndimage.find_objects(strip)
        for label in np.unique(values):
            rs, cs = objects[int(label) - 1]
            row_min[label] = min(row_min[label], start + rs.start)
            row_max[label] = max(row_max[label], start + rs.stop - 1)
            col_min[label] = min(col_min[label], cs.start)
            col_max[label] = max(col_max[label], cs.stop - 1)
        del rr, cc, values, objects
    present = np.nonzero(count[1:])[0] + 1
    ps = float(pixel_size_um)
    n = count[present].astype(np.float64)
    centroid = np.column_stack(
        [(sum_col[present] / n + 0.5) * ps, (sum_row[present] / n + 0.5) * ps]
    )
    bounds = np.column_stack(
        [
            col_min[present] * ps,
            row_min[present] * ps,
            (col_max[present] + 1) * ps,
            (row_max[present] + 1) * ps,
        ]
    ).astype(np.float64)
    return {
        "instance_id": present.astype(np.uint64),
        "centroid": centroid,
        "bounds": bounds,
        "area_px": count[present].astype(np.int64),
    }


def lookup_labels(
    mask: np.ndarray, x: np.ndarray, y: np.ndarray, pixel_size_um: float
) -> np.ndarray:
    """Return the label at each micron coordinate (corner rule; 0 off the mask).

    Args:
        mask: 2-D labels (a memmap is fine; only the hit pixels are read).
        x: X microns.
        y: Y microns.
        pixel_size_um: Microns per pixel.

    Returns:
        uint64 labels, 0 for background or coordinates off the mask.
    """
    from merxen.enrichment.label_lookup import PixelFrame, pixel_indices

    column, row = pixel_indices(
        np.asarray(x),
        np.asarray(y),
        PixelFrame(um_per_px=pixel_size_um),
        convention="corner",
    )
    n_rows, n_cols = mask.shape
    inside = (column >= 0) & (row >= 0) & (column < n_cols) & (row < n_rows)
    values = np.zeros(column.size, dtype=np.uint64)
    if inside.any():
        rr = row[inside]
        cc = column[inside]
        # Sorted reads keep a memmap lookup sequential.
        order = np.lexsort((cc, rr))
        found = np.empty(rr.size, dtype=np.uint64)
        found[order] = np.asarray(mask[rr[order], cc[order]], dtype=np.uint64)
        values[inside] = found
    return values
