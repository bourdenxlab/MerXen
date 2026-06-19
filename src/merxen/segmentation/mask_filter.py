"""Mask filtering utilities vendored from MOSAIK.

This module keeps the core regionprops-based filtering logic used by the
original notebook pipeline while trimming unrelated dependencies.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
from tqdm.auto import tqdm

_POOL_LABELED_MASKS: np.ndarray | None = None


def _region_stats_from_labeled_with_slice(
    labeled_masks: np.ndarray,
    label_and_slice: tuple[int, tuple[slice, slice]],
) -> tuple[int, float, float] | None:
    """Return ``(label_id, area, eccentricity)`` for one connected component."""
    label_id, slc = label_and_slice
    local_mask = labeled_masks[slc] == label_id
    if not local_mask.any():
        return None

    props = regionprops(local_mask.astype(np.uint8))
    if not props:
        return None

    region = props[0]
    return (label_id, float(region.area), float(region.eccentricity))


def _pool_init_labeled(labeled_masks: np.ndarray) -> None:
    """Set process-local global state for multiprocessing workers."""
    global _POOL_LABELED_MASKS
    _POOL_LABELED_MASKS = labeled_masks


def _region_stats_pool(
    label_and_slice: tuple[int, tuple[slice, slice]],
) -> tuple[int, float, float] | None:
    """Multiprocessing wrapper for per-label region statistics."""
    if _POOL_LABELED_MASKS is None:
        return None
    return _region_stats_from_labeled_with_slice(_POOL_LABELED_MASKS, label_and_slice)


def _compute_stats_serial(
    label_slices: list[tuple[int, tuple[slice, slice]]],
    labeled_masks: np.ndarray,
    *,
    show_progress: bool,
) -> list[tuple[int, float, float]]:
    """Compute per-label area/eccentricity without multiprocessing."""
    iterator: Iterable[tuple[int, tuple[slice, slice]]] = label_slices
    if show_progress:
        iterator = tqdm(
            label_slices,
            total=len(label_slices),
            desc="filter_masks_basic",
        )
    return [
        x
        for x in (
            _region_stats_from_labeled_with_slice(labeled_masks, item)
            for item in iterator
        )
        if x is not None
    ]


def filter_cell_by_regionprops(
    seg_masks: np.ndarray,
    max_eccentricity: float = 0.95,
    n_jobs: int | None = None,
    show_progress: bool = False,
    min_area_percentile: float = 10.0,
    min_area_px: float | None = None,
) -> np.ndarray:
    """Filter segmented masks by area and eccentricity.

    Args:
        seg_masks: Input segmentation mask. Non-zero pixels are treated as
            foreground and relabeled into connected components.
        max_eccentricity: Maximum allowed eccentricity for kept regions.
        n_jobs: Number of worker processes. ``None`` uses all available cores.
        show_progress: Whether to display progress bars.
        min_area_percentile: Percentile-derived area threshold if
            ``min_area_px`` is not provided.
        min_area_px: Absolute minimum area threshold in pixels.

    Returns:
        A relabeled mask (``int32``) containing only regions that pass filters.
    """
    labeled_masks = np.asarray(label(seg_masks), dtype=np.int32)
    max_label = int(labeled_masks.max())
    if max_label == 0:
        return np.zeros_like(seg_masks, dtype=np.int32)

    label_slices = [
        (label_id, slc)
        for label_id, slc in enumerate(find_objects(labeled_masks), start=1)
        if slc is not None
    ]
    if not label_slices:
        return np.zeros_like(seg_masks, dtype=np.int32)

    n_jobs = max(1, os.cpu_count() or 1) if n_jobs is None else max(1, int(n_jobs))
    use_parallel = n_jobs > 1 and len(label_slices) >= 64
    stats: list[tuple[int, float, float]]

    if use_parallel:
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=n_jobs,
                initializer=_pool_init_labeled,
                initargs=(labeled_masks,),
            ) as pool:
                results = pool.imap(_region_stats_pool, label_slices, chunksize=32)
                if show_progress:
                    results = tqdm(
                        results,
                        total=len(label_slices),
                        desc="filter_masks_basic",
                    )
                stats = [x for x in results if x is not None]
        except Exception:
            stats = _compute_stats_serial(
                label_slices,
                labeled_masks,
                show_progress=show_progress,
            )
    else:
        stats = _compute_stats_serial(
            label_slices,
            labeled_masks,
            show_progress=show_progress,
        )

    if not stats:
        return np.zeros_like(seg_masks, dtype=np.int32)

    # Preserve original label order semantics from MOSAIK.
    stats.sort(key=lambda x: x[0])
    areas = np.array([s[1] for s in stats], dtype=np.float64)
    if min_area_px is not None:
        min_area = float(min_area_px)
    else:
        area_pct = min(100.0, max(0.0, float(min_area_percentile)))
        min_area = float(np.percentile(areas, area_pct))

    keep_labels = np.array(
        [s[0] for s in stats if s[1] >= min_area and s[2] <= max_eccentricity],
        dtype=np.int32,
    )
    if keep_labels.size == 0:
        return np.zeros_like(seg_masks, dtype=np.int32)

    label_map = np.zeros(max_label + 1, dtype=np.int32)
    label_map[keep_labels] = np.arange(1, keep_labels.size + 1, dtype=np.int32)
    return label_map[labeled_masks]


def _chunk_rows_for_target(
    shape: tuple[int, int],
    dtype: np.dtype[Any],
    chunk_mb: int,
) -> int:
    """Return row count targeting a bounded temporary output chunk."""
    height, width = shape
    target_bytes = max(1, int(chunk_mb)) * 1024 * 1024
    row_bytes = max(1, int(width) * np.dtype(dtype).itemsize)
    return max(1, min(int(height), target_bytes // row_bytes))


def filter_labeled_mask_by_area(
    mask_path: Path | str,
    *,
    pixel_area_um2: float,
    min_area_um2: float | None = None,
    max_area_um2: float | None = None,
    output_path: Path | str | None = None,
    chunk_mb: int = 256,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Remove labeled mask objects outside absolute area bounds.

    The input ``.npy`` mask is loaded as a memory map, label areas are computed
    with ``bincount``, and the filtered mask is streamed to a temporary ``.npy``
    in row chunks before replacing the original file. Kept labels are compactly
    relabeled from 1..N so downstream transcript seed IDs remain dense.
    """
    mask_path = Path(mask_path)
    output_path = Path(output_path) if output_path is not None else mask_path
    min_area_um2 = None if min_area_um2 is None else float(min_area_um2)
    max_area_um2 = None if max_area_um2 is None else float(max_area_um2)
    pixel_area_um2 = float(pixel_area_um2)

    if min_area_um2 is None and max_area_um2 is None:
        return {
            "mask_path": str(output_path),
            "pixel_area_um2": pixel_area_um2,
            "n_labels": 0,
            "n_kept": 0,
            "n_removed_small": 0,
            "n_removed_large": 0,
            "written": False,
        }
    if not np.isfinite(pixel_area_um2) or pixel_area_um2 <= 0:
        raise ValueError(f"pixel_area_um2 must be positive, got {pixel_area_um2!r}")
    if min_area_um2 is not None and min_area_um2 < 0:
        raise ValueError(f"min_area_um2 must be non-negative, got {min_area_um2!r}")
    if max_area_um2 is not None and max_area_um2 < 0:
        raise ValueError(f"max_area_um2 must be non-negative, got {max_area_um2!r}")
    if (
        min_area_um2 is not None
        and max_area_um2 is not None
        and min_area_um2 > max_area_um2
    ):
        raise ValueError(
            "min_area_um2 must be <= max_area_um2 "
            f"(got {min_area_um2:g} > {max_area_um2:g})"
        )

    mask_mmap = np.load(mask_path, mmap_mode="r")
    if mask_mmap.ndim != 2:
        raise ValueError(f"Expected a 2D mask array, got shape={mask_mmap.shape}")
    if not np.issubdtype(mask_mmap.dtype, np.integer):
        raise TypeError(f"Expected an integer labeled mask, got {mask_mmap.dtype}")
    if np.issubdtype(mask_mmap.dtype, np.signedinteger) and int(mask_mmap.min()) < 0:
        raise ValueError("Labeled masks must not contain negative labels")

    max_label = int(mask_mmap.max())
    if max_label == 0:
        del mask_mmap
        return {
            "mask_path": str(output_path),
            "pixel_area_um2": pixel_area_um2,
            "n_labels": 0,
            "n_kept": 0,
            "n_removed_small": 0,
            "n_removed_large": 0,
            "written": False,
        }

    counts = np.bincount(mask_mmap.reshape(-1), minlength=max_label + 1)
    foreground = counts > 0
    foreground[0] = False
    areas_um2 = counts.astype(np.float64) * pixel_area_um2

    keep = foreground.copy()
    small = np.zeros_like(foreground)
    large = np.zeros_like(foreground)
    if min_area_um2 is not None:
        small = foreground & (areas_um2 < min_area_um2)
        keep &= ~small
    if max_area_um2 is not None:
        large = foreground & (areas_um2 > max_area_um2)
        keep &= ~large

    n_labels = int(np.count_nonzero(foreground))
    n_kept = int(np.count_nonzero(keep))
    n_removed_small = int(np.count_nonzero(small))
    n_removed_large = int(np.count_nonzero(large))

    if n_kept == n_labels and output_path == mask_path:
        del mask_mmap
        return {
            "mask_path": str(output_path),
            "pixel_area_um2": pixel_area_um2,
            "n_labels": n_labels,
            "n_kept": n_kept,
            "n_removed_small": n_removed_small,
            "n_removed_large": n_removed_large,
            "min_area_um2": min_area_um2,
            "max_area_um2": max_area_um2,
            "written": False,
        }

    dtype = np.dtype(mask_mmap.dtype)
    if n_kept > np.iinfo(dtype).max:
        raise OverflowError(f"Cannot relabel {n_kept:,} masks into dtype {dtype}")

    label_map = np.zeros(max_label + 1, dtype=dtype)
    label_map[keep] = np.arange(1, n_kept + 1, dtype=dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}.npy")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        out_mem = np.lib.format.open_memmap(
            str(tmp_path),
            mode="w+",
            dtype=dtype,
            shape=mask_mmap.shape,
        )
        chunk_rows = _chunk_rows_for_target(mask_mmap.shape, dtype, chunk_mb)
        row_starts: Iterable[int] = range(0, int(mask_mmap.shape[0]), chunk_rows)
        if show_progress:
            row_starts = tqdm(
                row_starts,
                total=int(np.ceil(mask_mmap.shape[0] / chunk_rows)),
                desc="area-filter mask",
                unit="chunk",
            )
        for y0 in row_starts:
            y1 = min(int(mask_mmap.shape[0]), int(y0) + chunk_rows)
            out_mem[y0:y1] = label_map[mask_mmap[y0:y1]]
        out_mem.flush()
        del out_mem, mask_mmap
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {
        "mask_path": str(output_path),
        "pixel_area_um2": pixel_area_um2,
        "n_labels": n_labels,
        "n_kept": n_kept,
        "n_removed_small": n_removed_small,
        "n_removed_large": n_removed_large,
        "min_area_um2": min_area_um2,
        "max_area_um2": max_area_um2,
        "written": True,
    }
