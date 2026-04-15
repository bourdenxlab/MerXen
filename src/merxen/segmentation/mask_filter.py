"""Mask filtering utilities vendored from MOSAIK.

This module keeps the core regionprops-based filtering logic used by the
original notebook pipeline while trimming unrelated dependencies.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterable

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
