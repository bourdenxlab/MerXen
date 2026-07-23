"""Mask-to-polygon geometry utilities vendored from MOSAIK."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterable

import numpy as np
from scipy.ndimage import find_objects
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from skimage.measure import find_contours
from tqdm.auto import tqdm

_POOL_SEG_MASKS: np.ndarray | None = None


def _polygon_from_bbox_with_mask(
    seg_masks: np.ndarray,
    label_and_slice: tuple[int, tuple[slice, slice]],
) -> BaseGeometry | None:
    """Extract one polygon for a labeled region using a tight bounding box."""
    label_id, slc = label_and_slice
    local_mask = seg_masks[slc] == label_id
    if not local_mask.any():
        return None

    # Add a one-pixel border so contours are not clipped by bbox edges.
    padded_mask = np.pad(local_mask.astype(np.uint8), pad_width=1, mode="constant")
    if padded_mask.shape[0] < 2 or padded_mask.shape[1] < 2:
        return None

    contours = find_contours(padded_mask, level=0.5)
    if not contours:
        return None

    contour_polygons: list[BaseGeometry] = []
    for contour in contours:
        contour[:, 0] += slc[0].start - 1
        contour[:, 1] += slc[1].start - 1
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        polygon = Polygon(contour[:, [1, 0]])
        if polygon.is_valid and not polygon.is_empty:
            contour_polygons.append(polygon)
    if not contour_polygons:
        return None

    geometry = contour_polygons[0]
    for polygon in contour_polygons[1:]:
        geometry = geometry.symmetric_difference(polygon)
    geometry = geometry.buffer(0)
    if geometry.is_valid and not geometry.is_empty:
        return geometry
    return None


def _pool_init(seg_masks: np.ndarray) -> None:
    """Set process-local segmentation mask for worker processes."""
    global _POOL_SEG_MASKS
    _POOL_SEG_MASKS = seg_masks


def _polygon_from_bbox_pool(
    label_and_slice: tuple[int, tuple[slice, slice]],
) -> BaseGeometry | None:
    """Multiprocessing wrapper for polygon extraction."""
    if _POOL_SEG_MASKS is None:
        return None
    return _polygon_from_bbox_with_mask(_POOL_SEG_MASKS, label_and_slice)


def _iter_polygons_serial(
    seg_masks: np.ndarray,
    label_slices: list[tuple[int, tuple[slice, slice]]],
    *,
    show_progress: bool,
) -> list[BaseGeometry]:
    """Serial polygon extraction fallback."""
    out: list[BaseGeometry] = []
    iterator: Iterable[tuple[int, tuple[slice, slice]]] = label_slices
    if show_progress:
        iterator = tqdm(label_slices, total=len(label_slices), desc="masks_to_polygons")
    for item in iterator:
        poly = _polygon_from_bbox_with_mask(seg_masks, item)
        if poly is not None:
            out.append(poly)
    return out


def masks_to_polygons(
    seg_masks: np.ndarray,
    factor_rescale: float = 0.0,
    n_jobs: int | None = None,
    show_progress: bool = False,
) -> list[BaseGeometry]:
    """Convert labeled masks to polygons.

    Args:
        seg_masks: 2D labeled mask array.
        factor_rescale: Optional isotropic polygon scaling factor. If 0,
            scaling is skipped for compatibility with the original notebook.
        n_jobs: Number of worker processes. ``None`` uses all CPUs.
        show_progress: Whether to display progress bars.

    Returns:
        List of valid polygons, one per connected component that could be
        converted.
    """
    return [
        polygon
        for _, polygon in masks_to_labeled_polygons(
            seg_masks,
            factor_rescale=factor_rescale,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
    ]


def masks_to_labeled_polygons(
    seg_masks: np.ndarray,
    factor_rescale: float = 0.0,
    n_jobs: int | None = None,
    show_progress: bool = False,
) -> list[tuple[int, BaseGeometry]]:
    """Convert a label image to polygons without losing label identity.

    Args:
        seg_masks: Two-dimensional labeled mask array.
        factor_rescale: Optional isotropic polygon scaling factor.
        n_jobs: Number of worker processes. ``None`` uses all CPUs.
        show_progress: Whether to display progress bars.

    Returns:
        Pairs of original positive label ID and valid polygon.
    """
    label_slices = [
        (label_id, slc)
        for label_id, slc in enumerate(find_objects(seg_masks), start=1)
        if slc is not None
    ]
    if not label_slices:
        return []

    n_jobs = max(1, os.cpu_count() or 1) if n_jobs is None else max(1, int(n_jobs))
    use_parallel = len(label_slices) >= 64 and n_jobs > 1
    labeled_polygons: list[tuple[int, BaseGeometry]]

    if use_parallel:
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=n_jobs, initializer=_pool_init, initargs=(seg_masks,)
            ) as pool:
                results = pool.imap(_polygon_from_bbox_pool, label_slices, chunksize=16)
                if show_progress:
                    results = tqdm(
                        results,
                        total=len(label_slices),
                        desc="masks_to_polygons",
                    )
                labeled_polygons = [
                    (label_id, polygon)
                    for (label_id, _), polygon in zip(
                        label_slices,
                        results,
                        strict=True,
                    )
                    if polygon is not None
                ]
        except Exception:
            labeled_polygons = []
            for label_id, label_slice in label_slices:
                polygon = _polygon_from_bbox_with_mask(
                    seg_masks,
                    (label_id, label_slice),
                )
                if polygon is not None:
                    labeled_polygons.append((label_id, polygon))
    else:
        iterator: Iterable[tuple[int, tuple[slice, slice]]] = label_slices
        if show_progress:
            iterator = tqdm(
                label_slices,
                total=len(label_slices),
                desc="masks_to_polygons",
            )
        labeled_polygons = []
        for label_id, label_slice in iterator:
            polygon = _polygon_from_bbox_with_mask(
                seg_masks,
                (label_id, label_slice),
            )
            if polygon is not None:
                labeled_polygons.append((label_id, polygon))

    if factor_rescale != 0:
        labeled_polygons = [
            (
                label_id,
                scale(
                    polygon,
                    xfact=float(factor_rescale),
                    yfact=float(factor_rescale),
                    origin=(0, 0),
                ),
            )
            for label_id, polygon in labeled_polygons
        ]
    return labeled_polygons
