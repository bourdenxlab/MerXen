"""Lazy pyramid + outline construction, ported from the napari viewer.

These functions reproduce the viewer's ``lazy_coarsened_pyramid`` /
``lazy_label_pyramid`` / outline builders exactly so a pre-built cache renders
identically to one the viewer would have made. The multiscale-tree assembly
reuses SpatialData's own ``dask_arrays_to_datatree`` (the same utility the
viewer calls), keeping the on-disk pyramid layout identical.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
from spatialdata.transformations import set_transformation

from merxen.viewer_cache.format import (
    PYRAMID_MAX_LEVELS,
    PYRAMID_MIN_SIZE,
    PYRAMID_TILE,
)


def lazy_coarsened_pyramid(
    base_data: Any,
    step: int,
    reducer: Any = None,
    min_size: int = PYRAMID_MIN_SIZE,
    max_levels: int = PYRAMID_MAX_LEVELS,
    tile: int = PYRAMID_TILE,
) -> list[Any]:
    """Build materialized-ready coarse levels for an image or label array.

    Each level downsamples the trailing ``(y, x)`` axes by ``step`` using
    ``reducer`` (``np.mean`` for intensity images, ``np.max`` for label ids so
    ids survive). Level 0 (the base) is intentionally EXCLUDED; the viewer reuses
    the existing lazy base array so the multi-gigapixel scale0 is never
    duplicated. Ported verbatim from the viewer's ``lazy_coarsened_pyramid``.
    """
    if reducer is None:
        reducer = np.mean
    step = max(2, int(step))
    data = da.asarray(base_data)
    ndim = data.ndim
    if ndim not in (2, 3):
        return []
    y_axis, x_axis = ndim - 2, ndim - 1
    dtype = data.dtype
    is_integer = np.issubdtype(dtype, np.integer)

    levels: list[Any] = []
    current = data
    prev_shape = tuple(int(s) for s in data.shape)
    while (
        len(levels) < max_levels
        and max(int(current.shape[y_axis]), int(current.shape[x_axis])) > min_size
    ):
        if int(current.shape[y_axis]) < step or int(current.shape[x_axis]) < step:
            break
        coarsened = da.coarsen(
            reducer, current, axes={y_axis: step, x_axis: step}, trim_excess=True
        )
        if reducer is np.mean and is_integer:
            coarsened = da.rint(coarsened).astype(dtype)
        else:
            coarsened = coarsened.astype(dtype)
        new_shape = tuple(int(s) for s in coarsened.shape)
        if new_shape[-2:] == prev_shape[-2:]:
            break
        chunks: tuple[int, ...]
        if ndim == 3:
            chunks = (new_shape[0], min(tile, new_shape[1]), min(tile, new_shape[2]))
        else:
            chunks = (min(tile, new_shape[0]), min(tile, new_shape[1]))
        coarsened = coarsened.rechunk(chunks)
        levels.append(coarsened)
        current = coarsened
        prev_shape = new_shape
    return levels


def lazy_label_pyramid(
    label_data: Any,
    min_size: int = PYRAMID_MIN_SIZE,
    max_levels: int = PYRAMID_MAX_LEVELS,
) -> list[Any]:
    """Build a lazy 2D label pyramid by max-pooling label ids (base included)."""
    data = da.asarray(label_data)
    levels: list[Any] = [data]
    while len(levels) < max_levels and max(int(axis) for axis in data.shape) > min_size:
        data = da.coarsen(np.max, data, axes={0: 2, 1: 2}, trim_excess=True)
        if data.shape == levels[-1].shape:
            break
        levels.append(data)
    return levels


def label_outline_mask_chunk(labels: Any, width: int = 1) -> np.ndarray:
    """Return a uint8 outline mask for a 2D label tile (viewer-verbatim)."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label tile, got shape {arr.shape}")

    fg = arr != 0
    outline = np.zeros(arr.shape, dtype=bool)
    outline[1:, :] |= fg[1:, :] & (arr[1:, :] != arr[:-1, :])
    outline[:-1, :] |= fg[:-1, :] & (arr[:-1, :] != arr[1:, :])
    outline[:, 1:] |= fg[:, 1:] & (arr[:, 1:] != arr[:, :-1])
    outline[:, :-1] |= fg[:, :-1] & (arr[:, :-1] != arr[:, 1:])

    width = int(width)
    if width > 1 and np.any(outline):
        try:
            from scipy.ndimage import binary_dilation

            outline = binary_dilation(outline, iterations=width - 1)
        except Exception:  # noqa: BLE001
            for _ in range(width - 1):
                expanded = outline.copy()
                expanded[1:, :] |= outline[:-1, :]
                expanded[:-1, :] |= outline[1:, :]
                expanded[:, 1:] |= outline[:, :-1]
                expanded[:, :-1] |= outline[:, 1:]
                outline = expanded

    return outline.astype(np.uint8, copy=False)


def lazy_outline_mask(label_data: Any, width: int) -> Any:
    """Build one lazy uint8 outline mask from one 2D label level."""
    width = max(1, int(width))
    labels = da.asarray(label_data)
    return labels.map_overlap(
        label_outline_mask_chunk,
        depth=max(1, width),
        boundary=0,
        trim=True,
        dtype=np.uint8,
        width=width,
    )


def _outline_width_for_level(
    width: int, base_shape: tuple[int, ...], level_shape: tuple[int, ...]
) -> int:
    """Scale outline width down for coarser pyramid levels (viewer-verbatim)."""
    if width <= 1:
        return 1
    y_factor = float(base_shape[0]) / max(1.0, float(level_shape[0]))
    x_factor = float(base_shape[1]) / max(1.0, float(level_shape[1]))
    scale_factor = max(1.0, y_factor, x_factor)
    return max(1, int(np.ceil(float(width) / scale_factor)))


def lazy_outline_pyramid_from_label_levels(
    label_levels: list[Any], width: int
) -> list[Any]:
    """Build outline masks independently from existing/synthetic label levels."""
    if len(label_levels) == 0:
        return []
    base_shape = tuple(int(axis) for axis in label_levels[0].shape)
    outlines: list[Any] = []
    for level in label_levels:
        level_shape = tuple(int(axis) for axis in level.shape)
        level_width = _outline_width_for_level(int(width), base_shape, level_shape)
        outlines.append(lazy_outline_mask(level, width=level_width))
    return outlines


def lazy_outline_pyramid(
    label_data: Any,
    width: int,
    min_size: int = PYRAMID_MIN_SIZE,
    max_levels: int = PYRAMID_MAX_LEVELS,
) -> list[Any]:
    """Build a lazy multiscale uint8 outline pyramid from a 2D label image."""
    width = max(1, int(width))
    label_levels = lazy_label_pyramid(
        label_data, min_size=min_size, max_levels=max_levels
    )
    return lazy_outline_pyramid_from_label_levels(label_levels, width=width)


def build_multiscale_tree(
    levels: list[Any],
    dims: tuple[str, ...],
    transform: Any,
    channels: list[Any] | None = None,
    dtype: Any = None,
) -> Any:
    """Assemble levels into a SpatialData multiscale DataTree with a transform.

    Mirrors the viewer's ``_datatree_from_levels``: cast each level, build the
    tree with SpatialData's ``dask_arrays_to_datatree``, and stamp the same
    ``global`` transform on every scale.
    """
    arrays = []
    for level in levels:
        arr = da.asarray(level)
        if dtype is not None:
            arr = arr.astype(dtype)
        arrays.append(arr)
    tree = dask_arrays_to_datatree(arrays, dims=dims, channels=channels)
    set_transformation(tree, {"global": transform}, set_all=True)
    return tree
