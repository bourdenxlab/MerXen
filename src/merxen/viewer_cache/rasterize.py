"""Polygon-to-label rasterization, ported from the napari viewer.

The pixel value written for each polygon is its *instance id* (the GeoDataFrame
index value) -- the same key the transcript ``assignment`` column, the
clustering-table instance key, and per-cell value tables join on. This must
match the viewer's ``_label_ids_for_shapes`` / ``rasterize_geometries_chunk``
exactly, or cell-type / value overlays colour the wrong cells.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

logger = logging.getLogger(__name__)


def coords_origin_step(coords: Any) -> tuple[float, float]:
    """Infer origin and step for a monotonic coordinate array (viewer-verbatim)."""
    if coords is None:
        return 0.0, 1.0
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    if arr.size == 1:
        return float(arr[0]), 1.0
    diffs = np.diff(arr)
    step = float(np.median(diffs))
    if not np.allclose(diffs, step, rtol=1e-3, atol=1e-6):
        step = float(diffs[0])
    return float(arr[0]), float(step)


def napari_affine_from_px_to_um(
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
    x_coords: Any = None,
    y_coords: Any = None,
) -> np.ndarray:
    """Build a 3x3 affine mapping (row, col) -> (y_um, x_um) (viewer-verbatim).

    ``x_transform``/``y_transform`` are the pixel->micron affine rows
    ``x_um = a*x_px + b*y_px + c`` and ``y_um = d*x_px + e*y_px + f``.
    """
    a, b, c = map(float, x_transform)
    d, e, f = map(float, y_transform)
    x_origin, x_step = coords_origin_step(x_coords)
    y_origin, y_step = coords_origin_step(y_coords)
    return np.array(
        [
            [e * y_step, d * x_step, d * x_origin + e * y_origin + f],  # y_um
            [b * y_step, a * x_step, a * x_origin + b * y_origin + c],  # x_um
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def pixel_window_global_bounds(
    affine: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[float, float, float, float]:
    """Return global x/y bounds (minx, miny, maxx, maxy) for a pixel window."""
    corners = np.array(
        [
            [float(y0), float(x0), 1.0],
            [float(y0), float(x1), 1.0],
            [float(y1), float(x0), 1.0],
            [float(y1), float(x1), 1.0],
        ],
        dtype=float,
    )
    yx = corners @ np.asarray(affine, dtype=float).T
    y_vals = yx[:, 0]
    x_vals = yx[:, 1]
    return (
        float(np.nanmin(x_vals)),
        float(np.nanmin(y_vals)),
        float(np.nanmax(x_vals)),
        float(np.nanmax(y_vals)),
    )


def query_geometries_for_bounds(
    gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Return geometries whose bounds intersect a global x/y bounding box."""
    query_box = box(*bounds)
    try:
        idx = gdf.sindex.query(query_box, predicate="intersects")
        return gdf.iloc[np.asarray(idx, dtype=np.int64)]
    except Exception:  # noqa: BLE001
        intersects = gdf.geometry.intersects(query_box)
        return gdf.loc[intersects]


def label_ids_for_shapes(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, Any]:
    """Return ``(id_series, dtype)`` giving each polygon its raster label id.

    Ported from the viewer's ``_label_ids_for_shapes``: each polygon is labelled
    with its own GeoDataFrame index value (the instance id). ``id_series`` is
    indexed by ``gdf.index``. ``dtype`` is the smallest unsigned integer holding
    every id (uint32, or uint64 for large ids such as merscope EntityIDs). Id 0
    collides with the transparent background; no real segmentation uses cell 0.
    """
    index = pd.Series(np.asarray(gdf.index), index=gdf.index)
    numeric = pd.to_numeric(index, errors="coerce")
    is_integer_ids = (
        len(numeric) > 0
        and numeric.notna().all()
        and (numeric >= 0).all()
        and np.array_equal(numeric.to_numpy(), np.floor(numeric.to_numpy()))
    )
    if is_integer_ids:
        ids = numeric.astype("int64")
    else:
        logger.warning(
            "Shapes index is not a non-negative integer id; rasterizing with "
            "positional codes. Cell-type colouring may not join for this mask."
        )
        ids = pd.Series(pd.factorize(np.asarray(gdf.index))[0] + 1, index=gdf.index)
    max_id = int(ids.max()) if len(ids) else 0
    dtype = np.uint32 if max_id <= np.iinfo(np.uint32).max else np.uint64
    return ids.astype(dtype), dtype


def rasterize_geometries_chunk(
    geometries: Any,
    labels: Any,
    shape: tuple[int, int],
    inv_affine: np.ndarray,
    y0: int = 0,
    x0: int = 0,
    dtype: Any = np.uint32,
) -> np.ndarray:
    """Rasterize global x/y geometries into one local label chunk (viewer-verbatim)."""
    from skimage.draw import polygon as draw_polygon

    out = np.zeros(tuple(int(v) for v in shape), dtype=dtype)
    inv = np.asarray(inv_affine, dtype=float)
    if inv.shape != (3, 3) or not np.isfinite(inv).all():
        raise ValueError("inv_affine must be a finite 3x3 matrix.")

    def draw_ring(coords: Any, value: Any) -> None:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
            return
        if not np.isfinite(coords[:, :2]).all():
            return
        xs = coords[:, 0]
        ys = coords[:, 1]
        rows = ys * inv[0, 0] + xs * inv[0, 1] + inv[0, 2] - float(y0)
        cols = ys * inv[1, 0] + xs * inv[1, 1] + inv[1, 2] - float(x0)
        if not np.isfinite(rows).all() or not np.isfinite(cols).all():
            return
        rr, cc = draw_polygon(rows, cols, shape=out.shape)
        if rr.size:
            out[rr, cc] = value

    for geom, label in zip(geometries, labels, strict=False):
        if geom is None or geom.is_empty:
            continue
        value = np.asarray(label, dtype=dtype).item()
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else (geom,)
        for part in parts:
            if part.is_empty:
                continue
            draw_ring(part.exterior.coords, value)
            for interior in part.interiors:
                draw_ring(interior.coords, 0)

    return out
