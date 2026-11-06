"""ATERA_PROSEG_PLAN: the tile grid, label ownership and windows of a section.

The plan reads only small parts of the source store: the build marker and
pixel size, the vendor table's ``instance_id`` and centroid, the vendor cell
polygons' bounding boxes and the build's 100 micron density grid (a row
estimate per tile). It never scans the transcripts or the masks.

- Cores are ``[c T, (c + 1) T) x [r T, (r + 1) T)`` microns from origin
  ``(0, 0)`` with ids ``c{c}_r{r}``.
- A vendor label belongs to the tile whose half-open core holds its centroid.
- A window starts as the core grown by the halo and is then grown to cover
  the bounding box of every owned cell plus ``owned_margin_um``; it may reach
  at most ``max_window_extension_um`` beyond the halo. It is clipped to the
  mask and snapped outwards to whole pixels.
- A tile that owns no label is not run.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from merxen.config import AteraProsegPlanConfig
from merxen.io.builders.atera import read_build_marker
from merxen.io.points_aggregates import read_points_aggregates
from merxen.platforms import find_um_per_px, platform_spec
from merxen.segmentation.atera_proseg import (
    PLAN_FILE_NAME,
    PLAN_FORMAT_VERSION,
    AteraProsegError,
    tile_id,
)

logger = logging.getLogger(__name__)

SOURCE_TABLE_KEY = "table"
SOURCE_CELL_SHAPES_KEY = "cell_boundaries"
CELL_LABELS_KEY = "cell_labels"


def mask_shape(store: Path, label_key: str = CELL_LABELS_KEY) -> tuple[int, int]:
    """Return ``(n_rows, n_cols)`` of level 0 of a label element.

    Args:
        store: The store.
        label_key: Label element.

    Returns:
        The level-0 shape.
    """
    level0 = _level0(Path(store) / "labels" / label_key)
    rows, cols = (int(value) for value in level0.shape[-2:])
    return rows, cols


def _level0(path: Path) -> zarr.Array:
    """Return level 0 (``s0``, else ``0``) of a multiscale label group."""
    group = zarr.open_group(str(path), mode="r")
    level0 = group["s0"] if "s0" in group else group["0"]
    if not isinstance(level0, zarr.Array):
        raise AteraProsegError(f"{path.name} has no level-0 array")
    return level0


def read_vendor_cells(store: Path) -> dict[str, np.ndarray]:
    """Return the vendor cells' ids, centroids and polygon bounds.

    Args:
        store: The source store (``tables/table``, ``shapes/cell_boundaries``)
            or a latest store (``table_original``, ``atera_cell_boundaries``).

    Returns:
        ``instance_id`` (uint64), ``centroid`` (``[n, 2]`` microns) and
        ``bounds`` (``[n, 4]`` x0, y0, x1, y1 microns), aligned by row.

    Raises:
        AteraProsegError: If a cell lacks a polygon or ids repeat.
    """
    import anndata as ad
    import geopandas as gpd

    store = Path(store)
    spec = platform_spec("ATERA")
    table_key = (
        SOURCE_TABLE_KEY
        if (store / "tables" / SOURCE_TABLE_KEY).exists()
        else "table_original"
    )
    shapes_key = (
        SOURCE_CELL_SHAPES_KEY
        if (store / "shapes" / SOURCE_CELL_SHAPES_KEY).exists()
        else spec.original_cell_shape_key
    )
    group = zarr.open_group(str(store / "tables" / table_key), mode="r")
    obs = ad.io.read_elem(group["obs"])
    obsm = group["obsm"]
    assert isinstance(obsm, zarr.Group)
    spatial = np.asarray(ad.io.read_elem(obsm["spatial"]), dtype=np.float64)
    ids = obs["instance_id"].to_numpy(dtype=np.uint64)
    if np.unique(ids).size != ids.size:
        raise AteraProsegError("The vendor table repeats an instance_id")
    shapes = gpd.read_parquet(store / "shapes" / shapes_key / "shapes.parquet")
    shape_ids = shapes["instance_id"].to_numpy(dtype=np.uint64)
    order = np.argsort(shape_ids)
    position = np.searchsorted(shape_ids[order], ids)
    position = np.minimum(position, max(shape_ids.size - 1, 0))
    if shape_ids.size == 0 or not np.array_equal(shape_ids[order][position], ids):
        raise AteraProsegError("A vendor cell has no polygon in the cell shapes")
    bounds = shapes.geometry.bounds.to_numpy(dtype=np.float64)[order][position]
    return {"instance_id": ids, "centroid": spatial[:, :2], "bounds": bounds}


def _core_index(values: np.ndarray, tile_um: float) -> np.ndarray:
    """Grid index of each coordinate under half-open cores ``[k T, (k + 1) T)``."""
    index = np.floor(values / tile_um).astype(np.int64)
    # A value an ulp below a core edge must not fall into the next core.
    index -= values < index * tile_um
    index += values >= (index + 1) * tile_um
    return index


def assign_owners(centroids: np.ndarray, tile_um: float) -> tuple[np.ndarray, ...]:
    """Return the grid column and row of the core holding each centroid.

    Args:
        centroids: ``[n, 2]`` x, y microns.
        tile_um: Core edge.

    Returns:
        ``(columns, rows)`` int64 arrays.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    if not np.isfinite(centroids).all():
        raise AteraProsegError("A vendor cell centroid is not finite")
    return (
        _core_index(centroids[:, 0], tile_um),
        _core_index(centroids[:, 1], tile_um),
    )


def _density_estimate(
    aggregates: Any, window_um: tuple[float, float, float, float]
) -> int | None:
    """Rows of the density bins that overlap a window (an upper estimate)."""
    if aggregates is None or aggregates.density.size == 0:
        return None
    x0, y0, x1, y1 = window_um
    bin_um = float(aggregates.density_bin_um)
    ox, oy = aggregates.density_origin_um
    n_y, n_x = aggregates.density.shape
    c0 = max(int(math.floor((x0 - ox) / bin_um)), 0)
    c1 = min(int(math.ceil((x1 - ox) / bin_um)), n_x)
    r0 = max(int(math.floor((y0 - oy) / bin_um)), 0)
    r1 = min(int(math.ceil((y1 - oy) / bin_um)), n_y)
    if c1 <= c0 or r1 <= r0:
        return 0
    return int(aggregates.density[r0:r1, c0:c1].sum())


def build_plan(cfg: AteraProsegPlanConfig) -> dict[str, Any]:
    """Compute the tile plan of a source store.

    Args:
        cfg: The plan configuration.

    Returns:
        The plan as written to ``tile_plan.json``.

    Raises:
        AteraProsegError: If the store is incomplete, a window would reach
            beyond the allowed extension, or the subset names unknown tiles.
    """
    source = Path(os.path.realpath(cfg.source_zarr_path))
    marker = read_build_marker(source)
    if marker is None or not marker.get("build_fingerprint"):
        raise AteraProsegError(
            f"[{cfg.dataset_name}] The source store has no build marker; run "
            "BUILD_SPATIALDATA to completion first"
        )
    if str(marker.get("platform", "")).upper() != "ATERA":
        raise AteraProsegError(f"[{cfg.dataset_name}] The source is not an ATERA store")
    pixel_size = float(find_um_per_px(source, "ATERA"))
    n_rows, n_cols = mask_shape(source)
    extent_um = (0.0, 0.0, n_cols * pixel_size, n_rows * pixel_size)
    cells = read_vendor_cells(source)
    tile_um = float(cfg.tile_um)
    halo = float(cfg.halo_um)
    margin = float(cfg.owned_margin_um)
    max_extension = (
        halo if cfg.max_window_extension_um is None else cfg.max_window_extension_um
    )
    columns, rows = assign_owners(cells["centroid"], tile_um)
    aggregates = read_points_aggregates(source)

    grid_cols = range(
        min(0, int(columns.min()) if columns.size else 0),
        max(
            math.ceil(extent_um[2] / tile_um),
            int(columns.max()) + 1 if columns.size else 0,
        ),
    )
    grid_rows = range(
        min(0, int(rows.min()) if rows.size else 0),
        max(math.ceil(extent_um[3] / tile_um), int(rows.max()) + 1 if rows.size else 0),
    )
    known = {tile_id(c, r) for c in grid_cols for r in grid_rows}
    subset = None if cfg.tile_subset is None else [str(t) for t in cfg.tile_subset]
    if subset is not None:
        unknown = sorted(set(subset) - known)
        if unknown:
            raise AteraProsegError(
                f"[{cfg.dataset_name}] tile_subset names tiles outside the grid: "
                f"{', '.join(unknown)}"
            )
    owner_ids = np.asarray([tile_id(c, r) for c, r in zip(columns, rows, strict=True)])
    tiles: list[dict[str, Any]] = []
    n_owned_total = 0
    for row in grid_rows:
        for column in grid_cols:
            tid = tile_id(column, row)
            core = (
                column * tile_um,
                row * tile_um,
                (column + 1) * tile_um,
                (row + 1) * tile_um,
            )
            owned = owner_ids == tid if owner_ids.size else np.zeros(0, dtype=bool)
            owned_labels = np.sort(cells["instance_id"][owned])
            n_owned_total += int(owned_labels.size)
            halo_window = (
                core[0] - halo,
                core[1] - halo,
                core[2] + halo,
                core[3] + halo,
            )
            window = list(halo_window)
            if owned_labels.size:
                box = cells["bounds"][owned]
                window[0] = min(window[0], float(box[:, 0].min()) - margin)
                window[1] = min(window[1], float(box[:, 1].min()) - margin)
                window[2] = max(window[2], float(box[:, 2].max()) + margin)
                window[3] = max(window[3], float(box[:, 3].max()) + margin)
                reach = max(
                    halo_window[0] - window[0],
                    halo_window[1] - window[1],
                    window[2] - halo_window[2],
                    window[3] - halo_window[3],
                )
                if reach > max_extension + 1e-9:
                    raise AteraProsegError(
                        f"[{cfg.dataset_name}] Tile {tid}: an owned cell reaches "
                        f"{reach:.1f} microns beyond the halo (at most "
                        f"{max_extension:.1f}); raise the halo or "
                        "max_window_extension_um"
                    )
            clipped = (
                max(window[0], extent_um[0]),
                max(window[1], extent_um[1]),
                min(window[2], extent_um[2]),
                min(window[3], extent_um[3]),
            )
            col0 = max(int(math.floor(clipped[0] / pixel_size)), 0)
            row0 = max(int(math.floor(clipped[1] / pixel_size)), 0)
            col1 = min(int(math.ceil(clipped[2] / pixel_size)), n_cols)
            row1 = min(int(math.ceil(clipped[3] / pixel_size)), n_rows)
            window_um = (
                col0 * pixel_size,
                row0 * pixel_size,
                col1 * pixel_size,
                row1 * pixel_size,
            )
            run = bool(owned_labels.size) and col1 > col0 and row1 > row0
            reason = None
            if not owned_labels.size:
                reason = "no vendor cell centroid in the core"
            elif not run:
                reason = "the window lies outside the mask"
            if run and subset is not None and tid not in subset:
                run, reason = False, "not in tile_subset"
            if not owned_labels.size and (col1 <= col0 or row1 <= row0):
                # A grid cell entirely off the mask with no cells: not listed.
                continue
            tiles.append(
                {
                    "tile_id": tid,
                    "grid": [int(column), int(row)],
                    "core_um": [float(v) for v in core],
                    "window_um": [float(v) for v in window_um],
                    "window_px": [col0, col1, row0, row1],
                    "n_owned": int(owned_labels.size),
                    "owned_labels": [int(v) for v in owned_labels],
                    "est_rows": _density_estimate(aggregates, window_um),
                    "run": run,
                    "reason": reason,
                }
            )
    if n_owned_total != cells["instance_id"].size:
        raise AteraProsegError(
            f"[{cfg.dataset_name}] {cells['instance_id'].size - n_owned_total} "
            "vendor labels have no owning tile"
        )
    plan = {
        "format_version": PLAN_FORMAT_VERSION,
        "dataset_name": cfg.dataset_name,
        "platform": "ATERA",
        "source_build_fingerprint": str(marker["build_fingerprint"]),
        "pixel_size_um": pixel_size,
        "mask_shape_px": [n_rows, n_cols],
        "grid": {
            "tile_um": tile_um,
            "halo_um": halo,
            "owned_margin_um": margin,
            "max_window_extension_um": float(max_extension),
            "origin_um": [0.0, 0.0],
        },
        "tile_subset": subset,
        "n_labels": int(cells["instance_id"].size),
        "n_tiles_run": sum(1 for tile in tiles if tile["run"]),
        "tiles": tiles,
    }
    logger.info(
        "[%s] Tile plan: %d tiles, %d run, %d vendor labels",
        cfg.dataset_name,
        len(tiles),
        plan["n_tiles_run"],
        plan["n_labels"],
    )
    return plan


def write_plan(cfg: AteraProsegPlanConfig) -> Path:
    """Compute the plan and write ``<output_dir>/tile_plan.json``.

    Args:
        cfg: The plan configuration.

    Returns:
        The written plan file.
    """
    plan = build_plan(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / PLAN_FILE_NAME
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


def read_plan(path: Path) -> dict[str, Any]:
    """Read a ``tile_plan.json``.

    Args:
        path: The plan file.

    Returns:
        The plan.
    """
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("format_version") != PLAN_FORMAT_VERSION:
        raise AteraProsegError(
            f"Unsupported tile plan version {plan.get('format_version')}"
        )
    return dict(plan)
