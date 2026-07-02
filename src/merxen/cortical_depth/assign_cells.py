"""Per-cell cortical-depth assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from merxen.cortical_depth.equivolumetric import EquivolumetricResult
from merxen.cortical_depth.laplace import LaplaceSolution, interpolate_scalar_field
from merxen.cortical_depth.ribbon import RibbonGrid, points_inside_mask
from merxen.cortical_depth.streamlines import Streamline, select_valid_streamlines
from merxen.io.transcript_io import first_existing_col

CORTICAL_DEPTH_COLUMNS = [
    "inside_cortical_ribbon",
    "cortical_depth_piece_id",
    "cortical_depth_piece_mode",
    "cortical_depth_annotation",
    "laplace_depth",
    "equivolumetric_depth",
    "distance_to_pia_um",
    "distance_to_wm_um",
    "streamline_thickness_um",
    "tangential_position_um",
    "nearest_streamline_id",
    "column_id",
    "cortical_depth_qc_flag",
]


@dataclass(frozen=True)
class CellCoordinateTable:
    """Cell IDs and x/y centroids in the active coordinate system."""

    cell_ids: pd.Index
    coordinates: np.ndarray
    source: str


@dataclass(frozen=True)
class StreamlineLookup:
    """Flattened KD-tree lookup over sampled streamline points."""

    tree: cKDTree
    streamline_ids: np.ndarray
    distance_to_pia_um: np.ndarray
    thickness_um: np.ndarray
    tangential_position_um: np.ndarray
    qc_flags: np.ndarray


def extract_cell_coordinates(
    table: ad.AnnData,
    *,
    sdata_obj: Any | None = None,
    shape_key: str | None = None,
) -> CellCoordinateTable:
    """Extract per-cell centroid coordinates from an AnnData table or shapes."""
    cell_ids = _table_cell_ids(table)
    if "spatial" in table.obsm:
        coords = np.asarray(table.obsm["spatial"], dtype=float)
        if coords.ndim == 2 and coords.shape[0] == table.n_obs and coords.shape[1] >= 2:
            return CellCoordinateTable(
                cell_ids=cell_ids,
                coordinates=coords[:, :2].astype(float, copy=False),
                source="obsm:spatial",
            )

    if sdata_obj is None or shape_key is None:
        raise ValueError(
            "Cell table does not contain obsm['spatial']; provide sdata_obj and "
            "shape_key so centroids can be derived from shapes."
        )
    if shape_key not in sdata_obj.shapes:
        raise KeyError(
            f"shape_key={shape_key!r} not found. "
            f"Available shapes: {list(sdata_obj.shapes.keys())}"
        )
    metrics = shape_centroids(sdata_obj.shapes[shape_key])
    positions = pd.Series(np.arange(table.n_obs), index=cell_ids)
    common = cell_ids.intersection(metrics.index)
    if len(common) == 0:
        raise ValueError(
            f"No table cells matched shapes in {shape_key!r}; cannot assign depth."
        )
    coords = np.full((table.n_obs, 2), np.nan, dtype=float)
    coords[positions.loc[common].to_numpy(), :] = metrics.loc[
        common, ["x", "y"]
    ].to_numpy(float)
    return CellCoordinateTable(
        cell_ids=cell_ids,
        coordinates=coords,
        source=f"shapes:{shape_key}",
    )


def shape_centroids(shapes: Any) -> pd.DataFrame:
    """Return robust x/y centroids indexed by normalized cell IDs."""
    gdf = shapes.compute() if hasattr(shapes, "compute") else shapes
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf)
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame({"geometry": gdf.geometry}, index=gdf.index)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    id_col = first_existing_col(
        gdf,
        ["cell_id", "cell", "cells", "cell_ID", "region", "label_id", "EntityID"],
    )
    ids = gdf.index.astype(str) if id_col is None else gdf[id_col].astype(str)
    centroids = gdf.geometry.centroid
    x = centroids.x.to_numpy(float)
    y = centroids.y.to_numpy(float)
    bad = ~np.isfinite(x) | ~np.isfinite(y)
    if bad.any():
        reps = gdf.geometry.representative_point()
        x[bad] = reps.x.to_numpy(float)[bad]
        y[bad] = reps.y.to_numpy(float)[bad]
    out = pd.DataFrame(
        {"x": x, "y": y},
        index=pd.Index(ids.astype(str), dtype=str, name="cell_id"),
    )
    return out[np.isfinite(out["x"]) & np.isfinite(out["y"])].loc[
        lambda df: ~df.index.duplicated(keep="first")
    ]


def assign_cortical_depth_to_cells(
    coordinates: CellCoordinateTable,
    solution: LaplaceSolution,
    equivolumetric: EquivolumetricResult,
    streamlines: list[Streamline],
    *,
    side_boundary_distance_um: float = 25.0,
) -> pd.DataFrame:
    """Assign depth metrics to cell coordinates."""
    grid = solution.grid
    points = np.asarray(coordinates.coordinates, dtype=float)
    inside = points_inside_mask(points, grid)
    laplace_depth = interpolate_scalar_field(solution.phi, grid.spec, points)
    equal_area_depth = interpolate_scalar_field(
        equivolumetric.depth,
        grid.spec,
        points,
    )

    assignments = pd.DataFrame(index=coordinates.cell_ids.astype(str))
    assignments["inside_cortical_ribbon"] = inside
    assignments["laplace_depth"] = laplace_depth
    assignments["equivolumetric_depth"] = equal_area_depth

    nearest = _nearest_streamline_values(points, streamlines)
    for column, values in nearest.items():
        assignments[column] = values

    side_distance = _side_distance_at_points(grid, points)
    near_side = side_distance <= float(side_boundary_distance_um)
    assignments["cortical_depth_qc_flag"] = _qc_flags(
        inside=inside,
        laplace_depth=laplace_depth,
        streamlines=assignments["nearest_streamline_id"].to_numpy(),
        near_side=near_side,
    )
    outside = ~inside
    value_columns = [
        "laplace_depth",
        "equivolumetric_depth",
        "distance_to_pia_um",
        "distance_to_wm_um",
        "streamline_thickness_um",
        "tangential_position_um",
        "nearest_streamline_id",
        "column_id",
    ]
    for column in value_columns:
        if column in assignments.columns:
            assignments.loc[outside, column] = np.nan
    return assignments


def apply_depth_columns(table: ad.AnnData, assignments: pd.DataFrame) -> ad.AnnData:
    """Return a copy of ``table`` with cortical-depth columns in ``obs``."""
    out = table.copy()
    aligned = assignments.reindex(_table_cell_ids(out).astype(str))
    for column in CORTICAL_DEPTH_COLUMNS:
        if column in aligned.columns:
            out.obs[column] = aligned[column].to_numpy()
    return out


def assignment_summary(assignments: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable summary for one assigned cell table."""
    inside = assignments["inside_cortical_ribbon"].astype(bool)
    assigned = inside & assignments["laplace_depth"].notna()
    thickness = pd.to_numeric(assignments["streamline_thickness_um"], errors="coerce")
    finite_thickness = thickness[np.isfinite(thickness)]
    return {
        "n_cells": int(len(assignments)),
        "n_inside_ribbon": int(inside.sum()),
        "n_outside_ribbon": int((~inside).sum()),
        "n_assigned_depth": int(assigned.sum()),
        "mean_streamline_thickness_um": _series_stat(finite_thickness, "mean"),
        "median_streamline_thickness_um": _series_stat(finite_thickness, "median"),
        "min_streamline_thickness_um": _series_stat(finite_thickness, "min"),
        "max_streamline_thickness_um": _series_stat(finite_thickness, "max"),
        "qc_flag_counts": {
            str(key): int(value)
            for key, value in assignments["cortical_depth_qc_flag"]
            .astype(str)
            .value_counts(dropna=False)
            .items()
        },
    }


def _nearest_streamline_values(
    points: np.ndarray,
    streamlines: list[Streamline],
) -> dict[str, np.ndarray]:
    n_cells = int(points.shape[0])
    base = {
        "distance_to_pia_um": np.full(n_cells, np.nan, dtype=float),
        "distance_to_wm_um": np.full(n_cells, np.nan, dtype=float),
        "streamline_thickness_um": np.full(n_cells, np.nan, dtype=float),
        "tangential_position_um": np.full(n_cells, np.nan, dtype=float),
        "nearest_streamline_id": np.full(n_cells, np.nan, dtype=float),
        "column_id": np.full(n_cells, np.nan, dtype=float),
    }
    valid = select_valid_streamlines(streamlines)
    lookup = build_streamline_lookup(valid or streamlines)
    if lookup is None or n_cells == 0:
        return base
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return base
    _dist, nearest = lookup.tree.query(points[finite], k=1)
    ids = lookup.streamline_ids[nearest]
    distance_to_pia = lookup.distance_to_pia_um[nearest]
    thickness = lookup.thickness_um[nearest]
    base["distance_to_pia_um"][finite] = distance_to_pia
    base["distance_to_wm_um"][finite] = np.maximum(thickness - distance_to_pia, 0.0)
    base["streamline_thickness_um"][finite] = thickness
    base["tangential_position_um"][finite] = lookup.tangential_position_um[nearest]
    base["nearest_streamline_id"][finite] = ids.astype(float)
    base["column_id"][finite] = ids.astype(float)
    return base


def build_streamline_lookup(streamlines: list[Streamline]) -> StreamlineLookup | None:
    """Build a flattened nearest-neighbor lookup for streamline samples."""
    points: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    thicknesses: list[np.ndarray] = []
    tangent: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for streamline in streamlines:
        pts = np.asarray(streamline.points, dtype=float)
        if pts.shape[0] == 0:
            continue
        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if cumulative[-1] > 0:
            cumulative = cumulative / cumulative[-1] * float(streamline.thickness_um)
        points.append(pts)
        ids.append(np.full(pts.shape[0], int(streamline.streamline_id), dtype=np.int32))
        distances.append(cumulative.astype(float, copy=False))
        thicknesses.append(np.full(pts.shape[0], float(streamline.thickness_um)))
        tangent.append(np.full(pts.shape[0], float(streamline.tangential_position_um)))
        flags.append(np.asarray([streamline.qc_flag] * pts.shape[0], dtype=object))
    if not points:
        return None
    tree_points = np.vstack(points)
    return StreamlineLookup(
        tree=cKDTree(tree_points),
        streamline_ids=np.concatenate(ids),
        distance_to_pia_um=np.concatenate(distances),
        thickness_um=np.concatenate(thicknesses),
        tangential_position_um=np.concatenate(tangent),
        qc_flags=np.concatenate(flags),
    )


def _table_cell_ids(table: ad.AnnData) -> pd.Index:
    if "cell_id" in table.obs.columns:
        return pd.Index(table.obs["cell_id"].astype(str), dtype=str)
    return pd.Index(table.obs_names.astype(str), dtype=str)


def _side_distance_at_points(grid: RibbonGrid, points: np.ndarray) -> np.ndarray:
    if not grid.side_boundary.any():
        return np.full(points.shape[0], np.inf, dtype=float)
    side_distance = ndimage.distance_transform_edt(~grid.side_boundary)
    side_distance = side_distance * float(grid.spec.resolution_um)
    rows, cols = grid.spec.points_to_indices(points)
    keep = (
        (rows >= 0) & (rows < grid.spec.height) & (cols >= 0) & (cols < grid.spec.width)
    )
    values = np.full(points.shape[0], np.inf, dtype=float)
    values[keep] = side_distance[rows[keep], cols[keep]]
    return values


def _qc_flags(
    *,
    inside: np.ndarray,
    laplace_depth: np.ndarray,
    streamlines: np.ndarray,
    near_side: np.ndarray,
) -> list[str]:
    flags: list[str] = []
    for in_ribbon, depth, streamline_id, is_near_side in zip(
        inside,
        laplace_depth,
        streamlines,
        near_side,
        strict=False,
    ):
        if not bool(in_ribbon):
            flags.append("outside_ribbon")
        elif not np.isfinite(depth):
            flags.append("no_laplace_depth")
        elif not np.isfinite(streamline_id):
            flags.append("no_streamline")
        elif bool(is_near_side):
            flags.append("near_side_boundary")
        else:
            flags.append("assigned")
    return flags


def _series_stat(series: pd.Series, name: str) -> float | None:
    if series.empty:
        return None
    value = getattr(series, name)()
    return None if pd.isna(value) else float(value)
