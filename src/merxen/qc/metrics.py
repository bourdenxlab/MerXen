"""Dataset-level QC metric computation."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd

from merxen.io.spatialdata_schema import choose_primary_points_key
from merxen.io.transcript_io import (
    assignment_mask_from_points,
    first_existing_col,
    to_pandas,
)
from merxen.memory import force_release

logger = logging.getLogger(__name__)


def _primary_polygon(geom: Any) -> Any | None:
    """Return a representative polygon for Polygon/MultiPolygon geometry."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return None


def _eccentricity_aspect(geom: Any) -> tuple[float, float]:
    """Estimate eccentricity and aspect ratio from polygon coordinates."""
    poly = _primary_polygon(geom)
    if poly is None:
        return np.nan, np.nan

    coords = np.asarray(poly.exterior.coords)
    if coords.shape[0] < 5:
        return np.nan, np.nan

    xy = coords[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.clip(eigvals, 1e-12, None))
    major = float(np.sqrt(eigvals[1]))
    minor = float(np.sqrt(eigvals[0]))
    if major <= 0:
        return np.nan, np.nan

    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor**2) / (major**2))))
    aspect_ratio = float(major / minor) if minor > 0 else np.nan
    return eccentricity, aspect_ratio


def _compute_cell_metrics_from_points(
    points_obj: Any,
    assign_col: str,
    gene_col: str,
) -> tuple[int, int, pd.DataFrame]:
    """Compute assignment counts and per-cell transcript/gene metrics."""
    cols = set(map(str, list(points_obj.columns)))
    has_background = "background" in cols
    point_cols = [assign_col, gene_col] + (["background"] if has_background else [])
    if hasattr(points_obj, "npartitions") and hasattr(points_obj, "partitions"):
        try:
            pts_small = points_obj[point_cols]
            n_total = int(pts_small.shape[0].compute())

            assigned_mask_dd = pts_small.map_partitions(
                assignment_mask_from_points,
                assign_col=assign_col,
                meta=("assigned", "bool"),
            )
            n_assigned = int(assigned_mask_dd.sum().compute())

            assigned_dd = pts_small[assigned_mask_dd]
            assigned_dd = assigned_dd.assign(
                cell_id_norm=assigned_dd[assign_col].astype(str)
            )

            trans_per_cell = assigned_dd.groupby("cell_id_norm").size().compute()
            trans_per_cell = trans_per_cell.rename("transcripts_per_cell")
            genes_per_cell = (
                assigned_dd.groupby("cell_id_norm")[gene_col].nunique().compute()
            )
            genes_per_cell = genes_per_cell.rename("genes_per_cell")
            cell_metrics = pd.concat([trans_per_cell, genes_per_cell], axis=1)
            cell_metrics = cell_metrics.reset_index(drop=False)
            return n_total, n_assigned, cell_metrics
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dask QC aggregation failed (%s); falling back to pandas.", exc
            )

    pts = to_pandas(points_obj)
    is_assigned = assignment_mask_from_points(pts[point_cols], assign_col=assign_col)
    n_total = int(len(pts))
    n_assigned = int(is_assigned.sum())

    assigned_pts = pts.loc[is_assigned, [assign_col, gene_col]].copy()
    assigned_pts["cell_id_norm"] = assigned_pts[assign_col].astype(str)
    trans_per_cell = (
        assigned_pts.groupby("cell_id_norm").size().rename("transcripts_per_cell")
    )
    genes_per_cell = assigned_pts.groupby("cell_id_norm")[gene_col].nunique()
    genes_per_cell = genes_per_cell.rename("genes_per_cell")
    cell_metrics = pd.concat([trans_per_cell, genes_per_cell], axis=1).reset_index(
        drop=False
    )
    return n_total, n_assigned, cell_metrics


def _choose_shape_key(sdata_obj: Any, preferred: str | None) -> str:
    """Resolve the shape key used for geometry QC."""
    if preferred is not None:
        if preferred not in sdata_obj.shapes:
            raise KeyError(
                f"Requested shape_key={preferred!r} not found. "
                f"Available shapes: {list(sdata_obj.shapes.keys())}"
            )
        return preferred
    if len(sdata_obj.shapes) == 0:
        raise RuntimeError("SpatialData object has no shapes for QC.")
    return str(list(sdata_obj.shapes.keys())[0])


def _choose_table_key(sdata_obj: Any, preferred: str | None) -> str | None:
    """Resolve the table key used for table-backed cell QC metrics."""
    if preferred is not None:
        if preferred not in sdata_obj.tables:
            raise KeyError(
                f"Requested table_key={preferred!r} not found. "
                f"Available tables: {list(sdata_obj.tables.keys())}"
            )
        return preferred
    return None


def _cell_ids_from_table(adata_obj: Any) -> pd.Index:
    """Return stable cell IDs from a SpatialData AnnData table."""
    attrs = dict(getattr(adata_obj, "uns", {}).get("spatialdata_attrs", {}))
    instance_key = attrs.get("instance_key")
    if isinstance(instance_key, str) and instance_key in adata_obj.obs.columns:
        values = adata_obj.obs[instance_key].astype(str).to_numpy()
        return pd.Index(values, dtype=str, name="cell_id_norm")
    return pd.Index(adata_obj.obs_names.astype(str), dtype=str, name="cell_id_norm")


def _compute_cell_metrics_from_table(adata_obj: Any) -> tuple[int, pd.DataFrame]:
    """Compute assigned transcript and gene counts from an AnnData cell table."""
    x_matrix = adata_obj.X
    transcripts_per_cell = np.asarray(x_matrix.sum(axis=1)).ravel()
    if hasattr(x_matrix, "getnnz"):
        genes_per_cell = np.asarray(x_matrix.getnnz(axis=1)).ravel()
    else:
        genes_per_cell = np.count_nonzero(np.asarray(x_matrix) > 0, axis=1)

    cell_metrics = pd.DataFrame(
        {
            "cell_id_norm": _cell_ids_from_table(adata_obj).astype(str),
            "transcripts_per_cell": transcripts_per_cell.astype(float),
            "genes_per_cell": genes_per_cell.astype(float),
        }
    )
    n_assigned = int(np.nansum(transcripts_per_cell))
    return n_assigned, cell_metrics


def compute_dataset_qc(
    latest_zarr_path: Path | str,
    dataset_name: str,
    *,
    table_key: str | None = None,
    shape_key: str | None = None,
) -> dict[str, Any]:
    """Compute geometry and assignment QC metrics for a dataset output zarr."""
    latest_zarr_path = Path(latest_zarr_path)
    logger.info("[%s] Loading latest output for QC: %s", dataset_name, latest_zarr_path)
    sdata = sd.read_zarr(latest_zarr_path)

    if len(sdata.shapes) == 0:
        raise RuntimeError(f"No shapes found in {latest_zarr_path}")
    if len(sdata.points) == 0:
        raise RuntimeError(f"No points found in {latest_zarr_path}")

    resolved_shape_key = _choose_shape_key(sdata, shape_key)
    resolved_table_key = _choose_table_key(sdata, table_key)
    shapes = sdata.shapes[resolved_shape_key]
    if "geometry" in shapes.columns:
        gdf = shapes[["geometry"]].copy()
    else:
        gdf = gpd.GeoDataFrame({"geometry": shapes.geometry})
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    geom_df = pd.DataFrame(index=gdf.index)
    geom_df["dataset"] = dataset_name
    geom_df["area"] = gdf.geometry.area.values
    geom_df["perimeter"] = gdf.geometry.length.values
    geom_df["convex_area"] = gdf.geometry.convex_hull.area.values
    geom_df["circularity"] = (
        4.0
        * np.pi
        * geom_df["area"]
        / np.clip(
            geom_df["perimeter"] ** 2,
            1e-12,
            None,
        )
    )
    geom_df["solidity"] = geom_df["area"] / np.clip(geom_df["convex_area"], 1e-12, None)
    ea = gdf.geometry.apply(_eccentricity_aspect)
    geom_df["eccentricity"] = [x[0] for x in ea]
    geom_df["aspect_ratio"] = [x[1] for x in ea]
    geom_df["log10_area"] = np.log10(np.clip(geom_df["area"].values, 1e-9, None))

    points_key = choose_primary_points_key(sdata)
    if points_key is None:
        raise RuntimeError("No transcript points found in SpatialData object.")
    points_obj = sdata.points[points_key]
    gene_col = first_existing_col(points_obj, ["feature_name", "gene", "target"])
    if gene_col is None:
        raise KeyError(
            f"No gene column found in points columns={list(points_obj.columns)}"
        )

    if resolved_table_key is not None:
        n_total = _point_count(points_obj)
        n_assigned, cell_metrics = _compute_cell_metrics_from_table(
            sdata.tables[resolved_table_key]
        )
    else:
        assign_col = first_existing_col(points_obj, ["assignment", "cell", "cell_id"])
        if assign_col is None:
            raise KeyError(
                "No assignment column found in points columns="
                f"{list(points_obj.columns)}"
            )
        n_total, n_assigned, cell_metrics = _compute_cell_metrics_from_points(
            points_obj,
            assign_col=assign_col,
            gene_col=gene_col,
        )
    cell_metrics["dataset"] = dataset_name
    pct_assigned = 100.0 * n_assigned / max(n_total, 1)

    summary = {
        "dataset": dataset_name,
        "latest_zarr_path": str(latest_zarr_path),
        "shape_key": resolved_shape_key,
        "table_key": resolved_table_key,
        "n_cells": int(len(gdf)),
        "n_transcripts_total": int(n_total),
        "n_transcripts_assigned": int(n_assigned),
        "pct_assigned": float(pct_assigned),
        "median_area": float(np.nanmedian(geom_df["area"])),
        "median_eccentricity": float(np.nanmedian(geom_df["eccentricity"])),
        "median_transcripts_per_cell": (
            float(np.nanmedian(cell_metrics["transcripts_per_cell"]))
            if len(cell_metrics)
            else np.nan
        ),
        "median_genes_per_cell": (
            float(np.nanmedian(cell_metrics["genes_per_cell"]))
            if len(cell_metrics)
            else np.nan
        ),
    }

    del sdata, gdf, shapes
    force_release(note=f"after QC {dataset_name}")

    return {
        "summary": summary,
        "geometry_metrics": geom_df,
        "cell_metrics": cell_metrics,
    }


def _point_count(points_obj: Any) -> int:
    """Return row count for pandas or dask-backed SpatialData points."""
    if hasattr(points_obj, "npartitions") and hasattr(points_obj, "partitions"):
        return int(points_obj.shape[0].compute())
    return int(len(points_obj))


def save_dataset_qc(
    qc_result: dict[str, Any],
    output_dir: Path | str,
    dataset_name: str,
) -> dict[str, Path]:
    """Persist QC outputs to CSV + pickle files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_name.lower()

    summary_path = output_dir / f"{stem}_qc_summary.csv"
    geom_path = output_dir / f"{stem}_geometry_metrics.csv"
    cell_path = output_dir / f"{stem}_cell_metrics.csv"
    pickle_path = output_dir / f"{stem}_qc.pkl"

    pd.DataFrame([qc_result["summary"]]).to_csv(summary_path, index=False)
    qc_result["geometry_metrics"].to_csv(geom_path, index=False)
    qc_result["cell_metrics"].to_csv(cell_path, index=False)

    with open(pickle_path, "wb") as f:
        pickle.dump(qc_result, f)

    return {
        "summary_csv": summary_path,
        "geometry_csv": geom_path,
        "cell_csv": cell_path,
        "pickle": pickle_path,
    }
