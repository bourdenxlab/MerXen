"""Feature extraction for Spateo alignment."""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from merxen.io.transcript_io import first_existing_col

logger = logging.getLogger(__name__)


def build_alignment_adata(
    sdata_obj: Any,
    *,
    platform: str,
    preferred_shape_key: str | None = None,
    preferred_table_key: str = "table",
    include_image_features: bool = True,
) -> ad.AnnData:
    """Build a cell-centroid AnnData object for section alignment.

    The first implementation uses the enriched cell-by-gene table plus cell
    centroids. Image-derived features are intentionally best-effort: if the
    SpatialData object does not expose a direct, unambiguous image-to-xy mapping,
    the gene features are used alone and this is recorded in ``uns``.
    """
    del include_image_features
    shape_key = _choose_shape_key(sdata_obj, preferred_shape_key)
    table_key = _choose_table_key(sdata_obj, preferred_table_key)

    centroids = _shape_centroids(sdata_obj.shapes[shape_key])
    table = sdata_obj.tables[table_key]
    adata = _align_table_to_centroids(table, centroids)
    adata.uns["merxen_alignment"] = {
        "platform": platform,
        "shape_key": shape_key,
        "table_key": table_key,
        "image_features": "skipped_no_unambiguous_image_xy_mapping",
    }
    return adata


def prepare_spateo_features(
    adata: ad.AnnData,
    *,
    normalize_total: float = 10_000.0,
    log1p: bool = True,
    use_hvg: bool = True,
    n_top_genes: int = 2_000,
) -> ad.AnnData:
    """Normalize/log-transform and optionally select variable genes."""
    out = adata.copy()
    out.X = _as_float_matrix(out.X)
    if normalize_total and normalize_total > 0:
        out.X = _normalize_total(out.X, target_sum=float(normalize_total))
    if log1p:
        out.X = _log1p_matrix(out.X)
    if use_hvg and out.n_vars > int(n_top_genes):
        keep = _top_variable_columns(out.X, int(n_top_genes))
        out = out[:, keep].copy()
    return out


def shared_gene_subset(
    fixed: ad.AnnData,
    moving: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subset two AnnData objects to shared feature names in the same order."""
    fixed_names = pd.Index(fixed.var_names.astype(str))
    moving_names = pd.Index(moving.var_names.astype(str))
    shared = fixed_names.intersection(moving_names)
    if len(shared) == 0:
        raise ValueError("No shared alignment features found between datasets")
    return fixed[:, shared].copy(), moving[:, shared].copy()


def _choose_shape_key(sdata_obj: Any, preferred: str | None) -> str:
    if preferred is not None and preferred in sdata_obj.shapes:
        return preferred
    if len(sdata_obj.shapes) == 0:
        raise RuntimeError("SpatialData object has no shapes for alignment")
    preferred_names = [
        "MOSAIK_proseg_aligned_nonrigid",
        "cell_boundaries_aligned_nonrigid",
        "merscope_cell_boundaries_aligned_nonrigid",
        "xenium_cell_boundaries_aligned_nonrigid",
        "MOSAIK_proseg",
        "cell_boundaries",
        "merscope_cell_boundaries",
        "xenium_cell_boundaries",
    ]
    for key in preferred_names:
        if key in sdata_obj.shapes:
            return key
    for key in sdata_obj.shapes:
        if str(key).endswith("_aligned_nonrigid"):
            return str(key)
    return str(list(sdata_obj.shapes.keys())[0])


def _choose_table_key(sdata_obj: Any, preferred: str) -> str:
    if preferred in sdata_obj.tables:
        return preferred
    if len(sdata_obj.tables) == 0:
        raise RuntimeError("SpatialData object has no tables for alignment")
    return str(list(sdata_obj.tables.keys())[0])


def _shape_centroids(shapes: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = shapes.copy()
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame({"geometry": gdf.geometry}, index=gdf.index)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    id_col = first_existing_col(
        gdf,
        ["cell_id", "cell", "cells", "cell_ID", "region", "label_id"],
    )
    ids = gdf.index.astype(str) if id_col is None else gdf[id_col].astype(str)
    cent_x, cent_y = _robust_centroid_xy(gdf)
    out = pd.DataFrame(
        {
            "cell_id": ids.astype(str).to_numpy(),
            "x": cent_x,
            "y": cent_y,
        },
        index=pd.Index(ids.astype(str), name="cell_id"),
    )
    out = out[np.isfinite(out["x"]) & np.isfinite(out["y"])]
    return out[~out.index.duplicated(keep="first")]


def _robust_centroid_xy(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid-like points while guarding against invalid-geometry blowups."""
    cent = gdf.geometry.centroid
    bounds = gdf.geometry.bounds
    x = cent.x.to_numpy(float)
    y = cent.y.to_numpy(float)
    minx = bounds["minx"].to_numpy(float)
    miny = bounds["miny"].to_numpy(float)
    maxx = bounds["maxx"].to_numpy(float)
    maxy = bounds["maxy"].to_numpy(float)
    bad = (
        ~np.isfinite(x)
        | ~np.isfinite(y)
        | (x < minx)
        | (x > maxx)
        | (y < miny)
        | (y > maxy)
    )
    if bad.any():
        reps = gdf.geometry.representative_point()
        rx = reps.x.to_numpy(float)
        ry = reps.y.to_numpy(float)
        use_rep = (
            bad
            & np.isfinite(rx)
            & np.isfinite(ry)
            & (rx >= minx)
            & (rx <= maxx)
            & (ry >= miny)
            & (ry <= maxy)
        )
        x[use_rep] = rx[use_rep]
        y[use_rep] = ry[use_rep]
        fallback = bad & ~use_rep
        x[fallback] = (minx[fallback] + maxx[fallback]) / 2.0
        y[fallback] = (miny[fallback] + maxy[fallback]) / 2.0
    return x, y


def _align_table_to_centroids(table: ad.AnnData, centroids: pd.DataFrame) -> ad.AnnData:
    table_ids = _table_cell_ids(table)
    common = pd.Index(table_ids).intersection(centroids.index)

    if len(common) >= 3:
        table_pos = pd.Series(np.arange(len(table_ids)), index=table_ids).loc[common]
        adata = table[table_pos.to_numpy(), :].copy()
        coords = centroids.loc[common, ["x", "y"]].to_numpy(float)
        adata.obs_names = common.astype(str)
    elif table.n_obs == len(centroids) and table.n_obs >= 3:
        adata = table.copy()
        coords = centroids[["x", "y"]].to_numpy(float)
        adata.obs_names = centroids.index.astype(str)
    else:
        raise ValueError(
            "Could not match cell table rows to shape centroids. "
            f"table_n={table.n_obs}, centroid_n={len(centroids)}, "
            f"common_n={len(common)}"
        )

    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obsm["spatial"] = coords
    genes = _table_gene_names(adata)
    adata.var_names = genes
    adata.var["gene"] = genes
    return adata


def _table_cell_ids(table: ad.AnnData) -> pd.Index:
    for col in ["cell_id", "cell", "cells", "cell_ID"]:
        if col in table.obs.columns:
            return pd.Index(table.obs[col].astype(str))
    return pd.Index(table.obs_names.astype(str))


def _table_gene_names(table: ad.AnnData) -> pd.Index:
    if "gene" in table.var.columns:
        return pd.Index(table.var["gene"].astype(str))
    return pd.Index(table.var_names.astype(str))


def _as_float_matrix(x: Any) -> Any:
    if sparse.issparse(x):
        return x.astype(np.float64).tocsr(copy=True)
    return np.asarray(x, dtype=np.float64).copy()


def _normalize_total(x: Any, *, target_sum: float) -> Any:
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).ravel()
        scale = np.divide(
            target_sum,
            row_sum,
            out=np.zeros_like(row_sum, dtype=np.float64),
            where=row_sum > 0,
        )
        return sparse.diags(scale).dot(x).tocsr()
    row_sum = x.sum(axis=1)
    scale = np.divide(
        target_sum,
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float64),
        where=row_sum > 0,
    )
    return x * scale[:, None]


def _log1p_matrix(x: Any) -> Any:
    if sparse.issparse(x):
        out = x.copy()
        out.data = np.log1p(out.data)
        return out
    return np.log1p(x)


def _top_variable_columns(x: Any, n_top: int) -> np.ndarray:
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        var = np.nanvar(x, axis=0)
    return np.argsort(var)[-int(n_top) :]
