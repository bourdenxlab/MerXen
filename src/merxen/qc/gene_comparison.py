"""Cross-platform gene-level comparison utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spatialdata as sd

from merxen.control_features import control_transcript_mask
from merxen.io.points_aggregates import read_points_aggregates
from merxen.io.spatialdata_schema import choose_primary_points_key
from merxen.io.transcript_io import (
    assignment_mask_from_points,
    first_existing_col,
    to_pandas,
)

logger = logging.getLogger(__name__)

# 10x cell-feature-matrix feature-type columns: ``feature_types`` in the
# Xenium HDF5 matrix (as read by spatialdata-io), ``feature_type`` in the
# Atera zarr matrix.
FEATURE_TYPE_COLUMNS: tuple[str, ...] = ("feature_types", "feature_type")

# ``var`` column holding gene names when ``var_names`` are feature ids, as in
# the Atera cell-feature matrix (Ensembl ids in ``var_names``).
FEATURE_NAME_COLUMN = "feature_name"

# Points column the build's per-gene totals (``gene_counts_kept``) count.
AGGREGATE_GENE_COLUMN = "feature_name"


def _table_gene_labels(adata: Any, gene_label: str) -> pd.Index:
    for column in (gene_label, FEATURE_NAME_COLUMN):
        if column in adata.var.columns:
            return pd.Index(adata.var[column].astype(str))
    return pd.Index(adata.var_names.astype(str))


def gene_totals_from_table(adata: Any, gene_label: str = "gene") -> pd.Series:
    """Aggregate per-gene totals from an AnnData table.

    Args:
        adata: AnnData table.
        gene_label: ``var`` column holding gene names. When absent, the
            ``feature_name`` column is used, else ``var_names``.

    Returns:
        Summed counts indexed by unique gene name.
    """
    genes = _table_gene_labels(adata, gene_label)
    totals = np.asarray(adata.X.sum(axis=0)).ravel()
    return pd.Series(totals, index=genes).groupby(level=0).sum().astype(float)


def feature_types_from_table(adata: Any, gene_label: str = "gene") -> pd.Series | None:
    """Return the platform feature type of each gene in a table, if recorded.

    Args:
        adata: AnnData table.
        gene_label: ``var`` column holding gene names, resolved as in
            :func:`gene_totals_from_table`.

    Returns:
        Feature types indexed by unique gene name (the first non-missing type
        of a repeated name), or ``None`` when ``var`` has no column from
        :data:`FEATURE_TYPE_COLUMNS`.
    """
    column = next((c for c in FEATURE_TYPE_COLUMNS if c in adata.var.columns), None)
    if column is None:
        return None
    types = pd.Series(
        adata.var[column].astype("string").to_numpy(),
        index=_table_gene_labels(adata, gene_label),
        dtype="string",
    )
    return types.groupby(level=0).first()


def _gene_totals_from_aggregates(
    sdata_obj: Any, points_key: str, gene_col: str
) -> pd.Series | None:
    """Return the build's per-gene totals of a store, when they can be trusted.

    An ATERA build records the rows it wrote per feature
    (``gene_counts_kept``). They replace a pass over every transcript while
    the points index still matches the parquet files and the points are
    named by ``feature_name``. XENIUM and MERSCOPE stores record none.
    """
    zarr_path = getattr(sdata_obj, "path", None)
    if zarr_path is None or gene_col != AGGREGATE_GENE_COLUMN:
        return None
    aggregates = read_points_aggregates(Path(zarr_path), points_key)
    if aggregates is None:
        return None
    if not aggregates.fingerprint_matches():
        logger.info(
            "Points aggregates of %s do not match its parquet files; counting genes "
            "with dask.",
            points_key,
        )
        return None
    counts = pd.Series(
        aggregates.gene_counts_kept,
        index=pd.Index(aggregates.feature_names, dtype=str),
        dtype=np.int64,
    )
    # A groupby over the points only sees features that occur.
    counts = counts[counts > 0]
    return counts.groupby(level=0).sum().astype(float)


def gene_totals_from_points(
    sdata_obj: Any,
    assigned_only: bool = False,
) -> pd.Series:
    """Aggregate per-gene transcript counts from the primary points table.

    Dask-backed points are counted with dask; only pandas points are
    materialised. When ``assigned_only`` is false and the store's build
    recorded per-gene totals that still match its parquet files (ATERA), those
    totals are returned without reading the points.

    Args:
        sdata_obj: SpatialData object with a transcript points element.
        assigned_only: Count only transcripts assigned to a cell.

    Returns:
        Counts indexed by unique gene name, as floats.
    """
    if len(sdata_obj.points) == 0:
        raise RuntimeError("No points found in spatialdata object.")

    points_key = choose_primary_points_key(sdata_obj)
    if points_key is None:
        raise RuntimeError("No transcript points found in SpatialData object.")
    pts = sdata_obj.points[points_key]

    gene_col = first_existing_col(pts, ["gene", "feature_name", "target"])
    if gene_col is None:
        raise KeyError(
            f"Could not find gene column in points {points_key}: {list(pts.columns)}"
        )

    if not assigned_only:
        fast = _gene_totals_from_aggregates(sdata_obj, points_key, gene_col)
        if fast is not None:
            return fast

    assign_col: str | None = None
    background_col = first_existing_col(pts, ["background"])
    if assigned_only:
        assign_col = first_existing_col(pts, ["assignment", "cell", "cell_id"])
        if assign_col is None and background_col is None:
            raise KeyError(
                "Could not find assignment/background column in points "
                f"{points_key}: {list(pts.columns)}"
            )

    if hasattr(pts, "npartitions") and hasattr(pts, "partitions"):
        if assigned_only:
            cols = [gene_col]
            if assign_col is not None:
                cols.append(assign_col)
            if background_col is not None:
                cols.append(background_col)
            work = pts[cols]
            mask = work.map_partitions(
                assignment_mask_from_points,
                assign_col=assign_col,
                background_col=background_col or "background",
                meta=("assigned", "bool"),
            )
            counts = work.loc[mask].groupby(gene_col).size().compute()
        else:
            counts = pts[[gene_col]].groupby(gene_col).size().compute()
    else:
        pdf = to_pandas(pts)
        if assigned_only:
            cols = [gene_col]
            if assign_col is not None:
                cols.append(assign_col)
            if background_col is not None:
                cols.append(background_col)
            mask = assignment_mask_from_points(
                pdf[cols],
                assign_col=assign_col,
                background_col=background_col or "background",
            )
            counts = pdf.loc[mask].groupby(gene_col).size()
        else:
            counts = pdf.groupby(gene_col).size()

    counts.index = counts.index.astype(str)
    return counts.groupby(level=0).sum().astype(float)


def apply_dataset_filter(
    gene_counts: pd.Series,
    dataset_name: str,
    *,
    feature_types: pd.Series | None = None,
) -> pd.Series:
    """Remove platform control features from gene counts.

    Uses the shared rules of :mod:`merxen.control_features`: a gene whose
    feature type is ``Gene Expression`` is kept and one whose type is a 10x
    control category is dropped; any other gene (no type, or a type such as
    ``Protein Expression``) is a control when its name matches the platform's
    control-name pattern or contains a control token.

    Args:
        gene_counts: Per-gene counts indexed by unique gene name.
        dataset_name: Platform name (any case); selects the name patterns.
            An unknown name checks every platform's patterns.
        feature_types: Optional feature types indexed by gene name, as
            returned by :func:`feature_types_from_table`; a repeated name uses
            its first non-missing type. Genes missing from it fall back to the
            name rules.

    Returns:
        A copy of ``gene_counts`` without the platform's control features.
    """
    names = gene_counts.index.astype(str)
    types = None
    if feature_types is not None:
        if not feature_types.index.is_unique:
            feature_types = feature_types.groupby(level=0).first()
        types = feature_types.reindex(names)
    controls = control_transcript_mask(
        names.to_numpy(dtype=object), platform=dataset_name, feature_type=types
    )
    return gene_counts.loc[~controls].copy()


def normalize_counts(gene_counts: pd.Series) -> tuple[pd.Series, float]:
    """Normalize counts by total transcripts and return (normalized, total)."""
    total = float(gene_counts.sum())
    if total <= 0:
        raise ValueError("Cannot normalize: total count is <= 0.")
    return gene_counts / total, total


def compare_df(x_counts: pd.Series, m_counts: pd.Series) -> pd.DataFrame:
    """Construct an aligned comparison DataFrame across shared genes."""
    common = sorted(set(x_counts.index.astype(str)) & set(m_counts.index.astype(str)))
    return pd.DataFrame(
        {
            "gene": common,
            "xenium": x_counts.reindex(common).values,
            "merscope": m_counts.reindex(common).values,
        }
    )


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a 1D linear model and return slope, intercept, and R²."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    denom = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if denom <= 0 else (1.0 - np.sum((y - yhat) ** 2) / denom)
    return float(slope), float(intercept), float(r2)


def compute_gene_comparison(
    xenium_sdata: Any,
    merscope_sdata: Any,
    *,
    xenium_table_key: str | None = None,
    merscope_table_key: str | None = None,
) -> dict[str, Any]:
    """Compute cross-platform gene totals, normalized counts, and fit metrics."""
    logger.info("[Gene Compare] Computing per-gene totals from points and tables")

    resolved_xenium_table_key = _choose_table_key(xenium_sdata, xenium_table_key)
    resolved_merscope_table_key = _choose_table_key(merscope_sdata, merscope_table_key)
    xenium_table = xenium_sdata.tables[resolved_xenium_table_key]
    merscope_table = merscope_sdata.tables[resolved_merscope_table_key]

    x_total_all = gene_totals_from_points(xenium_sdata, assigned_only=False)
    m_total_all = gene_totals_from_points(merscope_sdata, assigned_only=False)
    x_assigned_all = gene_totals_from_table(xenium_table)
    m_assigned_all = gene_totals_from_table(merscope_table)

    x_types = feature_types_from_table(xenium_table)
    m_types = feature_types_from_table(merscope_table)
    x_total = apply_dataset_filter(x_total_all, "XENIUM", feature_types=x_types)
    m_total = apply_dataset_filter(m_total_all, "MERSCOPE", feature_types=m_types)
    x_assigned = apply_dataset_filter(x_assigned_all, "XENIUM", feature_types=x_types)
    m_assigned = apply_dataset_filter(m_assigned_all, "MERSCOPE", feature_types=m_types)

    x_total_norm, x_total_sum = normalize_counts(x_total)
    m_total_norm, m_total_sum = normalize_counts(m_total)
    x_assigned_norm, x_assigned_sum = normalize_counts(x_assigned)
    m_assigned_norm, m_assigned_sum = normalize_counts(m_assigned)

    total_df = compare_df(x_total, m_total)
    assigned_df = compare_df(x_assigned, m_assigned)
    total_norm_df = compare_df(x_total_norm, m_total_norm)
    assigned_norm_df = compare_df(x_assigned_norm, m_assigned_norm)

    lx = np.log10(np.clip(total_norm_df["xenium"].to_numpy(float), 1e-12, None))
    ly = np.log10(np.clip(total_norm_df["merscope"].to_numpy(float), 1e-12, None))
    total_fit = fit_linear(lx, ly)

    lax = np.log10(np.clip(assigned_norm_df["xenium"].to_numpy(float), 1e-12, None))
    lay = np.log10(np.clip(assigned_norm_df["merscope"].to_numpy(float), 1e-12, None))
    assigned_fit = fit_linear(lax, lay)

    return {
        "total_counts_df": total_df,
        "assigned_counts_df": assigned_df,
        "total_normalized_df": total_norm_df,
        "assigned_normalized_df": assigned_norm_df,
        "totals": {
            "xenium_total_sum": x_total_sum,
            "merscope_total_sum": m_total_sum,
            "xenium_assigned_sum": x_assigned_sum,
            "merscope_assigned_sum": m_assigned_sum,
        },
        "table_keys": {
            "xenium": resolved_xenium_table_key,
            "merscope": resolved_merscope_table_key,
        },
        "fits": {
            "total_log10": {
                "slope": total_fit[0],
                "intercept": total_fit[1],
                "r2": total_fit[2],
            },
            "assigned_log10": {
                "slope": assigned_fit[0],
                "intercept": assigned_fit[1],
                "r2": assigned_fit[2],
            },
        },
    }


def gene_summary_df(counts: pd.Series, normalized: pd.Series) -> pd.DataFrame:
    """Construct a per-gene single-dataset summary DataFrame."""
    genes = sorted(set(counts.index.astype(str)) | set(normalized.index.astype(str)))
    return pd.DataFrame(
        {
            "gene": genes,
            "count": counts.reindex(genes).fillna(0.0).to_numpy(float),
            "normalized": normalized.reindex(genes).fillna(0.0).to_numpy(float),
        }
    )


def compute_gene_summary(
    sdata_obj: Any,
    dataset_name: str,
    *,
    table_key: str | None = None,
) -> dict[str, Any]:
    """Compute single-dataset total and assigned per-gene count summaries."""
    logger.info("[%s Gene Summary] Computing per-gene totals", dataset_name)
    resolved_table_key = _choose_table_key(sdata_obj, table_key)
    table = sdata_obj.tables[resolved_table_key]

    total_all = gene_totals_from_points(sdata_obj, assigned_only=False)
    assigned_all = gene_totals_from_table(table)

    feature_types = feature_types_from_table(table)
    total = apply_dataset_filter(total_all, dataset_name, feature_types=feature_types)
    assigned = apply_dataset_filter(
        assigned_all, dataset_name, feature_types=feature_types
    )

    total_norm, total_sum = normalize_counts(total)
    assigned_norm, assigned_sum = normalize_counts(assigned)

    return {
        "total_counts_df": gene_summary_df(total, total_norm),
        "assigned_counts_df": gene_summary_df(assigned, assigned_norm),
        "totals": {
            "total_sum": total_sum,
            "assigned_sum": assigned_sum,
        },
        "table_key": resolved_table_key,
    }


def compute_gene_comparison_from_paths(
    xenium_zarr_path: Path | str,
    merscope_zarr_path: Path | str,
    *,
    xenium_table_key: str | None = None,
    merscope_table_key: str | None = None,
) -> dict[str, Any]:
    """Load SpatialData zarrs and run cross-platform gene comparison."""
    xenium_sdata = sd.read_zarr(Path(xenium_zarr_path))
    merscope_sdata = sd.read_zarr(Path(merscope_zarr_path))
    return compute_gene_comparison(
        xenium_sdata=xenium_sdata,
        merscope_sdata=merscope_sdata,
        xenium_table_key=xenium_table_key,
        merscope_table_key=merscope_table_key,
    )


def compute_gene_summary_from_path(
    zarr_path: Path | str,
    dataset_name: str,
    *,
    table_key: str | None = None,
) -> dict[str, Any]:
    """Load one SpatialData zarr and compute single-dataset gene summaries."""
    sdata_obj = sd.read_zarr(Path(zarr_path))
    return compute_gene_summary(
        sdata_obj=sdata_obj,
        dataset_name=dataset_name,
        table_key=table_key,
    )


def _choose_table_key(sdata_obj: Any, preferred: str | None) -> str:
    """Resolve an AnnData table key for assigned-count summaries."""
    if preferred is not None:
        if preferred not in sdata_obj.tables:
            raise KeyError(
                f"Requested table_key={preferred!r} not found. "
                f"Available tables: {list(sdata_obj.tables.keys())}"
            )
        return preferred
    if "table" in sdata_obj.tables:
        return "table"
    if len(sdata_obj.tables) == 0:
        raise RuntimeError("SpatialData object has no AnnData tables.")
    return str(list(sdata_obj.tables.keys())[0])
