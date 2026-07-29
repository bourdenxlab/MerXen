"""Hybrid-segmentation diagnostics and comparison outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merxen.io.spatialdata_schema import INSTANCE_ID_COLUMN, canonical_instance_series
from merxen.plotting import save_figure

HYBRID_SHAPE_KEY = "MOSAIK_proseg_hybrid"
HYBRID_TABLE_KEY = "table_MOSAIK_proseg_hybrid"
CELLPOSE_TABLE_KEY = "table_MOSAIK_cellpose"
PROSEG_TABLE_KEY = "table_MOSAIK_proseg"
HYBRID_SOURCE_COLUMN = "hybrid_assignment_source"


def compute_hybrid_qc(
    sdata_obj: Any,
    *,
    dataset_name: str,
    points_key: str,
) -> dict[str, Any]:
    """Build per-cell, per-gene, and assignment-provenance hybrid QC tables."""
    if HYBRID_SHAPE_KEY not in sdata_obj.shapes:
        raise KeyError(f"Missing hybrid shape {HYBRID_SHAPE_KEY!r}")
    if HYBRID_TABLE_KEY not in sdata_obj.tables:
        raise KeyError(f"Missing hybrid table {HYBRID_TABLE_KEY!r}")

    hybrid_shapes = sdata_obj.shapes[HYBRID_SHAPE_KEY].copy()
    diagnostics = pd.DataFrame(hybrid_shapes.drop(columns="geometry")).copy()
    diagnostics.index = _shape_instance_index(hybrid_shapes)
    diagnostics.index.name = "cell_id"

    hybrid_counts = _cell_count_metrics(sdata_obj.tables[HYBRID_TABLE_KEY])
    diagnostics = diagnostics.join(
        hybrid_counts.add_prefix("hybrid_"),
        how="left",
    )
    for label, table_key in (
        ("cellpose", CELLPOSE_TABLE_KEY),
        ("proseg", PROSEG_TABLE_KEY),
    ):
        if table_key not in sdata_obj.tables:
            continue
        comparison = _cell_count_metrics(sdata_obj.tables[table_key])
        diagnostics = diagnostics.join(
            comparison.add_prefix(f"{label}_"),
            how="left",
        )
        for metric in ("transcripts", "genes"):
            diagnostics[f"hybrid_minus_{label}_{metric}"] = (
                diagnostics[f"hybrid_{metric}"] - diagnostics[f"{label}_{metric}"]
            )

    if {"hybrid_area_um2", "cellpose_area_um2"}.issubset(diagnostics.columns):
        diagnostics["hybrid_area_growth_um2"] = pd.to_numeric(
            diagnostics["hybrid_area_um2"], errors="coerce"
        ) - pd.to_numeric(diagnostics["cellpose_area_um2"], errors="coerce")
        diagnostics["hybrid_area_growth_fraction"] = diagnostics[
            "hybrid_area_growth_um2"
        ] / pd.to_numeric(
            diagnostics["cellpose_area_um2"],
            errors="coerce",
        ).clip(lower=1e-12)
        diagnostics["large_expansion_outlier"] = _upper_tail_flag(
            diagnostics["hybrid_area_growth_fraction"]
        )

    rejected_columns = [
        column
        for column in ("cap_rejected_external", "unsupported_external")
        if column in diagnostics.columns
    ]
    if rejected_columns:
        diagnostics["rejected_external_transcripts"] = (
            diagnostics[rejected_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
        )
        diagnostics["high_rejection_outlier"] = _upper_tail_flag(
            diagnostics["rejected_external_transcripts"]
        )

    source_summary = _assignment_source_summary(
        sdata_obj.points[points_key],
    )
    gene_changes = _gene_count_changes(sdata_obj.tables)
    map_gdf = hybrid_shapes[["geometry"]].copy()
    map_gdf.index = diagnostics.index
    for column in (
        "hybrid_area_growth_fraction",
        "rejected_external_transcripts",
        "large_expansion_outlier",
        "high_rejection_outlier",
    ):
        if column in diagnostics.columns:
            map_gdf[column] = diagnostics[column].to_numpy()

    fallback = (
        diagnostics["fallback_reason"].fillna("").astype(str).str.strip()
        if "fallback_reason" in diagnostics.columns
        else pd.Series("", index=diagnostics.index)
    )
    fallback_counts = fallback[fallback != ""].value_counts()
    fallback_reasons = pd.DataFrame(
        {
            "fallback_reason": fallback_counts.index.astype(str),
            "count": fallback_counts.to_numpy(dtype=np.int64),
        }
    )
    fallback_reasons["percent_of_hybrid_cells"] = (
        100.0 * fallback_reasons["count"] / max(len(diagnostics), 1)
    )
    source_counts = dict(
        zip(
            source_summary["assignment_source"],
            source_summary["count"],
            strict=True,
        )
    )
    n_sources = int(source_summary["count"].sum())
    summary = {
        "dataset": str(dataset_name),
        "n_hybrid_cells": int(len(diagnostics)),
        "n_fallback_cellpose": int((fallback != "").sum()),
        "pct_fallback_cellpose": float(
            100.0 * (fallback != "").sum() / max(len(diagnostics), 1)
        ),
        "n_assignment_source_transcripts": n_sources,
        "n_single_mask": int(source_counts.get("single_mask", 0)),
        "n_proseg_overlap": int(source_counts.get("proseg_overlap", 0)),
        "n_ambiguous_overlap": int(source_counts.get("ambiguous_overlap", 0)),
        "n_outside": int(source_counts.get("outside", 0)),
        "pct_ambiguous_overlap": float(
            100.0 * source_counts.get("ambiguous_overlap", 0) / max(n_sources, 1)
        ),
        "n_large_expansion_outliers": int(
            diagnostics.get(
                "large_expansion_outlier",
                pd.Series(False, index=diagnostics.index),
            ).sum()
        ),
        "n_high_rejection_outliers": int(
            diagnostics.get(
                "high_rejection_outlier",
                pd.Series(False, index=diagnostics.index),
            ).sum()
        ),
    }
    return {
        "summary": summary,
        "cell_diagnostics": diagnostics.reset_index(),
        "assignment_sources": source_summary,
        "fallback_reasons": fallback_reasons,
        "gene_count_changes": gene_changes,
        "map_gdf": map_gdf,
    }


def save_hybrid_qc(
    result: dict[str, Any],
    output_dir: Path | str,
    dataset_name: str,
) -> dict[str, Path]:
    """Persist hybrid QC tables and spatial diagnostic maps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(dataset_name).lower()
    paths = {
        "hybrid_cell_diagnostics": (output_dir / f"{stem}_hybrid_cell_diagnostics.csv"),
        "hybrid_assignment_sources": (
            output_dir / f"{stem}_hybrid_assignment_sources.csv"
        ),
        "hybrid_gene_count_changes": (
            output_dir / f"{stem}_hybrid_gene_count_changes.csv"
        ),
        "hybrid_fallback_reasons": (output_dir / f"{stem}_hybrid_fallback_reasons.csv"),
    }
    result["cell_diagnostics"].to_csv(
        paths["hybrid_cell_diagnostics"],
        index=False,
    )
    result["assignment_sources"].to_csv(
        paths["hybrid_assignment_sources"],
        index=False,
    )
    result["gene_count_changes"].to_csv(
        paths["hybrid_gene_count_changes"],
        index=False,
    )
    result["fallback_reasons"].to_csv(
        paths["hybrid_fallback_reasons"],
        index=False,
    )

    map_gdf = result["map_gdf"]
    if "hybrid_area_growth_fraction" in map_gdf.columns:
        path = output_dir / f"{stem}_hybrid_area_growth_map.png"
        _plot_hybrid_metric_map(
            map_gdf,
            "hybrid_area_growth_fraction",
            path,
            title=f"{dataset_name} hybrid area growth",
            colorbar_label="Area growth / Cellpose area",
            cmap="viridis",
            outlier_column="large_expansion_outlier",
        )
        paths["hybrid_area_growth_map"] = path
    if "rejected_external_transcripts" in map_gdf.columns:
        path = output_dir / f"{stem}_hybrid_rejected_transcripts_map.png"
        _plot_hybrid_metric_map(
            map_gdf,
            "rejected_external_transcripts",
            path,
            title=f"{dataset_name} rejected external transcripts",
            colorbar_label="Rejected transcripts",
            cmap="magma",
            outlier_column="high_rejection_outlier",
        )
        paths["hybrid_rejected_transcripts_map"] = path
    return paths


def _shape_instance_index(shapes: gpd.GeoDataFrame) -> pd.Index:
    values = (
        shapes[INSTANCE_ID_COLUMN]
        if INSTANCE_ID_COLUMN in shapes.columns
        else pd.Series(shapes.index, index=shapes.index)
    )
    canonical = canonical_instance_series(
        values,
        field_name=f"{HYBRID_SHAPE_KEY}.{INSTANCE_ID_COLUMN}",
    )
    return pd.Index(canonical.astype(str), dtype=str)


def _table_instance_index(table: ad.AnnData, *, table_name: str) -> pd.Index:
    attrs = dict(table.uns.get("spatialdata_attrs", {}))
    instance_key = str(attrs.get("instance_key", INSTANCE_ID_COLUMN))
    if instance_key not in table.obs.columns:
        raise KeyError(f"{table_name!r} lacks instance key {instance_key!r}")
    values = canonical_instance_series(
        table.obs[instance_key],
        field_name=f"{table_name}.{instance_key}",
    )
    return pd.Index(values.astype(str), dtype=str)


def _cell_count_metrics(table: ad.AnnData) -> pd.DataFrame:
    matrix = table.X
    transcripts = np.asarray(matrix.sum(axis=1)).ravel()
    if hasattr(matrix, "getnnz"):
        genes = np.asarray(matrix.getnnz(axis=1)).ravel()
    else:
        genes = np.count_nonzero(np.asarray(matrix) > 0, axis=1)
    return pd.DataFrame(
        {
            "transcripts": transcripts.astype(float),
            "genes": genes.astype(float),
        },
        index=_table_instance_index(table, table_name="cell-count table"),
    )


def _assignment_source_summary(points_obj: Any) -> pd.DataFrame:
    if HYBRID_SOURCE_COLUMN not in points_obj.columns:
        raise KeyError(f"Hybrid transcript points lack {HYBRID_SOURCE_COLUMN!r}")
    values = points_obj[HYBRID_SOURCE_COLUMN]
    counts = values.value_counts(dropna=False)
    if hasattr(counts, "compute"):
        counts = counts.compute()
    counts = pd.Series(counts)
    labels = ["missing" if pd.isna(value) else str(value) for value in counts.index]
    result = pd.DataFrame(
        {
            "assignment_source": labels,
            "count": counts.to_numpy(dtype=np.int64),
        }
    )
    total = int(result["count"].sum())
    result["percent"] = 100.0 * result["count"] / max(total, 1)
    preferred = {
        value: index
        for index, value in enumerate(
            ("single_mask", "proseg_overlap", "ambiguous_overlap", "outside")
        )
    }
    result["_order"] = result["assignment_source"].map(
        lambda value: preferred.get(value, len(preferred))
    )
    return (
        result.sort_values(
            ["_order", "assignment_source"],
            kind="stable",
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _gene_count_changes(tables: Any) -> pd.DataFrame:
    series_by_name: dict[str, pd.Series] = {}
    for label, table_key in (
        ("hybrid", HYBRID_TABLE_KEY),
        ("cellpose", CELLPOSE_TABLE_KEY),
        ("proseg", PROSEG_TABLE_KEY),
    ):
        if table_key not in tables:
            continue
        table = tables[table_key]
        genes = (
            table.var["gene"].astype(str)
            if "gene" in table.var.columns
            else pd.Series(table.var_names.astype(str), index=table.var_names)
        )
        totals = np.asarray(table.X.sum(axis=0)).ravel().astype(float)
        series_by_name[label] = (
            pd.Series(totals, index=np.asarray(genes, dtype=str)).groupby(level=0).sum()
        )
    frame = pd.DataFrame(series_by_name).fillna(0.0)
    frame.index.name = "gene"
    for label in ("cellpose", "proseg"):
        if label in frame.columns:
            frame[f"hybrid_minus_{label}"] = frame["hybrid"] - frame[label]
    return frame.reset_index()


def _upper_tail_flag(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(False, index=values.index)
    threshold = float(finite.quantile(0.99))
    return numeric.ge(threshold) & numeric.gt(0)


def _plot_hybrid_metric_map(
    map_gdf: gpd.GeoDataFrame,
    column: str,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    outlier_column: str | None = None,
) -> None:
    values = pd.to_numeric(map_gdf[column], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if bool(values.notna().any()):
        map_gdf.assign(_plot_value=values).plot(
            column="_plot_value",
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={"label": colorbar_label, "shrink": 0.75},
            linewidth=0,
        )
    else:
        map_gdf.boundary.plot(ax=ax, color="#6b7280", linewidth=0.2)
    if outlier_column is not None and outlier_column in map_gdf.columns:
        outliers = map_gdf[map_gdf[outlier_column].fillna(False).astype(bool)]
        if not outliers.empty:
            outliers.boundary.plot(
                ax=ax,
                color="#00ffff",
                linewidth=0.8,
            )
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)
