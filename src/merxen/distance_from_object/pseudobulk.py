"""Pseudobulk aggregation and paired near-vs-far differential expression."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def build_pair_pseudobulk(
    table: ad.AnnData,
    assignments: pd.DataFrame,
    *,
    pair_id: str,
    included_tissue_annotations: list[str],
    tissue_annotation_column: str = "cortical_depth_annotation",
    min_cells: int = 10,
) -> ad.AnnData:
    """Aggregate raw cell counts into near/far samples for one tissue block.

    Args:
        table: Source cell-by-gene AnnData with raw counts in ``layers['counts']``
            or ``X``.
        assignments: Distance metadata indexed by normalized cell ID.
        pair_id: Tissue-block identifier used as the paired design factor.
        included_tissue_annotations: Tissue labels eligible for aggregation.
        tissue_annotation_column: Assignment column containing tissue labels.
        min_cells: Minimum eligible cells required for a pseudobulk group.

    Returns:
        AnnData with zero, one, or two pseudobulk observations and raw summed
        integer counts. Only ``near`` and ``far`` groups are aggregated.

    Raises:
        KeyError: If the required tissue annotation is absent.
        ValueError: If counts are negative, non-finite, or non-integer.
    """
    if tissue_annotation_column not in assignments.columns:
        raise KeyError(
            f"Missing tissue annotation column {tissue_annotation_column!r}."
        )
    cell_ids = _table_cell_ids(table)
    aligned = assignments.reindex(cell_ids)
    matrix = _raw_counts_matrix(table)
    included = {str(value) for value in included_tissue_annotations}
    tissue = aligned[tissue_annotation_column].astype(str).isin(included).to_numpy()
    proximity = aligned["object_proximity"].astype(str).to_numpy()

    rows: list[sparse.csr_matrix] = []
    metadata: list[dict[str, Any]] = []
    for label in ("near", "far"):
        keep = tissue & (proximity == label)
        cell_count = int(keep.sum())
        if cell_count < int(min_cells):
            continue
        summed = sparse.csr_matrix(matrix[keep].sum(axis=0), dtype=np.int64)
        rows.append(summed)
        metadata.append(
            {
                "pair_id": str(pair_id),
                "proximity": label,
                "cell_count": cell_count,
            }
        )

    counts = (
        sparse.vstack(rows, format="csr", dtype=np.int64)
        if rows
        else sparse.csr_matrix((0, table.n_vars), dtype=np.int64)
    )
    obs_names = [f"{pair_id}__{row['proximity']}" for row in metadata]
    obs = pd.DataFrame(metadata, index=pd.Index(obs_names, name="pseudobulk_id"))
    if obs.empty:
        obs = pd.DataFrame(
            {
                "pair_id": pd.Series(dtype=str),
                "proximity": pd.Series(dtype=str),
                "cell_count": pd.Series(dtype=int),
            },
            index=pd.Index([], name="pseudobulk_id"),
        )
    var = table.var.copy()
    var.index = pd.Index(table.var_names.astype(str), name=table.var_names.name)
    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.layers["counts"] = counts.copy()
    result.uns["merxen_distance_from_object"] = {
        "pair_id": str(pair_id),
        "included_tissue_annotations": sorted(included),
        "tissue_annotation_column": tissue_annotation_column,
        "min_cells": int(min_cells),
    }
    return result


def combine_pair_pseudobulks(paths: Sequence[Path | str]) -> ad.AnnData:
    """Combine non-empty pair pseudobulks using their shared gene panel."""
    tables: list[ad.AnnData] = []
    for path in paths:
        table = ad.read_h5ad(path)
        if table.n_obs > 0:
            tables.append(table)
    if not tables:
        raise ValueError("No non-empty pair pseudobulk tables were found.")

    shared = set(tables[0].var_names.astype(str))
    for table in tables[1:]:
        shared &= set(table.var_names.astype(str))
    ordered_genes = [
        str(gene) for gene in tables[0].var_names.astype(str) if str(gene) in shared
    ]
    if not ordered_genes:
        raise ValueError("Pair pseudobulk tables have no shared genes.")
    subset = [table[:, ordered_genes].copy() for table in tables]
    combined = ad.concat(subset, axis=0, join="inner", merge="same")
    if combined.obs_names.duplicated().any():
        duplicates = combined.obs_names[combined.obs_names.duplicated()].tolist()
        raise ValueError(f"Duplicate pseudobulk IDs: {duplicates[:5]}")
    counts = _raw_counts_matrix(combined)
    combined.X = counts
    combined.layers["counts"] = counts.copy()
    return combined


def retain_complete_pairs(
    pseudobulk: ad.AnnData,
) -> tuple[ad.AnnData, list[str]]:
    """Retain pair IDs containing both near and far pseudobulk observations."""
    required = {"near", "far"}
    complete = [
        str(pair_id)
        for pair_id, group in pseudobulk.obs.groupby("pair_id", observed=True)
        if set(group["proximity"].astype(str)) >= required
    ]
    keep = pseudobulk.obs["pair_id"].astype(str).isin(complete).to_numpy()
    return pseudobulk[keep].copy(), complete


def run_paired_differential_expression(
    pseudobulk: ad.AnnData,
    *,
    n_cpus: int | None = None,
) -> pd.DataFrame:
    """Run PyDESeq2 with ``pair_id`` blocking and a near-vs-far contrast."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts = _raw_counts_matrix(pseudobulk)
    dense = counts.toarray()
    expressed = np.asarray(dense.sum(axis=0)).reshape(-1) > 0
    if not expressed.any():
        raise ValueError("All pseudobulk genes have zero counts.")
    counts_frame = pd.DataFrame(
        dense[:, expressed],
        index=pseudobulk.obs_names.astype(str),
        columns=pseudobulk.var_names.astype(str)[expressed],
        dtype=np.int64,
    )
    metadata = pseudobulk.obs[["pair_id", "proximity"]].copy()
    metadata.index = pseudobulk.obs_names.astype(str)
    metadata["pair_id"] = metadata["pair_id"].astype(str)
    metadata["proximity"] = pd.Categorical(
        metadata["proximity"].astype(str),
        categories=["far", "near"],
    )
    dataset = DeseqDataSet(
        counts=counts_frame,
        metadata=metadata,
        design="~ pair_id + proximity",
        n_cpus=n_cpus,
        quiet=True,
    )
    dataset.deseq2()
    statistics = DeseqStats(
        dataset,
        contrast=["proximity", "near", "far"],
        n_cpus=n_cpus,
        quiet=True,
    )
    statistics.summary()
    results = statistics.results_df.copy()
    results.index = pd.Index(results.index.astype(str), name="gene")
    return results.sort_values(["padj", "pvalue"], na_position="last")


def _raw_counts_matrix(table: ad.AnnData) -> sparse.csr_matrix:
    source = table.layers.get("counts", table.X)
    matrix = sparse.csr_matrix(source)
    data = np.asarray(matrix.data)
    if data.size and (not np.isfinite(data).all() or (data < 0).any()):
        raise ValueError("Raw count matrix must contain finite non-negative values.")
    rounded = np.rint(data)
    if data.size and not np.allclose(data, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("Pseudobulk requires unnormalized integer raw counts.")
    matrix.data = rounded.astype(np.int64, copy=False)
    return matrix.astype(np.int64)


def _table_cell_ids(table: ad.AnnData) -> pd.Index:
    if "cell_id" in table.obs.columns:
        return pd.Index(table.obs["cell_id"].astype(str), dtype=str)
    return pd.Index(table.obs_names.astype(str), dtype=str)
