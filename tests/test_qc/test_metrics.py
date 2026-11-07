"""Tests for QC metric helper logic."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from merxen.qc.metrics import (
    _compute_cell_metrics_from_points,
    _compute_cell_metrics_from_table,
    _eccentricity_aspect,
)


def test_eccentricity_aspect_returns_nan_for_none() -> None:
    """None geometry should return NaN metrics."""
    ecc, aspect = _eccentricity_aspect(None)
    assert np.isnan(ecc)
    assert np.isnan(aspect)


def test_compute_cell_metrics_from_points_with_pandas_input() -> None:
    """Per-cell metrics should aggregate transcripts and genes correctly."""
    points = pd.DataFrame(
        {
            "assignment": [1, 1, 2, 0, 2],
            "feature_name": ["A", "B", "A", "C", "A"],
        }
    )
    n_total, n_assigned, cell_metrics = _compute_cell_metrics_from_points(
        points,
        assign_col="assignment",
        gene_col="feature_name",
    )

    assert n_total == 5
    assert n_assigned == 4
    assert set(cell_metrics.columns) >= {
        "cell_id_norm",
        "transcripts_per_cell",
        "genes_per_cell",
    }

    row1 = cell_metrics[cell_metrics["cell_id_norm"] == "1"].iloc[0]
    row2 = cell_metrics[cell_metrics["cell_id_norm"] == "2"].iloc[0]
    assert int(row1["transcripts_per_cell"]) == 2
    assert int(row1["genes_per_cell"]) == 2
    assert int(row2["transcripts_per_cell"]) == 2
    assert int(row2["genes_per_cell"]) == 1


def test_compute_cell_metrics_from_points_uses_background_column() -> None:
    """ProSeg cell id 0 should be assigned when background is false."""
    points = pd.DataFrame(
        {
            "assignment": pd.Series([0, pd.NA, 2], dtype="UInt32"),
            "background": [False, True, False],
            "feature_name": ["A", "B", "A"],
        }
    )

    n_total, n_assigned, cell_metrics = _compute_cell_metrics_from_points(
        points,
        assign_col="assignment",
        gene_col="feature_name",
    )

    assert n_total == 3
    assert n_assigned == 2
    assert set(cell_metrics["cell_id_norm"]) == {"0", "2"}


def test_compute_cell_metrics_from_table_uses_instance_key() -> None:
    """Table-backed QC should aggregate row sums and preserve table cell IDs."""
    adata = ad.AnnData(
        X=np.array([[1, 0, 3], [0, 0, 2]], dtype=np.int64),
        obs=pd.DataFrame({"cell_id": ["a", "b"]}, index=["row1", "row2"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    adata.uns["spatialdata_attrs"] = {"instance_key": "cell_id"}

    n_assigned, cell_metrics = _compute_cell_metrics_from_table(adata)

    assert n_assigned == 6
    assert cell_metrics["cell_id_norm"].tolist() == ["a", "b"]
    assert cell_metrics["transcripts_per_cell"].tolist() == [4.0, 2.0]
    assert cell_metrics["genes_per_cell"].tolist() == [2.0, 1.0]


@pytest.mark.parametrize("is_sparse", [True, False])
def test_compute_cell_metrics_from_table_totals_are_exact_past_float32(
    is_sparse: bool,
) -> None:
    """Assigned totals past 2**24 stay exact for a float32 ``X`` on any platform.

    Summing this fixture in float32 gives 240,333,536 against the exact
    240,333,534, so the float64 per-cell sums and int64 total are what keep it
    right.
    """
    rng = np.random.default_rng(7)
    counts = rng.integers(1, 2_000, size=(60_000, 4)).astype(np.float32)
    table = ad.AnnData(X=sp.csr_matrix(counts) if is_sparse else counts)
    exact_total = int(counts.astype(np.int64).sum())
    assert exact_total > 2**24
    float32_total = int(np.nansum(np.asarray(sp.csr_matrix(counts).sum(axis=1))))
    assert float32_total != exact_total

    n_assigned, cell_metrics = _compute_cell_metrics_from_table(table)

    assert isinstance(n_assigned, int)
    assert n_assigned == exact_total
    assert np.array_equal(
        cell_metrics["transcripts_per_cell"].to_numpy(),
        counts.astype(np.int64).sum(axis=1).astype(float),
    )
    assert cell_metrics["transcripts_per_cell"].dtype == np.float64
