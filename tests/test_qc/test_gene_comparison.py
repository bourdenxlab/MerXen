"""Tests for gene comparison helper functions."""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merxen.qc.gene_comparison import (
    apply_dataset_filter,
    compare_df,
    compute_gene_summary,
    fit_linear,
    gene_totals_from_points,
    normalize_counts,
)


def test_apply_dataset_filter_xenium_removes_blank() -> None:
    """Xenium filter should drop blank control probes."""
    counts = pd.Series([10, 2, 4], index=["GAD1", "Blank-01", "AQP4"])
    out = apply_dataset_filter(counts, "XENIUM")
    assert "Blank-01" not in out.index
    assert set(out.index) == {"GAD1", "AQP4"}


def test_normalize_counts_returns_unit_sum() -> None:
    """Normalized gene counts should sum to 1."""
    counts = pd.Series([2.0, 3.0], index=["A", "B"])
    norm, total = normalize_counts(counts)
    assert np.isclose(total, 5.0)
    assert np.isclose(float(norm.sum()), 1.0)


def test_normalize_counts_raises_on_non_positive_total() -> None:
    """Normalization should fail for zero-total inputs."""
    counts = pd.Series([0.0, 0.0], index=["A", "B"])
    with pytest.raises(ValueError, match="total count"):
        normalize_counts(counts)


def test_compare_df_intersects_gene_sets() -> None:
    """Comparison DF should only include shared genes."""
    x = pd.Series([1.0, 2.0], index=["A", "B"])
    m = pd.Series([5.0, 6.0], index=["B", "C"])
    out = compare_df(x, m)
    assert out["gene"].tolist() == ["B"]
    assert out["xenium"].tolist() == [2.0]
    assert out["merscope"].tolist() == [5.0]


def test_fit_linear_returns_expected_slope() -> None:
    """Linear fit should recover slope/intercept on noiseless data."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    slope, intercept, r2 = fit_linear(x, y)
    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, 1.0)
    assert np.isclose(r2, 1.0)


def test_compute_gene_summary_uses_requested_table_key() -> None:
    """Assigned gene counts should come from the branch-specific table."""
    default = ad.AnnData(
        X=np.array([[10, 0]], dtype=np.int64),
        var=pd.DataFrame(index=["A", "B"]),
    )
    original = ad.AnnData(
        X=np.array([[0, 5]], dtype=np.int64),
        var=pd.DataFrame(index=["A", "B"]),
    )
    points = pd.DataFrame({"feature_name": ["A", "B", "B"]})
    sdata_obj = SimpleNamespace(
        tables={"table": default, "table_original": original},
        points={"transcripts": points},
    )

    out = compute_gene_summary(
        sdata_obj,
        "MERSCOPE",
        table_key="table_original",
    )

    assigned = out["assigned_counts_df"].set_index("gene")["count"]
    assert assigned["A"] == 0.0
    assert assigned["B"] == 5.0
    assert out["table_key"] == "table_original"


def test_gene_totals_from_points_uses_background_column_for_proseg() -> None:
    """Assigned-only point counts should keep ProSeg cell id 0."""
    points = pd.DataFrame(
        {
            "gene": ["A", "B", "A"],
            "assignment": pd.Series([0, pd.NA, 2], dtype="UInt32"),
            "background": [False, True, False],
        }
    )
    sdata_obj = SimpleNamespace(points={"transcripts": points})

    counts = gene_totals_from_points(sdata_obj, assigned_only=True)

    assert counts.to_dict() == {"A": 2.0}
