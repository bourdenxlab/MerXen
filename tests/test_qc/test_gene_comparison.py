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
    compute_gene_comparison,
    compute_gene_summary,
    fit_linear,
    gene_totals_from_points,
    normalize_counts,
)


def test_apply_dataset_filter_merscope_removes_blank() -> None:
    """MERSCOPE filter should drop blank control codewords."""
    counts = pd.Series([10, 2, 4], index=["GAD1", "Blank-01", "AQP4"])
    out = apply_dataset_filter(counts, "MERSCOPE")
    assert "Blank-01" not in out.index
    assert set(out.index) == {"GAD1", "AQP4"}


def test_apply_dataset_filter_xenium_removes_control_codewords() -> None:
    """Xenium filter should drop negative-control and unassigned features."""
    counts = pd.Series(
        [10, 1, 2, 3, 4],
        index=[
            "GAD1",
            "NegControlProbe_00001",
            "NegControlCodeword_0500",
            "UnassignedCodeword_0010",
            "AQP4",
        ],
    )
    out = apply_dataset_filter(counts, "xenium")
    assert set(out.index) == {"GAD1", "AQP4"}


# Gene symbols from Xenium and Atera panels whose names contain fragments of
# control prefixes or tokens; none of them is a control.
TOKEN_LIKE_GENES = [
    "GFAP",
    "HLA-DMB",
    "Nkx6-1",
    "BLNK",
    "NEGR1",
    "DEPP1",
    "UNC13A",
    "INTS1",
    "CNTNAP2",
    "ENST00000262410.10_MAPT3R_exon9_exon11",
]


@pytest.mark.parametrize("dataset_name", ["XENIUM", "MERSCOPE"])
def test_apply_dataset_filter_keeps_token_like_gene_names(dataset_name: str) -> None:
    """Gene names that only contain control-like fragments are kept."""
    counts = pd.Series(np.arange(len(TOKEN_LIKE_GENES)), index=TOKEN_LIKE_GENES)

    out = apply_dataset_filter(counts, dataset_name)

    assert out.index.tolist() == TOKEN_LIKE_GENES


def test_apply_dataset_filter_merscope_keeps_xenium_style_names() -> None:
    """The MERSCOPE rule drops only ``Blank`` codewords."""
    counts = pd.Series(
        [1, 2, 3, 4],
        index=["Gad1", "Blank-7", "Intergenic_Region_1", "DeprecatedCodeword_1"],
    )

    out = apply_dataset_filter(counts, "MERSCOPE")

    assert out.index.tolist() == ["Gad1", "Intergenic_Region_1", "DeprecatedCodeword_1"]


def test_apply_dataset_filter_xenium_keeps_unlisted_control_prefixes() -> None:
    """The Xenium filter lists three prefixes and misses the other controls."""
    counts = pd.Series(
        [10, 1, 2, 3, 4, 5],
        index=[
            "GAD1",
            "DeprecatedCodeword_0001",
            "Intergenic_Region_12",
            "BLANK_0006",
            "antisense_PROKR2",
            "NegControlProbe_00001",
        ],
    )

    out = apply_dataset_filter(counts, "XENIUM")

    assert out.index.tolist() == [
        "GAD1",
        "DeprecatedCodeword_0001",
        "Intergenic_Region_12",
        "BLANK_0006",
        "antisense_PROKR2",
    ]


def test_apply_dataset_filter_unknown_platform_is_unfiltered() -> None:
    """Platforms without a rule are returned as they are."""
    counts = pd.Series([1, 2], index=["GAD1", "NegControlProbe_00001"])

    out = apply_dataset_filter(counts, "ATERA")

    assert out.index.tolist() == ["GAD1", "NegControlProbe_00001"]


def test_compute_gene_summary_xenium_keeps_unlisted_controls_in_totals() -> None:
    """Deprecated and genomic-control features enter the Xenium totals."""
    table = ad.AnnData(
        X=np.array([[4, 1]], dtype=np.int64),
        var=pd.DataFrame(
            {"feature_types": ["Gene Expression", "Gene Expression"]},
            index=["A", "B"],
        ),
    )
    points = pd.DataFrame(
        {
            "feature_name": [
                "A",
                "A",
                "B",
                "NegControlCodeword_0500",
                "DeprecatedCodeword_0001",
                "Intergenic_Region_12",
            ]
        }
    )
    sdata_obj = SimpleNamespace(tables={"table": table}, points={"transcripts": points})

    out = compute_gene_summary(sdata_obj, "XENIUM")

    assert out["total_counts_df"]["gene"].tolist() == [
        "A",
        "B",
        "DeprecatedCodeword_0001",
        "Intergenic_Region_12",
    ]
    assert out["totals"] == {"total_sum": 5.0, "assigned_sum": 5.0}


def test_compute_gene_comparison_excludes_controls_from_totals() -> None:
    """Platform controls should not enter the normalization totals."""
    xenium_points = pd.DataFrame(
        {"feature_name": ["A", "A", "B", "NegControlProbe_00001"]}
    )
    merscope_points = pd.DataFrame({"gene": ["A", "B", "B", "Blank-0", "Blank-1"]})
    xenium_table = ad.AnnData(
        X=np.array([[2, 1, 5]], dtype=np.int64),
        var=pd.DataFrame(index=["A", "B", "UnassignedCodeword_0001"]),
    )
    merscope_table = ad.AnnData(
        X=np.array([[1, 2, 7]], dtype=np.int64),
        var=pd.DataFrame(index=["A", "B", "Blank-0"]),
    )
    xenium_sdata = SimpleNamespace(
        tables={"table": xenium_table}, points={"transcripts": xenium_points}
    )
    merscope_sdata = SimpleNamespace(
        tables={"table": merscope_table}, points={"transcripts": merscope_points}
    )

    out = compute_gene_comparison(xenium_sdata, merscope_sdata)

    assert out["totals"] == {
        "xenium_total_sum": 3.0,
        "merscope_total_sum": 3.0,
        "xenium_assigned_sum": 3.0,
        "merscope_assigned_sum": 3.0,
    }
    norm = out["total_normalized_df"].set_index("gene")
    assert np.isclose(norm.loc["A", "xenium"], 2.0 / 3.0)
    assert np.isclose(norm.loc["B", "merscope"], 2.0 / 3.0)


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


def test_compute_gene_summary_drops_merscope_blanks() -> None:
    """Single-platform MERSCOPE summaries should not list blank codewords."""
    table = ad.AnnData(
        X=np.array([[3, 1]], dtype=np.int64),
        var=pd.DataFrame(index=["A", "Blank-0"]),
    )
    points = pd.DataFrame({"gene": ["A", "A", "Blank-0", "Blank-3"]})
    sdata_obj = SimpleNamespace(tables={"table": table}, points={"transcripts": points})

    out = compute_gene_summary(sdata_obj, "MERSCOPE")

    assert out["total_counts_df"]["gene"].tolist() == ["A"]
    assert out["assigned_counts_df"]["gene"].tolist() == ["A"]
    assert out["totals"] == {"total_sum": 2.0, "assigned_sum": 3.0}
