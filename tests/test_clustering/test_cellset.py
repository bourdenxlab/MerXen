"""Tests for the shared table-cell selection (plan §4.4)."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from merxen.analysis.clustering_squidpy import (
    remove_control_features,
    run_scanpy_clustering,
)
from merxen.clustering.cellset import (
    LEGACY_TOTAL_COUNTS_KEY,
    restrict_to_table_cells,
    row_sums,
    select_table_cells,
)

MIN_COUNTS = 10
CONTROL_NAMES = ("Blank-1", "Blank-2", "NegControlProbe_00042", "UnassignedCodeword_7")


def _synthetic_adata(
    *,
    seed: int = 0,
    matrix_kind: str = "csr",
    dtype: type = np.float32,
    fractional: bool = False,
) -> ad.AnnData:
    """Return cells around the count threshold, with control features.

    Some objects reach ``MIN_COUNTS`` only through control counts, so the
    selection is right only if controls are removed first.
    """
    rng = np.random.default_rng(seed)
    n_genes = 30
    genes = [f"Gene{i}" for i in range(n_genes)]
    blocks = [range(0, 8), range(8, 16), range(16, 24)]
    rows = []
    for block in blocks:
        for _ in range(60):
            lam = np.full(n_genes + len(CONTROL_NAMES), 0.3)
            lam[list(block)] = 4.0
            lam[n_genes:] = 0.5
            rows.append(rng.poisson(lam))
    for _ in range(50):
        lam = np.full(n_genes + len(CONTROL_NAMES), 0.25)
        lam[n_genes:] = 2.5
        rows.append(rng.poisson(lam))
    counts = np.asarray(rows, dtype=float)
    # Objects exactly at, just below and just above the threshold, with
    # enough control counts to lift the below-threshold ones over it.
    for row, gene_total in ((-1, MIN_COUNTS), (-2, MIN_COUNTS - 1), (-3, 11)):
        counts[row, :n_genes] = 0
        counts[row, :gene_total] = 1
        counts[row, n_genes:] = 3
    if fractional:
        counts = counts * rng.uniform(0.85, 1.15, size=counts.shape)
    counts = counts.astype(dtype)
    matrix: np.ndarray | sparse.spmatrix | sparse.sparray
    if matrix_kind == "csr":
        matrix = sparse.csr_matrix(counts)
    elif matrix_kind == "csc":
        matrix = sparse.csc_matrix(counts)
    elif matrix_kind == "csr_array":
        matrix = sparse.csr_array(counts)
    else:
        matrix = counts
    var_names = genes + list(CONTROL_NAMES)
    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"cell_area": rng.uniform(5.0, 50.0, counts.shape[0])},
            index=[f"cell_{i}" for i in range(counts.shape[0])],
        ),
        var=pd.DataFrame(index=var_names),
    )


def _legacy_filter(adata: ad.AnnData) -> ad.AnnData:
    """The legacy cell filter before the refactor: controls, then filter_cells."""
    filtered = remove_control_features(adata)
    sc.pp.filter_cells(filtered, min_counts=MIN_COUNTS)
    return filtered


MATRIX_CASES = [
    ("csr", np.float32, False),
    ("csc", np.float64, False),
    ("csr_array", np.float32, False),
    ("dense", np.float32, False),
    ("dense", np.int32, False),
    ("csr", np.int64, False),
    ("csr", np.uint16, False),
    ("csr", np.float32, True),
    ("dense", np.float64, True),
]


@pytest.mark.parametrize(("matrix_kind", "dtype", "fractional"), MATRIX_CASES)
def test_select_table_cells_matches_legacy_filter(
    matrix_kind: str, dtype: type, fractional: bool
) -> None:
    """The selection equals scanpy's legacy filter, set and counts."""
    adata = _synthetic_adata(
        matrix_kind=matrix_kind, dtype=dtype, fractional=fractional
    )
    control_free = remove_control_features(adata)
    expected = _legacy_filter(adata)

    selection = select_table_cells(control_free, MIN_COUNTS)

    assert selection.table_cell_ids.equals(expected.obs_names)
    assert 0 < selection.n_table_cells < selection.n_objects
    legacy_mask, legacy_counts = sc.pp.filter_cells(
        control_free, min_counts=MIN_COUNTS, inplace=False
    )
    np.testing.assert_array_equal(selection.is_table_cell, legacy_mask)
    assert selection.total_counts.dtype == legacy_counts.dtype
    np.testing.assert_array_equal(selection.total_counts, legacy_counts)


@pytest.mark.parametrize(("matrix_kind", "dtype", "fractional"), MATRIX_CASES)
def test_restrict_to_table_cells_reproduces_filter_cells(
    matrix_kind: str, dtype: type, fractional: bool
) -> None:
    """In-place restriction gives the same AnnData as scanpy.pp.filter_cells."""
    adata = _synthetic_adata(
        matrix_kind=matrix_kind, dtype=dtype, fractional=fractional
    )
    expected = _legacy_filter(adata)
    actual = remove_control_features(adata)

    restrict_to_table_cells(actual, select_table_cells(actual, MIN_COUNTS))

    pd.testing.assert_frame_equal(actual.obs, expected.obs, check_exact=True)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    assert type(actual.X) is type(expected.X)
    assert actual.X.dtype == expected.X.dtype
    left = actual.X.toarray() if sparse.issparse(actual.X) else actual.X
    right = expected.X.toarray() if sparse.issparse(expected.X) else expected.X
    np.testing.assert_array_equal(left, right)
    assert actual.uns.keys() == expected.uns.keys()


def test_control_counts_do_not_count_towards_the_threshold() -> None:
    """Objects above the threshold only through controls are not table cells."""
    adata = _synthetic_adata()
    with_controls = select_table_cells(adata, MIN_COUNTS)
    without_controls = select_table_cells(remove_control_features(adata), MIN_COUNTS)

    lifted = with_controls.table_cell_ids.difference(without_controls.table_cell_ids)
    assert len(lifted) > 0
    assert "cell_229" in without_controls.table_cell_ids  # exactly MIN_COUNTS
    assert "cell_228" in lifted  # MIN_COUNTS - 1 genes + controls
    assert "cell_227" in without_controls.table_cell_ids


def test_run_scanpy_clustering_keeps_the_shared_cell_set() -> None:
    """Legacy clustering keeps exactly the selected cells, with n_counts."""
    adata = _synthetic_adata()
    selection = select_table_cells(remove_control_features(adata), MIN_COUNTS)

    clustered = run_scanpy_clustering(
        adata, min_counts=MIN_COUNTS, n_pcs=5, n_neighbors=10, use_gpu=False
    )

    assert clustered.obs_names.equals(selection.table_cell_ids)
    expected_counts = pd.Series(
        selection.total_counts, index=selection.obs_names, name="n_counts"
    ).loc[clustered.obs_names]
    pd.testing.assert_series_equal(
        clustered.obs[LEGACY_TOTAL_COUNTS_KEY], expected_counts, check_exact=True
    )


def test_select_table_cells_reports_counts_and_genes() -> None:
    """Totals and detected genes follow the matrix; the input is untouched."""
    matrix = sparse.csr_matrix(
        np.array([[0, 3, 7], [1, 0, 0], [0, 0, 0], [5, 5, 0]], dtype=np.float32)
    )
    adata = ad.AnnData(X=matrix, obs=pd.DataFrame(index=["a", "b", "c", "d"]))
    before = adata.copy()

    selection = select_table_cells(adata, 10)

    np.testing.assert_array_equal(selection.total_counts, [10, 1, 0, 10])
    np.testing.assert_array_equal(selection.n_genes, [2, 1, 0, 2])
    assert selection.n_genes.dtype == np.int64
    assert list(selection.table_cell_ids) == ["a", "d"]
    assert (selection.n_objects, selection.n_table_cells) == (4, 2)
    assert selection.min_counts == 10
    assert adata.obs_names.equals(before.obs_names)
    assert adata.obs.columns.empty
    np.testing.assert_array_equal(adata.X.toarray(), before.X.toarray())


def test_select_table_cells_with_zero_threshold_keeps_everything() -> None:
    """``min_counts=0`` keeps empty objects too."""
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))

    selection = select_table_cells(adata, 0)

    assert selection.n_table_cells == 3


def test_select_table_cells_rejects_negative_threshold() -> None:
    """A negative threshold is a configuration error."""
    adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="min_counts"):
        select_table_cells(adata, -1)


def test_restrict_to_table_cells_rejects_foreign_selection() -> None:
    """A selection made on other objects cannot be applied."""
    adata = _synthetic_adata()
    selection = select_table_cells(adata, MIN_COUNTS)
    other = adata[::2].copy()

    with pytest.raises(ValueError, match="different objects"):
        restrict_to_table_cells(other, selection)


def test_restrict_to_table_cells_can_leave_obs_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without a counts column only the subset changes; removals are logged."""
    adata = _synthetic_adata()
    selection = select_table_cells(adata, MIN_COUNTS)

    with caplog.at_level(logging.INFO, logger="merxen.clustering.cellset"):
        restrict_to_table_cells(adata, selection, total_counts_key=None)

    assert list(adata.obs.columns) == ["cell_area"]
    assert adata.n_obs == selection.n_table_cells
    assert "filtered out" in caplog.text


def test_row_sums_accepts_dense_and_sparse() -> None:
    """Row sums are 1-D for numpy arrays, sparse matrices and sparse arrays."""
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)

    for matrix in (values, sparse.csr_matrix(values), sparse.csr_array(values)):
        result = row_sums(matrix)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [3, 7])
