"""Table-cell selection shared by legacy and map-first clustering (plan §4.4).

The clustered H5AD and the SpatialData table written back hold exactly the
objects with ``total_counts >= min_counts`` after control features are
removed. Legacy ``run_scanpy_clustering`` and the annotation steps both select
cells with ``select_table_cells``, so their cell sets, and therefore MENDER's,
are identical.

The input must already be free of control features: legacy clustering removes
them with ``remove_control_features`` first, annotation with its control
registry (plan §8.4).

This module imports only numpy and pandas, so the GPU clustering environment
(no ``cell_type_mapper``, no SpatialData) can import it (plan §11.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# ``obs`` column in which legacy clustering records the counts per cell;
# ``scanpy.pp.filter_cells(adata, min_counts=...)`` wrote it before the
# selection moved here.
LEGACY_TOTAL_COUNTS_KEY: Final = "n_counts"


@dataclass(frozen=True, eq=False)
class TableCellSelection:
    """Which objects of an AnnData become table cells.

    Attributes:
        obs_names: Names of all input objects, in input order.
        total_counts: Counts per object summed over the input features, with
            the dtype numpy gives the row sum (as ``scanpy.pp.filter_cells``).
        n_genes: Number of features with a positive count per object, or
            ``None`` unless requested (``compute_n_genes``); legacy
            clustering never reads it, so it skips the extra pass.
        is_table_cell: ``total_counts >= min_counts`` per object.
        min_counts: The threshold used.
    """

    obs_names: pd.Index
    total_counts: np.ndarray
    n_genes: np.ndarray | None
    is_table_cell: np.ndarray
    min_counts: int

    @property
    def n_objects(self) -> int:
        """Return the number of input objects."""
        return int(self.is_table_cell.size)

    @property
    def n_table_cells(self) -> int:
        """Return the number of selected table cells."""
        return int(np.count_nonzero(self.is_table_cell))

    @property
    def table_cell_ids(self) -> pd.Index:
        """Return the names of the selected table cells, in input order."""
        return self.obs_names[self.is_table_cell]


def row_sums(matrix: Any) -> np.ndarray:
    """Return the row sums of a dense or sparse matrix as a 1-D array.

    Args:
        matrix: A numpy array or a scipy sparse matrix or array.

    Returns:
        One sum per row, in numpy's sum dtype (``int64`` for signed integers,
        ``uint64`` for unsigned ones, the input dtype for floats), which is
        also what ``scanpy.pp.filter_cells`` records.
    """
    return np.asarray(matrix.sum(axis=1)).ravel()


def select_table_cells(
    adata: ad.AnnData, min_counts: int, *, compute_n_genes: bool = False
) -> TableCellSelection:
    """Select the objects that become table cells.

    Args:
        adata: Objects x features, without control features; counts in ``X``.
        min_counts: Minimum total counts of a table cell
            (``ClusteringSquidpyConfig.min_counts``; 10 by default).
        compute_n_genes: Also count the detected features per object (the
            annotation steps need ``n_genes``). Off by default, so legacy
            clustering does exactly the work ``scanpy.pp.filter_cells`` did
            (no ``X > 0`` pass or temporary matrix).

    Returns:
        The selection; ``adata`` is not modified.

    Raises:
        ValueError: If ``min_counts`` is negative or ``adata.X`` is missing.
    """
    threshold = int(min_counts)
    if threshold < 0:
        raise ValueError(f"min_counts must be >= 0, got {min_counts}")
    matrix = adata.X
    if matrix is None:
        raise ValueError("select_table_cells needs counts in adata.X")
    total_counts = row_sums(matrix)
    return TableCellSelection(
        obs_names=adata.obs_names.copy(),
        total_counts=total_counts,
        n_genes=row_sums(matrix > 0) if compute_n_genes else None,
        is_table_cell=np.asarray(total_counts >= threshold, dtype=bool),
        min_counts=threshold,
    )


def restrict_to_table_cells(
    adata: ad.AnnData,
    selection: TableCellSelection,
    *,
    total_counts_key: str | None = LEGACY_TOTAL_COUNTS_KEY,
) -> None:
    """Subset ``adata`` in place to the selected table cells.

    Reproduces ``scanpy.pp.filter_cells(adata, min_counts=...)``: the counts of
    every object are first stored in ``obs[total_counts_key]``, then the
    objects below the threshold are dropped.

    Args:
        adata: The AnnData the selection was made on.
        selection: Output of ``select_table_cells`` for ``adata``.
        total_counts_key: ``obs`` column for the counts per cell, or ``None``
            to leave ``obs`` unchanged.

    Raises:
        ValueError: If the selection was made on different objects.
    """
    if not selection.obs_names.equals(adata.obs_names):
        raise ValueError("the table-cell selection was made on different objects")
    if total_counts_key is not None:
        adata.obs[total_counts_key] = selection.total_counts
    n_removed = selection.n_objects - selection.n_table_cells
    if n_removed > 0:
        logger.info(
            "filtered out %d cells that have less than %d counts",
            n_removed,
            selection.min_counts,
        )
    adata._inplace_subset_obs(selection.is_table_cell)
