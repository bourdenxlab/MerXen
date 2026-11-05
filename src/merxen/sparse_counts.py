"""Streaming accumulation of integer counts into a sparse matrix."""

from __future__ import annotations

import numpy as np
from scipy import sparse

_MAX_LINEAR_INDEX = np.iinfo(np.int64).max


class SparseCountAccumulator:
    """Count (row, column) pairs streamed in chunks and return a CSR matrix.

    Each chunk is reduced to its unique linear indices ``row * n_cols + col``
    with their counts. Reduced chunks wait until they hold as many entries as
    the sorted totals (and at least ``merge_threshold``), and are then merged
    into them in one linear pass, so the totals are rewritten a logarithmic
    number of times rather than once per chunk, and memory stays bounded by
    the number of distinct pairs rather than the number of rows added.
    Summing one CSR matrix per chunk instead rewrites the whole running
    matrix on every chunk.

    The matrix from :meth:`to_csr` equals summing one ``coo_matrix.tocsr()``
    per chunk into an empty ``csr_matrix`` of dtype int64: same data dtype,
    same index dtype, sorted column indices and no explicit zeros.

    Attributes:
        shape: ``(n_rows, n_cols)`` of the result.
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        *,
        merge_threshold: int = 1 << 24,
    ) -> None:
        """Create an empty accumulator.

        Args:
            n_rows: Number of matrix rows.
            n_cols: Number of matrix columns.
            merge_threshold: Minimum number of pending reduced entries before
                they are merged into the totals.

        Raises:
            ValueError: If a dimension is negative, the matrix has more cells
                than an int64 linear index can address, or
                ``merge_threshold`` is not positive.
        """
        n_rows = int(n_rows)
        n_cols = int(n_cols)
        if n_rows < 0 or n_cols < 0:
            raise ValueError(f"Matrix dimensions must be >= 0, got {(n_rows, n_cols)}")
        if n_rows * n_cols > _MAX_LINEAR_INDEX:
            raise ValueError(
                f"A {n_rows} x {n_cols} matrix exceeds the int64 linear index range"
            )
        if int(merge_threshold) <= 0:
            raise ValueError(f"merge_threshold must be > 0, got {merge_threshold}")
        self.shape = (n_rows, n_cols)
        self._merge_threshold = int(merge_threshold)
        self._keys: np.ndarray = np.zeros(0, dtype=np.int64)
        self._counts: np.ndarray = np.zeros(0, dtype=np.int64)
        self._pending_keys: list[np.ndarray] = []
        self._pending_counts: list[np.ndarray] = []
        self._n_pending = 0

    def add(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Add one count for each ``(rows[i], cols[i])`` pair.

        Args:
            rows: Row index of each observation.
            cols: Column index of each observation, the same length as
                ``rows``.

        Raises:
            ValueError: If the arrays differ in length or an index lies
                outside the matrix.
        """
        row_index = np.asarray(rows, dtype=np.int64).ravel()
        col_index = np.asarray(cols, dtype=np.int64).ravel()
        if row_index.shape != col_index.shape:
            raise ValueError(
                f"rows and cols differ in length: {row_index.size} != {col_index.size}"
            )
        if row_index.size == 0:
            return
        n_rows, n_cols = self.shape
        if (
            int(row_index.min()) < 0
            or int(row_index.max()) >= n_rows
            or int(col_index.min()) < 0
            or int(col_index.max()) >= n_cols
        ):
            raise ValueError(f"Index outside a {n_rows} x {n_cols} matrix")
        keys, counts = np.unique(row_index * n_cols + col_index, return_counts=True)
        self._pending_keys.append(keys)
        self._pending_counts.append(counts.astype(np.int64, copy=False))
        self._n_pending += int(keys.size)
        if self._n_pending >= max(self._merge_threshold, int(self._keys.size)):
            self._merge_pending()

    def to_csr(self) -> sparse.csr_matrix:
        """Return the accumulated counts as an int64 CSR matrix.

        Returns:
            The ``shape`` CSR matrix of counts in canonical form.
        """
        self._merge_pending()
        n_rows, n_cols = self.shape
        if self._keys.size == 0:
            return sparse.csr_matrix(self.shape, dtype=np.int64)
        rows, cols = np.divmod(self._keys, n_cols)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n_rows), out=indptr[1:])
        del rows
        return sparse.csr_matrix(
            (self._counts.copy(), cols, indptr),
            shape=self.shape,
        )

    def _merge_pending(self) -> None:
        """Fold the pending reduced chunks into the sorted running totals."""
        if not self._pending_keys:
            return
        pending_keys, pending_counts = _sum_duplicate_keys(
            np.concatenate(self._pending_keys),
            np.concatenate(self._pending_counts),
        )
        self._pending_keys.clear()
        self._pending_counts.clear()
        self._n_pending = 0
        # Both sides are sorted and unique, so known keys are summed in place
        # and new keys are inserted at their sorted positions without
        # re-sorting the totals.
        positions = np.searchsorted(self._keys, pending_keys)
        is_known = positions < self._keys.size
        is_known[is_known] = self._keys[positions[is_known]] == pending_keys[is_known]
        self._counts[positions[is_known]] += pending_counts[is_known]
        is_new = ~is_known
        self._keys = np.insert(self._keys, positions[is_new], pending_keys[is_new])
        self._counts = np.insert(
            self._counts, positions[is_new], pending_counts[is_new]
        )


def _sum_duplicate_keys(
    keys: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted unique keys and the summed counts of each.

    Args:
        keys: int64 linear indices, possibly repeated and unsorted.
        counts: int64 count of each entry of ``keys``.

    Returns:
        A tuple of (unique sorted keys, summed counts).
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    return sorted_keys[starts], np.add.reduceat(counts[order], starts)
