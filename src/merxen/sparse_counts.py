"""Streaming accumulation of integer counts into a sparse matrix."""

from __future__ import annotations

import numpy as np
from scipy import sparse

_MAX_LINEAR_INDEX = np.iinfo(np.int64).max
_MAX_INT32 = np.iinfo(np.int32).max


class SparseCountAccumulator:
    """Count (row, column) pairs streamed in chunks and return a CSR matrix.

    Each chunk is reduced to its unique linear indices ``row * n_cols + col``
    with their counts. Reduced chunks wait until they hold half as many
    entries as the sorted totals (and at least ``merge_threshold``), and are
    then merged into them in one linear pass, so each stored count is
    rewritten a bounded number of times on average rather than once per
    chunk. Summing one CSR matrix per chunk instead rewrites the whole
    running matrix on every chunk.

    Memory: the totals take 16 bytes per distinct (row, column) pair, and the
    pending entries 16 bytes each for at most ``max(merge_threshold,
    nnz / 2)`` entries plus one chunk. The peak, during a merge, is about 28
    bytes per stored count, a little above the 24 bytes of summing CSR
    matrices. :meth:`to_csr` hands the totals over to the result instead of
    copying them, so only the result (12 bytes per stored count with int32
    indices) is left once it returns.

    The matrix from :meth:`to_csr` equals summing one ``coo_matrix.tocsr()``
    per chunk into an empty ``csr_matrix`` of dtype int64: same data dtype,
    same index dtype (int32 unless a dimension or the number of stored counts
    exceeds the int32 range), sorted column indices and no explicit zeros.

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
        self._finished = False

    def add(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Add one count for each ``(rows[i], cols[i])`` pair.

        Args:
            rows: Row index of each observation.
            cols: Column index of each observation, the same length as
                ``rows``.

        Raises:
            ValueError: If the arrays differ in length or an index lies
                outside the matrix.
            RuntimeError: If :meth:`to_csr` has already been called.
        """
        self._check_not_finished()
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
        if self._n_pending >= max(self._merge_threshold, int(self._keys.size) // 2):
            self._merge_pending()

    def to_csr(self) -> sparse.csr_matrix:
        """Return the accumulated counts as an int64 CSR matrix.

        The accumulator hands its buffers to the result, so it can be called
        once and the accumulator takes no counts afterwards.

        Returns:
            The ``shape`` CSR matrix of counts in canonical form.

        Raises:
            RuntimeError: If :meth:`to_csr` has already been called.
        """
        self._check_not_finished()
        self._merge_pending()
        self._finished = True
        keys, counts = self._keys, self._counts
        self._keys = np.zeros(0, dtype=np.int64)
        self._counts = np.zeros(0, dtype=np.int64)
        n_rows, n_cols = self.shape
        if keys.size == 0:
            return sparse.csr_matrix(self.shape, dtype=np.int64)
        # scipy keeps int32 indices exactly when both dimensions and the
        # number of stored counts fit; building them in that dtype avoids an
        # int64 copy that the constructor would narrow again.
        index_dtype = (
            np.int32 if max(n_rows, n_cols, int(keys.size)) <= _MAX_INT32 else np.int64
        )
        row_starts = np.arange(n_rows + 1, dtype=np.int64) * n_cols
        indptr = np.searchsorted(keys, row_starts).astype(index_dtype, copy=False)
        del row_starts
        np.remainder(keys, n_cols, out=keys)
        cols = keys.astype(index_dtype, copy=False)
        del keys
        return sparse.csr_matrix((counts, cols, indptr), shape=self.shape)

    def _check_not_finished(self) -> None:
        """Raise once :meth:`to_csr` has taken the counts.

        Raises:
            RuntimeError: If :meth:`to_csr` has already been called.
        """
        if self._finished:
            raise RuntimeError("The counts were already taken by to_csr()")

    def _merge_pending(self) -> None:
        """Fold the pending reduced chunks into the sorted running totals."""
        if not self._pending_keys:
            return
        # Temporaries are released as soon as they are used, because a merge
        # can hold half as many entries as the totals.
        keys = np.concatenate(self._pending_keys)
        self._pending_keys.clear()
        order = np.argsort(keys)
        keys = keys[order]
        counts = np.concatenate(self._pending_counts)
        self._pending_counts.clear()
        counts = counts[order]
        del order
        self._n_pending = 0
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        keys = keys[starts]
        counts = np.add.reduceat(counts, starts)
        del starts
        # Both sides are sorted and unique: known keys are summed in place,
        # and new keys are placed at their sorted positions among the totals
        # without re-sorting them.
        positions = np.searchsorted(self._keys, keys)
        is_known = positions < self._keys.size
        is_known[is_known] = self._keys[positions[is_known]] == keys[is_known]
        self._counts[positions[is_known]] += counts[is_known]
        is_new = ~is_known
        del is_known
        new_keys = keys[is_new]
        del keys
        new_counts = counts[is_new]
        del counts
        destinations = positions[is_new]
        del positions, is_new
        destinations += np.arange(destinations.size, dtype=destinations.dtype)
        is_old = np.ones(self._keys.size + destinations.size, dtype=bool)
        is_old[destinations] = False
        self._keys = _interleave(self._keys, new_keys, is_old, destinations)
        del new_keys
        self._counts = _interleave(self._counts, new_counts, is_old, destinations)


def _interleave(
    old: np.ndarray,
    new: np.ndarray,
    is_old: np.ndarray,
    destinations: np.ndarray,
) -> np.ndarray:
    """Return ``old`` and ``new`` combined into one array.

    Args:
        old: Values kept in their order where ``is_old`` is true.
        new: Values written at ``destinations``.
        is_old: Boolean mask over the result, false exactly at
            ``destinations``.
        destinations: Positions of ``new`` in the result.

    Returns:
        An array of ``old``'s dtype and ``is_old``'s length.
    """
    merged = np.empty(is_old.size, dtype=old.dtype)
    merged[is_old] = old
    merged[destinations] = new
    return merged
