"""Tests for the streaming sparse count accumulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from merxen.sparse_counts import SparseCountAccumulator


def _per_chunk_csr_sum(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    """Sum one CSR matrix per chunk, as enrichment did before streaming."""
    total = sparse.csr_matrix(shape, dtype=np.int64)
    for rows, cols in chunks:
        if rows.size == 0:
            continue
        total = (
            total
            + sparse.coo_matrix(
                (np.ones(rows.size, dtype=np.int64), (rows, cols)), shape=shape
            ).tocsr()
        )
    return total


def _accumulate(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
    **kwargs: int,
) -> sparse.csr_matrix:
    accumulator = SparseCountAccumulator(*shape, **kwargs)
    for rows, cols in chunks:
        accumulator.add(rows, cols)
    return accumulator.to_csr()


def _random_chunks(
    shape: tuple[int, int],
    n_chunks: int,
    chunk_size: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chunks that repeat pairs both within and across chunks."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = shape
    hot_rows = rng.integers(0, n_rows, 64)
    hot_cols = rng.integers(0, n_cols, 64)
    chunks = []
    for _ in range(n_chunks):
        rows = rng.integers(0, n_rows, chunk_size)
        cols = rng.integers(0, n_cols, chunk_size)
        hot = rng.random(chunk_size) < 0.3
        rows[hot] = rng.choice(hot_rows, int(hot.sum()))
        cols[hot] = rng.choice(hot_cols, int(hot.sum()))
        chunks.append((rows, cols))
    return chunks


def _assert_identical_csr(
    actual: sparse.csr_matrix, expected: sparse.csr_matrix
) -> None:
    assert isinstance(actual, sparse.csr_matrix)
    assert actual.shape == expected.shape
    for name in ("data", "indices", "indptr"):
        actual_array = getattr(actual, name)
        expected_array = getattr(expected, name)
        assert actual_array.dtype == expected_array.dtype, name
        np.testing.assert_array_equal(actual_array, expected_array, err_msg=name)
    assert actual.has_canonical_format


@pytest.mark.parametrize("merge_threshold", [1, 500, 1 << 24])
def test_accumulator_matches_the_per_chunk_csr_sum(merge_threshold: int) -> None:
    """Every merge schedule gives the matrix that summed CSR chunks gave."""
    shape = (300, 40)
    chunks = _random_chunks(shape, n_chunks=25, chunk_size=400, seed=0)

    result = _accumulate(chunks, shape, merge_threshold=merge_threshold)

    _assert_identical_csr(result, _per_chunk_csr_sum(chunks, shape))


def test_accumulator_handles_a_whole_transcriptome_sized_matrix() -> None:
    """Linear indices beyond the int32 range keep row and column apart."""
    shape = (91_431, 24_677)
    chunks = _random_chunks(shape, n_chunks=40, chunk_size=25_000, seed=1)

    result = _accumulate(chunks, shape, merge_threshold=50_000)

    expected = _per_chunk_csr_sum(chunks, shape)
    assert int(np.max(chunks[0][0])) * shape[1] > np.iinfo(np.int32).max
    _assert_identical_csr(result, expected)
    assert int(result.sum()) == 40 * 25_000


def test_accumulator_matches_int64_indices_for_very_wide_matrices() -> None:
    """Index dtype follows scipy when a dimension exceeds the int32 range."""
    shape = (3, 2**31 + 10)
    chunks = [
        (np.array([0, 2, 2]), np.array([5, 2**31 + 9, 2**31 + 9])),
        (np.array([1, 0]), np.array([2**31, 5])),
    ]

    result = _accumulate(chunks, shape)

    expected = _per_chunk_csr_sum(chunks, shape)
    assert expected.indices.dtype == np.int64
    _assert_identical_csr(result, expected)


@pytest.mark.parametrize("shape", [(5, 4), (0, 4), (5, 0)])
def test_empty_accumulator_returns_an_empty_int64_matrix(
    shape: tuple[int, int],
) -> None:
    """Without counts the result equals an empty int64 CSR matrix."""
    accumulator = SparseCountAccumulator(*shape)
    accumulator.add(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))

    _assert_identical_csr(
        accumulator.to_csr(), sparse.csr_matrix(shape, dtype=np.int64)
    )


def test_accumulator_is_finished_once_its_result_is_taken() -> None:
    """``to_csr`` hands the counts over, so later use is an error."""
    accumulator = SparseCountAccumulator(2, 2)
    accumulator.add(np.array([0, 0, 1]), np.array([1, 1, 0]))

    result = accumulator.to_csr()

    np.testing.assert_array_equal(result.toarray(), [[0, 2], [1, 0]])
    with pytest.raises(RuntimeError, match="already taken"):
        accumulator.add(np.array([0]), np.array([0]))
    with pytest.raises(RuntimeError, match="already taken"):
        accumulator.to_csr()


def test_accumulator_merges_a_logarithmic_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending entries stay below half the totals, merged in growing batches."""
    merge_threshold = 100
    accumulator = SparseCountAccumulator(1000, 100, merge_threshold=merge_threshold)
    merge = SparseCountAccumulator._merge_pending
    n_merges = 0

    def _counting_merge(self: SparseCountAccumulator) -> None:
        nonlocal n_merges
        n_merges += 1
        merge(self)

    monkeypatch.setattr(SparseCountAccumulator, "_merge_pending", _counting_merge)
    keys = np.random.default_rng(3).permutation(1000 * 100)
    for chunk in np.array_split(keys, 500):
        accumulator.add(*np.divmod(chunk, 100))
        assert accumulator._n_pending < max(
            merge_threshold, accumulator._keys.size // 2
        )

    # 500 chunks of 200 new pairs merge whenever the pending pairs reach half
    # the totals: about log(500) / log(1.5) times, not once per chunk.
    assert n_merges <= 20
    assert accumulator.to_csr().nnz == keys.size


@pytest.mark.parametrize(
    ("rows", "cols"),
    [([0, 3], [0, 0]), ([0, -1], [0, 0]), ([0, 0], [0, 2]), ([0, 0], [-1, 0])],
)
def test_accumulator_rejects_indices_outside_the_matrix(
    rows: list[int], cols: list[int]
) -> None:
    """Out-of-range indices raise instead of folding into another cell."""
    accumulator = SparseCountAccumulator(3, 2)

    with pytest.raises(ValueError, match="outside"):
        accumulator.add(np.array(rows), np.array(cols))


def test_accumulator_rejects_mismatched_lengths() -> None:
    """Rows and columns must pair up one to one."""
    accumulator = SparseCountAccumulator(3, 2)

    with pytest.raises(ValueError, match="differ in length"):
        accumulator.add(np.array([0, 1]), np.array([0]))


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "merge_threshold", "message"),
    [
        (-1, 2, 1, "dimensions"),
        (2**32, 2**32, 1, "int64"),
        (2, 2, 0, "merge_threshold"),
    ],
)
def test_accumulator_rejects_invalid_construction(
    n_rows: int, n_cols: int, merge_threshold: int, message: str
) -> None:
    """Negative sizes, unaddressable shapes and a zero threshold are errors."""
    with pytest.raises(ValueError, match=message):
        SparseCountAccumulator(n_rows, n_cols, merge_threshold=merge_threshold)
