"""Streaming label-mask lookups (corner convention) and the native label check."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from merxen.enrichment import label_lookup as ll
from merxen.segmentation.cellpose import assign_labels_from_masks

PS = 0.21249222


def _labels(height: int = 23, width: int = 17, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 9, size=(height, width)).astype(np.uint32)


# --------------------------------------------------------------------------- #
# Pixel conventions
# --------------------------------------------------------------------------- #


def test_corner_takes_the_floor_and_center_rounds() -> None:
    """Pixel i covers [i*ps, (i+1)*ps) under corner; center rounds to nearest."""
    x = np.array([0.0, 0.4 * PS, 0.6 * PS, 1.0 * PS, 2.3 * PS, 2.999 * PS])
    y = np.zeros_like(x)
    frame = ll.PixelFrame(PS)
    corner, _ = ll.pixel_indices(x, y, frame, convention="corner")
    center, _ = ll.pixel_indices(x, y, frame, convention="center")
    assert corner.tolist() == [0, 0, 0, 1, 2, 2]
    assert center.tolist() == [0, 0, 1, 1, 2, 3]


def test_float32_precision_is_the_vendor_rule_on_pixel_edges() -> None:
    """float32(x) * float32(1/ps), not x / ps, decides a near-edge transcript.

    The coordinate is from the real crop (a QV 37.75 transcript the vendor
    assigned to the cell on the next row): in float64 it lies 0.000125 px below
    the edge, in the vendor's float32 arithmetic on it.
    """
    y = np.array([1706.3125])
    frame32 = ll.PixelFrame(PS)
    frame64 = ll.PixelFrame(PS, precision="float64")
    row32 = ll.pixel_indices(y, y, frame32)[1]
    row64 = ll.pixel_indices(y, y, frame64)[1]
    assert row64.tolist() == [int(np.floor(1706.3125 / PS))]
    assert row32.tolist() == [row64[0] + 1]
    assert row32.tolist() == [
        int(np.floor(np.float32(1706.3125) * np.float32(1.0 / PS)))
    ]


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_pixel_indices_follow_the_precision_formula(precision: str) -> None:
    rng = np.random.default_rng(9)
    x = (rng.uniform(-5, 20000, 20000) * 64).round() / 64
    column, _ = ll.pixel_indices(
        x,
        x,
        ll.PixelFrame(PS, precision=precision),  # type: ignore[arg-type]
    )
    if precision == "float32":
        expected = np.floor(x.astype(np.float32) * np.float32(1.0 / PS))
    else:
        expected = np.floor(x / PS)
    np.testing.assert_array_equal(column, expected.astype(np.int64))


def test_negative_coordinates_are_off_the_mask() -> None:
    """Negative coordinates floor below 0 and get label 0; rint would not."""
    labels = np.full((4, 4), 7, dtype=np.uint32)
    x = np.array([-0.4 * PS, -0.01, 0.5 * PS])
    y = np.array([0.5 * PS, 0.5 * PS, -0.125])
    column, row = ll.pixel_indices(x, y, ll.PixelFrame(PS))
    assert ll.lookup_labels(column, row, labels).tolist() == [0, 0, 0]
    # The centre convention puts x in [-0.5 ps, 0) on column 0.
    center = assign_labels_from_masks(
        x[:1], y[:1], labels, np.eye(2) / PS, np.zeros(2), pixel_convention="center"
    )
    assert center.tolist() == [7]


def test_float64_precision_upcasts_float32_coordinates() -> None:
    """float32 x and y give the pixels of their float64 values divided by ps."""
    rng = np.random.default_rng(3)
    x32 = (rng.uniform(0, 50, 5000) * 64).round().astype(np.float32) / np.float32(64)
    y32 = (rng.uniform(0, 50, 5000) * 64).round().astype(np.float32) / np.float32(64)
    frame = ll.PixelFrame(PS, precision="float64")
    column, row = ll.pixel_indices(x32, y32, frame)
    np.testing.assert_array_equal(
        column, np.floor(x32.astype(np.float64) / PS).astype(np.int64)
    )
    np.testing.assert_array_equal(
        row, np.floor(y32.astype(np.float64) / PS).astype(np.int64)
    )


def test_float32_pixel_size_is_used_as_given() -> None:
    """A float32 pixel size is widened exactly (not re-rounded to 8 digits)."""
    ps32 = np.float32(PS)
    frame = ll.PixelFrame(float(ps32), precision="float64")
    assert frame.um_per_px == float(ps32)
    x = np.array([float(ps32) * 10.5])
    assert ll.pixel_indices(x, x, frame)[0].tolist() == [10]


def test_an_origin_shifts_the_pixel_grid() -> None:
    for precision in ("float32", "float64"):
        frame = ll.PixelFrame(PS, origin_um=(10 * PS, -2 * PS), precision=precision)  # type: ignore[arg-type]
        column, row = ll.pixel_indices(
            np.array([10.5 * PS]), np.array([0.5 * PS]), frame
        )
        assert (column.tolist(), row.tolist()) == ([0], [2])
    with pytest.raises(ValueError, match="precision"):
        ll.PixelFrame(PS, precision="float16")  # type: ignore[arg-type]


def test_non_finite_coordinates_map_off_the_mask() -> None:
    labels = np.ones((3, 3), dtype=np.uint32)
    x = np.array([np.nan, 0.1, np.inf])
    column, row = ll.pixel_indices(x, np.full(3, 0.1), ll.PixelFrame(PS))
    assert ll.lookup_labels(column, row, labels).tolist() == [0, 1, 0]


def test_unknown_convention_and_pixel_size_are_refused() -> None:
    with pytest.raises(ValueError, match="pixel_convention"):
        ll.pixel_indices(np.zeros(1), np.zeros(1), ll.PixelFrame(PS), convention="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        ll.PixelFrame(0.0)


@pytest.mark.parametrize("block_px", [1, 4, 5, 4096])
@pytest.mark.parametrize("lazy", [False, True])
def test_blockwise_lookup_equals_direct_indexing(block_px: int, lazy: bool) -> None:
    """Reading only touched blocks (numpy or dask) equals indexing the mask."""
    labels = _labels()
    rng = np.random.default_rng(1)
    column = rng.integers(-3, labels.shape[1] + 3, 400)
    row = rng.integers(-3, labels.shape[0] + 3, 400)
    source = da.from_array(labels, chunks=(5, 4)) if lazy else labels
    got = ll.lookup_labels(column, row, source, block_px=block_px)
    inside = (
        (column >= 0)
        & (column < labels.shape[1])
        & (row >= 0)
        & (row < labels.shape[0])
    )
    expected = np.zeros(400, dtype=np.int64)
    expected[inside] = labels[row[inside], column[inside]]
    np.testing.assert_array_equal(got, expected)


def test_corner_lookup_matches_assign_labels_from_masks() -> None:
    """The streaming lookup equals the shared helper with pixel_convention=corner."""
    labels = _labels(31, 29, seed=5)
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 29 * PS + 1, 2000)
    y = rng.uniform(-1, 31 * PS + 1, 2000)
    column, row = ll.pixel_indices(x, y, ll.PixelFrame(PS))
    got = ll.lookup_labels(column, row, labels, block_px=7)
    # The helper works in float64; compare off the pixel edges.
    frac = np.minimum((x / PS) % 1.0, (y / PS) % 1.0)
    away = (frac > 1e-6) & (frac < 1 - 1e-6)
    helper = assign_labels_from_masks(
        x, y, labels, np.eye(2) / PS, np.zeros(2), pixel_convention="corner"
    )
    np.testing.assert_array_equal(got[away], helper[away])


# --------------------------------------------------------------------------- #
# Streaming over partitions
# --------------------------------------------------------------------------- #


def _write_partitions(tmp_path: Path, frame: pd.DataFrame, n_parts: int) -> list[Path]:
    paths = []
    bounds = np.linspace(0, len(frame), n_parts + 1).astype(int)
    for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        path = tmp_path / f"part.{index}.parquet"
        frame.iloc[start:stop].to_parquet(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("n_workers", [1, 4])
def test_stream_label_lookup_reads_parquet_partitions(
    tmp_path: Path, n_workers: int
) -> None:
    """Every row is looked up once; partition results come back in order."""
    labels = _labels(40, 40, seed=7)
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(-0.5, 40 * PS, 300).astype(np.float32),
            "y": rng.uniform(-0.5, 40 * PS, 300).astype(np.float32),
            "gene": rng.integers(0, 5, 300),
        }
    )
    paths = _write_partitions(tmp_path, frame, 5)

    def _reduce(part: pd.DataFrame, looked: dict[str, np.ndarray]) -> np.ndarray:
        assert set(part.columns) == {"x", "y", "gene"}
        return looked["cells"]

    results = ll.stream_label_lookup(
        paths,
        {"cells": labels},
        PS,
        columns=["gene"],
        reduce=_reduce,
        n_workers=n_workers,
        block_px=16,
    )
    column, row = ll.pixel_indices(
        frame["x"].to_numpy(), frame["y"].to_numpy(), ll.PixelFrame(PS)
    )
    np.testing.assert_array_equal(
        np.concatenate(results), ll.lookup_labels(column, row, labels)
    )


def test_stream_label_lookup_accepts_dask_and_pandas_points() -> None:
    import dask.dataframe as dd

    labels = _labels(20, 20, seed=8)
    frame = pd.DataFrame({"x": np.linspace(0, 4, 50), "y": np.linspace(4, 0, 50)})

    def _reduce(part: pd.DataFrame, looked: dict[str, np.ndarray]) -> np.ndarray:
        return looked["m"]

    from_pandas = ll.stream_label_lookup(
        frame, {"m": labels}, PS, columns=[], reduce=_reduce
    )
    from_dask = ll.stream_label_lookup(
        dd.from_pandas(frame, npartitions=3),
        {"m": labels},
        PS,
        columns=[],
        reduce=_reduce,
        n_workers=2,
    )
    np.testing.assert_array_equal(
        np.concatenate(from_pandas), np.concatenate(from_dask)
    )


class _CountingMask:
    """Numpy mask that counts the block reads (the cache misses)."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels
        self.shape = labels.shape
        self.reads = 0

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads += 1
        return self.labels[key]  # type: ignore[index]


@pytest.mark.parametrize("n_workers", [1, 4])
def test_the_block_cache_decodes_each_block_once_across_partitions(
    tmp_path: Path, n_workers: int
) -> None:
    """Partitions touching the same blocks share one decode per block."""
    labels = _labels(40, 40, seed=7)
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0, 40 * PS, 400).astype(np.float32),
            "y": rng.uniform(0, 40 * PS, 400).astype(np.float32),
        }
    )
    paths = _write_partitions(tmp_path, frame, 8)
    expected = ll.lookup_labels(
        *ll.pixel_indices(
            frame["x"].to_numpy(), frame["y"].to_numpy(), ll.PixelFrame(PS)
        ),
        labels,
    )

    def _run(cache_bytes: int) -> tuple[np.ndarray, int]:
        mask = _CountingMask(labels)
        results = ll.stream_label_lookup(
            paths,
            {"cells": mask},
            PS,
            columns=[],
            reduce=lambda part, looked: looked["cells"],
            n_workers=n_workers,
            block_px=16,
            block_cache_bytes=cache_bytes,
        )
        return np.concatenate(results), mask.reads

    cached, cached_reads = _run(ll.DEFAULT_BLOCK_CACHE_BYTES)
    uncached, uncached_reads = _run(0)
    np.testing.assert_array_equal(cached, expected)
    np.testing.assert_array_equal(uncached, expected)
    assert cached_reads == 9  # 3 x 3 blocks of 16 px over a 40 px mask
    assert uncached_reads == 8 * 9


def test_the_block_cache_evicts_least_recently_used_within_its_budget() -> None:
    block = np.zeros(10, dtype=np.uint32)  # 40 bytes
    cache = ll.BlockCache(100)
    loads: list[str] = []

    def _load(name: str) -> np.ndarray:
        loads.append(name)
        return block.copy()

    for name in ("a", "b", "a", "c", "b"):
        cache.get(name, lambda name=name: _load(name))
    # "b" was evicted by "c" (a was used more recently), then decoded again.
    assert loads == ["a", "b", "c", "b"]
    assert (cache.hits, cache.misses) == (1, 4)
    big = ll.BlockCache(10)
    big.get("x", lambda: _load("x"))
    big.get("x", lambda: _load("x"))
    assert loads[-2:] == ["x", "x"]  # larger than the budget: never kept
    with pytest.raises(ValueError, match="max_bytes"):
        ll.BlockCache(-1)


def test_the_block_cache_decodes_a_contended_block_once() -> None:
    """Threads asking for one block at once share a single decode."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cache = ll.BlockCache(1 << 20)
    started = threading.Event()
    release = threading.Event()
    loads = []

    def _load() -> np.ndarray:
        loads.append(1)
        started.set()
        release.wait(5)
        return np.ones(4, dtype=np.uint32)

    with ThreadPoolExecutor(4) as pool:
        first = pool.submit(cache.get, "k", _load)
        assert started.wait(5)
        others = [pool.submit(cache.get, "k", _load) for _ in range(3)]
        release.set()
        blocks = [first.result(), *(future.result() for future in others)]
    assert len(loads) == 1
    assert all(block is blocks[0] for block in blocks)


def test_a_failed_block_decode_is_raised_and_not_kept() -> None:
    cache = ll.BlockCache(1 << 20)

    def _fail() -> np.ndarray:
        raise OSError("unreadable chunk")

    with pytest.raises(OSError, match="unreadable chunk"):
        cache.get("k", _fail)
    assert cache.get("k", lambda: np.ones(2, dtype=np.uint32)).tolist() == [1, 1]


def test_partitions_are_ordered_naturally() -> None:
    names = ["part.10.parquet", "part.2.parquet", "part.1.parquet", "part.0.parquet"]
    ordered = ll.natural_partition_order([Path(name) for name in names])
    assert [path.name for path in ordered] == [
        "part.0.parquet",
        "part.1.parquet",
        "part.2.parquet",
        "part.10.parquet",
    ]


def test_masks_of_different_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="differ in shape"):
        ll.stream_label_lookup(
            pd.DataFrame({"x": [0.0], "y": [0.0]}),
            {"a": np.zeros((2, 2)), "b": np.zeros((3, 2))},
            PS,
            columns=[],
            reduce=lambda part, looked: None,
        )


def test_default_n_workers_reads_dask_num_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASK_NUM_WORKERS", "3")
    assert ll.default_n_workers() == 3
    monkeypatch.setenv("DASK_NUM_WORKERS", "zero")
    with pytest.raises(ValueError, match="DASK_NUM_WORKERS"):
        ll.default_n_workers()
    monkeypatch.delenv("DASK_NUM_WORKERS")
    assert 1 <= ll.default_n_workers() <= 8


# --------------------------------------------------------------------------- #
# Native label check on a hand-made mask
# --------------------------------------------------------------------------- #


def _hand_store() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Two cells; cell 2 owns nuclei 2 and 3; nucleus 1 belongs to cell 1."""
    cells = np.zeros((6, 8), dtype=np.uint32)
    cells[0:3, 0:3] = 1
    cells[0:6, 4:8] = 2
    nuclei = np.zeros_like(cells)
    nuclei[1, 1] = 1
    nuclei[1, 5] = 2
    nuclei[4, 6] = 3
    pixels = [(1, 1), (0, 0), (1, 5), (4, 6), (4, 6), (5, 7), (5, 0), (2, 2)]
    rows = np.array([p[0] for p in pixels])
    cols = np.array([p[1] for p in pixels])
    assignment = cells[rows, cols].astype(float)
    assignment[assignment == 0] = np.nan
    frame = pd.DataFrame(
        {
            "x": ((cols + 0.5) * PS).astype(np.float32),
            "y": ((rows + 0.5) * PS).astype(np.float32),
            "original_assignment": pd.array(assignment, dtype="UInt64"),
            "overlaps_nucleus": nuclei[rows, cols] > 0,
            "is_gene": [True, True, True, False, True, True, True, True],
        }
    )
    return cells, nuclei, frame


def _check(
    frame: pd.DataFrame, cells: np.ndarray, nuclei: np.ndarray, **kwargs: object
) -> tuple[dict, dict]:
    return ll.native_label_check(
        frame,
        cell_labels=cells,
        nucleus_labels=nuclei,
        nucleus_ids=np.array([1, 2, 3]),
        nucleus_cell_ids=np.array([1, 2, 2]),
        table_cell_ids=np.array([2, 1]),
        um_per_px=PS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_native_check_counts_nuclear_transcripts_per_owning_cell() -> None:
    """Both nuclei of cell 2 count for cell 2; nucleus ids never key cells."""
    cells, nuclei, frame = _hand_store()
    summary, per_cell = _check(frame, cells, nuclei)

    assert summary["passed"] is True
    assert summary["n_mismatch"] == 0
    assert summary["n_rows"] == 8
    # Table order is (cell 2, cell 1).
    assert per_cell["cell"].tolist() == [4, 3]
    assert per_cell["nucleus"].tolist() == [3, 1]
    assert per_cell["nucleus_gene"].tolist() == [2, 1]
    assert per_cell["cell_gene"].tolist() == [3, 3]
    nucleus = summary["nucleus"]
    assert nucleus["overlaps_agreement"] == 1.0
    assert nucleus["n_nucleus_outside_own_cell"] == 0
    assert nucleus["n_unknown_nucleus_labels"] == 0
    assert summary["failures"] == []
    assert set(summary["checks"]) == {
        "cell_assignment",
        "cell_labels_in_table",
        "nucleus_labels_known",
        "nuclei_inside_own_cell",
        "nucleus_overlaps",
    }


def test_native_check_reports_and_fails_on_vendor_disagreement() -> None:
    cells, nuclei, frame = _hand_store()
    frame.loc[1, "original_assignment"] = 2  # on cell 1's pixel
    frame.loc[6, "original_assignment"] = 1  # off every cell
    summary, _ = _check(frame, cells, nuclei)
    assert summary["n_mismatch_label_differs"] == 2
    assert summary["passed"] is False

    frame2 = _hand_store()[2]
    frame2.loc[0, "original_assignment"] = pd.NA  # vendor unassigned, on a cell
    summary2, _ = _check(frame2, cells, nuclei, tolerance=0.2)
    assert summary2["n_mismatch_vendor_unassigned"] == 1
    assert summary2["mismatch_allowed"] == 1
    assert summary2["passed"] is True


def test_native_check_flags_unknown_nuclei_and_overlap_disagreement() -> None:
    cells, nuclei, frame = _hand_store()
    nuclei = nuclei.copy()
    nuclei[2, 2] = 9  # a nucleus label without a shape row
    frame.loc[0, "overlaps_nucleus"] = False
    summary, per_cell = _check(frame, cells, nuclei)
    assert summary["nucleus"]["n_unknown_nucleus_labels"] == 1
    assert summary["nucleus"]["n_overlaps_vendor_only"] == 0
    assert summary["nucleus"]["n_overlaps_lookup_only"] == 2
    assert per_cell["nucleus"].tolist() == [3, 1]
    assert summary["n_mismatch"] == 0
    assert summary["passed"] is False
    assert summary["failures"] == ["nucleus_labels_known", "nucleus_overlaps"]


def test_a_nucleus_label_without_an_owning_cell_is_unknown() -> None:
    """A nucleus row whose cell_instance_id is 0 is unknown, not cell 0's."""
    cells, nuclei, frame = _hand_store()
    nuclei = nuclei.copy()
    nuclei[0, 0] = 4  # nucleus 4 exists but has no owning cell
    frame.loc[1, "overlaps_nucleus"] = True  # the row on pixel (0, 0)
    summary, per_cell = ll.native_label_check(
        frame,
        cell_labels=cells,
        nucleus_labels=nuclei,
        nucleus_ids=np.array([1, 2, 3, 4]),
        nucleus_cell_ids=np.array([1, 2, 2, 0]),
        table_cell_ids=np.array([2, 1]),
        um_per_px=PS,
    )
    assert summary["nucleus"]["n_unknown_nucleus_labels"] == 1
    assert summary["nucleus"]["n_nucleus_outside_own_cell"] == 0
    assert per_cell["nucleus"].tolist() == [3, 1]
    assert summary["failures"] == ["nucleus_labels_known"]
    assert summary["passed"] is False


def test_a_nucleus_on_another_cell_fails_the_check() -> None:
    """A wrong cell_instance_id (nucleus 1 said to be cell 2's) is caught."""
    cells, nuclei, frame = _hand_store()
    summary, per_cell = ll.native_label_check(
        frame,
        cell_labels=cells,
        nucleus_labels=nuclei,
        nucleus_ids=np.array([1, 2, 3]),
        nucleus_cell_ids=np.array([2, 2, 2]),
        table_cell_ids=np.array([2, 1]),
        um_per_px=PS,
    )
    assert summary["n_mismatch"] == 0
    assert summary["nucleus"]["n_nucleus_outside_own_cell"] == 1
    assert summary["checks"]["nuclei_inside_own_cell"] is False
    assert summary["failures"] == ["nuclei_inside_own_cell"]
    assert summary["passed"] is False


def test_a_cell_label_above_the_table_fails_the_check() -> None:
    cells, nuclei, frame = _hand_store()
    cells = cells.copy()
    cells[5, 0] = 7  # a cell label no table row has; row 6 sits on it
    frame.loc[6, "original_assignment"] = 7
    summary, _ = _check(frame, cells, nuclei)
    assert summary["n_mismatch"] == 0
    assert summary["n_cell_labels_outside_table"] == 1
    assert summary["failures"] == ["cell_labels_in_table"]
    assert summary["passed"] is False


def test_native_check_without_optional_columns() -> None:
    cells, nuclei, frame = _hand_store()
    summary, per_cell = _check(
        frame[["x", "y", "original_assignment"]],
        cells,
        nuclei,
        available_columns=["x", "y", "original_assignment"],
    )
    assert set(per_cell) == {"cell", "nucleus"}
    assert "overlaps_agreement" not in summary["nucleus"]


def test_native_check_validates_the_nucleus_inputs() -> None:
    cells, nuclei, frame = _hand_store()
    with pytest.raises(ValueError, match="positive and unique"):
        ll.native_label_check(
            frame,
            cell_labels=cells,
            nucleus_labels=nuclei,
            nucleus_ids=np.array([1, 1]),
            nucleus_cell_ids=np.array([1, 2]),
            table_cell_ids=np.array([1, 2]),
            um_per_px=PS,
        )
    with pytest.raises(ValueError, match="needs nucleus_ids"):
        ll.native_label_check(
            frame,
            cell_labels=cells,
            nucleus_labels=nuclei,
            nucleus_ids=None,
            nucleus_cell_ids=None,
            table_cell_ids=np.array([1, 2]),
            um_per_px=PS,
        )
