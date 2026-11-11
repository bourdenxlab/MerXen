"""QC of a native ATERA latest store (AT-6 WP-C).

QC runs on the reference native latest store
(``tests/test_io/atera_native_fixture.py``). Its transcript count and per-gene
totals come from the build aggregates while the points index matches the
parquet files, and fall back to dask streaming otherwise; the registration
check reads only the indexed files that overlap its windows. Nothing may load
every transcript into pandas.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa_arrow
import pyarrow.parquet as pq
import pytest
import spatialdata as sd
from dask.dataframe.dask_expr._collection import FrameBase
from test_io.atera_native_fixture import (
    CELL_SHAPES_KEY,
    POINTS_KEY,
    TABLE_KEY,
    make_native_reference,
)

from merxen.io import points_aggregates as pa
from merxen.qc import gene_comparison, metrics, registration
from merxen.qc.metrics import compute_dataset_qc
from merxen.qc.registration import (
    RegistrationCheckParams,
    compute_segmentation_registration_qc,
    transcripts_in_windows,
    window_parquet_files,
)


@pytest.fixture(scope="module")
def latest(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    _, store = make_native_reference(tmp_path_factory.mktemp("qc_native"))
    yield store


@pytest.fixture
def stale_latest(latest: Path, tmp_path: Path) -> Path:
    """A copy whose first parquet file lost a row after the build."""
    store = Path(shutil.copytree(latest, tmp_path / "stale.zarr"))
    first = sorted(pa.points_parquet_dir(store).glob("*.parquet"))[0]
    table = pq.read_table(first)
    pq.write_table(table.slice(0, table.num_rows - 1), first)
    return store


def _dask_count(store: Path) -> int:
    return int(sd.read_zarr(store).points[POINTS_KEY].shape[0].compute())


def _no_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("QC must not load every transcript into pandas")

    monkeypatch.setattr(metrics, "to_pandas", _refuse)
    monkeypatch.setattr(gene_comparison, "to_pandas", _refuse)


def _no_dask_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on any dask computation over a dataframe (a pass over the points).

    The dask fallbacks give the same numbers as the fast paths, so only a
    refused computation shows that a fast path was taken.
    """

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("QC must not stream the points when the index matches")

    monkeypatch.setattr(FrameBase, "compute", _refuse)


def _spy_read_table(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record the parquet files the registration check reads directly."""
    calls: list[Path] = []
    read_table = pq.read_table

    def _spy(path: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(Path(path))
        return read_table(path, *args, **kwargs)

    monkeypatch.setattr(registration.pq, "read_table", _spy)
    return calls


def _with_feature_counts(store: Path, extra: dict[str, int]) -> None:
    """Append features to the build's per-gene totals of a store copy."""
    v3 = store / "zarr.json"
    path = v3 if v3.is_file() else store / ".zattrs"
    document = json.loads(path.read_text())
    attrs = document["attributes"] if path == v3 else document
    transcripts = attrs[pa.ATERA_ATTRS_KEY]["transcripts"]
    transcripts["feature_names"] = [*transcripts["feature_names"], *extra]
    transcripts["gene_counts_kept"] = [
        *transcripts["gene_counts_kept"],
        *extra.values(),
    ]
    path.write_text(json.dumps(document))


def test_native_qc_counts_from_the_aggregates(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_pandas(monkeypatch)
    aggregates = pa.read_points_aggregates(latest)
    assert aggregates is not None and aggregates.fingerprint_matches()
    expected_count = _dask_count(latest)
    _no_dask_pass(monkeypatch)
    reads = _spy_read_table(monkeypatch)

    result = compute_dataset_qc(
        latest,
        "ATERA_SYNTH",
        table_key=TABLE_KEY,
        shape_key=CELL_SHAPES_KEY,
        registration_reference_shape_key=CELL_SHAPES_KEY,
    )

    summary = result["summary"]
    assert summary["n_transcripts_total"] == aggregates.n_points_written
    assert summary["n_transcripts_total"] == expected_count
    # The registration check read the indexed files, not the dask points.
    points_dir = aggregates.points_dir
    assert [path for path in reads if path.parent == points_dir]
    assert summary["shape_key"] == CELL_SHAPES_KEY
    assert summary["table_key"] == TABLE_KEY
    table = sd.read_zarr(latest).tables[TABLE_KEY]
    assert summary["n_cells"] == table.n_obs
    assert summary["n_transcripts_assigned"] == int(table.X.sum())


def test_reference_equal_to_tested_shape_is_skipped(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry reference of ATERA original is the tested shape itself."""
    calls: list[int] = []
    centroids = registration.shape_centroids

    def _counting(shapes: Any) -> np.ndarray:
        calls.append(1)
        return centroids(shapes)

    monkeypatch.setattr(registration, "shape_centroids", _counting)
    with_reference = compute_segmentation_registration_qc(
        sd.read_zarr(latest),
        shape_key=CELL_SHAPES_KEY,
        points_key=POINTS_KEY,
        reference_shape_key=CELL_SHAPES_KEY,
        zarr_path=latest,
    )
    assert len(calls) == 1  # only the tested shapes
    without = compute_segmentation_registration_qc(
        sd.read_zarr(latest),
        shape_key=CELL_SHAPES_KEY,
        points_key=POINTS_KEY,
        zarr_path=latest,
    )
    assert with_reference.to_dict() == without.to_dict()
    assert with_reference.shift_um is None
    assert all(row["shift_x_um"] is None for row in with_reference.per_window)


def test_point_count_fast_path_equals_dask(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = sd.read_zarr(latest).points[POINTS_KEY]
    expected = metrics._point_count(points)
    assert expected == _dask_count(latest)

    _no_dask_pass(monkeypatch)
    fast = metrics._point_count(points, latest, POINTS_KEY)

    assert fast == expected


def test_gene_totals_fast_path_equals_dask(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdata = sd.read_zarr(latest)
    points = sdata.points[POINTS_KEY]
    expected = points[["feature_name"]].groupby("feature_name").size().compute()
    expected.index = expected.index.astype(str)
    expected = expected[expected > 0].groupby(level=0).sum().astype(float)

    _no_pandas(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(
            gene_comparison, "_gene_totals_from_aggregates", lambda *_args: None
        )
        streamed = gene_comparison.gene_totals_from_points(sdata, assigned_only=False)
    fast = gene_comparison._gene_totals_from_aggregates(
        sdata, POINTS_KEY, "feature_name"
    )
    _no_dask_pass(monkeypatch)
    totals = gene_comparison.gene_totals_from_points(sdata, assigned_only=False)

    assert fast is not None
    pd.testing.assert_series_equal(totals, fast)
    # Same values, index and index name as the dask path.
    pd.testing.assert_series_equal(totals.sort_index(), streamed.sort_index())
    pd.testing.assert_series_equal(
        totals.sort_index(), expected.sort_index(), check_names=False
    )


def test_gene_totals_drop_features_without_transcripts(
    latest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature the build counted zero times is absent, as in a groupby."""
    store = Path(shutil.copytree(latest, tmp_path / "zero.zarr"))
    _with_feature_counts(store, {"ZERO_FEATURE": 0})
    aggregates = pa.read_points_aggregates(store)
    assert aggregates is not None and aggregates.fingerprint_matches()
    assert aggregates.gene_counts()["ZERO_FEATURE"] == 0
    sdata = sd.read_zarr(store)
    _no_dask_pass(monkeypatch)

    totals = gene_comparison.gene_totals_from_points(sdata, assigned_only=False)

    assert "ZERO_FEATURE" not in totals.index
    assert (totals > 0).all()
    assert totals.sum() == aggregates.n_points_written


def test_stale_index_falls_back_to_dask(
    stale_latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregates = pa.read_points_aggregates(stale_latest)
    assert aggregates is not None and not aggregates.fingerprint_matches()
    _no_pandas(monkeypatch)
    sdata = sd.read_zarr(stale_latest)
    points = sdata.points[POINTS_KEY]

    count = metrics._point_count(points, stale_latest, POINTS_KEY)
    totals = gene_comparison.gene_totals_from_points(sdata, assigned_only=False)

    assert count == _dask_count(stale_latest) == aggregates.n_points_written - 1
    assert (
        gene_comparison._gene_totals_from_aggregates(sdata, POINTS_KEY, "feature_name")
        is None
    )
    assert totals.sum() == count
    assert window_parquet_files(stale_latest, POINTS_KEY, [(0, 10, 0, 10)]) is None


def test_dask_failure_raises_without_pandas_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_pandas(monkeypatch)

    def _broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("synthetic dask failure")

    monkeypatch.setattr(metrics, "assignment_mask_from_points", _broken)
    points = dd.from_pandas(
        pd.DataFrame({"assignment": [1, 0, 2], "feature_name": ["A", "B", "A"]}),
        npartitions=2,
    )

    with pytest.raises(RuntimeError, match="does not fall back"):
        metrics._compute_cell_metrics_from_points(
            points,
            assign_col="assignment",
            gene_col="feature_name",
            allow_pandas_fallback=False,
        )


def test_dask_failure_keeps_the_pandas_fallback_without_aggregates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """XENIUM and MERSCOPE stores (no build aggregates) keep the old fallback."""
    real_mask = metrics.assignment_mask_from_points
    calls = {"dask": 0}

    def _broken_on_dask(frame: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(frame, pd.DataFrame) and calls["dask"] == 0:
            calls["dask"] += 1
            raise ValueError("synthetic dask failure")
        return real_mask(frame, *args, **kwargs)

    monkeypatch.setattr(metrics, "assignment_mask_from_points", _broken_on_dask)
    points = dd.from_pandas(
        pd.DataFrame({"assignment": [1, 0, 2], "feature_name": ["A", "B", "A"]}),
        npartitions=1,
    )

    with caplog.at_level("WARNING", logger=metrics.logger.name):
        n_total, n_assigned, cells = metrics._compute_cell_metrics_from_points(
            points, assign_col="assignment", gene_col="feature_name"
        )

    assert calls["dask"] == 1
    assert "falling back to pandas" in caplog.text
    assert (n_total, n_assigned) == (3, 2)
    assert sorted(cells["cell_id_norm"]) == ["1", "2"]


def test_qc_refuses_the_pandas_fallback_on_stores_with_aggregates(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compute_dataset_qc turns the fallback off for a store with aggregates."""
    seen: dict[str, bool] = {}

    def _record(*_args: Any, allow_pandas_fallback: bool = True, **_kw: Any) -> Any:
        seen["allow"] = allow_pandas_fallback
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(metrics, "_compute_cell_metrics_from_points", _record)
    # The native latest store names its assignment column original_assignment.
    real_first = metrics.first_existing_col

    def _first(frame: Any, candidates: list[str]) -> str | None:
        if "assignment" in candidates:
            return "original_assignment"
        return real_first(frame, candidates)

    monkeypatch.setattr(metrics, "first_existing_col", _first)
    with pytest.raises(RuntimeError, match="stop after recording"):
        compute_dataset_qc(
            latest,
            "ATERA_SYNTH",
            table_key=None,
            shape_key=CELL_SHAPES_KEY,
            registration_check=False,
        )
    assert seen == {"allow": False}


def _windows(latest: Path) -> list[tuple[float, float, float, float]]:
    aggregates = pa.read_points_aggregates(latest)
    assert aggregates is not None
    bounds = aggregates.bounds_um
    x_mid = (bounds["x_min"] + bounds["x_max"]) / 2.0
    y_mid = (bounds["y_min"] + bounds["y_max"]) / 2.0
    return [
        (bounds["x_min"], x_mid, bounds["y_min"], y_mid),
        (x_mid, bounds["x_max"] + 1.0, y_mid, bounds["y_max"] + 1.0),
        (1.0e6, 1.0e6 + 10.0, 1.0e6, 1.0e6 + 10.0),
    ]


def _sorted_rows(xy: np.ndarray) -> np.ndarray:
    return xy[np.lexsort((xy[:, 1], xy[:, 0]))]


@pytest.mark.parametrize("gene_col", [None, "feature_name"])
def test_window_pruning_matches_a_full_scan(latest: Path, gene_col: str | None) -> None:
    points = sd.read_zarr(latest).points[POINTS_KEY]
    windows = _windows(latest)
    files = window_parquet_files(latest, POINTS_KEY, windows)
    assert files is not None

    pruned = transcripts_in_windows(
        points,
        windows,
        x_col="x",
        y_col="y",
        gene_col=gene_col,
        platform="ATERA",
        parquet_files=files,
    )
    full = transcripts_in_windows(
        points, windows, x_col="x", y_col="y", gene_col=gene_col, platform="ATERA"
    )

    assert len(full) > 0
    np.testing.assert_array_equal(_sorted_rows(pruned), _sorted_rows(full))


def test_far_window_reads_no_file(latest: Path) -> None:
    far = [(1.0e6, 1.0e6 + 10.0, 1.0e6, 1.0e6 + 10.0)]

    files = window_parquet_files(latest, POINTS_KEY, far)
    xy = transcripts_in_windows(
        None, far, x_col="x", y_col="y", parquet_files=files or []
    )

    assert files == []
    assert xy.shape == (0, 2)


def test_registration_with_pruning_equals_full_scan(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdata = sd.read_zarr(latest)
    params = RegistrationCheckParams(
        window_size_um=16.0, radius_um=2.0, min_centroids=1
    )

    full = compute_segmentation_registration_qc(
        sdata, shape_key=CELL_SHAPES_KEY, points_key=POINTS_KEY, params=params
    )
    centroids = registration.shape_centroids(sdata.shapes[CELL_SHAPES_KEY])
    windows = registration.sample_registration_windows(centroids, params)
    expected_files = window_parquet_files(latest, POINTS_KEY, windows)
    assert expected_files is not None

    _no_dask_pass(monkeypatch)
    reads = _spy_read_table(monkeypatch)
    pruned = compute_segmentation_registration_qc(
        sdata,
        shape_key=CELL_SHAPES_KEY,
        points_key=POINTS_KEY,
        params=params,
        zarr_path=latest,
    )

    assert reads == expected_files
    assert full.mean_transcripts_at_centroids  # windows hold transcripts
    assert pruned.to_dict() == full.to_dict()


def test_xenium_store_is_never_pruned(tmp_path: Path) -> None:
    """Stores without build aggregates keep the dask scan."""
    store = tmp_path / "plain.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"attributes": {}}')

    assert window_parquet_files(store, POINTS_KEY, [(0, 1, 0, 1)]) is None
    assert window_parquet_files(None, POINTS_KEY, [(0, 1, 0, 1)]) is None


def test_atera_control_names_follow_the_registry() -> None:
    """ATERA controls use the 10x rules of merxen.control_features."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_name": [
                "SNAP25",
                "BLANK_0042",
                "antisense_GAD1",
                "NegControlProbe_7",
                "GAD1",
            ],
        }
    )

    xy = transcripts_in_windows(
        dd.from_pandas(frame, npartitions=2),
        [(0.0, 10.0, 0.0, 10.0)],
        x_col="x",
        y_col="y",
        gene_col="feature_name",
        platform="ATERA",
    )

    np.testing.assert_allclose(xy, [[1.0, 1.0], [5.0, 5.0]])


def test_pruning_follows_rows_moved_between_files(latest: Path, tmp_path: Path) -> None:
    """Rows swapped between two files keep every row count, so the index
    fingerprint still matches; pruning reads the footers and still finds them.
    """

    store = Path(shutil.copytree(latest, tmp_path / "swapped.zarr"))
    aggregates = pa.read_points_aggregates(store)
    assert aggregates is not None
    # Two files whose indexed boxes do not overlap.
    first, second = (
        entry
        for entry in aggregates.files
        if entry.name in {"part.0.parquet", "part.2.parquet"}
    )
    assert first.bbox_um is not None and second.bbox_um is not None
    assert first.bbox_um[1] > second.bbox_um[3]  # disjoint in y
    a = pq.read_table(aggregates.points_dir / first.name)
    b = pq.read_table(aggregates.points_dir / second.name)
    moved = a.slice(a.num_rows - 1, 1)
    pq.write_table(
        pa_arrow.concat_tables(
            [a.slice(0, a.num_rows - 1), b.slice(b.num_rows - 1, 1).cast(a.schema)]
        ),
        aggregates.points_dir / first.name,
    )
    pq.write_table(
        pa_arrow.concat_tables([b.slice(0, b.num_rows - 1), moved.cast(b.schema)]),
        aggregates.points_dir / second.name,
    )
    assert aggregates.fingerprint_matches()
    x = float(moved.column("x")[0].as_py())
    y = float(moved.column("y")[0].as_py())
    window = [(x - 0.01, x + 0.01, y - 0.01, y + 0.01)]
    # The index box of the file the row left still claims it.
    assert aggregates.files_intersecting((x - 0.01, y - 0.01, x + 0.01, y + 0.01)) == [
        aggregates.points_dir / first.name
    ]

    files = window_parquet_files(store, POINTS_KEY, window)
    points = sd.read_zarr(store).points[POINTS_KEY]
    pruned = transcripts_in_windows(
        points, window, x_col="x", y_col="y", parquet_files=files
    )
    full = transcripts_in_windows(points, window, x_col="x", y_col="y")

    assert files is not None and aggregates.points_dir / second.name in files
    assert len(full) == 1
    np.testing.assert_array_equal(_sorted_rows(pruned), _sorted_rows(full))


def test_pruning_without_coordinate_statistics_reads_every_file(latest: Path) -> None:
    """Points named other than the indexed x/y columns are never pruned away."""
    aggregates = pa.read_points_aggregates(latest)
    assert aggregates is not None
    far = [(1.0e6, 1.0e6 + 10.0, 1.0e6, 1.0e6 + 10.0)]

    files = window_parquet_files(
        latest, POINTS_KEY, far, x_col="global_x", y_col="global_y"
    )

    assert files == [aggregates.points_dir / entry.name for entry in aggregates.files]


def test_native_qc_assigned_total_is_exact(latest: Path) -> None:
    result = compute_dataset_qc(
        latest,
        "ATERA_SYNTH",
        table_key=TABLE_KEY,
        shape_key=CELL_SHAPES_KEY,
        registration_check=False,
    )
    table = sd.read_zarr(latest).tables[TABLE_KEY]
    expected = int(np.rint(np.asarray(table.X.sum(axis=1, dtype=np.float64))).sum())
    assert result["summary"]["n_transcripts_assigned"] == expected
