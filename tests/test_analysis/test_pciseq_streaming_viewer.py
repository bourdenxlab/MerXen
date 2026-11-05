"""Disk-backed viewer joins preserve exact IDs without whole-column computes."""

from __future__ import annotations

import pickle
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from merxen.analysis.pciseq_viewer import (
    _join_partition,
    _spot_index,
    _SpotIndex,
    _validate_native_ids,
)


def _index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SpotIndex:
    monkeypatch.setattr("merxen.analysis.pciseq_viewer._CHUNK_ROWS", 1)
    pd.DataFrame(
        {
            "transcript_id": pd.Series([2**64 - 1, 3, 2**63 + 1], dtype="uint64"),
            "instance_id": pd.array([17, None, 2**64 - 7], dtype="UInt64"),
            "gene_name": ["A", "NA", "B"],
        }
    ).to_csv(tmp_path / "spots.csv", index=False)
    return _spot_index(tmp_path / "spots.csv", None, tmp_path, 3)


def test_disk_index_is_small_serializable_and_joins_shuffled_native_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _index(tmp_path, monkeypatch)
    assert len(pickle.dumps(index)) < 2_000
    assert all(isinstance(array, np.memmap) for array in index.arrays())
    frame = pd.DataFrame(
        {
            "transcript_id": pd.Series([3, 71, 2**64 - 1, 2**63 + 1], dtype="uint64"),
            "gene": ["NA", "other", "A", "B"],
            "x": [1.25, 11.0, 21.5, 31.75],
            "y": [-19.0, -12.0, -2.5, 9.25],
        }
    )

    # Only individual delayed partitions may be materialized, never all IDs.
    def forbidden_compute(*args: object, **kwargs: object) -> None:
        pytest.fail("Native IDs were materialized as a full Dask Series")

    monkeypatch.setattr(dd.Series, "compute", forbidden_compute)
    assert _validate_native_ids(dd.from_pandas(frame, npartitions=3), index) == 4
    result = _join_partition(frame, index=index, gene_column="gene")
    pd.testing.assert_frame_equal(result[["x", "y"]], frame[["x", "y"]])
    assert result.assignment.tolist() == [pd.NA, pd.NA, 17, 2**64 - 7]
    assert str(result.assignment.dtype) == "UInt64"
    assert result.background.tolist() == [True, False, False, False]
    assert result.pciseq_assignment_status.tolist() == [
        "background",
        "not_modelled",
        "assigned",
        "assigned",
    ]


@pytest.mark.parametrize("fault", ["duplicate", "missing"])
def test_native_global_id_validation_spans_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    index = _index(tmp_path, monkeypatch)
    values = [3, 71, 2**64 - 1, 2**63 + 1]
    if fault == "duplicate":
        values[-1] = values[0]
    else:
        values[-1] = 99
    native = dd.from_pandas(
        pd.DataFrame({"transcript_id": pd.Series(values, dtype="uint64")}),
        npartitions=3,
    )
    with pytest.raises(ValueError, match="duplicate|absent"):
        _validate_native_ids(native, index)
