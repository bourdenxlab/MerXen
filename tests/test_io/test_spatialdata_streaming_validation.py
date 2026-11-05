"""Tests for the partition-by-partition deep schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

from merxen.io.spatialdata_io import prepare_source_spatialdata_contract
from merxen.io.spatialdata_schema import (
    SpatialDataContractError,
    validate_merxen_schema,
    validate_merxen_schema_streaming,
)

from .contract_synth import (
    N_PARTITIONS,
    ROWS_PER_PARTITION,
    PartitionReads,
    canonical_spatialdata_like,
    xenium_raw_spatialdata,
)

N_ROWS = ROWS_PER_PARTITION * N_PARTITIONS
TILES = ["t0"] * 10 + ["t1"] * 14


def _canonical_store(**columns: object) -> tuple[sd.SpatialData, PartitionReads]:
    """Return a contract-ready canonical Xenium-like store.

    Args:
        **columns: Column values to replace on the canonical points.

    Returns:
        The store after ``prepare_source_spatialdata_contract`` and the read
        counter of its points (reset).
    """
    reference, _ = xenium_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="XENIUM")
    frame = reference.points["transcripts"].compute().reset_index(drop=True)
    frame["source_tile"] = pd.Categorical(TILES)
    for column, values in columns.items():
        frame[column] = values
    raw, _ = xenium_raw_spatialdata()
    canonical, reads = canonical_spatialdata_like(
        reference, raw, frame=frame, read_delay_s=0.01
    )
    prepare_source_spatialdata_contract(
        canonical, platform="XENIUM", points_canonical=True
    )
    reads.reset()
    return canonical, reads


def test_streaming_validation_passes_and_reads_one_partition_at_a_time() -> None:
    """Valid points are read once, one partition in memory at a time."""
    sdata, reads = _canonical_store()

    summaries = validate_merxen_schema_streaming(
        sdata,
        expected_partition_rows={"transcripts": [ROWS_PER_PARTITION] * N_PARTITIONS},
        group_column="source_tile",
        expected_group_rows={"transcripts": {"t0": 10, "t1": 14}},
    )

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)
    assert reads.peak_in_flight == 1
    summary = summaries["transcripts"]
    assert summary.n_rows == N_ROWS
    assert summary.partition_rows == (ROWS_PER_PARTITION,) * N_PARTITIONS
    assert dict(summary.group_rows) == {"t0": 10, "t1": 14}
    assert (summary.transcript_id_min, summary.transcript_id_max) == (1, N_ROWS)
    assert summary.reread_partitions == 0
    validate_merxen_schema(sdata, deep=True)


def test_streaming_validation_rejects_a_duplicate_id_in_one_partition() -> None:
    """A repeated transcript ID inside a partition fails."""
    ids = np.arange(1, N_ROWS + 1, dtype=np.uint64)
    ids[3] = ids[2]
    sdata, _ = _canonical_store(transcript_id=ids)

    with pytest.raises(SpatialDataContractError, match="partition 0: duplicate"):
        validate_merxen_schema_streaming(sdata)


def test_streaming_validation_rejects_a_duplicate_id_across_partitions() -> None:
    """Overlapping ID ranges are re-read together and a repeat fails."""
    ids = np.arange(1, N_ROWS + 1, dtype=np.uint64)
    ids[ROWS_PER_PARTITION] = ids[0]
    sdata, _ = _canonical_store(transcript_id=ids)

    with pytest.raises(SpatialDataContractError, match=r"partitions \[0, 1\]"):
        validate_merxen_schema_streaming(sdata)


def test_streaming_validation_accepts_interleaved_unique_ids() -> None:
    """Unique IDs whose partition ranges interleave pass after one re-read."""
    ids = np.arange(1, N_ROWS + 1, dtype=np.uint64).reshape(-1, N_PARTITIONS)
    sdata, reads = _canonical_store(transcript_id=ids.T.ravel())

    summary = validate_merxen_schema_streaming(sdata)["transcripts"]

    assert summary.reread_partitions == N_PARTITIONS
    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 2)


def test_streaming_validation_rejects_a_non_positive_id() -> None:
    """Transcript IDs keep the positive-ID contract."""
    ids = np.arange(0, N_ROWS, dtype=np.uint64)
    sdata, _ = _canonical_store(transcript_id=ids)

    with pytest.raises(SpatialDataContractError, match="positive"):
        validate_merxen_schema_streaming(sdata)


def test_streaming_validation_rejects_partition_count_mismatch() -> None:
    """Partition row counts reconcile with the expected counts."""
    sdata, _ = _canonical_store()
    expected = [ROWS_PER_PARTITION] * N_PARTITIONS
    expected[-1] += 1

    with pytest.raises(SpatialDataContractError, match="partition row counts"):
        validate_merxen_schema_streaming(
            sdata, expected_partition_rows={"transcripts": expected}
        )


def test_streaming_validation_rejects_group_count_mismatch() -> None:
    """Rows per group (e.g. per tile) reconcile with the expected counts."""
    sdata, _ = _canonical_store()

    with pytest.raises(SpatialDataContractError, match="t2"):
        validate_merxen_schema_streaming(
            sdata,
            group_column="source_tile",
            expected_group_rows={"transcripts": {"t0": 10, "t1": 14, "t2": 1}},
        )


def test_streaming_validation_rejects_an_orphan_assignment() -> None:
    """Every assignment must reference a shape of its branch."""
    reference, _ = xenium_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="XENIUM")
    assignments = (
        reference.points["transcripts"]
        .compute()["original_assignment"]
        .reset_index(drop=True)
        .copy()
    )
    assignments.iloc[-1] = 999
    sdata, _ = _canonical_store(original_assignment=assignments)

    with pytest.raises(SpatialDataContractError, match=r"partition 3 .*\[999\]"):
        validate_merxen_schema_streaming(sdata)


def test_streaming_validation_rejects_table_shape_mismatch() -> None:
    """Table and shape identifier sets must be equal."""
    sdata, reads = _canonical_store()
    table = sdata.tables["table"]
    sdata.tables["table"] = table[1:].copy()

    with pytest.raises(SpatialDataContractError, match="identifier sets differ"):
        validate_merxen_schema_streaming(sdata)
    assert reads.total == 0


def test_streaming_validation_checks_its_arguments() -> None:
    """Expectations need a group column and must name existing points."""
    sdata, _ = _canonical_store()

    with pytest.raises(ValueError, match="group_column"):
        validate_merxen_schema_streaming(
            sdata, expected_group_rows={"transcripts": {"t0": 10}}
        )
    with pytest.raises(ValueError, match="missing_points"):
        validate_merxen_schema_streaming(
            sdata, expected_partition_rows={"missing_points": [1]}
        )
