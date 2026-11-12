"""Tests for the partition-by-partition deep schema validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import PointsModel

from merxen.io import spatialdata_schema
from merxen.io.spatialdata_io import prepare_source_spatialdata_contract
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    ORIGINAL_ID_NAMESPACE,
    SpatialDataContractError,
    register_segmentation_branch,
    validate_merxen_schema,
    validate_merxen_schema_streaming,
)

from .contract_synth import (
    N_PARTITIONS,
    ROWS_PER_PARTITION,
    PartitionReads,
    canonical_spatialdata_like,
    merscope_raw_spatialdata,
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


@pytest.mark.parametrize(
    ("platform", "builder"),
    [("XENIUM", xenium_raw_spatialdata), ("MERSCOPE", merscope_raw_spatialdata)],
    ids=["xenium", "merscope"],
)
def test_streaming_validation_passes_an_in_memory_default_contract(
    platform: str,
    builder: Callable[[], tuple[sd.SpatialData, PartitionReads]],
) -> None:
    """Partitions are read with their own numbers, so counted IDs stay unique.

    ``.partitions[k]`` would renumber the selection from 0 and restart the
    partition-order IDs of ``with_stable_transcript_ids`` at 1 in every
    partition.
    """
    sdata, reads = builder()
    prepare_source_spatialdata_contract(sdata, platform=platform)
    reads.reset()

    summary = validate_merxen_schema_streaming(sdata)["transcripts"]

    assert (summary.transcript_id_min, summary.transcript_id_max) == (1, N_ROWS)
    assert summary.reread_partitions == 0
    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)


def test_streaming_validation_rejects_a_duplicate_at_a_partition_boundary() -> None:
    """A repeat equal to the previous partition's largest ID is found."""
    ids = np.arange(1, N_ROWS + 1, dtype=np.uint64)
    ids[ROWS_PER_PARTITION] = ids[ROWS_PER_PARTITION - 1]
    sdata, _ = _canonical_store(transcript_id=ids)

    with pytest.raises(SpatialDataContractError, match=r"partitions \[0, 1\]"):
        validate_merxen_schema_streaming(sdata)


def _shuffled_ids() -> np.ndarray:
    """Return unique IDs whose ranges overlap in every partition."""
    return (
        np.random.default_rng(3).permutation(np.arange(1, N_ROWS + 1)).astype(np.uint64)
    )


def test_streaming_validation_buckets_rereads_within_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large overlap cluster is checked in ID buckets of bounded size."""
    sdata, reads = _canonical_store(transcript_id=_shuffled_ids())
    checked: list[int] = []
    unique_ids = spatialdata_schema._unique_transcript_ids

    def _record(ids: np.ndarray, *, label: str) -> np.ndarray:
        if "bucket" in label:
            checked.append(len(ids))
        return unique_ids(ids, label=label)

    monkeypatch.setattr(spatialdata_schema, "_unique_transcript_ids", _record)
    budget = ROWS_PER_PARTITION
    n_buckets = N_ROWS // budget

    summary = validate_merxen_schema_streaming(sdata, max_reread_rows=budget)[
        "transcripts"
    ]

    assert len(checked) == n_buckets
    assert sum(checked) == N_ROWS
    assert max(checked) <= 2 * budget
    assert summary.reread_partitions == n_buckets * N_PARTITIONS
    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1 + n_buckets)


def test_streaming_validation_finds_a_duplicate_in_a_bucket() -> None:
    """A repeated ID always lands in one bucket."""
    ids = _shuffled_ids()
    ids[-1] = ids[0]
    sdata, _ = _canonical_store(transcript_id=ids)

    with pytest.raises(SpatialDataContractError, match="bucket .* duplicate"):
        validate_merxen_schema_streaming(sdata, max_reread_rows=ROWS_PER_PARTITION)


def test_streaming_validation_refuses_a_bucket_over_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bucket that outgrows twice the budget raises instead of growing."""
    sdata, _ = _canonical_store(transcript_id=_shuffled_ids())
    monkeypatch.setattr(
        spatialdata_schema,
        "_id_buckets",
        lambda ids, n_buckets: np.zeros(len(ids), dtype=np.uint64),
    )

    with pytest.raises(ValueError, match="raise max_reread_rows"):
        validate_merxen_schema_streaming(sdata, max_reread_rows=ROWS_PER_PARTITION)
    with pytest.raises(ValueError, match="positive"):
        validate_merxen_schema_streaming(sdata, max_reread_rows=0)


def test_streaming_validation_checks_primary_points_without_a_registry() -> None:
    """The primary points are checked even when no branch is registered."""
    ids = np.arange(1, N_ROWS + 1, dtype=np.uint64)
    ids[3] = ids[2]
    sdata, _ = _canonical_store(transcript_id=ids)
    sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentations"] = {}

    with pytest.raises(SpatialDataContractError, match="partition 0: duplicate"):
        validate_merxen_schema_streaming(sdata)


def test_streaming_validation_rejects_a_different_partition_split() -> None:
    """Partition counts are compared one by one, not only in total."""
    sdata, _ = _canonical_store()
    expected = [ROWS_PER_PARTITION + 1, ROWS_PER_PARTITION - 1]
    expected += [ROWS_PER_PARTITION] * (N_PARTITIONS - 2)

    with pytest.raises(SpatialDataContractError, match="partition row counts"):
        validate_merxen_schema_streaming(
            sdata, expected_partition_rows={"transcripts": expected}
        )


def test_streaming_validation_rejects_an_unexpected_group() -> None:
    """A group missing from the expectations is expected to have no rows."""
    sdata, _ = _canonical_store()

    with pytest.raises(SpatialDataContractError, match=r"'t1': \(14, 0\)"):
        validate_merxen_schema_streaming(
            sdata,
            group_column="source_tile",
            expected_group_rows={"transcripts": {"t0": 10}},
        )


def test_streaming_validation_rejects_rows_without_a_group() -> None:
    """Rows with a null group value cannot be reconciled."""
    tiles = pd.Categorical([*TILES[:-3], None, None, None], categories=["t0", "t1"])
    sdata, _ = _canonical_store(source_tile=tiles)

    summary = validate_merxen_schema_streaming(sdata, group_column="source_tile")
    with pytest.raises(SpatialDataContractError, match="3 rows have no source_tile"):
        validate_merxen_schema_streaming(
            sdata,
            group_column="source_tile",
            expected_group_rows={"transcripts": {"t0": 10, "t1": 11}},
        )

    assert summary["transcripts"].null_group_rows == 3
    assert dict(summary["transcripts"].group_rows) == {"t0": 10, "t1": 11}


def test_streaming_validation_counts_groups_without_expectations() -> None:
    """An existing group column is counted even with nothing to reconcile."""
    sdata, _ = _canonical_store()

    summary = validate_merxen_schema_streaming(sdata, group_column="source_tile")

    assert dict(summary["transcripts"].group_rows) == {"t0": 10, "t1": 14}
    assert summary["transcripts"].null_group_rows == 0


def test_streaming_validation_rejects_a_missing_group_column() -> None:
    """Expectations on a column the points lack fail."""
    sdata, _ = _canonical_store()

    with pytest.raises(SpatialDataContractError, match="does not exist"):
        validate_merxen_schema_streaming(
            sdata,
            group_column="fov",
            expected_group_rows={"transcripts": {"0": N_ROWS}},
        )


def test_streaming_validation_rejects_points_without_transcript_ids() -> None:
    """A points element named in the expectations needs transcript IDs."""
    sdata, _ = _canonical_store()
    sdata.points["extra"] = PointsModel.parse(pd.DataFrame({"x": [1.0], "y": [2.0]}))

    with pytest.raises(SpatialDataContractError, match="lacks 'transcript_id'"):
        validate_merxen_schema_streaming(sdata, expected_partition_rows={"extra": [1]})


def test_streaming_validation_compares_coordinate_variant_matrices() -> None:
    """A coordinate variant must keep its parent's count matrix."""
    sdata, _ = _canonical_store()
    sdata.points["transcripts_aligned"] = sdata.points["transcripts"]
    sdata.shapes["cell_boundaries_aligned"] = sdata.shapes["cell_boundaries"]
    table = sdata.tables["table"].copy()
    table.X = table.X * 2
    sdata.tables["table_aligned"] = table
    register_segmentation_branch(
        sdata,
        "original_aligned",
        points_key="transcripts_aligned",
        assignment_column="original_assignment",
        shape_key="cell_boundaries_aligned",
        table_key="table_aligned",
        id_namespace=ORIGINAL_ID_NAMESPACE,
        coordinate_variant_of="original",
    )

    with pytest.raises(SpatialDataContractError, match="count matrix differs"):
        validate_merxen_schema_streaming(sdata)
