"""Chunk-independent pciSeq exports and bounded exact-identifier validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.analysis.pciseq import _summarize_spots
from merxen.analysis.pciseq_streaming import stream_spot_assignments


def _inputs(
    tmp_path: Path,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    prepared, computed = tmp_path / "prepared", tmp_path / "computed"
    prepared.mkdir()
    computed.mkdir()
    cell_map = pd.DataFrame(
        {
            "label": np.array([1, 2], dtype=np.uint32),
            "instance_id": np.array([2**64 - 1, 17], dtype=np.uint64),
        }
    )
    spot_map = pd.DataFrame(
        {
            "spot_index": [0, 1, 2, 3],
            "transcript_id": np.array(
                [2**64 - 3, 2**63 + 7, 2**53 + 1, 29], dtype=np.uint64
            ),
        }
    )
    spot_map.iloc[[2, 0, 3, 1]].to_csv(prepared / "spot_id_map.csv", index=False)
    pd.DataFrame({"gene_name": ["A", "NA", "B", "A"]}).to_csv(
        prepared / "spots.csv", index=False
    )
    records = [
        {
            "spot_id": 2,
            "neighbour": 0,
            "gene_name": "B",
            "x": 1.25,
            "neighbour_array": [2, 0],
            "neighbour_prob": [0.7, 0.299],
            "diagnostic": {"key": "value"},
            "optional": 1,
        },
        {
            "spot_id": 0,
            "neighbour": 1,
            "gene_name": "A",
            "x": 2.5,
            "neighbour_array": [1, 2],
            "neighbour_prob": [0.999, 0.001],
            "diagnostic": None,
            "optional": None,
        },
        {
            "spot_id": 3,
            "neighbour": 2,
            "gene_name": "A",
            "x": 3.75,
            "neighbour_array": [2],
            "neighbour_prob": [0.999],
            "diagnostic": ["unicode λ", None],
            "late_column": "late",
        },
        {
            "spot_id": 1,
            "neighbour": 1,
            "gene_name": "NA",
            "x": 4.125,
            "neighbour_array": [],
            "neighbour_prob": [],
            "diagnostic": "scalar",
            "optional": 4,
        },
    ]
    (computed / "gene_data.json").write_text(json.dumps(records))
    return prepared, computed, cell_map, spot_map, records


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 20_000])
def test_streamed_csv_matches_legacy_bytes_including_diagnostics(
    tmp_path: Path, chunk_rows: int
) -> None:
    prepared, computed, cells, spots, records = _inputs(tmp_path)
    expected = _summarize_spots(pd.DataFrame(records), spots, cells)
    target = tmp_path / "exports" / "spot_assignments.csv"
    stream_spot_assignments(prepared, computed, target, cells, 4, chunk_rows=chunk_rows)
    assert target.read_text() == expected.to_csv(index=False)
    result = pd.read_csv(
        target,
        dtype={"transcript_id": "uint64", "instance_id": "UInt64"},
        keep_default_na=False,
        na_values={"instance_id": [""]},
    )
    assert int(result.loc[1, "instance_id"]) == 2**64 - 1
    assert int(result.loc[1, "transcript_id"]) == 2**64 - 3
    assert result.gene_name.tolist() == ["B", "A", "A", "NA"]
    assert not list(target.parent.glob(".pciseq-spots-*"))


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "unknown",
        "missing",
        "wrong_gene",
        "unknown_cell",
        "negative_id",
        "fractional_id",
    ],
)
def test_streamed_records_reject_invalid_ids_across_chunks_without_replacing_output(
    tmp_path: Path, fault: str
) -> None:
    prepared, computed, cells, _, records = _inputs(tmp_path)
    if fault == "duplicate":
        records[-1]["spot_id"] = records[0]["spot_id"]
    elif fault == "unknown":
        records[-1]["spot_id"] = 9
    elif fault == "missing":
        records.pop()
    elif fault == "wrong_gene":
        records[-1]["gene_name"] = "B"
    elif fault == "unknown_cell":
        records[-1]["neighbour"] = 3
    elif fault == "negative_id":
        records[-1]["spot_id"] = -1
    else:
        records[-1]["spot_id"] = 1.5
    (computed / "gene_data.json").write_text(json.dumps(records))
    target = tmp_path / "spot_assignments.csv"
    target.write_text("previous output\n")
    with pytest.raises(ValueError):
        stream_spot_assignments(prepared, computed, target, cells, 4, chunk_rows=1)
    assert target.read_text() == "previous output\n"
    assert not list(tmp_path.glob(".pciseq-spots-*"))


@pytest.mark.parametrize(
    "fault",
    ["duplicate_transcript", "duplicate_index", "missing_index", "negative_transcript"],
)
def test_streamed_prepared_maps_validate_global_coverage_and_uniqueness(
    tmp_path: Path, fault: str
) -> None:
    prepared, computed, cells, spots, _ = _inputs(tmp_path)
    if fault == "duplicate_transcript":
        spots.loc[3, "transcript_id"] = spots.loc[0, "transcript_id"]
    elif fault == "duplicate_index":
        spots.loc[3, "spot_index"] = 0
    elif fault == "missing_index":
        spots = spots.iloc[:-1]
    else:
        spots["transcript_id"] = spots.transcript_id.astype(str)
        spots.loc[3, "transcript_id"] = "-1"
    spots.to_csv(prepared / "spot_id_map.csv", index=False)
    with pytest.raises(ValueError):
        stream_spot_assignments(
            prepared, computed, tmp_path / "out.csv", cells, 4, chunk_rows=1
        )


def test_schema_does_not_round_large_nullable_diagnostic_values(tmp_path: Path) -> None:
    prepared, computed, cells, spots, records = _inputs(tmp_path)
    for record, value in zip(records, [2**64 - 1, None, "text", 1], strict=True):
        record["mixed_diagnostic"] = value
    (computed / "gene_data.json").write_text(json.dumps(records))
    target = tmp_path / "out.csv"
    stream_spot_assignments(prepared, computed, target, cells, 4, chunk_rows=2)
    assert target.read_text() == _summarize_spots(
        pd.DataFrame(records), spots, cells
    ).to_csv(index=False)
