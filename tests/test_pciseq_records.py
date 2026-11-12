"""Streaming preserves the existing pciSeq JSON contract, including uint64 IDs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.pciseq_records import (
    iter_json_frames,
    iter_json_records,
    write_json_frames,
)


@pytest.mark.parametrize("read_size", [1, 3, 31, 1 << 20])
def test_record_boundaries_and_exact_integer_ids(
    tmp_path: Path, read_size: int
) -> None:
    records = [
        {"id": 2**64 - 1, "gene": 'a}b,["\\\u03bb', "nested": [[0.123456789012345]]},
        {"id": 2**63 + 13, "gene": "b", "nested": [], "missing": None},
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records, ensure_ascii=False))
    assert list(iter_json_records(path, read_size=read_size)) == records


@pytest.mark.parametrize("text", ["", "{}", "[{}", "[{},]", "[{}]x", "[1]", "[{} {}]"])
def test_invalid_or_incomplete_exports_are_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_json_records(path, read_size=2))


@pytest.mark.parametrize("chunk_rows", [1, 2, 20])
def test_frame_writer_matches_existing_export(tmp_path: Path, chunk_rows: int) -> None:
    frame = pd.DataFrame(
        {
            "id": np.asarray([2**64 - 1, 2**63 + 13, 0], dtype=np.uint64),
            "gene": ["a", "b", "\u03bb"],
            "prob": [0.123456789012345, 0.001, np.nan],
            "neighbour_array": [[1, 2], [0], []],
        }
    )
    path = tmp_path / "records.json"
    count = write_json_frames(
        path,
        (
            frame.iloc[start : start + chunk_rows]
            for start in range(0, len(frame), chunk_rows)
        ),
    )
    assert count == len(frame)
    assert path.read_text() == frame.to_json(orient="records", double_precision=15)
    frames = list(iter_json_frames(path, chunk_rows))
    assert sum(map(len, frames)) == len(frame)
    assert [int(value) for part in frames for value in part.id] == frame.id.tolist()


def test_empty_export(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    assert write_json_frames(path, []) == 0
    assert list(iter_json_records(path, read_size=1)) == []
