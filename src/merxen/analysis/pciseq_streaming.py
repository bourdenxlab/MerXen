"""Bounded-memory validation and export of exact-ID pciSeq spot records."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from merxen.pciseq_records import iter_json_records

CHUNK_ROWS = 20_000


def _frames(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    records = iter_json_records(path)
    while chunk := list(islice(records, chunk_rows)):
        # Defer inference until the global schema is known: a nullable integer
        # diagnostic must not lose precision in an early, locally float chunk.
        columns = dict.fromkeys(name for record in chunk for name in record)
        yield pd.DataFrame(
            {
                name: pd.Series(
                    [record.get(name, np.nan) for record in chunk], dtype=object
                )
                for name in columns
            }
        )


def integer_ids(
    values: Any, field_name: str, *, allow_zero: bool = False
) -> np.ndarray:
    """Parse exact uint64 identifiers, rejecting lossy floating-point IDs."""
    series = pd.Series(values, copy=False)
    if series.isna().any() or is_bool_dtype(series.dtype):
        raise ValueError(f"{field_name} contains invalid identifiers")
    if is_integer_dtype(series.dtype):
        if (series < 0).any():
            raise ValueError(f"{field_name} contains negative identifiers")
        result = np.asarray(series, dtype=np.uint64)
    elif is_float_dtype(series.dtype):
        numbers = series.to_numpy(dtype=float)
        if (
            not np.isfinite(numbers).all()
            or (numbers < 0).any()
            or (numbers > 2**53).any()
            or (numbers != np.floor(numbers)).any()
        ):
            raise ValueError(f"{field_name} contains unsafe identifiers")
        result = numbers.astype(np.uint64)
    else:
        strings = series.astype(str)
        if not strings.str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"{field_name} contains noninteger identifiers")
        try:
            result = np.fromiter(
                (int(value) for value in strings), dtype=np.uint64, count=len(series)
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{field_name} exceeds uint64 identifiers") from error
    if not allow_zero and (result == 0).any():
        raise ValueError(f"{field_name} requires positive identifiers")
    return result


def mapped_array(directory: Path, name: str, dtype: str, count: int) -> np.memmap:
    """Create a temporary contiguous disk-backed array."""
    return np.memmap(directory / name, mode="w+", dtype=dtype, shape=(count,))


def assert_unique_sorted(values: np.ndarray, field_name: str, chunk_rows: int) -> None:
    """Validate a sorted array in bounded slices, including chunk boundaries."""
    previous = None
    for start in range(0, len(values), chunk_rows):
        part = values[start : start + chunk_rows]
        if (previous is not None and part[0] == previous) or np.any(
            part[1:] == part[:-1]
        ):
            raise ValueError(f"{field_name} contains duplicate identifiers")
        previous = part[-1]


@dataclass
class _ColumnType:
    kinds: set[str] = field(default_factory=set)
    present: int = 0
    missing: bool = False
    minimum: int = 0
    maximum: int = 0

    def update(self, values: pd.Series) -> None:
        self.present += len(values)
        self.missing |= bool(values.isna().any())
        if not values.notna().any():
            return
        inferred = pd.api.types.infer_dtype(values, skipna=True)
        if inferred == "boolean":
            self.kinds.add("bool")
        elif inferred == "integer":
            self.kinds.add("int")
            self.minimum = min(self.minimum, int(values.min()))
            self.maximum = max(self.maximum, int(values.max()))
        elif inferred in {"floating", "mixed-integer-float", "decimal"}:
            self.kinds.add("float")
        else:
            self.kinds.add("object")

    def dtype(self, count: int) -> str:
        if (
            not self.kinds
            or "object" in self.kinds
            or ("bool" in self.kinds and len(self.kinds) > 1)
        ):
            return "object"
        missing = self.missing or self.present != count
        if "bool" in self.kinds:
            return "object" if missing else "bool"
        if "float" in self.kinds or missing:
            return "float64"
        if self.maximum > np.iinfo(np.int64).max:
            return "object" if self.minimum < 0 else "uint64"
        return "int64"


def _record_schema(path: Path, chunk_rows: int) -> tuple[dict[str, str], int]:
    columns: dict[str, _ColumnType] = {}
    count = 0
    for frame in _frames(path, chunk_rows):
        count += len(frame)
        for name in frame:
            columns.setdefault(name, _ColumnType()).update(frame[name])
    return {name: column.dtype(count) for name, column in columns.items()}, count


def _prepared_index(
    prepared_dir: Path, directory: Path, count: int, chunk_rows: int
) -> tuple[np.memmap, np.memmap, dict[str, int]]:
    transcript_ids = mapped_array(directory, "transcript_ids", "uint64", count)
    gene_codes = mapped_array(directory, "gene_codes", "uint32", count)
    seen = mapped_array(directory, "map_seen", "bool", count)
    for frame in pd.read_csv(
        prepared_dir / "spot_id_map.csv",
        dtype={"spot_index": "string", "transcript_id": "string"},
        chunksize=chunk_rows,
    ):
        indices = integer_ids(
            frame.spot_index, "Prepared spot map indices", allow_zero=True
        )
        if (
            (indices >= count).any()
            or len(np.unique(indices)) != len(indices)
            or seen[indices].any()
        ):
            raise ValueError("Prepared spot map indices must be sequential identifiers")
        transcript_ids[indices] = integer_ids(
            frame.transcript_id, "spot map transcript_id"
        )
        seen[indices] = True
    if not seen.all():
        raise ValueError("Prepared spot map indices must be sequential identifiers")
    ordered = mapped_array(directory, "ordered_transcript_ids", "uint64", count)
    for start in range(0, count, chunk_rows):
        ordered[start : start + chunk_rows] = transcript_ids[start : start + chunk_rows]
    ordered.sort(kind="quicksort")
    assert_unique_sorted(ordered, "spot map transcript_id", chunk_rows)
    del ordered, seen
    (directory / "ordered_transcript_ids").unlink()
    (directory / "map_seen").unlink()
    genes: dict[str, int] = {}
    position = 0
    for frame in pd.read_csv(
        prepared_dir / "spots.csv",
        usecols=["gene_name"],
        dtype={"gene_name": str},
        keep_default_na=False,
        chunksize=chunk_rows,
    ):
        if position + len(frame) > count:
            raise ValueError("Prepared transcript gene mapping has a different size")
        for name in frame.gene_name.unique():
            genes.setdefault(name, len(genes))
        gene_codes[position : position + len(frame)] = frame.gene_name.map(
            genes
        ).to_numpy(dtype=np.uint32)
        position += len(frame)
    if position != count:
        raise ValueError("Prepared transcript gene mapping has a different size")
    return transcript_ids, gene_codes, genes


def stream_spot_assignments(
    prepared_dir: Path,
    computed_dir: Path,
    output_path: Path,
    cell_map: pd.DataFrame,
    spot_count: int,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> None:
    """Validate all modeled records and write the legacy CSV without full frames.

    Dense model spot IDs address temporary memory-mapped arrays. Every raw
    diagnostic field is retained, and a schema pass preserves full-frame CSV
    numeric promotion and column ordering independently of the chunk size.
    """
    import json

    if (
        isinstance(spot_count, bool)
        or not isinstance(spot_count, int)
        or spot_count <= 0
    ):
        raise ValueError("Prepared transcript count must be a positive integer")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    schema, count = _record_schema(computed_dir / "gene_data.json", chunk_rows)
    if not {"spot_id", "neighbour", "gene_name"}.issubset(schema):
        raise ValueError("pciSeq spot output lacks required columns")
    if count != spot_count:
        raise ValueError("pciSeq output changed the prepared transcript identifiers")
    reserved = {"instance_id", "spot_index", "transcript_id"}
    if reserved.intersection(schema):
        raise ValueError(
            "pciSeq spot output collides with canonical identifier columns"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".pciseq-spots-", dir=output_path.parent
    ) as temporary:
        directory = Path(temporary)
        transcript_ids, expected_genes, gene_lookup = _prepared_index(
            prepared_dir, directory, spot_count, chunk_rows
        )
        seen = mapped_array(directory, "reported_seen", "bool", spot_count)
        labels = np.zeros(len(cell_map) + 1, dtype=np.uint64)
        labels[cell_map.label.to_numpy(dtype=np.uint32)] = (
            cell_map.instance_id.to_numpy(dtype=np.uint64)
        )
        target = directory / "spot_assignments.csv"
        first = True
        for frame in _frames(computed_dir / "gene_data.json", chunk_rows):
            indices = integer_ids(
                frame.spot_id, "pciSeq spot identifiers", allow_zero=True
            )
            if (
                (indices >= spot_count).any()
                or len(np.unique(indices)) != len(indices)
                or seen[indices].any()
            ):
                raise ValueError(
                    "pciSeq output changed the prepared transcript identifiers"
                )
            neighbors = integer_ids(
                frame.neighbour, "pciSeq assigned cell", allow_zero=True
            )
            if (neighbors >= len(labels)).any():
                raise ValueError("pciSeq returned an unknown assigned cell")
            codes = frame.gene_name.map(gene_lookup)
            if codes.isna().any() or not np.array_equal(
                codes.to_numpy(dtype=np.uint32), expected_genes[indices]
            ):
                raise ValueError(
                    "pciSeq output changed the prepared transcript gene mapping"
                )
            result = frame.reindex(columns=list(schema)).astype(schema)
            result["instance_id"] = pd.arrays.IntegerArray(
                labels[neighbors], neighbors == 0
            )
            result["spot_index"] = indices.astype(np.int64)
            result["transcript_id"] = transcript_ids[indices]
            for column in schema:
                if result[column].dtype == object:
                    result[column] = pd.Series(
                        [
                            json.dumps(value)
                            if isinstance(value, (list, dict))
                            else value
                            for value in result[column]
                        ],
                        index=result.index,
                        dtype=object,
                    )
            result.to_csv(target, mode="w" if first else "a", header=first, index=False)
            first = False
            seen[indices] = True
        if not seen.all():
            raise ValueError(
                "pciSeq output changed the prepared transcript identifiers"
            )
        os.replace(target, output_path)
