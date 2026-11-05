"""Read and write pciSeq's JSON record arrays without loading the whole file."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd


def iter_json_records(
    path: Path, *, read_size: int = 1 << 20
) -> Iterator[dict[str, Any]]:
    """Yield a JSON array's objects, preserving integer IDs and nested values.

    Memory is bounded by the read buffer and the largest individual record.
    Truncated arrays, trailing commas and extra trailing data are rejected.
    """
    if read_size < 1:
        raise ValueError("read_size must be positive")
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer = ""
        position = 0
        ended = False

        def more() -> bool:
            nonlocal buffer, position, ended
            buffer = buffer[position:]
            position = 0
            part = stream.read(read_size)
            buffer += part
            ended = not part
            return bool(part)

        def character() -> str:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if ended or not more():
                    return ""

        if character() != "[":
            raise ValueError(f"Expected a JSON record array: {path}")
        position += 1
        empty_allowed = True
        while True:
            next_character = character()
            if next_character == "]" and empty_allowed:
                position += 1
                break
            if next_character != "{":
                raise ValueError(f"Expected a JSON record object: {path}")
            while True:
                try:
                    value, stop = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if ended or not more():
                        raise ValueError(
                            f"Invalid or truncated JSON: {path}"
                        ) from error
            position = stop
            yield value
            separator = character()
            if separator == "]":
                position += 1
                break
            if separator != ",":
                raise ValueError(f"Expected a record separator: {path}")
            position += 1
            empty_allowed = False
        if character():
            raise ValueError(f"Unexpected data after JSON array: {path}")


def iter_json_frames(path: Path, chunk_rows: int = 20_000) -> Iterator[pd.DataFrame]:
    """Yield bounded DataFrames from pciSeq's standard JSON record array."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    records: list[dict[str, Any]] = []
    for record in iter_json_records(path):
        records.append(record)
        if len(records) == chunk_rows:
            yield pd.DataFrame.from_records(records)
            records = []
    if records:
        yield pd.DataFrame.from_records(records)


def write_json_frames(path: Path, frames: Iterable[pd.DataFrame]) -> int:
    """Write the same record format and precision as DataFrame.to_json.

    Returns the number of rows written. Frames retain their column and row order;
    no values are rounded beyond the existing pciSeq export convention.
    """
    count = 0
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write("[")
        for frame in frames:
            if frame.empty:
                continue
            if count:
                stream.write(",")
            stream.write(frame.to_json(orient="records", double_precision=15)[1:-1])
            count += len(frame)
        stream.write("]")
    return count
