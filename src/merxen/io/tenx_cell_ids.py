"""10x cell IDs: ``(prefix, suffix)`` uint32 pairs, strings and packed integers.

10x zarr outputs (Xenium and Atera ``cells.zarr.zip`` and ``transcripts.zarr.zip``)
store a cell ID as two uint32 columns, a cell prefix and a dataset suffix. The
cell-feature matrix names the same cell by a string such as ``aaaaokel-1`` and,
in Atera's ``obs.cell_id``, by the packed int64 ``(suffix << 32) | prefix``.

:func:`cell_id_str_from_prefix_suffix_uint32` is ported from spatialdata-io
(``spatialdata_io/readers/xenium.py``, release 0.6.0), Copyright (c) 2022, scverse®,
BSD 3-Clause License; see ``THIRD_PARTY_NOTICES.md`` and
``licenses/spatialdata-io-BSD-3-Clause.txt``. The port keeps the upstream
algorithm and adds input validation and type hints.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike

UNASSIGNED_CELL_ID_PART: Final = 0xFFFFFFFF
"""Prefix and suffix value of a transcript that no cell was assigned to."""

# Hex digits 0-9a-f become the letters a-p, so a prefix is all lower-case letters.
_HEX_SHIFT: Final = {str(i): chr(ord("a") + i) for i in range(10)} | {
    chr(ord("a") + i): chr(ord("a") + 10 + i) for i in range(6)
}


def cell_id_str_from_prefix_suffix_uint32(
    cell_id_prefix: ArrayLike, dataset_suffix: ArrayLike
) -> np.ndarray:
    """Convert 10x ``(prefix, suffix)`` cell IDs to their string form.

    The prefix is written in hexadecimal, each hex digit is shifted to the
    letters ``a``-``p``, the result is left-padded with ``a`` to eight letters
    and the suffix is appended after a dash (10x documentation, "cell ID").

    Args:
        cell_id_prefix: Cell ID prefixes (uint32 values).
        dataset_suffix: Dataset suffixes, one per prefix.

    Returns:
        Object array of cell ID strings, e.g. ``"aaaaokel-1"``.

    Raises:
        ValueError: If the inputs differ in length or hold negative values.
    """
    prefixes = np.asarray(cell_id_prefix).astype(np.int64, copy=False).ravel()
    suffixes = np.asarray(dataset_suffix).astype(np.int64, copy=False).ravel()
    if prefixes.shape != suffixes.shape:
        raise ValueError(
            f"{prefixes.size} cell ID prefixes but {suffixes.size} dataset suffixes"
        )
    if prefixes.size and (prefixes.min() < 0 or suffixes.min() < 0):
        raise ValueError("cell ID prefixes and suffixes must be non-negative")
    shifted = ["".join(_HEX_SHIFT[c] for c in format(int(p), "x")) for p in prefixes]
    return np.asarray(
        [
            prefix.rjust(8, "a") + f"-{int(suffix)}"
            for prefix, suffix in zip(shifted, suffixes, strict=True)
        ],
        dtype=object,
    )


def pack_cell_ids(cell_id_prefix: ArrayLike, dataset_suffix: ArrayLike) -> np.ndarray:
    """Pack 10x ``(prefix, suffix)`` cell IDs into one integer each.

    The packing is ``(suffix << 32) | prefix``, the int64 that Atera's
    cell-feature matrix stores as ``obs.cell_id``.

    Args:
        cell_id_prefix: Cell ID prefixes (uint32 values).
        dataset_suffix: Dataset suffixes, one per prefix.

    Returns:
        uint64 array of packed IDs.

    Raises:
        ValueError: If the inputs differ in shape.
    """
    prefixes = np.asarray(cell_id_prefix, dtype=np.uint64)
    suffixes = np.asarray(dataset_suffix, dtype=np.uint64)
    if prefixes.shape != suffixes.shape:
        raise ValueError(
            f"cell ID prefixes {prefixes.shape} and suffixes {suffixes.shape} differ"
        )
    return (suffixes << np.uint64(32)) | prefixes
