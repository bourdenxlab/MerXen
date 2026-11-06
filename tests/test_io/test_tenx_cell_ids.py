"""10x cell ID conversions vendored from spatialdata-io."""

from __future__ import annotations

import numpy as np
import pytest

from merxen.io.tenx_cell_ids import cell_id_str_from_prefix_suffix_uint32, pack_cell_ids


def test_known_cell_id_strings() -> None:
    prefixes = np.array([0, 1, 0xF, 59_979, 0xFFFFFFFF], dtype=np.uint32)
    suffixes = np.array([1, 1, 2, 1, 1], dtype=np.uint32)
    assert cell_id_str_from_prefix_suffix_uint32(prefixes, suffixes).tolist() == [
        "aaaaaaaa-1",
        "aaaaaaab-1",
        "aaaaaaap-2",
        "aaaaokel-1",
        "pppppppp-1",
    ]


def test_matches_spatialdata_io() -> None:
    xenium = pytest.importorskip("spatialdata_io.readers.xenium")
    rng = np.random.default_rng(0)
    prefixes = rng.integers(0, 2**32, 200, dtype=np.uint64).astype(np.uint32)
    suffixes = rng.integers(1, 4, 200).astype(np.uint32)
    expected = xenium.cell_id_str_from_prefix_suffix_uint32(prefixes, suffixes)
    np.testing.assert_array_equal(
        cell_id_str_from_prefix_suffix_uint32(prefixes, suffixes).astype(str),
        np.asarray(expected).astype(str),
    )


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="prefixes"):
        cell_id_str_from_prefix_suffix_uint32([1, 2], [1])


def test_pack_cell_ids() -> None:
    packed = pack_cell_ids(
        np.array([59_979, 7], np.uint32), np.array([1, 2], np.uint32)
    )
    assert packed.dtype == np.uint64
    assert packed.tolist() == [(1 << 32) | 59_979, (2 << 32) | 7]
    with pytest.raises(ValueError, match="differ"):
        pack_cell_ids(np.zeros(2), np.zeros(3))
