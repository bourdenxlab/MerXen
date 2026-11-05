"""The Z handling of ``scripts/manual_registration/compare_czi_readers.py``.

The script needs ``pylibCZIrw`` (LGPL, never a MerXen dependency) and ``czifile``,
so it is not run here. Its helper that reads a region as the maximum over Z takes
the reader as an argument; a stand-in with Z-dependent planes shows which planes it
reads and that it takes their maximum, which is what the converter writes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._scripts import load_script

compare_script = load_script("compare_czi_readers")


class _FakeDocument:
    """A pylibCZIrw-like reader whose plane ``z`` is ``10 * z`` plus one hot pixel."""

    def __init__(self) -> None:
        self.reads: list[dict[str, int]] = []

    def read(self, **kwargs: Any) -> np.ndarray:
        plane = kwargs["plane"]
        self.reads.append(dict(plane))
        size = compare_script.ROI_SIZE
        data = np.full((size, size, 1), 10 * plane["Z"], dtype=np.uint16)
        data[plane["Z"], 0, 0] = 1000 + plane["Z"]  # a different pixel per plane
        return data


def test_roi_reads_one_plane_by_default() -> None:
    document = _FakeDocument()

    roi = compare_script._pylibczi_roi(document, 5, 6, 2)

    assert document.reads == [{"C": 2, "Z": 0, "T": 0}]
    assert roi.shape == (compare_script.ROI_SIZE,) * 2
    assert roi[0, 0] == 1000 and roi[1, 1] == 0


def test_max_over_z_reads_every_plane_of_the_range_and_takes_the_maximum() -> None:
    document = _FakeDocument()

    roi = compare_script._pylibczi_roi_max_over_z(document, 5, 6, 1, (2, 5))

    assert [read["Z"] for read in document.reads] == [2, 3, 4]
    assert all(read["C"] == 1 and read["T"] == 0 for read in document.reads)
    # Every plane's hot pixel survives; elsewhere the highest plane (Z=4) wins.
    assert roi[2, 0] == 1002 and roi[3, 0] == 1003 and roi[4, 0] == 1004
    assert roi[10, 10] == 40
    assert roi[0, 0] == 40  # plane 2's hot pixel is at row 2, not row 0
