"""Seam clipping of the stitched ProSeg polygons (AT-9 follow-up, plan AT-7)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from merxen.segmentation.atera_proseg import AteraProsegError, seams
from merxen.segmentation.atera_proseg.seams import (
    SEAM_OVERLAP_TOLERANCE_UM2,
    clip_seam_overlaps,
    core_box,
    cross_tile_overlaps,
)

TILE = 8.0
# Groups in plan order: tile c0_r0 (0), c1_r0 (1), c2_r0 (2).
CORES = [core_box((0, 0), TILE), core_box((1, 0), TILE), core_box((2, 0), TILE)]


def test_core_box_is_the_half_open_core() -> None:
    assert core_box((1, 2), TILE).bounds == (8.0, 16.0, 16.0, 24.0)


def test_overlap_in_the_later_core_is_cut_from_the_earlier_cell() -> None:
    early = box(5.0, 1.0, 9.5, 4.0)  # tile 0, reaching into core 1
    late = box(8.5, 1.0, 12.0, 4.0)  # tile 1
    other = box(1.0, 5.0, 2.0, 6.0)
    result = clip_seam_overlaps([early, late, other], np.asarray([0, 1, 0]), CORES)
    assert result.geometries[1] is late and result.geometries[2] is other
    clipped = result.geometries[0]
    assert clipped.geom_type == "MultiPolygon"
    assert clipped.equals(box(5.0, 1.0, 8.5, 4.0))
    assert result.emptied == []
    (record,) = result.clipped
    assert record["position"] == 0 and record["against"] == [1]
    assert record["area_before_um2"] == pytest.approx(13.5)
    assert record["clipped_area_um2"] == pytest.approx(3.0)
    assert result.before is not None and result.after is not None
    assert result.before.n_pairs == 1
    assert result.before.max_area_um2 == pytest.approx(3.0)
    assert result.before.n_pairs_over_1um2 == 1
    assert result.after.n_pairs == 0


def test_an_overlap_across_the_core_edge_is_split_by_core() -> None:
    # Overlap [7, 9] x [1, 4]: [7, 8] lies in core 0, [8, 9] in core 1.
    first = box(5.0, 1.0, 9.0, 4.0)
    second = box(7.0, 1.0, 11.0, 4.0)
    result = clip_seam_overlaps([second, first], np.asarray([1, 0]), CORES)
    assert result.geometries[1].equals(box(5.0, 1.0, 8.0, 4.0))
    assert result.geometries[0].equals(box(8.0, 1.0, 11.0, 4.0))
    assert len(result.clipped) == 2


def test_an_overlap_in_neither_core_goes_to_the_earlier_tile() -> None:
    # The overlap lies in core c2_r0, a third tile's: the earlier tile (0)
    # keeps it, as for a transcript claimed by two tiles outside both cores.
    early = box(15.0, 1.0, 18.0, 4.0)
    late = box(17.0, 1.0, 20.0, 4.0)
    result = clip_seam_overlaps(
        [early, late], np.asarray([0, 1]), [CORES[0], core_box((5, 5), TILE)]
    )
    assert result.geometries[0] is early
    assert result.geometries[1].equals(box(18.0, 1.0, 20.0, 4.0))


def test_a_cell_inside_the_winner_is_emptied_and_listed() -> None:
    winner = box(9.0, 1.0, 13.0, 5.0)  # tile 1
    loser = box(10.0, 2.0, 11.0, 3.0)  # tile 0, wholly in core 1
    result = clip_seam_overlaps([loser, winner], np.asarray([0, 1]), CORES)
    assert result.emptied == [0]
    assert result.geometries[0] is None
    assert result.geometries[1] is winner
    assert result.clipped[0]["area_after_um2"] == 0.0
    assert result.clipped[0]["clipped_area_um2"] == pytest.approx(1.0)


def test_cells_of_one_tile_and_touching_cells_are_left_alone() -> None:
    first = box(1.0, 1.0, 4.0, 4.0)
    second = box(3.0, 1.0, 6.0, 4.0)  # same tile, overlapping
    touching = box(6.0, 1.0, 9.0, 4.0)  # other tile, shares an edge only
    result = clip_seam_overlaps([first, second, touching], np.asarray([0, 0, 1]), CORES)
    assert result.clipped == []
    inputs = [first, second, touching]
    assert all(a is b for a, b in zip(result.geometries, inputs, strict=True))


def test_a_cell_losing_to_two_neighbours_loses_both_parts() -> None:
    middle = box(6.0, 1.0, 10.0, 4.0)  # tile 0
    left = MultiPolygon([box(9.0, 1.0, 12.0, 2.0)])  # tile 1
    right = box(9.0, 3.0, 12.0, 6.0)  # tile 1
    result = clip_seam_overlaps([middle, left, right], np.asarray([0, 1, 1]), CORES)
    expected = box(6.0, 1.0, 10.0, 4.0).difference(
        box(9.0, 1.0, 10.0, 2.0).union(box(9.0, 3.0, 10.0, 4.0))
    )
    assert result.geometries[0].equals(expected)
    assert result.clipped[0]["against"] == [1, 2]
    _, _, area = cross_tile_overlaps(result.geometries, np.asarray([0, 1, 1]))
    assert area.size == 0


def test_an_invalid_polygon_is_repaired_before_clipping() -> None:
    bowtie = Polygon([(5, 1), (10, 4), (10, 1), (5, 4)])  # self-intersecting
    assert not bowtie.is_valid
    late = box(8.5, 0.0, 12.0, 5.0)
    result = clip_seam_overlaps([bowtie, late], np.asarray([0, 1]), CORES)
    assert result.geometries[0].is_valid
    assert result.geometries[0].bounds[2] == pytest.approx(8.5)


def test_a_remaining_overlap_above_the_tolerance_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A broken clip (the cell keeps its polygon) must not pass silently.
    monkeypatch.setattr(seams, "_as_multipolygon", lambda geometry: box(5, 1, 9.5, 4))
    with pytest.raises(AteraProsegError, match="still overlap"):
        clip_seam_overlaps(
            [box(5.0, 1.0, 9.5, 4.0), box(8.5, 1.0, 12.0, 4.0)],
            np.asarray([0, 1]),
            CORES,
        )
    assert SEAM_OVERLAP_TOLERANCE_UM2 == 0.01
