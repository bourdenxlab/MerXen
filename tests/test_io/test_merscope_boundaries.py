"""Tests for canonical MERSCOPE boundary input."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from spatialdata.transformations import Identity, get_transformation

from merxen.io.builders.merscope import (
    MerscopeKeys,
    _add_merscope_cell_polygons,
    _get_polygons,
)


def _write_boundaries(
    path: Path,
    rows: list[tuple[int, int, MultiPolygon]],
) -> Path:
    """Write ``(EntityID, ZIndex, geometry)`` rows as a vendor-style parquet."""
    boundaries_path = path / MerscopeKeys.BOUNDARIES_FILE
    gpd.GeoDataFrame(
        {
            "ID": np.arange(len(rows)),
            "EntityID": [entity for entity, _, _ in rows],
            "ZIndex": [z_index for _, z_index, _ in rows],
            "ZLevel": [1.5 * (z_index + 1) for _, z_index, _ in rows],
            "Geometry": [geometry for _, _, geometry in rows],
        },
        geometry="Geometry",
    ).to_parquet(boundaries_path)
    return boundaries_path


def _square(x: float, y: float, size: float) -> MultiPolygon:
    return MultiPolygon([box(x, y, x + size, y + size)])


def test_merscope_polygons_repair_self_intersections(tmp_path: Path) -> None:
    """Canonical boundaries should repair recoverable invalid polygons."""
    boundaries_path = tmp_path / "cell_boundaries.parquet"
    bowtie = Polygon([(1, 1), (6, 6), (1, 6), (6, 1), (1, 1)])
    gpd.GeoDataFrame(
        {
            "EntityID": [101, 102],
            "ZIndex": [0, 0],
            "boundary": [
                MultiPolygon([bowtie]),
                MultiPolygon([Polygon([(20, 20), (28, 20), (28, 28), (20, 28)])]),
            ],
        },
        geometry="boundary",
    ).to_parquet(boundaries_path)

    shapes = _get_polygons(boundaries_path, {"global": Identity()})

    assert list(shapes.index) == ["101", "102"]
    assert shapes.geometry.name == "geometry"
    assert bool(shapes.geometry.is_valid.all())


def test_merscope_polygons_keep_cells_absent_from_z0(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cells segmented only on upper planes get the union of their planes."""
    boundaries_path = _write_boundaries(
        tmp_path,
        [
            (1, 0, _square(0, 0, 4)),
            (2, 2, _square(10, 0, 4)),
            (2, 3, _square(11, 0, 4)),
            (3, 5, _square(20, 0, 3)),
        ],
    )

    with caplog.at_level(logging.INFO, logger="merxen.io.builders.merscope"):
        shapes = _get_polygons(boundaries_path, {"global": Identity()})

    assert list(shapes.index) == ["1", "2", "3"]
    assert set(shapes.geometry.geom_type) == {"MultiPolygon"}
    assert shapes.loc["1", "geometry"].equals(_square(0, 0, 4))
    assert shapes.loc["2", "geometry"].equals(MultiPolygon([box(10, 0, 15, 4)]))
    assert shapes.loc["3", "geometry"].equals(_square(20, 0, 3))
    # Non-geometry columns come from each cell's lowest plane.
    assert shapes["ZIndex"].tolist() == [0, 2, 5]
    assert (
        "2 of 3 cells have no ZIndex 0 polygon and 1 cells use the xy union"
        in caplog.text
    )


def test_merscope_polygons_union_keeps_disjoint_parts(tmp_path: Path) -> None:
    """Disjoint cross-sections and multi-part planes stay one MultiPolygon."""
    two_part_plane = MultiPolygon([box(0, 0, 2, 2), box(5, 0, 7, 2)])
    boundaries_path = _write_boundaries(
        tmp_path,
        [
            (7, 1, two_part_plane),
            (7, 2, _square(0, 10, 2)),
        ],
    )

    shapes = _get_polygons(boundaries_path, {"global": Identity()})

    geometry = shapes.loc["7", "geometry"]
    assert isinstance(geometry, MultiPolygon)
    assert len(geometry.geoms) == 3
    assert geometry.area == pytest.approx(12.0)


def test_merscope_polygons_skip_empty_planes(tmp_path: Path) -> None:
    """An empty plane polygon must not drop a cell that has other planes."""
    boundaries_path = _write_boundaries(
        tmp_path,
        [
            (4, 0, MultiPolygon()),
            (4, 1, _square(0, 0, 3)),
        ],
    )

    shapes = _get_polygons(boundaries_path, {"global": Identity()})

    assert list(shapes.index) == ["4"]
    assert shapes.loc["4", "geometry"].equals(_square(0, 0, 3))
    assert shapes.loc["4", "ZIndex"] == 1


def test_merscope_polygons_unchanged_when_planes_repeat_z0(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exports repeating one polygon on every plane keep their ZIndex 0 rows."""
    footprints = {
        30: MultiPolygon([box(0, 0, 3, 3), box(4, 0, 5, 1)]),
        10: _square(10, 0, 2),
        20: _square(20, 0, 5),
    }
    # Plane-major row order, as newer exports write it.
    rows = [
        (entity, z_index, footprint)
        for z_index in range(3)
        for entity, footprint in footprints.items()
    ]
    boundaries_path = _write_boundaries(tmp_path, rows)
    (tmp_path / "z0").mkdir()
    z0_only_path = _write_boundaries(
        tmp_path / "z0",
        [row for row in rows if row[1] == 0],
    )

    with caplog.at_level(logging.INFO, logger="merxen.io.builders.merscope"):
        shapes = _get_polygons(boundaries_path, {"global": Identity()})
    expected = _get_polygons(z0_only_path, {"global": Identity()})

    assert "vary across z planes" not in caplog.text
    assert list(shapes.index) == ["30", "10", "20"]
    assert shapes.index.equals(expected.index)
    assert list(shapes.columns) == list(expected.columns)
    non_geometry = [column for column in shapes.columns if column != "geometry"]
    assert shapes[non_geometry].equals(expected[non_geometry])
    assert np.array_equal(
        shapely.to_wkb(shapes.geometry.to_numpy()),
        shapely.to_wkb(expected.geometry.to_numpy()),
    )


def test_add_merscope_cell_polygons_uses_reader_key_and_transform(
    tmp_path: Path,
) -> None:
    """The spatialdata_io path gets all-plane polygons under its own key."""
    raw_dir = tmp_path / "slide_a" / "region_r1"
    images_dir = raw_dir / MerscopeKeys.IMAGES_DIR
    images_dir.mkdir(parents=True)
    matrix = np.array([[9.0, 0.0, 5.0], [0.0, 9.0, 7.0], [0.0, 0.0, 1.0]])
    np.savetxt(images_dir / MerscopeKeys.TRANSFORMATION_FILE, matrix)
    _write_boundaries(
        raw_dir,
        [
            (1, 0, _square(0, 0, 4)),
            (2, 4, _square(10, 0, 4)),
        ],
    )
    sdata = SimpleNamespace(shapes={})

    _add_merscope_cell_polygons(
        sdata,  # type: ignore[arg-type]
        path=raw_dir,
        region_name=None,
        slide_name=None,
    )

    shapes = sdata.shapes["slide_a_region_r1_polygons"]
    assert list(shapes.index) == ["1", "2"]
    transform = get_transformation(shapes, to_coordinate_system="global")
    assert np.allclose(
        transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        matrix,
    )


def test_add_merscope_cell_polygons_skips_missing_boundaries(tmp_path: Path) -> None:
    """Without a boundary parquet the reader result is left untouched."""
    sdata = SimpleNamespace(shapes={})

    _add_merscope_cell_polygons(
        sdata,  # type: ignore[arg-type]
        path=tmp_path,
        region_name="r",
        slide_name="s",
    )

    assert sdata.shapes == {}
