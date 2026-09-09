"""Tests for canonical MERSCOPE boundary input."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from spatialdata.transformations import Identity

from merxen.io.builders.merscope import _get_polygons


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
