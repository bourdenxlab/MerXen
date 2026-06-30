"""Synthetic tests for cortical-depth geometry, fields, and assignments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

from merxen.cortical_depth.assign_cells import (
    CellCoordinateTable,
    assign_cortical_depth_to_cells,
)
from merxen.cortical_depth.boundaries import (
    BoundaryAnnotations,
    load_boundary_annotations,
)
from merxen.cortical_depth.equivolumetric import compute_equal_area_depth
from merxen.cortical_depth.laplace import interpolate_scalar_field, solve_laplace_depth
from merxen.cortical_depth.ribbon import rasterize_cortical_ribbon
from merxen.cortical_depth.streamlines import trace_streamlines


def test_boundary_parsing_from_combined_geojson(tmp_path: Path) -> None:
    """Combined napari-style GeoJSON should resolve role-labelled features."""
    path = tmp_path / "annotations.geojson"
    data = {
        "type": "FeatureCollection",
        "features": [
            _line_feature([(0, 0), (100, 0)], role="pial_boundary"),
            _line_feature([(0, 50), (100, 50)], role="grey_white_boundary"),
            _line_feature([(0, 0), (0, 50)], role="side_boundary"),
            _polygon_feature([(40, 20), (50, 20), (50, 30), (40, 30)], role="tear"),
        ],
    }
    path.write_text(json.dumps(data))

    annotations = load_boundary_annotations(annotation_path=path)

    assert annotations.pial.length == 100
    assert annotations.wm.length == 100
    assert len(annotations.side_boundaries) == 1
    assert len(annotations.exclusions) == 1


def test_rectangular_ribbon_laplace_depth_is_approximately_linear() -> None:
    """A rectangular ribbon should produce monotone, nearly linear depth."""
    grid = rasterize_cortical_ribbon(
        _rectangle_annotations(),
        resolution_um=1.0,
        coordinate_unit_um=1.0,
        boundary_band_um=1.0,
    )
    solution = solve_laplace_depth(grid)
    points = np.array([[50, 5], [50, 25], [50, 45]], dtype=float)
    depths = interpolate_scalar_field(solution.phi, grid.spec, points)

    assert solution.converged
    assert solution.residual < 1e-8
    assert np.all(np.diff(depths) > 0)
    assert np.allclose(depths, [0.08, 0.50, 0.92], atol=0.08)


def test_streamlines_reach_white_matter_and_preserve_order() -> None:
    """Synthetic rectangular streamlines should remain ordered and reach WM."""
    grid = rasterize_cortical_ribbon(
        _rectangle_annotations(),
        resolution_um=1.0,
        coordinate_unit_um=1.0,
        boundary_band_um=1.0,
    )
    solution = solve_laplace_depth(grid)
    streamlines = trace_streamlines(
        solution.phi,
        grid,
        spacing_um=20.0,
        step_um=1.0,
        resample_points=21,
        side_boundary_distance_um=5.0,
    )

    reached = [line for line in streamlines if line.reached_wm]
    mid_x = np.array([line.points[10, 0] for line in streamlines])
    thickness = np.array([line.thickness_um for line in streamlines])

    assert len(reached) == len(streamlines)
    assert np.all(np.diff(mid_x) >= -1e-6)
    assert np.allclose(np.nanmedian(thickness), np.asarray(thickness).mean())
    assert np.allclose(np.nanmedian(thickness), 50.0, atol=2.0)


def test_cell_assignment_and_equal_area_depth_in_rectangle() -> None:
    """Cell assignment should fill required columns and flag outside cells."""
    grid = rasterize_cortical_ribbon(
        _rectangle_annotations(),
        resolution_um=1.0,
        coordinate_unit_um=1.0,
        boundary_band_um=1.0,
    )
    solution = solve_laplace_depth(grid)
    streamlines = trace_streamlines(
        solution.phi,
        grid,
        spacing_um=20.0,
        step_um=1.0,
        resample_points=31,
    )
    equal_area = compute_equal_area_depth(solution.phi, grid, streamlines)
    coords = CellCoordinateTable(
        pd.Index(["upper", "middle", "deep", "outside"]),
        np.array([[50, 5], [50, 25], [50, 45], [150, 25]], dtype=float),
        "synthetic",
    )

    assignments = assign_cortical_depth_to_cells(
        coords,
        solution,
        equal_area,
        streamlines,
    )

    assert not bool(assignments.loc["outside", "inside_cortical_ribbon"])
    assert assignments.loc["outside", "cortical_depth_qc_flag"] == "outside_ribbon"
    inside = assignments.loc[["upper", "middle", "deep"]]
    assert inside["laplace_depth"].is_monotonic_increasing
    assert inside["equivolumetric_depth"].is_monotonic_increasing
    assert np.allclose(inside["streamline_thickness_um"], 50.0, atol=2.0)


def test_curved_ribbon_produces_smooth_ordered_streamlines() -> None:
    """Curved synthetic ribbons should yield smooth non-crossing streamlines."""
    x = np.linspace(0, 120, 80)
    pial_y = 8.0 * np.sin(x / 120.0 * np.pi)
    wm_y = pial_y + 45.0 + 4.0 * np.cos(x / 120.0 * np.pi)
    annotations = BoundaryAnnotations(
        pial=LineString(np.column_stack([x, pial_y])),
        wm=LineString(np.column_stack([x, wm_y])),
    )
    grid = rasterize_cortical_ribbon(
        annotations,
        resolution_um=2.0,
        coordinate_unit_um=1.0,
        boundary_band_um=2.0,
    )
    solution = solve_laplace_depth(grid)
    streamlines = trace_streamlines(
        solution.phi,
        grid,
        spacing_um=15.0,
        step_um=1.0,
        resample_points=31,
    )
    reached = [line for line in streamlines if line.reached_wm]
    mid_x = np.array([line.points[15, 0] for line in reached])

    assert len(reached) >= int(0.8 * len(streamlines))
    assert np.all(np.diff(mid_x) >= -3.0)


def _rectangle_annotations() -> BoundaryAnnotations:
    return BoundaryAnnotations(
        pial=LineString([(0, 0), (100, 0)]),
        wm=LineString([(0, 50), (100, 50)]),
    )


def _line_feature(coords: list[tuple[float, float]], *, role: str) -> dict[str, object]:
    return {
        "type": "Feature",
        "properties": {"role": role},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _polygon_feature(
    coords: list[tuple[float, float]],
    *,
    role: str,
) -> dict[str, object]:
    return {
        "type": "Feature",
        "properties": {"role": role},
        "geometry": {
            "type": "Polygon",
            "coordinates": [list(Polygon(coords).exterior.coords)],
        },
    }
