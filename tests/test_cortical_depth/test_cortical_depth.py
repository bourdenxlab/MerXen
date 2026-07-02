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
    BoundaryPieceAnnotations,
    load_boundary_annotations,
)
from merxen.cortical_depth.equivolumetric import compute_equal_area_depth
from merxen.cortical_depth.laplace import interpolate_scalar_field, solve_laplace_depth
from merxen.cortical_depth.pipeline import (
    PieceDepthResult,
    _assign_piecewise_cortical_depth_to_cells,
)
from merxen.cortical_depth.ribbon import (
    _unique_valid_polygons,
    rasterize_cortical_ribbon,
)
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


def test_piece_aware_boundary_parsing_from_combined_geojson(tmp_path: Path) -> None:
    """Combined GeoJSON should preserve explicit tissue_piece_id groups."""
    path = tmp_path / "piece_annotations.geojson"
    data = {
        "type": "FeatureCollection",
        "features": [
            _line_feature(
                [(0, 0), (0, 80), (120, 80), (120, 0)],
                role="side_boundary",
            ),
            _line_feature(
                [(0, 10), (120, 10)],
                role="pial_boundary",
                tissue_piece_id="piece_a",
            ),
            _line_feature(
                [(0, 40), (120, 40)],
                role="gray_white_boundary",
                tissue_piece_id="piece_a",
            ),
            _line_feature(
                [(0, 55), (120, 55)],
                role="pial_boundary",
                tissue_piece_id="piece_b",
            ),
            _polygon_feature(
                [(0, 55), (120, 55), (120, 80), (0, 80)],
                role="cortical_ribbon",
                tissue_piece_id="piece_b",
            ),
        ],
    }
    path.write_text(json.dumps(data))

    annotations = load_boundary_annotations(annotation_path=path)

    assert [piece.tissue_piece_id for piece in annotations.pieces] == [
        "piece_a",
        "piece_b",
    ]
    assert annotations.pieces[0].piece_mode == "depth"
    assert annotations.pieces[1].piece_mode == "mask_qc_only"
    assert annotations.edge is not None


def test_closed_box_edge_depth_piece_rasterizes() -> None:
    """A closed single edge plus pial/WM lines should polygonize the depth piece."""
    edge = LineString([(0, 0), (120, 0), (120, 80), (0, 80), (0, 0)])
    piece = BoundaryPieceAnnotations(
        tissue_piece_id="piece_a",
        pial=LineString([(0, 15), (120, 15)]),
        wm=LineString([(0, 45), (120, 45)]),
    )

    grid = rasterize_cortical_ribbon(
        piece,
        edge_line=edge,
        resolution_um=2.0,
        coordinate_unit_um=1.0,
        boundary_band_um=2.0,
    )

    assert grid.mask.any()
    assert grid.pial_boundary.any()
    assert grid.wm_boundary.any()
    assert grid.tissue_piece_id == "piece_a"


def test_edge_overhang_with_near_snapped_endpoints_rasterizes() -> None:
    """Overdrawn edges and tiny endpoint offsets should still form a piece polygon."""
    edge = LineString([(0, -20), (0, 80), (100, 80), (100, -20)])
    piece = BoundaryPieceAnnotations(
        tissue_piece_id="piece_a",
        pial=LineString([(0.0001, 10), (100, 10)]),
        wm=LineString([(0, 50), (100.0001, 50)]),
    )

    grid = rasterize_cortical_ribbon(
        piece,
        edge_line=edge,
        resolution_um=2.0,
        coordinate_unit_um=1.0,
        boundary_band_um=2.0,
    )

    assert grid.mask.any()
    assert grid.pial_boundary.any()
    assert grid.wm_boundary.any()


def test_near_duplicate_candidate_polygons_are_deduplicated() -> None:
    """Projected and polygonized versions of one piece should not look ambiguous."""
    square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    near_square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.000001), (0.0, 10.0)])
    distinct = Polygon([(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)])

    assert _unique_valid_polygons([square, near_square, distinct]) == [
        square,
        distinct,
    ]


def test_pial_only_piece_rasterizes_mask_qc_only_and_assigns_cells() -> None:
    """Pial-only pieces should mark inside cells without depth values."""
    piece = BoundaryPieceAnnotations(
        tissue_piece_id="surface_only",
        pial=LineString([(0, 10), (100, 10)]),
        ribbon=Polygon([(0, 10), (100, 10), (100, 50), (0, 50)]),
    )
    grid = rasterize_cortical_ribbon(
        piece,
        resolution_um=2.0,
        coordinate_unit_um=1.0,
        boundary_band_um=2.0,
        require_wm=False,
    )
    coords = CellCoordinateTable(
        pd.Index(["inside", "outside"]),
        np.array([[50, 25], [50, 70]], dtype=float),
        "synthetic",
    )
    result = PieceDepthResult(
        tissue_piece_id="surface_only",
        piece_mode="mask_qc_only",
        grid=grid,
        solution=None,
        equal_area_depth=None,
        streamlines=[],
    )

    assignments = _assign_piecewise_cortical_depth_to_cells(
        coords,
        [result],
        side_boundary_distance_um=5.0,
    )

    assert bool(assignments.loc["inside", "inside_cortical_ribbon"])
    assert assignments.loc["inside", "cortical_depth_piece_id"] == "surface_only"
    assert assignments.loc["inside", "cortical_depth_piece_mode"] == "mask_qc_only"
    assert assignments.loc["inside", "cortical_depth_qc_flag"] == "pial_only_no_wm"
    assert np.isnan(assignments.loc["inside", "laplace_depth"])
    assert assignments.loc["outside", "cortical_depth_qc_flag"] == "outside_ribbon"


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


def _line_feature(
    coords: list[tuple[float, float]],
    *,
    role: str,
    tissue_piece_id: str | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = {"role": role}
    if tissue_piece_id is not None:
        properties["tissue_piece_id"] = tissue_piece_id
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _polygon_feature(
    coords: list[tuple[float, float]],
    *,
    role: str,
    tissue_piece_id: str | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = {"role": role}
    if tissue_piece_id is not None:
        properties["tissue_piece_id"] = tissue_piece_id
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [list(Polygon(coords).exterior.coords)],
        },
    }
