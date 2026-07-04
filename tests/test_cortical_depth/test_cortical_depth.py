"""Synthetic tests for cortical-depth geometry, fields, and assignments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from merxen.config import CorticalDepthTableConfig
from merxen.cortical_depth import streamlines as streamlines_module
from merxen.cortical_depth.assign_cells import (
    CellCoordinateTable,
    assign_cortical_depth_to_cells,
)
from merxen.cortical_depth.boundaries import (
    BoundaryAnnotations,
    BoundaryAnnotationSet,
    BoundaryPieceAnnotations,
    load_boundary_annotations,
)
from merxen.cortical_depth.equivolumetric import compute_equal_area_depth
from merxen.cortical_depth.laplace import interpolate_scalar_field, solve_laplace_depth
from merxen.cortical_depth.pipeline import (
    PieceDepthResult,
    _assign_piecewise_cortical_depth_to_cells,
    _classify_cell_tissue_annotations,
    _clustering_table_key,
)
from merxen.cortical_depth.plotting import (
    plot_cells_by_annotation,
    plot_depth_difference,
    plot_depth_violins_by_broad_class,
    plot_depth_violins_by_subcluster,
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


def test_cell_tissue_annotation_classifies_whole_sample() -> None:
    """Cells should be annotated as grey, white, excluded, or outside brain."""
    exclusion = Polygon([(45, 20), (55, 20), (55, 30), (45, 30)])
    edge = LineString([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
    piece = BoundaryPieceAnnotations(
        tissue_piece_id="piece_a",
        pial=LineString([(0, 10), (100, 10)]),
        wm=LineString([(0, 40), (100, 40)]),
        exclusions=(exclusion,),
    )
    annotations = BoundaryAnnotationSet(
        pieces=(piece,),
        edge=edge,
        side_boundaries=(edge,),
    )
    grid = rasterize_cortical_ribbon(
        piece,
        edge_line=edge,
        resolution_um=2.0,
        coordinate_unit_um=1.0,
        boundary_band_um=2.0,
    )
    result = PieceDepthResult(
        tissue_piece_id="piece_a",
        piece_mode="depth",
        grid=grid,
        solution=None,
        equal_area_depth=None,
        streamlines=[],
    )
    points = np.array(
        [
            [20, 20],
            [20, 70],
            [50, 25],
            [120, 20],
        ],
        dtype=float,
    )

    labels = _classify_cell_tissue_annotations(points, [result], annotations)

    assert labels.tolist() == [
        "grey_matter",
        "white_matter",
        "excluded",
        "outside_brain",
    ]


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


def test_streamlines_serial_and_parallel_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel tracing must reproduce serial output bit-for-bit and in order."""
    grid = rasterize_cortical_ribbon(
        _rectangle_annotations(),
        resolution_um=1.0,
        coordinate_unit_um=1.0,
        boundary_band_um=1.0,
    )
    solution = solve_laplace_depth(grid)
    trace_kwargs = {
        "spacing_um": 5.0,
        "step_um": 1.0,
        "resample_points": 21,
        "side_boundary_distance_um": 5.0,
    }
    serial = trace_streamlines(solution.phi, grid, n_jobs=1, **trace_kwargs)

    # Force the parallel branch regardless of seed count.
    monkeypatch.setattr(streamlines_module, "_PARALLEL_STREAMLINE_THRESHOLD", 1)
    parallel = trace_streamlines(solution.phi, grid, n_jobs=2, **trace_kwargs)

    assert len(serial) == len(parallel) > 1
    for expected, actual in zip(serial, parallel, strict=True):
        assert expected.streamline_id == actual.streamline_id
        assert expected.qc_flag == actual.qc_flag
        assert expected.reached_wm == actual.reached_wm
        assert expected.near_side_boundary == actual.near_side_boundary
        assert expected.thickness_um == actual.thickness_um
        assert expected.tangential_position_um == actual.tangential_position_um
        np.testing.assert_array_equal(expected.points, actual.points)


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


def test_cortical_depth_extra_plots_are_written(tmp_path: Path) -> None:
    """Difference and tissue-annotation plots should write PNG and PDF copies."""
    grid = rasterize_cortical_ribbon(
        _rectangle_annotations(),
        resolution_um=2.0,
        coordinate_unit_um=1.0,
        boundary_band_um=2.0,
    )
    solution = solve_laplace_depth(grid)
    streamlines = trace_streamlines(solution.phi, grid, spacing_um=20.0, step_um=1.0)
    equal_area = compute_equal_area_depth(solution.phi, grid, streamlines)
    cells = pd.DataFrame(
        {
            "x": [10.0, 20.0, 120.0],
            "y": [10.0, 30.0, 10.0],
            "cortical_depth_annotation": [
                "grey_matter",
                "white_matter",
                "outside_brain",
            ],
        }
    )

    difference_path = plot_depth_difference(
        tmp_path / "difference.png",
        grid,
        solution.phi,
        equal_area.depth,
    )
    annotation_path = plot_cells_by_annotation(
        tmp_path / "annotation.png",
        cells,
        [grid],
    )

    assert difference_path.exists()
    assert difference_path.with_suffix(".pdf").exists()
    assert annotation_path.exists()
    assert annotation_path.with_suffix(".pdf").exists()


def _clustered_depth_cells() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_per_group = 40
    records = []
    groups = [
        ("Neurons", "L2/3 IT", 0.25),
        ("Neurons", "L5 ET", 0.7),
        ("Astrocytes", "Astro-1", 0.5),
        ("Oligodendrocytes", "Oligo-1", 0.9),
    ]
    for broad_class, subcluster, center in groups:
        depths = np.clip(center + rng.normal(0.0, 0.05, n_per_group), 0.0, 1.0)
        for depth in depths:
            records.append(
                {
                    "laplace_depth": float(depth),
                    "equivolumetric_depth": float(np.clip(depth + 0.02, 0.0, 1.0)),
                    "broad_class": broad_class,
                    "subcluster_label": subcluster,
                }
            )
    # Cells outside the ribbon carry NaN depth and should be dropped from violins.
    records.append(
        {
            "laplace_depth": np.nan,
            "equivolumetric_depth": np.nan,
            "broad_class": "Mixed/Unknown",
            "subcluster_label": "Mixed/Unknown",
        }
    )
    return pd.DataFrame.from_records(records)


def test_depth_violins_by_broad_class_writes_png_and_pdf(tmp_path: Path) -> None:
    """Broad-class violin plots should write PNG and PDF copies per depth column."""
    cells = _clustered_depth_cells()
    for depth_column in ("laplace_depth", "equivolumetric_depth"):
        path = plot_depth_violins_by_broad_class(
            tmp_path / f"{depth_column}_broad.png",
            cells,
            depth_column=depth_column,
        )
        assert path.exists()
        assert path.with_suffix(".pdf").exists()


def test_depth_violins_by_subcluster_writes_png_and_pdf(tmp_path: Path) -> None:
    """Subcluster violin grids should write PNG and PDF copies per depth column."""
    cells = _clustered_depth_cells()
    for depth_column in ("laplace_depth", "equivolumetric_depth"):
        path = plot_depth_violins_by_subcluster(
            tmp_path / f"{depth_column}_sub.png",
            cells,
            depth_column=depth_column,
        )
        assert path.exists()
        assert path.with_suffix(".pdf").exists()


def test_depth_violins_handle_missing_annotations(tmp_path: Path) -> None:
    """Violin plots degrade gracefully when cluster columns are absent."""
    cells = pd.DataFrame({"laplace_depth": [0.1, 0.5, 0.9]})
    broad_path = plot_depth_violins_by_broad_class(
        tmp_path / "broad.png", cells, depth_column="laplace_depth"
    )
    subcluster_path = plot_depth_violins_by_subcluster(
        tmp_path / "sub.png", cells, depth_column="laplace_depth"
    )
    assert broad_path.exists()
    assert subcluster_path.exists()


def test_clustering_table_key_maps_segmentations() -> None:
    """Segmentation branches should map to their clustering_squidpy table keys."""
    reseg = CorticalDepthTableConfig(
        segmentation="reseg", table_key="table_MOSAIK_proseg"
    )
    original = CorticalDepthTableConfig(
        segmentation="original_seg", table_key="table_original"
    )
    assert _clustering_table_key(reseg) == "table_MOSAIK_proseg_clustering_squidpy"
    assert _clustering_table_key(original) == "table_original_clustering_squidpy"


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
