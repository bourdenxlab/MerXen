"""Pipeline entry point for cortical-depth computation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import contains_xy
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from skimage import io as skio
from spatialdata.models import TableModel

from merxen.config import CorticalDepthConfig, CorticalDepthTableConfig
from merxen.cortical_depth.assign_cells import (
    CORTICAL_DEPTH_COLUMNS,
    apply_depth_columns,
    assign_cortical_depth_to_cells,
    assignment_summary,
    extract_cell_coordinates,
)
from merxen.cortical_depth.boundaries import (
    BoundaryAnnotationSet,
    BoundaryPieceAnnotations,
    load_boundary_annotations,
)
from merxen.cortical_depth.equivolumetric import compute_equal_area_depth
from merxen.cortical_depth.laplace import solve_laplace_depth
from merxen.cortical_depth.plotting import (
    depth_contours_to_geojson,
    plot_cells_by_annotation,
    plot_cells_by_depth,
    plot_depth_difference,
    plot_depth_overlay,
    write_geojson,
)
from merxen.cortical_depth.ribbon import points_inside_mask, rasterize_cortical_ribbon
from merxen.cortical_depth.streamlines import (
    Streamline,
    streamlines_to_dataframe,
    streamlines_to_geojson,
    trace_streamlines,
)
from merxen.io.spatialdata_io import write_or_replace_element
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PieceDepthResult:
    tissue_piece_id: str
    piece_mode: str
    grid: Any
    solution: Any | None
    equal_area_depth: Any | None
    streamlines: list[Streamline]


def run_cortical_depth(config: CorticalDepthConfig) -> dict[str, Path]:
    """Run cortical-depth computation for one platform SpatialData zarr."""
    latest_path = Path(config.latest_zarr_path)
    output_dir = Path(config.output_dir)
    if not latest_path.exists():
        raise FileNotFoundError(f"[{config.dataset_name}] Missing zarr: {latest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_status(f"[{config.dataset_name}] Loading cortical-depth annotations")
    annotations = load_boundary_annotations(
        pial_path=config.pial_boundary_path,
        wm_path=config.wm_boundary_path,
        side_boundary_path=config.side_boundary_path,
        exclusion_path=config.exclusion_path,
        ribbon_path=config.ribbon_path,
        annotation_path=config.annotation_path,
        smoothing_window=config.boundary_smoothing_window,
    )
    piece_results = [
        _process_annotation_piece(
            config,
            piece,
            edge_line=annotations.edge,
            side_boundaries=annotations.side_boundaries,
        )
        for piece in annotations.pieces
    ]
    primary_depth_result = next(
        (result for result in piece_results if result.solution is not None), None
    )
    paths = _write_piece_geometry_outputs(
        output_dir=output_dir,
        dataset_name=config.dataset_name,
        piece_results=piece_results,
        contour_levels=config.contour_levels,
    )

    sdata_obj = sd.read_zarr(latest_path)
    table_summaries: dict[str, Any] = {}
    try:
        for table_config in config.tables:
            table_paths, summary = _annotate_table(
                sdata_obj=sdata_obj,
                table_config=table_config,
                config=config,
                output_dir=output_dir,
                annotations=annotations,
                piece_results=piece_results,
                primary_depth_result=primary_depth_result,
            )
            paths.update(table_paths)
            table_summaries[table_config.segmentation] = summary
    finally:
        del sdata_obj
        force_release(note=f"after {config.dataset_name} cortical depth")

    summary_path = output_dir / "cortical_depth_qc_summary.json"
    summary = _build_qc_summary(
        config=config,
        solution_residual=(
            primary_depth_result.solution.residual
            if primary_depth_result is not None
            and primary_depth_result.solution is not None
            else None
        ),
        streamlines=[line for result in piece_results for line in result.streamlines],
        table_summaries=table_summaries,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    paths["qc_summary"] = summary_path
    paths["latest_zarr"] = latest_path
    log_status(f"[{config.dataset_name}] Cortical-depth computation complete")
    return paths


def _process_annotation_piece(
    config: CorticalDepthConfig,
    piece: BoundaryPieceAnnotations,
    *,
    edge_line: Any,
    side_boundaries: tuple[Any, ...] = (),
) -> PieceDepthResult:
    raster_annotations = (
        piece.as_legacy_annotations(tuple(side_boundaries))
        if edge_line is None and side_boundaries
        else piece
    )
    grid = rasterize_cortical_ribbon(
        raster_annotations,
        resolution_um=config.raster_resolution_um,
        coordinate_unit_um=config.coordinate_unit_um,
        padding_um=config.raster_padding_um,
        boundary_band_um=config.boundary_band_um,
        edge_line=edge_line,
        require_wm=piece.wm is not None,
    )
    if piece.wm is None:
        return PieceDepthResult(
            tissue_piece_id=piece.tissue_piece_id,
            piece_mode=piece.piece_mode,
            grid=grid,
            solution=None,
            equal_area_depth=None,
            streamlines=[],
        )
    solution = solve_laplace_depth(grid)
    streamlines = trace_streamlines(
        solution.phi,
        grid,
        spacing_um=config.streamline_spacing_um,
        step_um=config.streamline_step_um,
        max_steps=config.streamline_max_steps,
        resample_points=config.streamline_resample_points,
        side_boundary_distance_um=config.side_boundary_distance_um,
    )
    equal_area = compute_equal_area_depth(solution.phi, grid, streamlines)
    return PieceDepthResult(
        tissue_piece_id=piece.tissue_piece_id,
        piece_mode=piece.piece_mode,
        grid=grid,
        solution=solution,
        equal_area_depth=equal_area,
        streamlines=streamlines,
    )


def _assign_piecewise_cortical_depth_to_cells(
    coords: Any,
    piece_results: list[PieceDepthResult],
    *,
    side_boundary_distance_um: float,
) -> pd.DataFrame:
    points = np.asarray(coords.coordinates, dtype=float)
    assignments = _empty_piecewise_assignments(
        coords.cell_ids.astype(str), points.shape[0]
    )
    unassigned = np.ones(points.shape[0], dtype=bool)
    for result in piece_results:
        inside = points_inside_mask(points, result.grid)
        take = unassigned & inside
        if not take.any():
            continue
        if result.solution is not None and result.equal_area_depth is not None:
            piece_assignments = assign_cortical_depth_to_cells(
                coords,
                result.solution,
                result.equal_area_depth,
                result.streamlines,
                side_boundary_distance_um=side_boundary_distance_um,
            )
            for column in piece_assignments.columns:
                assignments.loc[take, column] = piece_assignments.loc[
                    take, column
                ].to_numpy()
        else:
            assignments.loc[take, "inside_cortical_ribbon"] = True
            assignments.loc[take, "cortical_depth_qc_flag"] = "pial_only_no_wm"
        assignments.loc[take, "cortical_depth_piece_id"] = result.tissue_piece_id
        assignments.loc[take, "cortical_depth_piece_mode"] = result.piece_mode
        unassigned[take] = False
    return assignments


def _empty_piecewise_assignments(cell_ids: pd.Index, n_cells: int) -> pd.DataFrame:
    assignments = pd.DataFrame(index=cell_ids.astype(str))
    assignments["inside_cortical_ribbon"] = np.zeros(n_cells, dtype=bool)
    assignments["cortical_depth_piece_id"] = pd.Series(
        [None] * n_cells, index=assignments.index, dtype=object
    )
    assignments["cortical_depth_piece_mode"] = pd.Series(
        [None] * n_cells, index=assignments.index, dtype=object
    )
    for column in CORTICAL_DEPTH_COLUMNS:
        if column in assignments.columns:
            continue
        if column == "cortical_depth_qc_flag":
            assignments[column] = "outside_ribbon"
        elif column == "cortical_depth_annotation":
            assignments[column] = "outside_brain"
        else:
            assignments[column] = np.nan
    return assignments


def _annotate_table(
    *,
    sdata_obj: Any,
    table_config: CorticalDepthTableConfig,
    config: CorticalDepthConfig,
    output_dir: Path,
    annotations: BoundaryAnnotationSet,
    piece_results: list[PieceDepthResult],
    primary_depth_result: PieceDepthResult | None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    table_key = str(table_config.table_key)
    if table_key not in sdata_obj.tables:
        raise KeyError(
            f"[{config.dataset_name}] table_key={table_key!r} not found. "
            f"Available tables: {list(sdata_obj.tables.keys())}"
        )
    shape_key = _resolve_shape_key(
        sdata_obj,
        table=sdata_obj.tables[table_key],
        requested=table_config.shape_key,
        platform=config.platform,
    )
    coords = extract_cell_coordinates(
        sdata_obj.tables[table_key],
        sdata_obj=sdata_obj,
        shape_key=shape_key,
    )
    assignments = _assign_piecewise_cortical_depth_to_cells(
        coords,
        piece_results,
        side_boundary_distance_um=config.side_boundary_distance_um,
    )
    assignments["cortical_depth_annotation"] = _classify_cell_tissue_annotations(
        coords.coordinates,
        piece_results,
        annotations,
    )
    cells = assignments.copy()
    cells.insert(0, "cell_id", cells.index.astype(str))
    cells.insert(1, "x", coords.coordinates[:, 0])
    cells.insert(2, "y", coords.coordinates[:, 1])

    segmentation_dir = output_dir / table_config.segmentation
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    sample_stem = f"{config.dataset_name}_{table_config.segmentation}".lower()
    cells_path = segmentation_dir / f"{sample_stem}_cells_with_cortical_depth.parquet"
    cells.to_parquet(cells_path, index=False)

    plot_paths: dict[str, Path] = {}
    if primary_depth_result is not None:
        laplace_plot_path = segmentation_dir / f"{sample_stem}_cells_laplace_depth.png"
        plot_cells_by_depth(
            laplace_plot_path,
            cells,
            primary_depth_result.grid,
            value_column="laplace_depth",
        )
        plot_paths[f"{table_config.segmentation}_cells_laplace_depth_png"] = (
            laplace_plot_path
        )
        equivolumetric_plot_path = (
            segmentation_dir / f"{sample_stem}_cells_equivolumetric_depth.png"
        )
        plot_cells_by_depth(
            equivolumetric_plot_path,
            cells,
            primary_depth_result.grid,
            value_column="equivolumetric_depth",
        )
        plot_paths[f"{table_config.segmentation}_cells_equivolumetric_depth_png"] = (
            equivolumetric_plot_path
        )

    annotation_plot_path = (
        segmentation_dir / f"{sample_stem}_cells_tissue_annotation.png"
    )
    plot_cells_by_annotation(
        annotation_plot_path,
        cells,
        [result.grid for result in piece_results],
    )
    plot_paths[f"{table_config.segmentation}_cells_tissue_annotation_png"] = (
        annotation_plot_path
    )

    if config.write_spatialdata_table:
        updated = apply_depth_columns(sdata_obj.tables[table_key], assignments)
        parsed = _parse_table_for_spatialdata(
            updated,
            source_table=sdata_obj.tables[table_key],
            table_key=table_key,
            region=shape_key,
        )
        write_or_replace_element(
            sdata_obj,
            table_key,
            "tables",
            parsed,
            overwrite=True,
        )

    summary = assignment_summary(assignments)
    summary.update(
        {
            "table_key": table_key,
            "shape_key": shape_key,
            "coordinate_source": coords.source,
            "cells_path": str(cells_path),
        }
    )
    return (
        {f"{table_config.segmentation}_cells": cells_path, **plot_paths},
        summary,
    )


def _classify_cell_tissue_annotations(
    points: np.ndarray,
    piece_results: list[PieceDepthResult],
    annotations: BoundaryAnnotationSet,
) -> np.ndarray:
    """Classify cells as grey matter, white matter, excluded, or outside brain."""
    coords = np.asarray(points, dtype=float)
    labels = np.full(coords.shape[0], "outside_brain", dtype=object)

    brain_polygon = _brain_polygon_from_annotations(annotations)
    if brain_polygon is not None:
        labels[_points_inside_geometry(coords, brain_polygon)] = "white_matter"

    for result in piece_results:
        labels[_points_inside_geometry(coords, result.grid.polygon)] = "grey_matter"

    if annotations.exclusions:
        exclusions = unary_union(list(annotations.exclusions))
        labels[_points_inside_geometry(coords, exclusions)] = "excluded"
    return labels


def _brain_polygon_from_annotations(
    annotations: BoundaryAnnotationSet,
) -> Polygon | MultiPolygon | None:
    edge_lines = [annotations.edge] if annotations.edge is not None else []
    edge_lines.extend(annotations.side_boundaries)
    for line in edge_lines:
        polygon = _closed_line_polygon(line)
        if polygon is not None:
            return polygon
    return None


def _closed_line_polygon(line: LineString | None) -> Polygon | MultiPolygon | None:
    if line is None:
        return None
    coords = np.asarray(line.coords, dtype=float)
    if coords.shape[0] < 4:
        return None
    if not np.allclose(coords[0, :2], coords[-1, :2], rtol=0.0, atol=1e-9):
        return None
    polygon = Polygon(coords[:, :2])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if isinstance(polygon, Polygon | MultiPolygon) and not polygon.is_empty:
        return polygon
    return None


def _points_inside_geometry(points: np.ndarray, geometry: BaseGeometry) -> np.ndarray:
    coords = np.asarray(points, dtype=float)
    inside = np.zeros(coords.shape[0], dtype=bool)
    finite = np.isfinite(coords).all(axis=1)
    if not finite.any() or geometry.is_empty:
        return inside
    inside[finite] = contains_xy(geometry, coords[finite, 0], coords[finite, 1])
    return inside


def _write_piece_geometry_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    piece_results: list[PieceDepthResult],
    contour_levels: list[float],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    single_depth = len(piece_results) == 1 and piece_results[0].solution is not None
    for result in piece_results:
        piece_dir = (
            output_dir
            if single_depth
            else output_dir / "pieces" / _safe_piece_id(result.tissue_piece_id)
        )
        piece_dir.mkdir(parents=True, exist_ok=True)
        if result.solution is None or result.equal_area_depth is None:
            mask_path = piece_dir / "cortical_ribbon_mask.tif"
            skio.imsave(
                mask_path,
                (result.grid.mask.astype(np.uint8) * 255),
                check_contrast=False,
            )
            paths[f"{result.tissue_piece_id}_ribbon_mask"] = mask_path
            continue
        piece_paths = _write_geometry_outputs(
            output_dir=piece_dir,
            dataset_name=dataset_name
            if single_depth
            else f"{dataset_name}_{result.tissue_piece_id}",
            grid=result.grid,
            laplace_depth=result.solution.phi,
            equal_area_depth=result.equal_area_depth.depth,
            streamlines=result.streamlines,
            contour_levels=contour_levels,
        )
        for key, path in piece_paths.items():
            paths[key if single_depth else f"{result.tissue_piece_id}_{key}"] = path
        if not result.equal_area_depth.column_summary.empty:
            summary = result.equal_area_depth.column_summary.copy()
            summary.insert(0, "tissue_piece_id", result.tissue_piece_id)
            column_summary_path = piece_dir / "equivolumetric_column_summary.parquet"
            summary.to_parquet(column_summary_path, index=False)
            paths[
                "equivolumetric_column_summary"
                if single_depth
                else f"{result.tissue_piece_id}_equivolumetric_column_summary"
            ] = column_summary_path

    _write_combined_piece_vectors(
        output_dir=output_dir,
        piece_results=piece_results,
        contour_levels=contour_levels,
        paths=paths,
    )
    return paths


def _write_combined_piece_vectors(
    *,
    output_dir: Path,
    piece_results: list[PieceDepthResult],
    contour_levels: list[float],
    paths: dict[str, Path],
) -> None:
    streamline_features: list[dict[str, Any]] = []
    streamline_rows: list[pd.DataFrame] = []
    laplace_features: list[dict[str, Any]] = []
    equiv_features: list[dict[str, Any]] = []
    for result in piece_results:
        if result.solution is None or result.equal_area_depth is None:
            continue
        streamline_fc = streamlines_to_geojson(result.streamlines)
        for feature in streamline_fc.get("features", []):
            _add_piece_properties(feature, result)
            streamline_features.append(feature)
        df = streamlines_to_dataframe(result.streamlines)
        if not df.empty:
            df.insert(0, "tissue_piece_id", result.tissue_piece_id)
            df.insert(1, "piece_mode", result.piece_mode)
            streamline_rows.append(df)
        laplace_fc = depth_contours_to_geojson(
            result.solution.phi,
            result.grid,
            levels=contour_levels,
            property_name="laplace_depth",
        )
        for feature in laplace_fc.get("features", []):
            _add_piece_properties(feature, result)
            laplace_features.append(feature)
        equiv_fc = depth_contours_to_geojson(
            result.equal_area_depth.depth,
            result.grid,
            levels=contour_levels,
            property_name="equivolumetric_depth",
        )
        for feature in equiv_fc.get("features", []):
            _add_piece_properties(feature, result)
            equiv_features.append(feature)
    if streamline_features:
        streamlines_geojson = output_dir / "streamlines.geojson"
        write_geojson(
            {"type": "FeatureCollection", "features": streamline_features},
            streamlines_geojson,
        )
        paths["streamlines_geojson"] = streamlines_geojson
    if streamline_rows:
        streamlines_parquet = output_dir / "streamlines.parquet"
        pd.concat(streamline_rows, ignore_index=True).to_parquet(
            streamlines_parquet, index=False
        )
        paths["streamlines_parquet"] = streamlines_parquet
    if laplace_features:
        contours_geojson = output_dir / "depth_contours.geojson"
        write_geojson(
            {"type": "FeatureCollection", "features": laplace_features},
            contours_geojson,
        )
        paths["depth_contours_geojson"] = contours_geojson
    if equiv_features:
        equiv_contours_geojson = output_dir / "equivolumetric_depth_contours.geojson"
        write_geojson(
            {"type": "FeatureCollection", "features": equiv_features},
            equiv_contours_geojson,
        )
        paths["equivolumetric_contours_geojson"] = equiv_contours_geojson


def _add_piece_properties(feature: dict[str, Any], result: PieceDepthResult) -> None:
    properties = feature.setdefault("properties", {})
    properties["tissue_piece_id"] = result.tissue_piece_id
    properties["piece_mode"] = result.piece_mode


def _safe_piece_id(piece_id: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in str(piece_id)
    ).strip("_")
    return safe or "piece"


def _write_geometry_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    grid: Any,
    laplace_depth: np.ndarray,
    equal_area_depth: np.ndarray,
    streamlines: list[Streamline],
    contour_levels: list[float],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    mask_path = output_dir / "cortical_ribbon_mask.tif"
    skio.imsave(mask_path, (grid.mask.astype(np.uint8) * 255), check_contrast=False)
    paths["ribbon_mask"] = mask_path

    streamline_df = streamlines_to_dataframe(streamlines)
    streamlines_parquet = output_dir / "streamlines.parquet"
    streamline_df.to_parquet(streamlines_parquet, index=False)
    paths["streamlines_parquet"] = streamlines_parquet
    streamlines_geojson = output_dir / "streamlines.geojson"
    write_geojson(streamlines_to_geojson(streamlines), streamlines_geojson)
    paths["streamlines_geojson"] = streamlines_geojson

    contours_geojson = output_dir / "depth_contours.geojson"
    write_geojson(
        depth_contours_to_geojson(
            laplace_depth,
            grid,
            levels=contour_levels,
            property_name="laplace_depth",
        ),
        contours_geojson,
    )
    paths["depth_contours_geojson"] = contours_geojson

    equiv_contours_geojson = output_dir / "equivolumetric_depth_contours.geojson"
    write_geojson(
        depth_contours_to_geojson(
            equal_area_depth,
            grid,
            levels=contour_levels,
            property_name="equivolumetric_depth",
        ),
        equiv_contours_geojson,
    )
    paths["equivolumetric_contours_geojson"] = equiv_contours_geojson

    overlay_path = output_dir / f"{dataset_name.lower()}_cortical_depth_overlay.png"
    plot_depth_overlay(
        overlay_path,
        grid,
        laplace_depth,
        streamlines,
        contour_levels=contour_levels,
    )
    paths["overlay_png"] = overlay_path

    difference_path = (
        output_dir / f"{dataset_name.lower()}_laplace_equivolumetric_difference.png"
    )
    plot_depth_difference(
        difference_path,
        grid,
        laplace_depth,
        equal_area_depth,
    )
    paths["laplace_equivolumetric_difference_png"] = difference_path
    return paths


def _parse_table_for_spatialdata(
    table: ad.AnnData,
    *,
    source_table: ad.AnnData,
    table_key: str,
    region: str | None,
) -> ad.AnnData:
    out = table.copy()
    attrs = dict(source_table.uns.get("spatialdata_attrs", {}))
    region_key = str(attrs.get("region_key", "region"))
    instance_key = attrs.get("instance_key")
    if not isinstance(instance_key, str) or instance_key not in out.obs.columns:
        instance_key = "cell_id"
    if instance_key not in out.obs.columns:
        out.obs[instance_key] = out.obs_names.astype(str)
    parsed_region = region or _region_from_attrs(attrs) or str(table_key)
    out.obs[region_key] = pd.Categorical(
        [str(parsed_region)] * out.n_obs,
        categories=[str(parsed_region)],
    )
    out.uns.pop("spatialdata_attrs", None)
    return TableModel.parse(
        out,
        region=str(parsed_region),
        region_key=region_key,
        instance_key=str(instance_key),
    )


def _resolve_shape_key(
    sdata_obj: Any,
    *,
    table: ad.AnnData,
    requested: str | None,
    platform: str,
) -> str | None:
    if len(sdata_obj.shapes) == 0:
        return None
    if requested is not None:
        aligned = f"{requested}_aligned_nonrigid"
        if platform.upper() == "MERSCOPE" and aligned in sdata_obj.shapes:
            return aligned
        if requested not in sdata_obj.shapes:
            raise KeyError(
                f"Requested shape_key={requested!r} not found. "
                f"Available shapes: {list(sdata_obj.shapes.keys())}"
            )
        return requested
    region = _region_from_attrs(dict(table.uns.get("spatialdata_attrs", {})))
    if region is not None and region in sdata_obj.shapes:
        return region
    if region is not None and f"{region}_aligned_nonrigid" in sdata_obj.shapes:
        return f"{region}_aligned_nonrigid"
    return str(list(sdata_obj.shapes.keys())[0])


def _region_from_attrs(attrs: dict[str, Any]) -> str | None:
    region = attrs.get("region")
    if isinstance(region, str):
        return region
    if isinstance(region, list | tuple) and region:
        return str(region[0])
    return None


def _build_qc_summary(
    *,
    config: CorticalDepthConfig,
    solution_residual: float | None,
    streamlines: list[Streamline],
    table_summaries: dict[str, Any],
) -> dict[str, Any]:
    thickness = np.asarray([line.thickness_um for line in streamlines], dtype=float)
    finite = thickness[np.isfinite(thickness) & (thickness > 0)]
    failed = [line for line in streamlines if line.qc_flag != "ok"]
    warnings = _depth_warnings(streamlines, finite)
    return {
        "dataset_name": config.dataset_name,
        "platform": config.platform,
        "laplace_residual": None
        if solution_residual is None
        else float(solution_residual),
        "n_streamlines": int(len(streamlines)),
        "n_failed_or_flagged_streamlines": int(len(failed)),
        "mean_streamline_thickness_um": _array_stat(finite, "mean"),
        "median_streamline_thickness_um": _array_stat(finite, "median"),
        "min_streamline_thickness_um": _array_stat(finite, "min"),
        "max_streamline_thickness_um": _array_stat(finite, "max"),
        "streamline_qc_flag_counts": {
            str(key): int(value)
            for key, value in pd.Series([line.qc_flag for line in streamlines])
            .value_counts(dropna=False)
            .items()
        },
        "tables": table_summaries,
        "warnings": warnings,
    }


def _depth_warnings(
    streamlines: list[Streamline],
    finite_thickness: np.ndarray,
) -> list[str]:
    warnings: list[str] = []
    if not streamlines:
        warnings.append("no_streamlines_generated")
        return warnings
    failed_fraction = sum(line.qc_flag != "ok" for line in streamlines) / len(
        streamlines
    )
    if failed_fraction > 0.2:
        warnings.append(f"high_failed_streamline_fraction:{failed_fraction:.3f}")
    if finite_thickness.size >= 4:
        median = float(np.nanmedian(finite_thickness))
        if median > 0:
            abnormal = (finite_thickness < 0.5 * median) | (
                finite_thickness > 2.0 * median
            )
            if np.mean(abnormal) > 0.1:
                warnings.append("abnormal_local_thickness_variation")
    return warnings


def _array_stat(values: np.ndarray, name: str) -> float | None:
    if values.size == 0:
        return None
    if name == "mean":
        value = np.nanmean(values)
    elif name == "median":
        value = np.nanmedian(values)
    elif name == "min":
        value = np.nanmin(values)
    elif name == "max":
        value = np.nanmax(values)
    else:
        raise ValueError(f"Unknown array stat: {name}")
    return None if not np.isfinite(value) else float(value)
