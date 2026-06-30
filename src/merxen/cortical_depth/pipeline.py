"""Pipeline entry point for cortical-depth computation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from skimage import io as skio
from spatialdata.models import TableModel

from merxen.config import CorticalDepthConfig, CorticalDepthTableConfig
from merxen.cortical_depth.assign_cells import (
    apply_depth_columns,
    assign_cortical_depth_to_cells,
    assignment_summary,
    extract_cell_coordinates,
)
from merxen.cortical_depth.boundaries import load_boundary_annotations
from merxen.cortical_depth.equivolumetric import compute_equal_area_depth
from merxen.cortical_depth.laplace import solve_laplace_depth
from merxen.cortical_depth.plotting import (
    depth_contours_to_geojson,
    plot_cells_by_depth,
    plot_depth_overlay,
    write_geojson,
)
from merxen.cortical_depth.ribbon import rasterize_cortical_ribbon
from merxen.cortical_depth.streamlines import (
    Streamline,
    streamlines_to_dataframe,
    streamlines_to_geojson,
    trace_streamlines,
)
from merxen.io.spatialdata_io import write_or_replace_element
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)


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
    grid = rasterize_cortical_ribbon(
        annotations,
        resolution_um=config.raster_resolution_um,
        coordinate_unit_um=config.coordinate_unit_um,
        padding_um=config.raster_padding_um,
        boundary_band_um=config.boundary_band_um,
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

    paths = _write_geometry_outputs(
        output_dir=output_dir,
        dataset_name=config.dataset_name,
        grid=grid,
        laplace_depth=solution.phi,
        equal_area_depth=equal_area.depth,
        streamlines=streamlines,
        contour_levels=config.contour_levels,
    )
    if not equal_area.column_summary.empty:
        column_summary_path = output_dir / "equivolumetric_column_summary.parquet"
        equal_area.column_summary.to_parquet(column_summary_path, index=False)
        paths["equivolumetric_column_summary"] = column_summary_path

    sdata_obj = sd.read_zarr(latest_path)
    table_summaries: dict[str, Any] = {}
    try:
        for table_config in config.tables:
            table_paths, summary = _annotate_table(
                sdata_obj=sdata_obj,
                table_config=table_config,
                config=config,
                output_dir=output_dir,
                solution=solution,
                equal_area_depth=equal_area,
                streamlines=streamlines,
            )
            paths.update(table_paths)
            table_summaries[table_config.segmentation] = summary
    finally:
        del sdata_obj
        force_release(note=f"after {config.dataset_name} cortical depth")

    summary_path = output_dir / "cortical_depth_qc_summary.json"
    summary = _build_qc_summary(
        config=config,
        solution_residual=solution.residual,
        streamlines=streamlines,
        table_summaries=table_summaries,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    paths["qc_summary"] = summary_path
    paths["latest_zarr"] = latest_path
    log_status(f"[{config.dataset_name}] Cortical-depth computation complete")
    return paths


def _annotate_table(
    *,
    sdata_obj: Any,
    table_config: CorticalDepthTableConfig,
    config: CorticalDepthConfig,
    output_dir: Path,
    solution: Any,
    equal_area_depth: Any,
    streamlines: list[Streamline],
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
    assignments = assign_cortical_depth_to_cells(
        coords,
        solution,
        equal_area_depth,
        streamlines,
        side_boundary_distance_um=config.side_boundary_distance_um,
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

    plot_cells_by_depth(
        segmentation_dir / f"{sample_stem}_cells_laplace_depth.png",
        cells,
        solution.grid,
        value_column="laplace_depth",
    )
    plot_cells_by_depth(
        segmentation_dir / f"{sample_stem}_cells_equivolumetric_depth.png",
        cells,
        solution.grid,
        value_column="equivolumetric_depth",
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
        {f"{table_config.segmentation}_cells": cells_path},
        summary,
    )


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
    solution_residual: float,
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
        "laplace_residual": float(solution_residual),
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
