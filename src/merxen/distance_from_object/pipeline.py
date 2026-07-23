"""Pipeline entry points for distance-from-object analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import spatialdata as sd
from spatialdata.models import TableModel

from merxen.config import (
    DistanceFromObjectCohortConfig,
    DistanceFromObjectConfig,
    DistanceFromObjectTableConfig,
)
from merxen.cortical_depth.assign_cells import extract_cell_coordinates
from merxen.distance_from_object.annotations import (
    ObjectAnnotation,
    load_object_annotations,
    write_object_annotations,
)
from merxen.distance_from_object.distances import (
    apply_distance_columns,
    assign_distances_to_objects,
)
from merxen.distance_from_object.plotting import (
    plot_cell_distances,
    plot_proximity_counts,
    plot_volcano,
)
from merxen.distance_from_object.pseudobulk import (
    build_pair_pseudobulk,
    combine_pair_pseudobulks,
    retain_complete_pairs,
    run_paired_differential_expression,
)
from merxen.io.spatialdata_io import write_or_replace_element
from merxen.memory import force_release, log_status


def run_distance_from_object(config: DistanceFromObjectConfig) -> dict[str, Path]:
    """Annotate one platform zarr and write pair-level pseudobulk samples."""
    latest_path = Path(config.latest_zarr_path)
    output_dir = Path(config.output_dir)
    if not latest_path.exists():
        raise FileNotFoundError(f"[{config.dataset_name}] Missing zarr: {latest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_object_annotations(
        config.object_annotation_path,
        object_types=config.object_types,
    )
    normalized_annotations_path = write_object_annotations(
        output_dir / "registered_object_annotations.geojson",
        annotations,
    )
    paths: dict[str, Path] = {
        "registered_object_annotations": normalized_annotations_path,
        "latest_zarr": latest_path,
    }
    summaries: dict[str, Any] = {}

    log_status(f"[{config.dataset_name}] Loading SpatialData for object distances")
    sdata_obj = sd.read_zarr(latest_path)
    try:
        for table_config in config.tables:
            table_paths, summary = _annotate_table(
                sdata_obj=sdata_obj,
                table_config=table_config,
                config=config,
                annotations=annotations,
                output_dir=output_dir,
            )
            paths.update(table_paths)
            summaries[table_config.segmentation] = summary
    finally:
        del sdata_obj
        force_release(note=f"after {config.dataset_name} distance from object")

    summary_path = output_dir / "distance_from_object_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "pair_id": config.pair_id,
                "dataset_name": config.dataset_name,
                "platform": config.platform,
                "object_count": len(annotations),
                "object_type_counts": _object_type_counts(annotations),
                "near_distance_um": config.near_distance_um,
                "far_distance_um": config.far_distance_um,
                "max_distance_um": config.max_distance_um,
                "included_tissue_annotations": config.included_tissue_annotations,
                "tables": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths["summary"] = summary_path
    log_status(f"[{config.dataset_name}] Distance-from-object annotation complete")
    return paths


def run_distance_from_object_cohort(
    config: DistanceFromObjectCohortConfig,
) -> dict[str, Path]:
    """Run paired PyDESeq2 near-vs-far analyses across tissue-block pair IDs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for segmentation in config.segmentations:
        segmentation_dir = output_dir / segmentation
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        pair_paths = _find_pair_pseudobulk_paths(
            config.annotation_output_dirs,
            segmentation,
        )
        if not pair_paths:
            summaries[segmentation] = {
                "status": "skipped",
                "reason": "no_pair_pseudobulk_files",
            }
            continue
        try:
            combined = combine_pair_pseudobulks(pair_paths)
        except ValueError as exc:
            summaries[segmentation] = {
                "status": "skipped",
                "reason": str(exc),
            }
            continue
        paired, complete_pairs = retain_complete_pairs(combined)
        combined_path = segmentation_dir / "paired_pseudobulk_counts.h5ad"
        paired.write_h5ad(combined_path)
        paths[f"{segmentation}_pseudobulk"] = combined_path
        sample_metadata_path = segmentation_dir / "paired_pseudobulk_samples.csv"
        paired.obs.reset_index().to_csv(sample_metadata_path, index=False)
        paths[f"{segmentation}_samples"] = sample_metadata_path

        if len(complete_pairs) < config.min_pairs:
            summaries[segmentation] = {
                "status": "skipped",
                "reason": "insufficient_complete_pairs",
                "complete_pair_ids": complete_pairs,
                "required_pairs": config.min_pairs,
                "pseudobulk_samples": int(paired.n_obs),
                "genes": int(paired.n_vars),
            }
            continue
        log_status(
            f"[{config.platform}:{segmentation}] Running paired PyDESeq2 for "
            f"{len(complete_pairs)} tissue blocks"
        )
        results = run_paired_differential_expression(
            paired,
            n_cpus=config.n_cpus,
        )
        results_path = segmentation_dir / "near_vs_far_differential_expression.csv"
        results.reset_index().to_csv(results_path, index=False)
        results_parquet_path = results_path.with_suffix(".parquet")
        results.reset_index().to_parquet(results_parquet_path, index=False)
        volcano_path = segmentation_dir / "near_vs_far_volcano.png"
        plot_volcano(volcano_path, results)
        paths[f"{segmentation}_differential_expression"] = results_path
        paths[f"{segmentation}_differential_expression_parquet"] = results_parquet_path
        paths[f"{segmentation}_volcano"] = volcano_path
        summaries[segmentation] = {
            "status": "complete",
            "complete_pair_ids": complete_pairs,
            "pseudobulk_samples": int(paired.n_obs),
            "genes_tested": int(len(results)),
            "significant_genes_padj_0_05": int(
                pd.to_numeric(results["padj"], errors="coerce").lt(0.05).sum()
            ),
        }

    summary_path = output_dir / "distance_from_object_cohort_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "platform": config.platform,
                "design": "~ pair_id + proximity",
                "contrast": "near_vs_far",
                "segmentations": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths["summary"] = summary_path
    return paths


def _annotate_table(
    *,
    sdata_obj: Any,
    table_config: DistanceFromObjectTableConfig,
    config: DistanceFromObjectConfig,
    annotations: list[ObjectAnnotation],
    output_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    table_key = table_config.table_key
    if table_key not in sdata_obj.tables:
        raise KeyError(
            f"[{config.dataset_name}] table_key={table_key!r} not found. "
            f"Available tables: {list(sdata_obj.tables.keys())}"
        )
    table = sdata_obj.tables[table_key]
    tissue_column = config.tissue_annotation_column
    if tissue_column not in table.obs.columns:
        raise KeyError(
            f"[{config.dataset_name}:{table_config.segmentation}] Missing "
            f"{tissue_column!r} in {table_key!r}. Run compute_cortical_depth "
            "for this table before distance_from_object."
        )
    shape_key = _resolve_shape_key(
        sdata_obj,
        table=table,
        requested=table_config.shape_key,
        platform=config.platform,
    )
    coordinates = extract_cell_coordinates(
        table,
        sdata_obj=sdata_obj,
        shape_key=shape_key,
    )
    assignments = assign_distances_to_objects(
        coordinates,
        annotations,
        coordinate_unit_um=config.coordinate_unit_um,
        near_distance_um=config.near_distance_um,
        far_distance_um=config.far_distance_um,
        max_distance_um=config.max_distance_um,
    )
    assignments["cortical_depth_annotation"] = (
        table.obs[tissue_column].astype(str).to_numpy()
    )

    segmentation_dir = output_dir / table_config.segmentation
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    cells = assignments.copy()
    cells.insert(0, "cell_id", cells.index.astype(str))
    cells.insert(1, "x", coordinates.coordinates[:, 0])
    cells.insert(2, "y", coordinates.coordinates[:, 1])
    cells_path = segmentation_dir / "cells_with_object_distance.parquet"
    cells.to_parquet(cells_path, index=False)

    pseudobulk = build_pair_pseudobulk(
        table,
        assignments,
        pair_id=config.pair_id,
        included_tissue_annotations=config.included_tissue_annotations,
        tissue_annotation_column="cortical_depth_annotation",
        min_cells=config.min_cells_per_pseudobulk,
    )
    pseudobulk_path = segmentation_dir / "pseudobulk_counts.h5ad"
    pseudobulk.write_h5ad(pseudobulk_path)
    pseudobulk_samples_path = segmentation_dir / "pseudobulk_samples.csv"
    pseudobulk.obs.reset_index().to_csv(pseudobulk_samples_path, index=False)

    distance_plot_path = segmentation_dir / "cells_object_distance.png"
    plot_cell_distances(
        distance_plot_path,
        cells,
        annotations,
        max_distance_um=config.max_distance_um,
    )
    count_plot_path = segmentation_dir / "cells_object_proximity_counts.png"
    plot_proximity_counts(count_plot_path, cells)

    if config.write_spatialdata_table:
        updated = apply_distance_columns(table, assignments)
        parsed = _parse_table_for_spatialdata(
            updated,
            source_table=table,
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

    proximity_counts = {
        str(key): int(value)
        for key, value in assignments["object_proximity"]
        .fillna("missing")
        .value_counts()
        .items()
    }
    tissue_counts = {
        str(key): int(value)
        for key, value in assignments["cortical_depth_annotation"]
        .fillna("missing")
        .value_counts()
        .items()
    }
    paths = {
        f"{table_config.segmentation}_cells": cells_path,
        f"{table_config.segmentation}_pseudobulk": pseudobulk_path,
        f"{table_config.segmentation}_pseudobulk_samples": (pseudobulk_samples_path),
        f"{table_config.segmentation}_distance_plot": distance_plot_path,
        f"{table_config.segmentation}_count_plot": count_plot_path,
    }
    summary = {
        "table_key": table_key,
        "shape_key": shape_key,
        "coordinate_source": coordinates.source,
        "n_cells": int(len(assignments)),
        "n_inside_object": int(assignments["inside_object"].sum()),
        "proximity_counts": proximity_counts,
        "tissue_annotation_counts": tissue_counts,
        "pseudobulk_groups": int(pseudobulk.n_obs),
        "pseudobulk_cell_counts": {
            str(row.proximity): int(row.cell_count)
            for row in pseudobulk.obs.itertuples()
        },
    }
    return paths, summary


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
        instance_key = "instance_id" if "instance_id" in out.obs.columns else "cell_id"
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


def _region_from_attrs(attrs: dict[str, Any]) -> str | None:
    region = attrs.get("region")
    if isinstance(region, str):
        return region
    if isinstance(region, list | tuple) and region:
        return str(region[0])
    return None


def _object_type_counts(
    annotations: list[ObjectAnnotation],
) -> dict[str, int]:
    object_types = [annotation.object_type for annotation in annotations]
    counts = pd.Series(object_types).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _find_pair_pseudobulk_paths(
    roots: list[Path],
    segmentation: str,
) -> list[Path]:
    paths: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        direct = root / segmentation / "pseudobulk_counts.h5ad"
        if direct.exists():
            paths.append(direct)
            continue
        paths.extend(
            candidate
            for candidate in root.rglob("pseudobulk_counts.h5ad")
            if candidate.parent.name == segmentation
        )
    return sorted(set(paths))
