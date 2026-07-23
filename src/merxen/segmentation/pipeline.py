"""High-level segmentation pipeline orchestration."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata_io import xenium as xenium_reader

from merxen.config import SegmentationConfig
from merxen.io.image_source import (
    MERSCOPE_ZPROJ_IMAGE_NAME,
    build_image_source,
    fetch_merscope_projected_tile,
    fetch_tile,
    list_plane_keys,
    prepare_merscope_plane_sources,
)
from merxen.io.spatialdata_io import (
    convert_to_latest_zarr,
    upgrade_spatialdata_contract,
)
from merxen.io.spatialdata_schema import choose_primary_points_key
from merxen.io.transcript_io import resolve_col, write_proseg_csv_from_points
from merxen.memory import force_release, log_status
from merxen.path_utils import remove_path, stage_existing_output
from merxen.segmentation.cellpose import (
    build_cellpose_affine_to_microns,
    run_tiled_cellpose,
    synchronize_cellpose_probability_logits,
)
from merxen.segmentation.mask_filter import filter_labeled_mask_by_area
from merxen.segmentation.proseg import run_proseg_refinement
from merxen.segmentation.proseg_hybrid import (
    has_proseg_hybrid_refinement,
    run_proseg_hybrid_refinement,
)

logger = logging.getLogger(__name__)


def _pixel_area_um2_from_affine(
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> float:
    """Return square microns per mask pixel from pixel-to-micron affine terms."""
    return abs(
        (float(x_transform[0]) * float(y_transform[1]))
        - (float(x_transform[1]) * float(y_transform[0]))
    )


def _load_merscope_transform_matrix(config: SegmentationConfig) -> np.ndarray:
    """Load the MERSCOPE micron-to-mosaic transform matrix."""
    dataset = config.dataset
    candidates: list[Path] = []
    if dataset.transform_path is not None:
        candidates.append(Path(dataset.transform_path))
    candidates.append(Path(dataset.data_path) / "micron_to_mosaic_pixel_transform.csv")

    for candidate in candidates:
        if not candidate.exists():
            continue
        matrix = np.loadtxt(candidate)
        if matrix.shape == (3, 3):
            return matrix
    raise FileNotFoundError(
        "Could not determine MERSCOPE transform. "
        "Set dataset.transform_path or include "
        "'micron_to_mosaic_pixel_transform.csv' in the SpatialData zarr."
    )


def _load_xenium_transform_matrix(config: SegmentationConfig) -> np.ndarray:
    """Load or derive Xenium micron-to-pixel transform matrix."""
    dataset = config.dataset
    candidates: list[Path] = []
    if dataset.xenium_spec_path is not None:
        candidates.append(Path(dataset.xenium_spec_path))
    if dataset.transform_path is not None:
        candidates.append(Path(dataset.transform_path))
    candidates.extend(
        [
            Path(dataset.data_path) / "experiment.xenium",
            Path(dataset.data_path) / "specs.json",
            Path(dataset.data_path) / "specs" / "specs.json",
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in {".txt", ".csv"}:
            mat = np.loadtxt(candidate)
            if mat.shape == (3, 3):
                return mat
        try:
            spec = json.loads(candidate.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "pixel_size" in spec:
            mpp = float(spec["pixel_size"])
            return np.array(
                [
                    [1.0 / mpp, 0.0, 0.0],
                    [0.0, 1.0 / mpp, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
    raise FileNotFoundError(
        "Could not determine Xenium transform. "
        "Set dataset.xenium_spec_path or dataset.transform_path."
    )


def _load_dataset_sdata(
    config: SegmentationConfig,
) -> tuple[Any, Any, int, int, np.ndarray, Any]:
    """Load source SpatialData and return tile-fetch context for segmentation."""
    dataset = config.dataset
    platform = dataset.platform.upper()

    if platform == "MERSCOPE":
        sdata = sd.read_zarr(dataset.data_path)
        matrix = _load_merscope_transform_matrix(config)

        if MERSCOPE_ZPROJ_IMAGE_NAME in sdata.images:
            source = build_image_source(
                sdata.images[MERSCOPE_ZPROJ_IMAGE_NAME],
                requested_channels=dataset.channels,
                as_float32=True,
            )
            height, width, _ = source["shape"]
            log_status(
                f"[{dataset.name}] MERSCOPE projection image shape={height}x{width}, "
                f"channels={source['channels']}"
            )

            def fetch_tile_fn(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
                return fetch_tile(source, y0, y1, x0, x1)

            points_key = choose_primary_points_key(sdata)
            if points_key is None:
                raise RuntimeError(f"[{dataset.name}] No transcript points found.")
            return (
                sdata,
                fetch_tile_fn,
                height,
                width,
                matrix,
                sdata.points[points_key],
            )

        plane_keys = list_plane_keys(sdata.images, prefix=dataset.image_prefix)
        if dataset.z_range is None:
            selected_keys = [key for _, key in plane_keys]
        else:
            z0, z1 = dataset.z_range
            selected_keys = [key for z, key in plane_keys if z0 <= z <= z1]
        if not selected_keys:
            raise ValueError(
                f"[{dataset.name}] No MERSCOPE image planes selected. "
                "Check image_prefix and z_range."
            )

        plane_sources, height, width, use_channels = prepare_merscope_plane_sources(
            sdata,
            selected_keys=selected_keys,
            requested_channels=dataset.channels,
        )
        log_status(
            f"[{dataset.name}] MERSCOPE image shape={height}x{width}, "
            f"channels={use_channels}"
        )

        def fetch_tile_fn(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
            return fetch_merscope_projected_tile(plane_sources, y0, y1, x0, x1)

        points_key = choose_primary_points_key(sdata)
        if points_key is None:
            raise RuntimeError(f"[{dataset.name}] No transcript points found.")
        return sdata, fetch_tile_fn, height, width, matrix, sdata.points[points_key]

    if platform == "XENIUM":
        if Path(dataset.data_path).suffix == ".zarr":
            sdata = sd.read_zarr(dataset.data_path)
        else:
            sdata = xenium_reader(
                dataset.data_path,
                cells_table=False,
                cells_as_circles=False,
                cells_boundaries=False,
                nucleus_boundaries=False,
                cells_labels=False,
                nucleus_labels=False,
                transcripts=True,
                morphology_focus=True,
                morphology_mip=False,
                aligned_images=False,
            )
        matrix = _load_xenium_transform_matrix(config)
        if len(sdata.images) == 0:
            raise RuntimeError(f"[{dataset.name}] No Xenium images found.")
        image_key = list(sdata.images.keys())[0]
        source = build_image_source(
            sdata.images[image_key],
            requested_channels=dataset.channels,
            as_float32=True,
        )
        height, width, _ = source["shape"]
        log_status(
            f"[{dataset.name}] Xenium image='{image_key}' shape={height}x{width}, "
            f"channels={source['channels']}"
        )

        def fetch_tile_fn(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
            return fetch_tile(source, y0, y1, x0, x1)

        points_key = choose_primary_points_key(sdata)
        if points_key is None:
            raise RuntimeError(f"[{dataset.name}] No transcript points found.")
        return sdata, fetch_tile_fn, height, width, matrix, sdata.points[points_key]

    raise ValueError(f"Unsupported platform: {dataset.platform}")


def _write_progress(path: Path, data: dict) -> None:
    """Write progress JSON; best-effort, never raises."""
    try:
        data["updated_at"] = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        path.write_text(json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001
        pass


def run_cellpose_segmentation(
    config: SegmentationConfig,
    *,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Run Cellpose and prepare its mask-derived inputs for ProSeg."""
    dataset = config.dataset
    out_dir = Path(dataset.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_transcripts_csv = out_dir / "transcripts_for_proseg.csv"
    staged_mask_path = out_dir / "cellpose_masks_tiled.npy"
    staged_cellprob_path = out_dir / "cellpose_cellprobs_tiled.npy"
    staged_stitching_stats_path = out_dir / "cellpose_stitching_stats.json"
    transforms_path = out_dir / "cellpose_transforms.json"
    persistent_transcripts_csv = (
        Path(dataset.persistent_transcripts_path)
        if dataset.persistent_transcripts_path is not None
        else None
    )
    persistent_mask_path = (
        Path(dataset.persistent_mask_path)
        if dataset.persistent_mask_path is not None
        else None
    )
    persistent_cellprob_path = (
        Path(dataset.persistent_cellpose_cellprob_path)
        if dataset.persistent_cellpose_cellprob_path is not None
        else None
    )
    persistent_stitching_stats_path = (
        Path(dataset.persistent_cellpose_stitching_stats_path)
        if dataset.persistent_cellpose_stitching_stats_path is not None
        else None
    )
    transcripts_csv = persistent_transcripts_csv or staged_transcripts_csv
    mask_path = persistent_mask_path or staged_mask_path
    cellprob_path = persistent_cellprob_path or staged_cellprob_path
    stitching_stats_path = (
        persistent_stitching_stats_path or staged_stitching_stats_path
    )
    progress_path = out_dir / "cellpose_progress.json"
    _started_at = time.monotonic()

    def _stage_outputs() -> tuple[Path, Path, Path, Path, Path]:
        if transcripts_csv != staged_transcripts_csv:
            stage_existing_output(transcripts_csv, staged_transcripts_csv)
        if mask_path != staged_mask_path:
            stage_existing_output(mask_path, staged_mask_path)
        if cellprob_path != staged_cellprob_path:
            stage_existing_output(cellprob_path, staged_cellprob_path)
        if (
            stitching_stats_path.exists()
            and stitching_stats_path != staged_stitching_stats_path
        ):
            stage_existing_output(
                stitching_stats_path,
                staged_stitching_stats_path,
            )
        return (
            staged_transcripts_csv,
            staged_mask_path,
            staged_cellprob_path,
            staged_stitching_stats_path,
            transforms_path,
        )

    def _progress(stage: str, **extra: object) -> None:
        _write_progress(
            progress_path,
            {
                "dataset": dataset.name,
                "stage": stage,
                "elapsed_min": round((time.monotonic() - _started_at) / 60, 1),
                **extra,
            },
        )

    if (
        transcripts_csv.exists()
        and mask_path.exists()
        and cellprob_path.exists()
        and transforms_path.exists()
        and not force_rerun
    ):
        log_status(f"[{dataset.name}] Reusing existing Cellpose outputs")
        if not stitching_stats_path.exists():
            stitching_stats_path.parent.mkdir(parents=True, exist_ok=True)
            stitching_stats_path.write_text(
                json.dumps({"write_stitching_stats": False}, indent=2) + "\n"
            )
        (
            staged_transcripts,
            staged_mask,
            staged_cellprob,
            staged_stats,
            staged_transforms,
        ) = _stage_outputs()
        return {
            "transcripts_csv": staged_transcripts,
            "cellpose_mask_path": staged_mask,
            "cellpose_cellprob_path": staged_cellprob,
            "stitching_stats_path": staged_stats,
            "transforms_path": staged_transforms,
        }

    sdata, fetch_tile_fn, height, width, matrix, points_obj = _load_dataset_sdata(
        config
    )

    _progress("cellpose_starting")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path = run_tiled_cellpose(
        fetch_tile_fn=fetch_tile_fn,
        height=height,
        width=width,
        dataset_name=dataset.name,
        output_mask_path=mask_path,
        output_cellprob_path=cellprob_path,
        cellpose_config=config.cellpose,
        mask_filter_config=config.mask_filter,
        tiling_config=config.tiling,
        memory_config=config.memory,
        output_stitching_stats_path=stitching_stats_path,
        progress_callback=_progress,
    )
    _progress(
        "cellpose_done",
        stitching_stats_path=(
            str(stitching_stats_path) if stitching_stats_path.exists() else None
        ),
    )

    x_transform, y_transform = build_cellpose_affine_to_microns(
        matrix,
        scale_factor=1.0,
        x0=0.0,
        y0=0.0,
    )
    log_status(f"[{dataset.name}] mask->micron x_transform={x_transform}")
    log_status(f"[{dataset.name}] mask->micron y_transform={y_transform}")

    if (
        config.mask_filter.final_min_area_um2 is not None
        or config.mask_filter.final_max_area_um2 is not None
    ):
        pixel_area_um2 = _pixel_area_um2_from_affine(x_transform, y_transform)
        _progress(
            "cellpose_area_filtering",
            min_area_um2=config.mask_filter.final_min_area_um2,
            max_area_um2=config.mask_filter.final_max_area_um2,
            pixel_area_um2=pixel_area_um2,
        )
        filter_stats = filter_labeled_mask_by_area(
            mask_path,
            pixel_area_um2=pixel_area_um2,
            min_area_um2=config.mask_filter.final_min_area_um2,
            max_area_um2=config.mask_filter.final_max_area_um2,
            chunk_mb=config.mask_filter.final_filter_chunk_mb,
            show_progress=config.mask_filter.show_progress,
        )
        log_status(
            f"[{dataset.name}] Cellpose area filter kept "
            f"{filter_stats['n_kept']:,}/{filter_stats['n_labels']:,} masks "
            f"(removed small={filter_stats['n_removed_small']:,}, "
            f"large={filter_stats['n_removed_large']:,}; "
            f"bounds={config.mask_filter.final_min_area_um2}-"
            f"{config.mask_filter.final_max_area_um2} um2)"
        )
        _progress("cellpose_area_filtered", **filter_stats)
        force_release(note=f"after {dataset.name} Cellpose area filtering")

    synchronize_cellpose_probability_logits(mask_path, cellprob_path)

    x_col = resolve_col(points_obj, ["x", "global_x", "x_location"])
    y_col = resolve_col(points_obj, ["y", "global_y", "y_location"])
    z_col = resolve_col(points_obj, ["z", "global_z", "z_location"], required=False)
    gene_col = resolve_col(points_obj, ["gene", "feature_name", "target"])
    if x_col is None or y_col is None or gene_col is None:
        raise KeyError(
            "Could not resolve required points columns for ProSeg CSV export."
        )

    platform = dataset.platform.upper()
    qv_col = None
    min_qv = None
    excluded_gene_pattern = None
    if platform == "XENIUM":
        qv_col = resolve_col(
            points_obj, ["qv", "quality", "quality_value"], required=False
        )
        min_qv = dataset.min_qv
        excluded_gene_pattern = r"^(Deprecated|NegControl|Unassigned|Intergenic)"
    elif platform == "MERSCOPE":
        qv_col = resolve_col(
            points_obj,
            ["transcript_score", "qv", "quality", "quality_value"],
            required=False,
        )

    transcripts_csv.parent.mkdir(parents=True, exist_ok=True)
    mask_mmap = np.load(mask_path, mmap_mode="r")
    prep_stats = write_proseg_csv_from_points(
        points_obj=points_obj,
        csv_path=transcripts_csv,
        masks=mask_mmap,
        x_transform=x_transform,
        y_transform=y_transform,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        gene_col=gene_col,
        qv_col=qv_col,
        min_qv=min_qv,
        excluded_gene_pattern=excluded_gene_pattern,
        chunk_rows=config.memory.transcript_chunk_rows,
        dataset_name=dataset.name,
        status_every_chunks=config.memory.transcript_status_every_chunks,
        memory_check_every_chunks=config.memory.memory_check_every_chunks,
        max_ram_gb=config.memory.max_system_ram_gb,
        warn_ram_gb=config.memory.memory_warn_gb,
    )
    log_status(
        f"[{dataset.name}] Seeded transcripts for ProSeg: "
        f"{prep_stats['n_seeded']:,} ({prep_stats['pct_seeded']:.2f}%)"
    )

    transforms_path.write_text(
        json.dumps(
            {
                "x_transform": list(x_transform),
                "y_transform": list(y_transform),
            },
            indent=2,
        )
        + "\n"
    )
    if not stitching_stats_path.exists():
        stitching_stats_path.write_text(
            json.dumps({"write_stitching_stats": False}, indent=2) + "\n"
        )

    del mask_mmap, points_obj, sdata
    force_release(note=f"after {dataset.name} Cellpose segmentation")
    _progress("done")
    (
        staged_transcripts,
        staged_mask,
        staged_cellprob,
        staged_stats,
        staged_transforms,
    ) = _stage_outputs()
    return {
        "transcripts_csv": Path(staged_transcripts),
        "cellpose_mask_path": Path(staged_mask),
        "cellpose_cellprob_path": Path(staged_cellprob),
        "stitching_stats_path": Path(staged_stats),
        "transforms_path": Path(staged_transforms),
    }


def run_cellpose_nuclei_segmentation(
    config: SegmentationConfig,
    *,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Run an independently reusable DAPI-only Cellpose nuclei segmentation."""
    dataset = config.dataset
    out_dir = Path(dataset.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_mask_path = out_dir / "cellpose_nuclei_masks_tiled.npy"
    staged_stats_path = out_dir / "cellpose_nuclei_stitching_stats.json"
    persistent_mask_path = (
        Path(dataset.persistent_nuclei_mask_path)
        if dataset.persistent_nuclei_mask_path is not None
        else None
    )
    persistent_stats_path = (
        Path(dataset.persistent_nuclei_stitching_stats_path)
        if dataset.persistent_nuclei_stitching_stats_path is not None
        else None
    )
    mask_path = persistent_mask_path or staged_mask_path
    stats_path = persistent_stats_path or staged_stats_path
    progress_path = out_dir / "cellpose_nuclei_progress.json"
    started_at = time.monotonic()

    def _stage_outputs() -> tuple[Path, Path]:
        if mask_path != staged_mask_path:
            stage_existing_output(mask_path, staged_mask_path)
        if stats_path.exists() and stats_path != staged_stats_path:
            stage_existing_output(stats_path, staged_stats_path)
        return staged_mask_path, staged_stats_path

    def _progress(stage: str, **extra: object) -> None:
        _write_progress(
            progress_path,
            {
                "dataset": dataset.name,
                "stage": stage,
                "elapsed_min": round((time.monotonic() - started_at) / 60, 1),
                **extra,
            },
        )

    if mask_path.exists() and not force_rerun:
        log_status(f"[{dataset.name}] Reusing existing DAPI-only nuclei mask")
        if not stats_path.exists():
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(
                json.dumps({"write_stitching_stats": False}, indent=2) + "\n"
            )
        staged_mask, staged_stats = _stage_outputs()
        return {
            "nuclei_mask_path": staged_mask,
            "nuclei_stitching_stats_path": staged_stats,
        }

    nuclei_config = config.model_copy(deep=True)
    nuclei_config.dataset.channels = ["DAPI"]
    nuclei_config.cellpose = config.cellpose.model_copy(
        update={"model_type": config.nuclei_cellpose.model_type}
    )
    nuclei_config.mask_filter = config.nuclei_mask_filter

    sdata, fetch_tile_fn, height, width, matrix, points_obj = _load_dataset_sdata(
        nuclei_config
    )
    _progress("cellpose_nuclei_starting")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    run_tiled_cellpose(
        fetch_tile_fn=fetch_tile_fn,
        height=height,
        width=width,
        dataset_name=f"{dataset.name}:nuclei",
        output_mask_path=mask_path,
        cellpose_config=nuclei_config.cellpose,
        mask_filter_config=nuclei_config.mask_filter,
        tiling_config=nuclei_config.tiling,
        memory_config=nuclei_config.memory,
        output_stitching_stats_path=stats_path,
        progress_callback=_progress,
    )

    x_transform, y_transform = build_cellpose_affine_to_microns(
        matrix,
        scale_factor=1.0,
        x0=0.0,
        y0=0.0,
    )
    if (
        nuclei_config.mask_filter.final_min_area_um2 is not None
        or nuclei_config.mask_filter.final_max_area_um2 is not None
    ):
        filter_stats = filter_labeled_mask_by_area(
            mask_path,
            pixel_area_um2=_pixel_area_um2_from_affine(x_transform, y_transform),
            min_area_um2=nuclei_config.mask_filter.final_min_area_um2,
            max_area_um2=nuclei_config.mask_filter.final_max_area_um2,
            chunk_mb=nuclei_config.mask_filter.final_filter_chunk_mb,
            show_progress=nuclei_config.mask_filter.show_progress,
        )
        _progress("cellpose_nuclei_area_filtered", **filter_stats)
        log_status(
            f"[{dataset.name}] Nuclei area filter kept "
            f"{filter_stats['n_kept']:,}/{filter_stats['n_labels']:,} masks"
        )

    if not stats_path.exists():
        stats_path.write_text(
            json.dumps({"write_stitching_stats": False}, indent=2) + "\n"
        )
    del points_obj, sdata
    force_release(note=f"after {dataset.name} DAPI-only nuclei segmentation")
    _progress("done")
    staged_mask, staged_stats = _stage_outputs()
    return {
        "nuclei_mask_path": staged_mask,
        "nuclei_stitching_stats_path": staged_stats,
    }


def run_proseg_segmentation(
    config: SegmentationConfig,
    *,
    transcripts_csv: str | Path,
    cellpose_mask_path: str | Path,
    cellpose_cellprob_path: str | Path,
    transforms_path: str | Path,
    proseg_binary: str | Path | None = None,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Refine a prepared Cellpose prior with the CPU-only ProSeg tool."""
    dataset = config.dataset
    out_dir = Path(dataset.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_output = out_dir / "proseg_base_raw.zarr"
    staged_latest_output = out_dir / "proseg_base_latest.zarr"
    persistent_latest_output = (
        Path(dataset.persistent_latest_zarr_path)
        if dataset.persistent_latest_zarr_path is not None
        else None
    )
    latest_output = persistent_latest_output or staged_latest_output
    transcripts_csv = Path(transcripts_csv)
    cellpose_mask_path = Path(cellpose_mask_path)
    cellpose_cellprob_path = Path(cellpose_cellprob_path)
    transforms_path = Path(transforms_path)
    progress_path = out_dir / "proseg_progress.json"
    started_at = time.monotonic()

    def _progress(stage: str, **extra: object) -> None:
        _write_progress(
            progress_path,
            {
                "dataset": dataset.name,
                "stage": stage,
                "elapsed_min": round((time.monotonic() - started_at) / 60, 1),
                **extra,
            },
        )

    def _stage_latest() -> Path:
        if latest_output != staged_latest_output:
            stage_existing_output(latest_output, staged_latest_output)
        return staged_latest_output

    def _ensure_hybrid() -> None:
        is_spatialdata_store = (latest_output / "zarr.json").exists() or (
            latest_output / ".zgroup"
        ).exists()
        upgraded = (
            upgrade_spatialdata_contract(
                latest_output,
                quality_column_alias=(
                    "transcript_score"
                    if dataset.platform.upper() == "MERSCOPE"
                    else None
                ),
                platform=dataset.platform,
            )
            if is_spatialdata_store
            else False
        )
        if upgraded:
            _progress("spatialdata_schema_upgraded")
        if not config.proseg_hybrid.enabled:
            return
        if not force_rerun and has_proseg_hybrid_refinement(
            latest_output,
            config.proseg_hybrid,
        ):
            return
        _progress("proseg_hybrid_starting")
        summary = run_proseg_hybrid_refinement(
            latest_output,
            cellpose_mask_path,
            transforms_path,
            config.proseg_hybrid,
        )
        _progress("proseg_hybrid_done", **summary)

    if latest_output.exists() and not force_rerun:
        log_status(f"[{dataset.name}] Reusing existing latest output: {latest_output}")
        _ensure_hybrid()
        return {"latest_output": _stage_latest()}

    if raw_output.exists() and not force_rerun:
        latest_output.parent.mkdir(parents=True, exist_ok=True)
        convert_to_latest_zarr(
            raw_output,
            latest_output,
            quality_column_alias=(
                "transcript_score" if dataset.platform.upper() == "MERSCOPE" else None
            ),
            platform=dataset.platform,
        )
        remove_path(raw_output)
        _ensure_hybrid()
        return {"latest_output": _stage_latest()}

    transforms = json.loads(transforms_path.read_text())
    x_transform = tuple(float(value) for value in transforms["x_transform"])
    y_transform = tuple(float(value) for value in transforms["y_transform"])
    if len(x_transform) != 3 or len(y_transform) != 3:
        raise ValueError("Cellpose transform metadata must contain 3-value affines.")

    _progress("proseg_starting")
    proseg_params = config.proseg.model_dump()
    proseg_params.update(dataset.proseg_overrides)
    if proseg_binary is not None:
        proseg_params["binary_path"] = Path(proseg_binary)
    transcript_columns = set(pd.read_csv(transcripts_csv, nrows=0).columns)

    raw_out = run_proseg_refinement(
        transcripts_df=transcripts_csv,
        output_path=raw_output,
        proseg_binary=proseg_params["binary_path"],
        x_col="x_micron",
        y_col="y_micron",
        z_col="z_micron",
        gene_col="feature_name",
        cell_id_col="cell_id",
        transcript_id_col="transcript_id",
        qv_col="qv" if "qv" in transcript_columns else None,
        samples=int(proseg_params["samples"]),
        burnin_voxel_size=proseg_params.get("burnin_voxel_size"),
        voxel_size=float(proseg_params["voxel_size"]),
        voxel_layers=int(proseg_params["voxel_layers"]),
        nuclear_reassignment_prob=float(proseg_params["nuclear_reassignment_prob"]),
        diffusion_probability=float(proseg_params["diffusion_probability"]),
        cell_compactness=proseg_params.get("cell_compactness"),
        expand_initialized_cells=proseg_params.get("expand_initialized_cells"),
        use_cell_initialization=bool(
            proseg_params.get("use_cell_initialization", False)
        ),
        prior_seg_reassignment_prob=proseg_params.get("prior_seg_reassignment_prob"),
        max_transcript_nucleus_distance=proseg_params.get(
            "max_transcript_nucleus_distance"
        ),
        diffusion_sigma_far=proseg_params.get("diffusion_sigma_far"),
        cellpose_masks=cellpose_mask_path,
        cellpose_cellprobs=cellpose_cellprob_path,
        cellpose_x_transform=x_transform,
        cellpose_y_transform=y_transform,
        num_threads=int(proseg_params.get("num_threads", 12)),
        overwrite=True,
        progress_callback=_progress,
        proseg_samples=int(proseg_params["samples"]),
    )

    latest_output.parent.mkdir(parents=True, exist_ok=True)
    latest_out = convert_to_latest_zarr(
        raw_out,
        latest_output,
        quality_column_alias=(
            "transcript_score" if dataset.platform.upper() == "MERSCOPE" else None
        ),
        platform=dataset.platform,
    )
    remove_path(raw_out)
    _ensure_hybrid()
    staged_out = _stage_latest()
    log_status(f"[{dataset.name}] Wrote latest output: {latest_out}")
    force_release(note=f"after {dataset.name} ProSeg refinement")
    _progress("done")

    return {"latest_output": Path(staged_out)}


def run_segmentation_pipeline(
    config: SegmentationConfig,
    *,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Run Cellpose then ProSeg in one process for CLI compatibility."""
    dataset = config.dataset
    out_dir = Path(dataset.output_dir)
    staged_latest_output = out_dir / "proseg_base_latest.zarr"
    staged_transcripts_csv = out_dir / "transcripts_for_proseg.csv"
    staged_mask_path = out_dir / "cellpose_masks_tiled.npy"
    staged_cellprob_path = out_dir / "cellpose_cellprobs_tiled.npy"
    staged_transforms_path = out_dir / "cellpose_transforms.json"
    staged_nuclei_mask_path = out_dir / "cellpose_nuclei_masks_tiled.npy"
    persistent_latest_output = (
        Path(dataset.persistent_latest_zarr_path)
        if dataset.persistent_latest_zarr_path is not None
        else staged_latest_output
    )
    reusable_transcripts = (
        Path(dataset.persistent_transcripts_path)
        if dataset.persistent_transcripts_path
        else staged_transcripts_csv
    )
    reusable_mask = (
        Path(dataset.persistent_mask_path)
        if dataset.persistent_mask_path
        else staged_mask_path
    )
    reusable_cellprob = (
        Path(dataset.persistent_cellpose_cellprob_path)
        if dataset.persistent_cellpose_cellprob_path
        else staged_cellprob_path
    )
    reusable_nuclei_mask = (
        Path(dataset.persistent_nuclei_mask_path)
        if dataset.persistent_nuclei_mask_path
        else staged_nuclei_mask_path
    )
    hybrid_complete = not config.proseg_hybrid.enabled or has_proseg_hybrid_refinement(
        persistent_latest_output,
        config.proseg_hybrid,
    )

    if (
        persistent_latest_output.exists()
        and reusable_transcripts.exists()
        and reusable_mask.exists()
        and reusable_cellprob.exists()
        and reusable_nuclei_mask.exists()
        and (hybrid_complete or staged_transforms_path.exists())
        and not force_rerun
    ):
        if not hybrid_complete:
            run_proseg_hybrid_refinement(
                persistent_latest_output,
                reusable_mask,
                staged_transforms_path,
                config.proseg_hybrid,
            )
        if persistent_latest_output != staged_latest_output:
            stage_existing_output(persistent_latest_output, staged_latest_output)
        transcripts = reusable_transcripts
        mask = reusable_mask
        if transcripts != staged_transcripts_csv:
            stage_existing_output(transcripts, staged_transcripts_csv)
        if mask != staged_mask_path:
            stage_existing_output(mask, staged_mask_path)
        if reusable_cellprob != staged_cellprob_path:
            stage_existing_output(reusable_cellprob, staged_cellprob_path)
        if reusable_nuclei_mask != staged_nuclei_mask_path:
            stage_existing_output(reusable_nuclei_mask, staged_nuclei_mask_path)
        return {
            "latest_output": staged_latest_output,
            "transcripts_csv": staged_transcripts_csv,
            "cellpose_mask_path": staged_mask_path,
            "cellpose_cellprob_path": staged_cellprob_path,
            "nuclei_mask_path": staged_nuclei_mask_path,
        }

    cellpose_outputs = run_cellpose_segmentation(
        config,
        force_rerun=force_rerun,
    )
    nuclei_outputs = run_cellpose_nuclei_segmentation(
        config,
        force_rerun=force_rerun,
    )
    proseg_outputs = run_proseg_segmentation(
        config,
        transcripts_csv=cellpose_outputs["transcripts_csv"],
        cellpose_mask_path=cellpose_outputs["cellpose_mask_path"],
        cellpose_cellprob_path=cellpose_outputs["cellpose_cellprob_path"],
        transforms_path=cellpose_outputs["transforms_path"],
        force_rerun=force_rerun,
    )

    return {
        "latest_output": proseg_outputs["latest_output"],
        "transcripts_csv": cellpose_outputs["transcripts_csv"],
        "cellpose_mask_path": cellpose_outputs["cellpose_mask_path"],
        "cellpose_cellprob_path": cellpose_outputs["cellpose_cellprob_path"],
        "nuclei_mask_path": nuclei_outputs["nuclei_mask_path"],
    }
