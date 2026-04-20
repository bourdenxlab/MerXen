"""High-level segmentation pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import spatialdata as sd
from spatialdata_io import xenium as xenium_reader

from merxen.config import SegmentationConfig
from merxen.io.image_source import (
    build_image_source,
    fetch_merscope_projected_tile,
    fetch_tile,
    list_plane_keys,
    prepare_merscope_plane_sources,
)
from merxen.io.spatialdata_io import convert_to_latest_zarr
from merxen.io.transcript_io import resolve_col, write_proseg_csv_from_points
from merxen.memory import force_release, log_status
from merxen.segmentation.cellpose import (
    build_cellpose_affine_to_microns,
    run_tiled_cellpose,
)
from merxen.segmentation.proseg import run_proseg_refinement

logger = logging.getLogger(__name__)


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

        points_key = list(sdata.points.keys())[0]
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

        points_key = list(sdata.points.keys())[0]
        return sdata, fetch_tile_fn, height, width, matrix, sdata.points[points_key]

    raise ValueError(f"Unsupported platform: {dataset.platform}")


def run_segmentation_pipeline(
    config: SegmentationConfig,
    *,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Run Cellpose + ProSeg segmentation for one dataset configuration."""
    dataset = config.dataset
    out_dir = Path(dataset.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_output = out_dir / "proseg_base_raw.zarr"
    latest_output = out_dir / "proseg_base_latest.zarr"
    transcripts_csv = out_dir / "transcripts_for_proseg.csv"
    mask_path = out_dir / "cellpose_masks_tiled.npy"

    if latest_output.exists() and not force_rerun:
        log_status(f"[{dataset.name}] Reusing existing latest output: {latest_output}")
        return {
            "raw_output": raw_output,
            "latest_output": latest_output,
            "transcripts_csv": transcripts_csv,
            "cellpose_mask_path": mask_path,
        }

    if raw_output.exists() and not force_rerun:
        latest_out = convert_to_latest_zarr(raw_output, latest_output)
        return {
            "raw_output": raw_output,
            "latest_output": Path(latest_out),
            "transcripts_csv": transcripts_csv,
            "cellpose_mask_path": mask_path,
        }

    sdata, fetch_tile_fn, height, width, matrix, points_obj = _load_dataset_sdata(
        config
    )

    mask_path = run_tiled_cellpose(
        fetch_tile_fn=fetch_tile_fn,
        height=height,
        width=width,
        dataset_name=dataset.name,
        output_mask_path=mask_path,
        cellpose_config=config.cellpose,
        mask_filter_config=config.mask_filter,
        tiling_config=config.tiling,
        memory_config=config.memory,
    )

    x_transform, y_transform = build_cellpose_affine_to_microns(
        matrix,
        scale_factor=1.0,
        x0=0.0,
        y0=0.0,
    )
    log_status(f"[{dataset.name}] mask->micron x_transform={x_transform}")
    log_status(f"[{dataset.name}] mask->micron y_transform={y_transform}")

    x_col = resolve_col(points_obj, ["x", "global_x", "x_location"])
    y_col = resolve_col(points_obj, ["y", "global_y", "y_location"])
    z_col = resolve_col(points_obj, ["z", "global_z", "z_location"], required=False)
    gene_col = resolve_col(points_obj, ["gene", "feature_name", "target"])
    if x_col is None or y_col is None or gene_col is None:
        raise KeyError(
            "Could not resolve required points columns for ProSeg CSV export."
        )

    qv_col = None
    min_qv = None
    if dataset.platform.upper() == "XENIUM":
        qv_col = resolve_col(
            points_obj, ["qv", "quality", "quality_value"], required=False
        )
        min_qv = dataset.min_qv

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

    del mask_mmap, points_obj, sdata
    force_release(note=f"after {dataset.name} preprocessing, before ProSeg")

    proseg_params = config.proseg.model_dump()
    proseg_params.update(dataset.proseg_overrides)

    raw_out = run_proseg_refinement(
        transcripts_df=transcripts_csv,
        output_path=raw_output,
        proseg_binary=proseg_params["binary_path"],
        x_col="x_micron",
        y_col="y_micron",
        z_col="z_micron",
        gene_col="feature_name",
        cell_id_col="cell_id",
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
        cellpose_masks=mask_path,
        cellpose_x_transform=x_transform,
        cellpose_y_transform=y_transform,
        num_threads=int(proseg_params.get("num_threads", 12)),
        overwrite=True,
    )

    latest_out = convert_to_latest_zarr(raw_out, latest_output)
    log_status(f"[{dataset.name}] Wrote latest output: {latest_out}")
    force_release(note=f"after {dataset.name} full segmentation run")

    return {
        "raw_output": Path(raw_out),
        "latest_output": Path(latest_out),
        "transcripts_csv": transcripts_csv,
        "cellpose_mask_path": mask_path,
    }
