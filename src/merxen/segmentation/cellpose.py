"""Cellpose model execution, tiled segmentation, and mask-to-coordinate transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.ndimage import find_objects
from tqdm.auto import tqdm

from merxen._typing import AffineComponent
from merxen.config import CellposeConfig, MaskFilterConfig, MemoryConfig, TilingConfig
from merxen.io.image_source import prepare_cellpose_input
from merxen.memory import (
    clear_cuda_cache,
    enforce_memory_limit,
    force_release,
    log_status,
    memory_snapshot_gb,
)
from merxen.segmentation.mask_filter import filter_cell_by_regionprops

logger = logging.getLogger(__name__)

try:
    from cellpose import models as cellpose_models
except ImportError:  # pragma: no cover - exercised only in lightweight test envs
    cellpose_models = None


def build_cellpose_model(config: CellposeConfig) -> Any:
    """Instantiate a Cellpose model from config.

    Args:
        config: Cellpose configuration parameters.

    Returns:
        A CellposeModel instance.
    """
    if cellpose_models is None:
        raise RuntimeError(
            "Cellpose is not installed in this environment. "
            "Install the MerXen segmentation dependencies before running Cellpose."
        )

    kwargs: dict[str, Any] = {"gpu": config.gpu}
    if config.model_type is not None:
        kwargs["model_type"] = config.model_type
        kwargs["use_bfloat16"] = config.use_bfloat16
    return cellpose_models.CellposeModel(**kwargs)


def run_cellpose_model_eval(
    model: Any,
    img_seg: np.ndarray,
    config: CellposeConfig,
) -> tuple[np.ndarray, Any, Any]:
    """Run Cellpose inference on a single image tile.

    Args:
        model: A CellposeModel instance.
        img_seg: Input image array (H, W, 3) uint8.
        config: Cellpose configuration parameters.

    Returns:
        Tuple of (masks, flows, styles) from Cellpose eval.
    """
    masks, flows, styles = cast(
        tuple[np.ndarray, Any, Any],
        model.eval(
            img_seg,
            diameter=config.diameter,
            flow_threshold=config.flow_threshold,
            cellprob_threshold=config.cellprob_threshold,
            tile_overlap=config.tile_overlap,
            bsize=config.bsize,
        ),
    )
    return np.asarray(masks), flows, styles


def extract_cellpose_probability_logits(
    flows: Any,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    """Extract Cellpose's per-pixel cell-probability logits from model flows.

    Args:
        flows: Flow payload returned by ``CellposeModel.eval``.
        mask_shape: Expected two-dimensional mask shape.

    Returns:
        Float32 logits with the same shape as the mask.

    Raises:
        ValueError: If the Cellpose flow payload has no compatible probability map.
    """
    if not isinstance(flows, list | tuple) or len(flows) < 3:
        raise ValueError("Cellpose did not return a cell-probability flow map.")
    logits = np.asarray(flows[2]).squeeze()
    if logits.shape != tuple(mask_shape):
        raise ValueError(
            "Cellpose probability shape does not match masks: "
            f"{logits.shape} != {tuple(mask_shape)}"
        )
    return logits.astype(np.float32, copy=False)


def synchronize_cellpose_probability_logits(
    mask_path: Path | str,
    probability_path: Path | str,
    *,
    chunk_rows: int = 2048,
) -> None:
    """Zero probability logits wherever the final Cellpose mask is background.

    Args:
        mask_path: Final two-dimensional Cellpose label image.
        probability_path: Stitched float32 Cellpose probability-logit image.
        chunk_rows: Number of image rows updated per memory-bounded chunk.
    """
    masks = np.load(Path(mask_path), mmap_mode="r")
    probabilities = np.load(Path(probability_path), mmap_mode="r+")
    if masks.shape != probabilities.shape:
        raise ValueError(
            "Cellpose mask and probability shapes differ: "
            f"{masks.shape} != {probabilities.shape}"
        )
    rows = max(1, int(chunk_rows))
    for start in range(0, int(masks.shape[0]), rows):
        stop = min(start + rows, int(masks.shape[0]))
        probability_chunk = probabilities[start:stop]
        probability_chunk[masks[start:stop] == 0] = 0.0
    probabilities.flush()
    del masks, probabilities


def iter_core_tiles(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[dict[str, int]]:
    """Generate tiling coordinates with overlap for core-based segmentation.

    Each tile has an overlap region that is discarded after segmentation,
    keeping only the inner "core" region to avoid edge artifacts.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        tile_size: Full tile size (core + 2*overlap).
        overlap: Overlap on each side of the core.

    Returns:
        List of tile dicts with core and tile coordinates.

    Raises:
        ValueError: If tile_size - 2*overlap <= 0.
    """
    tile_size = int(tile_size)
    overlap = int(overlap)
    core = tile_size - (2 * overlap)
    if core <= 0:
        raise ValueError(
            f"Invalid tile config: tile_size={tile_size}, overlap={overlap}"
        )

    tiles = []
    ys = list(range(0, int(height), core))
    xs = list(range(0, int(width), core))

    for y_core0 in ys:
        y_core1 = min(y_core0 + core, int(height))
        y_tile0 = max(0, y_core0 - overlap)
        y_tile1 = min(int(height), y_core1 + overlap)
        y_core0_in_tile = y_core0 - y_tile0
        y_core1_in_tile = y_core0_in_tile + (y_core1 - y_core0)

        for x_core0 in xs:
            x_core1 = min(x_core0 + core, int(width))
            x_tile0 = max(0, x_core0 - overlap)
            x_tile1 = min(int(width), x_core1 + overlap)
            x_core0_in_tile = x_core0 - x_tile0
            x_core1_in_tile = x_core0_in_tile + (x_core1 - x_core0)

            tiles.append(
                {
                    "core_y0": y_core0,
                    "core_y1": y_core1,
                    "core_x0": x_core0,
                    "core_x1": x_core1,
                    "tile_y0": y_tile0,
                    "tile_y1": y_tile1,
                    "tile_x0": x_tile0,
                    "tile_x1": x_tile1,
                    "image_height": int(height),
                    "image_width": int(width),
                    "core_y0_in_tile": y_core0_in_tile,
                    "core_y1_in_tile": y_core1_in_tile,
                    "core_x0_in_tile": x_core0_in_tile,
                    "core_x1_in_tile": x_core1_in_tile,
                }
            )

    return tiles


def choose_working_tile_size(
    fetch_tile_fn: Any,
    height: int,
    width: int,
    model: Any,
    config: CellposeConfig,
    candidates: list[int],
    dataset_name: str,
) -> int:
    """Benchmark tile sizes and select the largest that fits in GPU memory.

    Probes a center tile at each candidate size, starting from the largest.
    Falls back to smaller sizes on OOM errors.

    Args:
        fetch_tile_fn: Callable(y0, y1, x0, x1) -> ndarray.
        height: Image height.
        width: Image width.
        model: Cellpose model for probing.
        config: Cellpose parameters.
        candidates: Tile sizes to try, largest first.
        dataset_name: Name for logging.

    Returns:
        The largest working tile size.

    Raises:
        RuntimeError: If no candidate tile size works.
    """
    cy = int(height) // 2
    cx = int(width) // 2

    for ts in candidates:
        ts = int(ts)
        y0 = max(0, cy - ts // 2)
        x0 = max(0, cx - ts // 2)
        y1 = min(int(height), y0 + ts)
        x1 = min(int(width), x0 + ts)
        y0 = max(0, y1 - ts)
        x0 = max(0, x1 - ts)

        log_status(f"[{dataset_name}] Probing Cellpose tile size={ts} on center tile")
        tile = img_seg = masks = flows = styles = None
        try:
            tile = fetch_tile_fn(y0, y1, x0, x1)
            _, img_seg, _ = prepare_cellpose_input(tile, factor_rescale=1.0)
            masks, flows, styles = run_cellpose_model_eval(model, img_seg, config)
            log_status(
                f"[{dataset_name}] Tile size {ts} succeeded "
                f"(labels in probe={int(np.max(masks))})"
            )
            return ts
        except (RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            is_oom = (
                "out of memory" in msg
                or ("cuda" in msg and "memory" in msg)
                or isinstance(e, MemoryError)
            )
            if is_oom:
                log_status(
                    f"[{dataset_name}] Tile size {ts} failed due to memory; "
                    "trying smaller tile"
                )
                continue
            raise
        finally:
            del tile, img_seg, masks, flows, styles
            clear_cuda_cache()
            force_release()

    raise RuntimeError(
        f"[{dataset_name}] Could not find a working tile size "
        f"from candidates={list(candidates)}"
    )


def _count_positive_labels(mask: np.ndarray) -> int:
    """Return the number of non-zero labels present in a mask."""
    labels = np.unique(mask)
    return int(np.count_nonzero(labels > 0))


def _empty_stitch_stats() -> dict[str, int]:
    """Create a per-tile stitching counter dictionary."""
    return {
        "raw_labels": 0,
        "filtered_labels": 0,
        "owned_labels": 0,
        "accepted_labels": 0,
        "duplicate_skipped": 0,
        "low_remaining_skipped": 0,
        "edge_touching_labels": 0,
        "edge_touching_skipped": 0,
        "conflict_pixels": 0,
        "filled_pixels": 0,
    }


def _label_touches_artificial_tile_edge(
    label_slice: tuple[slice, slice],
    tile_mask_shape: tuple[int, int],
    tile: dict[str, int],
) -> bool:
    """Return True when a label touches a tile edge that is not a global image edge."""
    y_slice, x_slice = label_slice
    tile_h, tile_w = tile_mask_shape
    return (
        (int(y_slice.start or 0) == 0 and int(tile["tile_y0"]) > 0)
        or (
            int(y_slice.stop or 0) >= tile_h
            and int(tile["tile_y1"]) < int(tile["image_height"])
        )
        or (int(x_slice.start or 0) == 0 and int(tile["tile_x0"]) > 0)
        or (
            int(x_slice.stop or 0) >= tile_w
            and int(tile["tile_x1"]) < int(tile["image_width"])
        )
    )


def _stitch_core_owned_tile_labels(
    tile_mask: np.ndarray,
    global_mask: np.ndarray,
    tile: dict[str, int],
    next_label: int,
    global_label_areas: dict[int, int],
    tiling_config: TilingConfig,
    tile_cellprobs: np.ndarray | None = None,
    global_cellprobs: np.ndarray | None = None,
) -> tuple[int, dict[str, int]]:
    """Paste whole core-owned objects from one tile into the global mask.

    A label is owned by a tile when its centroid falls inside the tile core.
    Owned objects are pasted from the full halo tile, not cropped to the core,
    so cells crossing a core boundary are not cut in half. Overlapping objects
    from neighboring tiles are treated as duplicates when their overlap with an
    existing global object is high enough.
    """
    stats = _empty_stitch_stats()
    tile_mask = np.asarray(tile_mask)
    stats["filtered_labels"] = _count_positive_labels(tile_mask)

    if stats["filtered_labels"] == 0:
        return int(next_label), stats

    label_slices = find_objects(tile_mask)
    core_y0 = int(tile["core_y0_in_tile"])
    core_y1 = int(tile["core_y1_in_tile"])
    core_x0 = int(tile["core_x0_in_tile"])
    core_x1 = int(tile["core_x1_in_tile"])

    for label_id, label_slice in enumerate(label_slices, start=1):
        if label_slice is None:
            continue

        y_slice, x_slice = label_slice
        local_mask = tile_mask[label_slice] == label_id
        candidate_area = int(np.count_nonzero(local_mask))
        if candidate_area == 0:
            continue

        yy, xx = np.nonzero(local_mask)
        centroid_y = float((yy + int(y_slice.start or 0)).mean())
        centroid_x = float((xx + int(x_slice.start or 0)).mean())
        if not (core_y0 <= centroid_y < core_y1 and core_x0 <= centroid_x < core_x1):
            continue

        stats["owned_labels"] += 1
        touches_edge = _label_touches_artificial_tile_edge(
            label_slice,
            (int(tile_mask.shape[0]), int(tile_mask.shape[1])),
            tile,
        )
        if touches_edge:
            stats["edge_touching_labels"] += 1
            if tiling_config.edge_touch_policy == "skip":
                stats["edge_touching_skipped"] += 1
                continue

        global_y0 = int(tile["tile_y0"]) + int(y_slice.start or 0)
        global_y1 = int(tile["tile_y0"]) + int(y_slice.stop or 0)
        global_x0 = int(tile["tile_x0"]) + int(x_slice.start or 0)
        global_x1 = int(tile["tile_x0"]) + int(x_slice.stop or 0)

        global_view = global_mask[global_y0:global_y1, global_x0:global_x1]
        existing_under_candidate = global_view[local_mask]
        conflict = existing_under_candidate > 0
        conflict_pixels = int(np.count_nonzero(conflict))
        stats["conflict_pixels"] += conflict_pixels

        duplicate = False
        if conflict_pixels > 0:
            existing_labels, intersections = np.unique(
                existing_under_candidate[conflict],
                return_counts=True,
            )
            max_iou = 0.0
            max_candidate_overlap = 0.0
            for existing_label, intersection in zip(
                existing_labels,
                intersections,
                strict=False,
            ):
                existing_label_int = int(existing_label)
                intersection_int = int(intersection)
                existing_area = int(global_label_areas.get(existing_label_int, 0))
                union = candidate_area + existing_area - intersection_int
                iou = intersection_int / union if union > 0 else 0.0
                candidate_overlap = intersection_int / candidate_area
                max_iou = max(max_iou, iou)
                max_candidate_overlap = max(max_candidate_overlap, candidate_overlap)

            duplicate = (
                max_iou >= tiling_config.duplicate_iou_threshold
                or max_candidate_overlap >= tiling_config.duplicate_overlap_fraction
            )

        if duplicate:
            stats["duplicate_skipped"] += 1
            continue

        remaining_pixels = candidate_area - conflict_pixels
        remaining_fraction = remaining_pixels / max(candidate_area, 1)
        if remaining_fraction < tiling_config.min_remaining_fraction:
            stats["low_remaining_skipped"] += 1
            continue

        fill_mask = local_mask & (global_view == 0)
        filled_pixels = int(np.count_nonzero(fill_mask))
        if filled_pixels == 0:
            stats["low_remaining_skipped"] += 1
            continue

        global_view[fill_mask] = np.uint32(next_label)
        if tile_cellprobs is not None and global_cellprobs is not None:
            probability_view = global_cellprobs[
                global_y0:global_y1,
                global_x0:global_x1,
            ]
            tile_probability_view = tile_cellprobs[label_slice]
            probability_view[fill_mask] = tile_probability_view[fill_mask]
        global_label_areas[int(next_label)] = filled_pixels
        stats["accepted_labels"] += 1
        stats["filled_pixels"] += filled_pixels
        next_label += 1

    return int(next_label), stats


def _merge_stitch_stats(total: dict[str, Any], tile_stats: dict[str, int]) -> None:
    """Accumulate per-tile stitching counters into a run-level stats dict."""
    for key, value in tile_stats.items():
        total[key] = int(total.get(key, 0)) + int(value)


def _write_stitching_stats(path: Path, stats: dict[str, Any]) -> None:
    """Write Cellpose stitching stats as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True))


def run_tiled_cellpose(
    fetch_tile_fn: Any,
    height: int,
    width: int,
    dataset_name: str,
    output_mask_path: Path,
    cellpose_config: CellposeConfig,
    mask_filter_config: MaskFilterConfig,
    tiling_config: TilingConfig,
    memory_config: MemoryConfig,
    output_stitching_stats_path: Path | None = None,
    output_cellprob_path: Path | None = None,
    progress_callback: Any = None,
) -> Path:
    """Run tiled Cellpose segmentation across a full image.

    Adaptively selects tile size based on available GPU memory, then
    processes all tiles with object-level stitching and per-tile mask filtering.

    Args:
        fetch_tile_fn: Callable(y0, y1, x0, x1) -> ndarray.
        height: Image height in pixels.
        width: Image width in pixels.
        dataset_name: Name for logging.
        output_mask_path: Path to write the output .npy mask.
        cellpose_config: Cellpose parameters.
        mask_filter_config: Mask filtering parameters.
        tiling_config: Tiling parameters.
        memory_config: Memory management parameters.
        output_stitching_stats_path: Optional path for stitching diagnostics JSON.
        output_cellprob_path: Optional path for stitched Cellpose probability logits.

    Returns:
        Path to the written mask .npy file.
    """
    output_mask_path = Path(output_mask_path)
    if output_mask_path.exists():
        output_mask_path.unlink()
    if output_stitching_stats_path is not None:
        output_stitching_stats_path = Path(output_stitching_stats_path)
        if output_stitching_stats_path.exists():
            output_stitching_stats_path.unlink()
    if output_cellprob_path is not None:
        output_cellprob_path = Path(output_cellprob_path)
        output_cellprob_path.parent.mkdir(parents=True, exist_ok=True)
        if output_cellprob_path.exists():
            output_cellprob_path.unlink()

    candidates = [
        int(x)
        for x in tiling_config.tile_size_candidates
        if int(x) >= int(tiling_config.min_tile_size)
    ]
    if not candidates:
        raise ValueError(
            "No valid tile sizes in tile_size_candidates after min_tile_size filter"
        )

    model = build_cellpose_model(cellpose_config)
    tile_size = choose_working_tile_size(
        fetch_tile_fn=fetch_tile_fn,
        height=height,
        width=width,
        model=model,
        config=cellpose_config,
        candidates=candidates,
        dataset_name=dataset_name,
    )

    tiles = iter_core_tiles(
        height=height,
        width=width,
        tile_size=tile_size,
        overlap=tiling_config.stitch_overlap_px,
    )
    log_status(
        f"[{dataset_name}] Running tiled Cellpose over {len(tiles)} tiles "
        f"(tile_size={tile_size}, stitch_overlap_px="
        f"{tiling_config.stitch_overlap_px})"
    )

    mask_mem = np.lib.format.open_memmap(
        str(output_mask_path),
        mode="w+",
        dtype=np.uint32,
        shape=(int(height), int(width)),
    )
    mask_mem[:] = 0
    cellprob_mem: np.memmap | None = None
    if output_cellprob_path is not None:
        cellprob_mem = cast(
            np.memmap,
            np.lib.format.open_memmap(
                str(output_cellprob_path),
                mode="w+",
                dtype=np.float32,
                shape=(int(height), int(width)),
            ),
        )
        cellprob_mem[:] = 0.0

    next_label = 1
    global_label_areas: dict[int, int] = {}
    tile_filter_n_jobs = max(1, min(mask_filter_config.n_jobs, 4))
    stitch_stats: dict[str, Any] = {
        "dataset_name": dataset_name,
        "height": int(height),
        "width": int(width),
        "tile_size": int(tile_size),
        "stitch_overlap_px": int(tiling_config.stitch_overlap_px),
        "tiles_total": len(tiles),
        "duplicate_iou_threshold": float(tiling_config.duplicate_iou_threshold),
        "duplicate_overlap_fraction": float(tiling_config.duplicate_overlap_fraction),
        "min_remaining_fraction": float(tiling_config.min_remaining_fraction),
        "edge_touch_policy": tiling_config.edge_touch_policy,
        "raw_labels": 0,
        "filtered_labels": 0,
        "owned_labels": 0,
        "accepted_labels": 0,
        "duplicate_skipped": 0,
        "low_remaining_skipped": 0,
        "edge_touching_labels": 0,
        "edge_touching_skipped": 0,
        "conflict_pixels": 0,
        "filled_pixels": 0,
        "final_labels": 0,
    }

    pbar = tqdm(
        tiles,
        total=len(tiles),
        desc=f"[{dataset_name}] Cellpose tiles",
        unit="tile",
    )

    for i, t in enumerate(pbar, start=1):
        enforce_memory_limit(
            stage=f"{dataset_name} tile {i}/{len(tiles)}",
            max_gb=memory_config.max_system_ram_gb,
            warn_gb=memory_config.memory_warn_gb,
        )

        tile = fetch_tile_fn(t["tile_y0"], t["tile_y1"], t["tile_x0"], t["tile_x1"])
        _, img_seg, _ = prepare_cellpose_input(tile, factor_rescale=1.0)

        masks, flows, styles = run_cellpose_model_eval(model, img_seg, cellpose_config)
        cellprob_logits = (
            extract_cellpose_probability_logits(flows, masks.shape)
            if cellprob_mem is not None
            else None
        )
        raw_label_count = _count_positive_labels(masks)

        if tiling_config.filter_per_tile:
            masks = filter_cell_by_regionprops(
                masks,
                max_eccentricity=mask_filter_config.max_eccentricity,
                n_jobs=tile_filter_n_jobs,
                show_progress=False,
                min_area_percentile=mask_filter_config.min_area_percentile,
                min_area_px=mask_filter_config.min_area_px,
            )

        next_label, tile_stats = _stitch_core_owned_tile_labels(
            tile_mask=masks,
            global_mask=mask_mem,
            tile=t,
            next_label=next_label,
            global_label_areas=global_label_areas,
            tiling_config=tiling_config,
            tile_cellprobs=cellprob_logits,
            global_cellprobs=cellprob_mem,
        )
        tile_stats["raw_labels"] = raw_label_count
        _merge_stitch_stats(stitch_stats, tile_stats)

        pbar.set_postfix_str(
            f"labels={next_label - 1:,} rss={memory_snapshot_gb()['rss_gb']:.1f}GB"
        )

        if i % int(tiling_config.status_every_tiles) == 0:
            log_status(
                f"[{dataset_name}] tile {i}/{len(tiles)} complete; "
                f"accepted labels={next_label - 1:,}; "
                f"duplicates skipped={stitch_stats['duplicate_skipped']:,}"
            )
            if progress_callback is not None:
                progress_callback(
                    "cellpose_tiling",
                    tiles_done=i,
                    tiles_total=len(tiles),
                    pct=round(100 * i / len(tiles), 1),
                    labels_found=next_label - 1,
                    duplicate_skipped=stitch_stats["duplicate_skipped"],
                )

        del tile, img_seg, masks, flows, styles, cellprob_logits
        clear_cuda_cache()

        if i % int(memory_config.memory_check_every_chunks) == 0:
            force_release()

    mask_mem.flush()
    if cellprob_mem is not None:
        cellprob_mem.flush()
    stitch_stats["final_labels"] = int(next_label - 1)
    if tiling_config.write_stitching_stats and output_stitching_stats_path is not None:
        _write_stitching_stats(output_stitching_stats_path, stitch_stats)
    del mask_mem, cellprob_mem, model
    clear_cuda_cache()
    force_release(note=f"after tiled Cellpose {dataset_name}")

    log_status(
        f"[{dataset_name}] Tiled Cellpose complete; wrote masks to {output_mask_path}"
    )
    if tiling_config.write_stitching_stats and output_stitching_stats_path is not None:
        log_status(
            f"[{dataset_name}] Wrote Cellpose stitching stats to "
            f"{output_stitching_stats_path}"
        )
    return output_mask_path


def build_cellpose_affine_to_microns(
    transform_matrix: np.ndarray,
    scale_factor: float,
    x0: float = 0.0,
    y0: float = 0.0,
) -> tuple[AffineComponent, AffineComponent]:
    """Construct an affine transform from Cellpose mask pixels to micron coordinates.

    Args:
        transform_matrix: 3x3 micron-to-pixel transform matrix.
        scale_factor: Scale factor applied during Cellpose (1.0 for native).
        x0: Pixel x-offset of the mask origin.
        y0: Pixel y-offset of the mask origin.

    Returns:
        Tuple of (x_transform, y_transform) where each is (a, b, offset).
    """
    m_inv = np.linalg.inv(transform_matrix)
    s = np.array(
        [
            [scale_factor, 0.0, float(x0)],
            [0.0, scale_factor, float(y0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    t = m_inv @ s
    x_transform = (float(t[0, 0]), float(t[0, 1]), float(t[0, 2]))
    y_transform = (float(t[1, 0]), float(t[1, 1]), float(t[1, 2]))
    return x_transform, y_transform


def invert_mask_affine(
    x_transform: AffineComponent,
    y_transform: AffineComponent,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the inverse of a pixel-to-micron affine transform.

    Args:
        x_transform: (a, b, tx) for x.
        y_transform: (a, b, ty) for y.

    Returns:
        Tuple of (A_inv, b) where A_inv is the 2x2 inverse matrix
        and b is the translation vector.
    """
    a = np.array(
        [
            [float(x_transform[0]), float(x_transform[1])],
            [float(y_transform[0]), float(y_transform[1])],
        ],
        dtype=float,
    )
    b = np.array([float(x_transform[2]), float(y_transform[2])], dtype=float)
    a_inv = np.linalg.inv(a)
    return a_inv, b


def assign_labels_from_masks(
    x_micron: np.ndarray,
    y_micron: np.ndarray,
    masks: np.ndarray,
    a_inv: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Assign transcript points to cell mask labels via inverse affine transform.

    Converts micron coordinates to pixel coordinates using the inverse affine,
    then looks up the mask label at each pixel location.

    Args:
        x_micron: X coordinates in microns.
        y_micron: Y coordinates in microns.
        masks: 2D labeled mask array (H, W).
        a_inv: 2x2 inverse affine matrix.
        b: Translation vector (2,).

    Returns:
        Array of cell labels (0 = unassigned).
    """
    coords = np.vstack([x_micron, y_micron])
    pix = a_inv @ (coords - b[:, None])

    x_px = np.rint(pix[0]).astype(np.int64)
    y_px = np.rint(pix[1]).astype(np.int64)

    h, w = masks.shape
    valid = (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)

    labels = np.zeros(len(x_micron), dtype=np.int32)
    labels[valid] = masks[y_px[valid], x_px[valid]].astype(np.int32, copy=False)
    return labels
