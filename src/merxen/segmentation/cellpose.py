"""Cellpose model execution, tiled segmentation, and mask-to-coordinate transforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from cellpose import models
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


def build_cellpose_model(config: CellposeConfig) -> models.CellposeModel:
    """Instantiate a Cellpose model from config.

    Args:
        config: Cellpose configuration parameters.

    Returns:
        A CellposeModel instance.
    """
    kwargs: dict[str, Any] = {"gpu": config.gpu}
    if config.model_type is not None:
        kwargs["model_type"] = config.model_type
    return models.CellposeModel(**kwargs)


def run_cellpose_model_eval(
    model: models.CellposeModel,
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
    model: models.CellposeModel,
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
        try:
            tile = fetch_tile_fn(y0, y1, x0, x1)
            _, img_seg, _ = prepare_cellpose_input(tile, factor_rescale=1.0)
            masks, flows, styles = run_cellpose_model_eval(model, img_seg, config)
            log_status(
                f"[{dataset_name}] Tile size {ts} succeeded "
                f"(labels in probe={int(np.max(masks))})"
            )
            del tile, img_seg, masks, flows, styles
            clear_cuda_cache()
            force_release()
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
                clear_cuda_cache()
                force_release()
                continue
            raise

    raise RuntimeError(
        f"[{dataset_name}] Could not find a working tile size "
        f"from candidates={list(candidates)}"
    )


def relabel_core_to_global(
    core_mask: np.ndarray,
    next_label: int,
) -> tuple[np.ndarray, int, int]:
    """Relabel a core tile mask to avoid label collisions when merging tiles.

    Args:
        core_mask: Labeled mask from a single tile's core region.
        next_label: The next available global label ID.

    Returns:
        Tuple of (relabeled_mask, updated_next_label, n_new_labels).
    """
    core_mask = core_mask.astype(np.int64, copy=False)
    labels = np.unique(core_mask)
    labels = labels[labels > 0]

    if labels.size == 0:
        return np.zeros(core_mask.shape, dtype=np.uint32), int(next_label), 0

    max_label = int(labels.max())
    new_ids = np.arange(
        int(next_label), int(next_label) + int(labels.size), dtype=np.uint32
    )

    if max_label < 20_000_000:
        lut = np.zeros(max_label + 1, dtype=np.uint32)
        lut[labels] = new_ids
        core_global = lut[core_mask]
    else:
        core_global = np.zeros(core_mask.shape, dtype=np.uint32)
        for old, new in zip(labels, new_ids, strict=False):
            core_global[core_mask == old] = new

    return core_global, int(next_label) + int(labels.size), int(labels.size)


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
) -> Path:
    """Run tiled Cellpose segmentation across a full image.

    Adaptively selects tile size based on available GPU memory, then
    processes all tiles with overlap-based stitching and per-tile mask filtering.

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

    Returns:
        Path to the written mask .npy file.
    """
    output_mask_path = Path(output_mask_path)
    if output_mask_path.exists():
        output_mask_path.unlink()

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
        overlap=tiling_config.tile_overlap,
    )
    log_status(
        f"[{dataset_name}] Running tiled Cellpose over {len(tiles)} tiles "
        f"(tile_size={tile_size}, overlap={tiling_config.tile_overlap})"
    )

    mask_mem = np.lib.format.open_memmap(
        str(output_mask_path),
        mode="w+",
        dtype=np.uint32,
        shape=(int(height), int(width)),
    )
    mask_mem[:] = 0

    next_label = 1
    total_new_labels = 0
    tile_filter_n_jobs = max(1, min(mask_filter_config.n_jobs, 4))

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

        if tiling_config.filter_per_tile:
            masks = filter_cell_by_regionprops(
                masks,
                max_eccentricity=mask_filter_config.max_eccentricity,
                n_jobs=tile_filter_n_jobs,
                show_progress=False,
                min_area_percentile=mask_filter_config.min_area_percentile,
                min_area_px=mask_filter_config.min_area_px,
            )

        core = masks[
            t["core_y0_in_tile"] : t["core_y1_in_tile"],
            t["core_x0_in_tile"] : t["core_x1_in_tile"],
        ]

        core_global, next_label, n_new = relabel_core_to_global(
            core, next_label=next_label
        )
        total_new_labels += n_new

        mask_mem[t["core_y0"] : t["core_y1"], t["core_x0"] : t["core_x1"]] = core_global

        pbar.set_postfix_str(
            f"labels={total_new_labels:,} rss={memory_snapshot_gb()['rss_gb']:.1f}GB"
        )

        if i % int(tiling_config.status_every_tiles) == 0:
            log_status(
                f"[{dataset_name}] tile {i}/{len(tiles)} complete; "
                f"accumulated labels={total_new_labels:,}"
            )

        del tile, img_seg, masks, flows, styles, core, core_global
        clear_cuda_cache()

        if i % int(memory_config.memory_check_every_chunks) == 0:
            force_release()

    mask_mem.flush()
    del mask_mem, model
    clear_cuda_cache()
    force_release(note=f"after tiled Cellpose {dataset_name}")

    log_status(
        f"[{dataset_name}] Tiled Cellpose complete; wrote masks to {output_mask_path}"
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
