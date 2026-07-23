"""Image-channel quantification over final Cellpose label masks."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.models import TableModel

from merxen.config import MaskImageQuantificationConfig
from merxen.io.image_source import build_image_source, fetch_tile
from merxen.io.spatialdata_io import write_or_replace_element
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    SOURCE_CELL_ID_COLUMN,
)
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)

MASK_IMAGE_QUANTIFICATION_TABLE_KEY = "table_MOSAIK_cellpose_image_quantification"
MOSAIK_CELLPOSE_SHAPE_NAME = "MOSAIK_cellpose"
IMAGE_QUANTIFICATION_STATS = ("min", "median", "mean", "max", "iqr")


@dataclass(frozen=True)
class MaskImageQuantificationResult:
    """In-memory quantification result before persistence."""

    table: ad.AnnData
    summary: dict[str, Any]


def build_mask_image_quantification_table(
    sdata_obj: Any,
    mask: np.ndarray,
    dataset_name: str,
    *,
    table_key: str = MASK_IMAGE_QUANTIFICATION_TABLE_KEY,
    shape_key: str = MOSAIK_CELLPOSE_SHAPE_NAME,
    tile_size: int = 2048,
) -> MaskImageQuantificationResult:
    """Quantify every image channel over nonzero Cellpose mask labels.

    Args:
        sdata_obj: SpatialData-like object containing image elements.
        mask: Two-dimensional final Cellpose label mask.
        dataset_name: Name used in logging and output summaries.
        table_key: SpatialData table key that will receive the result.
        shape_key: SpatialData shape region represented by the rows.
        tile_size: Square tile size used when streaming image and mask crops.

    Returns:
        AnnData table plus a JSON-serializable summary.
    """
    mask_arr = np.asarray(mask)
    label_ids, label_counts = _foreground_label_counts(mask_arr)
    if label_ids.size == 0:
        raise ValueError(f"[{dataset_name}] Cellpose mask contains no labels.")

    if not getattr(sdata_obj, "images", None):
        raise RuntimeError(f"[{dataset_name}] No image elements found to quantify.")

    log_status(
        f"[{dataset_name}] Quantifying {len(sdata_obj.images)} image element(s) "
        f"over {label_ids.size:,} Cellpose masks"
    )

    matrix_parts: list[np.ndarray] = []
    var_frames: list[pd.DataFrame] = []
    image_summaries: list[dict[str, Any]] = []

    for image_key in list(sdata_obj.images.keys()):
        image_matrix, image_var, image_summary = _quantify_image_element(
            image_key=str(image_key),
            image_obj=sdata_obj.images[image_key],
            mask=mask_arr,
            label_ids=label_ids,
            dataset_name=dataset_name,
            tile_size=int(tile_size),
        )
        matrix_parts.append(image_matrix)
        var_frames.append(image_var)
        image_summaries.append(image_summary)
        force_release(note=f"after {dataset_name} image quantification {image_key}")

    x_matrix = np.concatenate(matrix_parts, axis=1)
    var = pd.concat(var_frames, axis=0)
    obs = pd.DataFrame(
        index=pd.Index(
            label_ids.astype(str),
            dtype=str,
            name="obs_id",
        )
    )
    obs[INSTANCE_ID_COLUMN] = label_ids.astype(np.uint64, copy=False)
    obs[SOURCE_CELL_ID_COLUMN] = _cell_ids(label_ids)
    obs["label_id"] = label_ids.astype(np.int64, copy=False)
    obs["mask_pixel_count"] = label_counts.astype(np.int64, copy=False)
    obs["region"] = pd.Categorical([shape_key] * len(obs), categories=[shape_key])

    table = ad.AnnData(X=x_matrix, obs=obs, var=var)
    table.uns["mask_image_quantification"] = {
        "table_key": table_key,
        "shape_key": shape_key,
        "statistics": list(IMAGE_QUANTIFICATION_STATS),
    }

    summary = {
        "dataset_name": str(dataset_name),
        "table_key": str(table_key),
        "shape_key": str(shape_key),
        "n_cells": int(table.n_obs),
        "n_features": int(table.n_vars),
        "statistics": list(IMAGE_QUANTIFICATION_STATS),
        "images": image_summaries,
    }
    return MaskImageQuantificationResult(table=table, summary=summary)


def run_mask_image_quantification(
    config: MaskImageQuantificationConfig,
    *,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Read a SpatialData zarr, quantify image channels, and persist outputs."""
    latest_path = Path(config.latest_zarr_path)
    mask_path = Path(config.mask_path)
    output_dir = Path(config.output_dir)
    paths = _output_paths(output_dir, config.dataset_name)

    if not latest_path.exists():
        raise FileNotFoundError(f"[{config.dataset_name}] Missing zarr: {latest_path}")
    if not mask_path.exists():
        raise FileNotFoundError(
            f"[{config.dataset_name}] Missing Cellpose mask: {mask_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sdata_obj = sd.read_zarr(latest_path)
    try:
        if (
            not force_rerun
            and config.table_key in sdata_obj.tables
            and _sidecar_outputs_exist(paths)
        ):
            log_status(
                f"[{config.dataset_name}] Image quantification already exists; "
                "skipping."
            )
            return {"latest_zarr": latest_path, **paths}

        mask = np.load(mask_path, mmap_mode="r")
        result = build_mask_image_quantification_table(
            sdata_obj,
            mask,
            config.dataset_name,
            table_key=config.table_key,
            shape_key=config.shape_key,
            tile_size=config.tile_size,
        )
        parsed_table = TableModel.parse(
            result.table,
            region=config.shape_key,
            region_key="region",
            instance_key=INSTANCE_ID_COLUMN,
        )
        write_or_replace_element(
            sdata_obj,
            config.table_key,
            "tables",
            parsed_table,
            overwrite=True,
        )
        _write_sidecar_outputs(result.table, result.summary, paths)
        log_status(
            f"[{config.dataset_name}] Image quantification complete: {config.table_key}"
        )
        return {"latest_zarr": latest_path, **paths}
    finally:
        del sdata_obj
        force_release(note=f"after {config.dataset_name} mask image quantification")


def _quantify_image_element(
    *,
    image_key: str,
    image_obj: Any,
    mask: np.ndarray,
    label_ids: np.ndarray,
    dataset_name: str,
    tile_size: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    source = build_image_source(image_obj, requested_channels=None, as_float32=False)
    height, width, n_channels = source["shape"]
    mask_height, mask_width = mask.shape
    if (int(height), int(width)) != (int(mask_height), int(mask_width)):
        raise ValueError(
            f"[{dataset_name}] Image '{image_key}' shape {height}x{width} does not "
            f"match Cellpose mask shape {mask_height}x{mask_width}."
        )

    channel_names = _unique_channel_names(source, n_channels)
    values_by_channel: list[defaultdict[int, list[np.ndarray]]] = [
        defaultdict(list) for _ in range(int(n_channels))
    ]

    for y0, y1, x0, x1 in _iter_tiles(int(height), int(width), int(tile_size)):
        mask_tile = np.asarray(mask[y0:y1, x0:x1])
        foreground = mask_tile > 0
        if not foreground.any():
            continue

        image_tile = fetch_tile(source, y0, y1, x0, x1)
        tile_labels = mask_tile[foreground].astype(np.int64, copy=False)
        for channel_index in range(int(n_channels)):
            values = np.asarray(image_tile[..., channel_index][foreground])
            finite = np.isfinite(values)
            if not finite.any():
                continue
            _append_grouped_values(
                values_by_channel[channel_index],
                tile_labels[finite],
                values[finite],
            )
        del image_tile, mask_tile, foreground, tile_labels

    label_to_row = {int(label): i for i, label in enumerate(label_ids)}
    image_matrix = np.full(
        (len(label_ids), int(n_channels) * len(IMAGE_QUANTIFICATION_STATS)),
        np.nan,
        dtype=np.float64,
    )
    var_rows: list[dict[str, str]] = []
    feature_names: list[str] = []

    for channel_index, channel in enumerate(channel_names):
        offset = channel_index * len(IMAGE_QUANTIFICATION_STATS)
        for stat_name in IMAGE_QUANTIFICATION_STATS:
            feature_names.append(f"{image_key}__{channel}__{stat_name}")
            var_rows.append(
                {
                    "image_key": image_key,
                    "channel": channel,
                    "statistic": stat_name,
                }
            )

        for label, chunks in values_by_channel[channel_index].items():
            row_index = label_to_row.get(int(label))
            if row_index is None or not chunks:
                continue
            label_values = np.concatenate(chunks).astype(np.float64, copy=False)
            stat_slice = slice(offset, offset + len(IMAGE_QUANTIFICATION_STATS))
            image_matrix[row_index, stat_slice] = _compute_stats(label_values)

    var = pd.DataFrame(
        var_rows,
        index=pd.Index(feature_names, dtype=str, name="feature"),
    )
    summary = {
        "image_key": image_key,
        "height": int(height),
        "width": int(width),
        "n_channels": int(n_channels),
        "channels": channel_names,
    }
    return image_matrix, var, summary


def _append_grouped_values(
    store: defaultdict[int, list[np.ndarray]],
    labels: np.ndarray,
    values: np.ndarray,
) -> None:
    if labels.size == 0:
        return
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    sorted_values = values[order].astype(np.float64, copy=False)
    unique_labels, starts, counts = np.unique(
        sorted_labels,
        return_index=True,
        return_counts=True,
    )
    for label, start, count in zip(unique_labels, starts, counts, strict=True):
        stop = int(start) + int(count)
        store[int(label)].append(sorted_values[int(start) : stop].copy())


def _compute_stats(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full(len(IMAGE_QUANTIFICATION_STATS), np.nan, dtype=np.float64)
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return np.array(
        [
            np.min(values),
            q50,
            np.mean(values, dtype=np.float64),
            np.max(values),
            q75 - q25,
        ],
        dtype=np.float64,
    )


def _foreground_label_counts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.ndim != 2:
        raise ValueError(f"Cellpose mask must be 2D, got shape={mask.shape}")
    if np.any(mask < 0):
        raise ValueError("Cellpose mask labels must be non-negative")
    labels = np.asarray(mask).reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(labels)
    label_ids = np.flatnonzero(counts)
    label_ids = label_ids[label_ids > 0].astype(np.int64, copy=False)
    return label_ids, counts[label_ids]


def _unique_channel_names(source: Mapping[str, Any], n_channels: int) -> list[str]:
    channels = [str(channel) for channel in source.get("channels", [])]
    if len(channels) != int(n_channels):
        channels = [f"c{i}" for i in range(int(n_channels))]

    seen: dict[str, int] = {}
    unique: list[str] = []
    for idx, channel in enumerate(channels):
        base = channel or f"c{idx}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")
    return unique


def _iter_tiles(
    height: int,
    width: int,
    tile_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    for y0 in range(0, int(height), int(tile_size)):
        y1 = min(int(height), y0 + int(tile_size))
        for x0 in range(0, int(width), int(tile_size)):
            x1 = min(int(width), x0 + int(tile_size))
            yield y0, y1, x0, x1


def _cell_ids(label_ids: np.ndarray) -> list[str]:
    return [f"cellpose_{int(label)}" for label in label_ids]


def _output_paths(output_dir: Path, dataset_name: str) -> dict[str, Path]:
    prefix = str(dataset_name).lower()
    return {
        "wide_matrix": output_dir / f"{prefix}_mask_image_quantification.parquet",
        "feature_metadata": output_dir
        / f"{prefix}_mask_image_quantification_features.csv",
        "summary": output_dir / f"{prefix}_mask_image_quantification_summary.json",
    }


def _sidecar_outputs_exist(paths: Mapping[str, Path]) -> bool:
    return all(Path(path).exists() for path in paths.values())


def _write_sidecar_outputs(
    table: ad.AnnData,
    summary: dict[str, Any],
    paths: Mapping[str, Path],
) -> None:
    matrix = pd.DataFrame(
        np.asarray(table.X),
        index=table.obs_names.astype(str),
        columns=table.var_names.astype(str),
    )
    matrix.index.name = "cell_id"
    matrix.to_parquet(paths["wide_matrix"])
    table.var.to_csv(paths["feature_metadata"])

    summary_out = {
        **summary,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary_out, indent=2) + "\n")
