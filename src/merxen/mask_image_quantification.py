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
    canonical_instance_series,
)
from merxen.memory import force_release, log_status
from merxen.viewer_cache.format import is_derived_cache_key

logger = logging.getLogger(__name__)

MASK_IMAGE_QUANTIFICATION_TABLE_KEY = "table_MOSAIK_cellpose_image_quantification"
MOSAIK_CELLPOSE_SHAPE_NAME = "MOSAIK_cellpose"
PROSEG_HYBRID_TABLE_KEY = "table_MOSAIK_proseg_hybrid"
PROSEG_HYBRID_SHAPE_NAME = "MOSAIK_proseg_hybrid"
HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY = "cellpose_image_quantification"
HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX = "cellpose_image__"
IMAGE_QUANTIFICATION_STATS = ("min", "median", "mean", "max", "iqr")


@dataclass(frozen=True)
class MaskImageQuantificationResult:
    """In-memory quantification result before persistence."""

    table: ad.AnnData
    summary: dict[str, Any]


@dataclass(frozen=True)
class HybridImageQuantificationJoin:
    """Cellpose image features aligned onto the hybrid-cell identifier space."""

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

    images = getattr(sdata_obj, "images", None)
    if not images:
        raise RuntimeError(f"[{dataset_name}] No image elements found to quantify.")

    image_keys = [
        image_key for image_key in images if not is_derived_cache_key(str(image_key))
    ]
    if not image_keys:
        raise RuntimeError(
            f"[{dataset_name}] No source image elements found to quantify after "
            "excluding private viewer-cache images."
        )
    skipped_cache_images = len(images) - len(image_keys)
    log_status(
        f"[{dataset_name}] Quantifying {len(image_keys)} source image element(s) "
        f"over {label_ids.size:,} Cellpose masks; skipped "
        f"{skipped_cache_images} private viewer-cache image(s)"
    )

    matrix_parts: list[np.ndarray] = []
    var_frames: list[pd.DataFrame] = []
    image_summaries: list[dict[str, Any]] = []

    for image_key in image_keys:
        image_matrix, image_var, image_summary = _quantify_image_element(
            image_key=str(image_key),
            image_obj=images[image_key],
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
        existing_quantification = (
            not force_rerun
            and config.table_key in sdata_obj.tables
            and _sidecar_outputs_exist(paths)
        )
        if existing_quantification:
            if _hybrid_image_join_is_complete(
                sdata_obj,
                image_table_key=config.table_key,
            ):
                log_status(
                    f"[{config.dataset_name}] Image quantification and hybrid "
                    "feature join already exist; skipping."
                )
                return {"latest_zarr": latest_path, **paths}
            hybrid_join_summary = _write_hybrid_image_quantification_join(
                sdata_obj,
                sdata_obj.tables[config.table_key],
                image_table_key=config.table_key,
            )
            _update_sidecar_summary(paths["summary"], hybrid_join_summary)
            log_status(
                f"[{config.dataset_name}] Reused existing Cellpose image "
                "quantification and added the hybrid-cell feature join."
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
        hybrid_join_summary = _write_hybrid_image_quantification_join(
            sdata_obj,
            parsed_table,
            image_table_key=config.table_key,
        )
        result.summary["hybrid_join"] = hybrid_join_summary
        _write_sidecar_outputs(result.table, result.summary, paths)
        log_status(
            f"[{config.dataset_name}] Image quantification complete: {config.table_key}"
        )
        return {"latest_zarr": latest_path, **paths}
    finally:
        del sdata_obj
        force_release(note=f"after {config.dataset_name} mask image quantification")


def join_cellpose_image_quantification_to_hybrid(
    image_table: ad.AnnData,
    hybrid_table: ad.AnnData,
    *,
    image_table_key: str = MASK_IMAGE_QUANTIFICATION_TABLE_KEY,
) -> HybridImageQuantificationJoin:
    """Attach Cellpose-mask image features to matching hybrid cells.

    Hybrid cells retain the Cellpose label identifier, so this is an identifier
    join. The hybrid expression matrix remains untouched. Image measurements are
    exposed both as prefixed ``obs`` columns and as a compact ``obsm`` matrix.
    """
    image_ids = _table_instance_ids(image_table, table_name=image_table_key)
    hybrid_ids = _table_instance_ids(
        hybrid_table,
        table_name=PROSEG_HYBRID_TABLE_KEY,
    )
    image_positions = pd.Series(
        np.arange(image_table.n_obs, dtype=np.int64),
        index=pd.Index(image_ids.to_numpy(dtype=np.uint64), dtype="uint64"),
    )
    matched_positions = hybrid_ids.map(image_positions)
    matched = matched_positions.notna().to_numpy()

    source_matrix = image_table.X
    if hasattr(source_matrix, "toarray"):
        source_matrix = source_matrix.toarray()
    source_matrix = np.asarray(source_matrix, dtype=np.float64)
    joined_matrix = np.full(
        (hybrid_table.n_obs, image_table.n_vars),
        np.nan,
        dtype=np.float64,
    )
    if bool(matched.any()):
        joined_matrix[matched] = source_matrix[
            matched_positions.loc[matched].astype(int).to_numpy()
        ]

    updated = hybrid_table.copy()
    feature_names = image_table.var_names.astype(str).tolist()
    feature_obs_columns = _unique_spatialdata_obs_columns(
        [
            f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}{feature_name}"
            for feature_name in feature_names
        ],
        existing_columns=updated.obs.columns.astype(str).tolist(),
    )
    for feature_index, obs_column in enumerate(feature_obs_columns):
        updated.obs[obs_column] = joined_matrix[:, feature_index]

    for source_column in ("mask_pixel_count", "label_id"):
        if source_column not in image_table.obs.columns:
            continue
        values = pd.to_numeric(
            image_table.obs[source_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        joined_values = np.full(hybrid_table.n_obs, np.nan, dtype=float)
        if bool(matched.any()):
            joined_values[matched] = values[
                matched_positions.loc[matched].astype(int).to_numpy()
            ]
        updated.obs[f"cellpose_image_{source_column}"] = joined_values

    updated.obsm[HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY] = joined_matrix
    updated.uns[HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY] = {
        "source_table_key": str(image_table_key),
        "feature_names": feature_names,
        "obs_columns": feature_obs_columns,
        "obs_prefix": HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX,
        "n_matched_cells": int(matched.sum()),
        "n_unmatched_hybrid_cells": int((~matched).sum()),
    }
    return HybridImageQuantificationJoin(
        table=updated,
        summary={
            "status": "joined",
            "source_table_key": str(image_table_key),
            "target_table_key": PROSEG_HYBRID_TABLE_KEY,
            "n_hybrid_cells": int(hybrid_table.n_obs),
            "n_matched_cells": int(matched.sum()),
            "n_unmatched_hybrid_cells": int((~matched).sum()),
            "n_features": int(image_table.n_vars),
        },
    )


def _unique_spatialdata_obs_columns(
    names: list[str],
    *,
    existing_columns: list[str],
) -> list[str]:
    """Return deterministic, unique names accepted as SpatialData table columns."""
    used_lower = {str(column).lower() for column in existing_columns}
    result: list[str] = []
    for name in names:
        base = "".join(
            character if character.isalnum() or character in {"_", "-", "."} else "_"
            for character in str(name)
        )
        while base.startswith("__"):
            base = base[1:]
        if not base or base in {".", ".."}:
            base = "unnamed"
        if base == "_index":
            base = "index"

        candidate = base
        suffix = 1
        while candidate.lower() in used_lower:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_lower.add(candidate.lower())
        result.append(candidate)
    return result


def _write_hybrid_image_quantification_join(
    sdata_obj: Any,
    image_table: ad.AnnData,
    *,
    image_table_key: str,
) -> dict[str, Any]:
    if PROSEG_HYBRID_TABLE_KEY not in sdata_obj.tables:
        return {
            "status": "skipped",
            "reason": "hybrid_table_absent",
            "target_table_key": PROSEG_HYBRID_TABLE_KEY,
        }

    join = join_cellpose_image_quantification_to_hybrid(
        image_table,
        sdata_obj.tables[PROSEG_HYBRID_TABLE_KEY],
        image_table_key=image_table_key,
    )
    table = join.table
    table.uns.pop("spatialdata_attrs", None)
    table.obs["region"] = pd.Categorical(
        [PROSEG_HYBRID_SHAPE_NAME] * table.n_obs,
        categories=[PROSEG_HYBRID_SHAPE_NAME],
    )
    parsed = TableModel.parse(
        table,
        region=PROSEG_HYBRID_SHAPE_NAME,
        region_key="region",
        instance_key=INSTANCE_ID_COLUMN,
    )
    write_or_replace_element(
        sdata_obj,
        PROSEG_HYBRID_TABLE_KEY,
        "tables",
        parsed,
        overwrite=True,
    )
    return join.summary


def _hybrid_image_join_is_complete(
    sdata_obj: Any,
    *,
    image_table_key: str,
) -> bool:
    if PROSEG_HYBRID_TABLE_KEY not in sdata_obj.tables:
        return True
    if image_table_key not in sdata_obj.tables:
        return False
    marker = dict(
        sdata_obj.tables[PROSEG_HYBRID_TABLE_KEY].uns.get(
            HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY,
            {},
        )
    )
    expected_features = sdata_obj.tables[image_table_key].var_names.astype(str).tolist()
    return (
        marker.get("source_table_key") == str(image_table_key)
        and marker.get("feature_names") == expected_features
        and HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY
        in sdata_obj.tables[PROSEG_HYBRID_TABLE_KEY].obsm
    )


def _table_instance_ids(
    table: ad.AnnData,
    *,
    table_name: str,
) -> pd.Series:
    attrs = dict(table.uns.get("spatialdata_attrs", {}))
    instance_key = str(attrs.get("instance_key", INSTANCE_ID_COLUMN))
    if instance_key not in table.obs.columns:
        raise KeyError(
            f"{table_name!r} has no instance identifier column {instance_key!r}"
        )
    return canonical_instance_series(
        table.obs[instance_key],
        field_name=f"{table_name}.{instance_key}",
    )


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


def _update_sidecar_summary(
    summary_path: Path,
    hybrid_join_summary: Mapping[str, Any],
) -> None:
    summary = json.loads(summary_path.read_text())
    summary["hybrid_join"] = dict(hybrid_join_summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
