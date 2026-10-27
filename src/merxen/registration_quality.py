"""Per-cell quality scores for registered external DAPI images."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from merxen.io.image_source import image_to_cyx
from merxen.io.spatialdata_io import write_or_replace_element

REGISTRATION_QUALITY_ALGORITHM = "cell_dapi_agreement_v1"
REGISTRATION_QUALITY_UNS_KEY = "registered_image_quality"
REGISTRATION_QUALITY_TABLE_KEYS = {
    "table_MOSAIK_cellpose",
    "table_MOSAIK_proseg_hybrid",
}


@dataclass(frozen=True)
class RegistrationQualityResult:
    """Per-label registration measurements and their aggregate summary."""

    metrics: pd.DataFrame
    summary: dict[str, Any]


def compute_cell_registration_quality(
    reference: Any,
    moving: Any,
    nuclei_labels: Any,
    cell_labels: Any,
    *,
    downsample: int,
    source_pixel_size_um: float,
    smoothing_sigma_um: float = 0.68,
    tile_size: int = 1024,
    min_cell_pixels: int = 20,
    min_nuclear_pixels: int = 8,
    min_surrounding_pixels: int = 8,
) -> RegistrationQualityResult:
    """Measure local DAPI agreement for every Cellpose cell label.

    The registered and reference DAPI images are log transformed and lightly
    smoothed before two complementary measurements are made within each cell:

    * pixel-wise Pearson correlation measures agreement of nuclear shape;
    * nuclear contrast compares DAPI inside the Cellpose nuclear mask with the
      remainder of the Cellpose cell mask.

    The combined score is deliberately conservative. It is high only when the
    two DAPI patterns correlate and their nucleus-to-surrounding contrasts agree.
    Raw components are retained so analyses can apply different criteria later.
    """
    factor = int(downsample)
    if factor < 1:
        raise ValueError("downsample must be at least 1")
    source_arrays = [
        da.asarray(reference),
        da.asarray(moving),
        da.asarray(nuclei_labels),
        da.asarray(cell_labels),
    ]
    if any(array.ndim != 2 for array in source_arrays):
        raise ValueError("Registration quality inputs must all be two-dimensional")
    if len({tuple(array.shape) for array in source_arrays}) != 1:
        raise ValueError(
            "Registration quality inputs must share one image grid, got "
            f"{[tuple(array.shape) for array in source_arrays]}"
        )

    reference_ds = _downsample(source_arrays[0], factor, reducer=np.mean)
    moving_ds = _downsample(source_arrays[1], factor, reducer=np.mean)
    nuclei_ds = _downsample(source_arrays[2], factor, reducer=np.max)
    cells_ds = _downsample(source_arrays[3], factor, reducer=np.max)
    height, width = map(int, reference_ds.shape)
    max_label = int(cells_ds.max().compute())
    size = max_label + 1

    accumulators = {
        name: np.zeros(size, dtype=np.float64)
        for name in (
            "cell_count",
            "reference_sum",
            "moving_sum",
            "reference_square_sum",
            "moving_square_sum",
            "cross_sum",
            "nuclear_count",
            "reference_nuclear_sum",
            "moving_nuclear_sum",
            "surrounding_count",
            "reference_surrounding_sum",
            "moving_surrounding_sum",
        )
    }
    pixel_size_um = float(source_pixel_size_um) * factor
    smoothing_sigma_px = float(smoothing_sigma_um) / pixel_size_um
    halo = max(1, int(math.ceil(4.0 * smoothing_sigma_px)))

    for y0 in range(0, height, int(tile_size)):
        y1 = min(y0 + int(tile_size), height)
        for x0 in range(0, width, int(tile_size)):
            x1 = min(x0 + int(tile_size), width)
            ey0, ey1 = max(0, y0 - halo), min(height, y1 + halo)
            ex0, ex1 = max(0, x0 - halo), min(width, x1 + halo)
            reference_tile, moving_tile, nuclei_tile, cell_tile = da.compute(
                reference_ds[ey0:ey1, ex0:ex1],
                moving_ds[ey0:ey1, ex0:ex1],
                nuclei_ds[ey0:ey1, ex0:ex1],
                cells_ds[ey0:ey1, ex0:ex1],
            )
            reference_tile = gaussian_filter(
                np.log1p(np.maximum(reference_tile.astype(np.float32), 0.0)),
                smoothing_sigma_px,
            )
            moving_tile = gaussian_filter(
                np.log1p(np.maximum(moving_tile.astype(np.float32), 0.0)),
                smoothing_sigma_px,
            )
            core = (
                slice(y0 - ey0, y1 - ey0),
                slice(x0 - ex0, x1 - ex0),
            )
            _accumulate_tile(
                accumulators,
                reference_tile[core],
                moving_tile[core],
                nuclei_tile[core],
                cell_tile[core],
                size=size,
            )

    metrics = _finalize_metrics(
        accumulators,
        min_cell_pixels=int(min_cell_pixels),
        min_nuclear_pixels=int(min_nuclear_pixels),
        min_surrounding_pixels=int(min_surrounding_pixels),
    )
    valid_scores = metrics.loc[metrics["valid"], "score"]
    summary: dict[str, Any] = {
        "algorithm": REGISTRATION_QUALITY_ALGORITHM,
        "pixel_size_um": pixel_size_um,
        "downsample": factor,
        "smoothing_sigma_um": float(smoothing_sigma_um),
        "n_cells": int(len(metrics)),
        "n_valid_cells": int(metrics["valid"].sum()),
        "valid_fraction": float(metrics["valid"].mean()) if len(metrics) else 0.0,
        "score_definition": (
            "sqrt(clip((pearson-0.25)/0.70,0,1) * "
            "clip(1-|moving_contrast-reference_contrast|/0.25,0,1))"
        ),
    }
    if len(valid_scores):
        summary["score_percentiles"] = {
            str(percentile): float(np.percentile(valid_scores, percentile))
            for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        }
    return RegistrationQualityResult(metrics=metrics, summary=summary)


def score_registered_image_in_spatialdata(
    sdata_obj: Any,
    *,
    image_key: str,
    reference_image_key: str,
    reference_channel: str,
    moving_channel: str,
    cell_label_key: str,
    nuclei_label_key: str,
    source_pixel_size_um: float,
    target_pixel_size_um: float,
    smoothing_sigma_um: float,
    output_path: Path,
) -> dict[str, Any]:
    """Compute registration quality and attach it to Cellpose-derived tables."""
    required = {
        "images": [reference_image_key, image_key],
        "labels": [cell_label_key, nuclei_label_key],
    }
    missing = [
        f"{group}/{key}"
        for group, keys in required.items()
        for key in keys
        if key not in getattr(sdata_obj, group)
    ]
    if missing:
        return {
            "status": "skipped_missing_elements",
            "missing_elements": missing,
            "algorithm": REGISTRATION_QUALITY_ALGORITHM,
        }

    ratio = float(target_pixel_size_um) / float(source_pixel_size_um)
    downsample = int(round(ratio))
    if downsample < 1 or not math.isclose(ratio, downsample, rel_tol=0.01):
        raise ValueError(
            "registration_quality_pixel_size_um must be an integer multiple of "
            f"reference_pixel_size_um; got {target_pixel_size_um} / "
            f"{source_pixel_size_um}"
        )
    reference = image_to_cyx(sdata_obj.images[reference_image_key])
    moving = image_to_cyx(sdata_obj.images[image_key])
    reference_index = _channel_index(reference, reference_channel, reference_image_key)
    moving_index = _channel_index(moving, moving_channel, image_key)
    result = compute_cell_registration_quality(
        reference.isel(c=reference_index).data,
        moving.isel(c=moving_index).data,
        _label_array(sdata_obj.labels[nuclei_label_key]),
        _label_array(sdata_obj.labels[cell_label_key]),
        downsample=downsample,
        source_pixel_size_um=source_pixel_size_um,
        smoothing_sigma_um=smoothing_sigma_um,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output_path, index=False, compression="gzip")
    prefix = _column_prefix(image_key)
    updated_tables = []
    for table_key in list(map(str, sdata_obj.tables)):
        if table_key not in REGISTRATION_QUALITY_TABLE_KEYS:
            continue
        table = sdata_obj.tables[table_key]
        identifiers = _cellpose_identifiers(table)
        if identifiers is None:
            continue
        updated = table.copy()
        previous = [
            column for column in updated.obs.columns if str(column).startswith(prefix)
        ]
        if previous:
            updated.obs.drop(columns=previous, inplace=True)
        indexed = result.metrics.set_index("cellpose_label")
        positions = pd.to_numeric(identifiers, errors="coerce")
        for metric in (
            "score",
            "dapi_pearson",
            "reference_nuclear_contrast",
            "moving_nuclear_contrast",
            "nuclear_contrast_agreement",
            "cell_pixel_count",
            "nuclear_pixel_count",
            "surrounding_pixel_count",
        ):
            updated.obs[f"{prefix}{metric}"] = positions.map(indexed[metric]).to_numpy()
        valid_map = positions.map(indexed["valid"])
        updated.obs[f"{prefix}valid"] = valid_map.fillna(False).astype(bool).to_numpy()
        updated.uns[REGISTRATION_QUALITY_UNS_KEY] = {
            **dict(updated.uns.get(REGISTRATION_QUALITY_UNS_KEY, {})),
            str(image_key): {
                **result.summary,
                "reference_image_key": str(reference_image_key),
                "reference_channel": str(reference_channel),
                "moving_channel": str(moving_channel),
                "cell_label_key": str(cell_label_key),
                "nuclei_label_key": str(nuclei_label_key),
                "column_prefix": prefix,
            },
        }
        write_or_replace_element(
            sdata_obj,
            table_key,
            "tables",
            updated,
            overwrite=True,
        )
        updated_tables.append(table_key)

    return {
        "status": "complete",
        **result.summary,
        "updated_tables": updated_tables,
        "output_csv": str(output_path),
        "column_prefix": prefix,
    }


def _downsample(array: da.Array, factor: int, *, reducer: Any) -> da.Array:
    if factor == 1:
        return array
    return cast(
        da.Array,
        da.coarsen(
            reducer,
            array,
            axes={0: factor, 1: factor},
            trim_excess=True,
        ),
    )


def _accumulate_tile(
    accumulators: dict[str, np.ndarray],
    reference: np.ndarray,
    moving: np.ndarray,
    nuclei: np.ndarray,
    cells: np.ndarray,
    *,
    size: int,
) -> None:
    labels = np.asarray(cells, dtype=np.int64)
    cell_mask = labels > 0
    nuclear_mask = cell_mask & (np.asarray(nuclei) > 0)
    surrounding_mask = cell_mask & ~nuclear_mask

    def add(name: str, values: np.ndarray) -> None:
        accumulators[name] += np.bincount(
            labels.ravel(),
            weights=np.asarray(values, dtype=np.float64).ravel(),
            minlength=size,
        )[:size]

    add("cell_count", cell_mask)
    add("reference_sum", reference * cell_mask)
    add("moving_sum", moving * cell_mask)
    add("reference_square_sum", reference * reference * cell_mask)
    add("moving_square_sum", moving * moving * cell_mask)
    add("cross_sum", reference * moving * cell_mask)
    add("nuclear_count", nuclear_mask)
    add("reference_nuclear_sum", reference * nuclear_mask)
    add("moving_nuclear_sum", moving * nuclear_mask)
    add("surrounding_count", surrounding_mask)
    add("reference_surrounding_sum", reference * surrounding_mask)
    add("moving_surrounding_sum", moving * surrounding_mask)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan, dtype=np.float64),
            where=denominator > 0,
        ),
    )


def _finalize_metrics(
    values: dict[str, np.ndarray],
    *,
    min_cell_pixels: int,
    min_nuclear_pixels: int,
    min_surrounding_pixels: int,
) -> pd.DataFrame:
    count = values["cell_count"]
    reference_mean = _safe_divide(values["reference_sum"], count)
    moving_mean = _safe_divide(values["moving_sum"], count)
    covariance = _safe_divide(values["cross_sum"], count) - (
        reference_mean * moving_mean
    )
    reference_variance = _safe_divide(values["reference_square_sum"], count) - (
        reference_mean**2
    )
    moving_variance = _safe_divide(values["moving_square_sum"], count) - (
        moving_mean**2
    )
    denominator = np.sqrt(
        np.maximum(reference_variance, 0.0) * np.maximum(moving_variance, 0.0)
    )
    pearson = _safe_divide(covariance, denominator)

    nuclear_count = values["nuclear_count"]
    surrounding_count = values["surrounding_count"]
    reference_nuclear = _safe_divide(values["reference_nuclear_sum"], nuclear_count)
    moving_nuclear = _safe_divide(values["moving_nuclear_sum"], nuclear_count)
    reference_surrounding = _safe_divide(
        values["reference_surrounding_sum"], surrounding_count
    )
    moving_surrounding = _safe_divide(
        values["moving_surrounding_sum"], surrounding_count
    )
    reference_contrast = (reference_nuclear - reference_surrounding) / (
        reference_nuclear + reference_surrounding + 1e-6
    )
    moving_contrast = (moving_nuclear - moving_surrounding) / (
        moving_nuclear + moving_surrounding + 1e-6
    )
    contrast_agreement = np.clip(
        1.0 - np.abs(moving_contrast - reference_contrast) / 0.25,
        0.0,
        1.0,
    )
    shape_score = np.clip((pearson - 0.25) / 0.70, 0.0, 1.0)
    score = np.sqrt(shape_score * contrast_agreement)
    valid = (
        (count >= min_cell_pixels)
        & (nuclear_count >= min_nuclear_pixels)
        & (surrounding_count >= min_surrounding_pixels)
        & np.isfinite(pearson)
        & np.isfinite(reference_contrast)
        & np.isfinite(moving_contrast)
        & (reference_contrast > 0.0)
    )
    score[~valid] = np.nan

    labels = np.flatnonzero(count > 0)
    labels = labels[labels != 0]
    return pd.DataFrame(
        {
            "cellpose_label": labels.astype(np.uint64),
            "score": score[labels].astype(np.float32),
            "dapi_pearson": pearson[labels].astype(np.float32),
            "reference_nuclear_contrast": reference_contrast[labels].astype(np.float32),
            "moving_nuclear_contrast": moving_contrast[labels].astype(np.float32),
            "nuclear_contrast_agreement": contrast_agreement[labels].astype(np.float32),
            "cell_pixel_count": count[labels].astype(np.uint32),
            "nuclear_pixel_count": nuclear_count[labels].astype(np.uint32),
            "surrounding_pixel_count": surrounding_count[labels].astype(np.uint32),
            "valid": valid[labels],
        }
    )


def _channel_index(image: Any, channel: str, image_key: str) -> int:
    names = (
        [str(value) for value in image.coords["c"].values]
        if "c" in image.coords
        else [f"c{index}" for index in range(int(image.sizes["c"]))]
    )
    lowered = [name.lower() for name in names]
    wanted = str(channel).lower()
    if wanted not in lowered:
        raise ValueError(
            f"Channel {channel!r} not found in image {image_key!r}; channels: {names}"
        )
    return lowered.index(wanted)


def _label_array(label: Any) -> Any:
    if hasattr(label, "children"):
        names = sorted(
            label.children,
            key=lambda name: int(re.sub(r"\D", "", str(name)) or 0),
        )
        label = label[names[0]].ds["image"]
    return label.data if hasattr(label, "data") else label


def _cellpose_identifiers(table: Any) -> pd.Series | None:
    if "cellpose_label" in table.obs.columns:
        return table.obs["cellpose_label"]
    if "region" not in table.obs.columns or "instance_id" not in table.obs.columns:
        return None
    regions = table.obs["region"].astype(str)
    if not bool((regions == "MOSAIK_cellpose").all()):
        return None
    return table.obs["instance_id"]


def _column_prefix(image_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_key)).strip("_")
    return f"registration_quality__{safe}__"
