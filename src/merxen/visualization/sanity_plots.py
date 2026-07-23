"""Sanity overlay plotting helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as path_effects
from matplotlib.lines import Line2D
from shapely.geometry import box as shapely_box

from merxen.io.spatialdata_schema import choose_primary_points_key
from merxen.io.transcript_io import assignment_mask_from_points
from merxen.plotting import prepare_plot_output, save_figure

SANITY_CROP_SIZE_UM = 250.0
SANITY_MAX_TRANSCRIPTS_PER_PANEL = 2_000_000
SANITY_RANDOM_STATE = 42
SANITY_ASSIGNMENT_SHAPE_KEY = "MOSAIK_proseg"
SANITY_SCALE_BAR_UM = 100.0
SANITY_CROP_LOCATION_SAMPLE_N = 2_000
SANITY_SHAPE_STYLES: dict[str, tuple[str, str]] = {
    "MOSAIK_proseg": ("ProSeg", "#2ca02c"),
    "MOSAIK_cellpose": ("Cellpose-SAM", "#9467bd"),
    "merscope_cell_boundaries": ("Original segmentation", "#ff7f0e"),
    "xenium_cell_boundaries": ("Original segmentation", "#ff7f0e"),
}
SANITY_SHAPE_DRAW_ORDER: tuple[str, ...] = (
    "merscope_cell_boundaries",
    "xenium_cell_boundaries",
    "MOSAIK_cellpose",
    "MOSAIK_proseg",
)
SANITY_SHAPE_LEGEND_ORDER: tuple[str, ...] = (
    "ProSeg",
    "Cellpose-SAM",
    "Original segmentation",
)


@dataclass(frozen=True)
class _SanityPanelPlan:
    dataset_name: str
    display_bbox: tuple[float, float, float, float]
    aligned_bbox: tuple[float, float, float, float]
    prefer_aligned_vectors: bool


def _prepare_overlay_image(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Convert microscopy image data into a display-safe grayscale or RGB array."""
    arr = np.asarray(image)

    if arr.ndim == 2:
        return _normalize_channel(arr), "gray"

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image data, got shape {arr.shape}")

    if arr.shape[-1] == 1:
        return _normalize_channel(arr[..., 0]), "gray"

    arr = arr.astype(np.float32, copy=False)
    if arr.shape[-1] == 2:
        arr = np.concatenate([arr, np.zeros_like(arr[..., :1])], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    return _normalize_channel(arr), None


def _normalize_channel(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize an array to the [0, 1] range for plotting."""
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    values = arr[finite]
    lo, hi = np.percentile(values, (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.asarray(np.clip((arr - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)
    scaled[~finite] = 0.0
    return scaled


def plot_sanity_overlay(
    image: np.ndarray,
    output_path: Path | str,
    *,
    shapes: gpd.GeoDataFrame | None = None,
    points: pd.DataFrame | None = None,
    x_col: str = "x",
    y_col: str = "y",
    title: str = "Sanity Overlay",
    point_size: float = 1.5,
) -> Path:
    """Overlay shape boundaries and transcript points on an image crop."""
    output_path = prepare_plot_output(output_path)
    display_image, cmap = _prepare_overlay_image(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display_image, cmap=cmap)

    if shapes is not None and len(shapes) > 0:
        shapes.boundary.plot(ax=ax, linewidth=0.5, color="#F97316", alpha=0.85)

    if points is not None and len(points) > 0:
        ax.scatter(
            points[x_col].to_numpy(float),
            points[y_col].to_numpy(float),
            s=point_size,
            c="#06B6D4",
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_pair_sanity_crops(
    merscope_sdata: Any,
    xenium_sdata: Any,
    output_path: Path | str,
    *,
    merscope_zarr_path: Path | str | None = None,
    xenium_zarr_path: Path | str | None = None,
    crop_size_um: float = SANITY_CROP_SIZE_UM,
    assignment_shape_key: str | None = SANITY_ASSIGNMENT_SHAPE_KEY,
    merscope_assignment_shape_key: str | None = None,
    xenium_assignment_shape_key: str | None = None,
) -> Path:
    """Plot paired MOSAIK-style 250 um sanity crops for MERSCOPE and Xenium."""
    output_path = prepare_plot_output(output_path)
    merscope_plan, xenium_plan = _build_pair_sanity_plans(
        merscope_sdata,
        xenium_sdata,
        crop_size_um=crop_size_um,
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    plot_sanity_crop_panel(
        axes[0],
        merscope_sdata,
        "MERSCOPE",
        crop_bbox=merscope_plan.display_bbox,
        crop_size_um=crop_size_um,
        assignment_shape_key=(
            merscope_assignment_shape_key
            if merscope_assignment_shape_key is not None
            else assignment_shape_key
        ),
        prefer_aligned_vectors=merscope_plan.prefer_aligned_vectors,
        zarr_path=merscope_zarr_path,
    )
    plot_sanity_crop_panel(
        axes[1],
        xenium_sdata,
        "XENIUM",
        crop_bbox=xenium_plan.display_bbox,
        crop_size_um=crop_size_um,
        assignment_shape_key=(
            xenium_assignment_shape_key
            if xenium_assignment_shape_key is not None
            else assignment_shape_key
        ),
        prefer_aligned_vectors=xenium_plan.prefer_aligned_vectors,
        zarr_path=xenium_zarr_path,
    )
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)

    crop_location_path = output_path.with_name(
        f"{output_path.stem}_crop_location{output_path.suffix}"
    )
    plot_pair_crop_location(
        merscope_sdata,
        xenium_sdata,
        crop_location_path,
        merscope_plan=merscope_plan,
        xenium_plan=xenium_plan,
    )
    return output_path


def plot_single_sanity_crop(
    sdata_obj: Any,
    dataset_name: str,
    output_path: Path | str,
    *,
    zarr_path: Path | str | None = None,
    crop_size_um: float = SANITY_CROP_SIZE_UM,
    assignment_shape_key: str | None = SANITY_ASSIGNMENT_SHAPE_KEY,
) -> Path:
    """Plot one MOSAIK-style sanity crop for a single platform dataset."""
    output_path = prepare_plot_output(output_path)
    prefer_aligned = _has_aligned_vectors(sdata_obj)
    crop_bbox, _ = _choose_crop_bbox(
        sdata_obj,
        size_um=crop_size_um,
        center_xy=None,
        prefer_aligned=prefer_aligned,
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 8), constrained_layout=True)
    plot_sanity_crop_panel(
        ax,
        sdata_obj,
        dataset_name,
        crop_bbox=crop_bbox,
        crop_size_um=crop_size_um,
        assignment_shape_key=assignment_shape_key,
        prefer_aligned_vectors=prefer_aligned,
        zarr_path=zarr_path,
    )
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)

    crop_location_path = output_path.with_name(
        f"{output_path.stem}_crop_location{output_path.suffix}"
    )
    plot_single_crop_location(
        sdata_obj,
        dataset_name,
        crop_location_path,
        crop_bbox=crop_bbox,
        prefer_aligned_vectors=prefer_aligned,
    )
    return output_path


def plot_sanity_crop_panel(
    ax: plt.Axes,
    sdata_obj: Any,
    dataset_name: str,
    *,
    crop_size_um: float = SANITY_CROP_SIZE_UM,
    crop_bbox: tuple[float, float, float, float] | None = None,
    center_xy: tuple[float, float] | None = None,
    assignment_shape_key: str | None = SANITY_ASSIGNMENT_SHAPE_KEY,
    prefer_aligned_vectors: bool = True,
    zarr_path: Path | str | None = None,
) -> None:
    """Draw one MOSAIK-style image, shape, and assignment sanity crop panel."""
    if crop_bbox is None:
        bbox, ref_shape_key = _choose_crop_bbox(
            sdata_obj,
            size_um=crop_size_um,
            center_xy=center_xy,
            prefer_aligned=prefer_aligned_vectors,
        )
    else:
        bbox = crop_bbox
        ref_shape_key = _reference_shape_key(
            sdata_obj,
            prefer_aligned=prefer_aligned_vectors,
        )
    x0, y0, x1, y1 = bbox

    bg = _get_background_image_crop(sdata_obj, dataset_name, bbox, zarr_path=zarr_path)
    if bg is not None:
        ax.imshow(
            bg["rgb"],
            extent=bg["extent_um"],
            origin="lower",
            interpolation="nearest",
            alpha=0.95,
        )

    shape_keys = _ordered_sanity_shape_keys(
        sdata_obj,
        dataset_name=dataset_name,
        prefer_aligned=prefer_aligned_vectors,
    )
    shape_handles = []
    legend_labels_seen: set[str] = set()

    for shape_key in shape_keys:
        label, color = _sanity_shape_style(shape_key)
        try:
            shp_crop = _crop_single_shape(sdata_obj, shape_key, bbox)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{dataset_name}] Warning: failed to crop shapes[{shape_key}] ({exc})"
            )
            continue

        if len(shp_crop) == 0:
            continue

        shp_crop.boundary.plot(
            ax=ax,
            linewidth=0.75,
            color=color,
            alpha=0.95,
            zorder=_sanity_shape_zorder(shape_key),
        )
        if label in legend_labels_seen:
            continue
        legend_labels_seen.add(label)
        shape_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                label=f"{label} ({len(shp_crop):,})",
            )
        )

    tx_crop, _points_key, _assign_col = _crop_points(
        sdata_obj,
        bbox,
        max_points=SANITY_MAX_TRANSCRIPTS_PER_PANEL,
        random_state=SANITY_RANDOM_STATE,
        assignment_shape_key=assignment_shape_key,
        prefer_aligned_points=prefer_aligned_vectors,
        prefer_aligned_assignment=prefer_aligned_vectors,
    )

    tx_handles = []
    if len(tx_crop) > 0:
        unassigned = tx_crop[~tx_crop["assigned"]]
        assigned = tx_crop[tx_crop["assigned"]]

        if len(unassigned) > 0:
            ax.scatter(
                unassigned["x_um"],
                unassigned["y_um"],
                s=4,
                c="#d62728",
                alpha=0.50,
                rasterized=True,
            )
            tx_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color="#d62728",
                    label=f"Unassigned tx ({len(unassigned):,})",
                    markersize=5,
                )
            )

        if len(assigned) > 0:
            ax.scatter(
                assigned["x_um"],
                assigned["y_um"],
                s=4,
                c="yellow",
                alpha=0.50,
                rasterized=True,
            )
            tx_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color="yellow",
                    label=f"Assigned tx ({len(assigned):,})",
                    markersize=5,
                )
            )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(str(dataset_name).upper())
    if str(dataset_name).upper() == "XENIUM":
        _add_scale_bar(ax, bbox, length_um=SANITY_SCALE_BAR_UM)

    handles = tx_handles + shape_handles
    if handles:
        handles = _ordered_legend_handles(handles)
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)

    if ref_shape_key not in shape_keys:
        print(f"[{dataset_name}] Warning: reference shape {ref_shape_key} not plotted")


def plot_pair_crop_location(
    merscope_sdata: Any,
    xenium_sdata: Any,
    output_path: Path | str,
    *,
    merscope_plan: _SanityPanelPlan,
    xenium_plan: _SanityPanelPlan,
) -> Path:
    """Plot crop rectangles used for the paired sanity overlay."""
    output_path = prepare_plot_output(output_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    _plot_crop_location_panel(
        axes[0],
        merscope_sdata,
        "MERSCOPE raw",
        merscope_plan.display_bbox,
        prefer_aligned_vectors=False,
        dataset_name="MERSCOPE",
    )
    _plot_crop_location_panel(
        axes[1],
        merscope_sdata,
        "MERSCOPE aligned",
        merscope_plan.aligned_bbox,
        prefer_aligned_vectors=True,
        dataset_name="MERSCOPE",
    )
    _plot_crop_location_panel(
        axes[2],
        xenium_sdata,
        "XENIUM",
        xenium_plan.display_bbox,
        prefer_aligned_vectors=False,
        dataset_name="XENIUM",
    )
    save_figure(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_single_crop_location(
    sdata_obj: Any,
    dataset_name: str,
    output_path: Path | str,
    *,
    crop_bbox: tuple[float, float, float, float],
    prefer_aligned_vectors: bool,
) -> Path:
    """Plot the crop rectangle used for a single-platform sanity crop."""
    output_path = prepare_plot_output(output_path)
    label = str(dataset_name).upper()
    title = f"{label} {'aligned' if prefer_aligned_vectors else 'crop location'}"

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5), constrained_layout=True)
    _plot_crop_location_panel(
        ax,
        sdata_obj,
        title,
        crop_bbox,
        prefer_aligned_vectors=prefer_aligned_vectors,
        dataset_name=label,
    )
    save_figure(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def _build_pair_sanity_plans(
    merscope_sdata: Any,
    xenium_sdata: Any,
    *,
    crop_size_um: float,
) -> tuple[_SanityPanelPlan, _SanityPanelPlan]:
    if _has_aligned_vectors(merscope_sdata):
        shared_bbox = _choose_shared_aligned_bbox(
            merscope_sdata,
            xenium_sdata,
            crop_size_um=crop_size_um,
        )
        merscope_display_bbox = _merscope_raw_bbox_from_aligned_crop(
            merscope_sdata,
            shared_bbox,
            crop_size_um=crop_size_um,
        )
        return (
            _SanityPanelPlan(
                dataset_name="MERSCOPE",
                display_bbox=merscope_display_bbox,
                aligned_bbox=shared_bbox,
                prefer_aligned_vectors=False,
            ),
            _SanityPanelPlan(
                dataset_name="XENIUM",
                display_bbox=shared_bbox,
                aligned_bbox=shared_bbox,
                prefer_aligned_vectors=False,
            ),
        )

    merscope_bbox, _ = _choose_crop_bbox(
        merscope_sdata,
        size_um=crop_size_um,
        center_xy=None,
        prefer_aligned=False,
    )
    xenium_bbox, _ = _choose_crop_bbox(
        xenium_sdata,
        size_um=crop_size_um,
        center_xy=None,
        prefer_aligned=False,
    )
    return (
        _SanityPanelPlan(
            dataset_name="MERSCOPE",
            display_bbox=merscope_bbox,
            aligned_bbox=merscope_bbox,
            prefer_aligned_vectors=False,
        ),
        _SanityPanelPlan(
            dataset_name="XENIUM",
            display_bbox=xenium_bbox,
            aligned_bbox=xenium_bbox,
            prefer_aligned_vectors=False,
        ),
    )


def _has_aligned_vectors(sdata_obj: Any) -> bool:
    return any(
        str(key).endswith("_aligned_nonrigid") for key in sdata_obj.shapes
    ) or any(str(key).endswith("_aligned_nonrigid") for key in sdata_obj.points)


def _choose_shared_aligned_bbox(
    merscope_sdata: Any,
    xenium_sdata: Any,
    *,
    crop_size_um: float,
) -> tuple[float, float, float, float]:
    merscope_key = _reference_shape_key(merscope_sdata, prefer_aligned=True)
    xenium_key = _reference_shape_key(xenium_sdata, prefer_aligned=False)
    m_bounds = _shape_geometry_only(merscope_sdata.shapes[merscope_key]).total_bounds
    x_bounds = _shape_geometry_only(xenium_sdata.shapes[xenium_key]).total_bounds

    overlap = (
        max(float(m_bounds[0]), float(x_bounds[0])),
        max(float(m_bounds[1]), float(x_bounds[1])),
        min(float(m_bounds[2]), float(x_bounds[2])),
        min(float(m_bounds[3]), float(x_bounds[3])),
    )
    if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
        overlap = (
            float(x_bounds[0]),
            float(x_bounds[1]),
            float(x_bounds[2]),
            float(x_bounds[3]),
        )

    cx = 0.5 * (overlap[0] + overlap[2])
    cy = 0.5 * (overlap[1] + overlap[3])
    x0, x1 = _bounded_interval(cx, crop_size_um, overlap[0], overlap[2])
    y0, y1 = _bounded_interval(cy, crop_size_um, overlap[1], overlap[3])
    return (x0, y0, x1, y1)


def _merscope_raw_bbox_from_aligned_crop(
    sdata_obj: Any,
    aligned_bbox: tuple[float, float, float, float],
    *,
    crop_size_um: float,
) -> tuple[float, float, float, float]:
    raw_center = _raw_center_from_aligned_points(sdata_obj, aligned_bbox)
    raw_ref_key = _reference_shape_key(sdata_obj, prefer_aligned=False)
    raw_bounds = _shape_geometry_only(sdata_obj.shapes[raw_ref_key]).total_bounds
    if raw_center is None:
        raw_center = (
            0.5 * (float(raw_bounds[0]) + float(raw_bounds[2])),
            0.5 * (float(raw_bounds[1]) + float(raw_bounds[3])),
        )

    x0, x1 = _bounded_interval(
        raw_center[0],
        crop_size_um,
        float(raw_bounds[0]),
        float(raw_bounds[2]),
    )
    y0, y1 = _bounded_interval(
        raw_center[1],
        crop_size_um,
        float(raw_bounds[1]),
        float(raw_bounds[3]),
    )
    return (x0, y0, x1, y1)


def _raw_center_from_aligned_points(
    sdata_obj: Any,
    aligned_bbox: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    points_key = _reference_points_key(sdata_obj, prefer_aligned=True)
    pts = sdata_obj.points[points_key]
    x_col = _first_existing_col(pts, ["x", "x_micron", "global_x", "x_location"])
    y_col = _first_existing_col(pts, ["y", "y_micron", "global_y", "y_location"])
    raw_x_col = _first_existing_col(pts, ["raw_x"])
    raw_y_col = _first_existing_col(pts, ["raw_y"])
    if x_col is None or y_col is None or raw_x_col is None or raw_y_col is None:
        return None

    x0, y0, x1, y1 = aligned_bbox
    cols = [x_col, y_col, raw_x_col, raw_y_col]
    if hasattr(pts, "npartitions") and hasattr(pts, "partitions"):
        work = pts[cols]
        work = work[
            (work[x_col] >= x0)
            & (work[x_col] <= x1)
            & (work[y_col] >= y0)
            & (work[y_col] <= y1)
        ]
        pdf = work.compute()
    else:
        pdf = pd.DataFrame(pts[cols]).copy()
        pdf = pdf[
            (pdf[x_col] >= x0)
            & (pdf[x_col] <= x1)
            & (pdf[y_col] >= y0)
            & (pdf[y_col] <= y1)
        ]
    if pdf.empty:
        return None

    raw_x = pd.to_numeric(pdf[raw_x_col], errors="coerce")
    raw_y = pd.to_numeric(pdf[raw_y_col], errors="coerce")
    valid = np.isfinite(raw_x) & np.isfinite(raw_y)
    if not valid.any():
        return None
    return (float(raw_x[valid].median()), float(raw_y[valid].median()))


def _plot_crop_location_panel(
    ax: plt.Axes,
    sdata_obj: Any,
    title: str,
    bbox: tuple[float, float, float, float],
    *,
    prefer_aligned_vectors: bool,
    dataset_name: str,
) -> None:
    key = _reference_shape_key(sdata_obj, prefer_aligned=prefer_aligned_vectors)
    gdf = _shape_geometry_only(sdata_obj.shapes[key])
    if len(gdf) > SANITY_CROP_LOCATION_SAMPLE_N:
        gdf = gdf.sample(
            n=SANITY_CROP_LOCATION_SAMPLE_N,
            random_state=SANITY_RANDOM_STATE,
        )
    gdf.boundary.plot(ax=ax, linewidth=0.25, color="#6b7280", alpha=0.35)
    _draw_bbox(ax, bbox, color="#ef4444", linewidth=2.0)
    bounds = _shape_geometry_only(sdata_obj.shapes[key]).total_bounds
    ax.set_xlim(float(bounds[0]), float(bounds[2]))
    ax.set_ylim(float(bounds[1]), float(bounds[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if dataset_name.upper() == "MERSCOPE" and prefer_aligned_vectors:
        ax.text(
            0.02,
            0.98,
            "same aligned bbox as XENIUM",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )


def _draw_bbox(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    color: str,
    linewidth: float,
) -> None:
    x0, y0, x1, y1 = bbox
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=10)


def _ordered_sanity_shape_keys(
    sdata_obj: Any,
    *,
    dataset_name: str,
    prefer_aligned: bool,
) -> list[str]:
    available: dict[str, str] = {}
    suffix = "_aligned_nonrigid"
    for key_obj in sdata_obj.shapes:
        key = str(key_obj)
        canonical = _canonical_shape_key(key)
        if canonical not in SANITY_SHAPE_STYLES:
            continue
        if canonical.endswith("cell_boundaries"):
            if (
                dataset_name.upper() == "MERSCOPE"
                and canonical != "merscope_cell_boundaries"
            ):
                continue
            if (
                dataset_name.upper() == "XENIUM"
                and canonical != "xenium_cell_boundaries"
            ):
                continue
        is_aligned = key.endswith(suffix)
        if prefer_aligned != is_aligned:
            continue
        available[canonical] = key

    if not available and prefer_aligned:
        return _ordered_sanity_shape_keys(
            sdata_obj,
            dataset_name=dataset_name,
            prefer_aligned=False,
        )

    order = {name: i for i, name in enumerate(SANITY_SHAPE_DRAW_ORDER)}
    return [
        available[name]
        for name in sorted(available, key=lambda name: order.get(name, len(order)))
    ]


def _sanity_shape_style(shape_key: str) -> tuple[str, str]:
    canonical = _canonical_shape_key(shape_key)
    if canonical in SANITY_SHAPE_STYLES:
        return SANITY_SHAPE_STYLES[canonical]
    return (canonical, _fallback_shape_color(canonical))


def _sanity_shape_zorder(shape_key: str) -> int:
    canonical = _canonical_shape_key(shape_key)
    order = {name: i for i, name in enumerate(SANITY_SHAPE_DRAW_ORDER)}
    return 3 + order.get(canonical, 0)


def _ordered_legend_handles(handles: list[Line2D]) -> list[Line2D]:
    order = {label: i for i, label in enumerate(SANITY_SHAPE_LEGEND_ORDER)}

    def _label(handle: Line2D) -> str:
        return str(handle.get_label()).split(" (", 1)[0]

    def _handle_key(handle: Line2D) -> tuple[int, str]:
        return (order.get(_label(handle), -1), str(handle.get_label()))

    tx_handles = [h for h in handles if order.get(_label(h)) is None]
    shape_handles = [h for h in handles if order.get(_label(h)) is not None]
    return tx_handles + sorted(shape_handles, key=_handle_key)


def _canonical_shape_key(shape_key: str) -> str:
    suffix = "_aligned_nonrigid"
    key = str(shape_key)
    if key.endswith(suffix):
        return key[: -len(suffix)]
    return key


def _fallback_shape_color(shape_key: str) -> str:
    fallback_colors = (
        "#1f77b4",
        "#d62728",
        "#17becf",
        "#bcbd22",
        "#8c564b",
        "#e377c2",
    )
    return fallback_colors[sum(ord(ch) for ch in shape_key) % len(fallback_colors)]


def _add_scale_bar(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    length_um: float,
) -> None:
    x0, y0, x1, y1 = bbox
    x_span = x1 - x0
    y_span = y1 - y0
    if x_span <= 0 or y_span <= 0:
        return

    shown_length = min(length_um, x_span * 0.8)
    pad_x = x_span * 0.06
    pad_y = y_span * 0.06
    x_end = x1 - pad_x
    x_start = x_end - shown_length
    y = y0 + pad_y
    text_effects = [
        path_effects.Stroke(linewidth=2.5, foreground="black"),
        path_effects.Normal(),
    ]
    line_effects = [
        path_effects.Stroke(linewidth=6, foreground="black"),
        path_effects.Normal(),
    ]
    ax.plot(
        [x_start, x_end],
        [y, y],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
        path_effects=line_effects,
        zorder=10,
    )
    ax.text(
        0.5 * (x_start + x_end),
        y + y_span * 0.025,
        f"{shown_length:.0f} um",
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
        path_effects=text_effects,
        zorder=10,
    )


def _shape_geometry_only(shape_obj: Any) -> gpd.GeoDataFrame:
    if "geometry" in shape_obj.columns:
        gdf = shape_obj[["geometry"]].copy()
    else:
        gdf = gpd.GeoDataFrame({"geometry": shape_obj.geometry}, index=shape_obj.index)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def _reference_shape_key_with_mode(
    sdata_obj: Any,
    *,
    prefer_aligned: bool,
) -> str:
    suffix = "_aligned_nonrigid"
    candidates = [
        "MOSAIK_proseg",
        "MOSAIK_cellpose",
        "merscope_cell_boundaries",
        "xenium_cell_boundaries",
    ]
    if prefer_aligned:
        for base_key in candidates:
            key = f"{base_key}{suffix}"
            if key in sdata_obj.shapes:
                return key
    for key in candidates:
        if key in sdata_obj.shapes:
            return key
    if prefer_aligned:
        for key in sdata_obj.shapes:
            if str(key).endswith(suffix) and "nucleus" not in str(key).lower():
                return str(key)
    for key in sdata_obj.shapes:
        if "nucleus" not in str(key).lower() and str(key) != "cell_boundaries":
            return str(key)
    if len(sdata_obj.shapes) == 0:
        raise RuntimeError("No shapes found in SpatialData object.")
    return str(list(sdata_obj.shapes.keys())[0])


def _reference_shape_key(
    sdata_obj: Any,
    *,
    prefer_aligned: bool = True,
) -> str:
    return _reference_shape_key_with_mode(sdata_obj, prefer_aligned=prefer_aligned)


def _bounded_interval(
    center: float,
    size: float,
    min_v: float,
    max_v: float,
) -> tuple[float, float]:
    span = max_v - min_v
    if span <= size:
        return float(min_v), float(max_v)
    half = size / 2.0
    lo = center - half
    hi = center + half
    if lo < min_v:
        hi += min_v - lo
        lo = min_v
    if hi > max_v:
        lo -= hi - max_v
        hi = max_v
    return float(lo), float(hi)


def _choose_crop_bbox(
    sdata_obj: Any,
    *,
    size_um: float,
    center_xy: tuple[float, float] | None,
    prefer_aligned: bool = True,
) -> tuple[tuple[float, float, float, float], str]:
    ref_key = _reference_shape_key(sdata_obj, prefer_aligned=prefer_aligned)
    gdf = _shape_geometry_only(sdata_obj.shapes[ref_key])
    if len(gdf) == 0:
        raise RuntimeError(f"No non-empty geometries in shapes[{ref_key}]")

    minx, miny, maxx, maxy = gdf.total_bounds
    if center_xy is None:
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
    else:
        cx, cy = center_xy

    x0, x1 = _bounded_interval(cx, size_um, minx, maxx)
    y0, y1 = _bounded_interval(cy, size_um, miny, maxy)
    return (x0, y0, x1, y1), ref_key


def _crop_single_shape(
    sdata_obj: Any,
    shape_key: str,
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    gdf = _shape_geometry_only(sdata_obj.shapes[shape_key])
    crop_poly = shapely_box(*bbox)
    keep = gdf.geometry.intersects(crop_poly)
    return gdf.loc[keep].copy()


def _assign_points_by_shape(
    points_pdf: pd.DataFrame,
    shapes_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    n_points = len(points_pdf)
    if n_points == 0 or len(shapes_gdf) == 0:
        return np.zeros(n_points, dtype=bool)

    pts_gdf = gpd.GeoDataFrame(
        {"_row_id": np.arange(n_points, dtype=np.int64)},
        geometry=gpd.points_from_xy(
            points_pdf["x_um"].to_numpy(),
            points_pdf["y_um"].to_numpy(),
        ),
    )
    shp = shapes_gdf[["geometry"]].copy()

    try:
        joined = gpd.sjoin(
            pts_gdf[["_row_id", "geometry"]],
            shp,
            how="left",
            predicate="within",
        )
        matched = joined.loc[
            joined["index_right"].notna(),
            "_row_id",
        ].to_numpy(dtype=np.int64, copy=False)
        assigned = np.zeros(n_points, dtype=bool)
        if matched.size:
            assigned[np.unique(matched)] = True
        return assigned
    except Exception:  # noqa: BLE001
        return _assign_points_by_shape_fallback(pts_gdf, shp)


def _assign_points_by_shape_fallback(
    pts_gdf: gpd.GeoDataFrame,
    shapes_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    assigned = np.zeros(len(pts_gdf), dtype=bool)
    try:
        sindex = shapes_gdf.sindex
        use_sindex = sindex is not None
    except Exception:  # noqa: BLE001
        sindex = None
        use_sindex = False

    for i, point in enumerate(pts_gdf.geometry.values):
        candidates = (
            list(sindex.intersection(point.bounds))
            if use_sindex
            else range(len(shapes_gdf))
        )
        if not candidates:
            continue
        assigned[i] = any(shapes_gdf.geometry.iloc[j].covers(point) for j in candidates)
    return assigned


def _crop_points(
    sdata_obj: Any,
    bbox: tuple[float, float, float, float],
    *,
    max_points: int | None,
    random_state: int,
    assignment_shape_key: str | None,
    prefer_aligned_points: bool = True,
    prefer_aligned_assignment: bool = True,
) -> tuple[pd.DataFrame, str, str | None]:
    if len(sdata_obj.points) == 0:
        raise RuntimeError("No points found in SpatialData object.")

    points_key = _reference_points_key(
        sdata_obj,
        prefer_aligned=prefer_aligned_points,
    )
    pts = sdata_obj.points[points_key]
    x_col = _first_existing_col(
        pts,
        ["x", "global_x", "x_location", "x_micron", "observed_x", "x_global_px"],
    )
    y_col = _first_existing_col(
        pts,
        ["y", "global_y", "y_location", "y_micron", "observed_y", "y_global_px"],
    )
    assign_col = _first_existing_col(pts, ["assignment", "cell", "cell_id"])
    background_col = _first_existing_col(pts, ["background"])
    if x_col is None or y_col is None:
        raise KeyError(f"Could not resolve x/y columns in points[{points_key}]")
    if (
        prefer_aligned_assignment
        and assignment_shape_key is not None
        and f"{assignment_shape_key}_aligned_nonrigid" in sdata_obj.shapes
    ):
        assignment_shape_key = f"{assignment_shape_key}_aligned_nonrigid"

    x0, y0, x1, y1 = bbox
    cols = [x_col, y_col]
    if assign_col is not None:
        cols.append(assign_col)
    if background_col is not None and background_col not in cols:
        cols.append(background_col)

    if hasattr(pts, "npartitions") and hasattr(pts, "partitions"):
        work = pts[cols]
        work = work[
            (work[x_col] >= x0)
            & (work[x_col] <= x1)
            & (work[y_col] >= y0)
            & (work[y_col] <= y1)
        ]
        pdf = work.compute()
    else:
        pdf = pd.DataFrame(pts[cols]).copy()
        pdf = pdf[
            (pdf[x_col] >= x0)
            & (pdf[x_col] <= x1)
            & (pdf[y_col] >= y0)
            & (pdf[y_col] <= y1)
        ].copy()

    pdf = pdf.rename(columns={x_col: "x_um", y_col: "y_um"})

    if assignment_shape_key is not None:
        if assignment_shape_key not in sdata_obj.shapes:
            raise KeyError(
                f"assignment_shape_key='{assignment_shape_key}' not found in shapes. "
                f"Available: {list(sdata_obj.shapes.keys())}"
            )
        shape_crop = _crop_single_shape(sdata_obj, assignment_shape_key, bbox)
        pdf["assigned"] = _assign_points_by_shape(pdf, shape_crop)
        assign_col = f"shape:{assignment_shape_key}"
    elif (assign_col is not None and assign_col in pdf.columns) or (
        background_col is not None and background_col in pdf.columns
    ):
        pdf["assigned"] = assignment_mask_from_points(
            pdf,
            assign_col=assign_col,
            background_col=background_col or "background",
        ).values
        assign_col = assign_col or background_col
    else:
        pdf["assigned"] = True

    if max_points is not None and len(pdf) > max_points:
        pdf = pdf.sample(n=max_points, random_state=random_state)

    return pdf, points_key, assign_col


def _reference_points_key(
    sdata_obj: Any,
    *,
    prefer_aligned: bool = True,
) -> str:
    primary = choose_primary_points_key(sdata_obj)
    if primary is None:
        raise RuntimeError("No transcript points found in SpatialData object.")
    if prefer_aligned:
        aligned = f"{primary}_aligned_nonrigid"
        if aligned in sdata_obj.points:
            return aligned
    return primary


def _get_scale0_dataarray(image_elem: Any) -> Any:
    if hasattr(image_elem, "keys") and "scale0" in image_elem:
        node = image_elem["scale0"]
        if hasattr(node, "ds"):
            if "image" in node.ds:
                return node.ds["image"]
            if len(node.ds.data_vars) > 0:
                return next(iter(node.ds.data_vars.values()))
    if hasattr(image_elem, "ds"):
        if "image" in image_elem.ds:
            return image_elem.ds["image"]
        if len(image_elem.ds.data_vars) > 0:
            return next(iter(image_elem.ds.data_vars.values()))
    return image_elem


def _pick_channel_name(channel_labels: list[str], preferred: list[str]) -> str | None:
    lower = [str(label).lower() for label in channel_labels]
    for preferred_name in preferred:
        preferred_lower = preferred_name.lower()
        for i, label in enumerate(lower):
            if label == preferred_lower or preferred_lower in label:
                return channel_labels[i]
    return None


def _norm01(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(values[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return np.asarray(np.clip((values - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)


def _get_background_image_crop(
    sdata_obj: Any,
    dataset_name: str,
    bbox: tuple[float, float, float, float],
    *,
    zarr_path: Path | str | None,
) -> dict[str, Any] | None:
    if len(sdata_obj.images) == 0:
        return None

    if dataset_name.upper() == "MERSCOPE":
        image_key = _pick_merscope_image_key(sdata_obj.images)
        ch2_pref = ["PolyT", "18S"]
    else:
        image_key = (
            "morphology_focus"
            if "morphology_focus" in sdata_obj.images
            else list(sdata_obj.images.keys())[0]
        )
        ch2_pref = ["18S", "PolyT"]

    try:
        da = _get_scale0_dataarray(sdata_obj.images[image_key])
        x_transform, y_transform = _resolve_dataset_mask_affine(
            dataset_name,
            zarr_path=zarr_path,
        )
        crop = _crop_image_dataarray_to_bbox(da, bbox, x_transform, y_transform)
    except Exception as exc:  # noqa: BLE001
        print(f"[{dataset_name}] Warning: failed to crop background image ({exc})")
        return None

    if crop is None:
        return None

    channels = [str(c) for c in crop.coords["c"].values] if "c" in crop.coords else []
    ch_dapi = _pick_channel_name(channels, ["DAPI"]) if channels else None
    ch_rna = _pick_channel_name(channels, ch2_pref) if channels else None

    if "c" in crop.dims:
        dapi_da = crop.sel(c=ch_dapi) if ch_dapi is not None else crop.isel(c=0)
        if ch_rna is not None:
            rna_da = crop.sel(c=ch_rna)
        else:
            rna_da = crop.isel(c=1 if crop.sizes["c"] > 1 else 0)
    else:
        dapi_da = crop
        rna_da = crop

    dapi = _norm01(_to_numpy(dapi_da))
    rna = _norm01(_to_numpy(rna_da))
    rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype=np.float32)
    rgb[..., 2] = dapi
    rgb[..., 1] = rna

    extent_um = _crop_extent_um(crop, x_transform, y_transform)
    return {
        "image_key": image_key,
        "channels_used": {"dapi": ch_dapi, "rna_like": ch_rna},
        "rgb": rgb,
        "extent_um": extent_um,
        "transform": {"x_transform": x_transform, "y_transform": y_transform},
    }


def _pick_merscope_image_key(images: Any) -> str:
    if "MERSCOPE_z_projection" in images:
        return "MERSCOPE_z_projection"
    projection_key = next(
        (key for key in images if "projection" in str(key).lower()),
        None,
    )
    if projection_key is not None:
        return str(projection_key)
    return str(list(images.keys())[0])


def _crop_image_dataarray_to_bbox(
    da: Any,
    bbox: tuple[float, float, float, float],
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> Any | None:
    x0_um, y0_um, x1_um, y1_um = bbox
    corners_um = [
        (x0_um, y0_um),
        (x0_um, y1_um),
        (x1_um, y0_um),
        (x1_um, y1_um),
    ]
    corners_px = [
        _affine_um_to_px(xu, yu, x_transform, y_transform) for xu, yu in corners_um
    ]
    px_vals = np.array([point[0] for point in corners_px], dtype=float)
    py_vals = np.array([point[1] for point in corners_px], dtype=float)
    px0, px1 = float(px_vals.min() - 1.0), float(px_vals.max() + 1.0)
    py0, py1 = float(py_vals.min() - 1.0), float(py_vals.max() + 1.0)

    crop = da.sel(x=_coord_slice(da, "x", px0, px1), y=_coord_slice(da, "y", py0, py1))
    if crop.sizes.get("x", 0) == 0 or crop.sizes.get("y", 0) == 0:
        return None
    return crop


def _coord_slice(da: Any, dim: str, lo: float, hi: float) -> slice:
    coord = np.asarray(da.coords[dim].values)
    if coord.size >= 2 and coord[0] > coord[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


def _crop_extent_um(
    crop: Any,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    xv = np.asarray(crop.coords["x"].values)
    yv = np.asarray(crop.coords["y"].values)
    px_corners = [
        (float(xv.min()), float(yv.min())),
        (float(xv.min()), float(yv.max())),
        (float(xv.max()), float(yv.min())),
        (float(xv.max()), float(yv.max())),
    ]
    um_corners = [
        _affine_px_to_um(px, py, x_transform, y_transform) for px, py in px_corners
    ]
    x_um_vals = [corner[0] for corner in um_corners]
    y_um_vals = [corner[1] for corner in um_corners]
    return (min(x_um_vals), max(x_um_vals), min(y_um_vals), max(y_um_vals))


def _resolve_dataset_mask_affine(
    dataset_name: str,
    *,
    zarr_path: Path | str | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    dataset = dataset_name.upper()
    if dataset == "MERSCOPE":
        matrix = _load_merscope_transform_matrix(zarr_path)
        if matrix is not None:
            return _matrix_pixel_to_micron_affine(matrix)
        return (0.108, 0.0, 0.0), (0.0, 0.108, 0.0)

    if dataset == "XENIUM":
        mpp = _find_xenium_microns_per_pixel(zarr_path)
        if mpp is None:
            mpp = 0.2125
        return (float(mpp), 0.0, 0.0), (0.0, float(mpp), 0.0)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_merscope_transform_matrix(zarr_path: Path | str | None) -> np.ndarray | None:
    candidates = _sidecar_candidates(zarr_path, "micron_to_mosaic_pixel_transform.csv")
    for candidate in candidates:
        if not candidate.exists():
            continue
        matrix = np.loadtxt(candidate)
        if matrix.shape == (3, 3):
            return matrix
    return None


def _find_xenium_microns_per_pixel(zarr_path: Path | str | None) -> float | None:
    candidates = []
    candidates.extend(_sidecar_candidates(zarr_path, "experiment.xenium"))
    candidates.extend(_sidecar_candidates(zarr_path, "specs.json"))
    if zarr_path is not None:
        base = Path(zarr_path)
        candidates.extend(
            [
                base / "specs" / "specs.json",
                base.parent / "specs" / "specs.json",
            ]
        )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in {".txt", ".csv"}:
            matrix = np.loadtxt(candidate)
            if matrix.shape == (3, 3) and float(matrix[0, 0]) != 0.0:
                return 1.0 / float(matrix[0, 0])
        try:
            data = json.loads(candidate.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "pixel_size" in data:
            return float(data["pixel_size"])
        if "microns_per_pixel" in data:
            return float(data["microns_per_pixel"])
    return None


def _sidecar_candidates(zarr_path: Path | str | None, name: str) -> list[Path]:
    if zarr_path is None:
        return []
    base = Path(zarr_path)
    return [
        base / name,
        base.parent / name,
        base.parent.parent / name,
    ]


def _matrix_pixel_to_micron_affine(
    micron_to_pixel_matrix: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    inverse = np.linalg.inv(micron_to_pixel_matrix)
    x_transform = (float(inverse[0, 0]), float(inverse[0, 1]), float(inverse[0, 2]))
    y_transform = (float(inverse[1, 0]), float(inverse[1, 1]), float(inverse[1, 2]))
    return x_transform, y_transform


def _affine_um_to_px(
    x_um: float,
    y_um: float,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[float, float]:
    matrix = np.array(
        [
            [float(x_transform[0]), float(x_transform[1])],
            [float(y_transform[0]), float(y_transform[1])],
        ],
        dtype=float,
    )
    offset = np.array([float(x_transform[2]), float(y_transform[2])], dtype=float)
    px, py = np.linalg.inv(matrix) @ (np.array([x_um, y_um], dtype=float) - offset)
    return float(px), float(py)


def _affine_px_to_um(
    x_px: float,
    y_px: float,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[float, float]:
    x_um = (
        float(x_transform[0]) * float(x_px)
        + float(x_transform[1]) * float(y_px)
        + float(x_transform[2])
    )
    y_um = (
        float(y_transform[0]) * float(x_px)
        + float(y_transform[1]) * float(y_px)
        + float(y_transform[2])
    )
    return float(x_um), float(y_um)


def _first_existing_col(df_like: Any, candidates: list[str]) -> str | None:
    cols = set(map(str, list(df_like.columns)))
    for col in candidates:
        if col in cols:
            return col
    return None


def _to_numpy(data_array: Any) -> np.ndarray:
    data = data_array.compute() if hasattr(data_array, "compute") else data_array
    return np.asarray(data)
