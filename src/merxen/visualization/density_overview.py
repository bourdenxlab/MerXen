"""Transcript density overview plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from tqdm.auto import tqdm

from merxen.io.spatialdata_schema import choose_primary_points_key
from merxen.platforms import find_platform_spec
from merxen.plotting import prepare_plot_output, save_figure
from merxen.visualization.raster_density import (
    DensityRaster,
    percentile_vmax,
    points_bounds,
    single_colour_cmap,
    stream_hist2d,
)

MERSCOPE_TRANSCRIPT_COLOR = "#1f77b4"
XENIUM_TRANSCRIPT_COLOR = "#d62728"
ATERA_TRANSCRIPT_COLOR = "#2ca02c"
TRANSCRIPT_OVERVIEW_SAMPLE_N = 250_000
TRANSCRIPT_OVERVIEW_CROP_SAMPLE_N = 200_000
TRANSCRIPT_OVERVIEW_RANDOM_STATE = 42
TRANSCRIPT_OVERVIEW_POINT_SIZE = 0.3
TRANSCRIPT_OVERVIEW_CROP_BBOX_UM = (4000.0, 4000.0, 6000.0, 6000.0)
TRANSCRIPT_OVERVIEW_HEATMAP_BINS = 400
TRANSCRIPT_OVERVIEW_HEATMAP_CMAP = "magma"
TRANSCRIPT_OVERVIEW_HEATMAP_LOG = False
TRANSCRIPT_OVERVIEW_HEATMAP_VMIN = None
TRANSCRIPT_OVERVIEW_HEATMAP_VMAX = 1700.0
RASTER_OVERVIEW_VMAX_PERCENTILE = 99.5
RASTER_OVERVIEW_FULL_BIN_UM = 10.0
RASTER_OVERVIEW_CROP_BIN_UM = 1.0


def density_hist2d(
    points_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    bins: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D histogram density from point coordinates."""
    x = pd.to_numeric(points_df[x_col], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(points_df[y_col], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("No valid point coordinates available for density histogram.")
    hist, x_edges, y_edges = np.histogram2d(x[valid], y[valid], bins=bins)
    return hist, x_edges, y_edges


def plot_density_overview(
    points_df: pd.DataFrame,
    output_path: Path | str,
    *,
    x_col: str = "x",
    y_col: str = "y",
    bins: int = 400,
    title: str = "Transcript Density",
) -> Path:
    """Render a full-field transcript density heatmap from point coordinates."""
    output_path = prepare_plot_output(output_path)
    hist, x_edges, y_edges = density_hist2d(
        points_df, x_col=x_col, y_col=y_col, bins=bins
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = (
        float(x_edges[0]),
        float(x_edges[-1]),
        float(y_edges[0]),
        float(y_edges[-1]),
    )
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Transcript count")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_transcript_overview(
    merscope_sdata: Any,
    xenium_sdata: Any,
    output_path: Path | str,
    *,
    sample_n: int = TRANSCRIPT_OVERVIEW_SAMPLE_N,
    crop_sample_n: int = TRANSCRIPT_OVERVIEW_CROP_SAMPLE_N,
    random_state: int = TRANSCRIPT_OVERVIEW_RANDOM_STATE,
    point_size: float = TRANSCRIPT_OVERVIEW_POINT_SIZE,
    crop_bbox_um: tuple[float, float, float, float] = TRANSCRIPT_OVERVIEW_CROP_BBOX_UM,
    heatmap_bins: int = TRANSCRIPT_OVERVIEW_HEATMAP_BINS,
    heatmap_cmap: str = TRANSCRIPT_OVERVIEW_HEATMAP_CMAP,
    heatmap_log: bool = TRANSCRIPT_OVERVIEW_HEATMAP_LOG,
    heatmap_vmin: float | None = TRANSCRIPT_OVERVIEW_HEATMAP_VMIN,
    heatmap_vmax: float | None = TRANSCRIPT_OVERVIEW_HEATMAP_VMAX,
) -> Path:
    """Render the paired 3x2 transcript overview from the MOSAIK notebook."""
    output_path = prepare_plot_output(output_path)

    merscope_full = _points_bounds_and_sample_xy(
        merscope_sdata,
        sample_n=sample_n,
        random_state=random_state,
    )
    xenium_full = _points_bounds_and_sample_xy(
        xenium_sdata,
        sample_n=sample_n,
        random_state=random_state,
    )

    merscope_crop = _points_sample_xy_in_bbox(
        merscope_sdata,
        bbox=crop_bbox_um,
        sample_n=crop_sample_n,
        random_state=random_state,
    )
    xenium_crop = _points_sample_xy_in_bbox(
        xenium_sdata,
        bbox=crop_bbox_um,
        sample_n=crop_sample_n,
        random_state=random_state,
    )

    m_cx, m_cy, m_span = _center_and_span(merscope_full["bounds"])
    x_cx, x_cy, x_span = _center_and_span(xenium_full["bounds"])
    common_span = max(m_span, x_span)
    half = 0.5 * common_span

    m_xrange = (m_cx - half, m_cx + half)
    m_yrange = (m_cy - half, m_cy + half)
    x_xrange = (x_cx - half, x_cx + half)
    x_yrange = (x_cy - half, x_cy + half)

    m_heat = _hist2d_all_points_streaming(
        merscope_sdata,
        x_range=m_xrange,
        y_range=m_yrange,
        bins=heatmap_bins,
        dataset_name="MERSCOPE",
    )
    x_heat = _hist2d_all_points_streaming(
        xenium_sdata,
        x_range=x_xrange,
        y_range=x_yrange,
        bins=heatmap_bins,
        dataset_name="XENIUM",
    )

    m_h = m_heat["hist"].T
    x_h = x_heat["hist"].T
    norm, m_show, x_show = _build_heatmap_display(
        m_h,
        x_h,
        log_scale=heatmap_log,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    x0, y0, x1, y1 = crop_bbox_um
    fig, axes = plt.subplots(3, 2, figsize=(16, 20), constrained_layout=True)

    axes[0, 0].imshow(
        m_show,
        origin="lower",
        extent=(m_xrange[0], m_xrange[1], m_yrange[0], m_yrange[1]),
        cmap=heatmap_cmap,
        norm=norm,
        aspect="equal",
    )
    axes[0, 0].set_title("MERSCOPE transcript density (all transcripts)")
    axes[0, 0].set_xlabel("x (microns)")
    axes[0, 0].set_ylabel("y (microns)")

    im1 = axes[0, 1].imshow(
        x_show,
        origin="lower",
        extent=(x_xrange[0], x_xrange[1], x_yrange[0], x_yrange[1]),
        cmap=heatmap_cmap,
        norm=norm,
        aspect="equal",
    )
    axes[0, 1].set_title("XENIUM transcript density (all transcripts)")
    axes[0, 1].set_xlabel("x (microns)")
    axes[0, 1].set_ylabel("y (microns)")

    cbar = fig.colorbar(im1, ax=[axes[0, 0], axes[0, 1]], shrink=0.9)
    label_suffix = " (log scale)" if heatmap_log else ""
    cbar.set_label(f"Transcript count per bin{label_suffix}")

    m_df = merscope_full["sampled"]
    axes[1, 0].scatter(
        m_df["x_um"],
        m_df["y_um"],
        s=point_size,
        c=MERSCOPE_TRANSCRIPT_COLOR,
        alpha=0.15,
        rasterized=True,
    )
    axes[1, 0].set_xlim(m_xrange)
    axes[1, 0].set_ylim(m_yrange)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("MERSCOPE transcripts (subsample, full)")
    axes[1, 0].set_xlabel("x (microns)")
    axes[1, 0].set_ylabel("y (microns)")

    x_df = xenium_full["sampled"]
    axes[1, 1].scatter(
        x_df["x_um"],
        x_df["y_um"],
        s=point_size,
        c=XENIUM_TRANSCRIPT_COLOR,
        alpha=0.15,
        rasterized=True,
    )
    axes[1, 1].set_xlim(x_xrange)
    axes[1, 1].set_ylim(x_yrange)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("XENIUM transcripts (subsample, full)")
    axes[1, 1].set_xlabel("x (microns)")
    axes[1, 1].set_ylabel("y (microns)")

    mc_df = merscope_crop["sampled"]
    axes[2, 0].scatter(
        mc_df["x_um"],
        mc_df["y_um"],
        s=point_size,
        c=MERSCOPE_TRANSCRIPT_COLOR,
        alpha=0.05,
        rasterized=True,
    )
    axes[2, 0].set_xlim(x0, x1)
    axes[2, 0].set_ylim(y0, y1)
    axes[2, 0].set_aspect("equal")
    axes[2, 0].set_title(f"MERSCOPE crop x=[{x0:.0f},{x1:.0f}], y=[{y0:.0f},{y1:.0f}]")
    axes[2, 0].set_xlabel("x (microns)")
    axes[2, 0].set_ylabel("y (microns)")

    xc_df = xenium_crop["sampled"]
    axes[2, 1].scatter(
        xc_df["x_um"],
        xc_df["y_um"],
        s=point_size,
        c=XENIUM_TRANSCRIPT_COLOR,
        alpha=0.05,
        rasterized=True,
    )
    axes[2, 1].set_xlim(x0, x1)
    axes[2, 1].set_ylim(y0, y1)
    axes[2, 1].set_aspect("equal")
    axes[2, 1].set_title(f"XENIUM crop x=[{x0:.0f},{x1:.0f}], y=[{y0:.0f},{y1:.0f}]")
    axes[2, 1].set_xlabel("x (microns)")
    axes[2, 1].set_ylabel("y (microns)")

    save_figure(fig, output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_single_transcript_overview(
    sdata_obj: Any,
    dataset_name: str,
    output_path: Path | str,
    *,
    sample_n: int = TRANSCRIPT_OVERVIEW_SAMPLE_N,
    crop_sample_n: int = TRANSCRIPT_OVERVIEW_CROP_SAMPLE_N,
    random_state: int = TRANSCRIPT_OVERVIEW_RANDOM_STATE,
    point_size: float = TRANSCRIPT_OVERVIEW_POINT_SIZE,
    crop_bbox_um: tuple[float, float, float, float] | None = None,
    crop_size_um: float = 2_000.0,
    heatmap_bins: int = TRANSCRIPT_OVERVIEW_HEATMAP_BINS,
    heatmap_cmap: str = TRANSCRIPT_OVERVIEW_HEATMAP_CMAP,
    heatmap_log: bool = TRANSCRIPT_OVERVIEW_HEATMAP_LOG,
    heatmap_vmin: float | None = TRANSCRIPT_OVERVIEW_HEATMAP_VMIN,
    heatmap_vmax: float | None = TRANSCRIPT_OVERVIEW_HEATMAP_VMAX,
    zarr_path: Path | str | None = None,
) -> Path:
    """Render a one-dataset transcript density, full scatter, and crop overview.

    Platforms with ``rasterised_transcript_plots`` (whole-transcriptome panels)
    draw all three panels as density rasters over every transcript instead
    (:func:`plot_rasterised_transcript_overview`); ``sample_n``,
    ``crop_sample_n``, ``random_state``, ``point_size``, ``heatmap_log`` and
    ``heatmap_vmax`` do not apply to them.
    """
    spec = find_platform_spec(dataset_name)
    if spec is not None and spec.rasterised_transcript_plots:
        store = zarr_path if zarr_path is not None else getattr(sdata_obj, "path", None)
        if store is None:
            raise ValueError(
                f"{spec.name} transcript overviews stream the points from the "
                "store; pass zarr_path or a store-backed SpatialData object."
            )
        return plot_rasterised_transcript_overview(
            store,
            dataset_name,
            output_path,
            points_key=_reference_points_key(sdata_obj),
            crop_bbox_um=crop_bbox_um,
            crop_size_um=crop_size_um,
            heatmap_bins=heatmap_bins,
            heatmap_cmap=heatmap_cmap,
            heatmap_vmin=heatmap_vmin,
        )
    output_path = prepare_plot_output(output_path)
    dataset_label = str(dataset_name).upper()
    point_color = (
        MERSCOPE_TRANSCRIPT_COLOR
        if dataset_label == "MERSCOPE"
        else XENIUM_TRANSCRIPT_COLOR
    )

    full = _points_bounds_and_sample_xy(
        sdata_obj,
        sample_n=sample_n,
        random_state=random_state,
    )
    cx, cy, span = _center_and_span(full["bounds"])
    span = max(float(span), 1.0)
    half = 0.5 * span
    x_range = (cx - half, cx + half)
    y_range = (cy - half, cy + half)

    if crop_bbox_um is None:
        crop_bbox_um = _centered_bbox(full["bounds"], crop_size_um)
    crop = _points_sample_xy_in_bbox(
        sdata_obj,
        bbox=crop_bbox_um,
        sample_n=crop_sample_n,
        random_state=random_state,
    )

    heat = _hist2d_all_points_streaming(
        sdata_obj,
        x_range=x_range,
        y_range=y_range,
        bins=heatmap_bins,
        dataset_name=dataset_label,
    )
    heat_values = heat["hist"].T
    norm, heat_show, _ = _build_heatmap_display(
        heat_values,
        heat_values,
        log_scale=heatmap_log,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    fig, axes = plt.subplots(3, 1, figsize=(8, 18), constrained_layout=True)
    im = axes[0].imshow(
        heat_show,
        origin="lower",
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        cmap=heatmap_cmap,
        norm=norm,
        aspect="equal",
    )
    axes[0].set_title(f"{dataset_label} transcript density (all transcripts)")
    axes[0].set_xlabel("x (microns)")
    axes[0].set_ylabel("y (microns)")
    label_suffix = " (log scale)" if heatmap_log else ""
    fig.colorbar(im, ax=axes[0], shrink=0.85).set_label(
        f"Transcript count per bin{label_suffix}"
    )

    full_df = full["sampled"]
    axes[1].scatter(
        full_df["x_um"],
        full_df["y_um"],
        s=point_size,
        c=point_color,
        alpha=0.15,
        rasterized=True,
    )
    axes[1].set_xlim(x_range)
    axes[1].set_ylim(y_range)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"{dataset_label} transcripts (subsample, full)")
    axes[1].set_xlabel("x (microns)")
    axes[1].set_ylabel("y (microns)")

    x0, y0, x1, y1 = crop_bbox_um
    crop_df = crop["sampled"]
    axes[2].scatter(
        crop_df["x_um"],
        crop_df["y_um"],
        s=point_size,
        c=point_color,
        alpha=0.08,
        rasterized=True,
    )
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y0, y1)
    axes[2].set_aspect("equal")
    axes[2].set_title(
        f"{dataset_label} crop x=[{x0:.0f},{x1:.0f}], y=[{y0:.0f},{y1:.0f}]"
    )
    axes[2].set_xlabel("x (microns)")
    axes[2].set_ylabel("y (microns)")

    save_figure(fig, output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_rasterised_transcript_overview(
    zarr_path: Path | str,
    dataset_name: str,
    output_path: Path | str,
    *,
    points_key: str = "transcripts",
    crop_bbox_um: tuple[float, float, float, float] | None = None,
    crop_size_um: float = 2_000.0,
    heatmap_bins: int = TRANSCRIPT_OVERVIEW_HEATMAP_BINS,
    heatmap_cmap: str = TRANSCRIPT_OVERVIEW_HEATMAP_CMAP,
    heatmap_vmin: float | None = TRANSCRIPT_OVERVIEW_HEATMAP_VMIN,
    full_bin_um: float = RASTER_OVERVIEW_FULL_BIN_UM,
    crop_bin_um: float = RASTER_OVERVIEW_CROP_BIN_UM,
    vmax_percentile: float = RASTER_OVERVIEW_VMAX_PERCENTILE,
) -> Path:
    """Render the three overview panels as density over every transcript.

    Nothing is sampled: each panel streams the points element of the store one
    parquet file at a time (:func:`raster_density.stream_hist2d`), and the crop
    reads only the files the points index places in the crop.

    - Panel 0: ``heatmap_bins`` square bins over the section; the colour
      limit is the ``vmax_percentile`` of the non-empty bins.
    - Panel 1: ``full_bin_um`` bins over the section, log colour scale.
    - Panel 2: ``crop_bin_um`` bins over the crop, log colour scale.

    Args:
        zarr_path: Store whose points are binned.
        dataset_name: Platform label used in titles.
        output_path: PNG path (a PDF is written beside it).
        points_key: Points element.
        crop_bbox_um: Crop ``(x0, y0, x1, y1)``; centred on the section when
            ``None``.
        crop_size_um: Crop edge when ``crop_bbox_um`` is ``None``.
        heatmap_bins: Bins per axis of panel 0.
        heatmap_cmap: Colormap of panel 0.
        heatmap_vmin: Lower colour limit of panel 0 (default 0).
        full_bin_um: Bin edge of panel 1.
        crop_bin_um: Bin edge of panel 2.
        vmax_percentile: Percentile of non-empty bins used as colour maximum.

    Returns:
        ``output_path``.
    """
    output_path = prepare_plot_output(output_path)
    zarr_path = Path(zarr_path)
    dataset_label = str(dataset_name).upper()
    bounds = points_bounds(zarr_path, points_key)
    cx, cy, span = _center_and_span(bounds)
    span = max(float(span), 1.0)
    half = 0.5 * span
    x_range = (cx - half, cx + half)
    y_range = (cy - half, cy + half)
    square = (x_range[0], y_range[0], x_range[1], y_range[1])
    if crop_bbox_um is None:
        crop_bbox_um = _centered_bbox(bounds, crop_size_um)

    heat = stream_hist2d(
        zarr_path, points_key, square, span / max(int(heatmap_bins), 1)
    )
    full = stream_hist2d(zarr_path, points_key, square, full_bin_um)
    crop = stream_hist2d(zarr_path, points_key, crop_bbox_um, crop_bin_um)

    colour = ATERA_TRANSCRIPT_COLOR
    cmap = single_colour_cmap(colour, name=f"{dataset_label.lower()}_density")
    fig, axes = plt.subplots(3, 1, figsize=(8, 18), constrained_layout=True)

    heat_values = heat.combined().astype(np.float64)
    vmin = 0.0 if heatmap_vmin is None else float(heatmap_vmin)
    vmax = max(percentile_vmax(heat_values, vmax_percentile), vmin + 1.0)
    im = axes[0].imshow(
        heat_values,
        origin="lower",
        extent=heat.extent,
        cmap=heatmap_cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        aspect="equal",
        interpolation="nearest",
    )
    axes[0].set_xlim(x_range)
    axes[0].set_ylim(y_range)
    axes[0].set_title(
        f"{dataset_label} transcript density (all transcripts, {heat.total():,})"
    )
    fig.colorbar(im, ax=axes[0], shrink=0.85).set_label(
        f"Transcript count per {heat.bin_um:.3g} um bin "
        f"(max = {vmax_percentile:g}th percentile)"
    )

    _draw_log_density(
        fig,
        axes[1],
        full,
        cmap=cmap,
        vmax_percentile=vmax_percentile,
        title=f"{dataset_label} transcripts (all transcripts, "
        f"{full.bin_um:g} um bins, {full.total():,})",
    )
    axes[1].set_xlim(x_range)
    axes[1].set_ylim(y_range)

    x0, y0, x1, y1 = crop_bbox_um
    _draw_log_density(
        fig,
        axes[2],
        crop,
        cmap=cmap,
        vmax_percentile=vmax_percentile,
        title=f"{dataset_label} crop x=[{x0:.0f},{x1:.0f}], y=[{y0:.0f},{y1:.0f}] "
        f"(all transcripts, {crop.bin_um:g} um bins, {crop.total():,})",
    )
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y0, y1)
    for ax in axes:
        ax.set_xlabel("x (microns)")
        ax.set_ylabel("y (microns)")

    save_figure(fig, output_path, dpi=220)
    plt.close(fig)
    return output_path


def _draw_log_density(
    fig: Any,
    ax: Any,
    raster: DensityRaster,
    *,
    cmap: Any,
    vmax_percentile: float,
    title: str,
) -> None:
    values = raster.combined().astype(np.float64)
    vmax = percentile_vmax(values, vmax_percentile)
    shown = np.ma.masked_less_equal(values, 0.0)
    im = ax.imshow(
        shown,
        origin="lower",
        extent=raster.extent,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=1.0, vmax=max(vmax, 1.0 + 1e-9)),
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85).set_label(
        f"Transcripts per {raster.bin_um:g} um bin (log scale)"
    )


def _resolve_points_xy_cols(
    sdata_obj: Any,
) -> tuple[str, Any, str, str]:
    if len(sdata_obj.points) == 0:
        raise RuntimeError("No points found in SpatialData object.")
    pts_key = _reference_points_key(sdata_obj)
    pts = sdata_obj.points[pts_key]
    x_col = _first_existing_col(
        pts,
        ["x", "x_micron", "global_x", "x_location", "observed_x", "x_global_px"],
    )
    y_col = _first_existing_col(
        pts,
        ["y", "y_micron", "global_y", "y_location", "observed_y", "y_global_px"],
    )
    if x_col is None or y_col is None:
        raise KeyError(f"Could not resolve x/y columns for points[{pts_key}]")
    return pts_key, pts, x_col, y_col


def _points_bounds_and_sample_xy(
    sdata_obj: Any,
    *,
    sample_n: int,
    random_state: int,
) -> dict[str, Any]:
    pts_key, pts, x_col, y_col = _resolve_points_xy_cols(sdata_obj)
    work = pts[[x_col, y_col]]

    if hasattr(work, "npartitions") and hasattr(work, "compute"):
        minx = float(work[x_col].min().compute())
        maxx = float(work[x_col].max().compute())
        miny = float(work[y_col].min().compute())
        maxy = float(work[y_col].max().compute())

        total_n = int(work.map_partitions(len, meta=("n", "i8")).sum().compute())
        if total_n <= sample_n:
            sampled = work.compute()
        else:
            frac = float(sample_n) / float(total_n)
            sampled = work.sample(frac=frac, random_state=random_state).compute()
        sampled = sampled.rename(columns={x_col: "x_um", y_col: "y_um"})
    else:
        pdf = pd.DataFrame(work).copy()
        minx = float(pdf[x_col].min())
        maxx = float(pdf[x_col].max())
        miny = float(pdf[y_col].min())
        maxy = float(pdf[y_col].max())
        sampled = pdf.rename(columns={x_col: "x_um", y_col: "y_um"})

    if len(sampled) > sample_n:
        sampled = sampled.sample(n=sample_n, random_state=random_state)

    return {
        "points_key": pts_key,
        "sampled": sampled,
        "bounds": (minx, miny, maxx, maxy),
    }


def _points_sample_xy_in_bbox(
    sdata_obj: Any,
    *,
    bbox: tuple[float, float, float, float],
    sample_n: int,
    random_state: int,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    pts_key, pts, x_col, y_col = _resolve_points_xy_cols(sdata_obj)
    work = pts[[x_col, y_col]]

    if hasattr(work, "npartitions") and hasattr(work, "compute"):
        crop = work[
            (work[x_col] >= x0)
            & (work[x_col] <= x1)
            & (work[y_col] >= y0)
            & (work[y_col] <= y1)
        ]
        total_n = int(crop.map_partitions(len, meta=("n", "i8")).sum().compute())
        if total_n <= sample_n:
            sampled = crop.compute()
        else:
            frac = float(sample_n) / float(total_n)
            sampled = crop.sample(frac=frac, random_state=random_state).compute()
        sampled = sampled.rename(columns={x_col: "x_um", y_col: "y_um"})
    else:
        pdf = pd.DataFrame(work).copy()
        sampled = pdf[
            (pdf[x_col] >= x0)
            & (pdf[x_col] <= x1)
            & (pdf[y_col] >= y0)
            & (pdf[y_col] <= y1)
        ].copy()
        sampled = sampled.rename(columns={x_col: "x_um", y_col: "y_um"})

    if len(sampled) > sample_n:
        sampled = sampled.sample(n=sample_n, random_state=random_state)

    return {
        "points_key": pts_key,
        "sampled": sampled,
        "bbox": bbox,
    }


def _hist2d_all_points_streaming(
    sdata_obj: Any,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: int,
    dataset_name: str,
) -> dict[str, Any]:
    pts_key, pts, x_col, y_col = _resolve_points_xy_cols(sdata_obj)
    work = pts[[x_col, y_col]]
    hist = np.zeros((bins, bins), dtype=np.uint64)

    if hasattr(work, "npartitions") and hasattr(work, "get_partition"):
        nparts = int(work.npartitions)
        iterator = tqdm(
            range(nparts),
            desc=f"[{dataset_name}] density histogram partitions",
            unit="part",
        )
        for i in iterator:
            pdf = work.get_partition(i).compute()
            _add_hist2d_partition(hist, pdf, x_col, y_col, x_range, y_range, bins)
    else:
        pdf = pd.DataFrame(work).copy()
        _add_hist2d_partition(hist, pdf, x_col, y_col, x_range, y_range, bins)

    return {
        "points_key": pts_key,
        "hist": hist,
        "n_total": int(hist.sum()),
        "x_col": x_col,
        "y_col": y_col,
    }


def _add_hist2d_partition(
    hist: np.ndarray,
    pdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: int,
) -> None:
    if len(pdf) == 0:
        return
    xv = pd.to_numeric(pdf[x_col], errors="coerce").to_numpy(np.float64)
    yv = pd.to_numeric(pdf[y_col], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(xv) & np.isfinite(yv)
    if not np.any(valid):
        return
    h, _, _ = np.histogram2d(
        xv[valid],
        yv[valid],
        bins=bins,
        range=[x_range, y_range],
    )
    hist += h.astype(np.uint64, copy=False)


def _build_heatmap_display(
    m_h: np.ndarray,
    x_h: np.ndarray,
    *,
    log_scale: bool,
    vmin: float | None,
    vmax: float | None,
) -> tuple[mcolors.Normalize, np.ndarray, np.ndarray]:
    shared_vmax_auto = float(max(np.nanmax(m_h), np.nanmax(x_h), 1.0))
    positive_mins = []
    if np.any(m_h > 0):
        positive_mins.append(float(np.nanmin(m_h[m_h > 0])))
    if np.any(x_h > 0):
        positive_mins.append(float(np.nanmin(x_h[x_h > 0])))
    shared_vmin_positive_auto = float(min(positive_mins)) if positive_mins else 1.0

    if log_scale:
        auto_vmin = shared_vmin_positive_auto
        auto_vmax = shared_vmax_auto
    else:
        auto_vmin = 0.0
        auto_vmax = shared_vmax_auto

    heat_vmin = float(vmin) if vmin is not None else auto_vmin
    heat_vmax = float(vmax) if vmax is not None else auto_vmax

    if log_scale and heat_vmin <= 0:
        raise ValueError("For log heatmap scale, heatmap_vmin must be > 0")
    if heat_vmax <= heat_vmin:
        raise ValueError(
            f"Invalid heatmap limits: vmin={heat_vmin} must be < vmax={heat_vmax}."
        )

    if log_scale:
        norm: mcolors.Normalize = mcolors.LogNorm(vmin=heat_vmin, vmax=heat_vmax)
        return norm, np.where(m_h > 0, m_h, np.nan), np.where(x_h > 0, x_h, np.nan)

    norm = mcolors.Normalize(vmin=heat_vmin, vmax=heat_vmax)
    return norm, m_h, x_h


def _center_and_span(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    minx, miny, maxx, maxy = bounds
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    span = max(maxx - minx, maxy - miny)
    return cx, cy, span


def _centered_bbox(
    bounds: tuple[float, float, float, float],
    size_um: float,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    size = max(1.0, min(float(size_um), max(maxx - minx, maxy - miny)))
    half = 0.5 * size
    return (cx - half, cy - half, cx + half, cy + half)


def _reference_points_key(sdata_obj: Any) -> str:
    primary = choose_primary_points_key(sdata_obj)
    if primary is None:
        raise RuntimeError("No transcript points found in SpatialData object.")
    aligned = f"{primary}_aligned_nonrigid"
    return aligned if aligned in sdata_obj.points else primary


def _first_existing_col(df_like: Any, candidates: list[str]) -> str | None:
    cols = list(df_like.columns)
    for col in candidates:
        if col in cols:
            return col
    return None
