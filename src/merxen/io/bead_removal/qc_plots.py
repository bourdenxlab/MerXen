"""Figures of the bead-removal QC report.

Every function draws one PNG of ``bead_removal_qc/`` with a matplotlib ``Figure``
that is not attached to pyplot (the Agg canvas), so drawing never changes the
backend of the process. matplotlib is imported inside :func:`_figure`; a build
without bead removal never loads it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from merxen.io.bead_removal.params import FLUX_TARGET
from merxen.io.bead_removal.psf import kern2d

if TYPE_CHECKING:
    from merxen.io.bead_removal.calibrate import BeadCalibration
    from merxen.io.bead_removal.qc import QcLocation, WindowResult


def _figure(width: float, height: float) -> Any:
    """Create a matplotlib Figure that is not attached to pyplot (Agg canvas)."""
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), layout="constrained")


def _grey_limits(*images: np.ndarray) -> tuple[float, float]:
    """Display range shared by images: 1st to 99.7th percentile of the first."""
    lo, hi = np.nanpercentile(images[0], [1.0, 99.7])
    return float(lo), float(max(hi, lo + 1.0))


def write_psf_montage(calibration: BeadCalibration, path: Path) -> None:
    """Draw the PSF library: radial profiles per plane and 2-D renders (Q3).

    Top row: the pooled template of z0..z7 (colour), the per-tile templates of z0..z3
    (grey, they show how well the calibration tiles agree) and the ring radius of the
    7 + 22.8 z law (dotted). Bottom row: the 2-D render of each pooled template.

    Args:
        calibration: The calibration (per-tile templates are kept in its library
            info).
        path: Output PNG.
    """
    lib = calibration.library
    tile_templates = lib.info.get("tile_templates", {})
    rows = lib.summary()
    n_planes = len(lib.tmpls)
    fig = _figure(2.4 * n_planes, 5.6)
    axes = fig.subplots(2, n_planes, squeeze=False)
    for k in range(n_planes):
        ax = axes[0, k]
        tmpl = lib.tmpls[k]
        for per in tile_templates.values():
            if k in per:
                ax.plot(tmpl.nodes[: len(per[k])], per[k], color="0.65", lw=0.8)
        ax.plot(tmpl.nodes, tmpl.vals, color="C3" if k < 4 else "C0", lw=1.5)
        if k >= 1:
            ax.axvline(rows[k]["law_radius_px"], color="k", ls=":", lw=0.8)
        ax.set_title(
            f"z{k}: ring {rows[k]['ring_radius_px']:.1f} px "
            f"(law {rows[k]['law_radius_px']:.1f})\n"
            f"flux {rows[k]['flux_rel']:.2f} (target {FLUX_TARGET[k]:.2f})",
            fontsize=7,
        )
        ax.set_xlabel("radius (px)", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("response per unit z0 amplitude", fontsize=7)
            ax.set_yscale("symlog", linthresh=0.05)
        kern, half = kern2d(tmpl)
        ax2 = axes[1, k]
        scale = float(np.abs(kern).max()) or 1.0
        ax2.imshow(
            kern,
            cmap="magma",
            vmin=-0.05 * scale,
            vmax=0.35 * scale if k else scale,
            extent=(-half, half, half, -half),
        )
        ax2.tick_params(labelsize=6)
    fig.suptitle(
        "PSF library: pooled templates (colour), per-tile templates z0-z3 (grey), "
        "7 + 22.8 z ring law (dotted)",
        fontsize=9,
    )
    fig.savefig(path, dpi=80)


def write_overview_figure(
    path: Path,
    *,
    dapi_overview: np.ndarray,
    valid: np.ndarray,
    tissue: np.ndarray,
    calibration: BeadCalibration,
    cores: Sequence[Sequence[int]],
    locations: Sequence[QcLocation],
    crop_px: int,
    factor: int = 16,
) -> None:
    """Draw the 1/16 overview with tissue, calibration tiles and chunk grid (Q4).

    Args:
        path: Output PNG.
        dapi_overview: 1/16 DAPI overview.
        valid: Valid mask.
        tissue: Tissue mask.
        calibration: The calibration (tiles; green = passed, red = failed).
        cores: Chunk cores.
        locations: Before/after crops (marked in blue).
        crop_px: Crop edge.
        factor: Down-sampling of the overview.
    """
    from matplotlib.patches import Rectangle

    f = factor
    fig = _figure(
        9.0, 9.0 * dapi_overview.shape[0] / max(dapi_overview.shape[1], 1) + 0.6
    )
    ax = fig.subplots()
    shown = np.where(valid, np.log1p(np.maximum(dapi_overview, 0.0)), np.nan)
    ax.imshow(shown, cmap="gray", interpolation="nearest")
    if tissue.any():
        ax.contour(tissue.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
    for y0, y1, x0, x1 in cores:
        ax.add_patch(
            Rectangle(
                (x0 / f, y0 / f),
                (x1 - x0) / f,
                (y1 - y0) / f,
                fill=False,
                ec="yellow",
                lw=0.4,
                alpha=0.6,
            )
        )
    for tile in calibration.tiles:
        size = float(tile.get("tile_px", 2048))
        colour = "lime" if tile.get("status") == "ok" else "red"
        ax.add_patch(
            Rectangle(
                (tile["x0"] / f, tile["y0"] / f),
                size / f,
                size / f,
                fill=False,
                ec=colour,
                lw=1.4,
            )
        )
        ax.text(
            tile["x0"] / f,
            tile["y0"] / f,
            str(tile.get("index")),
            color=colour,
            fontsize=7,
        )
    for loc in locations:
        ax.add_patch(
            Rectangle(
                (loc.x0 / f, loc.y0 / f),
                crop_px / f,
                crop_px / f,
                fill=False,
                ec="deepskyblue",
                lw=1.0,
                ls="--",
            )
        )
    ax.set_title(
        "DAPI overview (1/16): tissue outline (cyan), chunk grid (yellow), calibration "
        "tiles (green passed, red failed), QC crops (blue)",
        fontsize=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=100)


def write_map_figure(
    path: Path,
    panels: Sequence[tuple[str, np.ndarray, str]],
    *,
    cell: int,
    cores: Sequence[Sequence[int]],
    shape: tuple[int, int],
    suptitle: str,
) -> None:
    """Draw one or more map panels with the chunk grid overlaid."""
    fig = _figure(6.0 * len(panels), 6.0 * shape[0] / max(shape[1], 1) + 0.8)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (title, image, label) in zip(axes, panels, strict=True):
        extent = (0, image.shape[1] * cell, image.shape[0] * cell, 0)
        finite = image[np.isfinite(image)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        vmin = float(finite.min()) if finite.size and finite.min() < 0 else 0.0
        im = ax.imshow(
            image, extent=extent, cmap="viridis", vmin=vmin, vmax=max(vmax, vmin + 1e-9)
        )
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        for y0, y1, x0, x1 in cores:
            ax.plot(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="w", lw=0.4, alpha=0.7
            )
        ax.set_xlim(0, shape[1])
        ax.set_ylim(shape[0], 0)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(suptitle, fontsize=9)
    fig.savefig(path, dpi=90)


def write_before_after(
    windows: Sequence[WindowResult],
    corrected_crops: Sequence[Mapping[str, np.ndarray]],
    path: Path,
    dapi_path: Path,
    crops_path: Path,
    *,
    reference_stain: str,
    display_px: int,
) -> None:
    """Draw the before/after crops (Q10) and save the full-resolution crops.

    One row per location: raw projection, corrected projection (the build's own
    output), raw - corrected, and the reference stain's plane 1 raw and corrected
    (from the QC re-fit). A second figure shows one DAPI pair. The crops
    (uint16) are also stored as an ``.npz`` for inspection at full resolution.

    Args:
        windows: Re-fitted windows (one per location).
        corrected_crops: For each window, the build's corrected projection crop of
            every stain (``{stain: array}``, same size as the window's crop).
        path: Output PNG of the reference stain.
        dapi_path: Output PNG of the DAPI pair.
        crops_path: Output ``.npz``.
        reference_stain: Name of the reference stain (the figure's stain).
        display_px: Edge of the central part that is shown.
    """
    if not windows:
        return
    fig = _figure(15.0, 3.0 * len(windows) + 0.5)
    axes = fig.subplots(len(windows), 5, squeeze=False)
    arrays: dict[str, np.ndarray] = {}
    for r, (w, built) in enumerate(zip(windows, corrected_crops, strict=True)):
        crop = w.reference_raw.shape[0]
        lo = (crop - display_px) // 2
        sl = (slice(lo, lo + display_px), slice(lo, lo + display_px))
        raw = w.reference_raw[sl]
        corrected = built[reference_stain][sl].astype(np.float32)
        z1_raw = w.reference_raw_z1[sl]
        z1_cor = w.reference_corrected_z1[sl]
        v0, v1 = _grey_limits(raw)
        z0v, z1v = _grey_limits(z1_raw)
        diff = np.maximum(raw - corrected, 0.0)
        panels = (
            (raw, "raw projection", v0, v1, "gray"),
            (corrected, "corrected projection", v0, v1, "gray"),
            (
                diff,
                "raw - corrected",
                0.0,
                float(max(np.percentile(diff, 99.9), 1.0)),
                "magma",
            ),
            (z1_raw, "z1 raw", z0v, z1v, "gray"),
            (z1_cor, "z1 corrected", z0v, z1v, "gray"),
        )
        for c, (img, title, vmin, vmax, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{w.location.note}: {title}" if c == 0 else title, fontsize=7)
        arrays[f"loc{r}_{reference_stain}_raw"] = np.clip(
            np.rint(raw), 0, 65535
        ).astype(np.uint16)
        arrays[f"loc{r}_{reference_stain}_corrected"] = np.clip(
            np.rint(corrected), 0, 65535
        ).astype(np.uint16)
        arrays[f"loc{r}_origin_yx"] = np.array(
            [w.location.y0 + lo, w.location.x0 + lo], dtype=np.int64
        )
    fig.suptitle(
        f"{reference_stain}: raw vs corrected (raw and corrected share a grey scale; "
        "z1 panels are re-fitted on the crop window)",
        fontsize=9,
    )
    fig.savefig(path, dpi=70)

    first = windows[0]
    dapi_names = [s for s in first.stain_raw if s != reference_stain]
    if dapi_names:
        stain = "DAPI" if "DAPI" in dapi_names else dapi_names[0]
        crop = first.stain_raw[stain].shape[0]
        lo = (crop - display_px) // 2
        sl = (slice(lo, lo + display_px), slice(lo, lo + display_px))
        raw = first.stain_raw[stain][sl]
        corrected = corrected_crops[0][stain][sl].astype(np.float32)
        diff = np.maximum(raw - corrected, 0.0)
        fig = _figure(10.0, 3.8)
        axes2 = fig.subplots(1, 3)
        v0, v1 = _grey_limits(raw)
        axes2[0].imshow(raw, cmap="gray", vmin=v0, vmax=v1, interpolation="nearest")
        axes2[1].imshow(
            corrected, cmap="gray", vmin=v0, vmax=v1, interpolation="nearest"
        )
        axes2[2].imshow(
            diff,
            cmap="magma",
            vmin=0,
            vmax=max(float(np.percentile(diff, 99.9)), 1.0),
            interpolation="nearest",
        )
        for ax, title in zip(
            axes2, ("raw", "corrected", "raw - corrected"), strict=True
        ):
            ax.set_title(f"{stain} {title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{first.location.note}", fontsize=8)
        fig.savefig(dapi_path, dpi=80)
        arrays[f"loc0_{stain}_raw"] = np.clip(np.rint(raw), 0, 65535).astype(np.uint16)
        arrays[f"loc0_{stain}_corrected"] = np.clip(
            np.rint(corrected), 0, 65535
        ).astype(np.uint16)
    np.savez_compressed(crops_path, **arrays)  # type: ignore[arg-type]


def write_change_overview(
    path: Path, change: Mapping[str, np.ndarray], *, cores: Sequence[Sequence[int]]
) -> None:
    """Draw the 1/16 overview of the largest change of each corrected stain (Q11).

    Chunk-shaped blocks in this picture would mean a chunk-dependent correction.

    Args:
        path: Output PNG.
        change: ``{stain: [h/16, w/16] max of (raw - corrected)}``.
        cores: Chunk cores (not drawn: a visible seam must not be explained by an
            overlay).
    """
    stains = list(change)
    first = change[stains[0]]
    fig = _figure(
        6.0 * len(stains), 6.0 * first.shape[0] / max(first.shape[1], 1) + 0.8
    )
    axes = fig.subplots(1, len(stains), squeeze=False)[0]
    for ax, stain in zip(axes, stains, strict=True):
        image = change[stain]
        vmax = float(np.percentile(image, 99.9)) if image.size else 1.0
        im = ax.imshow(
            image, cmap="magma", vmin=0.0, vmax=max(vmax, 1.0), interpolation="nearest"
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="counts")
        ax.set_title(f"{stain}: max(raw - corrected) per 16 px block", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=90)


def write_residual_profile_figure(profiles: Mapping[str, Any], path: Path) -> None:
    """Draw the median residual profiles, tissue against off tissue (Q9)."""
    planes = sorted({int(z) for g in profiles["groups"].values() for z in g["planes"]})
    if not planes:
        return
    fig = _figure(4.2 * len(planes), 6.0)
    axes = fig.subplots(2, len(planes), squeeze=False)
    radius = np.asarray(profiles["radius_px"])
    for r, name in enumerate(("tissue", "off_tissue")):
        group = profiles["groups"][name]
        for c, z in enumerate(planes):
            ax = axes[r, c]
            entry = group["planes"].get(str(z))
            if entry:
                ax.plot(radius, entry["raw"]["profile"], color="0.5", label="raw")
                ax.plot(
                    radius, entry["corrected"]["profile"], color="C3", label="corrected"
                )
                ax.axvline(entry["ring_radius_px"], color="k", ls=":", lw=0.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(
                f"{name.replace('_', ' ')} z{z} (n={group['n_beads']})", fontsize=8
            )
            ax.tick_params(labelsize=6)
            if r == 1:
                ax.set_xlabel("radius (px)", fontsize=7)
            if c == 0:
                ax.set_ylabel("median profile, 1e-3 x A0", fontsize=7)
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
            break
    fig.suptitle(
        "Residual around bright isolated beads (annulus 150-170 px subtracted)",
        fontsize=9,
    )
    fig.savefig(path, dpi=80)
