"""Quality-control record of a bead-removal build.

Two parts live here:

* Cheap statistics that every chunk computes from arrays it already holds
  (:func:`chunk_qc_probe`): the residual-ring scores of bright isolated beads on
  the raw and the corrected projection, and a 1/16 overview of the largest
  change of the projection. The chunk driver calls it and the results travel in
  the per-chunk statistics.
* The report written into ``bead_removal_qc/`` inside the built zarr
  (:func:`write_bead_removal_qc`): summary and warnings, calibration record, PSF
  library and montage, overview with the calibration tiles, bead table, density
  and stretch maps, per-chunk table, residual-ring and residual-profile
  numbers, before/after crops and the change overview. It is what a user reads
  to decide whether to trust a sample.

The module is plain numpy until a figure is drawn: matplotlib is imported inside
the functions that draw, with the Agg canvas, so a build without bead removal
never loads it. A failure of the QC must never cost the projection, so the
builder calls the writer inside a guard.
"""

from __future__ import annotations

import csv
import functools
import gzip
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

from merxen.io.bead_removal.apply import SATURATION_LEVEL
from merxen.io.bead_removal.calibrate import (
    DEFAULT_PX_PER_UM,
    BeadCalibration,
    build_overview,
    ring_tscores,
    tissue_mask_from_overview,
)
from merxen.io.bead_removal.params import (
    BEAD_REMOVAL_VERSION,
    DEFAULT_PARAMS,
    FLUX_TARGET,
    PsfSubtractParams,
)
from merxen.io.bead_removal.psf import to_jsonable
from merxen.io.bead_removal.qc_plots import (
    write_before_after,
    write_change_overview,
    write_map_figure,
    write_overview_figure,
    write_psf_montage,
    write_residual_profile_figure,
)

if TYPE_CHECKING:
    from merxen.io.bead_removal.driver import BeadRemovalRunResult
    from merxen.io.bead_removal.fill import FillParams

logger = logging.getLogger(__name__)

QC_DIRNAME = "bead_removal_qc"
METHOD_NAME = "psf_subtract"

# Down-sampling of the change overview and of the tissue overview (the same grid).
OVERVIEW_FACTOR = 16

# --- residual-ring score (Q9)
# Beads this bright (A0 counts) are scored; a neighbour above ISOLATION_A0 within
# ISOLATION_PX makes a bead non-isolated.
BRIGHT_A0 = 1600.0
ISOLATION_A0 = 400.0
ISOLATION_PX = 40.0
# At most this many beads per chunk (the brightest isolated ones).
MAX_RING_BEADS_PER_CHUNK = 50
# Scored beads keep this far from the core edge, so that the sector windows lie
# inside the projection and every neighbour is in the same core.
RING_MARGIN_PX = 72
CONTROL_POINTS_PER_CHUNK = 50
CONTROL_CLEARANCE_PX = 60.0
# A ring or disc is "visible" above this sector t-score.
VISIBLE_T = 3.0
# Disc: the cell-sized bright disc a bead leaves in a max projection, scored out
# to the z1 ring radius plus this many px, against an annulus 8-20 px outside.
DISC_EXTRA_PX = 4.0
# The score is reported without a warning below this many scored beads.
MIN_SCORED_FOR_WARNING = 20
RING_WARN_ABOVE = 0.05
DISC_WARN_ABOVE = 0.20

# --- warnings
# A calibration whose weakest passing tile is this close to the gate is fragile.
DECISION_MARGIN_WARN = 0.02
FILL_GUARD_SHARE = 0.5
QC_MAX_MB = 50.0

# --- residual profiles (Q9, second part)
PROFILE_RADIUS_PX = 170
PROFILE_PLANES = (1, 2, 3, 4)
PROFILE_ISOLATION_PX = 50.0
PROFILE_BASELINE_PX = (150, 170)

# Window halo of the QC re-fits (the production halo is 256 px or more).
QC_HALO_PX = 256

# Calibration warnings are passed through; these prefixes give them a code.
_CALIBRATION_WARNING_CODES: tuple[tuple[str, str], ...] = (
    ("low_tissue_tiles", "low_tissue_tiles"),
    ("cost forecast", "forecast"),
    ("classifier recall including", "classifier_recall"),
    ("held-out classifier recall", "classifier_recall"),
    ("classifier thresholds", "classifier_thresholds"),
    ("stain ", "antibody_rho"),
    ("focus check", "focus_check"),
    ("stretch fallback", "stretch_fallback"),
)


# ---------------------------------------------------------------------------
# sector t-scores
_DISC_CACHE: dict[
    tuple[float, int], tuple[int, np.ndarray, np.ndarray, np.ndarray]
] = {}


def _disc_layout(
    r_disc: float, nsec: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return (half size, disc mask, annulus mask, sector index) of a disc score."""
    key = (round(float(r_disc), 1), nsec)
    if key not in _DISC_CACHE:
        half = int(math.ceil(r_disc + 20.0)) + 1
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
        radius = np.hypot(yy, xx)
        sector = (
            ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
        ).ravel()
        on = (radius <= r_disc).ravel()
        flank = ((radius >= r_disc + 8.0) & (radius <= r_disc + 20.0)).ravel()
        _DISC_CACHE[key] = (half, on, flank, sector)
    return _DISC_CACHE[key]


def disc_tscores(
    img: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    r_disc: float,
    nsec: int = 16,
) -> np.ndarray:
    """Sector t-score of the bright disc a bead leaves around its centre.

    In each of ``nsec`` angular sectors the mean of the disc (r <= ``r_disc``) minus
    the mean of the annulus ``r_disc`` + 8 .. + 20 px; t = mean / (sd /
    sqrt(nsec)). A disc is present in every sector (high t), tissue structure is
    not.

    Args:
        img: 2-D image.
        ys: Centre rows (px).
        xs: Centre columns (px).
        r_disc: Disc radius (px).
        nsec: Number of sectors.

    Returns:
        t-scores; NaN where the window leaves the image or holds non-finite values.
    """
    half, on, flank, sector = _disc_layout(r_disc, nsec)
    height, width = img.shape
    on_count = np.bincount(sector[on], minlength=nsec).astype(float)
    flank_count = np.bincount(sector[flank], minlength=nsec).astype(float)
    t = np.full(len(ys), np.nan)
    for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
        y, x = int(round(y)), int(round(x))
        if (
            y - half < 0
            or x - half < 0
            or y + half + 1 > height
            or x + half + 1 > width
        ):
            continue
        w = img[y - half : y + half + 1, x - half : x + half + 1].ravel()
        if not np.isfinite(w).all():
            continue
        d = np.bincount(sector[on], w[on], minlength=nsec) / np.maximum(
            on_count, 1
        ) - np.bincount(sector[flank], w[flank], minlength=nsec) / np.maximum(
            flank_count, 1
        )
        sd = d.std(ddof=1)
        t[i] = d.mean() / (sd / np.sqrt(nsec) + 1e-6)
    return t


def _scores(
    img: np.ndarray, ys: np.ndarray, xs: np.ndarray, radii: np.ndarray, mode: str
) -> np.ndarray:
    """Score each centre at its own (0.5 px rounded) radius."""
    out = np.full(len(ys), np.nan)
    for i, (y, x, r) in enumerate(zip(ys, xs, radii, strict=True)):
        r = float(np.round(r * 2.0) / 2.0)
        if mode == "ring":
            out[i] = ring_tscores(img, np.array([y]), np.array([x]), r)[0]
        else:
            out[i] = disc_tscores(img, np.array([y]), np.array([x]), r + DISC_EXTRA_PX)[
                0
            ]
    return out


def _mode_counts(
    raw: np.ndarray, corrected: np.ndarray
) -> dict[str, int | float | None]:
    """Count the beads whose ring (or disc) is visible before and after."""
    ok = np.isfinite(raw) & np.isfinite(corrected)
    n = int(ok.sum())
    return {
        "n": n,
        "vis_raw": int((raw[ok] > VISIBLE_T).sum()),
        "vis_corr": int((corrected[ok] > VISIBLE_T).sum()),
        "t_raw_median": float(np.median(raw[ok])) if n else None,
        "t_corr_median": float(np.median(corrected[ok])) if n else None,
    }


def residual_ring_scores(
    raw: np.ndarray,
    corrected: np.ndarray,
    beads: np.ndarray,
    *,
    origin_yx: tuple[int, int],
    r1_px: float,
    ell_default: float,
    seed: int = 0,
    max_beads: int = MAX_RING_BEADS_PER_CHUNK,
    margin_px: int = RING_MARGIN_PX,
    n_control: int = CONTROL_POINTS_PER_CHUNK,
) -> dict[str, Any]:
    """Score how much of a bright bead's z1 ring and disc survives the correction.

    The beads scored are the brightest isolated ones (A0 >= ``BRIGHT_A0``, not
    pruned, not an aggregate, no other bead above ``ISOLATION_A0`` within
    ``ISOLATION_PX``, centre at least ``margin_px`` from the edge of the image), at
    most ``max_beads``. Each bead's ring radius is the library z1 radius times its
    fitted stretch. The same score at random control points of the raw image gives
    the false-visible rate of the statistic on this tissue.

    Args:
        raw: Raw max projection of the reference stain (2-D, float).
        corrected: Corrected projection of the same area.
        beads: Bead table ``[n, 8]`` (``model.beads_table`` columns, absolute
            coordinates).
        origin_yx: Absolute coordinate of pixel (0, 0) of the images.
        r1_px: Library z1 ring radius (px).
        ell_default: Stretch used for beads without a fitted value.
        seed: Seed of the control points.
        max_beads: Cap on the scored beads.
        margin_px: Edge margin.
        n_control: Number of control points.

    Returns:
        JSON-able dict with ``n_candidates`` (isolated bright beads before the
        cap) and, for ``ring``, ``disc`` and ``control``, the counts needed to
        pool several chunks.
    """
    height, width = raw.shape
    empty = {"n": 0, "vis_raw": 0, "vis_corr": 0, "t_raw_median": None}
    result: dict[str, Any] = {
        "n_candidates": 0,
        "ring": dict(empty, t_corr_median=None),
        "disc": dict(empty, t_corr_median=None),
        "control": {"n": 0, "ring_vis_raw": 0, "disc_vis_raw": 0},
    }
    inside_possible = height > 2 * margin_px and width > 2 * margin_px
    if not inside_possible:
        return result
    table = np.asarray(beads, np.float64).reshape(-1, 8)
    ys = table[:, 0] - origin_yx[0]
    xs = table[:, 1] - origin_yx[1]
    ell = np.where(np.isfinite(table[:, 6]), table[:, 6], ell_default)
    radius = r1_px * np.exp(ell)
    listed = table[:, 2] >= ISOLATION_A0
    modelled = table[:, 7] == 0
    tree_all = cKDTree(np.c_[ys[listed], xs[listed]]) if listed.any() else None

    bright = (
        modelled
        & (table[:, 2] >= BRIGHT_A0)
        & (table[:, 4] == 0)
        & (ys >= margin_px)
        & (ys < height - margin_px)
        & (xs >= margin_px)
        & (xs < width - margin_px)
    )
    keep = np.flatnonzero(bright)
    if keep.size and tree_all is not None:
        near = tree_all.query_ball_point(
            np.c_[ys[keep], xs[keep]], ISOLATION_PX, return_length=True
        )
        # The bead itself is in the tree (it is above ISOLATION_A0).
        keep = keep[np.asarray(near) <= 1]
    result["n_candidates"] = int(keep.size)
    keep = keep[np.argsort(-table[keep, 2], kind="stable")][:max_beads]
    if keep.size:
        raw_f = np.asarray(raw, np.float32)
        cor_f = np.asarray(corrected, np.float32)
        for mode in ("ring", "disc"):
            result[mode] = _mode_counts(
                _scores(raw_f, ys[keep], xs[keep], radius[keep], mode),
                _scores(cor_f, ys[keep], xs[keep], radius[keep], mode),
            )

    if n_control > 0:
        rng = np.random.default_rng(seed)
        cy = rng.uniform(margin_px, height - margin_px, n_control)
        cx = rng.uniform(margin_px, width - margin_px, n_control)
        if tree_all is not None:
            dist, _ = tree_all.query(np.c_[cy, cx])
            far = dist > CONTROL_CLEARANCE_PX
            cy, cx = cy[far], cx[far]
        if cy.size:
            radii = np.full(cy.size, r1_px * np.exp(ell_default))
            raw_f = np.asarray(raw, np.float32)
            t_ring = _scores(raw_f, cy, cx, radii, "ring")
            t_disc = _scores(raw_f, cy, cx, radii, "disc")
            result["control"] = {
                "n": int(np.isfinite(t_ring).sum()),
                "ring_vis_raw": int((t_ring > VISIBLE_T).sum()),
                "disc_vis_raw": int((t_disc > VISIBLE_T).sum()),
            }
    return result


def pool_ring_scores(
    chunk_scores: Iterable[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Pool the per-chunk residual-ring counts into dataset fractions.

    Args:
        chunk_scores: ``stats["ring_score"]`` of every chunk (None entries are
            skipped).

    Returns:
        ``n_candidates`` and, for ``ring`` and ``disc``, ``n``, the visible counts
        and fractions before and after; for ``control`` the false-visible
        fractions of the raw image.
    """
    pooled: dict[str, Any] = {"n_candidates": 0, "n_chunks": 0}
    sums = {m: {"n": 0, "vis_raw": 0, "vis_corr": 0} for m in ("ring", "disc")}
    control = {"n": 0, "ring_vis_raw": 0, "disc_vis_raw": 0}
    for score in chunk_scores:
        if not score:
            continue
        pooled["n_chunks"] += 1
        pooled["n_candidates"] += int(score.get("n_candidates", 0))
        for mode, acc in sums.items():
            for key in acc:
                acc[key] += int(score[mode][key])
        for key in control:
            control[key] += int(score["control"][key])

    def _frac(num: int, den: int) -> float | None:
        return float(num / den) if den else None

    for mode, acc in sums.items():
        pooled[mode] = {
            **acc,
            "frac_raw": _frac(acc["vis_raw"], acc["n"]),
            "frac_corr": _frac(acc["vis_corr"], acc["n"]),
        }
    pooled["control"] = {
        **control,
        "ring_frac_raw": _frac(control["ring_vis_raw"], control["n"]),
        "disc_frac_raw": _frac(control["disc_vis_raw"], control["n"]),
    }
    return pooled


# ---------------------------------------------------------------------------
# the per-chunk probe
def _block_max(diff: np.ndarray, factor: int = OVERVIEW_FACTOR) -> np.ndarray:
    """Maximum of non-negative values over ``factor`` x ``factor`` blocks."""
    height, width = diff.shape
    pad_h, pad_w = -height % factor, -width % factor
    if pad_h or pad_w:
        diff = np.pad(diff, ((0, pad_h), (0, pad_w)))
    blocks = diff.reshape(
        diff.shape[0] // factor, factor, diff.shape[1] // factor, factor
    )
    out: np.ndarray = blocks.max(axis=(1, 3))
    return out


def chunk_qc_probe(
    *,
    stacks: Mapping[str, Any],
    window_origin: tuple[int, int],
    core: tuple[int, int, int, int],
    z_layers: Sequence[int],
    projections: Mapping[str, np.ndarray],
    beads: np.ndarray,
    calibration: BeadCalibration,
    index: int,
) -> dict[str, Any]:
    """Compute the QC statistics of one chunk from arrays the chunk already holds.

    Args:
        stacks: The window's raw planes, ``{stain: {z: plane}}`` (or arrays
            ``[nz, Y, X]`` with plane index = z).
        window_origin: Absolute coordinate of window pixel (0, 0).
        core: (y0, y1, x0, x1) of the core in absolute mosaic px.
        z_layers: Projected planes.
        projections: Corrected uint16 core projection of every corrected stain.
        beads: Bead table of the core (absolute coordinates).
        calibration: Dataset calibration (library z1 radius, default stretch,
            reference stain).
        index: Chunk index (seeds the control points).

    Returns:
        ``{"ring": <residual_ring_scores dict>, "change": {stain: float32 array}}``;
        ``change`` holds, per 16 px block of the core, the largest value of
        (raw projection - corrected projection).
    """
    wy0, wx0 = window_origin
    cy0, cy1, cx0, cx1 = core
    sy = slice(cy0 - wy0, cy1 - wy0)
    sx = slice(cx0 - wx0, cx1 - wx0)
    raw_proj: dict[str, np.ndarray] = {}
    change: dict[str, np.ndarray] = {}
    for stain, corrected in projections.items():
        raw: np.ndarray | None = None
        for z in z_layers:
            plane = np.asarray(stacks[stain][int(z)])[sy, sx]
            if raw is None:
                raw = plane.astype(np.float32, copy=True)
            else:
                np.maximum(raw, plane, out=raw)
        if raw is None:
            continue
        raw_proj[stain] = raw
        diff = raw - corrected.astype(np.float32)
        change[stain] = _block_max(np.maximum(diff, 0.0)).astype(np.float32)
    ref = calibration.reference_stain
    ring: dict[str, Any] = {}
    if ref in raw_proj and ref in projections:
        r1_px = float(calibration.library.summary()[1]["ring_radius_px"])
        ring = residual_ring_scores(
            raw_proj[ref],
            projections[ref].astype(np.float32),
            beads,
            origin_yx=(cy0, cx0),
            r1_px=r1_px,
            ell_default=float(calibration.ell_default),
            seed=1000 + int(index),
        )
    return {"ring": ring, "change": change}


# ---------------------------------------------------------------------------
# warnings
@dataclass
class QcWarning:
    """One QC warning.

    Attributes:
        code: Short identifier of the kind of warning.
        message: Text for the log and the summary.
    """

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-able form."""
        return {"code": self.code, "message": self.message}


def calibration_warning_code(message: str) -> str:
    """Return the code of a warning raised by the calibration pass.

    Args:
        message: Warning text.

    Returns:
        A short code, ``"calibration"`` for an unknown kind.
    """
    for prefix, code in _CALIBRATION_WARNING_CODES:
        if message.startswith(prefix):
            return code
    return "calibration"


def collect_qc_warnings(
    *,
    calibration: BeadCalibration,
    chunk_stats: Sequence[Mapping[str, Any]],
    ring: Mapping[str, Any] | None,
    reference_stain: str | None = None,
) -> list[QcWarning]:
    """List every warning of a build.

    The calibration's own warnings (low tissue-free area, forecast above 12 h,
    classifier recall, antibody response, ...) are passed through with a code. The
    chunk pass adds: saturated reference pixels, a calibration decision that is
    close to the gate, a fill that mostly does nothing, and a residual bead ring or
    disc that is still visible after the correction.

    Args:
        calibration: The dataset calibration.
        chunk_stats: Per-chunk statistics of the driver.
        ring: Pooled residual-ring scores (``pool_ring_scores``), or None.
        reference_stain: Stain whose fill counters are checked (default: the
            calibration's reference stain).

    Returns:
        The warnings, calibration ones first.
    """
    warnings: list[QcWarning] = [
        QcWarning(calibration_warning_code(str(m)), str(m))
        for m in calibration.decision.get("warnings", [])
    ]

    saturated = sum(
        int(s.get("model", {}).get("n_saturated_px", 0)) for s in chunk_stats
    )
    if saturated:
        n_chunks = sum(
            1 for s in chunk_stats if s.get("model", {}).get("n_saturated_px")
        )
        warnings.append(
            QcWarning(
                "saturation",
                f"{saturated} reference-stain pixels at or above "
                f"{SATURATION_LEVEL:.0f} counts in {n_chunks} chunk(s): saturated "
                "beads are not modelled in this version and may be under-corrected",
            )
        )

    margin = calibration.decision.get("margin_pass")
    if margin is not None and float(margin) < DECISION_MARGIN_WARN:
        warnings.append(
            QcWarning(
                "calibration_margin",
                f"the weakest passing calibration tile is only {float(margin):.4f} "
                f"above the ring-prominence gate (< {DECISION_MARGIN_WARN}): the "
                "bead-presence decision is close to its limit",
            )
        )

    stain = reference_stain or calibration.reference_stain
    fill_tot: dict[str, float] = {}
    for s in chunk_stats:
        for key, val in s.get("fill", {}).get(stain, {}).items():
            if isinstance(val, int | float):
                fill_tot[key] = fill_tot.get(key, 0.0) + float(val)
    cand = fill_tot.get("cand_px", 0.0)
    if cand > 0 and fill_tot.get("dense_skipped_px", 0.0) > FILL_GUARD_SHARE * cand:
        warnings.append(
            QcWarning(
                "fill_guard",
                f"the fill skipped {fill_tot['dense_skipped_px']:.0f} of "
                f"{cand:.0f} candidate pixels as densely coated (premise guard): "
                "the fill is mostly inert on this sample",
            )
        )

    if ring:
        ring_mode, disc_mode = ring.get("ring", {}), ring.get("disc", {})
        n_scored = int(ring_mode.get("n", 0))
        if n_scored >= MIN_SCORED_FOR_WARNING:
            ring_frac = ring_mode.get("frac_corr")
            disc_frac = disc_mode.get("frac_corr")
            if ring_frac is not None and ring_frac > RING_WARN_ABOVE:
                warnings.append(
                    QcWarning(
                        "residual_ring",
                        f"the z1 ring of {ring_frac:.3f} of {n_scored} bright "
                        f"isolated beads is still visible after the correction "
                        f"(> {RING_WARN_ABOVE})",
                    )
                )
            if disc_frac is not None and disc_frac > DISC_WARN_ABOVE:
                warnings.append(
                    QcWarning(
                        "residual_disc",
                        f"the bright disc of {disc_frac:.3f} of "
                        f"{int(disc_mode.get('n', 0))} bright isolated beads is "
                        f"still visible after the correction (> {DISC_WARN_ABOVE})",
                    )
                )
    return warnings


# ---------------------------------------------------------------------------
# tables
def beads_chunk_index(beads: np.ndarray, cores: Sequence[Sequence[int]]) -> np.ndarray:
    """Assign each bead to the chunk whose core holds its centre.

    Args:
        beads: Bead table ``[n, 8]`` (absolute y, x in the first columns).
        cores: ``(y0, y1, x0, x1)`` of the chunks, in chunk order.

    Returns:
        int array [n] with the chunk index, -1 where no core holds the bead.
    """
    out = np.full(len(beads), -1, dtype=int)
    for i, (y0, y1, x0, x1) in enumerate(cores):
        inside = (
            (beads[:, 0] >= y0)
            & (beads[:, 0] < y1)
            & (beads[:, 1] >= x0)
            & (beads[:, 1] < x1)
            & (out < 0)
        )
        out[inside] = i
    return out


def chunk_rows(
    chunk_stats: Sequence[Mapping[str, Any]], z_layers: Sequence[int]
) -> list[dict[str, Any]]:
    """Flatten the per-chunk statistics into table rows (Q8).

    Args:
        chunk_stats: Per-chunk statistics of the driver.
        z_layers: Projected planes (names the per-plane columns).

    Returns:
        One flat dict per chunk.
    """
    rows: list[dict[str, Any]] = []
    for s in chunk_stats:
        core = s.get("core", [None] * 4)
        model = s.get("model", {})
        row: dict[str, Any] = {
            "chunk": s.get("index"),
            "core_y0": core[0],
            "core_y1": core[1],
            "core_x0": core[2],
            "core_x1": core[3],
            "n_candidates": model.get("n_cand"),
            "n_beads": model.get("n_bead"),
            "n_rescued": model.get("n_rescued"),
            "n_blobs": model.get("n_blob"),
            "n_pruned": model.get("n_pruned"),
            "n_beads_in_core": s.get("n_beads_in_core"),
            "gamma_med": model.get("gamma_med"),
            "ell_median": model.get("ell_median"),
            "ell_sd": model.get("ell_sd"),
            "status": model.get("status"),
            "wall_s": s.get("wall_s"),
            "cpu_s": s.get("cpu_s"),
            "peak_rss_gb": s.get("peak_rss_gb"),
            "polyt_px_saturated": model.get("n_saturated_px"),
        }
        if "gpu_s" in s:  # GPU runs only: seconds the window kept the GPU busy
            row["gpu_s"] = s["gpu_s"]
        for stain, per in s.get("per_stain", {}).items():
            for i, z in enumerate(z_layers):
                for key in ("frac_gt10", "frac_gt100", "removed_sum", "removed_p99"):
                    values = per.get(key, [])
                    if i < len(values):
                        row[f"{stain}_z{int(z)}_{key}"] = values[i]
        ring = s.get("ring_score") or {}
        for mode in ("ring", "disc"):
            part = ring.get(mode)
            if part:
                row[f"{mode}_n"] = part["n"]
                row[f"{mode}_vis_raw"] = part["vis_raw"]
                row[f"{mode}_vis_corr"] = part["vis_corr"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dict rows as CSV (the union of the keys, first-seen order)."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_beads_table(
    path: Path, beads: np.ndarray, chunk_index: np.ndarray, max_mb: float = 25.0
) -> dict[str, Any]:
    """Write the modelled beads as ``beads.csv.gz`` (Q5).

    Coordinates and amplitudes are rounded so the file stays small; if it would
    exceed ``max_mb``, only beads with A0 >= ``ISOLATION_A0`` are kept.

    Args:
        path: Output file.
        beads: Bead table ``[n, 8]``.
        chunk_index: Chunk of each bead.
        max_mb: Size limit of the file.

    Returns:
        ``{"rows": n, "truncated": bool}``.
    """
    header = "y,x,A0,a,blob,gamma,ell,pruned,chunk\n"

    def _write(rows: np.ndarray, idx: np.ndarray) -> None:
        with gzip.open(path, "wt", compresslevel=6, newline="") as handle:
            handle.write(header)
            for row, chunk in zip(rows, idx, strict=True):
                handle.write(
                    f"{row[0]:.1f},{row[1]:.1f},{row[2]:.0f},{row[3]:.0f},"
                    f"{int(row[4])},{row[5]:.3f},{row[6]:.3f},{int(row[7])},"
                    f"{int(chunk)}\n"
                )

    _write(beads, chunk_index)
    truncated = False
    if path.stat().st_size > max_mb * 1e6:
        keep = beads[:, 2] >= ISOLATION_A0
        _write(beads[keep], chunk_index[keep])
        truncated = True
    return {
        "rows": int(len(beads))
        if not truncated
        else int((beads[:, 2] >= ISOLATION_A0).sum()),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# JSON helpers
def _clean(value: Any) -> Any:
    """Make a structure strictly JSON-able (NaN and inf become None)."""
    value = to_jsonable(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_clean(data), indent=1, sort_keys=True))


# ---------------------------------------------------------------------------
# QC re-fits of a few windows (before/after crops and residual profiles)
@dataclass
class QcLocation:
    """A place the QC re-fits and shows.

    Attributes:
        kind: ``"calibration"`` (a tissue-free calibration tile) or ``"tissue"``
            (a tissue chunk with many bright beads).
        y0: First row of the crop (absolute mosaic px, multiple of 16).
        x0: First column of the crop.
        note: Short description for the figure title.
    """

    kind: str
    y0: int
    x0: int
    note: str = ""


def select_qc_locations(
    *,
    calibration: BeadCalibration,
    beads: np.ndarray,
    cores: Sequence[Sequence[int]],
    tissue: np.ndarray,
    valid: np.ndarray,
    mosaic_shape: tuple[int, int],
    crop_px: int = 1024,
    n_calibration: int = 3,
    n_tissue: int = 3,
) -> list[QcLocation]:
    """Choose the crops of the before/after figure and the residual profiles (Q10).

    The calibration tiles with the most beads, then the tissue chunks with the most
    bright beads (the crop of ``crop_px`` inside the chunk holding most of them).
    Crops never overlap.

    Args:
        calibration: The calibration (its tiles).
        beads: Bead table ``[n, 8]``.
        cores: Chunk cores.
        tissue: Tissue mask of the 1/16 overview.
        valid: Valid mask of the 1/16 overview.
        mosaic_shape: (height, width) of the mosaic.
        crop_px: Crop edge.
        n_calibration: Number of calibration-tile crops.
        n_tissue: Number of tissue-chunk crops.

    Returns:
        The locations, calibration ones first.
    """
    height, width = mosaic_shape
    crop_px = int(min(crop_px, (height // 16) * 16, (width // 16) * 16))
    if crop_px < 64:
        return []
    chosen: list[QcLocation] = []

    def _place(kind: str, cy: float, cx: float, note: str) -> None:
        y0 = int(round((cy - crop_px / 2) / 16.0) * 16)
        x0 = int(round((cx - crop_px / 2) / 16.0) * 16)
        y0 = int(np.clip(y0, 0, ((height - crop_px) // 16) * 16))
        x0 = int(np.clip(x0, 0, ((width - crop_px) // 16) * 16))
        for other in chosen:
            if abs(other.y0 - y0) < crop_px and abs(other.x0 - x0) < crop_px:
                return
        chosen.append(QcLocation(kind, y0, x0, note))

    tiles = [t for t in calibration.tiles if t.get("status") == "ok"]
    tiles.sort(key=lambda t: -(t.get("n_bead") or 0))
    for tile in tiles:
        if sum(loc.kind == "calibration" for loc in chosen) >= n_calibration:
            break
        size = float(tile.get("tile_px", 2048))
        _place(
            "calibration",
            float(tile["y0"]) + size / 2,
            float(tile["x0"]) + size / 2,
            f"calibration tile {tile.get('index')}",
        )

    bright = (
        beads[(beads[:, 2] >= BRIGHT_A0) & (beads[:, 7] == 0)] if len(beads) else beads
    )
    f = OVERVIEW_FACTOR
    counts: dict[int, int] = {}
    fractions: dict[int, float] = {}
    for i, (y0, y1, x0, x1) in enumerate(cores):
        blocks = (slice(y0 // f, -(-y1 // f)), slice(x0 // f, -(-x1 // f)))
        n_valid = int(valid[blocks].sum())
        if not n_valid:
            continue
        fractions[i] = float(tissue[blocks].sum()) / n_valid
        counts[i] = int(
            (
                (bright[:, 0] >= y0)
                & (bright[:, 0] < y1)
                & (bright[:, 1] >= x0)
                & (bright[:, 1] < x1)
            ).sum()
        )
    # Tissue chunks (at least half tissue; any tissue if none qualifies), most
    # bright beads first.
    pool: list[int] = []
    for floor in (0.5, 0.01):
        pool = [i for i, frac in fractions.items() if frac >= floor and counts[i] > 0]
        if pool:
            break
    pool.sort(key=lambda i: (-counts[i], i))
    step = max(crop_px // 4, 16)
    for i in pool:
        if sum(loc.kind == "tissue" for loc in chosen) >= n_tissue:
            break
        y0, y1, x0, x1 = cores[i]
        in_core = bright[
            (bright[:, 0] >= y0)
            & (bright[:, 0] < y1)
            & (bright[:, 1] >= x0)
            & (bright[:, 1] < x1)
        ]
        best, best_n = (float(y0 + crop_px / 2), float(x0 + crop_px / 2)), -1
        for cy in np.arange(
            y0 + crop_px / 2, max(y1 - crop_px / 2, y0 + crop_px / 2) + 1, step
        ):
            for cx in np.arange(
                x0 + crop_px / 2, max(x1 - crop_px / 2, x0 + crop_px / 2) + 1, step
            ):
                n = int(
                    (
                        (np.abs(in_core[:, 0] - cy) < crop_px / 2)
                        & (np.abs(in_core[:, 1] - cx) < crop_px / 2)
                    ).sum()
                )
                if n > best_n:
                    best, best_n = (float(cy), float(cx)), n
        _place("tissue", best[0], best[1], f"tissue chunk {i}")
    return chosen


@dataclass
class WindowResult:
    """Re-fit of one QC window.

    Attributes:
        location: Where the crop is.
        reference_raw: Raw reference-stain max projection of the crop.
        reference_corrected_z1: Corrected plane 1 of the reference stain (crop).
        reference_raw_z1: Raw plane 1 of the reference stain (crop).
        stain_raw: Raw max projection of every corrected stain (crop).
        stain_refit: The re-fit's corrected max projection (crop).
        profiles: Radial profile arrays per bead: ``raw`` and ``corrected``
            ``[n, n_planes, R + 1]`` in units of A0.
        profile_planes: Planes of the profiles.
        bead_yx: Absolute (y, x) of the profiled beads.
        n_beads: Beads of the re-fit inside the crop.
    """

    location: QcLocation
    reference_raw: np.ndarray
    reference_raw_z1: np.ndarray
    reference_corrected_z1: np.ndarray
    stain_raw: dict[str, np.ndarray] = field(default_factory=dict)
    stain_refit: dict[str, np.ndarray] = field(default_factory=dict)
    profiles: dict[str, np.ndarray] = field(default_factory=dict)
    profile_planes: list[int] = field(default_factory=list)
    bead_yx: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    n_beads: int = 0


def _profile_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (radius index, counts per radius, baseline mask) of a profile window."""
    r = PROFILE_RADIUS_PX
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    radius = np.hypot(yy, xx)
    rb = np.round(radius).astype(int)
    counts = np.bincount(rb.ravel(), minlength=r + 2)
    lo, hi = PROFILE_BASELINE_PX
    return rb, counts, (radius >= lo) & (radius <= hi)


def refit_window(
    reader: Any,
    calibration: BeadCalibration,
    location: QcLocation,
    *,
    z_layers: Sequence[int],
    params: PsfSubtractParams,
    fill: FillParams | None,
    crop_px: int,
    halo_px: int = QC_HALO_PX,
) -> WindowResult:
    """Correct one window again and keep what the QC figures need.

    The window is the crop plus ``halo_px`` on every side, fitted with the planes
    1..max(z_layers) so that the planes of the residual profiles exist. The model
    is the one of the build (same calibration, same decisions); only the window
    differs, so near its edge the result differs from the build by a few counts.

    Args:
        reader: MosaicPlaneReader with the corrected stains.
        calibration: The dataset calibration.
        location: The crop.
        z_layers: Planes of the build.
        params: Algorithm parameters.
        fill: Fill configuration of the build (None = off).
        crop_px: Crop edge.
        halo_px: Halo around the crop.

    Returns:
        The crops and the residual profiles of the isolated bright beads inside the
        crop.
    """
    from merxen.io.bead_removal.apply import correct_window
    from merxen.io.bead_removal.driver import planes_needed, read_window_stacks
    from merxen.io.bead_removal.model import beads_table

    zs = [int(z) for z in z_layers]
    zs_all = list(range(1, max(zs) + 1))
    ref = calibration.reference_stain
    window = (
        location.y0 - halo_px,
        location.y0 + crop_px + halo_px,
        location.x0 - halo_px,
        location.x0 + crop_px + halo_px,
    )
    stacks = read_window_stacks(reader, planes_needed(calibration, zs_all), window)
    out, model, _ = correct_window(
        stacks,
        origin_yx=(window[0], window[2]),
        calibration=calibration,
        z_layers=zs_all,
        params=params,
        fill=fill,
    )
    inner = (slice(halo_px, halo_px + crop_px), slice(halo_px, halo_px + crop_px))
    result = WindowResult(
        location=location,
        reference_raw=np.zeros(0),
        reference_raw_z1=np.asarray(stacks[ref][1])[inner].astype(np.float32),
        reference_corrected_z1=np.asarray(out[ref][0])[inner].astype(np.float32),
    )
    for stain, corrected in out.items():
        raw = None
        refit = None
        for z in zs:
            plane = np.asarray(stacks[stain][z])
            raw = plane if raw is None else np.maximum(raw, plane)
            fixed = corrected[z - 1]
            refit = fixed if refit is None else np.maximum(refit, fixed)
        assert raw is not None and refit is not None
        result.stain_raw[stain] = raw[inner].astype(np.float32)
        result.stain_refit[stain] = refit[inner].astype(np.float32)
    result.reference_raw = result.stain_raw[ref]

    table = beads_table(model, (window[0], window[2]))
    if not len(table):
        return result
    in_crop = (
        (table[:, 0] >= location.y0)
        & (table[:, 0] < location.y0 + crop_px)
        & (table[:, 1] >= location.x0)
        & (table[:, 1] < location.x0 + crop_px)
    )
    result.n_beads = int((in_crop & (table[:, 7] == 0)).sum())
    modelled = table[:, 7] == 0
    listed = modelled & (table[:, 2] >= ISOLATION_A0)
    tree = cKDTree(table[listed, :2]) if listed.any() else None
    pick = np.flatnonzero(
        in_crop & modelled & (table[:, 2] >= BRIGHT_A0) & (table[:, 4] == 0)
    )
    if tree is not None and pick.size:
        near = tree.query_ball_point(
            table[pick, :2], PROFILE_ISOLATION_PX, return_length=True
        )
        pick = pick[np.asarray(near) <= 1]
    planes = [z for z in PROFILE_PLANES if z in zs_all]
    result.profile_planes = planes
    if not pick.size or not planes:
        return result
    rb, counts, baseline = _profile_geometry()
    r = PROFILE_RADIUS_PX
    raw_prof = np.zeros((pick.size, len(planes), r + 1), np.float32)
    cor_prof = np.zeros_like(raw_prof)
    for b, row in enumerate(pick):
        y = int(round(table[row, 0] - window[0]))
        x = int(round(table[row, 1] - window[2]))
        a0 = float(table[row, 2])
        for pi, z in enumerate(planes):
            for target, source in (
                (raw_prof, stacks[ref][z]),
                (cor_prof, out[ref][z - 1]),
            ):
                w = np.asarray(source)[y - r : y + r + 1, x - r : x + r + 1].astype(
                    np.float32
                )
                if w.shape != (2 * r + 1, 2 * r + 1):
                    target[b, pi] = np.nan
                    continue
                w = w - w[baseline].mean()
                target[b, pi] = (
                    np.bincount(rb.ravel(), w.ravel(), minlength=r + 2)
                    / np.maximum(counts, 1)
                )[: r + 1] / a0
    result.profiles = {"raw": raw_prof, "corrected": cor_prof}
    result.bead_yx = table[pick, :2]
    return result


def summarise_profiles(
    windows: Sequence[WindowResult],
    calibration: BeadCalibration,
    tissue: np.ndarray,
) -> dict[str, Any]:
    """Pool the residual profiles of all windows into tissue and off-tissue medians.

    The profile of a bead is the azimuthal mean of (plane - mean of the 150-170 px
    annulus) / A0, in units of 1e-3 A0. Over-subtraction shows as a negative
    residual in the disc or at the ring, a radius mismatch as a dipole. Under
    tissue the z3 / z4 rings are only partly removed (documented limit R1); the
    numbers here make that visible per sample.

    Args:
        windows: Re-fitted windows.
        calibration: The calibration (ring radius per plane).
        tissue: Tissue mask of the 1/16 overview.

    Returns:
        JSON-able dict: per group (``tissue``, ``off_tissue``) the bead count and,
        per plane, the band medians and the median profile for raw and corrected.
    """
    planes = sorted({z for w in windows for z in w.profile_planes})
    rows = calibration.library.summary()
    groups: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"raw": [], "corrected": []} for name in ("tissue", "off_tissue")
    }
    for w in windows:
        if not w.profiles:
            continue
        for b, (y, x) in enumerate(w.bead_yx):
            by = min(int(y // OVERVIEW_FACTOR), tissue.shape[0] - 1)
            bx = min(int(x // OVERVIEW_FACTOR), tissue.shape[1] - 1)
            name = "tissue" if bool(tissue[by, bx]) else "off_tissue"
            for kind in ("raw", "corrected"):
                padded = np.full(
                    (len(planes), PROFILE_RADIUS_PX + 1), np.nan, np.float32
                )
                for pi, z in enumerate(w.profile_planes):
                    padded[planes.index(z)] = w.profiles[kind][b, pi]
                groups[name][kind].append(padded)
    radii = np.arange(PROFILE_RADIUS_PX + 1)
    out: dict[str, Any] = {
        "unit": "1e-3 x A0",
        "radius_px": radii.tolist(),
        "baseline_annulus_px": list(PROFILE_BASELINE_PX),
        "groups": {},
    }
    for name, parts in groups.items():
        n = len(parts["raw"])
        entry: dict[str, Any] = {"n_beads": n, "planes": {}}
        out["groups"][name] = entry
        if not n:
            continue
        raw = np.stack(parts["raw"])
        cor = np.stack(parts["corrected"])
        for pi, z in enumerate(planes):
            rc = float(rows[z]["ring_radius_px"])
            width = max(4.0, 0.25 * rc)
            bands = {
                "disc": radii < rc - width,
                "ring": (radii >= rc - width / 2) & (radii <= rc + width / 2),
                "outer": (radii > rc + width) & (radii <= rc + 2.5 * width),
            }
            plane_entry: dict[str, Any] = {
                "ring_radius_px": rc,
                "bands_px": {
                    "disc": [0.0, rc - width],
                    "ring": [rc - width / 2, rc + width / 2],
                    "outer": [rc + width, rc + 2.5 * width],
                },
            }
            for kind, arr in (("raw", raw), ("corrected", cor)):
                per_bead = arr[:, pi]
                plane_entry[kind] = {
                    band: float(
                        np.nanmedian(np.nanmean(per_bead[:, mask], axis=1)) * 1e3
                    )
                    for band, mask in bands.items()
                }
                plane_entry[kind]["profile"] = (
                    np.nanmedian(per_bead, axis=0) * 1e3
                ).tolist()
            entry["planes"][str(z)] = plane_entry
    return out


def _cell_grid(
    ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int], cell: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell row and column of each point and the grid shape."""
    n_rows, n_cols = -(-shape[0] // cell), -(-shape[1] // cell)
    iy = np.clip((ys // cell).astype(int), 0, n_rows - 1)
    ix = np.clip((xs // cell).astype(int), 0, n_cols - 1)
    return iy, ix


def bead_density_map(
    beads: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
    px_per_um: float,
    *,
    min_a0: float,
    cell: int = 512,
) -> np.ndarray:
    """Count modelled beads per mm2 on a ``cell`` px grid.

    Args:
        beads: Bead table ``[n, 8]``.
        valid: Valid mask of the 1/16 overview.
        shape: Mosaic shape (px).
        px_per_um: Pixel scale.
        min_a0: Only beads with at least this amplitude are counted.
        cell: Grid cell edge (px).

    Returns:
        Beads per mm2 per cell, NaN where a cell holds no valid pixel.
    """
    n_rows, n_cols = -(-shape[0] // cell), -(-shape[1] // cell)
    sel = beads[(beads[:, 7] == 0) & (beads[:, 2] >= min_a0)] if len(beads) else beads
    counts = np.zeros((n_rows, n_cols))
    if len(sel):
        iy, ix = _cell_grid(sel[:, 0], sel[:, 1], shape, cell)
        np.add.at(counts, (iy, ix), 1.0)
    per = max(cell // OVERVIEW_FACTOR, 1)
    padded = np.zeros((n_rows * per, n_cols * per), bool)
    h, w = min(valid.shape[0], padded.shape[0]), min(valid.shape[1], padded.shape[1])
    padded[:h, :w] = valid[:h, :w]
    frac = padded.reshape(n_rows, per, n_cols, per).mean(axis=(1, 3))
    area_mm2 = (cell / px_per_um / 1000.0) ** 2
    density = counts / area_mm2
    return np.where(frac > 0, density, np.nan)


def stretch_field_map(
    beads: np.ndarray, shape: tuple[int, int], *, cell: int = 256
) -> np.ndarray:
    """Median fitted stretch (ell) per ``cell`` px cell; NaN where no bead has one."""
    n_rows, n_cols = -(-shape[0] // cell), -(-shape[1] // cell)
    out = np.full((n_rows, n_cols), np.nan)
    sel = beads[(beads[:, 7] == 0) & np.isfinite(beads[:, 6])] if len(beads) else beads
    if not len(sel):
        return out
    iy, ix = _cell_grid(sel[:, 0], sel[:, 1], shape, cell)
    flat = iy * n_cols + ix
    order = np.argsort(flat, kind="stable")
    flat, ell = flat[order], sel[order, 6]
    for cell_id in np.unique(flat):
        lo, hi = np.searchsorted(flat, [cell_id, cell_id + 1])
        out[cell_id // n_cols, cell_id % n_cols] = float(np.median(ell[lo:hi]))
    return out


# ---------------------------------------------------------------------------
# the report
def load_change_overview(work_dir: Path) -> dict[str, np.ndarray]:
    """Read the driver's 1/16 change overview (empty when the file is missing)."""
    path = Path(work_dir) / "change_overview.npz"
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files if not k.startswith("_")}


def _dir_size_bytes(folder: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()}


def write_bead_removal_qc(
    qc_dir: str | Path,
    *,
    calibration: BeadCalibration,
    run: BeadRemovalRunResult,
    images_dir: str | Path,
    projection: Any,
    stainings: Sequence[str],
    z_layers: Sequence[int],
    params: PsfSubtractParams = DEFAULT_PARAMS,
    fill: FillParams | None = None,
    crop_px: int = 1024,
    n_calibration_locations: int = 3,
    n_tissue_locations: int = 3,
) -> dict[str, Any]:
    """Write the QC folder of a bead-removal build.

    Everything is derived from the calibration, the driver's run record and files
    (bead table, chunk statistics, change overview), a 1/16 re-read of the raw
    mosaic for the tissue mask, and a re-fit of a few windows for the before/after
    crops and the residual profiles. A part that fails is recorded under
    ``errors`` and does not stop the others.

    Args:
        qc_dir: Folder to write (created; its parent must exist).
        calibration: The dataset calibration.
        run: The driver's run record (statistics in memory, files in
            ``run.work_dir``).
        images_dir: Folder with the raw mosaic planes.
        projection: The built projection ``[channel, y, x]`` (a zarr array or
            anything that slices like one).
        stainings: Channel names of ``projection``.
        z_layers: Projected planes.
        params: Algorithm parameters of the build.
        fill: Fill configuration of the build (None = off).
        crop_px: Edge of the before/after crops (the re-fit window adds 256 px of
            halo on each side).
        n_calibration_locations: Crops on calibration tiles.
        n_tissue_locations: Crops on tissue chunks.

    Returns:
        The summary that is written to ``summary.json``.
    """
    from merxen.io.bead_removal.reader import MosaicPlaneReader

    t_start = time.time()
    out = Path(qc_dir)
    out.mkdir(exist_ok=True)
    work = Path(run.work_dir)
    zs = [int(z) for z in z_layers]
    ref = calibration.reference_stain
    chunk_stats = list(run.chunk_stats)
    cores = [s["core"] for s in chunk_stats]
    errors: list[dict[str, str]] = []

    def _guard(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a QC part must not stop the others
            logger.exception("[BEADS] QC part %s failed", name)
            errors.append({"part": name, "error": f"{type(exc).__name__}: {exc}"})
            return None

    beads_path = work / "beads.npy"
    beads = np.load(beads_path) if beads_path.exists() else np.zeros((0, 8))
    beads = np.asarray(beads, np.float64).reshape(-1, 8)
    chunk_index = beads_chunk_index(beads, cores)

    # Q1 inputs that need no image.
    pooled_ring = pool_ring_scores(s.get("ring_score") for s in chunk_stats)
    warnings = collect_qc_warnings(
        calibration=calibration, chunk_stats=chunk_stats, ring=pooled_ring
    )
    px_per_um = float(calibration.forecast.get("px_per_um", DEFAULT_PX_PER_UM))

    # Q2, Q3, Q5, Q8.
    _write_json(out / "calibration.json", _calibration_record(calibration))
    calibration.library.save(out / "psf_library.npz")
    _guard(
        "psf_montage", lambda: write_psf_montage(calibration, out / "psf_montage.png")
    )
    beads_info = _guard(
        "beads_table",
        lambda: write_beads_table(out / "beads.csv.gz", beads, chunk_index),
    )
    _guard(
        "chunks_table",
        lambda: _write_csv(out / "chunks.csv", chunk_rows(chunk_stats, zs)),
    )

    # Overview, tissue mask and maps.
    reader = MosaicPlaneReader(
        images_dir,
        sorted(set(calibration.corrected_stains) | {ref, "DAPI"}),
        _planes_for(calibration, zs),
    )
    overview: dict[str, Any] = {}
    windows: list[WindowResult] = []
    profiles: dict[str, Any] = {}
    try:

        def _overview() -> None:
            z_mid = int(calibration.tissue.get("z_mid", zs[len(zs) // 2]))
            dapi_ov, valid_ov = build_overview(
                reader, reference_stain=ref, dapi_stain="DAPI", z_mid=z_mid
            )
            tissue_mask, _ = tissue_mask_from_overview(dapi_ov, valid_ov)
            overview.update(dapi=dapi_ov, valid=valid_ov, tissue=tissue_mask)

        _guard("overview", _overview)
        shape = reader.shape
        locations: list[QcLocation] = []
        if overview:
            locations = (
                _guard(
                    "locations",
                    lambda: select_qc_locations(
                        calibration=calibration,
                        beads=beads,
                        cores=cores,
                        tissue=overview["tissue"],
                        valid=overview["valid"],
                        mosaic_shape=shape,
                        crop_px=crop_px,
                        n_calibration=n_calibration_locations,
                        n_tissue=n_tissue_locations,
                    ),
                )
                or []
            )
            _guard(
                "overview_figure",
                lambda: write_overview_figure(
                    out / "overview.png",
                    dapi_overview=overview["dapi"],
                    valid=overview["valid"],
                    tissue=overview["tissue"],
                    calibration=calibration,
                    cores=cores,
                    locations=locations,
                    crop_px=min(crop_px, *shape),
                ),
            )
            _guard(
                "density_figure",
                lambda: write_map_figure(
                    out / "bead_density.png",
                    [
                        (
                            f"modelled beads with A0 >= {a0:.0f}",
                            bead_density_map(
                                beads, overview["valid"], shape, px_per_um, min_a0=a0
                            ),
                            "beads per mm2",
                        )
                        for a0 in (ISOLATION_A0, BRIGHT_A0)
                    ],
                    cell=512,
                    cores=cores,
                    shape=shape,
                    suptitle="Bead density on a 512 px grid (chunk grid in white)",
                ),
            )
        _guard(
            "stretch_figure",
            lambda: write_map_figure(
                out / "stretch_field.png",
                [
                    (
                        "median stretch ell per 256 px cell",
                        stretch_field_map(beads, shape, cell=256),
                        "ell",
                    )
                ],
                cell=256,
                cores=cores,
                shape=shape,
                suptitle="Fitted stretch field (a field-of-view tilt shows here)",
            ),
        )
        change = load_change_overview(work)
        if change:
            _guard(
                "change_overview",
                lambda: write_change_overview(
                    out / "change_overview.png", change, cores=cores
                ),
            )

        # Q9 profiles and Q10 crops: a re-fit of a few windows.
        window_crop = int(min(crop_px, *shape))

        def _refit(loc: QcLocation) -> WindowResult:
            return refit_window(
                reader,
                calibration,
                loc,
                z_layers=zs,
                params=params,
                fill=fill,
                crop_px=window_crop,
            )

        for loc in locations:
            result = _guard(
                f"window_{loc.kind}_{loc.y0}_{loc.x0}", functools.partial(_refit, loc)
            )
            if result is not None:
                windows.append(result)
        if windows and overview:
            profiles = (
                _guard(
                    "residual_profiles",
                    lambda: summarise_profiles(
                        windows, calibration, overview["tissue"]
                    ),
                )
                or {}
            )
            if profiles:
                _write_json(out / "residual_profiles.json", profiles)
                _guard(
                    "residual_profile_figure",
                    lambda: write_residual_profile_figure(
                        profiles, out / "residual_profiles.png"
                    ),
                )
        if windows:
            built = []
            for w in windows:
                size = w.reference_raw.shape[0]
                built.append(
                    {
                        stain: np.asarray(
                            projection[
                                list(stainings).index(stain),
                                w.location.y0 : w.location.y0 + size,
                                w.location.x0 : w.location.x0 + size,
                            ]
                        )
                        for stain in w.stain_raw
                    }
                )
            _guard(
                "before_after",
                lambda: write_before_after(
                    windows,
                    built,
                    out / "before_after.png",
                    out / "before_after_dapi.png",
                    out / "before_after_crops.npz",
                    reference_stain=ref,
                    display_px=int(min(crop_px, *shape) * 3 // 4),
                ),
            )
    finally:
        reader.close()

    summary = _summary_record(
        calibration=calibration,
        run=run,
        chunk_stats=chunk_stats,
        beads=beads,
        beads_info=beads_info,
        overview=overview,
        px_per_um=px_per_um,
        pooled_ring=pooled_ring,
        profiles=profiles,
        windows=windows,
        locations=locations,
        warnings=warnings,
        errors=errors,
        params=params,
        fill=fill,
        z_layers=zs,
        qc_s=time.time() - t_start,
        crop_px=crop_px,
    )
    _write_json(out / "summary.json", summary)
    sizes = _dir_size_bytes(out)
    total_mb = sum(sizes.values()) / 1e6
    summary["files"] = sizes
    summary["total_size_mb"] = total_mb
    if total_mb > QC_MAX_MB:
        warnings.append(
            QcWarning(
                "qc_size", f"the QC folder is {total_mb:.1f} MB (> {QC_MAX_MB:.0f} MB)"
            )
        )
        summary["warnings"] = [w.to_dict() for w in warnings]
        summary["status"] = "warn"
    _write_json(out / "summary.json", summary)
    _log_summary(summary)
    return summary


def _planes_for(calibration: BeadCalibration, zs: Sequence[int]) -> list[int]:
    """Planes every QC read needs: 0..max(z_layers, 2)."""
    return list(range(0, max(2, max(zs)) + 1))


#: Residual risk of the rescue, stated in the QC summary whenever the rescue ran.
RESCUE_RESIDUAL_RISK = (
    "the rescue can remove a compact tissue object in flat surroundings (a cell-edge "
    "punctum) with a bead PSF; it is rare (about 0.08 rescues per 1024 px tissue "
    "tile on the measured bead-free tissue) and the local damage was at most about "
    "600 counts within about 30 px"
)


def _rescue_record(
    calibration: BeadCalibration, chunk_stats: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Describe the z1-ring rescue of the classifier for summary.json.

    Args:
        calibration: The dataset calibration (its thresholds and classifier info).
        chunk_stats: Per-chunk statistics of the driver.

    Returns:
        Whether the rescue was active, its per-dataset gate with the raw statistic it
        was clipped from, the rescued counts of the calibration tiles and of the
        windows (``n_rescued`` counts the beads of the windows including their
        halos, like ``n_beads``), and the residual risk when it was active.
    """
    thresholds = calibration.thresholds
    classifier = calibration.decision.get("classifier", {})
    n_beads = sum(int(s.get("model", {}).get("n_bead") or 0) for s in chunk_stats)
    n_rescued = sum(int(s.get("model", {}).get("n_rescued") or 0) for s in chunk_stats)
    record: dict[str, Any] = {
        "active": bool(thresholds.rescue_active),
        "rescue_sd1_max": float(thresholds.rescue_sd1_max),
        "sd1_rel_p98_raw": classifier.get("sd1_rel_p98_raw"),
        "sd1_rel_clamped": "rescue_sd1_max" in classifier.get("clamped", []),
        "rescue_zprof_zmax": float(thresholds.rescue_zprof_zmax),
        "rescue_tz1_min": float(thresholds.rescue_tz1_min),
        "n_rescued_tiles": classifier.get("n_rescued_tiles"),
        "rescued_fraction_tiles": classifier.get("rescued_fraction_tiles"),
        "n_rescued_windows": int(n_rescued),
        "rescued_fraction": float(n_rescued / n_beads) if n_beads else None,
    }
    if not thresholds.rescue_active:
        record["off_reason"] = classifier.get("rescue_off_reason", "not derived")
    else:
        record["residual_risk"] = RESCUE_RESIDUAL_RISK
    return record


def _calibration_record(calibration: BeadCalibration) -> dict[str, Any]:
    """Return calibration.json (Q2): the calibration and the library vs its laws."""
    record = calibration.to_dict()
    record["library_vs_law"] = [
        dict(row, flux_target=float(FLUX_TARGET[int(row["plane"])]))
        for row in calibration.library.summary()
    ]
    return record


def _summary_record(
    *,
    calibration: BeadCalibration,
    run: BeadRemovalRunResult,
    chunk_stats: Sequence[Mapping[str, Any]],
    beads: np.ndarray,
    beads_info: Mapping[str, Any] | None,
    overview: Mapping[str, Any],
    px_per_um: float,
    pooled_ring: Mapping[str, Any],
    profiles: Mapping[str, Any],
    windows: Sequence[WindowResult],
    locations: Sequence[QcLocation],
    warnings: Sequence[QcWarning],
    errors: Sequence[Mapping[str, str]],
    params: PsfSubtractParams,
    fill: FillParams | None,
    z_layers: Sequence[int],
    qc_s: float,
    crop_px: int,
) -> dict[str, Any]:
    """Assemble summary.json (Q1)."""
    all_warnings = list(warnings)
    if errors:
        all_warnings.append(
            QcWarning(
                "qc_incomplete",
                "parts of the QC report could not be produced: "
                + ", ".join(e["part"] for e in errors),
            )
        )
    valid = overview.get("valid")
    area_mm2 = (
        float(valid.sum()) * OVERVIEW_FACTOR**2 / (px_per_um**2 * 1e6)
        if valid is not None
        else None
    )
    modelled = beads[beads[:, 7] == 0] if len(beads) else beads

    def _density(min_a0: float) -> float | None:
        if not area_mm2:
            return None
        return float((modelled[:, 2] >= min_a0).sum()) / area_mm2

    fill_totals: dict[str, dict[str, float]] = {}
    for s in chunk_stats:
        for stain, counters in s.get("fill", {}).items():
            acc = fill_totals.setdefault(stain, {})
            for key, val in counters.items():
                if isinstance(val, int | float):
                    acc[key] = acc.get(key, 0.0) + float(val)
    decision = calibration.decision
    classifier = decision.get("classifier", {})
    summary: dict[str, Any] = {
        "status": "warn" if all_warnings else "pass",
        "warnings": [w.to_dict() for w in all_warnings],
        "errors": list(errors),
        "method": METHOD_NAME,
        "version": BEAD_REMOVAL_VERSION,
        "params_sha256": calibration.params_sha256,
        "calibration_sha256": calibration.calibration_sha256(),
        "z_layers": list(z_layers),
        "corrected_stains": list(calibration.corrected_stains),
        "reference_stain": calibration.reference_stain,
        "n_chunks": int(run.n_chunks),
        "n_beads_listed": int(len(beads)),
        "n_beads_modelled": int(len(modelled)),
        "valid_area_mm2": area_mm2,
        "beads_per_mm2_a0_400": _density(ISOLATION_A0),
        "beads_per_mm2_a0_1600": _density(BRIGHT_A0),
        "runtime_s": {
            "calibration": decision.get("timing_s", {}).get("total_s"),
            "chunks": float(run.wall_s),
            "qc": float(qc_s),
        },
        "n_workers": int(run.n_workers),
        "device": run.device,
        "backend": dict(run.backend),
        **({"gpu": dict(run.gpu)} if run.gpu is not None else {}),
        "peak_rss_gb": {
            "process_tree": float(run.peak_tree_rss_gb),
            "process_tree_anon": float(run.peak_tree_anon_gb),
            "worker": float(run.peak_worker_rss_gb),
            "worker_anon": float(run.peak_worker_anon_gb),
        },
        "calibration": {
            "n_tiles": decision.get("n_tiles"),
            "n_pass": decision.get("n_pass"),
            "required": decision.get("required"),
            "gate": decision.get("gate"),
            "margin_pass": decision.get("margin_pass"),
            "focus": decision.get("focus"),
            "thresholds": calibration.thresholds.to_dict(),
            "rho": {k: [v[0], v[1]] for k, v in calibration.rho.items()},
            "forecast": calibration.forecast,
            "tissue": {
                k: calibration.tissue.get(k)
                for k in ("threshold_counts", "tissue_fraction", "low_tissue_tiles")
            },
        },
        # Reported only (never a warning): how much the calibration tiles
        # disagree on the PSF library.
        "library_spread": decision.get("library_spread", {}).get("summary", {}),
        "classifier": {
            k: classifier.get(k)
            for k in (
                "recall",
                "n_recall_reference",
                "recall_with_clusters",
                "n_recall_reference_with_clusters",
                "recall_in_sample",
                "n_reference",
                "used_defaults",
            )
        },
        "rescue": _rescue_record(calibration, chunk_stats),
        "saturation": {
            "n_pixels": int(
                sum(
                    int(s.get("model", {}).get("n_saturated_px", 0))
                    for s in chunk_stats
                )
            ),
            "level": SATURATION_LEVEL,
        },
        "fill": {
            "enabled": bool(fill is not None and fill.enabled),
            "counters": fill_totals,
        },
        "residual_ring": dict(pooled_ring),
        "residual_ring_limits": {
            "z1_ring_visible_corrected": RING_WARN_ABOVE,
            "disc_visible_corrected": DISC_WARN_ABOVE,
            "min_beads_for_warning": MIN_SCORED_FOR_WARNING,
        },
        "residual_profiles": {
            name: group["n_beads"] for name, group in profiles.get("groups", {}).items()
        },
        "crops": {
            "crop_px": int(crop_px),
            "locations": [
                {"kind": loc.kind, "y0": loc.y0, "x0": loc.x0, "note": loc.note}
                for loc in locations
            ],
            "refit_windows": len(windows),
        },
        "beads_table": dict(beads_info) if beads_info else None,
    }
    return summary


def _log_summary(summary: Mapping[str, Any]) -> None:
    """Log the final QC summary (Q12); every warning is logged as a warning."""
    calibration = summary["calibration"]
    logger.info(
        "[BEADS] QC calibration: %s of %s tiles passed (%s required), margin above "
        "the gate %s, thresholds %s, forecast %s h wall (%s backend)",
        calibration.get("n_pass"),
        calibration.get("n_tiles"),
        calibration.get("required"),
        calibration.get("margin_pass"),
        calibration.get("thresholds"),
        calibration.get("forecast", {}).get("wall_hours"),
        calibration.get("forecast", {}).get("backend"),
    )
    ring = summary["residual_ring"]
    logger.info(
        "[BEADS] QC %s: %d chunks, %d beads modelled (%.0f / mm2 with A0 >= 400), "
        "residual z1 ring visible %s (raw %s), disc %s (raw %s), %.1f MB",
        summary["status"],
        summary["n_chunks"],
        summary["n_beads_modelled"],
        summary["beads_per_mm2_a0_400"] or float("nan"),
        _fmt_frac(ring.get("ring", {}).get("frac_corr")),
        _fmt_frac(ring.get("ring", {}).get("frac_raw")),
        _fmt_frac(ring.get("disc", {}).get("frac_corr")),
        _fmt_frac(ring.get("disc", {}).get("frac_raw")),
        summary.get("total_size_mb", float("nan")),
    )
    rescue = summary.get("rescue") or {}
    if rescue.get("active"):
        logger.info(
            "[BEADS] QC rescue: %s beads rescued in the windows (%s of the beads), "
            "texture gate %.4f (raw P98 %s), z-profile to plane %d",
            rescue.get("n_rescued_windows"),
            _fmt_frac(rescue.get("rescued_fraction")),
            rescue["rescue_sd1_max"],
            _fmt_frac(rescue.get("sd1_rel_p98_raw")),
            int(rescue["rescue_zprof_zmax"]),
        )
    gpu = summary.get("gpu")
    if gpu:
        logger.info(
            "[BEADS] QC GPU backend: waited %.1f s for the GPU lock (%s), %.1f s of "
            "GPU work, %.1f s per window (median), %.1f s (max)",
            gpu["lock_wait_s"],
            "lock file set" if gpu["locked"] else "no lock file",
            gpu["gpu_s"],
            gpu["gpu_s_per_window"]["median"],
            gpu["gpu_s_per_window"]["max"],
        )
    for warning in summary["warnings"]:
        logger.warning(
            "[BEADS] QC warning (%s): %s", warning["code"], warning["message"]
        )


def _fmt_frac(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
