"""Optional bounded, one-sided, chunk-invariant fill of residual bead light.

Experimental and off by default. The subtraction
(:func:`merxen.io.bead_removal.apply.apply_bead_model`) can leave residual bead
light where the model under-predicts, mostly on the rings of the brightest
aggregates in plane 1. This fill replaces the pixels where the model's own
uncertainty is large by an interpolation from the surrounding pixels, but only
where that interpolation is trustworthy: where the pixels around the hole are
flat. Off tissue the background is flat to within the pixel noise; in tissue the
fill does nothing ("flat background only, inert in tissue"), so its benefit is
small and limited to off-tissue bead rings.

Why flat surroundings only: inpainting cannot tell a bead residual from tissue.
Inside a bead footprint every pixel above the interpolation (cell bodies,
processes, ridges) looks like residual. On a perfect subtraction (no bead
residual at all) an earlier version of this fill still changed 2.9 % of the
pixels and removed 91 counts per changed pixel, and no texture measure (local sd
of the high-passed plane) identifies tissue that is smooth. Restricting the fill
to flat surroundings removes that damage by construction.

Guarantees (all covered by tests):

* ``FillParams.enabled`` is False by default; a disabled fill returns the
  subtraction result unchanged (bit-identical).
* The result is never brighter than the subtraction result and never lower than
  ``corrected - eps x model light`` (max-filtered over the mask dilation): the
  fill can only remove what the model could have missed, and only that much.
* Chunk invariance: every operation has a fixed support (Gaussian filters of
  fixed sigma, 8 px block sums, a maximum filter of fixed radius for the
  bright-source restriction) that is below the 256 px halo (see
  :func:`support_px`; :class:`FillParams` refuses settings whose support exceeds
  254 px), and no statistic depends on the window extent. The only inputs besides
  the pixels are dataset-global constants (the per-plane pixel noise of the
  calibration). The arrays must start on the absolute 16 px grid
  (``correct_window`` guarantees it) because the block sums are aligned to the
  array origin.
* Only candidate regions are processed (crops around regions where the model
  uncertainty exceeds the gate), so the cost follows the number of bright beads,
  not the window area.

The interpolation is a normalised-convolution cascade from coarse to fine levels
with fixed Gaussian supports; a hole that is larger than the coarsest support is
left unchanged (never invented).

Hybrid mode (``FillParams.mode == "hybrid"``) is the original, more aggressive
behaviour of the method, offered as an option. It fills wherever the model's own
uncertainty ``eps x light`` exceeds ``gate x`` the local texture sd of the
corrected plane (robust sd of the high-passed plane in blocks aligned to the
absolute mosaic grid), also in tissue. There is no flatness gate, no noise
deadband (hard one-sided clip), no bright-source restriction and no dense-coating
skip, so tissue puncta next to a bead can be dimmed. The bound is unchanged: the
result is never brighter than the subtraction result and never lower than
``corrected - eps x model light`` (max-filtered over the mask dilation). The
texture map adds to the support (see :func:`support_px`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

from merxen.io.bead_removal.params import NZ_MAX

logger = logging.getLogger(__name__)

# Largest allowed reach (px) of the bright-source restriction.
MAX_REACH_PX = 64
# halo 256 - 2: the largest support that stays chunk invariant inside the
# default halo.
MAX_SUPPORT_PX = 254
# Coarse grid (px) of the candidate regions, the coarse fill levels and the
# flatness test.
BLOCK = 8
# Candidate blocks closer than this (in blocks) share one crop.
MERGE_BLOCKS = 6
# Side (blocks, 264 px) of the box in which the candidate coverage is measured.
COVER_BLOCKS = 33
# A block takes part in the flatness test if at least this fraction of its
# pixels is known.
MIN_KNOWN_BLOCK = 0.25
# FillParams fields that only the hybrid mode reads (not hashed in flat mode).
_HYBRID_ONLY_FIELDS = (
    "mode",
    "texture_block",
    "texture_hp_sigma",
    "texture_smooth",
    "texture_radius_blocks",
)
_SQUARE = ndi.generate_binary_structure(2, 2)
_CROSS = ndi.generate_binary_structure(2, 1)


@dataclass(frozen=True)
class FillParams:
    """Settings of the bounded fill.

    The defaults are the recommended configuration used when the option is
    switched on. The fill is off unless ``enabled`` is True; with
    ``enabled=False`` every other field is ignored (but still validated).

    Attributes:
        enabled: Master switch (default False).
        stains: Stains the fill runs on. Only the reference stain can run: the
            pixel-noise floor, which every decision of the fill is scaled by, is
            the calibration's per-plane noise of the reference stain. Must equal
            ``(reference_stain,)``.
        reference_stain: Stain whose planes the calibration's noise constants
            belong to.
        eps: Per plane (index = z): the fill may remove at most
            ``eps[z] x model light`` below the subtraction result. 0 switches a
            plane off. 0.3 covers about the 95th percentile of the measured
            residual / model light ratio at bright pixels on synthetic ground
            truth (the median is negative: the subtraction over-subtracts).
        gate: A pixel is a fill candidate where ``eps x light > gate x noise
            floor``.
        floor_frac: Noise floor = ``floor_frac x`` the dataset-global pixel
            noise of the plane.
        dilation: Dilation (px, cross structure) of the candidate mask before
            filling; also the support of the bound.
        peak_snr: Restriction to bright sources: a candidate pixel is kept only
            if a bright candidate pixel (``eps x light`` at least ``peak_snr x``
            the global pixel noise of the plane) lies within ``bright_reach_px``
            of it. With a pixel noise of 37 counts and eps 0.3, 3 corresponds to
            beads with z0 amplitude of about 1,600-3,000 counts at plane 1 (and
            about 4,000-6,000 at plane 2). 0 disables the restriction.
        bright_reach_px: Reach (px, Chebyshev) of the bright-source restriction.
            It is bounded on purpose: a connected-component test has an unbounded
            reach (faint candidate bridges of several hundred px were measured),
            which made the result depend on the window cut. At most
            ``MAX_REACH_PX``; it adds to :func:`support_px`.
        planes: Planes the fill may act on (None = all planes passed). Plane 0 is
            never passed by ``correct_window``.
        flat_max: Flatness limit, in units of the noise floor. The fill acts only
            where the known pixels around the hole (outside the candidate set),
            measured as 8 px block means with the coarse fill cascade, vary by no
            more than ``flat_max x`` the pixel noise. 1.0 = "flat to within one
            pixel-noise sd": at a noise of 37 counts the sd of the block means is
            13-55 counts (10th-90th percentile) at off-tissue candidate blocks
            and 40-520 counts in tissue (median 100-290); 95 % of the candidate
            pixels of off-tissue windows pass, 98-100 % of those of tissue
            windows do not. Must be > 0.
        deadband: One-sided soft threshold, in units of the noise floor: only the
            part of the excess over the interpolation that exceeds
            ``deadband x`` the pixel noise is removed. 2 keeps the fill from
            clipping pixel noise (a hard one-sided clip of Gaussian noise removes
            0.8 sd per changed pixel; with 2 sd of deadband 0.4 sd and on 2.3 % of
            the pixels). 0 = hard clip.
        max_cover: Premise check. Inpainting needs surroundings to interpolate
            from. A candidate pixel is not filled when more than this fraction of
            the 264 px box around it is candidate (dense, bright bead coating: the
            whole plane is candidate and the fill would only subtract a bounded
            share of the bead pedestal). Isolated beads measure 0.2-0.35 (95th
            percentile 0.5-0.6), a dense bright coating 0.85-1.0. 1 = off.
        fine_sigmas: Gaussian sigmas (px) of the fine fill levels (full
            resolution).
        coarse_sigmas: Gaussian sigmas (in 8 px blocks) of the coarse fill
            levels; 4 blocks = 32 px.
        tau: Known-weight fraction at which a level is trusted completely.
        smooth: Gaussian sigma (px) of the final smoothing inside the mask
            (0 = off).
        mode: ``"flat"`` (default) acts on flat background only, as described
            above. ``"hybrid"`` is the original behaviour: it fills where
            ``eps x light > gate x max(texture sd, noise floor)`` in tissue as
            well, and may dim tissue puncta close to a bead. In hybrid mode
            ``flat_max``, ``deadband``, ``peak_snr``, ``bright_reach_px`` and
            ``max_cover`` are ignored; the effective values are no flatness gate,
            a hard one-sided clip (deadband 0), no bright-source restriction
            (peak_snr 0) and no dense-coating skip (max_cover 1.0). The fill
            cascade, the dilation and the bound ``removal <= eps x light`` are
            the same.
        texture_block: Hybrid only. Side (px) of the blocks of the texture map;
            the blocks are aligned to the absolute mosaic grid, so the result does
            not depend on how the mosaic is cut into windows.
        texture_hp_sigma: Hybrid only. Gaussian sigma (px) of the high-pass
            (``plane - gaussian(plane)``, computed over valid pixels only) whose
            robust (MAD) sd per block is the texture sd.
        texture_smooth: Hybrid only. Gaussian sigma (in blocks) with which the
            per-block sd is smoothed.
        texture_radius_blocks: Hybrid only. Radius (blocks) at which the
            smoothing kernel is truncated. The original smoothing (sigma 1 block,
            scipy's default truncation at 4 sigma) reaches 4 blocks = 128 px; with
            the 24 px high-pass and the 144 px coarse path the total support
            would be 328 px, over ``MAX_SUPPORT_PX``. Truncated at 1 block (kernel
            0.27, 0.45, 0.27) the total is 232 px, so this is the default. A
            larger radius is refused when the support no longer fits.

    Raises:
        ValueError: A field is out of range, ``stains`` is not
            ``(reference_stain,)``, or the support of the settings exceeds
            ``MAX_SUPPORT_PX``.
    """

    enabled: bool = False
    stains: tuple[str, ...] = ("PolyT",)
    reference_stain: str = "PolyT"
    eps: tuple[float, ...] = (0.3,) * NZ_MAX
    gate: float = 1.0
    floor_frac: float = 1.0
    dilation: int = 2
    peak_snr: float = 3.0
    bright_reach_px: int = 64
    planes: tuple[int, ...] | None = None
    flat_max: float = 1.0
    deadband: float = 2.0
    max_cover: float = 0.6
    fine_sigmas: tuple[float, ...] = (1.5, 3.0, 6.0)
    coarse_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    tau: float = 0.5
    smooth: float = 1.5
    mode: Literal["flat", "hybrid"] = "flat"
    texture_block: int = 32
    texture_hp_sigma: float = 6.0
    texture_smooth: float = 1.0
    texture_radius_blocks: int = 1

    def __post_init__(self) -> None:
        """Validate the settings (see the class docstring for the rules)."""
        if tuple(self.stains) != (self.reference_stain,):
            raise ValueError(
                "the fill runs on the reference stain only (its noise floor is "
                f"calibrated): stains must be ({self.reference_stain!r},); "
                f"got {tuple(self.stains)}"
            )
        if len(self.eps) != NZ_MAX or min(self.eps) < 0 or max(self.eps) > 1:
            raise ValueError(f"eps must have {NZ_MAX} values in [0, 1]; got {self.eps}")
        if (
            self.gate <= 0
            or self.floor_frac <= 0
            or self.dilation < 0
            or not 0 < self.tau <= 1
        ):
            raise ValueError(
                "gate and floor_frac must be > 0, dilation >= 0 and 0 < tau <= 1"
            )
        if self.flat_max <= 0 or self.deadband < 0 or self.smooth < 0:
            raise ValueError("flat_max must be > 0, deadband and smooth >= 0")
        if not self.fine_sigmas or not self.coarse_sigmas:
            raise ValueError("fine_sigmas and coarse_sigmas must not be empty")
        if not 1 <= self.bright_reach_px <= MAX_REACH_PX:
            raise ValueError(
                f"bright_reach_px must be in [1, {MAX_REACH_PX}]; "
                f"got {self.bright_reach_px}"
            )
        if self.mode not in ("flat", "hybrid"):
            raise ValueError(f"mode must be 'flat' or 'hybrid'; got {self.mode!r}")
        if (
            self.texture_block < 1
            or self.texture_hp_sigma <= 0
            or self.texture_smooth <= 0
            or self.texture_radius_blocks < 0
        ):
            raise ValueError(
                "texture_block must be >= 1, texture_hp_sigma and texture_smooth "
                "> 0 and texture_radius_blocks >= 0"
            )
        total = support_px(self)["total"]
        if total > MAX_SUPPORT_PX:
            # Chunk invariance needs support <= halo - 2; refuse the setting
            # instead of silently depending on the chunk grid.
            raise ValueError(
                f"the fill's support ({total} px) exceeds {MAX_SUPPORT_PX} px "
                "(halo 256 - 2)"
            )


def fill_params_sha256(params: FillParams) -> str:
    """Hash every fill parameter (recorded in the provenance when it is enabled).

    Args:
        params: The fill parameters.

    Returns:
        Hex SHA-256 of the sorted-key JSON dump of the dataclass. The fields that
        only hybrid mode reads (``_HYBRID_ONLY_FIELDS``) are left out in flat
        mode, so the hash of a flat configuration is the one it had before the
        hybrid mode existed.
    """
    fields = dataclasses.asdict(params)
    if params.mode == "flat":
        for name in _HYBRID_ONLY_FIELDS:
            del fields[name]
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _gauss_radius(sigma: float) -> int:
    """Radius (px) of a Gaussian filter with scipy's default truncation."""
    return int(4.0 * sigma + 0.5)


def _effective_gates(params: FillParams) -> tuple[float, float, float]:
    """Return the (peak_snr, max_cover, deadband) in force for the mode.

    Hybrid mode switches the bright-source restriction, the dense-coating skip and
    the noise deadband off whatever the fields say; flat mode uses the fields.
    """
    if params.mode == "hybrid":
        return 0.0, 1.0, 0.0
    return params.peak_snr, params.max_cover, params.deadband


def _effective_params(params: FillParams) -> FillParams:
    """Return ``params`` with the mode's effective gates (see ``_effective_gates``).

    Flat mode returns the object itself, so its processing is untouched.
    """
    if params.mode == "flat":
        return params
    peak_snr, max_cover, deadband = _effective_gates(params)
    return dataclasses.replace(
        params, peak_snr=peak_snr, max_cover=max_cover, deadband=deadband
    )


def _texture_reach_px(params: FillParams) -> int:
    """Reach (px) of the candidate set of hybrid mode through the texture map.

    A pixel's texture sd is the smoothed sd of the blocks within
    ``texture_radius_blocks`` blocks of its own block (at most ``texture_block - 1``
    px away from the pixel at the block edge), and each block sd reads the
    high-passed plane over the block, which reads the plane one high-pass radius
    further.
    """
    return (
        _gauss_radius(params.texture_hp_sigma)
        + (params.texture_radius_blocks + 1) * params.texture_block
    )


def support_px(params: FillParams) -> dict[str, int]:
    """Distance over which a pixel's result depends on other pixels, per stage.

    The total must stay below the halo.

    Args:
        params: The fill parameters.

    Returns:
        Dict with the support (px) of the fine path ("fine": mask dilation, fine
        fill levels, final smoothing), of the candidate-coverage test ("cover",
        added to "fine"), of the candidate set ("reach", added to both paths: the
        candidate set, which the coverage test, the flatness test and the coarse
        hole are computed from, depends on pixels that far away), of the coarse
        path ("coarse": coarse fill field and flatness test) and the overall
        maximum ("total"). "reach" is the bright-source reach in flat mode and the
        reach of the texture map in hybrid mode (high-pass radius plus the blocks
        of the sd smoothing, see ``texture_radius_blocks``). The flat defaults
        give 32, 136, 64, 144 and 232, the hybrid defaults 32, 0, 88, 144 and 232.
    """
    peak_snr, max_cover, _deadband = _effective_gates(params)
    smooth = _gauss_radius(params.smooth) if params.smooth else 0
    fine = _gauss_radius(max(params.fine_sigmas)) + params.dilation + smooth
    # The coverage test is window-level, outside the crops.
    cover = (COVER_BLOCKS // 2 + 1) * BLOCK if max_cover < 1.0 else 0
    coarse = (_gauss_radius(max(params.coarse_sigmas)) + 1 + 1) * BLOCK
    if params.mode == "hybrid":
        reach = _texture_reach_px(params)
    else:
        reach = params.bright_reach_px if peak_snr > 0 else 0
    return {
        "fine": fine,
        "cover": cover,
        "reach": reach,
        "coarse": coarse,
        "total": max(fine + cover, coarse) + reach,
    }


def _crop_margin(params: FillParams) -> int:
    """Margin (px, multiple of the block) added around a region's crop."""
    return int(-(-support_px(params)["fine"] // BLOCK) * BLOCK + BLOCK)


# ---------------------------------------------------------------------------
def _cascade(
    v: np.ndarray,
    w: np.ndarray,
    sigmas: Sequence[float],
    tau: float,
    f: np.ndarray | None,
    have: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised-convolution cascade from the largest to the smallest sigma.

    This is the pull-push idea with fixed supports. At every level the local
    weighted mean is blended over the coarser estimate with confidence
    ``min(1, known fraction / tau)``.

    Args:
        v: Known values (0 elsewhere).
        w: Known weight in [0, 1].
        sigmas: Gaussian sigmas, largest first.
        tau: Known-weight fraction at which a level is trusted completely.
        f: Coarser estimate to blend over (None = start from the first level).
        have: Where ``f`` has data (None together with ``f``).

    Returns:
        (estimate, have): ``have`` is False where no known pixel was within reach
        of any level.
    """
    for s in sigmas:
        gv = gaussian_filter(v, s, mode="constant")
        gw = gaussian_filter(w, s, mode="constant")
        ok = gw > 1e-3
        val = np.where(ok, gv / np.maximum(gw, 1e-6), 0.0).astype(np.float32)
        if f is None or have is None:
            f, have = val, ok
        else:
            conf = np.where(ok, np.minimum(gw / tau, 1.0), 0.0).astype(np.float32)
            f = np.where(have, conf * val + (1.0 - conf) * f, val).astype(np.float32)
            have = have | ok
    if f is None or have is None:
        raise ValueError("the cascade needs at least one sigma")
    return f, have


def _block_sums(a: np.ndarray, hb: int, wb: int) -> np.ndarray:
    """Sum an array over BLOCK x BLOCK blocks (shape [hb * BLOCK, wb * BLOCK])."""
    return np.asarray(a.reshape(hb, BLOCK, wb, BLOCK).sum(axis=(1, 3)))


def _coarse_fields(
    c: np.ndarray,
    valid: np.ndarray,
    cand: np.ndarray,
    floor: float,
    p: FillParams,
    with_flatness: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-plane coarse fields on the 8 px block grid, from pixels off the hole.

    The hole is defined from the model light alone, so both fields depend on
    nothing beyond a few blocks around the pixel.

    Args:
        c: Plane after the subtraction.
        valid: Pixels that are valid (> 0 in the raw plane).
        cand: Candidate pixels.
        floor: Noise floor (counts).
        p: Fill parameters.
        with_flatness: Compute the flatness test (flat mode). Without it every
            block with data counts as flat (hybrid mode has no flatness gate).

    Returns:
        (fill field, have, flat): the coarse interpolation of the known pixels,
        where it has data, and the blocks where the known pixels around are flat,
        i.e. the sd of their 8 px block means (same cascade as the fill field,
        computed in float64) is at most ``flat_max x floor``. Blocks without
        enough known surroundings are not flat.
    """
    h, w = c.shape
    hb, wb = h // BLOCK, w // BLOCK
    cand_b = cand.reshape(hb, BLOCK, wb, BLOCK).any(axis=(1, 3))
    iterations = max(1, int(np.ceil((p.dilation + 2) / BLOCK)))
    hole_b = ndi.binary_dilation(cand_b, structure=_SQUARE, iterations=iterations)
    area = float(BLOCK * BLOCK)
    kn64 = np.where(hole_b, 0.0, _block_sums(valid, hb, wb) / area)
    kn = kn64.astype(np.float32)
    kv = np.where(
        hole_b, 0.0, _block_sums(np.where(valid, c, 0.0), hb, wb) / area
    ).astype(np.float32)
    sigmas = sorted(p.coarse_sigmas, reverse=True)
    fc, have = _cascade(kv, kn, sigmas, p.tau, None, None)
    if not with_flatness:
        return fc, have, np.ones(have.shape, bool)
    # Flatness: weighted mean and mean square of the known block means (float64:
    # the means are ~1e3-1e4, the sd ~1e1).
    sm = np.where(
        hole_b,
        0.0,
        _block_sums(np.where(valid, c, 0.0).astype(np.float64), hb, wb) / area,
    )
    ok_b = kn64 >= MIN_KNOWN_BLOCK
    mu = np.where(ok_b, sm / np.maximum(kn64, 1e-6), 0.0)
    wgt = np.where(ok_b, kn64, 0.0)
    m1: np.ndarray | None = None
    m2: np.ndarray | None = None
    have_v: np.ndarray | None = None
    for s in sigmas:
        gw = gaussian_filter(wgt, s, mode="constant")
        ok = gw > 1e-3
        g1 = np.where(
            ok,
            gaussian_filter(wgt * mu, s, mode="constant") / np.maximum(gw, 1e-6),
            0.0,
        )
        g2 = np.where(
            ok,
            gaussian_filter(wgt * mu * mu, s, mode="constant") / np.maximum(gw, 1e-6),
            0.0,
        )
        if m1 is None or m2 is None or have_v is None:
            m1, m2, have_v = g1, g2, ok
        else:
            conf = np.where(ok, np.minimum(gw / p.tau, 1.0), 0.0)
            m1 = np.where(have_v, conf * g1 + (1.0 - conf) * m1, g1)
            m2 = np.where(have_v, conf * g2 + (1.0 - conf) * m2, g2)
            have_v = have_v | ok
    if m1 is None or m2 is None or have_v is None:
        raise ValueError("coarse_sigmas must not be empty")
    flat = have_v & (np.maximum(m2 - m1 * m1, 0.0) <= (p.flat_max * floor) ** 2)
    return fc, have, flat


def _upsample_coarse(
    fc: np.ndarray, have: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (block-centred) upsample of the coarse field over a crop.

    Args:
        fc: Coarse fill field on the block grid.
        have: Where the coarse field has data.
        y0: Crop start row (px, multiple of BLOCK).
        y1: Crop end row (px, multiple of BLOCK).
        x0: Crop start column (px, multiple of BLOCK).
        x1: Crop end column (px, multiple of BLOCK).

    Returns:
        (field, ok) over the crop; ``ok`` is False where the interpolation
        touches a block without data.
    """
    by0, by1, bx0, bx1 = y0 // BLOCK, y1 // BLOCK, x0 // BLOCK, x1 // BLOCK
    ya, yb = max(by0 - 1, 0), min(by1 + 1, fc.shape[0])
    xa, xb = max(bx0 - 1, 0), min(bx1 + 1, fc.shape[1])
    f = np.where(have[ya:yb, xa:xb], fc[ya:yb, xa:xb], 0.0).astype(np.float32)
    hv = have[ya:yb, xa:xb].astype(np.float32)
    fu = ndi.zoom(f, BLOCK, order=1, mode="nearest", grid_mode=True)
    hu = ndi.zoom(hv, BLOCK, order=1, mode="nearest", grid_mode=True)
    oy, ox = (by0 - ya) * BLOCK, (bx0 - xa) * BLOCK
    fu = fu[oy : oy + (y1 - y0), ox : ox + (x1 - x0)]
    hu = hu[oy : oy + (y1 - y0), ox : ox + (x1 - x0)]
    # A pixel whose interpolation touches a block without data counts as without
    # data.
    okc = hu > 0.999
    return np.where(okc, fu / np.maximum(hu, 1e-6), 0.0).astype(np.float32), okc


def _block_mad_sd(blocks: np.ndarray) -> np.ndarray:
    """Robust (MAD) sd of each row of ``blocks``, ignoring NaN.

    Rows without a NaN take the plain-median path (partition based, much faster
    than ``nanmedian``); every row has at least one finite value.

    Args:
        blocks: Array [n, pixels_per_block] of high-passed values (NaN = invalid).

    Returns:
        float32 vector [n] of ``1.4826 x MAD``.
    """
    sd = np.empty(len(blocks), np.float32)
    complete = np.isfinite(blocks).all(axis=1)
    for rows, median in ((complete, np.median), (~complete, np.nanmedian)):
        if not rows.any():
            continue
        sub = blocks[rows]
        centre = median(sub, axis=1)
        sd[rows] = 1.4826 * median(np.abs(sub - centre[:, None]), axis=1)
    return sd


def _texture_sd_map(
    c: np.ndarray,
    valid: np.ndarray,
    wanted: np.ndarray,
    origin_yx: tuple[int, int],
    p: FillParams,
) -> np.ndarray:
    """Local texture sd of a plane, on blocks aligned to the absolute mosaic grid.

    Robust (MAD) sd of the high-passed plane per ``texture_block`` px block,
    smoothed over the neighbouring blocks, upsampled to pixels. It is the sd map
    of the original hybrid method with two changes that make it chunk invariant:
    the blocks sit on the absolute grid (``origin_yx`` is the absolute coordinate
    of pixel (0, 0)), and invalid pixels take no part (the high-pass and the
    smoothing are normalised convolutions over valid pixels, a block needs at least
    ``MIN_KNOWN_BLOCK`` valid pixels), so the zeros outside the mosaic do not
    create texture at the mosaic edge. Only the blocks that the smoothing of a
    ``wanted`` pixel reads are computed.

    Args:
        c: Plane after the subtraction.
        valid: Valid pixels.
        wanted: Pixels whose sd is needed (the other pixels get sd 0).
        origin_yx: Absolute mosaic coordinate of pixel (0, 0).
        p: Fill parameters (texture fields).

    Returns:
        float32 array of the plane's shape: texture sd in counts, 0 where there is
        no valid block within reach.
    """
    h, w = c.shape
    size = p.texture_block
    radius = p.texture_radius_blocks
    weight = valid.astype(np.float32)
    low_pass = gaussian_filter(
        np.where(valid, c, 0.0).astype(np.float32), p.texture_hp_sigma, mode="constant"
    ) / np.maximum(gaussian_filter(weight, p.texture_hp_sigma, mode="constant"), 1e-6)
    # Pad to whole blocks of the absolute grid; NaN marks "no pixel".
    oy, ox = origin_yx[0] % size, origin_yx[1] % size
    hb, wb = -(-(h + oy) // size), -(-(w + ox) // size)
    high_pass = np.full((hb * size, wb * size), np.nan, np.float32)
    high_pass[oy : oy + h, ox : ox + w] = np.where(valid, c - low_pass, np.nan)
    wanted_px = np.zeros((hb * size, wb * size), bool)
    wanted_px[oy : oy + h, ox : ox + w] = wanted
    wanted_b = np.asarray(wanted_px.reshape(hb, size, wb, size).any(axis=(1, 3)))
    need = (
        ndi.binary_dilation(wanted_b, structure=_SQUARE, iterations=radius)
        if radius
        else wanted_b
    )
    rows, cols = np.nonzero(need)
    blocks = (
        high_pass.reshape(hb, size, wb, size)
        .transpose(0, 2, 1, 3)[rows, cols]
        .reshape(len(rows), -1)
    )
    ok = np.isfinite(blocks).sum(axis=1) >= MIN_KNOWN_BLOCK * size * size
    block_sd = np.zeros((hb, wb), np.float32)
    block_ok = np.zeros((hb, wb), np.float32)
    if ok.any():
        block_sd[rows[ok], cols[ok]] = _block_mad_sd(blocks[ok])
        block_ok[rows[ok], cols[ok]] = 1.0
    if radius:
        # Truncated Gaussian over the neighbouring blocks, normalised by the blocks
        # that have an sd (the original used mode="nearest" at the array edge).
        truncate = (radius + 0.25) / p.texture_smooth
        num = gaussian_filter(
            block_sd, p.texture_smooth, mode="constant", truncate=truncate
        )
        den = gaussian_filter(
            block_ok, p.texture_smooth, mode="constant", truncate=truncate
        )
        block_sd = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0).astype(
            np.float32
        )
    full = np.repeat(np.repeat(block_sd, size, axis=0), size, axis=1)
    return np.ascontiguousarray(full[oy : oy + h, ox : ox + w])


def _fill_crop(
    c: np.ndarray,
    u: np.ndarray,
    valid: np.ndarray,
    cand: np.ndarray,
    f0: np.ndarray,
    have0: np.ndarray,
    floor: float,
    p: FillParams,
    ctr: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fill one crop.

    Args:
        c: Crop of the plane after the subtraction.
        u: Crop of ``eps x model light``.
        valid: Valid pixels of the crop.
        cand: Candidate pixels of the crop.
        f0: Coarse interpolation over the crop (start of the fine cascade).
        have0: Where ``f0`` has data.
        floor: Noise floor (counts).
        p: Fill parameters.
        ctr: Counters; ``unfilled_px`` is added to.

    Returns:
        (new values, mask of the pixels that may be changed).
    """
    m0 = (u > p.gate * floor) & cand
    if not m0.any():
        return c, np.zeros(c.shape, bool)
    m = (
        ndi.binary_dilation(m0, structure=_CROSS, iterations=p.dilation)
        if p.dilation
        else m0
    )
    m &= valid
    if not m.any():
        return c, m
    known = valid & ~m
    f, have = _cascade(
        np.where(known, c, 0.0).astype(np.float32),
        known.astype(np.float32),
        sorted(p.fine_sigmas, reverse=True),
        p.tau,
        f0,
        have0,
    )
    if p.smooth:
        fm = gaussian_filter(
            np.where(m, f, 0.0).astype(np.float32), p.smooth, mode="constant"
        )
        wm = gaussian_filter(m.astype(np.float32), p.smooth, mode="constant")
        f = np.where(m & (wm > 0.5) & have, fm / np.maximum(wm, 1e-3), f)
    udil = ndi.grey_dilation(u, size=2 * p.dilation + 1) if p.dilation else u
    # One-sided, bounded by the (dilated) model uncertainty, and only the part
    # beyond the noise deadband.
    rem = np.minimum(np.maximum(c - f - p.deadband * floor, 0.0), udil)
    ok = m & have
    ctr["unfilled_px"] = ctr.get("unfilled_px", 0.0) + float((m & ~have).sum())
    return np.where(ok, c - rem, c).astype(np.float32), ok


def _fill_plane(
    c: np.ndarray,
    light: np.ndarray,
    raw: np.ndarray,
    eps: float,
    noise: float,
    p: FillParams,
    ctr: dict[str, float],
    origin_yx: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Fill one plane (see the module docstring).

    Args:
        c: Plane after the subtraction.
        light: Model light of the plane.
        raw: Input plane (0 = invalid pixel).
        eps: Share of the model light the fill may remove at most.
        noise: Dataset-global pixel noise of the plane.
        p: Fill parameters, with the mode's effective gates (``_effective_params``).
        ctr: Counters of the plane (filled in place).
        origin_yx: Absolute mosaic coordinate of pixel (0, 0); used by the texture
            map of hybrid mode.

    Returns:
        The new plane, same shape as ``c``.
    """
    h0, w0 = c.shape
    ph, pw = (-h0) % BLOCK, (-w0) % BLOCK
    if ph or pw:  # zeros = invalid pixels
        c, light, raw = (np.pad(a, ((0, ph), (0, pw))) for a in (c, light, raw))
    h, w = c.shape
    valid = raw > 0
    u = (eps * light).astype(np.float32)
    floor = max(p.floor_frac * noise, 1.0)
    cand = (u > p.gate * floor) & valid
    if p.mode == "hybrid" and cand.any():
        # Texture gate: the model's uncertainty must exceed the local texture sd of
        # the corrected plane, not only the global noise floor.
        sd = _texture_sd_map(c, valid, cand, origin_yx, p)
        textured = cand & (u > p.gate * np.maximum(sd, floor))
        ctr["texture_dropped_px"] = ctr.get("texture_dropped_px", 0.0) + float(
            (cand & ~textured).sum()
        )
        cand = textured
    if p.peak_snr > 0 and cand.any():
        # Bounded bright-source restriction: keep a candidate pixel only if a
        # bright candidate pixel lies within `bright_reach_px` (square window). A
        # window-level connected-component test would let a faint candidate
        # bridge pull in sources that lie outside the window, so the result would
        # depend on the chunk cut.
        bright = cand & (u >= p.peak_snr * noise)
        size = 2 * p.bright_reach_px + 1
        near = ndi.maximum_filter(
            bright.astype(np.uint8), size=size, mode="constant", cval=0
        ).astype(bool)
        ctr["reach_dropped_px"] = ctr.get("reach_dropped_px", 0.0) + float(
            (cand & ~near).sum()
        )
        cand = cand & near
    if not cand.any():
        return c[:h0, :w0]
    ctr["cand_px"] = ctr.get("cand_px", 0.0) + float(cand.sum())
    # The hole of the coarse fields is every candidate, also the ones skipped
    # below.
    fc, havec, flat_b = _coarse_fields(
        c, valid, cand, floor, p, with_flatness=p.mode == "flat"
    )
    keep = cand
    if p.max_cover < 1.0:
        frac_b = cand.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(
            axis=(1, 3), dtype=np.float32
        )
        cover = ndi.uniform_filter(frac_b, COVER_BLOCKS, mode="constant") / (
            ndi.uniform_filter(np.ones_like(frac_b), COVER_BLOCKS, mode="constant")
        )
        dense = np.repeat(np.repeat(cover > p.max_cover, BLOCK, axis=0), BLOCK, axis=1)
        ctr["dense_skipped_px"] = ctr.get("dense_skipped_px", 0.0) + float(
            (keep & dense).sum()
        )
        keep = keep & ~dense
    if p.mode == "flat":
        flat = np.repeat(np.repeat(flat_b, BLOCK, axis=0), BLOCK, axis=1)
        ctr["not_flat_px"] = ctr.get("not_flat_px", 0.0) + float((keep & ~flat).sum())
        cand = keep & flat
    else:
        cand = keep  # hybrid has no flatness gate
    if not cand.any():
        return c[:h0, :w0]
    # Regions: candidate blocks merged over a few blocks, one crop per region.
    margin = _crop_margin(p)
    cand_b = np.asarray(
        cand.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).any(axis=(1, 3))
    )
    merged = ndi.binary_dilation(cand_b, structure=_SQUARE, iterations=MERGE_BLOCKS)
    lab_b = np.asarray(ndi.label(merged, structure=_SQUARE)[0])
    out = c.copy()
    for i, sl in enumerate(ndi.find_objects(lab_b), start=1):
        sub = np.asarray(cand_b[sl] & (lab_b[sl] == i))
        if not sub.any():
            continue
        ys, xs = np.nonzero(sub)
        # `own` = the pixels this crop may write: candidate blocks of the region
        # plus one block (covers the dilation).
        oy0 = (sl[0].start + int(ys.min()) - 1) * BLOCK
        oy1 = (sl[0].start + int(ys.max()) + 2) * BLOCK
        ox0 = (sl[1].start + int(xs.min()) - 1) * BLOCK
        ox1 = (sl[1].start + int(xs.max()) + 2) * BLOCK
        oy0, ox0, oy1, ox1 = max(oy0, 0), max(ox0, 0), min(oy1, h), min(ox1, w)
        y0, y1 = max(oy0 - margin, 0), min(oy1 + margin, h)
        x0, x1 = max(ox0 - margin, 0), min(ox1 + margin, w)
        f0, have0 = _upsample_coarse(fc, havec, y0, y1, x0, x1)
        nv, ok = _fill_crop(
            c[y0:y1, x0:x1],
            u[y0:y1, x0:x1],
            valid[y0:y1, x0:x1],
            cand[y0:y1, x0:x1],
            f0,
            have0,
            floor,
            p,
            ctr,
        )
        if not ok.any():
            continue
        sy, sx = slice(oy0 - y0, oy1 - y0), slice(ox0 - x0, ox1 - x0)
        own = ok[sy, sx]
        out[oy0:oy1, ox0:ox1][own] = nv[sy, sx][own]
    out = np.minimum(out, c)  # never brighter
    return np.asarray(out[:h0, :w0], dtype=np.float32)


def apply_bounded_fill(
    corrected: np.ndarray,
    model_light: np.ndarray,
    *,
    raw: np.ndarray,
    z_layers: Sequence[int],
    config: FillParams,
    stain: str,
    noise: Mapping[int, float] | None = None,
    stats: dict[str, Any] | None = None,
    origin_yx: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Remove residual bead light left by the subtraction.

    Bounded, one-sided inpainting: on flat background only in flat mode, wherever
    the model uncertainty exceeds the local texture in hybrid mode. See the module
    docstring for the guarantees.

    Args:
        corrected: Subtraction result [n_z, Y, X] float32 (planes in ``z_layers``
            order); the array must start on the absolute 16 px grid
            (``correct_window`` guarantees it).
        model_light: Light attributed to beads, ``rho x max(M, 0)``, same shape.
        raw: Input planes, same shape (0 = invalid pixel, left unchanged).
        z_layers: Plane index (z) of each slice along axis 0.
        config: Fill settings; ``config.enabled`` False returns ``corrected``
            unchanged (bit-identical).
        stain: Stain name (the fill only runs for ``config.stains``).
        noise: Dataset-global pixel noise per plane (index = z) of the reference
            stain, from the calibration. Required when the fill is enabled for
            this stain; never estimated from the window.
        stats: Optional dict that receives the footprint counters (totals and
            ``per_plane``): ``cand_px`` (candidate pixels after the bright-source
            restriction), ``reach_dropped_px`` (dropped by it),
            ``dense_skipped_px`` (premise guard), ``not_flat_px`` (surroundings
            not flat: tissue, texture), ``unfilled_px`` (no known pixel within
            reach), ``changed_px`` and ``removed_sum``. Hybrid mode adds
            ``texture_dropped_px`` (candidates below the local texture sd) and
            reports no ``reach_dropped_px``, ``dense_skipped_px`` or
            ``not_flat_px``.
        origin_yx: Absolute mosaic coordinate of pixel (0, 0). Hybrid mode aligns
            its texture blocks to the absolute grid with it, and requires a
            multiple of ``BLOCK`` (``correct_window`` passes a multiple of 16).
            Flat mode does not read it.

    Returns:
        float32 array [n_z, Y, X], never brighter than ``corrected``, never lower
        than ``corrected - eps x light`` (max filtered over the mask dilation),
        equal to ``corrected`` where ``raw == 0``.

    Raises:
        ValueError: ``noise`` is missing or lacks a processed plane while the fill
            is enabled for ``stain``; ``origin_yx`` is not a multiple of ``BLOCK``
            in hybrid mode.
    """
    if not config.enabled or stain not in config.stains:
        return corrected
    if config.mode == "hybrid" and (origin_yx[0] % BLOCK or origin_yx[1] % BLOCK):
        raise ValueError(
            f"origin_yx must be a multiple of {BLOCK} px in hybrid mode; "
            f"got {origin_yx}"
        )
    config = _effective_params(config)
    if noise is None:
        raise ValueError(
            "fill needs the dataset-global per-plane noise (calibration); it is "
            "never estimated per window"
        )
    out = np.array(corrected, dtype=np.float32, copy=True)
    tot: dict[str, float] = {}
    for i, k in enumerate(z_layers):
        k = int(k)
        eps = float(config.eps[min(k, len(config.eps) - 1)])
        if eps <= 0 or (config.planes is not None and k not in config.planes):
            continue
        if k not in noise:
            raise ValueError(
                f"fill needs the global noise of plane {k}; the calibration "
                f"provides {sorted(noise)}"
            )
        ctr: dict[str, float] = {}
        out[i] = _fill_plane(
            out[i],
            np.asarray(model_light[i], np.float32),
            np.asarray(raw[i], np.float32),
            eps,
            float(noise[k]),
            config,
            ctr,
            origin_yx,
        )
        ctr["changed_px"] = float((out[i] < corrected[i]).sum())
        ctr["removed_sum"] = float((corrected[i] - out[i]).sum())
        for key, val in ctr.items():
            tot[key] = tot.get(key, 0.0) + val
        if stats is not None:
            stats.setdefault("per_plane", {})[k] = ctr
        logger.debug(
            "[BEADS] fill %s z%d: %s",
            stain,
            k,
            {name: round(count, 1) for name, count in ctr.items()},
        )
    if stats is not None:
        stats.update(tot)
    result: np.ndarray = np.where(
        np.asarray(raw) > 0, np.minimum(out, corrected), corrected
    ).astype(np.float32)
    return result
