"""GPU port of the hybrid mode of the bounded bead-removal fill.

The numpy implementation (:mod:`merxen.io.bead_removal.fill`, ``mode="hybrid"``) is
the specification; this module reproduces it per plane on the device, with the same
operations in the same order: the candidate mask (``eps x light`` above the noise
floor and, with ``texture_gate``, above the texture-sd gate: high-pass, robust
per-block sd on blocks aligned to the absolute mosaic grid, truncated block
smoothing), its cross dilation, the coarse (8 px block) and fine Gaussian
fill cascades with the ``tau`` trust weights and the final smoothing, the one-sided
bound ``removal <= eps x model light`` and the never-brighter clamp, and the same
counters.

One difference in organisation, none in result: the numpy version fills one crop per
region of candidates (to bound its cost on the host), the device version filters the
whole window plane. The crops carry a margin that is at least the support of the fill
(:func:`merxen.io.bead_removal.fill.support_px`), so a pixel's result depends on the
same neighbourhood in both and only float32 rounding differs. Every operation is
deterministic (convolutions with deterministic algorithms, sorts, reductions, no
scatter-adds), so two runs give bitwise identical planes.

Only the hybrid mode is ported. The flat mode runs on the CPU
(:func:`merxen.io.bead_removal.backend.resolve_device`).
"""

# Plane and block dimensions follow the names of the numpy module they port from.
# mypy: disable-error-code="no-any-return"
# (the torch tensor-method stubs return Any)

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F  # noqa: N812

from merxen.io.bead_removal.fill import (
    BLOCK,
    MIN_KNOWN_BLOCK,
    FillParams,
    effective_params,
    fill_dilation,
    fill_eps,
)

from . import ops

logger = logging.getLogger(__name__)

# Robust sd of a Gaussian from the median absolute deviation.
MAD_TO_SD = 1.4826


@functools.lru_cache(maxsize=64)
def _kernel(sigma: float, truncate: float) -> torch.Tensor:
    """Cached 1-D Gaussian kernel (scipy's radius ``int(truncate x sigma + 0.5)``)."""
    return ops.gauss_kernel(sigma, truncate)


def _gauss(x: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Gaussian filter with zero padding (scipy ``mode="constant"``) of a stack.

    Args:
        x: float32 tensor [n, Y, X]; the planes are filtered independently.
        sigma: Gaussian sigma (px).
        truncate: Kernel radius in sigmas (scipy's default is 4).

    Returns:
        float32 tensor [n, Y, X].
    """
    kernel = _kernel(float(sigma), float(truncate))
    radius = (kernel.numel() - 1) // 2
    y = x.unsqueeze(1)
    y = F.conv2d(y, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    y = F.conv2d(y, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return y.squeeze(1)


def _cascade(
    v: torch.Tensor,
    w: torch.Tensor,
    sigmas: Sequence[float],
    tau: float,
    f: torch.Tensor | None,
    have: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised-convolution cascade from the largest to the smallest sigma.

    Port of ``merxen.io.bead_removal.fill._cascade``.

    Args:
        v: Known values (0 elsewhere), [Y, X].
        w: Known weight in [0, 1], [Y, X].
        sigmas: Gaussian sigmas, largest first.
        tau: Known-weight fraction at which a level is trusted completely.
        f: Coarser estimate to blend over (None = start from the first level).
        have: Where ``f`` has data (None together with ``f``).

    Returns:
        (estimate, have).
    """
    vw = torch.stack([v, w])
    for sigma in sigmas:
        g = _gauss(vw, sigma)
        gv, gw = g[0], g[1]
        ok = gw > 1e-3
        val = torch.where(ok, gv / gw.clamp(min=1e-6), torch.zeros_like(gv))
        if f is None or have is None:
            f, have = val, ok
        else:
            conf = torch.where(ok, (gw / tau).clamp(max=1.0), torch.zeros_like(gw))
            f = torch.where(have, conf * val + (1.0 - conf) * f, val)
            have = have | ok
    if f is None or have is None:
        raise ValueError("the cascade needs at least one sigma")
    return f, have


def _any_blocks(mask: torch.Tensor, size: int) -> torch.Tensor:
    """True where any pixel of a ``size`` x ``size`` block of a bool plane is set."""
    h, w = mask.shape
    return mask.reshape(h // size, size, w // size, size).any(dim=3).any(dim=1)


def _dilate_square(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of a bool plane by a (2 radius + 1) square (zero border)."""
    if radius == 0:
        return mask
    k = 2 * radius + 1
    pooled = F.max_pool2d(
        mask.to(torch.float16)[None, None], kernel_size=k, stride=1, padding=radius
    )
    return pooled[0, 0] > 0


def _dilate_cross(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Binary dilation of a bool plane by the 4-neighbour cross (zero border)."""
    out = mask
    for _ in range(iterations):
        grown = out.clone()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _block_sums(a: torch.Tensor, hb: int, wb: int) -> torch.Tensor:
    """Sum a plane over BLOCK x BLOCK blocks, in float64: [hb, wb]."""
    return a.reshape(hb, BLOCK, wb, BLOCK).sum(dim=(1, 3), dtype=torch.float64)


def _coarse_fields(
    c: torch.Tensor,
    valid: torch.Tensor,
    cand: torch.Tensor,
    p: FillParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-plane coarse fill field on the 8 px block grid (hybrid: no flatness).

    Port of ``merxen.io.bead_removal.fill._coarse_fields`` without the flatness test.

    Args:
        c: Plane after the subtraction.
        valid: Valid pixels.
        cand: Candidate pixels.
        p: Fill parameters.

    Returns:
        (fill field, have): the coarse interpolation of the known pixels and where
        it has data.
    """
    h, w = c.shape
    hb, wb = h // BLOCK, w // BLOCK
    iterations = max(1, -(-(fill_dilation(p) + 2) // BLOCK))
    hole_b = _any_blocks(cand, BLOCK)
    for _ in range(iterations):
        hole_b = _dilate_square(hole_b, 1)
    area = float(BLOCK * BLOCK)
    zero = torch.zeros((), dtype=torch.float64, device=c.device)
    kn = torch.where(hole_b, zero, _block_sums(valid, hb, wb) / area).float()
    kv = torch.where(
        hole_b,
        zero,
        _block_sums(torch.where(valid, c, torch.zeros_like(c)), hb, wb) / area,
    ).float()
    sigmas = sorted(p.coarse_sigmas, reverse=True)
    return _cascade(kv, kn, sigmas, p.tau, None, None)


def _upsample_coarse(
    fc: torch.Tensor, have: torch.Tensor, h: int, w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (block-centred) upsample of the coarse field over the whole plane.

    Port of ``merxen.io.bead_removal.fill._upsample_coarse`` (scipy ``zoom`` with
    ``grid_mode=True``, ``order=1``, ``mode="nearest"``): a pixel whose
    interpolation touches a block without data counts as without data.

    Returns:
        (field, ok) of the plane's shape.
    """
    f = torch.where(have, fc, torch.zeros_like(fc))
    fu = ops.bilinear_blocks(f, h, w, BLOCK)
    hu = ops.bilinear_blocks(have.float(), h, w, BLOCK)
    ok = hu > 0.999
    return torch.where(ok, fu / hu.clamp(min=1e-6), torch.zeros_like(fu)), ok


def _texture_sd_map(
    c: torch.Tensor,
    valid: torch.Tensor,
    wanted: torch.Tensor,
    origin_yx: tuple[int, int],
    p: FillParams,
) -> torch.Tensor:
    """Local texture sd of a plane, on blocks aligned to the absolute mosaic grid.

    Port of ``merxen.io.bead_removal.fill._texture_sd_map``: robust (MAD) sd of the
    high-passed plane per ``texture_block`` px block (a block needs
    ``MIN_KNOWN_BLOCK`` valid pixels), smoothed over the neighbouring blocks with the
    truncated Gaussian and upsampled to pixels. Only the blocks that the smoothing of
    a ``wanted`` pixel reads are computed.

    Args:
        c: Plane after the subtraction.
        valid: Valid pixels.
        wanted: Pixels whose sd is needed (the other pixels get sd 0).
        origin_yx: Absolute mosaic coordinate of pixel (0, 0).
        p: Fill parameters (texture fields).

    Returns:
        float32 plane of the same shape: texture sd in counts, 0 where there is no
        valid block within reach.
    """
    h, w = c.shape
    size = p.texture_block
    radius = p.texture_radius_blocks
    dev = c.device
    zero = torch.zeros_like(c)
    low_num = _gauss(
        torch.stack([torch.where(valid, c, zero), valid.float()]), p.texture_hp_sigma
    )
    low_pass = low_num[0] / low_num[1].clamp(min=1e-6)
    del low_num
    # Pad to whole blocks of the absolute grid; NaN marks "no pixel".
    oy, ox = origin_yx[0] % size, origin_yx[1] % size
    hb, wb = -(-(h + oy) // size), -(-(w + ox) // size)
    high_pass = torch.full((hb * size, wb * size), float("nan"), device=dev)
    nan = torch.full_like(c, float("nan"))
    high_pass[oy : oy + h, ox : ox + w] = torch.where(valid, c - low_pass, nan)
    del low_pass, nan
    wanted_px = torch.zeros((hb * size, wb * size), dtype=torch.bool, device=dev)
    wanted_px[oy : oy + h, ox : ox + w] = wanted
    need = _dilate_square(_any_blocks(wanted_px, size), radius)
    del wanted_px
    rows, cols = torch.nonzero(need, as_tuple=True)
    blocks = high_pass.reshape(hb, size, wb, size).permute(0, 2, 1, 3)[rows, cols]
    blocks = blocks.reshape(len(rows), -1)
    del high_pass
    finite = torch.isfinite(blocks)
    ok = finite.sum(dim=1) >= MIN_KNOWN_BLOCK * size * size
    block_sd = torch.zeros((hb, wb), device=dev)
    block_ok = torch.zeros((hb, wb), device=dev)
    if bool(ok.any()):
        sub, known = blocks[ok], finite[ok]
        centre = ops.masked_median_lastdim(sub, known)
        deviation = torch.where(
            known, (sub - centre[:, None]).abs(), torch.zeros_like(sub)
        )
        sd = MAD_TO_SD * ops.masked_median_lastdim(deviation, known)
        block_sd[rows[ok], cols[ok]] = sd
        block_ok[rows[ok], cols[ok]] = 1.0
    if radius:
        # Truncated Gaussian over the neighbouring blocks, normalised by the blocks
        # that have an sd.
        truncate = (radius + 0.25) / p.texture_smooth
        smoothed = _gauss(torch.stack([block_sd, block_ok]), p.texture_smooth, truncate)
        num, den = smoothed[0], smoothed[1]
        block_sd = torch.where(
            den > 1e-3, num / den.clamp(min=1e-6), torch.zeros_like(num)
        )
    full = block_sd.repeat_interleave(size, dim=0).repeat_interleave(size, dim=1)
    return full[oy : oy + h, ox : ox + w].contiguous()


def fill_plane(
    corrected: torch.Tensor,
    light: torch.Tensor,
    raw: torch.Tensor,
    *,
    plane: int,
    params: FillParams,
    noise: Mapping[int, float],
    origin_yx: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, float] | None]:
    """Fill one plane in hybrid mode (device version of the numpy fill).

    Port of ``merxen.io.bead_removal.fill._fill_plane`` for ``mode="hybrid"`` plus the
    plane selection and the final clamp of ``apply_bounded_fill``. The result is never
    brighter than ``corrected``, never lower than ``corrected - eps x light`` (max
    filtered over the mask dilation) and equal to ``corrected`` where ``raw == 0``.

    Args:
        corrected: Plane after the subtraction, float32 [Y, X] on the device.
        light: Light attributed to beads, ``rho x max(M, 0)``, same shape.
        raw: Input plane, same shape (0 = invalid pixel).
        plane: Plane index (z); selects ``eps`` and the planes the fill acts on.
        params: Fill parameters (hybrid mode, enabled).
        noise: Dataset-global pixel noise per plane of the reference stain.
        origin_yx: Absolute mosaic coordinate of pixel (0, 0); a multiple of BLOCK
            (``correct_window`` passes a multiple of 16).

    Returns:
        (plane, counters): the new plane (``corrected`` itself when the plane is not
        processed or nothing changes) and the plane's counters (``cand_px``,
        ``texture_dropped_px`` (with ``texture_gate`` only), ``unfilled_px``,
        ``changed_px``, ``removed_sum``, as the numpy fill reports them), or
        ``None`` when the plane is skipped (``eps`` 0 or ``plane`` not in
        ``params.planes``).

    Raises:
        ValueError: The mode is not hybrid, ``origin_yx`` is not a multiple of BLOCK,
            or the noise of ``plane`` is missing.
    """
    if params.mode != "hybrid":
        raise ValueError(
            f"the GPU fill implements the hybrid mode only; got {params.mode!r}"
        )
    if origin_yx[0] % BLOCK or origin_yx[1] % BLOCK:
        raise ValueError(
            f"origin_yx must be a multiple of {BLOCK} px in hybrid mode; "
            f"got {origin_yx}"
        )
    plane_eps = fill_eps(params)
    eps = float(plane_eps[min(plane, len(plane_eps) - 1)])
    if eps <= 0 or (params.planes is not None and plane not in params.planes):
        return corrected, None
    if plane not in noise:
        raise ValueError(
            f"fill needs the global noise of plane {plane}; the calibration "
            f"provides {sorted(noise)}"
        )
    p = effective_params(params)
    ctr: dict[str, float] = {}
    h0, w0 = corrected.shape
    ph, pw = (-h0) % BLOCK, (-w0) % BLOCK
    c, lt, rw = corrected, light, raw
    if ph or pw:  # zeros = invalid pixels
        c, lt, rw = (F.pad(a, (0, pw, 0, ph)) for a in (c, lt, rw))
    h, w = c.shape
    valid = rw > 0
    u = eps * lt
    floor = max(p.floor_frac * float(noise[plane]), 1.0)
    cand = (u > p.gate * floor) & valid
    if p.texture_gate and bool(cand.any()):
        # Texture gate (opt-in): the model's uncertainty must exceed the local
        # texture sd of the corrected plane, not only the global noise floor.
        sd = _texture_sd_map(c, valid, cand, origin_yx, p)
        textured = cand & (u > p.gate * sd.clamp(min=floor))
        ctr["texture_dropped_px"] = float((cand & ~textured).sum())
        cand = textured
        del sd, textured
    result = c
    if bool(cand.any()):
        ctr["cand_px"] = float(cand.sum())
        fc, have_c = _coarse_fields(c, valid, cand, p)
        f0, have0 = _upsample_coarse(fc, have_c, h, w)
        del fc, have_c
        m0 = (u > p.gate * floor) & cand
        dilation = fill_dilation(p)
        m = _dilate_cross(m0, dilation) & valid
        del m0
        if bool(m.any()):
            known = valid & ~m
            f, have = _cascade(
                torch.where(known, c, torch.zeros_like(c)),
                known.float(),
                sorted(p.fine_sigmas, reverse=True),
                p.tau,
                f0,
                have0,
            )
            del known, f0, have0
            if p.smooth:
                smoothed = _gauss(
                    torch.stack([torch.where(m, f, torch.zeros_like(f)), m.float()]),
                    p.smooth,
                )
                fm, wm = smoothed[0], smoothed[1]
                f = torch.where(m & (wm > 0.5) & have, fm / wm.clamp(min=1e-3), f)
                del smoothed, fm, wm
            udil = (
                F.max_pool2d(
                    u[None, None],
                    kernel_size=2 * dilation + 1,
                    stride=1,
                    padding=dilation,
                )[0, 0]
                if dilation
                else u
            )
            # One-sided, bounded by the (dilated) model uncertainty (hybrid: no
            # noise deadband).
            removal = torch.minimum((c - f - p.deadband * floor).clamp(min=0.0), udil)
            usable = m & have
            ctr["unfilled_px"] = float((m & ~have).sum())
            result = torch.where(usable, c - removal, c)
            del f, have, udil, removal, usable
    # Never brighter than the subtraction result; invalid pixels stay as they are.
    result = torch.where(valid, torch.minimum(result, c), c)[:h0, :w0]
    ctr["changed_px"] = float((result < corrected).sum())
    ctr["removed_sum"] = float((corrected - result).sum(dtype=torch.float64))
    logger.debug(
        "[BEADS] gpu fill z%d: %s",
        plane,
        {name: round(count, 1) for name, count in ctr.items()},
    )
    return result, ctr
