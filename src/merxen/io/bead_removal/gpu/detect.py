"""GPU port of ``merxen.io.bead_removal.detect.detect_sources`` (z0 bead detection and
classification).

All image-sized work (DoG, peak mask) and all per-candidate window statistics (centre
minus annulus, azimuthal ring profile, null control positions) run on the device,
batched over candidates.  The greedy non-maximum suppression and the tiny per-candidate
classification stay on the host (about 10^4 candidates).
"""

# Window dimensions and matrices keep the capital letters of the formulas they port
# from the CPU modules (H, W, B, M, K, A0 ...), so the two can be read side by side.
# ruff: noqa: N803, N806

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.spatial import cKDTree

from merxen.io.bead_removal.detect import (
    ClassifierThresholds,
    Sources,
    _sector_index,
    empty_sources,
    nms,
    ring_tscores,
)
from merxen.io.bead_removal.params import DEFAULT_PARAMS, PsfSubtractParams

from . import ops

logger = logging.getLogger(__name__)

_yy, _xx = np.mgrid[-20:21, -20:21]
_RR = np.hypot(_yy, _xx)
_ANN = (_RR >= 14) & (_RR <= 20)
_AY_NP, _AX_NP = _yy[_ANN], _xx[_ANN]
_C3 = np.mgrid[-1:2, -1:2]
_CHUNK_ELEMS = 24_000_000


def _consts(
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ay = torch.from_numpy(_AY_NP.astype(np.int64)).to(dev)
    ax = torch.from_numpy(_AX_NP.astype(np.int64)).to(dev)
    cy = torch.from_numpy(_C3[0].ravel().astype(np.int64)).to(dev)
    cx = torch.from_numpy(_C3[1].ravel().astype(np.int64)).to(dev)
    return ay, ax, cy, cx


def centre_minus_bg(
    img: torch.Tensor, py: torch.Tensor, px: torch.Tensor
) -> torch.Tensor:
    """3x3 centre mean minus the median of the annulus r 14-20 px at integer positions
    (edge-replicated), float32.

    Args:
        img: 2-D float32 plane on the device.
        py, px: int64 positions (any number; processed in chunks).

    Returns:
        float32 tensor [n] of counts above the local background.
    """
    ay, ax, cy, cx = _consts(img.device)
    Y, X = img.shape
    flat = img.reshape(-1)
    out = torch.empty(py.shape[0], dtype=torch.float32, device=img.device)
    step = max(1, _CHUNK_ELEMS // ay.numel())
    for s in range(0, py.shape[0], step):
        y = py[s : s + step, None]
        x = px[s : s + step, None]
        c = flat[(y + cy).clamp(0, Y - 1) * X + (x + cx).clamp(0, X - 1)].mean(1)
        a = flat[(y + ay).clamp(0, Y - 1) * X + (x + ax).clamp(0, X - 1)]
        out[s : s + step] = c - ops.median_lastdim(a)
    return out


def _null_sd(
    img: torch.Tensor,
    py: torch.Tensor,
    px: torch.Tensor,
    n: int = 24,
    img2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-source sd of the centre-minus-annulus statistic at ``n`` control positions
    60-140 px away (golden-angle)."""
    k = np.arange(n)
    ang = (k * 2.399963229728653) % (2 * np.pi)
    rad = 60.0 + 80.0 * ((k * 0.618034) % 1.0)
    dy = torch.from_numpy(np.round(rad * np.sin(ang)).astype(np.int64)).to(img.device)
    dx = torch.from_numpy(np.round(rad * np.cos(ang)).astype(np.int64)).to(img.device)
    Y, X = img.shape
    qy = (py[:, None] + dy[None, :]).clamp(0, Y - 1).reshape(-1)
    qx = (px[:, None] + dx[None, :]).clamp(0, X - 1).reshape(-1)
    v = centre_minus_bg(img, qy, qx)
    if img2 is not None:
        v = v - centre_minus_bg(img2, qy, qx)
    v = v.reshape(py.shape[0], n)
    med = ops.median_lastdim(v).unsqueeze(1)
    return 1.4826 * ops.median_lastdim((v - med).abs())


def ring_ratio(
    img: torch.Tensor,
    ys: torch.Tensor,
    xs: torch.Tensor,
    rin: float = 8.0,
    rout: float = 15.0,
    bg_in: float = 18.0,
    bg_out: float = 24.0,
) -> torch.Tensor:
    """Azimuthal-profile ring amplitude (max over r in [rin, rout]) minus the background
    annulus median, batched.

    Args:
        img: 2-D float32 plane.
        ys, xs: float64 sub-pixel centres.

    Returns:
        float64 tensor [n].
    """
    dev = img.device
    Y, X = img.shape
    R = int(np.ceil(bg_out)) + 1
    ar = torch.arange(-R, R + 1, device=dev, dtype=torch.int64)
    gy = ar[:, None].to(torch.float64)
    gx = ar[None, :].to(torch.float64)
    nb = int(rout - rin) + 1
    out = torch.empty(ys.shape[0], dtype=torch.float64, device=dev)
    flat = img.reshape(-1)
    step = max(1, _CHUNK_ELEMS // ((2 * R + 1) ** 2) // 2)
    for s in range(0, ys.shape[0], step):
        y = ys[s : s + step]
        x = xs[s : s + step]
        iy = torch.round(y).to(torch.int64)
        ix = torch.round(x).to(torch.int64)
        w = flat[
            (iy[:, None, None] + ar[None, :, None]).clamp(0, Y - 1) * X
            + (ix[:, None, None] + ar[None, None, :]).clamp(0, X - 1)
        ]
        r = torch.hypot(
            gy[None] - (y - iy)[:, None, None], gx[None] - (x - ix)[:, None, None]
        )
        wd = w.to(torch.float64).reshape(w.shape[0], -1)
        r = r.reshape(r.shape[0], -1)
        bg = ops.masked_median_lastdim(wd, (r >= bg_in) & (r <= bg_out))
        idx = torch.floor(r).to(torch.int64)
        prof = []
        for b in range(nb):
            m = idx == int(rin) + b
            cnt = m.sum(1).clamp(min=1)
            prof.append((wd * m).sum(1) / cnt)
        out[s : s + step] = torch.stack(prof, 1).max(1).values - bg
    return out


def _thin_close_peaks(pk: np.ndarray, spacing: int) -> np.ndarray:
    """Greedy thinning of peaks closer than ``spacing`` (Chebyshev), in priority order =
    row order.

    Same rule as the final ``ensure_spacing`` step of skimage's peak_local_max (a point
    rejects every neighbour at a distance < spacing unless it was rejected itself), but
    only the peaks that have such a neighbour are visited: after the (2 spacing + 1)^2
    maximum filter that happens on exact intensity plateaus only.
    """
    if pk.shape[0] < 2:
        return pk
    pairs = cKDTree(pk).query_pairs(
        float(spacing) - 0.5, p=np.inf, output_type="ndarray"
    )
    if pairs.shape[0] == 0:
        return pk
    adj: dict[int, list[int]] = {}
    for a, b in pairs:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    rejected: set[int] = set()
    for i in sorted(adj):
        if i not in rejected:
            rejected.update(adj[i])
    keep = np.ones(pk.shape[0], bool)
    keep[list(rejected)] = False
    return pk[keep]


def _ring_tscores_device(
    img: torch.Tensor,
    ys: np.ndarray,
    xs: np.ndarray,
    r_ring: float,
    nsec: int = 16,
) -> np.ndarray:
    """Return ``ring_tscores`` of the CPU module for a few centres of a device plane.

    This serves the rescue candidates (a handful per window). The window of every
    centre is cut on the device (the same integer rounding and bounds rule as the CPU
    function: a window that leaves the image gives NaN) and the CPU function runs on
    the host copy, so the t-scores are the CPU's.

    Args:
        img: 2-D plane on the device.
        ys: Centre rows (px).
        xs: Centre columns (px).
        r_ring: Ring radius (px).
        nsec: Number of angular sectors.

    Returns:
        float array of t-scores (NaN where the window leaves the image).
    """
    half = _sector_index(r_ring, nsec)[0]
    height, width = img.shape
    out = np.full(len(ys), np.nan)
    for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
        iy, ix = int(round(y)), int(round(x))
        if (
            iy - half < 0
            or ix - half < 0
            or iy + half + 1 > height
            or ix + half + 1 > width
        ):
            continue
        crop = img[iy - half : iy + half + 1, ix - half : ix + half + 1].cpu().numpy()
        centre = np.array([float(half)])
        out[i] = ring_tscores(crop, centre, centre, r_ring, nsec)[0]
    return out


def detect_sources(
    polyt0: torch.Tensor,
    polyt1: torch.Tensor,
    dapi0: torch.Tensor | None,
    dapi1: torch.Tensor | None,
    *,
    dog_noise: float,
    thresholds: ClassifierThresholds,
    params: PsfSubtractParams = DEFAULT_PARAMS,
    z_planes: dict[int, torch.Tensor] | None = None,
) -> Sources:
    """Detect and classify bead candidates on z0 of the reference stain (GPU).

    Args:
        polyt0, polyt1: Reference-stain planes 0 and 1 on the device (float32, 0 =
            invalid pixel).
        dapi0, dapi1: DAPI planes 0 and 1 (None = no spectral test).
        dog_noise: Dataset-global DoG noise (detection threshold = ``k_sigma`` x this).
        thresholds: Per-dataset classifier thresholds.
        params: Fixed parameters.
        z_planes: {z: device plane} of the reference stain at z >= 2; read only by
            the z-profile test of the rescue (``thresholds.rescue_zprof_zmax >= 2``).
            A plane that is not given is not tested, as on the CPU.

    Returns:
        ``merxen.io.bead_removal.detect.Sources`` (host numpy) with the same fields and
        conventions as the CPU function.
    """
    p = params
    dev = polyt0.device
    p0 = polyt0
    Y, X = p0.shape
    valid = p0 > 0
    sm = ops.gaussian_filter(p0, 1.0)
    d = sm - ops.gaussian_filter(p0, 3.0)
    thr = float(p.k_sigma * float(dog_noise))
    md = int(p.peak_min_distance)
    mx = F.max_pool2d(d[None, None], 2 * md + 1, 1, md)[0, 0]
    mask = (d == mx) & (d > thr)
    b = int(p.border)
    if b > 0:
        mask[:b] = False
        mask[-b:] = False
        mask[:, :b] = False
        mask[:, -b:] = False
    if p.edge_margin_px > 0 and not bool(valid.all()):
        # The margin is measured from invalid pixels only: the window border is not an
        # edge (replicate padding = scipy "nearest"), so a window with one invalid pixel
        # keeps the candidates that a fully valid window keeps.
        m = int(p.edge_margin_px)
        inv = F.pad(
            (~valid).to(torch.float32)[None, None], (m, m, m, m), mode="replicate"
        )
        # minimum filter of the valid mask
        inner = F.max_pool2d(inv, 2 * m + 1, 1, 0)[0, 0] == 0
        mask &= inner
    pk = mask.nonzero()
    if pk.shape[0] == 0:
        return empty_sources()
    inten = d[pk[:, 0], pk[:, 1]].cpu().numpy()
    pk_h = pk.cpu().numpy()[np.argsort(-inten, kind="stable")]
    pk_h = _thin_close_peaks(pk_h, md)
    py_t = torch.from_numpy(pk_h[:, 0].astype(np.int64)).to(dev)
    px_t = torch.from_numpy(pk_h[:, 1].astype(np.int64)).to(dev)
    A0_t = centre_minus_bg(p0, py_t, px_t)
    A0 = A0_t.cpu().numpy()
    py, px = pk_h[:, 0], pk_h[:, 1]
    keep = nms(py, px, A0, p.nms_px)
    py, px, A0 = py[keep], px[keep], A0[keep]
    py_t = torch.from_numpy(py.astype(np.int64)).to(dev)
    px_t = torch.from_numpy(px.astype(np.int64)).to(dev)
    # sub-pixel parabolic refinement on the 1 px smoothed image (vectorised; edge
    # rows/cols keep the integer position)
    sy, sx = py.astype(float).copy(), px.astype(float).copy()
    inside = (py > 0) & (py < Y - 1) & (px > 0) & (px < X - 1)
    ii = np.where(inside)[0]
    smf = sm
    yy, xx = py_t[ii], px_t[ii]
    a_, b_, c_ = (
        smf[yy - 1, xx].double().cpu().numpy(),
        smf[yy, xx].double().cpu().numpy(),
        smf[yy + 1, xx].double().cpu().numpy(),
    )
    den = a_ - 2 * b_ + c_
    neg = den < 0
    sy[ii[neg]] += np.clip(0.5 * (a_[neg] - c_[neg]) / den[neg], -0.7, 0.7)
    a_, b_, c_ = (
        smf[yy, xx - 1].double().cpu().numpy(),
        smf[yy, xx].double().cpu().numpy(),
        smf[yy, xx + 1].double().cpu().numpy(),
    )
    den = a_ - 2 * b_ + c_
    neg = den < 0
    sx[ii[neg]] += np.clip(0.5 * (a_[neg] - c_[neg]) / den[neg], -0.7, 0.7)
    r1 = centre_minus_bg(polyt1, py_t, px_t).cpu().numpy() / np.maximum(A0, 1.0)
    if dapi0 is not None and dapi1 is not None:
        c0 = centre_minus_bg(dapi0, py_t, px_t).cpu().numpy()
        c1 = centre_minus_bg(dapi1, py_t, px_t).cpu().numpy()
        rD0 = c0 / np.maximum(A0, 1.0)
        rD = (c0 - c1) / np.maximum(A0, 1.0)
    else:
        rD0 = rD = np.full(len(py), np.nan)
    A0s = np.maximum(A0, 1.0)
    rring = (
        ring_ratio(p0, torch.from_numpy(sy).to(dev), torch.from_numpy(sx).to(dev))
        .cpu()
        .numpy()
        / A0s
    )
    sd1 = _null_sd(polyt1, py_t, px_t).cpu().numpy()
    th = thresholds
    is_bead = (p.a0_min <= A0) & (r1 < np.maximum(th.r1_max, 3.5 * sd1 / A0s))
    if dapi0 is not None and dapi1 is not None:
        sdD0 = _null_sd(dapi0, py_t, px_t).cpu().numpy()
        sdD = _null_sd(dapi0, py_t, px_t, img2=dapi1).cpu().numpy()
        ok0 = np.abs(rD0 - th.rd0_center) <= np.maximum(th.rd_tol, 3.5 * sdD0 / A0s)
        okD = np.abs(rD - th.rd_center) <= np.maximum(th.rd_tol, 3.5 * sdD / A0s)
        is_bead &= ok0 | okD
    ring_ok = (rring >= th.ring_min) & (rring <= th.ring_max)
    sd1_rel = sd1 / A0s
    rescued = np.zeros(len(py), bool)
    ring_t1 = np.full(len(py), np.nan)
    if th.rescue_active:
        # The rescue path of the CPU function (z0 ring too low, flat z1 surroundings,
        # no secondary focus at z2..zmax, coherent raw z1 ring). Only a handful of
        # candidates reach the ring test.
        cand = np.where(
            is_bead
            & ~ring_ok
            & (rring < th.ring_min)
            & (th.rescue_a0_min <= A0)
            & (sd1_rel <= th.rescue_sd1_max)
        )[0]
        if len(cand) and th.rescue_zprof_zmax >= 2:
            for z in range(2, int(th.rescue_zprof_zmax) + 1):
                if not len(cand):
                    break
                plane_z = (z_planes or {}).get(z)
                if plane_z is None:
                    continue
                cy = torch.from_numpy(py[cand].astype(np.int64)).to(dev)
                cx = torch.from_numpy(px[cand].astype(np.int64)).to(dev)
                contrast = centre_minus_bg(plane_z, cy, cx).cpu().numpy() / A0s[cand]
                tol = np.maximum(
                    th.r1_max, 3.5 * _null_sd(plane_z, cy, cx).cpu().numpy() / A0s[cand]
                )
                cand = cand[contrast <= tol]
        if len(cand):
            t_cand = _ring_tscores_device(
                polyt1, sy[cand], sx[cand], th.rescue_ring_radius_px
            )
            ring_t1[cand] = t_cand
            rescued[cand[np.nan_to_num(t_cand, nan=-1e9) >= th.rescue_tz1_min]] = True
    is_bead = (is_bead & ring_ok) | rescued
    srcs = Sources(
        sy,
        sx,
        A0.astype(np.float64),
        A0.astype(np.float64).copy(),
        is_bead,
        rD,
        r1,
        np.zeros(len(py), bool),
    )
    srcs.rring = rring
    srcs.rescued = rescued
    srcs.sd1_rel = sd1_rel
    srcs.t1 = ring_t1
    logger.debug(
        "[BEADS-GPU] detect: %d candidates, %d beads (%d rescued)",
        srcs.n,
        int(is_bead.sum()),
        int(rescued.sum()),
    )
    return srcs
