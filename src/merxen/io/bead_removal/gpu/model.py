"""GPU port of ``merxen.io.bead_removal.model.fit_window_library``: blobs, stretch field
and the plane-by-plane library fit."""

# Window dimensions and matrices keep the capital letters of the formulas they port
# from the CPU modules (H, W, B, M, K, A0 ...), so the two can be read side by side.
# ruff: noqa: N803, N806
# mypy: disable-error-code="no-any-return"
# (the torch.fft and tensor-method stubs return Any)

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike

from merxen.io.bead_removal.detect import Sources, window_bounds
from merxen.io.bead_removal.model import (
    _GRID,
    BeadModel,
    WindowCalibration,
    _consensus_gamma,
    _prune_from_low_planes,
    smooth_field,
    top_per_block,
)
from merxen.io.bead_removal.params import BETA, DEFAULT_PARAMS, PsfSubtractParams
from merxen.io.bead_removal.psf import Template

from . import ops
from .detect import detect_sources
from .fit import FitOpts, Frame, PlaneFitter, PlaneResult, fit_plane, grid_colours
from .stamps import Geom, TmplDev, kern2d, make_geom, point_stamps, radius_grid

logger = logging.getLogger(__name__)


def _in_image(geom: Geom, Y: int, X: int) -> torch.Tensor:
    """float32 [B, Hb, Wb] 1 where the window pixel lies inside the image."""
    ry = (geom.yy >= 0) & (geom.yy < Y)
    cx = (geom.xx >= 0) & (geom.xx < X)
    return (ry[:, :, None] & cx[:, None, :]).to(torch.float32)


def _frame_with_model(
    img: torch.Tensor, B: torch.Tensor, M: torch.Tensor, pad: int
) -> Frame:
    fr = Frame(img, B, pad)
    Y, X = img.shape
    fr.M.view(Y + 2 * pad, fr.Wp)[pad : pad + Y, pad : pad + X] = M
    return fr


# ------------------------------------------------------------------------------------
# blobs
def _fft_same(
    x: torch.Tensor, Kf: torch.Tensor, kshape: tuple[int, int]
) -> torch.Tensor:
    """scipy fftconvolve(x, K, 'same') (odd kernel): central part of the full
    convolution; ``Kf`` = rfft2 of K."""
    H = x.shape[0] + kshape[0] - 1
    W = x.shape[1] + kshape[1] - 1
    full = torch.fft.irfft2(torch.fft.rfft2(x, s=(H, W)) * Kf, s=(H, W))
    oy, ox = (kshape[0] - 1) // 2, (kshape[1] - 1) // 2
    return full[oy : oy + x.shape[0], ox : ox + x.shape[1]]


def _rl(win: torch.Tensor, K: torch.Tensor, iters: int) -> torch.Tensor:
    """Richardson-Lucy deconvolution (float64) of the window ``win`` (>= 0) with the
    unit-sum kernel K."""
    H = win.shape[0] + K.shape[0] - 1
    W = win.shape[1] + K.shape[1] - 1
    Kfa = torch.fft.rfft2(K, s=(H, W))
    Kfb = torch.fft.rfft2(torch.flip(K, (0, 1)), s=(H, W))
    kshape = (K.shape[0], K.shape[1])
    u = torch.full(
        win.shape, max(float(win.mean()), 1.0), dtype=torch.float64, device=win.device
    )
    for _ in range(iters):
        est = _fft_same(u, Kfa, kshape) + 1e-3
        u = u * _fft_same(torch.clamp(win / est, 0, 5.0), Kfb, kshape)
    return u


def _point_clipped(
    tm: TmplDev,
    a: float,
    s: float,
    yc: float,
    xc: float,
    shape: tuple[int, int],
    p: PsfSubtractParams,
    R: float,
) -> tuple[int, int, int, int, torch.Tensor]:
    """CPU ``_stamp`` of a point source on the clipped window: (y0, y1, x0, x1, h) with
    h float32 [h, w]."""
    y0, y1, x0, x1 = window_bounds(shape, yc, xc, R)
    dev = ops.device()
    dy = (
        torch.arange(y0, y1, device=dev, dtype=torch.float32) - float(np.float32(yc))
    )[:, None]
    dx = (
        torch.arange(x0, x1, device=dev, dtype=torch.float32) - float(np.float32(xc))
    )[None, :]
    r = torch.sqrt(dy * dy + dx * dx)
    return y0, y1, x0, x1, (a * tm.eval(r, float(s))).to(torch.float32)


def _rl_batch(win: torch.Tensor, K: torch.Tensor, iters: int) -> torch.Tensor:
    """Batched Richardson-Lucy (float64): ``win`` [B, H, W] (>= 0), ``K`` [kh, kw]
    unit-sum kernel shared by all."""
    Bn, H, W = win.shape
    Hf, Wf = H + K.shape[0] - 1, W + K.shape[1] - 1
    Kfa = torch.fft.rfft2(K, s=(Hf, Wf))[None]
    Kfb = torch.fft.rfft2(torch.flip(K, (0, 1)), s=(Hf, Wf))[None]
    oy, ox = (K.shape[0] - 1) // 2, (K.shape[1] - 1) // 2

    def same(x: torch.Tensor, Kf: torch.Tensor) -> torch.Tensor:
        full = torch.fft.irfft2(torch.fft.rfft2(x, s=(Hf, Wf)) * Kf, s=(Hf, Wf))
        return full[:, oy : oy + H, ox : ox + W]

    mean = win.mean((1, 2)).clamp(min=1.0)
    u = mean[:, None, None].expand(Bn, H, W).clone()
    for _ in range(iters):
        est = same(u, Kfa) + 1e-3
        u = u * same(torch.clamp(win / est, 0, 5.0), Kfb)
    return u


def build_blobs(
    img0: torch.Tensor,
    B0: torch.Tensor,
    src: Sources,
    f0: PlaneResult,
    p: PsfSubtractParams,
    a0_median: float,
    opts: FitOpts,
) -> np.ndarray:
    """GPU ``merxen.io.bead_removal.model.build_blobs``: aperture flux ratios (batched),
    then Richardson-Lucy source maps.

    The RL maps are computed colour by colour (blobs whose windows and stamps cannot
    overlap are processed together, brightest colour first) instead of strictly one
    after the other; blobs whose 81 px window is clipped by the image border, and
    ``opts.mode == "seq"``, use the strictly sequential order of the CPU code.

    Args:
        img0, B0: z0 plane and envelope (device).
        src: Sources; updated in place (``u``, ``a``, ``blob``, ``q``).
        f0: z0 point-source fit (PlaneResult; ``model`` on the device).
        p: Parameters.
        a0_median: Dataset-global median bead amplitude.
        opts: Execution options.

    Returns:
        Indices of the blobs.
    """
    dev = img0.device
    Y, X = img0.shape
    t0 = f0.tmpl
    tm = TmplDev.from_template(t0)
    phi_ap = t0.flux(p.r_ap)
    N = src.n
    q = np.zeros(N)
    snrf = np.zeros(N)
    R = t0.r_sup * 1.0 + p.win_extra  # f0.s == 1 at z0
    Hb = int(np.ceil(2 * R)) + 3
    pad = Hb + 2
    fr = _frame_with_model(img0, B0, f0.model, pad)
    idx = np.where(src.is_bead)[0]
    per = max(1, opts.px_budget // (Hb * Hb))

    def f64(a: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, np.float64), device=dev)

    yc_all, xc_all, a_all, g_all, c_all = (
        f64(src.y),
        f64(src.x),
        f64(src.a),
        f64(f0.g),
        f64(f0.c),
    )
    s_all = f64(f0.s)
    for i0 in range(0, len(idx), per):
        ii = torch.as_tensor(idx[i0 : i0 + per], device=dev)
        yc, xc = yc_all[ii], xc_all[ii]
        geom = make_geom(yc, xc, torch.full_like(yc, R), Hb, Hb, pad, fr.Wp)
        h = point_stamps(tm, a_all[ii], s_all[ii], yc, xc, geom)
        base = fr.base[geom.flat]
        Mw = fr.M[geom.flat]
        own = g_all[ii].to(torch.float32)[:, None, None] * h
        res = base - (Mw - own) - c_all[ii].to(torch.float32)[:, None, None]
        r = radius_grid(geom, yc, xc)
        ap = ((r <= p.r_ap).to(torch.float32)) * geom.mask * _in_image(geom, Y, X)
        fl = (res * ap).sum((1, 2), dtype=torch.float64).cpu().numpy()
        nap = ap.sum((1, 2), dtype=torch.float64).cpu().numpy()
        sel = idx[i0 : i0 + per]
        q[sel] = fl / np.maximum(src.A0[sel] * phi_ap, 1e-6)
        snrf[sel] = fl / (f0.noise * np.sqrt(np.maximum(nap, 1)))
    src.q = q
    cand = (
        src.is_bead
        & (q > p.blob_q)
        & (snrf > p.blob_snr)
        & (p.blob_a_rel * a0_median <= src.A0)
    )
    if not cand.any():
        return np.where(src.blob)[0]
    K0, R0 = kern2d(tm, 1.0)
    Phi = float(K0.sum())
    Kn = (K0 / Phi).to(torch.float64)
    hb = p.blob_half
    order = [j for j in np.argsort(-src.A0, kind="stable") if cand[j]]
    # windows of the RL maps: interior (unclipped) blobs are batched by colour, the
    # others go one by one
    wb = {j: window_bounds((Y, X), src.y[j], src.x[j], hb) for j in order}
    interior = [
        j
        for j in order
        if wb[j][0] > 0 and wb[j][2] > 0 and wb[j][1] < Y and wb[j][3] < X
    ]
    steps: list = []
    if opts.mode == "seq":
        steps = [[j] for j in order]
    else:
        border = [j for j in order if j not in set(interior)]
        ia = np.asarray(interior, np.int64)
        if ia.size:
            # stamp removal / source map / window extents
            Dcell = float(2 * max(t0.r_sup + p.win_extra, hb + R0) + 4)
            oy, ox = opts.origin_yx if opts.anchor else (0, 0)
            groups = grid_colours(src.y[ia] + oy, src.x[ia] + ox, src.A0[ia], Dcell)
            steps = [list(ia[g]) for g in groups]
        steps += [[j] for j in border]
    for grp in steps:
        _blob_step(fr, tm, t0, K0, Kn, Phi, R0, grp, src, f0, p, Y, X, hb, wb, dev)
    return np.where(src.blob)[0]


def _blob_step(
    fr: Frame,
    tm: TmplDev,
    t0: Template,
    K0: torch.Tensor,
    Kn: torch.Tensor,
    Phi: float,
    R0: int,
    grp: list[int],
    src: Sources,
    f0: PlaneResult,
    p: PsfSubtractParams,
    Y: int,
    X: int,
    hb: int,
    wb: dict[int, tuple[int, int, int, int]],
    dev: torch.device,
) -> None:
    """RL source maps of a group of blobs with disjoint zones (one blob = the CPU
    sequential step)."""
    ids = np.asarray(grp, np.int64)

    def f64(a: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, np.float64), device=dev)

    yc, xc = f64(src.y[ids]), f64(src.x[ids])
    # 1. remove the point stamp of every blob of the group from the model
    Hs = int(np.ceil(2 * (t0.r_sup + p.win_extra))) + 3
    geom = make_geom(
        yc, xc, torch.full_like(yc, t0.r_sup + p.win_extra), Hs, Hs, fr.pad, fr.Wp
    )
    h = point_stamps(
        tm,
        f64(src.a[ids]),
        torch.ones(len(ids), dtype=torch.float64, device=dev),
        yc,
        xc,
        geom,
    )
    fr.M.index_add_(
        0,
        geom.flat.reshape(-1),
        (
            -(1.0 * h)
            * torch.as_tensor(f0.g[ids], device=dev, dtype=torch.float32)[:, None, None]
        ).reshape(-1),
    )
    # 2. windows (all of one shape; batches are split by shape)
    shapes: dict[tuple[int, int], list[int]] = {}
    for b, j in enumerate(ids):
        y0, y1, x0, x1 = wb[j]
        shapes.setdefault((y1 - y0, x1 - x0), []).append(b)
    for (hh, ww), bl in shapes.items():
        sel = np.asarray(bl)
        jj = ids[sel]
        y0t = torch.as_tensor([wb[j][0] for j in jj], device=dev)
        x0t = torch.as_tensor([wb[j][2] for j in jj], device=dev)
        ay = torch.arange(hh, device=dev)
        ax = torch.arange(ww, device=dev)
        yy = y0t[:, None] + ay[None, :]
        xx = x0t[:, None] + ax[None, :]
        flat = (yy + fr.pad)[:, :, None] * fr.Wp + (xx + fr.pad)[:, None, :]
        Iw = (
            fr.base[flat]
            - fr.M[flat]
            - torch.as_tensor(f0.c[jj], device=dev, dtype=torch.float32)[:, None, None]
        ).to(torch.float64)
        rr = torch.hypot(
            yy.to(torch.float64)[:, :, None] - f64(src.y[jj])[:, None, None],
            xx.to(torch.float64)[:, None, :] - f64(src.x[jj])[:, None, None],
        )
        ann = ((rr >= hb - 15) & (rr <= hb - 3)).reshape(len(jj), -1)
        flatv = Iw.reshape(len(jj), -1)
        big = torch.finfo(flatv.dtype).max
        cnt = ann.sum(1)
        srt = torch.where(ann, flatv, torch.full_like(flatv, big)).sort(-1).values
        pct = ops.percentile_lastdim(srt, 30.0, cnt)
        Iw = torch.where((cnt > 0)[:, None, None], Iw - pct[:, None, None], Iw)
        Ic = torch.clamp(Iw, min=0)
        u = (_rl_batch(Ic, Kn, p.rl_iter) / Phi).to(torch.float32)
        u = torch.where(rr > hb - 3, torch.zeros_like(u), u)
        u_np = u.cpu().numpy()
        for b, j in enumerate(jj):
            src.u[int(j)] = (int(wb[j][0]), int(wb[j][2]), u_np[b])
            src.a[j] = float(u[b].sum())
            src.blob[j] = True
        full = torch.fft.irfft2(
            torch.fft.rfft2(u, s=(hh + K0.shape[0] - 1, ww + K0.shape[1] - 1))
            * torch.fft.rfft2(K0, s=(hh + K0.shape[0] - 1, ww + K0.shape[1] - 1))[None],
            s=(hh + K0.shape[0] - 1, ww + K0.shape[1] - 1),
        )
        fy = (y0t - R0)[:, None] + torch.arange(full.shape[1], device=dev)[None, :]
        fx = (x0t - R0)[:, None] + torch.arange(full.shape[2], device=dev)[None, :]
        fl = (fy + fr.pad)[:, :, None] * fr.Wp + (fx + fr.pad)[:, None, :]
        fr.M.index_add_(0, fl.reshape(-1), full.reshape(-1))


# ------------------------------------------------------------------------------------
# stretch field
def stretch_search(
    img: torch.Tensor,
    B: torch.Tensor,
    M: torch.Tensor,
    src: Sources,
    g: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    tmpl: Template,
    k: int,
    noise: float,
    p: PsfSubtractParams,
    cand: np.ndarray,
    grid: np.ndarray,
    opts: FitOpts,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched ``_stretch_search``: grid search of the template dilation per candidate
    (weighted SSE after refitting gain and offset), then a parabola through the log-SSE
    minimum."""
    dev = img.device
    Y, X = img.shape
    N = src.n
    s_new = s.copy()
    snr = np.zeros(N)
    ci = np.where(cand)[0]
    if ci.size == 0:
        return s_new, snr
    tm = TmplDev.from_template(tmpl)
    sig_eff = noise * p.tex
    gmax_grid = float(grid.max())
    Rw = (tmpl.r_sup + p.win_extra) * gmax_grid
    Hb = int(np.ceil(2 * Rw)) + 3
    pad = Hb + 2
    fr = _frame_with_model(img, B, M, pad)

    def f64(a: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, np.float64), device=dev)

    per = max(1, opts.px_budget // (Hb * Hb) // 3)
    log_sse = np.zeros((len(ci), len(grid)))
    for i0 in range(0, len(ci), per):
        sel = ci[i0 : i0 + per]
        yc, xc, a0 = f64(src.y[sel]), f64(src.x[sel]), f64(src.a[sel])
        gi0 = f64(g[sel])
        geom = make_geom(yc, xc, torch.full_like(yc, Rw), Hb, Hb, pad, fr.Wp)
        base = fr.base[geom.flat]
        Mw = fr.M[geom.flat]
        inimg = _in_image(geom, Y, X) * geom.mask
        vw = fr.valid[geom.flat].to(torch.float32) * geom.mask
        r = radius_grid(geom, yc, xc)
        own = (a0 * gi0).to(torch.float32)[:, None, None] * tm.eval(
            r, f64(s[sel]).to(torch.float32)[:, None, None]
        )
        res = base - Mw + own * geom.mask
        t1 = tm.eval(r, 1.0)
        snr_b = (
            a0
            * torch.clamp(gi0, min=0.3)
            * torch.sqrt((t1 * t1 * inimg).sum((1, 2), dtype=torch.float64))
            / sig_eff
        )
        snr[sel] = snr_b.cpu().numpy()
        cb = f64(c[sel])
        a32 = a0.to(torch.float32)[:, None, None]
        for jg, sg_ in enumerate(grid):
            h = a32 * tm.eval(r, float(sg_)) * geom.mask
            gi = torch.clamp(gi0, min=0.3)
            cc = cb
            w = vw
            for _ in range(2):
                e = (
                    res
                    - gi.to(torch.float32)[:, None, None] * h
                    - cc.to(torch.float32)[:, None, None]
                )
                sig = torch.sqrt(
                    sig_eff**2
                    + (p.rel_tol * gi.to(torch.float32)[:, None, None] * h) ** 2
                )
                w = vw / (e / (p.huber_k * sig)).clamp(min=1.0)
                s_hh = (w * h * h).sum((1, 2), dtype=torch.float64)
                s_h1 = (w * h).sum((1, 2), dtype=torch.float64)
                s_11 = w.sum((1, 2), dtype=torch.float64)
                s_hr = (w * h * res).sum((1, 2), dtype=torch.float64)
                s_1r = (w * res).sum((1, 2), dtype=torch.float64)
                a00, a01, a11 = s_hh + 1e-9, s_h1, s_11 + 1e-9
                det = a00 * a11 - a01 * a01
                gi = ((s_hr * a11 - a01 * s_1r) / det).clamp(0.0, p.g_max)
                cc = (a00 * s_1r - a01 * s_hr) / det
            e = (
                res
                - gi.to(torch.float32)[:, None, None] * h
                - cc.to(torch.float32)[:, None, None]
            )
            log_sse[i0 : i0 + len(sel), jg] = (
                torch.log((w * e * e).sum((1, 2), dtype=torch.float64)).cpu().numpy()
            )
    ok = snr[ci] >= p.stretch_snr
    j = np.argmin(log_sse, axis=1)
    dl = np.log(grid[1]) - np.log(grid[0])
    ls = np.log(grid[j])
    inner = (j > 0) & (j < len(grid) - 1)
    jj = np.clip(j, 1, len(grid) - 2)
    r_ = np.arange(len(ci))
    a_, b_, c_ = log_sse[r_, jj - 1], log_sse[r_, jj], log_sse[r_, jj + 1]
    den = a_ - 2 * b_ + c_
    dj = np.where(den > 0, 0.5 * (a_ - c_) / np.where(den > 0, den, 1.0), 0.0)
    ls = np.where(inner, np.log(grid[jj]) + np.clip(dj, -1, 1) * dl, ls)
    upd = ci[ok]
    s_new[upd] = np.exp(ls[ok])
    return s_new, snr


def estimate_stretch_field(
    imgs: Mapping[int, torch.Tensor],
    Bs: Mapping[int, torch.Tensor],
    src: Sources,
    fits0: Mapping[int, PlaneResult],
    p: PsfSubtractParams,
    *,
    origin_yx: tuple[int, int],
    ell_default: float,
    ks: Sequence[int] = (0, 1, 2),
    opts: FitOpts,
) -> tuple[np.ndarray, int, float]:
    """GPU ``estimate_stretch_field``; the candidate rule and the smooth field are the
    CPU functions."""
    N = src.n
    num = np.zeros(N)
    den = np.zeros(N)
    for k in ks:
        f = fits0[k]
        assert src.sel is not None
        cand = src.sel.copy()
        est = src.a * f.g * f.tmpl.vals.max() / (p.tex * f.noise)
        cand &= est >= p.stretch_snr
        cand = top_per_block(
            est, cand, src.y, src.x, origin_yx, p.stretch_block_px, p.stretch_per_block
        )
        sn, snr = stretch_search(
            imgs[k],
            Bs[k],
            f.model,
            src,
            f.g,
            f.c,
            f.s,
            f.tmpl,
            k,
            f.noise,
            p,
            cand,
            _GRID,
            opts,
        )
        ok = (
            cand
            & (snr >= p.stretch_snr)
            & (sn > _GRID[0] * 1.01)
            & (sn < _GRID[-1] * 0.99)
        )
        w = (snr * BETA[k]) ** 2
        num[ok] += w[ok] * np.log(sn[ok]) / BETA[k]
        den[ok] += w[ok]
    m = den > 0
    pts = np.c_[src.y, src.x]
    ell_b = np.where(m, num / np.maximum(den, 1e-12), 0.0)
    ell = smooth_field(
        pts[m], ell_b[m], np.sqrt(den[m]), pts, p.stretch_sigma, kmin=p.stretch_kmin
    )
    return (
        np.where(np.isfinite(ell), ell, ell_default),
        int(m.sum()),
        (float(np.median(ell_b[m])) if m.any() else float("nan")),
    )


# ------------------------------------------------------------------------------------
def fit_window_library(
    polyt: dict[int, torch.Tensor],
    dapi: dict[int, torch.Tensor],
    *,
    origin_yx: tuple[int, int],
    calibration: WindowCalibration,
    params: PsfSubtractParams = DEFAULT_PARAMS,
    planes: Sequence[int],
    opts: FitOpts | None = None,
    timers: dict[str, float] | None = None,
) -> BeadModel:
    """GPU ``fit_window_library`` (steps B3-B8) for device planes; see
    ``merxen.io.bead_removal.model.fit_window_library``.

    Args:
        polyt: {z: float32 device plane} for the reference stain (0, 1, 2 and
            ``planes``).
        dapi: {z: device plane} for planes 0 and 1.
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0) (multiple of 16).
        calibration: Dataset calibration.
        params: Fixed parameters.
        planes: Output planes (> 2 are fitted only when listed).
        opts: Execution options (Gauss-Seidel mode ...).
        timers: Optional dict that receives per-stage wall seconds
            (device-synchronised).

    Returns:
        ``merxen.io.bead_removal.model.BeadModel`` whose ``fits[k].model`` and ``B[k]``
        are device tensors.
    """
    p = params
    opts = dataclasses.replace(
        opts or FitOpts(), origin_yx=(int(origin_yx[0]), int(origin_yx[1]))
    )
    tm_ = timers if timers is not None else {}

    def stamp(name: str, t0: float) -> float:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tm_[name] = tm_.get(name, 0.0) + (t1 - t0)
        tm_["peak_after_" + name] = torch.cuda.max_memory_allocated() / 1e9
        return t1

    t = time.perf_counter()
    t00 = t
    fit_planes = sorted({0, 1, 2} | {int(k) for k in planes if k > 2})
    lib = calibration.library
    src = detect_sources(
        polyt[0],
        polyt[1],
        dapi.get(0),
        dapi.get(1),
        dog_noise=calibration.dog_noise_z0,
        thresholds=calibration.thresholds,
        params=p,
        z_planes={
            k: polyt[k] for k in calibration.thresholds.rescue_planes if k in polyt
        },
    )
    info: dict = dict(
        n_cand=int(src.n),
        n_bead=int(src.is_bead.sum()),
        n_rescued=int(src.rescued.sum()) if src.rescued is not None else 0,
    )
    t = stamp("detect", t)
    if not src.is_bead.any():
        return BeadModel(
            src,
            {},
            {},
            {},
            dict(info, status="no beads", time_s=time.perf_counter() - t00),
        )
    imgs = {k: polyt[k] for k in fit_planes}
    n_sat = 0
    if np.isfinite(p.sat_level):
        for k in fit_planes:
            sat = imgs[k] >= p.sat_level
            n_sat += int(sat.sum())
            imgs[k] = torch.where(sat, torch.zeros_like(imgs[k]), imgs[k])
    info["n_saturated_px"] = n_sat
    p_z0 = dataclasses.replace(p, g_max=6.0) if n_sat else p
    B = {k: ops.lower_envelope(imgs[k], imgs[k] > 0) for k in fit_planes}
    t = stamp("lower_envelope", t)
    noise = {k: float(calibration.noise[k]) for k in fit_planes}
    lo, hi = calibration.a0_sel
    a0_median = calibration.a0_median
    use = src.is_bead.copy()
    src.cand_bead = use.copy()
    sel = use & (lo <= src.A0) & (hi >= src.A0)
    t_det = time.perf_counter() - t00
    # ---- z0 point sources
    f0 = fit_plane(
        imgs[0],
        B[0],
        src,
        0,
        p_z0,
        lib.tmpls[0],
        use=use,
        noise=noise[0],
        refine_pos=True,
        opts=opts,
    )
    src.a = src.A0 * f0.g
    f0.g[:] = 1.0
    t = stamp("fit z0 (pass 1)", t)
    blobs = build_blobs(imgs[0], B[0], src, f0, p, a0_median, opts)
    info["n_blob"] = int(len(blobs))
    t = stamp("blobs", t)
    assert src.q is not None
    single = use & ~src.blob
    sel = single & (src.q < 1.15) & (src.a >= lo * 0.8) & (src.a <= hi)
    src.sel = sel
    f0 = fit_plane(
        imgs[0],
        B[0],
        src,
        0,
        p_z0,
        lib.tmpls[0],
        use=use,
        noise=noise[0],
        refine_pos=True,
        opts=opts,
    )
    src.a = np.where(single, src.a * f0.g, src.a)
    f0.g[single] = 1.0
    fits = {0: f0}
    t = stamp("fit z0 (pass 2)", t)
    t_z0 = time.perf_counter() - t00
    for k in (1, 2):
        fits[k] = fit_plane(
            imgs[k], B[k], src, k, p, lib.tmpls[k], use=use, noise=noise[k], opts=opts
        )
    t = stamp("fit z1,z2 (pass 1)", t)
    ell, nfield, ell_meas = estimate_stretch_field(
        imgs,
        B,
        src,
        fits,
        p,
        origin_yx=origin_yx,
        ell_default=float(calibration.ell_default),
        ks=(0, 1, 2),
        opts=opts,
    )
    info["ell_meas_median"] = ell_meas
    src.ell = ell
    info["n_stretch_beads"] = nfield
    info["ell_median"] = float(np.median(ell))
    info["ell_sd"] = float(np.std(ell))
    t = stamp("stretch field", t)
    t_str = time.perf_counter() - t00
    for k in (1, 2):
        fits[k] = fit_plane(
            imgs[k],
            B[k],
            src,
            k,
            p,
            lib.tmpls[k],
            use=use,
            s=np.exp(BETA[k] * ell),
            noise=noise[k],
            opts=opts,
        )
    t = stamp("fit z1,z2 (pass 2)", t)
    low = [1, 2]
    # The device model keeps tensors where the CPU dataclasses are typed as numpy.
    pruned, n_evid = _prune_from_low_planes(src, fits, low, use, p)  # type: ignore[arg-type]
    info["n_pruned"] = int(pruned.sum())
    info["n_ring_evidence"] = n_evid
    pr = np.where(pruned)[0]
    for k in low:
        if pr.size:
            fitter = PlaneFitter(
                imgs[k],
                B[k],
                src,
                k,
                p,
                lib.tmpls[k],
                fits[k].s,
                np.ones(src.n, bool),
                noise[k],
                None,
                opts,
            )
            M = fits[k].model
            fitter.frame.M.view(fitter.frame.Y + 2 * fitter.frame.pad, fitter.frame.Wp)[
                fitter.frame.pad : fitter.frame.pad + fitter.frame.Y,
                fitter.frame.pad : fitter.frame.pad + fitter.frame.X,
            ] = M
            fitter.g = torch.as_tensor(fits[k].g, device=M.device)
            fitter.remove(pr)
            fits[k].model = fitter.frame.crop_model()
    use = use & ~pruned
    src.is_bead = src.is_bead & ~pruned
    sel = sel & ~pruned
    src.sel = sel
    gamma = _consensus_gamma(src, fits, low, p)  # type: ignore[arg-type]
    t = stamp("prune", t)
    for k in [k for k in fit_planes if k > 2]:
        fits[k] = fit_plane(
            imgs[k],
            B[k],
            src,
            k,
            p,
            lib.tmpls[k],
            use=use,
            s=np.exp(BETA[k] * ell),
            noise=noise[k],
            gp_arr=gamma,
            opts=opts,
        )
        t = stamp(f"fit z{k}", t)
    info["gamma_med"] = float(np.median(gamma[use])) if use.any() else 1.0
    info["n_beads_final"] = int(use.sum())
    info["status"] = "ok"
    info["time_s"] = time.perf_counter() - t00
    info["time_detect_s"], info["time_z0_s"], info["time_stretch_s"] = (
        t_det,
        t_z0 - t_det,
        t_str - t_z0,
    )
    return BeadModel(src, fits, B, noise, info)  # type: ignore[arg-type]
