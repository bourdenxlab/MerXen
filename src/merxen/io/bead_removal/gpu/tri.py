"""Fused Triton kernels for the per-source window fits (best-case GPU implementation).

One program handles the whole window of one source: it generates the radial-template
stamp on the fly, reads the image / model / validity from the padded frame, runs the
three re-weighting iterations of the robust gain-offset fit and returns (gain, offset);
a second kernel adds the stamp times a gain to the model.  No ``[B, H, W]`` temporary is
ever materialised, so a batch costs a handful of launches instead of several hundred.

Windows inside one batch must be disjoint (colour batches), so the model update is a
plain read-modify-write.
"""

# The jitted kernels take raw device pointers and ``tl.constexpr`` values: Triton reads
# the parameter annotations itself, so they stay unannotated here, and the kernel
# arguments keep the short names of the formulas in the module docstring.
# ruff: noqa: ANN001, ANN202, N803, N806

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .fit import Frame

NUM_WARPS = 4


@triton.jit
def _stamp(tab_ptr, n_tab, ds_inv, ay, ax, yc32, xc32, a_norm, s_inv, tmask):
    """Stamp value a * H(r/s) / s^2 at integer pixel coordinates (ay, ax) (float32), 0
    beyond the table."""
    dy = ay.to(tl.float32) - yc32
    dx = ax.to(tl.float32) - xc32
    r = tl.sqrt(dy * dy + dx * dx)
    t = r * s_inv * ds_inv
    i0f = tl.floor(t)
    i0 = tl.minimum(tl.maximum(i0f, 0.0), (n_tab - 2).to(tl.float32)).to(tl.int32)
    f = t - i0.to(tl.float32)
    v0 = tl.load(tab_ptr + i0, mask=tmask, other=0.0)
    v1 = tl.load(tab_ptr + i0 + 1, mask=tmask, other=0.0)
    h = v0 + (v1 - v0) * f
    h = tl.where(t <= (n_tab - 1).to(tl.float32), h, 0.0)
    return h * a_norm


@triton.jit
def _fit_kernel(
    base_ptr,
    valid_ptr,
    m_ptr,
    tab_ptr,
    yc_ptr,
    xc_ptr,
    a_ptr,
    s_ptr,
    gold_ptr,
    c0_ptr,
    gp_ptr,
    lamg_ptr,
    gmax_ptr,
    gi_out,
    ci_out,
    Wp,
    pad,
    n_tab,
    ds_inv,
    r_sup,
    win_extra,
    sig_eff,
    rel_tol,
    huber_k,
    BY: tl.constexpr,
    BX: tl.constexpr,
):
    b = tl.program_id(0)
    yc = tl.load(yc_ptr + b)
    xc = tl.load(xc_ptr + b)
    a = tl.load(a_ptr + b)
    s = tl.load(s_ptr + b)
    gold = tl.load(gold_ptr + b)
    c0 = tl.load(c0_ptr + b)
    gp = tl.load(gp_ptr + b)
    lam_g = tl.load(lamg_ptr + b)
    gmax = tl.load(gmax_ptr + b)
    R = r_sup * s + win_extra
    y0 = tl.floor(yc - R)
    y1 = tl.ceil(yc + R) + 1.0
    x0 = tl.floor(xc - R)
    x1 = tl.ceil(xc + R) + 1.0
    hh = (y1 - y0).to(tl.int32)
    ww = (x1 - x0).to(tl.int32)
    y0i = y0.to(tl.int32)
    x0i = x0.to(tl.int32)
    yc32 = yc.to(tl.float32)
    xc32 = xc.to(tl.float32)
    s32 = s.to(tl.float32)
    a_norm = (a / (s * s)).to(tl.float32)
    s_inv = 1.0 / s32
    gold32 = gold.to(tl.float32)
    s2 = sig_eff * sig_eff
    lam_c = 1.0 / 9.0
    gi = gold
    ci = c0
    ar_y = tl.arange(0, BY)[:, None]
    ar_x = tl.arange(0, BX)[None, :]
    for _it in range(3):
        gi32 = gi.to(tl.float32)
        ci32 = ci.to(tl.float32)
        s_hh = gold * 0.0  # float64 accumulators
        s_h1 = gold * 0.0
        s_11 = gold * 0.0
        s_hr = gold * 0.0
        s_1r = gold * 0.0
        for ry in range(0, hh, BY):
            iy = ry + ar_y
            ay = y0i + iy
            for rx in range(0, ww, BX):
                ix = rx + ar_x
                ax = x0i + ix
                msk = (iy < hh) & (ix < ww)
                lin = (ay + pad) * Wp + (ax + pad)
                base = tl.load(base_ptr + lin, mask=msk, other=0.0)
                mv = tl.load(m_ptr + lin, mask=msk, other=0.0)
                v = tl.load(valid_ptr + lin, mask=msk, other=0).to(tl.float32)
                h = _stamp(
                    tab_ptr, n_tab, ds_inv, ay, ax, yc32, xc32, a_norm, s_inv, msk
                )
                res = base - mv + gold32 * h
                gh = gi32 * h
                e = res - gh - ci32
                sig = tl.sqrt(s2 + (rel_tol * gh) * (rel_tol * gh))
                t = e / (huber_k * sig)
                w = v / tl.maximum(t, 1.0)
                wh = w * h
                s_hh += tl.sum(tl.sum(wh * h, 1), 0).to(tl.float64)
                s_h1 += tl.sum(tl.sum(wh, 1), 0).to(tl.float64)
                s_11 += tl.sum(tl.sum(w, 1), 0).to(tl.float64)
                s_hr += tl.sum(tl.sum(wh * res, 1), 0).to(tl.float64)
                s_1r += tl.sum(tl.sum(w * res, 1), 0).to(tl.float64)
        a00 = s_hh + lam_g
        a01 = s_h1
        a11 = s_11 + lam_c
        b0 = s_hr + lam_g * gp
        b1 = s_1r
        det = a00 * a11 - a01 * a01
        g_new = (b0 * a11 - a01 * b1) / det
        g_new = tl.minimum(tl.maximum(g_new, 0.0), gmax)
        ci = (a00 * b1 - a01 * b0) / det
        gi = g_new
    tl.store(gi_out + b, gi)
    tl.store(ci_out + b, ci)


@triton.jit
def _pos_kernel(
    base_ptr,
    valid_ptr,
    m_ptr,
    tab_ptr,
    yc_ptr,
    xc_ptr,
    a_ptr,
    s_ptr,
    gold_ptr,
    gi_ptr,
    ci_ptr,
    dy_out,
    dx_out,
    Wp,
    pad,
    n_tab,
    ds_inv,
    r_sup,
    win_extra,
    sig_eff,
    rel_tol,
    huber_k,
    BY: tl.constexpr,
    BX: tl.constexpr,
):
    """z0: one Gauss-Newton step of the sub-pixel position (finite-difference stamp
    derivatives, step 0.25 px)."""
    b = tl.program_id(0)
    yc = tl.load(yc_ptr + b)
    xc = tl.load(xc_ptr + b)
    a = tl.load(a_ptr + b)
    s = tl.load(s_ptr + b)
    gold = tl.load(gold_ptr + b)
    gi = tl.load(gi_ptr + b)
    ci = tl.load(ci_ptr + b)
    R = r_sup * s + win_extra
    y0 = tl.floor(yc - R)
    y1 = tl.ceil(yc + R) + 1.0
    x0 = tl.floor(xc - R)
    x1 = tl.ceil(xc + R) + 1.0
    hh = (y1 - y0).to(tl.int32)
    ww = (x1 - x0).to(tl.int32)
    y0i = y0.to(tl.int32)
    x0i = x0.to(tl.int32)
    yc32 = yc.to(tl.float32)
    xc32 = xc.to(tl.float32)
    s32 = s.to(tl.float32)
    a_norm = (a / (s * s)).to(tl.float32)
    s_inv = 1.0 / s32
    gold32 = gold.to(tl.float32)
    gi32 = gi.to(tl.float32)
    ci32 = ci.to(tl.float32)
    d = 0.25
    s2 = sig_eff * sig_eff
    ar_y = tl.arange(0, BY)[:, None]
    ar_x = tl.arange(0, BX)[None, :]
    A00 = gold * 0.0
    A01 = gold * 0.0
    A11 = gold * 0.0
    b0 = gold * 0.0
    b1 = gold * 0.0
    hmax = gold * 0.0
    for ry in range(0, hh, BY):
        iy = ry + ar_y
        ay = y0i + iy
        for rx in range(0, ww, BX):
            ix = rx + ar_x
            ax = x0i + ix
            msk = (iy < hh) & (ix < ww)
            lin = (ay + pad) * Wp + (ax + pad)
            base = tl.load(base_ptr + lin, mask=msk, other=0.0)
            mv = tl.load(m_ptr + lin, mask=msk, other=0.0)
            v = tl.load(valid_ptr + lin, mask=msk, other=0).to(tl.float32)
            h = _stamp(tab_ptr, n_tab, ds_inv, ay, ax, yc32, xc32, a_norm, s_inv, msk)
            hyp = _stamp(
                tab_ptr, n_tab, ds_inv, ay, ax, yc32 + d, xc32, a_norm, s_inv, msk
            )
            hym = _stamp(
                tab_ptr, n_tab, ds_inv, ay, ax, yc32 - d, xc32, a_norm, s_inv, msk
            )
            hxp = _stamp(
                tab_ptr, n_tab, ds_inv, ay, ax, yc32, xc32 + d, a_norm, s_inv, msk
            )
            hxm = _stamp(
                tab_ptr, n_tab, ds_inv, ay, ax, yc32, xc32 - d, a_norm, s_inv, msk
            )
            hy = (hyp - hym) / (2.0 * d)
            hx = (hxp - hxm) / (2.0 * d)
            res = base - mv + gold32 * h
            gh = gi32 * h
            e = res - gh - ci32
            sig = tl.sqrt(s2 + (rel_tol * gh) * (rel_tol * gh))
            w = v / tl.maximum(e / (huber_k * sig), 1.0)
            c0 = gi32 * hy
            c1 = gi32 * hx
            A00 += tl.sum(tl.sum(w * c0 * c0, 1), 0).to(tl.float64)
            A01 += tl.sum(tl.sum(w * c0 * c1, 1), 0).to(tl.float64)
            A11 += tl.sum(tl.sum(w * c1 * c1, 1), 0).to(tl.float64)
            b0 += tl.sum(tl.sum(w * c0 * e, 1), 0).to(tl.float64)
            b1 += tl.sum(tl.sum(w * c1 * e, 1), 0).to(tl.float64)
            hm = tl.max(tl.max(tl.where(msk, h, -1e30), 1), 0)
            hmax = tl.maximum(hmax, hm.to(tl.float64))
    reg = 1e-6 * (1.0 + A00 + A11)
    A00 = A00 + reg
    A11 = A11 + reg
    det = A00 * A11 - A01 * A01
    dy = (b0 * A11 - A01 * b1) / det
    dx = (A00 * b1 - A01 * b0) / det
    dy = tl.minimum(tl.maximum(dy, -0.6), 0.6)
    dx = tl.minimum(tl.maximum(dx, -0.6), 0.6)
    faint = gi * hmax < 20.0 * sig_eff
    dy = tl.where(faint, 0.0, dy)
    dx = tl.where(faint, 0.0, dx)
    tl.store(dy_out + b, dy)
    tl.store(dx_out + b, dx)


@triton.jit
def _add_kernel(
    m_ptr,
    tab_ptr,
    yc_ptr,
    xc_ptr,
    a_ptr,
    s_ptr,
    gain_ptr,
    Wp,
    pad,
    n_tab,
    ds_inv,
    r_sup,
    win_extra,
    BY: tl.constexpr,
    BX: tl.constexpr,
):
    """M[window of source b] += gain[b] * stamp_b  (windows of one launch are
    disjoint)."""
    b = tl.program_id(0)
    gain = tl.load(gain_ptr + b)
    yc = tl.load(yc_ptr + b)
    xc = tl.load(xc_ptr + b)
    a = tl.load(a_ptr + b)
    s = tl.load(s_ptr + b)
    R = r_sup * s + win_extra
    y0 = tl.floor(yc - R)
    y1 = tl.ceil(yc + R) + 1.0
    x0 = tl.floor(xc - R)
    x1 = tl.ceil(xc + R) + 1.0
    hh = (y1 - y0).to(tl.int32)
    ww = (x1 - x0).to(tl.int32)
    y0i = y0.to(tl.int32)
    x0i = x0.to(tl.int32)
    yc32 = yc.to(tl.float32)
    xc32 = xc.to(tl.float32)
    s32 = s.to(tl.float32)
    a_norm = (a / (s * s)).to(tl.float32)
    s_inv = 1.0 / s32
    g32 = gain.to(tl.float32)
    ar_y = tl.arange(0, BY)[:, None]
    ar_x = tl.arange(0, BX)[None, :]
    for ry in range(0, hh, BY):
        iy = ry + ar_y
        ay = y0i + iy
        for rx in range(0, ww, BX):
            ix = rx + ar_x
            ax = x0i + ix
            msk = (iy < hh) & (ix < ww)
            lin = (ay + pad) * Wp + (ax + pad)
            h = _stamp(tab_ptr, n_tab, ds_inv, ay, ax, yc32, xc32, a_norm, s_inv, msk)
            mv = tl.load(m_ptr + lin, mask=msk, other=0.0)
            tl.store(m_ptr + lin, mv + g32 * h, mask=msk)


def _tile(Hb: int) -> tuple[int, int]:
    """(BY, BX): tile of the window loop; BX = smallest power of two >= the window width
    (<= 256)."""
    bx = 32
    while bx < Hb and bx < 256:
        bx *= 2
    by = max(1, 1024 // bx)
    return by, bx


def fit_batch(
    frame: Frame,
    tab: torch.Tensor,
    ds: float,
    n_tab: int,
    r_sup: float,
    win_extra: float,
    Hb: int,
    yc: torch.Tensor,
    xc: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
    gold: torch.Tensor,
    c0: torch.Tensor,
    gp: torch.Tensor,
    lam_g: torch.Tensor,
    gmax: torch.Tensor,
    sig_eff: float,
    rel_tol: float,
    huber_k: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust gain / offset fit of a batch of sources (all per-source tensors float64
    [B]).  Returns (gi, ci)."""
    B = yc.shape[0]
    gi = torch.empty(B, dtype=torch.float64, device=yc.device)
    ci = torch.empty_like(gi)
    by, bx = _tile(Hb)
    _fit_kernel[(B,)](
        frame.base,
        frame.valid,
        frame.M,
        tab,
        yc,
        xc,
        a,
        s,
        gold,
        c0,
        gp,
        lam_g,
        gmax,
        gi,
        ci,
        frame.Wp,
        frame.pad,
        n_tab,
        1.0 / ds,
        r_sup,
        win_extra,
        sig_eff,
        rel_tol,
        huber_k,
        BY=by,
        BX=bx,
        num_warps=NUM_WARPS,
    )
    return gi, ci


def position_step(
    frame: Frame,
    tab: torch.Tensor,
    ds: float,
    n_tab: int,
    r_sup: float,
    win_extra: float,
    Hb: int,
    yc: torch.Tensor,
    xc: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
    gold: torch.Tensor,
    gi: torch.Tensor,
    ci: torch.Tensor,
    sig_eff: float,
    rel_tol: float,
    huber_k: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One bounded position step per source (see ``_pos_kernel``); returns (dy, dx)."""
    B = yc.shape[0]
    dy = torch.empty(B, dtype=torch.float64, device=yc.device)
    dx = torch.empty_like(dy)
    by, bx = _tile(Hb)
    _pos_kernel[(B,)](
        frame.base,
        frame.valid,
        frame.M,
        tab,
        yc,
        xc,
        a,
        s,
        gold,
        gi,
        ci,
        dy,
        dx,
        frame.Wp,
        frame.pad,
        n_tab,
        1.0 / ds,
        r_sup,
        win_extra,
        sig_eff,
        rel_tol,
        huber_k,
        BY=by,
        BX=bx,
        num_warps=NUM_WARPS,
    )
    return dy, dx


def add_stamps(
    frame: Frame,
    tab: torch.Tensor,
    ds: float,
    n_tab: int,
    r_sup: float,
    win_extra: float,
    Hb: int,
    yc: torch.Tensor,
    xc: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
    gain: torch.Tensor,
) -> None:
    """M += gain * stamp for a batch of sources with disjoint windows."""
    B = yc.shape[0]
    by, bx = _tile(Hb)
    _add_kernel[(B,)](
        frame.M,
        tab,
        yc,
        xc,
        a,
        s,
        gain,
        frame.Wp,
        frame.pad,
        n_tab,
        1.0 / ds,
        r_sup,
        win_extra,
        BY=by,
        BX=bx,
        num_warps=NUM_WARPS,
    )
