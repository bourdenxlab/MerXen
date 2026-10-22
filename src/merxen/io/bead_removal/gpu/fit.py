"""Batched GPU version of the per-plane forced-photometry fit
(``merxen.io.bead_removal.model.fit_plane`` with a fixed library template).

Algorithmic differences from the CPU code (validated against the CPU code):

* ``mode="color"`` (production): the sequential Gauss-Seidel back-fitting (brightest
  first) is replaced by multicolour Gauss-Seidel.  Sources are placed on a grid of cells
  of size ``2 R_max + 2`` px; cells of the same parity class cannot contain interacting
  windows, and the sources inside one cell are ranked by brightness, so colour = 4 x
  rank + parity. Within a colour all windows are disjoint, so the batched update is
  *exactly* the sequential one for that colour; the order of the colours differs from
    the brightness order (same fixed point, different path in 3 sweeps).
* ``mode="seq"`` (oracle for the tests): one source per step in brightness order, i.e.
  the CPU algorithm itself; it reproduces the numpy code to float32 rounding.  Slow,
  never used in production.
* Blobs (extended sources with a stored source map) are updated one by one at the start
  of each sweep.
* The one-sided Huber re-weighting, the 2x2 ridge solution and the clipping of the gain
  follow ``merxen.io.bead_removal.model._solve_gc``.
"""

# Window dimensions and matrices keep the capital letters of the formulas they port
# from the CPU modules (H, W, B, M, K, A0 ...), so the two can be read side by side.
# ruff: noqa: N803, N806

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from merxen.io.bead_removal.detect import Sources
from merxen.io.bead_removal.params import PsfSubtractParams
from merxen.io.bead_removal.psf import Template

from .stamps import Geom, TmplDev, make_geom, point_stamps

logger = logging.getLogger(__name__)

PX_BUDGET = 14_000_000  # window pixels per batch (tensor [B, Hb, Wb])


@dataclass
class FitOpts:
    """Execution options of the GPU fit (not algorithm parameters).

    The defaults are the production configuration: colour update order, fused Triton
    kernels, colour cells anchored on the absolute mosaic grid with a fixed cell size.
    ``mode="seq"`` (brightness order, one source per step) is the test oracle;
    ``triton=False`` runs the same algorithm with plain torch (slower, for hosts without
    a working Triton).
    """

    mode: str = "color"  # "color" (production) | "seq" (oracle)
    px_budget: int = PX_BUDGET
    # absolute mosaic coordinate of window pixel (0, 0): anchors the colour grid
    origin_yx: tuple[int, int] = (0, 0)
    # colour cells on the absolute grid with a fixed cell size (chunk independent
    # ordering)
    anchor: bool = True
    # largest radius dilation the anchored cell size allows for (larger: window-relative
    # cells)
    s_cap: float = 1.6
    # seq mode: random update order (order-sensitivity checks) instead of the
    # brightness order
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("color", "seq"):
            raise ValueError(
                f"unknown fit mode {self.mode!r}; expected 'color' or 'seq'"
            )

    triton: bool = True  # fused Triton kernels for the point-source updates


@dataclass
class PlaneResult:
    """Output of one plane fit; ``model`` is a device tensor [Y, X]."""

    tmpl: Template
    g: np.ndarray
    c: np.ndarray
    s: np.ndarray
    model: torch.Tensor
    noise: float


class Frame:
    """Zero-padded device images of one plane: base = I - B, valid mask, model M (all
    [Y+2P, X+2P])."""

    def __init__(self, img: torch.Tensor, B: torch.Tensor, pad: int) -> None:
        self.Y, self.X = img.shape
        self.pad = pad
        self.Wp = self.X + 2 * pad
        f = torch.nn.functional.pad
        self.base = f(img - B, (pad, pad, pad, pad)).reshape(-1).contiguous()
        self.valid = (
            f((img > 0).to(torch.uint8), (pad, pad, pad, pad)).reshape(-1).contiguous()
        )
        self.M = torch.zeros(
            (self.Y + 2 * pad) * self.Wp, dtype=torch.float32, device=img.device
        )

    def crop_model(self) -> torch.Tensor:
        p = self.pad
        return self.M.view(self.Y + 2 * p, self.Wp)[
            p : p + self.Y, p : p + self.X
        ].contiguous()


def grid_colours(
    y: np.ndarray, x: np.ndarray, A0: np.ndarray, D: float
) -> list[np.ndarray]:
    """Partition sources into groups whose square windows of side <= D cannot overlap
    (multicolour ordering).

    Sources are put on a grid of D x D cells; two cells of the same parity class (cy %
    2, cx % 2) are at least one cell apart, so windows of side <= D in them are
    disjoint.  Inside one cell the sources are ranked by brightness (A0) and get
    different colours, colour = 4 x rank + parity.  Returns index arrays into the
    inputs, ascending colour.
    """
    n = len(y)
    if n == 0:
        return []
    cy = np.floor(np.asarray(y) / D).astype(np.int64)
    cx = np.floor(np.asarray(x) / D).astype(np.int64)
    order = np.argsort(-np.asarray(A0), kind="stable")
    key = cy[order] * 100003 + cx[order]
    o2 = np.argsort(key, kind="stable")
    ks = key[o2]
    start = np.r_[True, ks[1:] != ks[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    rank = np.empty(n, np.int64)
    rank[order[o2]] = np.arange(n) - first
    colour = rank * 4 + (cy % 2) * 2 + (cx % 2)
    srt = np.argsort(colour, kind="stable")
    cs = colour[srt]
    cut = np.r_[0, np.where(cs[1:] != cs[:-1])[0] + 1, n]
    return [srt[cut[i] : cut[i + 1]] for i in range(len(cut) - 1)]


@dataclass
class BlobGroup:
    """A colour of blobs: stacked stamp windows (common padded size) and their
    geometry."""

    idx_np: np.ndarray
    idx: torch.Tensor
    geom: Geom
    h: torch.Tensor


# ------------------------------------------------------------------------------------
# robust solve of a batch
def _irls_sums(
    res: torch.Tensor,
    h: torch.Tensor,
    vw: torch.Tensor,
    g32: torch.Tensor,
    c32: torch.Tensor,
    s2: float,
    rt: float,
    hk: float,
) -> tuple[torch.Tensor, ...]:
    """Weighted sums of one re-weighting iteration: (S_hh, S_h1, S_11, S_hr, S_1r) per
    source, float64 [B]."""
    gh = g32 * h
    e = res - gh - c32
    sig = torch.sqrt(s2 + (rt * gh) ** 2)
    w = vw / torch.clamp(e / (hk * sig), min=1.0)
    wh = w * h
    return (
        (wh * h).sum((1, 2), dtype=torch.float64),
        wh.sum((1, 2), dtype=torch.float64),
        w.sum((1, 2), dtype=torch.float64),
        (wh * res).sum((1, 2), dtype=torch.float64),
        (w * res).sum((1, 2), dtype=torch.float64),
    )


def _solve_gc_impl(
    res: torch.Tensor,
    h: torch.Tensor,
    vw: torch.Tensor,
    sig_eff: float,
    lam_g: torch.Tensor,
    gp: torch.Tensor,
    g0: torch.Tensor,
    c0: torch.Tensor,
    gmax: torch.Tensor,
    rt: float,
    hk: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ``merxen.io.bead_removal.model._solve_gc`` (3 re-weighting iterations);
    all image tensors [B, Hb, Wb] float32, per-source tensors [B] float64."""
    gi, ci = g0, c0
    lam_c = 1.0 / 9.0
    s2 = sig_eff * sig_eff
    for _ in range(3):
        s_hh, s_h1, s_11, s_hr, s_1r = _irls_sums(
            res,
            h,
            vw,
            gi.to(torch.float32)[:, None, None],
            ci.to(torch.float32)[:, None, None],
            s2,
            rt,
            hk,
        )
        a00 = s_hh + lam_g
        a01 = s_h1
        a11 = s_11 + lam_c
        b0 = s_hr + lam_g * gp
        b1 = s_1r
        det = a00 * a11 - a01 * a01
        gi = torch.minimum(((b0 * a11 - a01 * b1) / det).clamp(min=0.0), gmax)
        ci = (a00 * b1 - a01 * b0) / det
    return gi, ci


def solve_gc(
    res: torch.Tensor,
    h: torch.Tensor,
    vw: torch.Tensor,
    sig_eff: float,
    lam_g: torch.Tensor,
    gp: torch.Tensor,
    g0: torch.Tensor,
    c0: torch.Tensor,
    gmax: torch.Tensor,
    p: PsfSubtractParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched robust one-sided-Huber ridge fit of gain and offset.

    See ``_solve_gc_impl``; the number of re-weighting iterations is fixed at 3.
    """
    return _solve_gc_impl(
        res,
        h,
        vw,
        float(sig_eff),
        lam_g,
        gp,
        g0,
        c0,
        gmax,
        float(p.rel_tol),
        float(p.huber_k),
    )


# ------------------------------------------------------------------------------------
# per-plane fitter
class PlaneFitter:
    """State of one plane fit (device tensors of the sources, padded frame, stamps
    cache)."""

    def __init__(
        self,
        img: torch.Tensor,
        B: torch.Tensor,
        src: Sources,
        k: int,
        p: PsfSubtractParams,
        tmpl: Template,
        s: np.ndarray,
        use: np.ndarray,
        noise: float,
        gp_arr: np.ndarray | None,
        opts: FitOpts,
    ) -> None:
        self.k, self.p, self.opts, self.noise = k, p, opts, float(noise)
        self.src, self.tmpl = src, tmpl
        self.tm = TmplDev.from_template(tmpl)
        dev = img.device
        self.dev = dev
        N = src.n
        self.N = N
        self.s_h = np.asarray(s, float)
        smax = float(self.s_h.max()) if N else 1.0
        blob = src.blob.astype(bool)
        self.Rmax = tmpl.r_sup * smax + p.win_extra
        self.Hb = int(np.ceil(2 * self.Rmax)) + 3
        # a window of common size Hb may start up to Hb - h_b px before the source's own
        # window
        pad = self.Hb + 2
        self.frame = Frame(img, B, pad)

        def f64(a: ArrayLike) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, np.float64), device=dev)

        self.yc, self.xc = f64(src.y).clone(), f64(src.x).clone()
        self.a = f64(src.a)
        self.s = f64(s)
        self.g = (
            f64(np.full(N, p.g_prior) if gp_arr is None else gp_arr).clone()
            if k > 0 or gp_arr is not None
            else f64(np.ones(N))
        )
        self.c = torch.zeros(N, dtype=torch.float64, device=dev)
        self.use_h = np.asarray(use, bool)
        self.blob_h = blob
        sg = np.full(N, p.g_prior_sd[k])
        gmax = np.full(N, p.g_max)
        if k > 0:
            sg[blob] = p.blob_g_sd
        gmax[blob] = p.blob_g_max
        self.sg = f64(sg)
        self.gmax = f64(gmax)
        self.lam_g = (self.noise * p.tex) ** 2 / self.sg**2
        gp: np.ndarray = np.full(N, p.g_prior)
        if gp_arr is not None and k > 0:
            gp = np.asarray(gp_arr, float)
        self.gp = f64(gp)

    # -- geometry / stamps
    def _geom(
        self,
        idx: torch.Tensor,
        yc: torch.Tensor | None = None,
        xc: torch.Tensor | None = None,
    ) -> Geom:
        yc = self.yc[idx] if yc is None else yc
        xc = self.xc[idx] if xc is None else xc
        R = self.tmpl.r_sup * self.s[idx] + self.p.win_extra
        return make_geom(yc, xc, R, self.Hb, self.Hb, self.frame.pad, self.frame.Wp)

    def _points(
        self,
        idx: torch.Tensor,
        yc: torch.Tensor | None = None,
        xc: torch.Tensor | None = None,
    ) -> tuple[Geom, torch.Tensor]:
        yc = self.yc[idx] if yc is None else yc
        xc = self.xc[idx] if xc is None else xc
        geom = self._geom(idx, yc, xc)
        h = point_stamps(self.tm, self.a[idx], self.s[idx], yc, xc, geom)
        return geom, h

    def _blob_groups(self, js: np.ndarray, seq: bool = False) -> list[BlobGroup]:
        """Colour groups of blobs with their stacked stamp windows.

        A blob's stamp is the full convolution of its source map ``u`` with the dilated
        plane kernel; its fit window is that stamp plus a ``win_extra`` margin.  The
        blobs of a group are stacked in a common (zero padded) size and the true
        per-blob window is restored through a row / column mask, so every sum equals the
        CPU window sum.
        """
        if len(js) == 0:
            return []
        dev, we = self.dev, int(self.p.win_extra)
        pad, Wp = self.frame.pad, self.frame.Wp
        total = self.frame.M.numel()
        r_sup = self.tmpl.r_sup
        rows = []
        for j in js:
            uy0, ux0, u = self.src.u[int(j)]
            R = int(np.ceil(r_sup * float(self.s_h[j])))
            rows.append((uy0, ux0, u.shape[0], u.shape[1], R))
        info = np.asarray(rows, np.int64)
        sy = info[:, 2] + 2 * info[:, 4] + 2 * we
        sx = info[:, 3] + 2 * info[:, 4] + 2 * we
        if seq:
            groups = [np.array([i]) for i in range(len(js))]
        else:
            cyy = info[:, 0] - info[:, 4] - we + sy / 2.0
            cxx = info[:, 1] - info[:, 4] - we + sx / 2.0
            if self.opts.anchor and float(self.s_h.max()) <= self.opts.s_cap:
                # blob map <= 83 px
                D = float(
                    2 * 40 + 3 + 2 * int(np.ceil(r_sup * self.opts.s_cap)) + 2 * we + 4
                )
                cyy, cxx = cyy + self.opts.origin_yx[0], cxx + self.opts.origin_yx[1]
            else:
                D = float(max(sy.max(), sx.max()) + 4)
            groups = grid_colours(cyy, cxx, self.src.A0[js], D)
        out = []
        for gsel in groups:
            ids = js[gsel]
            inf = info[gsel]
            B = len(ids)
            Rm, Hu, Wu = (
                int(inf[:, 4].max()),
                int(inf[:, 2].max()),
                int(inf[:, 3].max()),
            )
            U = np.zeros((B, Hu, Wu), np.float32)
            for b, j in enumerate(ids):
                u = self.src.u[int(j)][2]
                U[b, : u.shape[0], : u.shape[1]] = u
            ar = torch.arange(-Rm, Rm + 1, device=dev, dtype=torch.float64)
            r = torch.hypot(ar[:, None], ar[None, :]).to(torch.float32)
            sb = torch.as_tensor(self.s_h[ids], device=dev, dtype=torch.float32)[
                :, None, None
            ]
            K = self.tm.eval(r[None], sb)
            Hf, Wf = Hu + 2 * Rm, Wu + 2 * Rm
            full = torch.fft.irfft2(
                torch.fft.rfft2(torch.as_tensor(U, device=dev), s=(Hf, Wf))
                * torch.fft.rfft2(K, s=(Hf, Wf)),
                s=(Hf, Wf),
            )
            h = torch.nn.functional.pad(full, (we, we, we, we)).contiguous()
            Hm, Wm = h.shape[1], h.shape[2]
            oy = torch.as_tensor(inf[:, 0] - Rm - we, device=dev)
            ox = torch.as_tensor(inf[:, 1] - Rm - we, device=dev)
            offy = torch.as_tensor(Rm - inf[:, 4], device=dev)
            offx = torch.as_tensor(Rm - inf[:, 4], device=dev)
            hy = torch.as_tensor(inf[:, 2] + 2 * inf[:, 4] + 2 * we, device=dev)
            hx = torch.as_tensor(inf[:, 3] + 2 * inf[:, 4] + 2 * we, device=dev)
            ay = torch.arange(Hm, device=dev)
            ax = torch.arange(Wm, device=dev)
            rowm = (
                (ay[None, :] >= offy[:, None]) & (ay[None, :] < (offy + hy)[:, None])
            ).to(torch.float32)
            colm = (
                (ax[None, :] >= offx[:, None]) & (ax[None, :] < (offx + hx)[:, None])
            ).to(torch.float32)
            yy = oy[:, None] + ay[None, :]
            xx = ox[:, None] + ax[None, :]
            flat = ((yy + pad)[:, :, None] * Wp + (xx + pad)[:, None, :]).clamp_(
                0, total - 1
            )
            geom = Geom(
                flat,
                rowm[:, :, None] * colm[:, None, :],
                yy.to(torch.float32),
                xx.to(torch.float32),
            )
            out.append(
                BlobGroup(ids, torch.as_tensor(ids, device=dev), geom, h * geom.mask)
            )
        return out

    def _scatter(self, M: torch.Tensor, flat: torch.Tensor, vals: torch.Tensor) -> None:
        """M[flat] += vals.

        The windows of a batch are disjoint and the padding adds exact zeros, so the
        result does not depend on the order of the atomic adds.
        """
        M.index_add_(0, flat.reshape(-1), vals.reshape(-1))

    # -- batch kernels
    def _gather(self, geom: Geom) -> tuple[torch.Tensor, torch.Tensor]:
        fr = self.frame
        base = fr.base[geom.flat]
        vw = fr.valid[geom.flat].to(torch.float32) * geom.mask
        return base, vw

    def render(
        self, idx_t: torch.Tensor, sign: float = 1.0, g: torch.Tensor | None = None
    ) -> None:
        """M += sign * g * stamp for the given sources (a colour of disjoint
        windows)."""
        gg = self.g if g is None else g
        geom, h = self._points(idx_t)
        h *= (sign * gg[idx_t]).to(torch.float32)[:, None, None]
        self._scatter(self.frame.M, geom.flat, h)

    def update(
        self, idx_t: torch.Tensor, refine: bool, bg: BlobGroup | None = None
    ) -> None:
        """Refit gain / offset of a batch of sources against the current model and
        update the model.

        The model ``M`` of the frame is read and updated in place.
        """
        p = self.p
        fr = self.frame
        M = fr.M
        if bg is not None:
            geom, h = bg.geom, bg.h
        else:
            geom, h = self._points(idx_t)
        base, vw = self._gather(geom)
        gold = self.g[idx_t]
        res = base - M[geom.flat] + gold.to(torch.float32)[:, None, None] * h
        sig_eff = self.noise * p.tex
        gi, ci = solve_gc(
            res,
            h,
            vw,
            sig_eff,
            self.lam_g[idx_t],
            self.gp[idx_t],
            gold,
            self.c[idx_t],
            self.gmax[idx_t],
            p,
        )
        if refine:
            dy, dx = self._position_step(idx_t, res, vw, gi, ci, geom)
            moved = (dy != 0) | (dx != 0)
            if bool(moved.any()):
                # remove the old stamp, move the source, add the new one (CPU: Mw -=
                # gold*h; ...; M += gi*h)
                yc2 = self.yc[idx_t] + dy
                xc2 = self.xc[idx_t] + dx
                geom2, h2 = self._points(idx_t, yc2, xc2)
                m32 = moved.to(torch.float32)[:, None, None]
                self._scatter(
                    M, geom.flat, (-(gold.to(torch.float32)[:, None, None] * h) * m32)
                )
                self._scatter(
                    M, geom2.flat, ((gi.to(torch.float32)[:, None, None] * h2) * m32)
                )
                self.yc[idx_t] = torch.where(moved, yc2, self.yc[idx_t])
                self.xc[idx_t] = torch.where(moved, xc2, self.xc[idx_t])
                keep = (~moved).to(torch.float32)[:, None, None]
                self._scatter(
                    M,
                    geom.flat,
                    (((gi - gold).to(torch.float32)[:, None, None] * h) * keep),
                )
                self.g[idx_t], self.c[idx_t] = gi, ci
                return
        self._scatter(M, geom.flat, ((gi - gold).to(torch.float32)[:, None, None] * h))
        self.g[idx_t], self.c[idx_t] = gi, ci

    def _position_step(
        self,
        idx_t: torch.Tensor,
        res: torch.Tensor,
        vw: torch.Tensor,
        gi: torch.Tensor,
        ci: torch.Tensor,
        geom: Geom,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z0: one bounded Gauss-Newton step on the sub-pixel position (``_solve_gcp``),
        batched; zero where too faint."""
        p = self.p
        d = 0.25
        yc, xc = self.yc[idx_t], self.xc[idx_t]
        h0 = point_stamps(self.tm, self.a[idx_t], self.s[idx_t], yc, xc, geom)
        sig_eff = self.noise * p.tex

        def hh(dy: float, dx: float) -> torch.Tensor:
            return point_stamps(
                self.tm, self.a[idx_t], self.s[idx_t], yc + dy, xc + dx, geom
            )

        hy = (hh(d, 0.0) - hh(-d, 0.0)) / (2 * d)
        hx = (hh(0.0, d) - hh(0.0, -d)) / (2 * d)
        g32 = gi.to(torch.float32)[:, None, None]
        e = res - g32 * h0 - ci.to(torch.float32)[:, None, None]
        sig = torch.sqrt(sig_eff**2 + (p.rel_tol * g32 * h0) ** 2)
        w = vw / (e / (p.huber_k * sig)).clamp(min=1.0)
        c0, c1 = g32 * hy, g32 * hx
        A00 = (w * c0 * c0).sum((1, 2), dtype=torch.float64)
        A01 = (w * c0 * c1).sum((1, 2), dtype=torch.float64)
        A11 = (w * c1 * c1).sum((1, 2), dtype=torch.float64)
        b0 = (w * c0 * e).sum((1, 2), dtype=torch.float64)
        b1 = (w * c1 * e).sum((1, 2), dtype=torch.float64)
        reg = 1e-6 * (1.0 + A00 + A11)
        A00 = A00 + reg
        A11 = A11 + reg
        det = A00 * A11 - A01 * A01
        dy = ((b0 * A11 - A01 * b1) / det).clamp(-0.6, 0.6)
        dx = ((A00 * b1 - A01 * b0) / det).clamp(-0.6, 0.6)
        hmax = h0.reshape(h0.shape[0], -1).max(1).values.to(torch.float64)
        faint = gi * hmax < 20 * sig_eff
        dy = torch.where(faint, torch.zeros_like(dy), dy)
        dx = torch.where(faint, torch.zeros_like(dx), dx)
        return dy, dx

    # -- colours
    def colours(self, members: np.ndarray) -> list[np.ndarray]:
        """Groups of sources with pairwise disjoint windows (see module docstring), in
        update order."""
        if members.size == 0:
            return []
        if self.opts.mode == "seq":
            o = members[np.argsort(-self.src.A0[members], kind="stable")]
            if self.opts.seed is not None:
                o = np.random.default_rng(self.opts.seed + self.k).permutation(o)
            return [np.array([i]) for i in o]
        y, x = self.src.y[members], self.src.x[members]
        if self.opts.anchor and float(self.s_h.max()) <= self.opts.s_cap:
            # cells on the ABSOLUTE grid with a size that does not depend on this
            # window's own dilation field, so that a source gets the same colour (and
            # update rank) in every window that contains its neighbourhood
            D = (
                2 * int(np.ceil(self.tmpl.r_sup * self.opts.s_cap + self.p.win_extra))
                + 7
            )
            y, x = y + self.opts.origin_yx[0], x + self.opts.origin_yx[1]
        else:
            D = self.Hb + 4  # side of the common window size + margin
        return [members[g] for g in grid_colours(y, x, self.src.A0[members], D)]

    def chunks(self, idx: np.ndarray) -> list[torch.Tensor]:
        per = max(1, self.opts.px_budget // (self.Hb * self.Hb))
        return [
            torch.as_tensor(idx[i : i + per], device=self.dev)
            for i in range(0, len(idx), per)
        ]

    def _run_triton(self, act: np.ndarray, refine_pos: bool) -> None:
        """Same as ``run`` for point sources with the fused kernels of ``tri.py`` (blobs
        use the eager path)."""
        from . import tri

        p, dev = self.p, self.dev
        tab, ds, n_tab = self.tm.vals.contiguous(), self.tm.ds, self.tm.n
        r_sup, we = float(self.tmpl.r_sup), float(p.win_extra)
        sig_eff = self.noise * p.tex
        fr = self.frame

        def batch(members: np.ndarray) -> dict[str, torch.Tensor]:
            ii = torch.as_tensor(members, device=dev)
            return dict(
                idx=ii,
                yc=self.yc[ii].clone(),
                xc=self.xc[ii].clone(),
                a=self.a[ii].contiguous(),
                s=self.s[ii].contiguous(),
                lam_g=self.lam_g[ii].contiguous(),
                gp=self.gp[ii].contiguous(),
                gmax=self.gmax[ii].contiguous(),
                g=self.g[ii].clone(),
                c=self.c[ii].clone(),
            )

        pts = np.where(self.use_h & ~self.blob_h)[0]
        for grp in self.colours(pts):
            bt = batch(grp)
            tri.add_stamps(
                fr,
                tab,
                ds,
                n_tab,
                r_sup,
                we,
                self.Hb,
                bt["yc"],
                bt["xc"],
                bt["a"],
                bt["s"],
                bt["g"],
            )
        self._render_blobs()
        act_blob = np.where(act & self.blob_h)[0]
        act_pts = np.where(act & ~self.blob_h)[0]
        steps: list[tuple[str, Any]]
        if self.opts.mode == "seq":
            # exact sequential Gauss-Seidel in brightness order (CPU order), one source
            # per step: validation mode
            order = [i for i in np.argsort(-self.src.A0, kind="stable") if act[i]]
            if self.opts.seed is not None:
                order = list(
                    np.random.default_rng(self.opts.seed + self.k).permutation(order)
                )
            bgs = {int(g.idx_np[0]): g for g in self._blob_groups(act_blob, seq=True)}
            steps = [
                ("b", bgs[int(i)]) if self.blob_h[i] else ("p", batch(np.array([i])))
                for i in order
            ]
        else:
            steps = [("b", g) for g in self._blob_groups(act_blob)] + [
                ("p", batch(g)) for g in self.colours(act_pts)
            ]
        batches = [st[1] for st in steps if st[0] == "p"]
        for sw in range(p.n_sweeps):
            refine = refine_pos and sw >= 1
            for kind, bt in steps:
                if kind == "b":
                    self.update(bt.idx, False, bg=bt)
                    continue
                gold = bt["g"]
                gi, ci = tri.fit_batch(
                    fr,
                    tab,
                    ds,
                    n_tab,
                    r_sup,
                    we,
                    self.Hb,
                    bt["yc"],
                    bt["xc"],
                    bt["a"],
                    bt["s"],
                    gold,
                    bt["c"],
                    bt["gp"],
                    bt["lam_g"],
                    bt["gmax"],
                    sig_eff,
                    float(p.rel_tol),
                    float(p.huber_k),
                )
                if refine:
                    dy, dx = tri.position_step(
                        fr,
                        tab,
                        ds,
                        n_tab,
                        r_sup,
                        we,
                        self.Hb,
                        bt["yc"],
                        bt["xc"],
                        bt["a"],
                        bt["s"],
                        gold,
                        gi,
                        ci,
                        sig_eff,
                        float(p.rel_tol),
                        float(p.huber_k),
                    )
                    moved = (dy != 0) | (dx != 0)
                    tri.add_stamps(
                        fr,
                        tab,
                        ds,
                        n_tab,
                        r_sup,
                        we,
                        self.Hb,
                        bt["yc"],
                        bt["xc"],
                        bt["a"],
                        bt["s"],
                        torch.where(moved, -gold, gi - gold),
                    )
                    bt["yc"] = torch.where(moved, bt["yc"] + dy, bt["yc"])
                    bt["xc"] = torch.where(moved, bt["xc"] + dx, bt["xc"])
                    tri.add_stamps(
                        fr,
                        tab,
                        ds,
                        n_tab,
                        r_sup,
                        we,
                        self.Hb,
                        bt["yc"],
                        bt["xc"],
                        bt["a"],
                        bt["s"],
                        torch.where(moved, gi, torch.zeros_like(gi)),
                    )
                else:
                    tri.add_stamps(
                        fr,
                        tab,
                        ds,
                        n_tab,
                        r_sup,
                        we,
                        self.Hb,
                        bt["yc"],
                        bt["xc"],
                        bt["a"],
                        bt["s"],
                        gi - gold,
                    )
                bt["g"], bt["c"] = gi, ci
        for bt in batches:
            self.g[bt["idx"]], self.c[bt["idx"]] = bt["g"], bt["c"]
            self.yc[bt["idx"]], self.xc[bt["idx"]] = bt["yc"], bt["xc"]

    def run(self, act: np.ndarray, refine_pos: bool) -> None:
        """Initial render of every ``use`` source, then ``n_sweeps`` of back-fitting of
        the ``act`` sources."""
        p = self.p
        if self.opts.triton:
            return self._run_triton(act, refine_pos)
        pts = np.where(self.use_h & ~self.blob_h)[0]
        for grp in self.colours(pts):
            for ch in self.chunks(grp):
                self.render(ch)
        self._render_blobs()
        act_blob = np.where(act & self.blob_h)[0]
        act_pts = np.where(act & ~self.blob_h)[0]
        groups = self.colours(act_pts)
        blob_groups = self._blob_groups(act_blob, seq=self.opts.mode == "seq")
        for sw in range(p.n_sweeps):
            refine = refine_pos and sw >= 1
            for bg in blob_groups:
                self.update(bg.idx, False, bg=bg)
            for grp in groups:
                for ch in self.chunks(grp):
                    self.update(ch, refine)

    def _render_blobs(self, js: np.ndarray | None = None, sign: float = 1.0) -> None:
        """M += sign * g * stamp for blobs (default: every blob in ``use``)."""
        if js is None:
            js = np.where(self.use_h & self.blob_h)[0]
        for bg in self._blob_groups(js):
            self._scatter(
                self.frame.M,
                bg.geom.flat,
                bg.h * (sign * self.g[bg.idx]).to(torch.float32)[:, None, None],
            )

    def remove(self, idx: np.ndarray) -> None:
        """Subtract the model of the given (point) sources from M (pruning)."""
        pts = np.asarray([i for i in idx if not self.blob_h[i]], np.int64)
        for grp in self.colours(pts):
            for ch in self.chunks(grp):
                self.render(ch, sign=-1.0)
        self._render_blobs(
            np.asarray([i for i in idx if self.blob_h[i]], np.int64), sign=-1.0
        )


def g_peak(tmpl: Template) -> float:
    return float(np.max(np.abs(tmpl.vals)))


def fit_plane(
    img: torch.Tensor,
    B: torch.Tensor,
    src: Sources,
    k: int,
    p: PsfSubtractParams,
    tmpl: Template,
    *,
    use: np.ndarray,
    noise: float,
    s: np.ndarray | None = None,
    refine_pos: bool = False,
    gp_arr: np.ndarray | None = None,
    opts: FitOpts | None = None,
) -> PlaneResult:
    """GPU ``fit_plane`` with a fixed library template (production mode).

    Args:
        img, B: Plane and lower envelope (device float32 [Y, X]); 0 = invalid pixel.
        src: Sources (host); positions are updated in place when ``refine_pos``.
        k: Plane index.
        p: Parameters.
        tmpl: Library template of plane k.
        use: Boolean mask of the sources with fitted gains.
        noise: Plane noise.
        s: Per-source radius dilation (default 1).
        refine_pos: z0 only: refine sub-pixel positions in the last two sweeps.
        gp_arr: Per-source gain prior for planes >= 1 (consensus gain).
        opts: Execution options.

    Returns:
        PlaneResult (gains ``g``, offsets ``c`` on the host; rendered model on the
        device).
    """
    opts = opts or FitOpts()
    N = src.n
    if k == 0 and p.rel_tol0 != p.rel_tol:
        import dataclasses

        p = dataclasses.replace(p, rel_tol=p.rel_tol0)
    s = np.ones(N) if s is None else np.asarray(s, float)
    fitter = PlaneFitter(img, B, src, k, p, tmpl, s, use, noise, gp_arr, opts)
    if k >= 2:
        act = use & ((src.a * g_peak(tmpl) * p.g_max) >= p.active_snr * noise * p.tex)
    else:
        act = use.copy()
    fitter.run(act, refine_pos)
    g = fitter.g.cpu().numpy()
    c = fitter.c.cpu().numpy()
    if refine_pos:
        src.y = fitter.yc.cpu().numpy()
        src.x = fitter.xc.cpu().numpy()
    return PlaneResult(tmpl, g, c, s, fitter.frame.crop_model(), float(noise))
