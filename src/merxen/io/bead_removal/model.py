"""Forced-photometry bead model of one reference-stain window.

Empirical radial PSF templates, per-bead per-plane gains fitted by robust
Gauss-Seidel back-fitting, Richardson-Lucy source maps for extended aggregates
and a smooth stretch field for the within-field-of-view change of the PSF
radius. There are two entry points:

* :func:`fit_window_library` is the production fit: the PSF library and every
  dataset statistic come from the dataset calibration.
* :func:`fit_tile_local` is the calibration fit: the templates are stacked from
  the off-tissue tile itself.

The model of plane k is::

    y_k = B_k + tissue + sum_j g_jk * [ u_j (*) H_k(. / s_jk) / s_jk^2 ] + noise

where B_k is the lower envelope, g_jk the gain of source j, u_j its z0 amplitude
map (a delta for a point source), H_k the template of plane k and s_jk the
radius dilation of source j on plane k.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from merxen.io.bead_removal.detect import (
    ClassifierThresholds,
    PlaneStack,
    Sources,
    detect_sources,
    huber1,
    lower_envelope,
    radius_grid,
    robust_noise,
    window_bounds,
)
from merxen.io.bead_removal.params import (
    BETA,
    DEFAULT_PARAMS,
    FLUX_TARGET,
    PsfSubtractParams,
)
from merxen.io.bead_removal.psf import (
    PsfLibrary,
    Template,
    derive_high_templates,
    kern2d,
    ring_prominence,
)

logger = logging.getLogger(__name__)

# Cache of the full (unclipped) model image of one blob source: window origin
# (y0, x0) and the convolved image.
_BlobCache = dict[int, tuple[int, int, np.ndarray]]

# Grid of the radius dilations searched per source by the stretch estimate.
_GRID = np.exp(np.linspace(np.log(0.80), np.log(1.27), 13))


class WindowCalibration(Protocol):
    """What the production fit needs from a dataset calibration.

    The calibration module provides the concrete class; the fit only reads
    these attributes, so any object that has them can be used (the unit tests
    use a synthetic one).
    """

    @property
    def library(self) -> PsfLibrary:
        """PSF library with one template per plane z0..z7."""

    @property
    def thresholds(self) -> ClassifierThresholds:
        """Per-dataset classifier thresholds."""

    @property
    def noise(self) -> Mapping[int, float]:
        """Pixel noise sd of the reference stain per plane (counts)."""

    @property
    def dog_noise_z0(self) -> float:
        """Dataset-global DoG noise of z0 (the detection threshold unit)."""

    @property
    def a0_sel(self) -> tuple[float, float]:
        """Range of z0 amplitudes of the compact beads that define the fit."""

    @property
    def a0_median(self) -> float:
        """Dataset-global median bead amplitude (blob candidacy scale)."""

    @property
    def ell_default(self) -> float:
        """Log-stretch used where the stretch field has too few samples."""


# ----------------------------------------------------------------------------
# stamps (unit-gain model image of one source on one plane)
def _blob_full(
    src: Sources, j: int, tmpl: Template, s: float
) -> tuple[int, int, np.ndarray]:
    """Convolve the amplitude map of blob ``j`` with the dilated template.

    Returns:
        (row, column) of the image origin in the window, and the full image.
    """
    uy0, ux0, amp_map = src.u[j]
    kernel, reach = kern2d(tmpl, s)
    full = fftconvolve(amp_map, kernel, mode="full").astype(np.float32)
    return uy0 - reach, ux0 - reach, full


def _stamp(
    src: Sources,
    j: int,
    tmpl: Template,
    s: float,
    shape: tuple[int, int],
    p: PsfSubtractParams,
    cache: _BlobCache,
) -> tuple[int, int, int, int, np.ndarray]:
    """Render the unit-gain model of source ``j`` on an image window.

    The window extends ``win_extra`` px beyond the template support so that the
    local offset is constrained.

    Args:
        src: Sources.
        j: Source index.
        tmpl: Template of the plane.
        s: Radius dilation of the source on this plane.
        shape: Shape of the image the stamp is cut for.
        p: Fixed parameters.
        cache: Blob images already rendered (filled for blobs).

    Returns:
        (y0, y1, x0, x1, h): the window bounds and the model image on it.
    """
    if src.blob[j]:
        if j not in cache:
            cache[j] = _blob_full(src, j, tmpl, s)
        fy0, fx0, full = cache[j]
        full_h, full_w = full.shape
        height, width = shape
        y0 = max(0, fy0 - p.win_extra)
        y1 = min(height, fy0 + full_h + p.win_extra)
        x0 = max(0, fx0 - p.win_extra)
        x1 = min(width, fx0 + full_w + p.win_extra)
        h = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), np.float32)
        ya, yb = max(y0, fy0), min(y1, fy0 + full_h)
        xa, xb = max(x0, fx0), min(x1, fx0 + full_w)
        if yb > ya and xb > xa:
            h[ya - y0 : yb - y0, xa - x0 : xb - x0] = full[
                ya - fy0 : yb - fy0, xa - fx0 : xb - fx0
            ]
        return y0, y1, x0, x1, h
    reach = tmpl.r_sup * s + p.win_extra
    y0, y1, x0, x1 = window_bounds(shape, src.y[j], src.x[j], reach)
    r = radius_grid(src.y[j], src.x[j], y0, y1, x0, x1)
    return y0, y1, x0, x1, (src.a[j] * tmpl(r, s)).astype(np.float32)


@dataclass
class PlaneFit:
    """Result of the fit of one plane.

    Attributes:
        tmpl: Template of the plane (library or stacked).
        g: Gain per source.
        c: Local offset per source.
        s: Radius dilation per source.
        model: Rendered bead light on the window (sum of gain x unit stamps).
        noise: Pixel noise sd used for the plane.
    """

    tmpl: Template
    g: np.ndarray
    c: np.ndarray
    s: np.ndarray
    model: np.ndarray
    noise: float


def _solve_gc(
    res: np.ndarray,
    h: np.ndarray,
    valid: np.ndarray,
    sig_eff: float,
    sg: float,
    g0: float,
    gp: float,
    p: PsfSubtractParams,
    ci0: float = 0.0,
    gmax: float | None = None,
) -> tuple[float, float]:
    """Fit the gain and offset of one stamp to the residual image.

    The fit is a robust (one-sided Huber) ridge regression.

    Args:
        res: Residual image with the source's own light still in it.
        h: Unit-gain stamp.
        valid: 1.0 on valid pixels, 0.0 elsewhere.
        sig_eff: Effective pixel noise (noise x texture factor).
        sg: Prior sd of the gain.
        g0: Starting gain.
        gp: Prior mean of the gain.
        p: Fixed parameters.
        ci0: Starting offset.
        gmax: Upper bound of the gain (default ``p.g_max``).

    Returns:
        (gain, offset).
    """
    gmax = p.g_max if gmax is None else gmax
    gi, ci = g0, ci0
    lam_g = sig_eff**2 / sg**2
    lam_c = 1.0 / 9.0
    for _ in range(3):
        e = res - gi * h - ci
        w = valid * huber1(
            e, np.sqrt(sig_eff**2 + (p.rel_tol * gi * h) ** 2), p.huber_k
        )
        s_hh = float(np.sum(w * h * h))
        s_h1 = float(np.sum(w * h))
        s_11 = float(np.sum(w))
        s_hr = float(np.sum(w * h * res))
        s_1r = float(np.sum(w * res))
        normal = np.array(
            [[s_hh + lam_g, s_h1], [s_h1, s_11 + lam_c]], dtype=np.float64
        )
        rhs = np.array([s_hr + lam_g * gp, s_1r], dtype=np.float64)
        gi, ci = np.linalg.solve(normal, rhs)
        gi = float(np.clip(gi, 0.0, gmax))
    return gi, float(ci)


def _stack_template(
    img: np.ndarray,
    env: np.ndarray,
    rendered: np.ndarray,
    src: Sources,
    g: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    tmpl_prev: Template | None,
    k: int,
    noise: float,
    p: PsfSubtractParams,
    use: np.ndarray,
    flux0: float | None = None,
) -> Template:
    """Stack a radial template from aligned, scale-normalised source stamps.

    The stamps are those of the selected compact singles (weights ~ amplitude^2)
    with their neighbours removed through the current model. Calibration (local
    mode) only.

    Args:
        img: Plane image.
        env: Lower envelope of the plane.
        rendered: Current model image of the plane.
        src: Sources.
        g: Gains per source.
        c: Offsets per source.
        s: Radius dilations per source.
        tmpl_prev: Template of the previous outer iteration (None on the first).
        k: Plane index.
        noise: Pixel noise sd of the plane.
        p: Fixed parameters.
        use: Sources that are stacked.
        flux0: z0 template flux (flux cap of the planes >= 3).

    Returns:
        The stacked, smoothed and tapered template.
    """
    ds = p.ds
    r_sup = p.r_sup[k]
    n = int(np.ceil(r_sup / ds)) + 1
    num = np.zeros(n + 2)
    den = np.zeros(n + 2)
    r_win = r_sup + p.win_extra
    base = img - env
    for i in np.where(use)[0]:
        yc, xc, a0 = src.y[i], src.x[i], src.a[i]
        si = s[i]
        y0, y1, x0, x1 = window_bounds(img.shape, yc, xc, r_win * si)
        r = radius_grid(yc, xc, y0, y1, x0, x1)
        v = base[y0:y1, x0:x1] - rendered[y0:y1, x0:x1]
        if a0 <= 0:
            continue
        ge = float(np.clip(g[i], 0.4, 2.5))
        amp = a0 * ge
        if tmpl_prev is not None:
            v = v + a0 * g[i] * tmpl_prev(r, si)
            ci = c[i]
        else:
            ann = (r >= r_sup * si) & (r <= r_win * si)
            ci = np.median(v[ann]) if ann.any() else 0.0
        v = v - ci
        valid = img[y0:y1, x0:x1] > 0
        w = valid.astype(np.float32) * (amp * amp)
        if tmpl_prev is not None:
            mp = amp * tmpl_prev(r, si)
            w = w * huber1(
                v - mp,
                np.sqrt((noise * p.tex) ** 2 + (p.rel_tol * mp) ** 2),
                p.huber_k,
            )
        rr = r / si
        inside = rr < r_sup
        u = v[inside] * si * si / amp
        ww = w[inside]
        j = rr[inside] / ds
        j0 = np.floor(j).astype(np.int64)
        f = j - j0
        num += np.bincount(j0, ww * (1 - f) * u, minlength=n + 2)[: n + 2]
        num += np.bincount(j0 + 1, ww * f * u, minlength=n + 2)[: n + 2]
        den += np.bincount(j0, ww * (1 - f), minlength=n + 2)[: n + 2]
        den += np.bincount(j0 + 1, ww * f, minlength=n + 2)[: n + 2]
    vals = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)[:n]
    smooth = p.smooth_px[k] / ds
    if smooth > 0:
        vals = gaussian_filter1d(vals, smooth, mode="nearest")
    nodes = np.arange(n) * ds
    taper = np.clip((r_sup - nodes) / (r_sup * 0.12), 0, 1)
    vals = vals * (0.5 - 0.5 * np.cos(np.pi * taper))
    tmpl = Template(ds, r_sup, vals)
    if k >= 3 and flux0 is not None:
        # Planes >= 3 are faint; their stacked flux must not exceed the flux the
        # conservation law allows.
        cap = FLUX_TARGET[k] * flux0 * p.flux_tol
        flux = tmpl.flux(1e9)
        if flux > cap:
            tmpl.vals = tmpl.vals * (cap / flux)
    return tmpl


def _g_peak(tmpl: Template) -> float:
    """Return the largest absolute value of a template."""
    return float(np.max(np.abs(tmpl.vals)))


def _render_all(
    rendered: np.ndarray,
    src: Sources,
    tmpl: Template,
    s: np.ndarray,
    g: np.ndarray,
    use: np.ndarray,
    p: PsfSubtractParams,
    cache: _BlobCache,
) -> None:
    """Render the model of all used sources into ``rendered`` (in place)."""
    rendered[:] = 0
    for j in np.where(use)[0]:
        if g[j] == 0:
            continue
        y0, y1, x0, x1, h = _stamp(src, j, tmpl, s[j], rendered.shape, p, cache)
        rendered[y0:y1, x0:x1] += g[j] * h


def _solve_gcp(
    res: np.ndarray,
    src: Sources,
    i: int,
    tmpl: Template,
    s: float,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    valid: np.ndarray,
    sig_eff: float,
    sg: float,
    g0: float,
    gp: float,
    p: PsfSubtractParams,
    ci0: float,
) -> tuple[float, float, float, float]:
    """Fit gain and offset, then take one bounded Gauss-Newton step on position.

    Used on z0 only.

    Returns:
        (gain, offset, row shift, column shift) with shifts clipped to +-0.6 px.
    """
    step = 0.25
    yc, xc, a0 = src.y[i], src.x[i], src.a[i]

    def stamp_at(dy: float, dx: float) -> np.ndarray:
        r = radius_grid(yc + dy, xc + dx, y0, y1, x0, x1)
        stamp: np.ndarray = (a0 * tmpl(r, s)).astype(np.float32)
        return stamp

    h = stamp_at(0, 0)
    gi, ci = _solve_gc(res, h, valid, sig_eff, sg, g0, gp, p, ci0)
    if gi * float(h.max()) < 20 * sig_eff:  # too faint for a measurable shift
        return gi, ci, 0.0, 0.0
    hy = (stamp_at(step, 0) - stamp_at(-step, 0)) / (2 * step)
    hx = (stamp_at(0, step) - stamp_at(0, -step)) / (2 * step)
    e = res - gi * h - ci
    w = valid * huber1(e, np.sqrt(sig_eff**2 + (p.rel_tol * gi * h) ** 2), p.huber_k)
    cols = [gi * hy, gi * hx]
    normal = np.array(
        [[np.sum(w * cols[a] * cols[b]) for b in range(2)] for a in range(2)],
        dtype=np.float64,
    )
    normal += np.eye(2) * (1e-6 * (1.0 + normal[0, 0] + normal[1, 1]))
    rhs = np.array([np.sum(w * cols[a] * e) for a in range(2)], dtype=np.float64)
    dy, dx = np.linalg.solve(normal, rhs)
    return gi, ci, float(np.clip(dy, -0.6, 0.6)), float(np.clip(dx, -0.6, 0.6))


def _sweeps(
    img: np.ndarray,
    env: np.ndarray,
    rendered: np.ndarray,
    src: Sources,
    g: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    tmpl: Template,
    k: int,
    noise: float,
    p: PsfSubtractParams,
    use: np.ndarray,
    sweeps: int,
    cache: _BlobCache,
    refine_pos: bool = False,
    gp_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Seidel back-fitting.

    Every source is refitted (brightest first) with its neighbours removed
    through the rendered model, which is updated in place.

    Args:
        img: Plane image.
        env: Lower envelope of the plane.
        rendered: Current model image (updated in place).
        src: Sources (z0 positions are refined in place when ``refine_pos``).
        g: Gains per source (updated in place).
        c: Offsets per source (updated in place).
        s: Radius dilations per source.
        tmpl: Template of the plane.
        k: Plane index.
        noise: Pixel noise sd of the plane.
        p: Fixed parameters.
        use: Sources that are fitted.
        sweeps: Number of sweeps.
        cache: Blob images already rendered.
        refine_pos: z0 only: refine sub-pixel positions from the second sweep.
        gp_arr: Per-source gain prior for planes >= 1 (consensus gain).

    Returns:
        (gains, offsets).
    """
    sig_eff = noise * p.tex
    sg = p.g_prior_sd[k]
    order = np.argsort(-src.A0, kind="stable")
    base = img - env
    for sweep in range(sweeps):
        for i in order:
            if not use[i]:
                continue
            y0, y1, x0, x1, h = _stamp(src, i, tmpl, s[i], img.shape, p, cache)
            g_old = g[i]
            gp = p.g_prior if (gp_arr is None or k == 0) else float(gp_arr[i])
            window = rendered[y0:y1, x0:x1]
            res = base[y0:y1, x0:x1] - window + g_old * h
            valid = (img[y0:y1, x0:x1] > 0).astype(np.float32)
            if refine_pos and not src.blob[i] and sweep >= 1:
                gi, ci, dy, dx = _solve_gcp(
                    res,
                    src,
                    i,
                    tmpl,
                    s[i],
                    y0,
                    y1,
                    x0,
                    x1,
                    valid,
                    sig_eff,
                    sg,
                    g_old,
                    gp,
                    p,
                    c[i],
                )
                if dy != 0.0 or dx != 0.0:
                    # Remove the old stamp, move the source, add the new one.
                    window -= g_old * h
                    src.y[i] += dy
                    src.x[i] += dx
                    ny0, ny1, nx0, nx1, h = _stamp(
                        src, i, tmpl, s[i], img.shape, p, cache
                    )
                    rendered[ny0:ny1, nx0:nx1] += gi * h
                    g[i], c[i] = gi, ci
                    continue
            else:
                gi, ci = _solve_gc(
                    res,
                    h,
                    valid,
                    sig_eff,
                    p.blob_g_sd if (src.blob[i] and k > 0) else sg,
                    g_old,
                    gp,
                    p,
                    c[i],
                    p.blob_g_max if src.blob[i] else None,
                )
            window += (gi - g_old) * h
            g[i], c[i] = gi, ci
    return g, c


def fit_plane(
    img: np.ndarray,
    env: np.ndarray,
    src: Sources,
    k: int,
    p: PsfSubtractParams = DEFAULT_PARAMS,
    tmpl0: Template | None = None,
    s: np.ndarray | None = None,
    use: np.ndarray | None = None,
    sel: np.ndarray | None = None,
    noise: float | None = None,
    refine_pos: bool = False,
    fixed_template: Template | None = None,
    gp_arr: np.ndarray | None = None,
    flux0: float | None = None,
) -> PlaneFit:
    """Fit the template and gains of one plane.

    Args:
        img: 2-D plane (0 = invalid pixel).
        env: Lower envelope (smooth background) of the plane.
        src: Sources (positions and amplitudes at z0).
        k: Plane index.
        p: Fixed parameters.
        tmpl0: Starting template when stacking (calibration).
        s: Per-source radius dilation (default 1).
        use: Sources with fitted gains.
        sel: Sources stacked into the template (calibration only).
        noise: Pixel noise of the plane (global in production).
        refine_pos: z0 only: refine sub-pixel positions in the last sweeps.
        fixed_template: Library template; None stacks the template from ``sel``
            (calibration).
        gp_arr: Per-source gain prior for planes >= 1 (consensus gain).
        flux0: z0 template flux (flux cap of the stacked planes >= 3).

    Returns:
        The plane fit with the rendered model (sum of gain x unit stamps).
    """
    n_src = src.n
    if k == 0 and p.rel_tol0 != p.rel_tol:
        p = dataclasses.replace(p, rel_tol=p.rel_tol0)
    if noise is None:
        noise = robust_noise(img, img > 0)
    g = (
        np.full(n_src, p.g_prior if k > 0 else 1.0)
        if gp_arr is None
        else np.array(gp_arr, float)
    )
    c = np.zeros(n_src)
    s = np.ones(n_src) if s is None else s
    use = np.ones(n_src, bool) if use is None else use
    sel = use & ~src.blob if sel is None else sel
    if k >= 2 and sel.sum() > p.max_sel_hi:
        top = np.sort(src.a[sel])[-p.max_sel_hi]
        sel = sel & (src.a >= top)
    rendered = np.zeros(img.shape, np.float32)
    tmpl = fixed_template if fixed_template is not None else tmpl0
    cache: _BlobCache = {}
    n_outer = 1 if fixed_template is not None else p.n_outer
    for it in range(n_outer):
        if fixed_template is None:
            tmpl = _stack_template(
                img, env, rendered, src, g, c, s, tmpl, k, noise, p, sel, flux0
            )
        assert tmpl is not None
        cache.clear()
        # Sources whose rings stay far below the noise are rendered with the
        # prior gain, not fitted.
        active = (
            use & ((src.a * _g_peak(tmpl) * p.g_max) >= p.active_snr * noise * p.tex)
            if k >= 2
            else use
        )
        _render_all(rendered, src, tmpl, s, g, use, p, cache)
        g, c = _sweeps(
            img,
            env,
            rendered,
            src,
            g,
            c,
            s,
            tmpl,
            k,
            noise,
            p,
            active,
            p.n_sweeps,
            cache,
            refine_pos=(refine_pos and it == n_outer - 1),
            gp_arr=gp_arr,
        )
        if fixed_template is None and k == 0:
            bright = (
                sel & (src.a >= np.percentile(src.a[sel], 40)) if sel.any() else sel
            )
            gain_med = float(np.median(g[bright])) if bright.any() else 1.0
            if gain_med > 1e-3:
                tmpl.vals = tmpl.vals * gain_med
                g = g / gain_med
                rendered *= gain_med
                for j in list(cache):
                    cache[j] = (cache[j][0], cache[j][1], cache[j][2] * gain_med)
        logger.debug(
            "[BEADS] plane %d outer %d: median g %.3f",
            k,
            it,
            float(np.median(g[use])) if use.any() else 0.0,
        )
    assert tmpl is not None
    return PlaneFit(tmpl, g, c, s, rendered, noise)


# ----------------------------------------------------------------------------
# blobs (extended aggregates): Richardson-Lucy source maps
def _richardson_lucy(img: np.ndarray, kernel: np.ndarray, iters: int) -> np.ndarray:
    """Deconvolve an image window with a unit-sum kernel (Richardson-Lucy).

    Args:
        img: Image window, clipped to >= 0.
        kernel: Unit-sum point-spread kernel.
        iters: Number of iterations.

    Returns:
        The estimated source map.
    """
    estimate = np.full(img.shape, max(float(img.mean()), 1.0), np.float64)
    flipped = kernel[::-1, ::-1]
    for _ in range(iters):
        blurred = fftconvolve(estimate, kernel, mode="same") + 1e-3
        estimate = estimate * fftconvolve(
            np.clip(img / blurred, 0, 5.0), flipped, mode="same"
        )
    return estimate


def build_blobs(
    img0: np.ndarray,
    env0: np.ndarray,
    src: Sources,
    f0: PlaneFit,
    p: PsfSubtractParams,
    a0_median: float,
) -> np.ndarray:
    """Replace point sources by RL source maps for extended sources.

    A source is extended when its aperture flux / point-source flux ratio ``q``
    exceeds ``blob_q``.

    Args:
        img0: The z0 plane.
        env0: Lower envelope of the z0 plane.
        src: Sources; updated in place (``u``, ``a``, ``blob``, ``q``).
        f0: z0 point-source fit.
        p: Fixed parameters.
        a0_median: Dataset-global median bead amplitude. The blob candidacy
            threshold is ``blob_a_rel`` x this; the prototype used the window
            median, which made the decision depend on the chunk.

    Returns:
        Indices of the blobs.
    """
    t0 = f0.tmpl
    base = img0 - env0
    phi_ap = t0.flux(p.r_ap)
    q = np.zeros(src.n)
    snr_flux = np.zeros(src.n)
    for i in np.where(src.is_bead)[0]:
        y0, y1, x0, x1, h = _stamp(src, i, t0, f0.s[i], img0.shape, p, {})
        r = radius_grid(src.y[i], src.x[i], y0, y1, x0, x1)
        aperture = r <= p.r_ap
        own = f0.g[i] * h
        res = base[y0:y1, x0:x1] - (f0.model[y0:y1, x0:x1] - own) - f0.c[i]
        flux = float(res[aperture].sum())
        q[i] = flux / max(src.A0[i] * phi_ap, 1e-6)
        snr_flux[i] = flux / (f0.noise * np.sqrt(max(aperture.sum(), 1)))
    src.q = q
    cand = (
        src.is_bead
        & (q > p.blob_q)
        & (snr_flux > p.blob_snr)
        & (p.blob_a_rel * a0_median <= src.A0)
    )
    k0, _ = kern2d(t0, 1.0)
    phi = float(k0.sum())
    kernel_norm = (k0 / phi).astype(np.float64)
    model0 = f0.model.copy()
    for j in [j for j in np.argsort(-src.A0, kind="stable") if cand[j]]:
        half = p.blob_half
        y0, y1, x0, x1 = window_bounds(img0.shape, src.y[j], src.x[j], half)
        # Remove this source's point model from the model (blob flag is still
        # False).
        sy0, sy1, sx0, sx1, point = _stamp(src, j, t0, f0.s[j], img0.shape, p, {})
        model0[sy0:sy1, sx0:sx1] -= f0.g[j] * point
        window = (base[y0:y1, x0:x1] - model0[y0:y1, x0:x1] - f0.c[j]).astype(
            np.float64
        )
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(yy - src.y[j], xx - src.x[j])
        annulus = (rr >= half - 15) & (rr <= half - 3)
        if annulus.any():
            # Local baseline (tissue) of the blob window.
            window = window - np.percentile(window[annulus], 30)
        clipped = np.clip(window, 0, None)
        # Amplitude density: conv(u, H_0) ~ window.
        amp_map = (_richardson_lucy(clipped, kernel_norm, p.rl_iter) / phi).astype(
            np.float32
        )
        amp_map[rr > half - 3] = 0
        src.u[j] = (y0, x0, amp_map)
        src.a[j] = float(amp_map.sum())
        src.blob[j] = True
        fy0, fx0, full = _blob_full(src, j, t0, 1.0)
        ya, xa = max(0, fy0), max(0, fx0)
        yb = min(model0.shape[0], fy0 + full.shape[0])
        xb = min(model0.shape[1], fx0 + full.shape[1])
        model0[ya:yb, xa:xb] += full[ya - fy0 : yb - fy0, xa - fx0 : xb - fx0]
    return np.where(src.blob)[0]


# ----------------------------------------------------------------------------
# stretch field
def _stretch_search(
    img: np.ndarray,
    env: np.ndarray,
    rendered: np.ndarray,
    src: Sources,
    g: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    tmpl: Template,
    noise: float,
    p: PsfSubtractParams,
    cand: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the template dilation of every candidate source on a grid.

    For each dilation the gain and offset are refitted and the weighted sum of
    squared errors is recorded; the optimum is refined by a parabola in log
    space.

    Args:
        img: Plane image.
        env: Lower envelope of the plane.
        rendered: Current model image of the plane.
        src: Sources.
        g: Gains per source.
        c: Offsets per source.
        s: Current radius dilations per source.
        tmpl: Template of the plane.
        noise: Pixel noise sd of the plane.
        p: Fixed parameters.
        cand: Sources that are searched.
        grid: Dilations to try (ascending).

    Returns:
        (new dilation per source, S/N of the template response per source).
    """
    sig_eff = noise * p.tex
    base = img - env
    n_src = src.n
    s_new = s.copy()
    snr = np.zeros(n_src)
    gmax = float(grid.max())
    for i in np.where(cand)[0]:
        yc, xc, a0 = src.y[i], src.x[i], src.a[i]
        reach = (tmpl.r_sup + p.win_extra) * gmax
        y0, y1, x0, x1 = window_bounds(img.shape, yc, xc, reach)
        r = radius_grid(yc, xc, y0, y1, x0, x1)
        own = a0 * g[i] * tmpl(r, s[i])
        res = base[y0:y1, x0:x1] - rendered[y0:y1, x0:x1] + own
        valid = (img[y0:y1, x0:x1] > 0).astype(np.float32)
        snr[i] = a0 * max(g[i], 0.3) * np.sqrt(np.sum(tmpl(r, 1.0) ** 2)) / sig_eff
        if snr[i] < p.stretch_snr:
            continue
        sse_list = []
        for scale in grid:
            h = (a0 * tmpl(r, scale)).astype(np.float32)
            gi, ci = max(g[i], 0.3), c[i]
            for _ in range(2):
                e = res - gi * h - ci
                w = valid * huber1(
                    e, np.sqrt(sig_eff**2 + (p.rel_tol * gi * h) ** 2), p.huber_k
                )
                s_hh, s_h1, s_11 = np.sum(w * h * h), np.sum(w * h), np.sum(w)
                s_hr, s_1r = np.sum(w * h * res), np.sum(w * res)
                gi, ci = np.linalg.solve(
                    np.array([[s_hh + 1e-9, s_h1], [s_h1, s_11 + 1e-9]]),
                    [s_hr, s_1r],
                )
                gi = float(np.clip(gi, 0.0, p.g_max))
            e = res - gi * h - ci
            sse_list.append(np.sum(w * e * e))
        sse = np.array(sse_list)
        best = int(np.argmin(sse))
        d_log = np.log(grid[1]) - np.log(grid[0])
        if 0 < best < len(grid) - 1:
            before = np.log(sse[best - 1])
            centre = np.log(sse[best])
            after = np.log(sse[best + 1])
            curvature = before - 2 * centre + after
            shift = 0.5 * (before - after) / curvature if curvature > 0 else 0.0
            log_s = np.log(grid[best]) + np.clip(shift, -1, 1) * d_log
        else:
            log_s = np.log(grid[best])
        s_new[i] = np.exp(log_s)
    return s_new, snr


def smooth_field(
    pts: np.ndarray,
    val: np.ndarray,
    wgt: np.ndarray,
    q: np.ndarray,
    sigma: float,
    kmin: int = 4,
) -> np.ndarray:
    """Evaluate a Gaussian-weighted median of ``val`` at the query points.

    The weighted median is robust against outliers.

    Args:
        pts: Sample positions [n, 2].
        val: Sample values [n].
        wgt: Sample weights [n].
        q: Query positions [m, 2].
        sigma: Gaussian scale in px.
        kmin: Minimum number of samples within 2.5 sigma of a query point.

    Returns:
        Array [m] with NaN where fewer than ``kmin`` samples lie within 2.5 sigma.
    """
    out = np.full(len(q), np.nan)
    if len(pts) == 0:
        return out
    tree = cKDTree(pts)
    for i in range(len(q)):
        idx = tree.query_ball_point(q[i], 2.5 * sigma)
        if len(idx) < kmin:
            continue
        idx = np.asarray(idx)
        d = np.linalg.norm(pts[idx] - q[i], axis=1)
        w = wgt[idx] * np.exp(-0.5 * (d / sigma) ** 2)
        v = val[idx]
        order = np.argsort(v)
        cum_w = np.cumsum(w[order])
        out[i] = v[order][min(np.searchsorted(cum_w, cum_w[-1] / 2), len(order) - 1)]
    return out


def top_per_block(
    score: np.ndarray,
    cand: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    origin_yx: tuple[int, int],
    block: int,
    n_keep: int,
) -> np.ndarray:
    """Keep, per absolute block, the candidates with the highest score.

    The block index is floor(absolute coordinate / block) with absolute = window
    coordinate + origin, so the choice does not depend on where a chunk window
    starts. Ties are broken by the smaller y, then the smaller x.

    Args:
        score: Per-source score (higher wins).
        cand: Boolean candidate mask.
        y: Source rows in window coordinates.
        x: Source columns in window coordinates.
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0).
        block: Block size in px.
        n_keep: Candidates kept per block.

    Returns:
        Boolean mask (a subset of ``cand``).
    """
    idx = np.where(cand)[0]
    out = np.zeros_like(cand)
    if idx.size == 0:
        return out
    by = np.floor((y[idx] + origin_yx[0]) / block).astype(np.int64) + 1_000_000
    bx = np.floor((x[idx] + origin_yx[1]) / block).astype(np.int64) + 1_000_000
    order = np.lexsort((x[idx], y[idx], -score[idx], bx, by))  # last key sorts first
    gid = by[order] * 4_000_000 + bx[order]
    start = np.r_[True, gid[1:] != gid[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - first
    out[idx[order[rank < n_keep]]] = True
    return out


def estimate_stretch_field(
    imgs: dict[int, np.ndarray],
    envs: dict[int, np.ndarray],
    src: Sources,
    fits0: dict[int, PlaneFit],
    p: PsfSubtractParams,
    *,
    origin_yx: tuple[int, int],
    ell_default: float,
    ks: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, int, float]:
    """Estimate the smooth log-stretch field ell at every source.

    The field is the within-field-of-view tilt or bowl of the PSF radius.
    Candidates are compact singles with S/N >= ``stretch_snr``; per plane at most
    ``stretch_per_block`` per absolute ``stretch_block_px`` block (brightest
    first), so the choice is independent of the chunk grid. Sources with fewer
    than ``stretch_kmin`` samples within 2.5 sigma get the dataset-global
    ``ell_default``.

    Args:
        imgs: Plane -> image.
        envs: Plane -> lower envelope.
        src: Sources (``sel`` must be set).
        fits0: Plane -> first fit of that plane.
        p: Fixed parameters.
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0).
        ell_default: Dataset-global log-stretch fallback.
        ks: Planes that measure the stretch.

    Returns:
        (ell per source, number of sources with a direct measurement, median of
        the direct measurements or NaN).
    """
    assert src.sel is not None
    n_src = src.n
    num = np.zeros(n_src)
    den = np.zeros(n_src)
    for k in ks:
        fit = fits0[k]
        cand = src.sel.copy()
        est = src.a * fit.g * fit.tmpl.vals.max() / (p.tex * fit.noise)
        cand &= est >= p.stretch_snr
        cand = top_per_block(
            est, cand, src.y, src.x, origin_yx, p.stretch_block_px, p.stretch_per_block
        )
        s_new, snr = _stretch_search(
            imgs[k],
            envs[k],
            fit.model,
            src,
            fit.g,
            fit.c,
            fit.s,
            fit.tmpl,
            fit.noise,
            p,
            cand,
            _GRID,
        )
        ok = cand & (snr >= p.stretch_snr) & (s_new > _GRID[0] * 1.01)
        ok &= s_new < _GRID[-1] * 0.99
        w = (snr * BETA[k]) ** 2
        num[ok] += w[ok] * np.log(s_new[ok]) / BETA[k]
        den[ok] += w[ok]
    measured = den > 0
    pts = np.c_[src.y, src.x]
    ell_b = np.where(measured, num / np.maximum(den, 1e-12), 0.0)
    ell = smooth_field(
        pts[measured],
        ell_b[measured],
        np.sqrt(den[measured]),
        pts,
        p.stretch_sigma,
        kmin=p.stretch_kmin,
    )
    median = float(np.median(ell_b[measured])) if measured.any() else float("nan")
    return np.where(np.isfinite(ell), ell, ell_default), int(measured.sum()), median


# ----------------------------------------------------------------------------
# reference-stain model
@dataclass
class BeadModel:
    """Fitted model of one window.

    Attributes:
        src: Sources (``is_bead`` excludes pruned sources; ``a`` fitted
            amplitudes; ``ell`` stretch field).
        fits: Plane -> plane fit (``model`` = rendered bead light on the window).
        B: Plane -> lower envelope of the reference stain.
        noise: Plane -> noise sd used.
        info: Counts, timings and ``status`` ("ok", "no beads", "no valid pixel",
            or a calibration-only failure message).
    """

    src: Sources
    fits: dict[int, PlaneFit]
    B: dict[int, np.ndarray]  # noqa: N815
    noise: dict[int, float]
    info: dict[str, Any]


def _remove_sources(
    fit: PlaneFit, src: Sources, idx: np.ndarray, p: PsfSubtractParams
) -> None:
    """Subtract the model of the given sources from a fitted plane (in place).

    Used when sources are pruned after the fit.
    """
    for j in idx:
        if fit.g[j] == 0:
            continue
        y0, y1, x0, x1, h = _stamp(src, j, fit.tmpl, fit.s[j], fit.model.shape, p, {})
        fit.model[y0:y1, x0:x1] -= fit.g[j] * h


def _prune_from_low_planes(
    src: Sources,
    fits: dict[int, PlaneFit],
    low: list[int],
    use: np.ndarray,
    p: PsfSubtractParams,
) -> tuple[np.ndarray, int]:
    """Prune sources without a ring on the final z1/z2 fits.

    The consensus gain of z1 and z2 is information weighted; sources whose
    consensus gain is far from 1 are not beads. ``src.g_pre`` and ``src.w_pre``
    are set.

    Returns:
        (boolean mask of the pruned sources, number of sources with ring
        evidence).
    """
    w_sum = np.zeros(src.n)
    g_sum = np.zeros(src.n)
    for k in low:
        fit = fits[k]
        w_k = (src.a * float(np.max(np.abs(fit.tmpl.vals))) / (p.tex * fit.noise)) ** 2
        g_sum += w_k * fit.g
        w_sum += w_k
    g_pre = np.where(w_sum > 0, g_sum / np.maximum(w_sum, 1e-9), 1.0)
    src.g_pre, src.w_pre = g_pre, w_sum
    pruned = (
        use
        & (w_sum >= p.prune_info)
        & (
            (g_pre < np.where(src.blob, p.prune_gamma_blob, p.prune_gamma))
            | (g_pre > p.prune_gamma_hi)
        )
    )
    pruned |= use & ~src.blob & (src.a > p.a_ratio_max * src.A0)
    n_evidence = int((use & ~pruned & (w_sum >= p.prune_info) & (g_pre >= 0.5)).sum())
    return pruned, n_evidence


def _consensus_gamma(
    src: Sources, fits: dict[int, PlaneFit], low: list[int], p: PsfSubtractParams
) -> np.ndarray:
    """Return the consensus gain (shrunk to 1) of the z1/z2 fits.

    It is the prior gain of the planes >= 3.
    """
    w_sum = np.ones(src.n)
    g_sum = np.ones(src.n)
    for k in low:
        fit = fits[k]
        w_k = (src.a * float(np.max(np.abs(fit.tmpl.vals))) / (p.tex * fit.noise)) ** 2
        w_k = np.minimum(w_k, 400.0)
        g_sum += w_k * fit.g
        w_sum += w_k
    consensus: np.ndarray = np.clip(g_sum / w_sum, 0.0, 1.5)
    return consensus


def _prepare_images(
    polyt: PlaneStack, ks: Sequence[int], p: PsfSubtractParams
) -> tuple[dict[int, np.ndarray], int]:
    """Read the planes as float32 and mark saturated pixels invalid.

    Saturation handling is off by default (``sat_level`` is infinite).

    Returns:
        (plane -> image, number of saturated pixels).
    """
    imgs = {k: np.asarray(polyt[k], np.float32) for k in ks}
    n_sat = 0
    if np.isfinite(p.sat_level):
        for k in ks:
            sat = imgs[k] >= p.sat_level
            n_sat += int(sat.sum())
            if sat.any():
                # 0 = invalid pixel.
                imgs[k] = np.where(sat, 0, imgs[k]).astype(np.float32)
    return imgs, n_sat


def fit_window_library(
    polyt: PlaneStack,
    dapi: PlaneStack | None,
    *,
    origin_yx: tuple[int, int],
    calibration: WindowCalibration,
    params: PsfSubtractParams = DEFAULT_PARAMS,
    planes: Sequence[int],
) -> BeadModel:
    """Fit one window with the dataset library and the global constants.

    Args:
        polyt: Reference-stain planes (array [nz, Y, X] or {z: plane}); planes
            0, 1, 2 and every plane in ``planes`` must be present. Pixels equal
            to 0 are invalid.
        dapi: DAPI planes 0 and 1 (spectral test).
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0). It must be
            a multiple of the envelope block (16) so that the block grids do not
            depend on the chunk (``correct_window`` pads when it is not).
        calibration: Dataset calibration (library, thresholds, noise, A0 bounds,
            default stretch).
        params: Fixed parameters.
        planes: Planes whose output is needed; planes 0..2 are always fitted,
            planes > 2 only when listed.

    Returns:
        The bead model. With no bead the model is empty
        (``info["status"] == "no beads"``) and the output is the identity.
    """
    p = params
    t_start = time.time()
    fit_planes = sorted({0, 1, 2} | {int(k) for k in planes if k > 2})
    lib = calibration.library
    src = detect_sources(
        polyt,
        dapi,
        dog_noise=calibration.dog_noise_z0,
        thresholds=calibration.thresholds,
        params=p,
    )
    info: dict[str, Any] = {
        "n_cand": int(src.n),
        "n_bead": int(src.is_bead.sum()),
        "n_rescued": int(src.rescued.sum()) if src.rescued is not None else 0,
    }
    if not src.is_bead.any():
        return BeadModel(
            src, {}, {}, {}, dict(info, status="no beads", time_s=time.time() - t_start)
        )
    imgs, n_sat = _prepare_images(polyt, fit_planes, p)
    info["n_saturated_px"] = n_sat
    # Saturated cores: the wings must be allowed to scale up.
    p_z0 = dataclasses.replace(p, g_max=6.0) if n_sat else p
    envs = {k: lower_envelope(imgs[k], imgs[k] > 0) for k in fit_planes}
    noise = {k: float(calibration.noise[k]) for k in fit_planes}
    lo, hi = calibration.a0_sel
    a0_median = calibration.a0_median
    use = src.is_bead.copy()
    src.cand_bead = use.copy()
    sel = use & (lo <= src.A0) & (hi >= src.A0)
    t_detect = time.time() - t_start
    # ---- z0, point sources
    f0 = fit_plane(
        imgs[0],
        envs[0],
        src,
        0,
        p_z0,
        use=use,
        sel=sel,
        noise=noise[0],
        refine_pos=True,
        fixed_template=lib.tmpls[0],
    )
    src.a = src.A0 * f0.g
    f0.g[:] = 1.0
    # ---- blobs
    blobs = build_blobs(imgs[0], envs[0], src, f0, p, a0_median)
    info["n_blob"] = int(len(blobs))
    single = use & ~src.blob
    assert src.q is not None
    sel = single & (src.q < 1.15) & (src.a >= lo * 0.8) & (src.a <= hi)
    src.sel = sel
    f0 = fit_plane(
        imgs[0],
        envs[0],
        src,
        0,
        p_z0,
        use=use,
        sel=sel,
        noise=noise[0],
        refine_pos=True,
        fixed_template=lib.tmpls[0],
    )
    src.a = np.where(single, src.a * f0.g, src.a)
    f0.g[single] = 1.0
    fits: dict[int, PlaneFit] = {0: f0}
    t_z0 = time.time() - t_start
    # ---- stretch field from z0, z1, z2 (singles)
    for k in (1, 2):
        fits[k] = fit_plane(
            imgs[k],
            envs[k],
            src,
            k,
            p,
            use=use,
            sel=sel,
            noise=noise[k],
            fixed_template=lib.tmpls[k],
        )
    ell, n_field, ell_meas = estimate_stretch_field(
        imgs,
        envs,
        src,
        fits,
        p,
        origin_yx=origin_yx,
        ell_default=float(calibration.ell_default),
        ks=(0, 1, 2),
    )
    info["ell_meas_median"] = ell_meas
    src.ell = ell
    info["n_stretch_beads"] = n_field
    info["ell_median"] = float(np.median(ell))
    info["ell_sd"] = float(np.std(ell))
    t_stretch = time.time() - t_start
    # ---- planes 1, 2 with the stretch field; then planes >= 3 with the
    # per-source consensus gain as prior
    for k in (1, 2):
        fits[k] = fit_plane(
            imgs[k],
            envs[k],
            src,
            k,
            p,
            use=use,
            sel=sel,
            s=np.exp(BETA[k] * ell),
            noise=noise[k],
            fixed_template=lib.tmpls[k],
        )
    low = [1, 2]
    pruned, n_evidence = _prune_from_low_planes(src, fits, low, use, p)
    info["n_pruned"] = int(pruned.sum())
    info["n_ring_evidence"] = n_evidence
    for k in low:
        _remove_sources(fits[k], src, np.where(pruned)[0], p)
    use = use & ~pruned
    src.is_bead = src.is_bead & ~pruned
    sel = sel & ~pruned
    src.sel = sel
    gamma = _consensus_gamma(src, fits, low, p)
    for k in [k for k in fit_planes if k > 2]:
        fits[k] = fit_plane(
            imgs[k],
            envs[k],
            src,
            k,
            p,
            use=use,
            sel=sel,
            s=np.exp(BETA[k] * ell),
            noise=noise[k],
            gp_arr=gamma,
            fixed_template=lib.tmpls[k],
        )
    # The prototype refits z0 here; z0 is never an output, so the refit is
    # skipped.
    info["gamma_med"] = float(np.median(gamma[use])) if use.any() else 1.0
    info["n_beads_final"] = int(use.sum())
    info["status"] = "ok"
    info["time_s"] = time.time() - t_start
    info["time_detect_s"] = t_detect
    info["time_z0_s"] = t_z0 - t_detect
    info["time_stretch_s"] = t_stretch - t_z0
    return BeadModel(src, fits, envs, noise, info)


def fit_tile_local(
    polyt: PlaneStack,
    dapi: PlaneStack | None,
    *,
    params: PsfSubtractParams = DEFAULT_PARAMS,
    thresholds: ClassifierThresholds | None = None,
    min_ring_prom: float | None = None,
    n_planes: int = 4,
) -> BeadModel:
    """Fit one off-tissue calibration tile in local mode.

    The templates are stacked from the tile itself.

    Args:
        polyt: Reference-stain planes 0..n_planes-1.
        dapi: DAPI planes 0..1 (spectral test) or None.
        params: Fixed parameters.
        thresholds: Classifier thresholds (default: the parameter defaults).
        min_ring_prom: Minimum z1 ring prominence of the stacked template
            (default ``params.min_ring_prom``).
        n_planes: Planes 0..n_planes-1 are fitted (4: z0..z3; higher planes are
            derived by scaling).

    Returns:
        The bead model. ``info["status"]`` is "ok" or the reason the tile does not
        show beads (then the tile is not used for the library).
        ``fits[k].tmpl`` are the stacked templates and ``info`` carries the ring
        prominence and radius, the counts and ``gamma_med``.
    """
    p = (
        params
        if min_ring_prom is None
        else dataclasses.replace(params, min_ring_prom=min_ring_prom)
    )
    th = thresholds or ClassifierThresholds.from_params(p)
    t_start = time.time()
    ks_all = list(range(n_planes))
    src = detect_sources(polyt, dapi, dog_noise=None, thresholds=th, params=p)
    info: dict[str, Any] = {"n_cand": int(src.n), "n_bead": int(src.is_bead.sum())}
    if src.is_bead.sum() < 8:
        return BeadModel(src, {}, {}, {}, dict(info, status="too few beads"))
    imgs, n_sat = _prepare_images(polyt, ks_all, p)
    info["n_saturated_px"] = n_sat
    envs = {k: lower_envelope(imgs[k], imgs[k] > 0) for k in ks_all}
    noise = {k: robust_noise(imgs[k], imgs[k] > 0) for k in ks_all}
    use = src.is_bead
    src.cand_bead = use.copy()
    lo, hi = np.percentile(src.A0[use], p.sel_pct)
    sel = use & (lo <= src.A0) & (hi >= src.A0)
    f0 = fit_plane(
        imgs[0],
        envs[0],
        src,
        0,
        p,
        use=use,
        sel=sel,
        noise=noise[0],
        refine_pos=True,
    )
    src.a = src.A0 * f0.g
    f0.g[:] = 1.0
    blobs = build_blobs(
        imgs[0], envs[0], src, f0, p, float(np.median(src.A0[src.is_bead]))
    )
    info["n_blob"] = int(len(blobs))
    single = use & ~src.blob
    assert src.q is not None
    sel = single & (src.q < 1.15) & (src.a >= lo * 0.8) & (src.a <= hi)
    if sel.sum() < 8:
        sel = single.copy()
    src.sel = sel
    f0 = fit_plane(
        imgs[0],
        envs[0],
        src,
        0,
        p,
        use=use,
        sel=sel,
        noise=noise[0],
        refine_pos=True,
    )
    src.a = np.where(single, src.a * f0.g, src.a)
    f0.g[single] = 1.0
    fits: dict[int, PlaneFit] = {0: f0}
    for k in (1, 2):
        fits[k] = fit_plane(
            imgs[k], envs[k], src, k, p, use=use, sel=sel, noise=noise[k]
        )
    prominence, ring_radius = ring_prominence(fits[1].tmpl, 20.0, 40.0)
    info["z1_ring_prominence"] = prominence
    info["z1_ring_radius"] = ring_radius
    if prominence < p.min_ring_prom or not (20.5 <= ring_radius <= 39.5):
        return BeadModel(
            src,
            {},
            {},
            {},
            dict(info, status="no bead PSF (no z1 ring in the stacked template)"),
        )
    src.sel = sel
    ell, n_field, ell_meas = estimate_stretch_field(
        imgs, envs, src, fits, p, origin_yx=(0, 0), ell_default=0.0, ks=(0, 1, 2)
    )
    info["ell_meas_median"] = ell_meas
    src.ell = ell
    info["n_stretch_beads"] = n_field
    info["ell_median"] = float(np.median(ell))
    info["ell_sd"] = float(np.std(ell))
    flux0 = f0.tmpl.flux(1e9)
    for k in (1, 2):
        fits[k] = fit_plane(
            imgs[k],
            envs[k],
            src,
            k,
            p,
            use=use,
            sel=sel,
            s=np.exp(BETA[k] * ell),
            noise=noise[k],
            flux0=flux0,
        )
    low = [1, 2]
    pruned, n_evidence = _prune_from_low_planes(src, fits, low, use, p)
    info["n_pruned"] = int(pruned.sum())
    info["n_ring_evidence"] = n_evidence
    if n_evidence < p.min_evidence:
        return BeadModel(
            src,
            {},
            {},
            {},
            dict(
                info,
                status="no bead evidence (too few sources with a fitted z1/z2 ring)",
            ),
        )
    for k in low:
        _remove_sources(fits[k], src, np.where(pruned)[0], p)
    use = use & ~pruned
    src.is_bead = src.is_bead & ~pruned
    sel = sel & ~pruned
    src.sel = sel
    gamma = _consensus_gamma(src, fits, low, p)
    for k in [k for k in ks_all if k >= 3]:
        fits[k] = fit_plane(
            imgs[k],
            envs[k],
            src,
            k,
            p,
            use=use,
            sel=sel,
            s=np.exp(BETA[k] * ell),
            noise=noise[k],
            gp_arr=gamma,
            flux0=flux0,
        )
    # z0 template of the library: refit with the final stretch field (the
    # stacked z0 profile is the library's z0).
    fits[0] = fit_plane(
        imgs[0],
        envs[0],
        src,
        0,
        p,
        use=use,
        sel=sel,
        s=np.exp(BETA[0] * ell),
        noise=noise[0],
    )
    info["gamma_med"] = float(np.median(gamma[use])) if use.any() else 1.0
    info["n_sel"] = int(sel.sum())
    info["status"] = "ok"
    info["time_s"] = time.time() - t_start
    return BeadModel(src, fits, envs, noise, info)


def derive_library_templates(
    tmpls: list[Template], params: PsfSubtractParams = DEFAULT_PARAMS
) -> list[Template]:
    """Derive the templates of z4..z7 from a z0..z3 list.

    A thin wrapper kept here for the calibration pooling.
    """
    return derive_high_templates(tmpls, params)


def beads_table(model: BeadModel, origin_yx: tuple[int, int]) -> np.ndarray:
    """List the sources that were classified as beads, pruned ones included.

    Args:
        model: Fitted window model.
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0).

    Returns:
        float64 array [n, 8] with columns: absolute y, absolute x (mosaic px),
        A0, fitted amplitude a, blob (0/1), consensus gain gamma, stretch ell,
        pruned (0/1).
    """
    s = model.src
    if s.n == 0 or s.cand_bead is None or not s.cand_bead.any():
        return np.zeros((0, 8))
    m = s.cand_bead
    gamma = s.g_pre if s.g_pre is not None else np.full(s.n, np.nan)
    ell = s.ell if s.ell is not None else np.full(s.n, np.nan)
    return np.column_stack(
        [
            s.y[m] + origin_yx[0],
            s.x[m] + origin_yx[1],
            s.A0[m],
            s.a[m],
            s.blob[m].astype(float),
            gamma[m],
            ell[m],
            (~s.is_bead[m]).astype(float),
        ]
    )
