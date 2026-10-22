"""Batched radial-template stamps and window geometry on the device.

A "stamp" is the unit-gain model image of one source on one plane: ``a * H_k(r / s) /
s^2`` on a square window of half size ``R = r_sup * s + win_extra`` around the source.
Windows are generated for a whole batch of sources at once in a common (padded) size
``Hb x Wb`` together with row / column masks that reproduce the exact per-source extent
of the CPU code (``window_bounds``).  Images live in a zero-padded frame (pad ``P`` px,
valid = 0 in the pad) so that windows that leave the image are clipped by the validity
weights, exactly like the CPU window clipping.
"""

# Window dimensions and matrices keep the capital letters of the formulas they port
# from the CPU modules (H, W, B, M, K, A0 ...), so the two can be read side by side.
# ruff: noqa: N803, N806
# mypy: disable-error-code="no-any-return"
# (the torch.fft and tensor-method stubs return Any)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from merxen.io.bead_removal.psf import Template

from . import ops


@dataclass
class TmplDev:
    """Radial template on the device (float32 node table; uniform node spacing
    ``ds``)."""

    vals: torch.Tensor
    ds: float
    n: int
    r_sup: float

    @classmethod
    def from_template(cls, t: Template) -> TmplDev:
        return cls(
            torch.from_numpy(np.asarray(t.vals, np.float32)).to(ops.device()),
            float(t.ds),
            int(t.n),
            float(t.r_sup),
        )

    def eval(self, r: torch.Tensor, s: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Profile at radii ``r`` (float32) dilated by ``s`` (broadcastable): interp(r /
        s) / s^2, zero beyond the table (numpy.interp(..., right=0))."""
        norm: torch.Tensor | float
        if isinstance(s, torch.Tensor):
            t = r / (s * self.ds)
            norm = 1.0 / (s * s)
        else:
            t = r / (s * self.ds)
            norm = 1.0 / (s * s)
        i0 = t.floor().clamp_(0, self.n - 2)
        f = t - i0
        i0 = i0.to(torch.int64)
        v0 = self.vals[i0]
        v1 = self.vals[i0 + 1]
        out = v0 + (v1 - v0) * f
        out = torch.where(t <= (self.n - 1), out, torch.zeros_like(out))
        return out * norm


@dataclass
class Geom:
    """Window geometry of a batch (all on the device).

    Attributes:
        flat: int64 [B, Hb, Wb] flat indices into the padded frame.
        mask: float32 [B, Hb, Wb] 1 inside the exact per-source window, 0 in the
            common-size padding.
        yy, xx: float32 [B, Hb] / [B, Wb] absolute (unpadded) pixel coordinates of the
        window rows / columns.
    """

    flat: torch.Tensor
    mask: torch.Tensor
    yy: torch.Tensor
    xx: torch.Tensor


def make_geom(
    yc: torch.Tensor,
    xc: torch.Tensor,
    R: torch.Tensor,
    Hb: int,
    Wb: int,
    pad: int,
    Wp: int,
) -> Geom:
    """Windows of half size ``R`` around (yc, xc) (float64 [B]) in the padded frame (row
    stride ``Wp``)."""
    dev = yc.device
    y0 = torch.floor(yc - R)
    y1 = torch.ceil(yc + R) + 1
    x0 = torch.floor(xc - R)
    x1 = torch.ceil(xc + R) + 1
    arh = torch.arange(Hb, device=dev, dtype=torch.float64)
    arw = torch.arange(Wb, device=dev, dtype=torch.float64)
    rowm = (arh[None, :] < (y1 - y0)[:, None]).to(torch.float32)
    colm = (arw[None, :] < (x1 - x0)[:, None]).to(torch.float32)
    yy = y0[:, None] + arh[None, :]
    xx = x0[:, None] + arw[None, :]
    flat = (yy.to(torch.int64) + pad)[:, :, None] * Wp + (xx.to(torch.int64) + pad)[
        :, None, :
    ]
    mask = rowm[:, :, None] * colm[:, None, :]
    return Geom(flat, mask, yy.to(torch.float32), xx.to(torch.float32))


def radius_grid(geom: Geom, yc: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
    """float32 distance of every window pixel to its source (same float32 arithmetic as
    ``detect.radius_grid``)."""
    dy = geom.yy - yc.to(torch.float32)[:, None]
    dx = geom.xx - xc.to(torch.float32)[:, None]
    return torch.sqrt(dy[:, :, None] * dy[:, :, None] + dx[:, None, :] * dx[:, None, :])


def point_stamps(
    tm: TmplDev,
    a: torch.Tensor,
    s: torch.Tensor,
    yc: torch.Tensor,
    xc: torch.Tensor,
    geom: Geom,
) -> torch.Tensor:
    """Unit-gain stamps ``a * tmpl(r, s)`` of a batch of point sources, float32 [B, Hb,
    Wb] (outside the window: 0)."""
    r = radius_grid(geom, yc, xc)
    h = (
        tm.eval(r, s.to(torch.float32)[:, None, None])
        * a.to(torch.float32)[:, None, None]
    )
    return h * geom.mask


def kern2d(tm: TmplDev, s: float) -> tuple[torch.Tensor, int]:
    """2-D kernel of a template dilated by ``s`` and its half size R (kernel (2R+1)^2),
    float32 (``psf.kern2d``)."""
    R = int(np.ceil(tm.r_sup * s))
    ar = torch.arange(-R, R + 1, device=ops.device(), dtype=torch.float64)
    r = torch.hypot(ar[:, None], ar[None, :]).to(torch.float32)
    return tm.eval(r, float(s)), R


def fft_conv_full(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Full linear convolution of two 2-D float32 tensors via FFT."""
    H = u.shape[0] + K.shape[0] - 1
    W = u.shape[1] + K.shape[1] - 1
    fu = torch.fft.rfft2(u, s=(H, W))
    fk = torch.fft.rfft2(K, s=(H, W))
    return torch.fft.irfft2(fu * fk, s=(H, W))
