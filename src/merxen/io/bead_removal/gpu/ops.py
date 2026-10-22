"""Low-level torch helpers that reproduce the numpy / scipy primitives used by the CPU
modules on the GPU.

Every helper mimics the exact numerical convention of the CPU function it replaces
(scipy 'reflect' / 'nearest' padding, numpy 'linear' percentile, numpy median of an even
count = mean of the two middle values, scipy rank filters), so the GPU path differs from
the CPU path only by float32 rounding and by the algorithmic changes that are documented
in ``fit.py``.
"""

# Window dimensions and matrices keep the capital letters of the formulas they port
# from the CPU modules (H, W, B, M, K, A0 ...), so the two can be read side by side.
# ruff: noqa: N803, N806
# mypy: disable-error-code="no-any-return"
# (the torch.fft and tensor-method stubs return Any)

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# The envelope constants and the nearest-block fill are the CPU module's public names,
# so both backends share one definition.
from merxen.io.bead_removal.detect import (
    ENVELOPE_FILTER_PCT,
    ENVELOPE_GAUSS_REACH_BLOCKS,
    ENVELOPE_GAUSS_SIGMA_BLOCKS,
)
from merxen.io.bead_removal.detect import (
    fill_undefined_blocks as _fill_undefined_blocks_host,
)

_DEV: torch.device | None = None


def device() -> torch.device:
    """The CUDA device (cached)."""
    global _DEV
    if _DEV is None:
        _DEV = torch.device("cuda")
    return _DEV


def configure(deterministic: bool = True) -> None:
    """Numerics settings: no TF32 anywhere (counts are compared at the 1-count level),
    deterministic kernels."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Every kernel used is deterministic except atomic scatter-adds; those only run on
    # batches of disjoint windows (the colour ordering), where the result cannot depend
    # on the order of the adds.
    torch.use_deterministic_algorithms(False)


def to_dev(
    a: np.ndarray | torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """numpy array -> device tensor (float64 stays float64 unless ``dtype`` is
    given)."""
    t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
    t = t.to(device(), non_blocking=True)
    return t if dtype is None else t.to(dtype)


# ------------------------------------------------------------------------------------
# padding and separable filters
def pad_dim(x: torch.Tensor, n: int, dim: int, mode: str) -> torch.Tensor:
    """Pad ``n`` samples on both sides of ``dim`` with scipy conventions: 'reflect' =
    half-sample symmetric (d c b a | a b c d | d c b a), 'nearest' = edge replicate,
    'constant' = zeros."""
    if n == 0:
        return x
    L = x.shape[dim]
    if mode == "reflect":
        if n > L:
            raise ValueError("pad larger than the dimension is not supported")
        head = x.narrow(dim, 0, n).flip(dim)
        tail = x.narrow(dim, L - n, n).flip(dim)
    elif mode == "nearest":
        shp = list(x.shape)
        shp[dim] = n
        head = x.narrow(dim, 0, 1).expand(*shp)
        tail = x.narrow(dim, L - 1, 1).expand(*shp)
    elif mode == "constant":
        shp = list(x.shape)
        shp[dim] = n
        head = tail = x.new_zeros(shp)
    else:
        raise ValueError(mode)
    return torch.cat([head, x, tail], dim)


def gauss_kernel(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """scipy.ndimage._gaussian_kernel1d(sigma, 0, radius) with radius =
    int(truncate*sigma+0.5), float32."""
    r = int(truncate * float(sigma) + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 / (sigma * sigma) * x * x)
    w /= w.sum()
    return torch.from_numpy(w.astype(np.float32)).to(device())


def gaussian_filter(
    x: torch.Tensor, sigma: float, mode: str = "reflect", truncate: float = 4.0
) -> torch.Tensor:
    """scipy.ndimage.gaussian_filter of a 2-D float32 tensor (separable, given
    boundary mode and truncation, default 4)."""
    k = gauss_kernel(sigma, truncate)
    r = (k.numel() - 1) // 2
    y = x[None, None]
    y = F.conv2d(pad_dim(y, r, 2, mode), k.view(1, 1, -1, 1))
    y = F.conv2d(pad_dim(y, r, 3, mode), k.view(1, 1, 1, -1))
    return y[0, 0]


# ------------------------------------------------------------------------------------
# order statistics
def median_lastdim(v: torch.Tensor) -> torch.Tensor:
    """numpy.median along the last dimension (even count: mean of the two middle
    values)."""
    n = v.shape[-1]
    s, _ = v.sort(-1)
    return 0.5 * (s[..., (n - 1) // 2] + s[..., n // 2])


def masked_median_lastdim(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """numpy.median of ``v[m]`` per row for a boolean mask with a per-row count (rows
    must have >= 1 True)."""
    big = torch.finfo(v.dtype).max
    s, _ = torch.where(m, v, torch.full_like(v, big)).sort(-1)
    n = m.sum(-1)
    lo = torch.gather(s, -1, ((n - 1) // 2).unsqueeze(-1)).squeeze(-1)
    hi = torch.gather(s, -1, (n // 2).unsqueeze(-1)).squeeze(-1)
    return 0.5 * (lo + hi)


def percentile_lastdim(
    v: torch.Tensor, pct: float, count: torch.Tensor | None = None
) -> torch.Tensor:
    """numpy.percentile(..., method='linear') along the last dim.

    ``count``: number of leading valid entries per row after sorting (entries beyond it
    must already be +inf); None = all entries valid.
    """
    s, _ = v.sort(-1)
    n = v.shape[-1]
    if count is None:
        pos = pct / 100.0 * (n - 1)
        lo = int(math.floor(pos))
        f = pos - lo
        hi = min(lo + 1, n - 1)
        return s[..., lo] * (1.0 - f) + s[..., hi] * f
    cnt = count.to(torch.float64)
    pos_t = pct / 100.0 * (cnt - 1.0).clamp(min=0)
    lo_t = pos_t.floor().to(torch.int64)
    f_t = (pos_t - lo_t.to(torch.float64)).to(v.dtype)
    hi_t = torch.minimum(lo_t + 1, (count - 1).clamp(min=0))
    a = torch.gather(s, -1, lo_t.unsqueeze(-1)).squeeze(-1)
    b = torch.gather(s, -1, hi_t.unsqueeze(-1)).squeeze(-1)
    return a * (1.0 - f_t) + b * f_t


def rank_filter_2d(c: torch.Tensor, size: int, pct: float) -> torch.Tensor:
    """scipy.ndimage.percentile_filter(c, pct, size=size, mode='nearest') of a small 2-D
    tensor (block grid)."""
    r = size // 2
    p = pad_dim(pad_dim(c, r, 0, "nearest"), r, 1, "nearest")
    u = (
        p.unfold(0, size, 1)
        .unfold(1, size, 1)
        .reshape(c.shape[0], c.shape[1], size * size)
    )
    rank = int(float(size * size) * pct / 100.0)
    if rank == size * size:
        rank -= 1
    # sort (not kthvalue): kthvalue has no deterministic CUDA kernel
    return u.sort(-1).values[..., rank]


# ------------------------------------------------------------------------------------
# lower envelope
def masked_percentile_blocks(
    a: torch.Tensor, ok: torch.Tensor, pct: float
) -> torch.Tensor:
    """Linear-interpolation percentile of the ``ok`` entries of every row
    (``merxen.io.bead_removal.detect._masked_percentile_rows``).

    Args:
        a: float32 [n, m] values.
        ok: bool [n, m], True where the entry takes part.
        pct: Percentile in percent.

    Returns:
        float32 [n]; NaN for a row without a valid entry.  For a row whose entries are
        all valid the result equals the plain percentile of the row (same virtual index,
        numpy's interpolation formula).
    """
    n_ok = ok.sum(-1)
    # invalid entries go last
    srt, _ = torch.where(ok, a, torch.full_like(a, float("inf"))).sort(-1)
    cnt = n_ok.clamp(min=1)
    pos = (cnt - 1).to(torch.float64) * (pct / 100.0)
    lo = pos.floor().to(torch.int64)
    hi = torch.minimum(lo + 1, cnt - 1)
    g = (pos - lo.to(torch.float64)).to(a.dtype)
    lo_v = torch.gather(srt, -1, lo.unsqueeze(-1)).squeeze(-1)
    hi_v = torch.gather(srt, -1, hi.unsqueeze(-1)).squeeze(-1)
    # avoid inf - inf in rows without a valid entry
    lo_v = torch.where(n_ok > 0, lo_v, torch.zeros_like(lo_v))
    hi_v = torch.where(n_ok > 0, hi_v, torch.zeros_like(hi_v))
    d = hi_v - lo_v
    res = torch.where(g >= 0.5, hi_v - d * (1.0 - g), lo_v + d * g)  # numpy's lerp
    return torch.where(n_ok > 0, res, torch.full_like(res, float("nan")))


def rank_filter_2d_nan(c: torch.Tensor, size: int, pct: float) -> torch.Tensor:
    """``merxen.io.bead_removal.detect._percentile_filter_nan``: percentile filter of a
    small 2-D grid that ignores NaN entries.

    The rank is ``floor(n_valid * pct / 100)`` among the valid entries of the
    (edge-replicated) neighbourhood, which is scipy's rule; where the neighbourhood
    holds no NaN the result is identical to ``rank_filter_2d``.  NaN where the
    neighbourhood has no valid entry.
    """
    r = size // 2
    p = pad_dim(pad_dim(c, r, 0, "nearest"), r, 1, "nearest")
    u = (
        p.unfold(0, size, 1)
        .unfold(1, size, 1)
        .reshape(c.shape[0], c.shape[1], size * size)
    )
    n_ok = (~torch.isnan(u)).sum(-1)
    srt = u.sort(-1).values  # NaN sorts last
    rank = torch.minimum(
        torch.floor(n_ok.to(torch.float64) * (pct / 100.0)).to(torch.int64),
        (n_ok - 1).clamp(min=0),
    )
    res = torch.gather(srt, -1, rank.unsqueeze(-1)).squeeze(-1)
    return torch.where(n_ok > 0, res, torch.full_like(res, float("nan")))


def fill_undefined_blocks(c: torch.Tensor, reach: int) -> torch.Tensor:
    """Give NaN blocks the value of the nearest defined block within ``reach`` blocks
    (chessboard), else 0.

    The block grid is tiny (a few hundred blocks per side), so the nearest-block search
    of the reference (``merxen.io.bead_removal.detect.fill_undefined_blocks``, a scipy
    chessboard distance transform) runs on the host: the tie-breaking between equally
    near blocks is then identical to the CPU path by construction, for a negligible
    cost.
    """
    if not bool(torch.isnan(c).any()):
        return c
    out = _fill_undefined_blocks_host(c.cpu().numpy(), reach)
    return torch.from_numpy(np.ascontiguousarray(out)).to(c.device)


def lower_envelope(
    img: torch.Tensor,
    valid: torch.Tensor | None = None,
    block: int = 16,
    win_blocks: int = 9,
    pct: float = 20.0,
    sub: int = 1,
) -> torch.Tensor:
    """GPU port of ``merxen.io.bead_removal.detect.lower_envelope`` (block percentile ->
    percentile filter -> gaussian -> bilinear).

    Invalid pixels are handled locally, like on the CPU, so a valid pixel's envelope
    never depends on the rest of the
    window: the block percentile uses the valid pixels of each block only (same
        statistic and same sub-sampling as for a
    fully valid window), the block-grid filter ignores blocks without a valid pixel, and
    blocks that stay undefined are filled from the nearest defined block within a fixed
    reach.

    Args:
        img: 2-D float32 plane on the device.
        valid: Optional boolean mask of usable pixels.
        block: Block size in px.
        win_blocks: Size (blocks) of the percentile filter over the block grid.
        pct: Percentile within a block.
        sub: Sub-sampling of the pixels within a block (the same pattern whatever the
            validity).

    Returns:
        float32 tensor of the same shape.
    """
    Y, X = img.shape
    by, bx = Y // block, X // block
    a = (
        img[: by * block, : bx * block]
        .reshape(by, block, bx, block)
        .permute(0, 2, 1, 3)
        .reshape(by, bx, -1)
    )
    a_s = a[:, :, ::sub] if sub > 1 else a
    if valid is None or bool(valid.all()):
        c = percentile_lastdim(a_s, pct)
        c = rank_filter_2d(c, win_blocks, ENVELOPE_FILTER_PCT)
    else:
        assert valid is not None
        v = (
            valid[: by * block, : bx * block]
            .reshape(by, block, bx, block)
            .permute(0, 2, 1, 3)
            .reshape(by, bx, -1)
        )
        v_s = v[:, :, ::sub] if sub > 1 else v
        c = masked_percentile_blocks(
            a_s.reshape(by * bx, -1), v_s.reshape(by * bx, -1), pct
        ).reshape(by, bx)
        c = rank_filter_2d_nan(c, win_blocks, ENVELOPE_FILTER_PCT)
        reach = max(1, ENVELOPE_GAUSS_REACH_BLOCKS + 1 - win_blocks // 2)
        c = fill_undefined_blocks(c, reach)
    c = gaussian_filter(c, ENVELOPE_GAUSS_SIGMA_BLOCKS, "nearest")
    return bilinear_blocks(c, Y, X, block)


def bilinear_blocks(c: torch.Tensor, Y: int, X: int, block: int) -> torch.Tensor:
    """scipy map_coordinates(order=1, mode='nearest') of the block grid at the pixel
    centres (separable lerp)."""
    by, bx = c.shape
    dev = c.device

    def axis(n: int, nb: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = (
            (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) / block - 0.5
        ).clamp(0, nb - 1)
        i0 = t.floor().to(torch.int64).clamp(max=nb - 1)
        i1 = (i0 + 1).clamp(max=nb - 1)
        w = (t - i0.to(torch.float64)).to(torch.float32)
        return i0, i1, w

    y0, y1, wy = axis(Y, by)
    x0, x1, wx = axis(X, bx)
    cx = c[:, x0] * (1 - wx) + c[:, x1] * wx  # [by, X]
    return cx[y0] * (1 - wy)[:, None] + cx[y1] * wy[:, None]
