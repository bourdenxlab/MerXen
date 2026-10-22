"""Bead detection and classification on plane z0 of the reference stain.

Also holds the small image helpers shared by the model (noise estimates, the
lower envelope, window geometry). The DoG noise and the classifier thresholds
are dataset-global inputs: a chunk must never decide anything from its own
statistics.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from .params import DEFAULT_PARAMS, ENVELOPE_BLOCK_PX, PsfSubtractParams

logger = logging.getLogger(__name__)

__all__ = [
    "ENVELOPE_BLOCK_PX",
    "ENVELOPE_FILTER_PCT",
    "ENVELOPE_GAUSS_REACH_BLOCKS",
    "ENVELOPE_GAUSS_SIGMA_BLOCKS",
    "ClassifierThresholds",
    "PlaneStack",
    "Sources",
    "centre_minus_bg",
    "detect_sources",
    "dog_noise_of",
    "empty_sources",
    "fill_undefined_blocks",
    "huber1",
    "lower_envelope",
    "nms",
    "null_sd",
    "radius_grid",
    "ring_ratio",
    "ring_tscores",
    "robust_noise",
    "window_bounds",
]

#: A stack is an array [nz, Y, X] (plane index = z) or a mapping {z: 2-D array}.
PlaneStack: TypeAlias = np.ndarray | Mapping[int, np.ndarray]


@dataclass(frozen=True)
class ClassifierThresholds:
    """Per-dataset thresholds of the bead / not-bead classifier.

    They are derived once in the calibration pass.

    Attributes:
        ring_min: Lowest z0 ring / centre ratio of a bead.
        ring_max: Highest z0 ring / centre ratio of a bead.
        r1_max: Highest PolyT z1 / z0 centre ratio of a bead.
        rd_center: Expected (DAPI z0 - z1) / PolyT z0 contrast ratio of a bead.
        rd0_center: Expected DAPI z0 / PolyT z0 contrast ratio of a bead.
        rd_tol: Tolerance of the DAPI ratios (grows with local texture).
        rescue_tz1_min: Minimum sector t-score of the raw z1 ring of a candidate that
            fails the z0 ring test on the low side and is rescued; 0 switches the
            rescue off.
        rescue_ring_radius_px: z1 ring radius of the PSF library (px) at which the
            rescue ring is measured.
        rescue_a0_min: Minimum z0 amplitude (counts) of a rescued candidate (the ring
            t-score of a faint source is noise).
        rescue_sd1_max: Texture gate of the rescue: highest z1 null sd / A0
            (``Sources.sd1_rel``) of a rescued candidate, derived per dataset by the
            calibration. 0 (the default, and the value of every calibration written
            before the gate existed) switches the rescue off.
        rescue_zprof_zmax: Highest plane of the "no secondary focus" test of the
            rescue (the contrast / A0 of the candidate at z2 .. zmax must stay within
            ``max(r1_max, 3.5 null_sd / A0)``); 0 = test off.
    """

    ring_min: float = 0.30
    ring_max: float = 1.30
    r1_max: float = 0.25
    rd_center: float = 0.097
    rd0_center: float = 0.106
    rd_tol: float = 0.12
    rescue_tz1_min: float = 0.0
    rescue_ring_radius_px: float = 0.0
    rescue_a0_min: float = 1000.0
    rescue_sd1_max: float = 0.0
    rescue_zprof_zmax: float = 0.0

    #: Names of the fields that only act through the rescue.
    _RESCUE_FIELDS = (
        "rescue_tz1_min",
        "rescue_ring_radius_px",
        "rescue_a0_min",
        "rescue_sd1_max",
        "rescue_zprof_zmax",
    )

    @classmethod
    def from_params(
        cls, params: PsfSubtractParams = DEFAULT_PARAMS
    ) -> ClassifierThresholds:
        """Take the defaults from the parameter set.

        Args:
            params: Fixed parameters; their classifier defaults are the validated
                values of the reference dataset.

        Returns:
            Thresholds equal to the parameter defaults.
        """
        return cls(
            params.ring_min,
            params.ring_max,
            params.r1_max,
            params.rd_center,
            params.rd0_center,
            params.rd_tol,
        )

    @property
    def rescue_active(self) -> bool:
        """Whether the z1-ring rescue is on (all three of its switches are set)."""
        return (
            self.rescue_tz1_min > 0
            and self.rescue_ring_radius_px > 0
            and self.rescue_sd1_max > 0
        )

    @property
    def rescue_planes(self) -> tuple[int, ...]:
        """Reference-stain planes (beyond 0 and 1) that the rescue reads.

        Returns:
            The planes 2 .. ``rescue_zprof_zmax`` of the z-profile test, or an empty
            tuple while the rescue or that test is off.
        """
        if not self.rescue_active or self.rescue_zprof_zmax < 2:
            return ()
        return tuple(range(2, int(self.rescue_zprof_zmax) + 1))

    def to_dict(self) -> dict[str, float]:
        """Return the thresholds as a JSON-able dict of floats."""
        return {k: float(v) for k, v in self.__dict__.items()}

    def to_hash_dict(self) -> dict[str, float]:
        """Return the thresholds that decide a correction, for the calibration hash.

        The rescue fields are left out while the rescue is off, so a calibration
        without an active rescue keeps the hash it had before the rescue existed.

        Returns:
            The thresholds as floats, without the rescue fields unless the rescue is
            active.
        """
        out = self.to_dict()
        if not self.rescue_active:
            for name in self._RESCUE_FIELDS:
                out.pop(name)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClassifierThresholds:
        """Read thresholds written by :meth:`to_dict`.

        Fields that this version does not know (written by an earlier development
        version) are ignored; fields that are missing keep their defaults, so a
        calibration written before the rescue existed loads with the rescue off.

        Args:
            data: Mapping of threshold names to values.

        Returns:
            The thresholds.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})


# ----------------------------------------------------------------------------
# generic helpers
def robust_noise(img: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Estimate the pixel noise sd from the residual of a 5x5 mean filter.

    Robust to structure; the residual is sub-sampled by 3.

    Args:
        img: 2-D plane.
        valid: Optional boolean mask of usable pixels.

    Returns:
        Robust noise standard deviation in counts (1.0 if no valid pixel).
    """
    residual = (img - ndi.uniform_filter(img, 5))[::3, ::3]
    if valid is not None:
        residual = residual[valid[::3, ::3]]
    if residual.size == 0:
        return 1.0
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(1.4826 * mad * np.sqrt(25.0 / 24.0))


def dog_noise_of(plane: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Estimate the robust sd (1.4826 x MAD) of the DoG(1, 3 px) response.

    Args:
        plane: 2-D z0 plane.
        valid: Optional boolean mask of usable pixels (default: pixels > 0).

    Returns:
        The DoG noise used as the detection threshold unit (1.0 if no valid
        pixel).
    """
    p0 = np.asarray(plane, np.float32)
    if valid is None:
        valid = p0 > 0
    dog = gaussian_filter(p0, 1) - gaussian_filter(p0, 3)
    vals = dog[valid]
    if not vals.size:
        return 1.0
    return float(1.4826 * np.median(np.abs(vals - np.median(vals))))


#: Sigma (blocks) of the Gaussian smoothing of the envelope's block grid.
ENVELOPE_GAUSS_SIGMA_BLOCKS = 2.0
#: Reach (blocks) of that smoothing: scipy truncates a Gaussian at 4 sigma.
ENVELOPE_GAUSS_REACH_BLOCKS = int(4.0 * ENVELOPE_GAUSS_SIGMA_BLOCKS + 0.5)
#: Percentile (percent) of the envelope's second, block-grid filter.
ENVELOPE_FILTER_PCT = 30.0


def _masked_percentile_rows(vals: np.ndarray, ok: np.ndarray, pct: float) -> np.ndarray:
    """Take the percentile of the ``ok`` entries of every row.

    Uses numpy's default (linear interpolation) percentile definition. It is
    sort based so that rows with different numbers of valid entries can be
    processed together; for a row whose entries are all valid the result equals
    ``np.percentile(row, pct)`` (same virtual index and interpolation formula).

    Args:
        vals: Array [n, m].
        ok: Boolean array [n, m], True where the entry takes part.
        pct: Percentile in percent.

    Returns:
        Array [n] of the ``vals`` dtype; NaN for a row without a valid entry.
    """
    n_ok = ok.sum(axis=1)
    # Invalid entries go last in the sort.
    ordered = np.sort(np.where(ok, vals, np.inf), axis=1)
    pos = (np.maximum(n_ok, 1) - 1) * (pct / 100.0)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, np.maximum(n_ok, 1) - 1)
    frac = pos - lo
    rows = np.arange(len(vals))
    below, above = ordered[rows, lo], ordered[rows, hi]
    # Rows without a valid entry: avoid inf - inf.
    below = np.where(n_ok > 0, below, 0)
    above = np.where(n_ok > 0, above, 0)
    gap = above - below
    res = np.where(frac >= 0.5, above - gap * (1.0 - frac), below + gap * frac)
    return np.where(n_ok > 0, res, np.nan).astype(vals.dtype)


def _block_percentile(
    pixels: np.ndarray, valid: np.ndarray | None, pct: float, sub: int
) -> np.ndarray:
    """Take the percentile of the (sub-sampled) valid pixels of every block.

    The same statistic is used whatever the validity of the window: blocks
    without an invalid pixel go through ``np.percentile`` on the sub-sampled
    pixels, blocks with some invalid pixels use the same sub-sampling pattern and
    the same interpolation on their valid pixels only. One invalid pixel
    therefore changes only its own block.

    Args:
        pixels: Pixels per block [by, bx, block * block].
        valid: Validity per pixel, same shape, or None (all valid).
        pct: Percentile in percent.
        sub: Sub-sampling stride within a block.

    Returns:
        Array [by, bx]; NaN for blocks without any valid pixel.
    """
    sampled = pixels[:, :, ::sub] if sub > 1 else pixels
    if valid is None:
        whole: np.ndarray = np.percentile(sampled, pct, axis=2)
        return whole
    valid_sampled = valid[:, :, ::sub] if sub > 1 else valid
    n_ok = valid_sampled.sum(axis=2)
    is_full = n_ok == sampled.shape[2]
    is_partial = (n_ok > 0) & ~is_full
    out = np.full(sampled.shape[:2], np.nan, np.float64)
    if is_full.any():
        out[is_full] = np.percentile(sampled[is_full], pct, axis=1)
    if is_partial.any():
        out[is_partial] = _masked_percentile_rows(
            sampled[is_partial], valid_sampled[is_partial], pct
        )
    return out.astype(pixels.dtype)


def _percentile_filter_nan(grid: np.ndarray, pct: float, size: int) -> np.ndarray:
    """Apply ``ndi.percentile_filter(grid, pct, size, mode="nearest")`` ignoring NaN.

    Where a neighbourhood contains no NaN the result is exactly scipy's (same
    rank rule). Elsewhere the rank is taken among the valid neighbours only
    (``floor(n_valid x pct / 100)``, scipy's rule); NaN where there is none.

    Args:
        grid: 2-D array that may contain NaN.
        pct: Percentile in percent.
        size: Edge of the square filter window.

    Returns:
        Filtered array; NaN where the window holds no valid entry.
    """
    is_nan = np.isnan(grid)
    out: np.ndarray = ndi.percentile_filter(
        np.where(is_nan, 0, grid), pct, size=size, mode="nearest"
    )
    if not is_nan.any():
        return out
    radius = size // 2
    touched = ndi.maximum_filter(is_nan.astype(np.uint8), size=size, mode="nearest") > 0
    ys, xs = np.nonzero(touched)
    padded = np.pad(grid, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    windows = windows[ys, xs].reshape(len(ys), -1)
    n_ok = (~np.isnan(windows)).sum(axis=1)
    ordered = np.sort(windows, axis=1)  # NaN sorts last
    rank = np.minimum((n_ok * pct) // 100, np.maximum(n_ok - 1, 0)).astype(np.intp)
    res = ordered[np.arange(len(ys)), rank]
    out = out.astype(np.float64 if out.dtype == np.float64 else np.float32, copy=True)
    out[ys, xs] = np.where(n_ok > 0, res, np.nan)
    return out


def fill_undefined_blocks(grid: np.ndarray, reach: int) -> np.ndarray:
    """Give NaN blocks the value of the nearest defined block within ``reach``.

    Distance is the chessboard distance in blocks; blocks beyond it get 0. The
    fill is a pure function of the neighbourhood (never of a window statistic),
    so a block that can influence a valid pixel receives the same value whatever
    the window around it. Blocks farther than ``reach`` from every defined block
    cannot influence a valid pixel (see :func:`lower_envelope`) and only serve
    the pixels that are invalid anyway.

    Public because the GPU backend must reproduce exactly this fill.

    Args:
        grid: 2-D block grid; NaN marks an undefined block.
        reach: Maximum distance (blocks) over which a value is propagated.

    Returns:
        Grid without NaN, same dtype.
    """
    undefined = np.isnan(grid)
    if not undefined.any():
        return grid
    if undefined.all():
        return np.zeros_like(grid)
    dist, idx = ndi.distance_transform_cdt(
        undefined, metric="chessboard", return_indices=True
    )
    return np.where(dist <= reach, grid[idx[0], idx[1]], 0).astype(grid.dtype)


def lower_envelope(
    img: np.ndarray,
    valid: np.ndarray | None = None,
    block: int = ENVELOPE_BLOCK_PX,
    win_blocks: int = 9,
    pct: float = 20.0,
    sub: int = 1,
) -> np.ndarray:
    """Compute a smooth lower envelope (a local low percentile) of a plane.

    The steps are: block percentile, percentile filter over the block grid,
    gaussian smoothing, bilinear interpolation back to pixels.

    The block grid starts at pixel 0 of ``img``; a caller that wants
    chunk-independent results passes windows whose absolute origin is a multiple
    of ``block``.

    Invalid pixels are handled locally so that a valid pixel's envelope never
    depends on the rest of the window: the block percentile uses the valid pixels
    of each block only (the same statistic and sub-sampling as for a fully valid
    window), the block-grid percentile filter ignores blocks without a valid
    pixel, and blocks that remain undefined are filled from the nearest defined
    block within a fixed reach. That reach plus the filter radius covers the
    Gaussian smoothing and the bilinear interpolation
    (``ENVELOPE_GAUSS_REACH_BLOCKS + 1`` blocks); blocks beyond it get 0 and can
    only influence pixels whose own block has no valid pixel within the filter
    radius, i.e. invalid output pixels. A fully valid window takes exactly the
    original computation.

    Args:
        img: 2-D plane.
        valid: Optional boolean mask of usable pixels.
        block: Block size in px.
        win_blocks: Size (blocks) of the percentile filter over the block grid.
        pct: Percentile within a block.
        sub: Sub-sampling of the pixels within a block (4 for the faint stains:
            speed).

    Returns:
        float32 array of the same shape.
    """
    height, width = img.shape
    n_by, n_bx = height // block, width // block

    def to_blocks(plane: np.ndarray) -> np.ndarray:
        cropped = plane[: n_by * block, : n_bx * block]
        return (
            cropped.reshape(n_by, block, n_bx, block)
            .transpose(0, 2, 1, 3)
            .reshape(n_by, n_bx, -1)
        )

    pixels = to_blocks(img)
    valid_blocks = None
    if valid is not None and not bool(valid.all()):
        valid_blocks = to_blocks(valid)
    grid = _block_percentile(pixels, valid_blocks, pct, sub)
    if valid_blocks is None:
        grid = ndi.percentile_filter(
            grid, ENVELOPE_FILTER_PCT, size=win_blocks, mode="nearest"
        )
    else:
        grid = _percentile_filter_nan(grid, ENVELOPE_FILTER_PCT, win_blocks)
        reach = max(1, ENVELOPE_GAUSS_REACH_BLOCKS + 1 - win_blocks // 2)
        grid = fill_undefined_blocks(grid, reach)
    grid = gaussian_filter(grid, ENVELOPE_GAUSS_SIGMA_BLOCKS, mode="nearest")
    yy = np.clip((np.arange(height) + 0.5) / block - 0.5, 0, n_by - 1)
    xx = np.clip((np.arange(width) + 0.5) / block - 0.5, 0, n_bx - 1)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    envelope: np.ndarray = ndi.map_coordinates(grid, [gy, gx], order=1, mode="nearest")
    return envelope.astype(np.float32)


def huber1(e: np.ndarray, sig: float | np.ndarray, k: float) -> np.ndarray:
    """Compute one-sided Huber weights.

    Positive residuals above ``k * sig`` (tissue, unmodelled neighbours) are
    down-weighted; negative ones keep full weight.

    Args:
        e: Residuals (data minus model).
        sig: Noise sd (scalar or array broadcastable to ``e``).
        k: Huber threshold in units of ``sig``.

    Returns:
        float32 weights in (0, 1].
    """
    t = e / (k * sig)
    return np.where(t > 1.0, 1.0 / np.maximum(t, 1.0), 1.0).astype(np.float32)


def window_bounds(
    shape: tuple[int, int], yc: float, xc: float, half: float
) -> tuple[int, int, int, int]:
    """Clip a square window around a point to the image.

    Args:
        shape: Image shape (height, width).
        yc: Centre row.
        xc: Centre column.
        half: Half size of the window in px.

    Returns:
        Integer window ``(y0, y1, x0, x1)`` with exclusive ends.
    """
    height, width = shape
    y0 = max(0, int(np.floor(yc - half)))
    y1 = min(height, int(np.ceil(yc + half)) + 1)
    x0 = max(0, int(np.floor(xc - half)))
    x1 = min(width, int(np.ceil(xc + half)) + 1)
    return y0, y1, x0, x1


def radius_grid(yc: float, xc: float, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Compute the distance of every pixel of a window to a centre.

    Args:
        yc: Centre row.
        xc: Centre column.
        y0: First window row.
        y1: End window row (exclusive).
        x0: First window column.
        x1: End window column (exclusive).

    Returns:
        float32 array of shape (y1 - y0, x1 - x0).
    """
    dy = (np.arange(y0, y1, dtype=np.float32) - np.float32(yc))[:, None]
    dx = (np.arange(x0, x1, dtype=np.float32) - np.float32(xc))[None, :]
    dist: np.ndarray = np.sqrt(dy * dy + dx * dx)
    return dist


# ----------------------------------------------------------------------------
# detection at z0
_OFFSET_Y, _OFFSET_X = np.mgrid[-20:21, -20:21]
_OFFSET_R = np.hypot(_OFFSET_Y, _OFFSET_X)
_ANNULUS = (_OFFSET_R >= 14) & (_OFFSET_R <= 20)
_ANNULUS_Y, _ANNULUS_X = _OFFSET_Y[_ANNULUS], _OFFSET_X[_ANNULUS]
_CENTRE3_Y, _CENTRE3_X = (g.ravel() for g in np.mgrid[-1:2, -1:2])


def centre_minus_bg(
    img: np.ndarray, py: np.ndarray, px: np.ndarray, pad: np.ndarray | None = None
) -> np.ndarray:
    """Measure counts above background at integer positions.

    The measure is the 3x3 centre mean minus the median of the annulus
    r 14-20 px.

    Args:
        img: 2-D plane.
        py: Rows of the positions.
        px: Columns of the positions.
        pad: Optional ``img`` edge-padded by 20 px (to share it between calls).

    Returns:
        Array with one value per position.
    """
    py = np.asarray(py)
    px = np.asarray(px)
    if pad is None:
        pad = np.pad(img, 20, mode="edge")
    centre = pad[py[:, None] + 20 + _CENTRE3_Y, px[:, None] + 20 + _CENTRE3_X].mean(1)
    background = np.median(
        pad[py[:, None] + 20 + _ANNULUS_Y, px[:, None] + 20 + _ANNULUS_X], axis=1
    )
    contrast: np.ndarray = centre - background
    return contrast


def nms(y: np.ndarray, x: np.ndarray, score: np.ndarray, radius: float) -> np.ndarray:
    """Suppress non-maxima greedily by score within ``radius`` (stable order).

    Args:
        y: Rows of the points.
        x: Columns of the points.
        score: Score per point; the highest-scoring point of a neighbourhood
            survives.
        radius: Suppression radius in px.

    Returns:
        Boolean keep mask.
    """
    order = np.argsort(-np.asarray(score), kind="stable")
    pts = np.c_[y, x]
    tree = cKDTree(pts)
    keep = np.zeros(len(pts), bool)
    dead = np.zeros(len(pts), bool)
    for i in order:
        if dead[i]:
            continue
        keep[i] = True
        dead[tree.query_ball_point(pts[i], radius)] = True
    return keep


# Ring test of the rescue path (and the calibration's reference set).
_SectorLayout = tuple[int, np.ndarray, np.ndarray, np.ndarray]
_SECTOR_CACHE: dict[tuple[float, int], _SectorLayout] = {}


def _sector_index(r_ring: float, nsec: int = 16) -> _SectorLayout:
    """Return the cached sector layout of a ring of radius ``r_ring``.

    Args:
        r_ring: Ring radius (px).
        nsec: Number of angular sectors.

    Returns:
        (half window R, flat ring-band mask, flat flank mask, flat sector index) of
        the (2R + 1) x (2R + 1) window around a centre.
    """
    key = (round(float(r_ring), 1), nsec)
    if key not in _SECTOR_CACHE:
        w = max(4.0, 0.25 * r_ring)
        half = int(np.ceil(r_ring + 1.5 * w)) + 1
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
        r = np.hypot(yy, xx)
        sec = ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
        on = np.abs(r - r_ring) <= w / 2
        flank = ((r >= r_ring - 1.5 * w) & (r <= r_ring - w)) | (
            (r >= r_ring + w) & (r <= r_ring + 1.5 * w)
        )
        _SECTOR_CACHE[key] = (half, on.ravel(), flank.ravel(), sec.ravel())
    return _SECTOR_CACHE[key]


def ring_tscores(
    img: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    r_ring: float,
    nsec: int = 16,
) -> np.ndarray:
    """Sector t-score of a defocused ring at radius ``r_ring`` around each centre.

    In each of ``nsec`` angular sectors the mean of the ring band
    (|r - r_ring| <= w / 2) minus the mean of the two flank bands, with
    w = max(4, 0.25 r_ring); t = mean / (sd / sqrt(nsec)). A bead ring is present at
    every azimuth (high t); tissue structure is not.

    Args:
        img: 2-D plane.
        ys: Centre rows (px).
        xs: Centre columns (px).
        r_ring: Ring radius (px).
        nsec: Number of sectors.

    Returns:
        float array of t-scores; NaN where the window leaves the image or holds
        non-finite values.
    """
    half, on, flank, sec = _sector_index(r_ring, nsec)
    height, width = img.shape
    t = np.full(len(ys), np.nan)
    on_count = np.bincount(sec[on], minlength=nsec).astype(float)
    flank_count = np.bincount(sec[flank], minlength=nsec).astype(float)
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
        d = np.bincount(sec[on], w[on], minlength=nsec) / np.maximum(
            on_count, 1
        ) - np.bincount(sec[flank], w[flank], minlength=nsec) / np.maximum(
            flank_count, 1
        )
        sd = d.std(ddof=1)
        t[i] = d.mean() / (sd / np.sqrt(nsec) + 1e-6)
    return t


@dataclass
class Sources:
    """Detected sources of one window (positions in window pixel coordinates).

    Attributes:
        y: Sub-pixel centre rows.
        x: Sub-pixel centre columns.
        A0: Detection amplitude (counts above local background at z0).
        a: Fitted z0 amplitude (equivalent point-source amplitude).
        is_bead: Classified as a bead (and not pruned).
        rD: DAPI (z0 - z1) / A0 ratio.
        r1: PolyT z1 / z0 centre ratio.
        blob: Extended source (modelled with a source map).
        u: Blob index -> (y0, x0, amplitude-density map).
        ell: Log-stretch at plane 1 (field value at the source).
        sel: Compact singles that sample the stretch field.
        q: Aperture flux / point-source flux.
        g_pre: Consensus gain (diagnostic).
        t1: z1 ring t-score (diagnostic).
        rring: z0 ring ratio (diagnostic).
        w_pre: Information weight (diagnostic).
        cand_bead: Classified as a bead before pruning (``is_bead`` loses the
            pruned ones).
        rescued: Accepted by the z1-ring rescue although the z0 ring test failed.
        sd1_rel: z1 null sd / A0, the local tissue texture relative to the
            amplitude (the gate of the rescue, a calibration reference feature).
    """

    y: np.ndarray
    x: np.ndarray
    A0: np.ndarray  # noqa: N815
    a: np.ndarray
    is_bead: np.ndarray
    rD: np.ndarray  # noqa: N815
    r1: np.ndarray
    blob: np.ndarray
    u: dict[int, Any] = field(default_factory=dict)
    ell: np.ndarray | None = None
    sel: np.ndarray | None = None
    q: np.ndarray | None = None
    g_pre: np.ndarray | None = None
    t1: np.ndarray | None = None
    rring: np.ndarray | None = None
    w_pre: np.ndarray | None = None
    cand_bead: np.ndarray | None = None
    rescued: np.ndarray | None = None
    sd1_rel: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Number of sources."""
        return len(self.y)


def empty_sources() -> Sources:
    """Return a :class:`Sources` object with no source."""
    zeros = np.zeros(0)
    return Sources(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros.astype(bool),
        zeros,
        zeros,
        zeros.astype(bool),
    )


def null_sd(
    img: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    n: int = 24,
    img2: np.ndarray | None = None,
) -> np.ndarray:
    """Measure the tissue-texture noise floor of the amplitude statistic.

    This is the per-source sd of the centre-minus-annulus statistic at ``n``
    control positions 60-140 px away (a deterministic golden-angle pattern).

    Args:
        img: 2-D plane.
        py: Rows of the sources.
        px: Columns of the sources.
        n: Number of control positions per source.
        img2: Optional second plane; its statistic is subtracted from the first
            (z0 minus z1 contrast).

    Returns:
        Array with one robust sd (1.4826 x MAD) per source.
    """
    pad = np.pad(img, 20, mode="edge")
    k = np.arange(n)
    angle = (k * 2.399963229728653) % (2 * np.pi)
    radius = 60.0 + 80.0 * ((k * 0.618034) % 1.0)
    dy = np.round(radius * np.sin(angle)).astype(int)
    dx = np.round(radius * np.cos(angle)).astype(int)
    height, width = img.shape
    qy = np.clip(np.asarray(py)[:, None] + dy[None, :], 0, height - 1).ravel()
    qx = np.clip(np.asarray(px)[:, None] + dx[None, :], 0, width - 1).ravel()
    stat = centre_minus_bg(img, qy, qx, pad)
    if img2 is not None:
        stat = stat - centre_minus_bg(img2, qy, qx)
    stat = stat.reshape(len(py), n)
    med = np.median(stat, axis=1, keepdims=True)
    sd: np.ndarray = 1.4826 * np.median(np.abs(stat - med), axis=1)
    return sd


def ring_ratio(
    img: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    rin: float = 8.0,
    rout: float = 15.0,
    bg_in: float = 18.0,
    bg_out: float = 24.0,
) -> np.ndarray:
    """Measure the ring amplitude of every source.

    The amplitude is the maximum of the azimuthal profile over r in
    [``rin``, ``rout``] minus the median of a local background annulus.

    Args:
        img: 2-D plane.
        ys: Source rows (sub-pixel).
        xs: Source columns (sub-pixel).
        rin: Inner radius of the ring search in px.
        rout: Outer radius of the ring search in px.
        bg_in: Inner radius of the background annulus in px.
        bg_out: Outer radius of the background annulus in px.

    Returns:
        Ring amplitude in counts per source.
    """
    half = int(np.ceil(bg_out)) + 1
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    out = np.full(len(ys), np.nan)
    pad = np.pad(img, half, mode="edge")
    for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
        iy, ix = int(round(y)), int(round(x))
        window = pad[iy : iy + 2 * half + 1, ix : ix + 2 * half + 1]
        r = np.hypot(yy - (y - iy), xx - (x - ix))
        background = np.median(window[(r >= bg_in) & (r <= bg_out)])
        idx = np.floor(r).astype(int)
        in_ring = (idx >= int(rin)) & (idx <= int(rout))
        counts = np.bincount(idx[in_ring] - int(rin), minlength=int(rout - rin) + 1)
        profile = np.bincount(
            idx[in_ring] - int(rin), window[in_ring], minlength=int(rout - rin) + 1
        ) / np.maximum(counts, 1)
        out[i] = profile.max() - background
    return out


def detect_sources(
    polyt: PlaneStack,
    dapi: PlaneStack | None,
    *,
    dog_noise: float | None,
    thresholds: ClassifierThresholds,
    params: PsfSubtractParams = DEFAULT_PARAMS,
) -> Sources:
    """Detect and classify bead candidates on z0 of the reference stain.

    Args:
        polyt: Reference-stain planes; planes 0 and 1 are read (array
            [nz, Y, X] or {z: plane}), and planes 2 .. ``rescue_zprof_zmax`` for
            the rescue path.
        dapi: DAPI planes 0 and 1 for the spectral test, or None (calibration of
            a stain-less test only).
        dog_noise: Dataset-global DoG noise (the detection threshold is
            ``k_sigma`` x this). None estimates it from this window (calibration
            tiles only; production passes the global value).
        thresholds: Per-dataset classifier thresholds.
        params: Fixed parameters.

    Returns:
        Sources with ``is_bead`` set; candidates within ``edge_margin_px`` of an
        invalid pixel are dropped. ``rescued`` marks the beads that failed the z0
        ring test and were accepted on their z1 ring (``t1`` holds that ring's
        t-score for the candidates that reached the ring test, NaN elsewhere).
    """
    p = params
    p0 = np.asarray(polyt[0], np.float32)
    valid = p0 > 0
    dog = gaussian_filter(p0, 1) - gaussian_filter(p0, 3)
    noise = dog_noise_of(p0, valid) if dog_noise is None else float(dog_noise)
    peaks = peak_local_max(
        dog,
        min_distance=p.peak_min_distance,
        threshold_abs=p.k_sigma * noise,
        exclude_border=p.border,
    )
    if len(peaks) and not bool(valid.all()) and p.edge_margin_px > 0:
        # Zeros outside the mosaic create a step edge that the DoG and the
        # annulus statistics misread. The margin is measured from invalid pixels
        # only: the window border is not an edge ("nearest"), so a window with
        # one invalid pixel keeps the candidates that a fully valid window keeps.
        inner = ndi.minimum_filter(
            valid.astype(np.uint8), size=2 * p.edge_margin_px + 1, mode="nearest"
        )
        peaks = peaks[inner[peaks[:, 0], peaks[:, 1]] > 0]
    if len(peaks) == 0:
        return empty_sources()
    py, px = peaks[:, 0], peaks[:, 1]
    amp0 = centre_minus_bg(p0, py, px)
    keep = nms(py, px, amp0, p.nms_px)
    py, px, amp0 = py[keep], px[keep], amp0[keep]
    smooth = gaussian_filter(p0, 1.0)
    height, width = p0.shape
    sy, sx = py.astype(float).copy(), px.astype(float).copy()
    # Sub-pixel position: parabola through the three smoothed pixels around the
    # peak along each axis, clipped so that a noisy peak cannot move far.
    inside = (py > 0) & (py < height - 1) & (px > 0) & (px < width - 1)
    for i in np.where(inside)[0]:
        y, x = py[i], px[i]
        before, centre, after = smooth[y - 1, x], smooth[y, x], smooth[y + 1, x]
        curvature = before - 2 * centre + after
        if curvature < 0:
            sy[i] += np.clip(0.5 * (before - after) / curvature, -0.7, 0.7)
        before, centre, after = smooth[y, x - 1], smooth[y, x], smooth[y, x + 1]
        curvature = before - 2 * centre + after
        if curvature < 0:
            sx[i] += np.clip(0.5 * (before - after) / curvature, -0.7, 0.7)
    p1 = np.asarray(polyt[1], np.float32)
    r1 = centre_minus_bg(p1, py, px) / np.maximum(amp0, 1.0)
    # DAPI bead signal: (a) z0 centre-minus-annulus contrast, (b) z0 minus z1
    # contrast (cancels smooth tissue nuclei, keeps the sharp bead dot). A source
    # passes the spectral test if either is compatible with a bead.
    if dapi is not None:
        d0 = np.asarray(dapi[0], np.float32)
        d1 = np.asarray(dapi[1], np.float32)
        c0 = centre_minus_bg(d0, py, px)
        c1 = centre_minus_bg(d1, py, px)
        r_dapi0 = c0 / np.maximum(amp0, 1.0)
        r_dapi = (c0 - c1) / np.maximum(amp0, 1.0)
    else:
        r_dapi0 = r_dapi = np.full(len(py), np.nan)
    # Spectral / z1 classification; tolerances grow with the local tissue texture
    # (sd of the high-pass image / A0).
    amp0_safe = np.maximum(amp0, 1.0)
    rring = ring_ratio(p0, sy, sx) / amp0_safe
    sd1 = null_sd(p1, py, px)
    th = thresholds
    is_bead = (amp0 >= p.a0_min) & (r1 < np.maximum(th.r1_max, 3.5 * sd1 / amp0_safe))
    if dapi is not None:
        sd_dapi0 = null_sd(d0, py, px)
        sd_dapi = null_sd(d0, py, px, img2=d1)
        ok0 = np.abs(r_dapi0 - th.rd0_center) <= np.maximum(
            th.rd_tol, 3.5 * sd_dapi0 / amp0_safe
        )
        ok_diff = np.abs(r_dapi - th.rd_center) <= np.maximum(
            th.rd_tol, 3.5 * sd_dapi / amp0_safe
        )
        is_bead &= ok0 | ok_diff
    ring_ok = (rring >= th.ring_min) & (rring <= th.ring_max)
    sd1_rel = sd1 / amp0_safe
    rescued = np.zeros(len(py), bool)
    ring_t1 = np.full(len(py), np.nan)
    if th.rescue_active:
        # Rescue path: a source that is spectrally and in z1 a bead but whose z0
        # ring is disturbed (neighbouring beads, a cluster) is accepted when
        #  (1) its amplitude is large enough for a ring test,
        #  (2) the tissue texture around it is low relative to its amplitude
        #      (sd1_rel <= rescue_sd1_max: the rescue is for beads in flat
        #      surroundings, a textured neighbourhood is where a tissue object can
        #      mimic a z1 ring),
        #  (3) it has no secondary focus at z2 .. zmax (a bead is focused at z0
        #      only; planes missing from the stack are skipped),
        #  (4) its raw z1 ring is coherent at every azimuth (a bead signature that
        #      tissue puncta, focused at z0, do not have).
        # The tests run cheap first; the decision does not depend on the order.
        cand = np.where(
            is_bead
            & ~ring_ok
            & (rring < th.ring_min)
            & (amp0 >= th.rescue_a0_min)
            & (sd1_rel <= th.rescue_sd1_max)
        )[0]
        if len(cand) and th.rescue_zprof_zmax >= 2:
            for z in range(2, int(th.rescue_zprof_zmax) + 1):
                if not len(cand):
                    break
                if isinstance(polyt, Mapping):
                    plane_z = polyt.get(z)
                else:
                    plane_z = polyt[z] if z < len(polyt) else None
                if plane_z is None:
                    continue
                plane_z = np.asarray(plane_z, np.float32)
                contrast = (
                    centre_minus_bg(plane_z, py[cand], px[cand]) / amp0_safe[cand]
                )
                tol = np.maximum(
                    th.r1_max,
                    3.5 * null_sd(plane_z, py[cand], px[cand]) / amp0_safe[cand],
                )
                cand = cand[contrast <= tol]
        if len(cand):
            # Measured at the detection centre (the refined centre of the fit is
            # not known here).
            t_cand = ring_tscores(p1, sy[cand], sx[cand], th.rescue_ring_radius_px)
            ring_t1[cand] = t_cand
            rescued[cand[np.nan_to_num(t_cand, nan=-1e9) >= th.rescue_tz1_min]] = True
    is_bead = (is_bead & ring_ok) | rescued
    sources = Sources(
        sy,
        sx,
        amp0.astype(np.float64),
        amp0.astype(np.float64).copy(),
        is_bead,
        r_dapi,
        r1,
        np.zeros(len(py), bool),
    )
    sources.rring = rring
    sources.rescued = rescued
    sources.sd1_rel = sd1_rel
    sources.t1 = ring_t1
    logger.debug(
        "[BEADS] detect: %d candidates, %d beads (%d rescued)",
        sources.n,
        int(is_bead.sum()),
        int(rescued.sum()),
    )
    return sources
