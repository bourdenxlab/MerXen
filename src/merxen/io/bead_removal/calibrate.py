"""Per-dataset bead calibration: the calibration pass and its data class.

:func:`calibrate_bead_model` is the once-per-dataset pass that runs before any chunk is
corrected. Everything a chunk needs (the PSF library, the classifier thresholds, the
noise and amplitude constants, the stain responses) is decided here, never inside a
chunk. A dataset that cannot be corrected raises :class:`BeadCalibrationError` (no
beads, ambiguous, beads not focused, no tissue-free area).

The classifier thresholds are derived per dataset from a reference set of bright,
isolated, spectrally confirmed beads and checked on held-out sample-grid tiles. The
held-out recall is reported and warns as decided by the owner: the cluster-inclusive
recall warns below 0.75 when it rests on at least 50 beads; the isolated recall warns
below 0.90 only when it rests on at least 20 beads. Datasets whose beads fail the z0
ring test in large numbers (strongly varying focus, dense clusters) keep those beads
uncorrected; the warning says so.

A gated rescue recovers part of those beads: a candidate that fails the z0 ring test on
the low side is accepted on a coherent raw z1 ring when the texture around it is low
(``rescue_sd1_max``, the P98 of the isolated reference beads clipped to
``RESCUE_SD1_RANGE``) and it has no secondary focus at z2..z6. The rescue is off for
calibrations without the gate and when too few texture measurements exist.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

from merxen.config import MerscopeBeadRemovalConfig
from merxen.io.bead_removal.detect import (
    ClassifierThresholds,
    centre_minus_bg,
    detect_sources,
    dog_noise_of,
    huber1,
    lower_envelope,
    nms,
    null_sd,
    ring_ratio,
    ring_tscores,
    robust_noise,
    window_bounds,
)
from merxen.io.bead_removal.params import (
    BEAD_REMOVAL_VERSION,
    DEFAULT_PARAMS,
    PsfSubtractParams,
    params_sha256,
)
from merxen.io.bead_removal.psf import (
    PsfLibrary,
    Template,
    derive_high_templates,
    to_jsonable,
)

logger = logging.getLogger(__name__)

#: Default pixel size of the mosaics (px per micron) when the transform file next to
#: the images cannot be read.
DEFAULT_PX_PER_UM = 9.2593
#: Warn when the forecast wall time of the chunk pass (resolved backend) exceeds this.
FORECAST_WARN_WALL_H = 12.0
#: A window is fitted with its halo: the beads of the halo are fitted twice (once per
#: neighbouring window), so the fitted bead count is this factor times the number of
#: beads in the cores (measured on a development mosaic: 301,007 fitted vs 244,861 in
#: the cores = 1.23).
FORECAST_HALO_FACTOR = 1.23
#: CPU cost model: (workers, CPU-s per bead in the cores, halo re-fits included)
#: measured on a development mosaic. 5 / 32 workers: the 52-chunk scaling runs; 8: the
#: fill-off timing run; 16: the full-mosaic run (169 chunks; the 52-chunk subset gave
#: 0.199, so the 5- and 8-worker points may under-estimate a whole mosaic by up to
#: ~25 %). The cost per bead rises with the number of simultaneous workers (shared
#: caches / memory bandwidth of the host); between the points the model interpolates
#: linearly, below 5 workers it keeps the 5-worker value, above 32 it extrapolates flat.
FORECAST_CPU_S_PER_BEAD = ((5, 0.106), (8, 0.116), (16, 0.250), (32, 0.481))
#: Fraction of the ideal wall time (CPU-s / workers) that is actually reached:
#: (workers, efficiency), measured as above.
FORECAST_CPU_EFFICIENCY = ((5, 0.94), (16, 0.97), (32, 0.88))
#: GPU cost model: seconds per window = a + b x fitted beads per window, through two
#: measured points (fitted beads per window incl. halo, seconds per 4096 px core window
#: on a quiet host): a typical-density development mosaic (mean window, 1,630 beads in
#: the cores x 1.23) and a dense, bright-bead one (6,200 fitted beads).
FORECAST_GPU_POINTS = ((1630.0 * FORECAST_HALO_FACTOR, 5.7), (6200.0, 13.6))
#: Core edge of a chunk (px) used to count the windows of the GPU model.
FORECAST_CORE_PX = 4096
#: Minimum number of reference beads for the per-dataset classifier thresholds (else
#: the defaults are kept).
MIN_REFERENCE_BEADS = 50
#: Focus check: median z1 / z0 centre ratio of bright classified beads must stay below
#: this.
FOCUS_R1_MAX = 0.30
#: Edge (px) of the dataset-global sample tiles (the 4 x 4 grid over the valid
#: bounding box).
GRID_TILE_PX = 1024
#: Grid size (n x n tiles of ``GRID_TILE_PX`` over the valid bounding box) of the
#: classifier reference / held-out recall tiles. 4 = the sample grid itself (default).
#: Diagnostic only: a denser grid gives more reference beads in sparse-bead datasets.
REFERENCE_GRID_N = 4
#: Minimum amplitude (counts) of a reference bead / of a candidate in the focus check.
REFERENCE_A0_MIN = 1000.0
#: Held-out recall of the isolated reference beads below this raises the QC warning,
#: but only when it rests on at least ``MIN_RECALL_REFERENCE`` beads (fewer: reported,
#: not warned as low).
RECALL_WARN_BELOW = 0.90
MIN_RECALL_REFERENCE = 20
#: Warn when the cluster-inclusive held-out recall is below this and rests on at least
#: ``MIN_RECALL_WITH_CLUSTERS_REFERENCE`` beads. The isolated reference set misses the
#: beads that fail in clusters, which is where a classifier problem shows.
RECALL_WITH_CLUSTERS_WARN_BELOW = 0.75
MIN_RECALL_WITH_CLUSTERS_REFERENCE = 50
#: Rescue path: minimum sector t-score of the raw z1 ring (the reference-set test of
#: ``confirmed_mask`` uses 3.0; the rescue is one unit stricter).
RESCUE_TZ1_MIN = 4.0
#: Texture gate of the rescue: ``rescue_sd1_max`` is the P98 of ``sd1_rel`` (z1 null
#: sd / A0) of the isolated reference beads that set ``r1_max``, clipped to this range
#: (fixed in advance). Below the range nothing is flat enough to be told from noise;
#: above it a rescue would be accepted in textured tissue.
RESCUE_SD1_RANGE = (0.02, 0.08)
#: Highest plane of the rescue's "no secondary focus" test (planes 2..k are read when
#: all of them exist).
RESCUE_ZPROF_ZMAX = 6
#: Two reference beads closer than this (px, mosaic coordinates) are the same bead seen
#: through two overlapping tiles.
SAME_BEAD_TOL_PX = 3.0
#: Planes of the per-tile templates (the library derives z4..z7 from z3), reported in
#: the library spread.
LIBRARY_SPREAD_PLANES = (1, 2, 3)

Backend = Literal["cpu", "cuda"]


class BeadCalibrationError(RuntimeError):
    """The dataset cannot be corrected.

    Raised for: no beads, ambiguous bead presence, beads not focused at plane 0, no
    tissue-free area, or unusable input.

    Attributes:
        report: Per-tile table / measured values for logging (JSON-able dict).
    """

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report: dict[str, Any] = report or {}


@dataclass
class BeadCalibration:
    """Everything a chunk needs from the dataset-level calibration pass.

    No chunk decides anything on its own: the library, thresholds and constants below
    are read-only inputs of the chunk pass.

    Attributes:
        version: BEAD_REMOVAL_VERSION the calibration was made for.
        params_sha256: Hash of the PsfSubtractParams used.
        corrected_stains: Stains that are corrected (v1: PolyT, DAPI).
        reference_stain: Stain carrying the bead model (PolyT).
        library: PSF library z0..z7.
        thresholds: Per-dataset classifier thresholds.
        noise: Reference-stain pixel noise per plane (counts).
        dog_noise_z0: Robust sd of the DoG(1, 3 px) response of the reference z0.
        a0_sel: (low, high) bounds of the amplitude range of singles sampling the
            stretch field.
        a0_median: Median bead amplitude (blob candidacy).
        ell_default: Stretch value for sources without enough neighbours.
        rho: Stain -> (bead response relative to the reference, standard error);
            insignificant values are 0.
        tiles: Per calibration tile record (QC).
        decision: Bead-presence decision, margins, focus check, classifier reference
            set, library spread, warnings (QC).
        tissue: Tissue threshold / fraction, candidate-tile statistics (QC).
        forecast: Cost forecast (QC).
    """

    version: int
    params_sha256: str
    corrected_stains: list[str]
    reference_stain: str
    library: PsfLibrary
    thresholds: ClassifierThresholds
    noise: dict[int, float]
    dog_noise_z0: float
    a0_sel: tuple[float, float]
    a0_median: float
    ell_default: float
    rho: dict[str, tuple[float, float]]
    tiles: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    tissue: dict[str, Any] = field(default_factory=dict)
    forecast: dict[str, Any] = field(default_factory=dict)

    def calibration_sha256(self) -> str:
        """Hash of everything that decides how a chunk is corrected.

        The QC records (tiles, decision, tissue, forecast, timings) are left out, so
        the hash is the same for the same calibrated model however many workers made
        it and survives a ``save`` / ``load`` round trip. Provenance and the
        segmentation sidecar compare it to tell whether two builds used the same
        calibration.

        Returns:
            Hex SHA-256 digest.
        """
        digest = hashlib.sha256()
        meta = {
            "version": int(self.version),
            "params_sha256": self.params_sha256,
            "corrected_stains": list(self.corrected_stains),
            "reference_stain": self.reference_stain,
            "thresholds": self.thresholds.to_hash_dict(),
            "noise": {str(k): float(v) for k, v in sorted(self.noise.items())},
            "dog_noise_z0": float(self.dog_noise_z0),
            "a0_sel": [float(self.a0_sel[0]), float(self.a0_sel[1])],
            "a0_median": float(self.a0_median),
            "ell_default": float(self.ell_default),
            "rho": {k: [float(v[0]), float(v[1])] for k, v in sorted(self.rho.items())},
        }
        digest.update(json.dumps(meta, sort_keys=True).encode())
        for tmpl in self.library.tmpls:
            digest.update(repr((float(tmpl.ds), float(tmpl.r_sup))).encode())
            digest.update(np.ascontiguousarray(tmpl.vals, dtype="<f8").tobytes())
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Describe the calibration as JSON-able data.

        The library is summarised; its arrays are written by :meth:`save`.

        Returns:
            Dictionary of the calibration fields plus ``calibration_sha256``.
        """
        lib = self.library.to_dict()
        lib["info"] = {
            k: v for k, v in lib.get("info", {}).items() if k != "tile_templates"
        }
        out: dict[str, Any] = to_jsonable(
            {
                "version": self.version,
                "params_sha256": self.params_sha256,
                "calibration_sha256": self.calibration_sha256(),
                "corrected_stains": list(self.corrected_stains),
                "reference_stain": self.reference_stain,
                "library": lib,
                "thresholds": self.thresholds.to_dict(),
                "noise": {str(k): float(v) for k, v in self.noise.items()},
                "dog_noise_z0": float(self.dog_noise_z0),
                "a0_sel": [float(self.a0_sel[0]), float(self.a0_sel[1])],
                "a0_median": float(self.a0_median),
                "ell_default": float(self.ell_default),
                "rho": {k: [float(v[0]), float(v[1])] for k, v in self.rho.items()},
                "tiles": self.tiles,
                "decision": self.decision,
                "tissue": self.tissue,
                "forecast": self.forecast,
            }
        )
        return out

    def save(self, directory: str | Path) -> None:
        """Write the calibration into a directory.

        Writes ``calibration.json`` and ``psf_library.npz`` (and
        ``tile_templates.npz`` when the per-tile templates are kept).

        Args:
            directory: Output directory (created if needed).
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration.json").write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=True, allow_nan=True)
        )
        self.library.save(out_dir / "psf_library.npz")
        tile_templates = self.library.info.get("tile_templates")
        if tile_templates:
            arrays: dict[str, Any] = {
                f"t{i}_z{k}": np.asarray(v)
                for i, per in tile_templates.items()
                for k, v in per.items()
            }
            np.savez(out_dir / "tile_templates.npz", **arrays)

    @classmethod
    def load(cls, directory: str | Path) -> BeadCalibration:
        """Read a calibration written by :meth:`save`.

        Args:
            directory: Directory holding ``calibration.json`` and
                ``psf_library.npz``.

        Returns:
            The calibration.
        """
        in_dir = Path(directory)
        data = json.loads((in_dir / "calibration.json").read_text())
        library = PsfLibrary.load(in_dir / "psf_library.npz")
        if (in_dir / "tile_templates.npz").exists():
            archive = np.load(in_dir / "tile_templates.npz")
            tile_templates: dict[int, dict[int, np.ndarray]] = {}
            for name in archive.files:
                tile, plane = name[1:].split("_z")
                tile_templates.setdefault(int(tile), {})[int(plane)] = archive[name]
            library.info["tile_templates"] = tile_templates
        return cls(
            version=int(data["version"]),
            params_sha256=data["params_sha256"],
            corrected_stains=list(data["corrected_stains"]),
            reference_stain=data["reference_stain"],
            library=library,
            thresholds=ClassifierThresholds.from_dict(data["thresholds"]),
            noise={int(k): float(v) for k, v in data["noise"].items()},
            dog_noise_z0=float(data["dog_noise_z0"]),
            a0_sel=(float(data["a0_sel"][0]), float(data["a0_sel"][1])),
            a0_median=float(data["a0_median"]),
            ell_default=float(data["ell_default"]),
            rho={k: (float(v[0]), float(v[1])) for k, v in data["rho"].items()},
            tiles=data.get("tiles", []),
            decision=data.get("decision", {}),
            tissue=data.get("tissue", {}),
            forecast=data.get("forecast", {}),
        )


# Process pool helpers.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def _single_thread_env() -> Iterator[None]:
    """Set the BLAS/OpenMP thread variables to 1 for children, then restore them."""
    old = {k: os.environ.get(k) for k in _THREAD_ENV}
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _pool(n_workers: int) -> Iterator[cf.ProcessPoolExecutor | None]:
    """Yield a spawn-context pool of single-threaded workers (None: run inline)."""
    if n_workers <= 1:
        yield None
        return
    with _single_thread_env():
        ex = cf.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
        )
        try:
            yield ex
        finally:
            ex.shutdown(wait=True, cancel_futures=True)


def _map(
    ex: cf.ProcessPoolExecutor | None, fn: Callable[[dict], Any], jobs: Sequence[dict]
) -> list[Any]:
    """Results of ``fn`` over ``jobs`` in job order (exceptions propagate)."""
    if ex is None:
        return [fn(j) for j in jobs]
    return list(ex.map(fn, jobs))


def _open_reader(job: dict) -> Any:
    from .reader import MosaicPlaneReader

    return MosaicPlaneReader(job["images_dir"], job["stains"], job["planes"])


# Overview, tissue mask and distance from tissue.
def build_overview(
    reader: Any, *, reference_stain: str, dapi_stain: str, z_mid: int, factor: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 1/``factor`` DAPI overview and the validity mask of the mosaic.

    The mask is the reference stain's z0 > 0.

    Args:
        reader: MosaicPlaneReader holding both planes.
        reference_stain: Stain whose z0 > 0 marks the valid mosaic area.
        dapi_stain: Stain used for the tissue mask.
        z_mid: DAPI plane (middle of the corrected planes).
        factor: Down-sampling (every ``factor``-th row, mean of ``factor`` columns).

    Returns:
        (dapi overview float32, valid overview bool).
    """
    dapi_ov = reader.overview(dapi_stain, z_mid, factor)
    ref_ov = reader.overview(reference_stain, 0, factor)
    h, w = (
        min(dapi_ov.shape[0], ref_ov.shape[0]),
        min(dapi_ov.shape[1], ref_ov.shape[1]),
    )
    return dapi_ov[:h, :w].astype(np.float32), ref_ov[:h, :w] > 0


def tissue_mask_from_overview(
    dapi_ov: np.ndarray,
    valid_ov: np.ndarray,
    *,
    sigma: float = 2.0,
    min_object: int = 64,
    max_hole: int = 1024,
) -> tuple[np.ndarray, dict]:
    """Tissue mask of an overview.

    Smoothed log DAPI, Otsu threshold over the valid pixels, small objects removed
    and small holes filled. The smoothing is normalised by the valid area so that the
    zeros outside the mosaic do not darken tissue at the mosaic border.

    Args:
        dapi_ov: DAPI overview (counts).
        valid_ov: Valid-pixel mask.
        sigma: Gaussian sigma in overview pixels.
        min_object: Objects smaller than this (overview px) are removed.
        max_hole: Holes smaller than this (overview px) are filled.

    Returns:
        (tissue bool mask, info with threshold_counts and tissue_fraction).
    """
    logd = np.log1p(np.maximum(dapi_ov.astype(np.float64), 0.0))
    w = valid_ov.astype(np.float64)
    den = gaussian_filter(w, sigma)
    smooth = np.where(
        den > 1e-3, gaussian_filter(logd * w, sigma) / np.maximum(den, 1e-3), 0.0
    )
    vals = smooth[valid_ov]
    if vals.size < 2 or float(np.ptp(vals)) < 1e-6:
        return np.zeros(valid_ov.shape, bool), dict(
            threshold_counts=float("nan"), tissue_fraction=0.0
        )
    thr = float(threshold_otsu(vals))
    mask = (smooth > thr) & valid_ov
    lab, n = ndi.label(mask)
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        mask = (sizes[lab] >= min_object) & mask
    lab, n = ndi.label(~mask)
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = sizes[lab] < max_hole
        mask = mask | (small & (lab > 0))
    mask &= valid_ov
    frac = float(mask.sum() / max(int(valid_ov.sum()), 1))
    return mask, dict(threshold_counts=float(np.expm1(thr)), tissue_fraction=frac)


def off_tissue_distance(
    tissue: np.ndarray, valid_ov: np.ndarray, factor: int = 16
) -> np.ndarray:
    """Distance (mosaic px) of valid non-tissue pixels to the nearest tissue/invalid."""
    free = valid_ov & ~tissue
    if free.all():
        return np.full(free.shape, 1e9, np.float64)
    dist: np.ndarray = ndi.distance_transform_edt(free) * factor
    return dist


# Calibration tiles.
def count_bead_like_peaks(
    img: np.ndarray, params: PsfSubtractParams = DEFAULT_PARAMS
) -> int:
    """Count the bead-like sharp peaks of one plane.

    A peak counts when it is a DoG peak with A0 >= a0_min that survives
    non-maximum suppression and whose z0 ring ratio lies within the defaults. The
    tile's own DoG noise is used; this only ranks tiles and estimates the bead
    density.

    Args:
        img: 2-D plane (0 = invalid).
        params: Fixed parameters.

    Returns:
        Count of peaks that look like an in-focus bead (point source with a ring at
        r 8-15 px).
    """
    p = params
    valid = img > 0
    noise = dog_noise_of(img, valid)
    d = gaussian_filter(img.astype(np.float32), 1) - gaussian_filter(
        img.astype(np.float32), 3
    )
    pk = peak_local_max(
        d,
        min_distance=p.peak_min_distance,
        threshold_abs=p.k_sigma * noise,
        exclude_border=p.border,
    )
    if len(pk) == 0:
        return 0
    py, px = pk[:, 0], pk[:, 1]
    a0 = centre_minus_bg(img, py, px)
    m = a0 >= p.a0_min
    py, px, a0 = py[m], px[m], a0[m]
    if len(py) == 0:
        return 0
    k = nms(py, px, a0, p.nms_px)
    py, px, a0 = py[k], px[k], a0[k]
    rr = ring_ratio(img, py, px) / np.maximum(a0, 1.0)
    return int(((rr >= p.ring_min) & (rr <= p.ring_max)).sum())


def candidate_tiles(
    valid_ov: np.ndarray,
    d_off: np.ndarray,
    *,
    shape_px: tuple[int, int],
    tile_px: int,
    min_dist_px: float,
    factor: int = 16,
) -> dict[str, np.ndarray]:
    """List the candidate tiles with their off-tissue statistics.

    The tiles have an edge of ``tile_px``, a stride of half a tile, and lie inside the
    mosaic.

    Args:
        valid_ov: Valid mask of the overview.
        d_off: Distance to tissue/invalid (mosaic px) of the overview.
        shape_px: Mosaic (height, width) in px.
        tile_px: Tile edge (multiple of ``factor``).
        min_dist_px: Distance that counts as tissue-free.
        factor: Overview factor.

    Returns:
        Dict of arrays (y0, x0, valid_frac, off_frac, mean_d_off), one entry per
        tile.
    """
    t = max(tile_px // factor, 1)
    stride = max(t // 2, 1)
    h, w = valid_ov.shape
    ii = np.pad(
        np.cumsum(np.cumsum(valid_ov.astype(np.float64), 0), 1), ((1, 0), (1, 0))
    )
    io = np.pad(
        np.cumsum(np.cumsum((d_off >= min_dist_px) & valid_ov, 0), 1).astype(
            np.float64
        ),
        ((1, 0), (1, 0)),
    )
    dd = np.minimum(d_off, 4.0 * max(shape_px)) * valid_ov
    idd = np.pad(np.cumsum(np.cumsum(dd, 0), 1), ((1, 0), (1, 0)))

    def box(integral: np.ndarray, iy: int, ix: int) -> float:
        return float(
            integral[iy + t, ix + t]
            - integral[iy, ix + t]
            - integral[iy + t, ix]
            + integral[iy, ix]
        )

    rows = []
    for iy in range(0, max(h - t, -1) + 1, stride):
        for ix in range(0, max(w - t, -1) + 1, stride):
            y0, x0 = iy * factor, ix * factor
            if y0 + tile_px > shape_px[0] or x0 + tile_px > shape_px[1]:
                continue
            area = float(t * t)
            rows.append(
                (
                    y0,
                    x0,
                    box(ii, iy, ix) / area,
                    box(io, iy, ix) / area,
                    box(idd, iy, ix) / area,
                )
            )
    a = np.array(rows, np.float64).reshape(-1, 5)
    return dict(
        y0=a[:, 0].astype(int),
        x0=a[:, 1].astype(int),
        valid_frac=a[:, 2],
        off_frac=a[:, 3],
        mean_d_off=a[:, 4],
    )


def rank_tiles(
    cand: dict[str, np.ndarray], *, min_off_frac: float, min_valid_frac: float = 0.99
) -> list[int]:
    """Rank the tiles that have enough valid and tissue-free area.

    Ranking is by off fraction, then mean distance from tissue, then position.

    Returns:
        Tile indices, best first.
    """
    ok = np.where(
        (cand["valid_frac"] >= min_valid_frac) & (cand["off_frac"] >= min_off_frac)
    )[0]
    return sorted(
        ok.tolist(),
        key=lambda i: (
            -cand["off_frac"][i],
            -cand["mean_d_off"][i],
            cand["y0"][i],
            cand["x0"][i],
        ),
    )


def greedy_select(
    centres: np.ndarray, counts: Sequence[int], n_tiles: int, min_sep_px: float
) -> list[int]:
    """Choose tiles greedily by bead-like peak count.

    Ties go to the better-ranked tile; centres must be at least ``min_sep_px`` apart.

    Args:
        centres: [n, 2] tile centres in rank order.
        counts: Peak count per tile.
        n_tiles: Maximum number of tiles.
        min_sep_px: Minimum centre distance.

    Returns:
        Indices (into the rank order) of the chosen tiles, in the order chosen.
    """
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= n_tiles:
            break
        if all(np.hypot(*(centres[i] - centres[j])) >= min_sep_px for j in chosen):
            chosen.append(i)
    return chosen


#: Overlap (fraction of a tile's area shared with a tile of an earlier set) that later
#: tile sets may accept, tried in order.
TILE_SET_OVERLAP_LEVELS = (0.0, 0.25, 0.5, 0.75)


def _overlap_fraction(centres: np.ndarray, c: int, tile_px: int) -> np.ndarray:
    """Fraction of every tile's area covered by tile ``c`` (equal-size tiles)."""
    fy = np.clip(1.0 - np.abs(centres[:, 0] - centres[c, 0]) / tile_px, 0.0, 1.0)
    fx = np.clip(1.0 - np.abs(centres[:, 1] - centres[c, 1]) / tile_px, 0.0, 1.0)
    overlap: np.ndarray = fy * fx
    return overlap


def select_tile_sets_detail(
    centres: np.ndarray,
    counts: Sequence[int],
    n_tiles: int,
    min_sep_px: float,
    *,
    tile_px: int,
    n_sets: int = 1,
    min_tiles: int = 3,
) -> list[tuple[list[int], float]]:
    """Choose successive tile sets that overlap earlier sets as little as possible.

    Each set is a ``greedy_select`` over the tiles that remain. Set 0 is exactly
    ``greedy_select(centres, counts, n_tiles, min_sep_px)`` (the production choice).
    A later set first uses only tiles that share no pixel with an earlier set; when
    that leaves fewer than ``min_tiles`` (a mosaic with little tissue-free area cannot
    supply three disjoint sets) the accepted overlap is raised through
    ``TILE_SET_OVERLAP_LEVELS`` until ``min_tiles`` are found. The sets exist for the
    calibration-reproducibility test (private ``_tile_offset`` hook of
    :func:`calibrate_bead_model`).

    Args:
        centres: [n, 2] tile centres in rank order.
        counts: Peak count per tile.
        n_tiles: Maximum number of tiles per set.
        min_sep_px: Minimum centre distance inside a set.
        tile_px: Tile edge.
        n_sets: Number of sets to build.
        min_tiles: A later set needs at least this many tiles at the accepted
            overlap level.

    Returns:
        ``n_sets`` pairs (indices into the rank order, overlap level used); a set
        may be empty when nothing is left.
    """
    sets: list[tuple[list[int], float]] = []
    used: list[int] = []
    for n in range(n_sets):
        if n == 0:
            sets.append((greedy_select(centres, counts, n_tiles, min_sep_px), 0.0))
            used += sets[-1][0]
            continue
        best: tuple[list[int], float] = ([], TILE_SET_OVERLAP_LEVELS[-1])
        for level in TILE_SET_OVERLAP_LEVELS:
            blocked = np.zeros(len(counts), bool)
            for c in used:
                blocked |= _overlap_fraction(centres, c, tile_px) > level + 1e-9
            avail = np.where(~blocked)[0]
            chosen = (
                [
                    int(avail[k])
                    for k in greedy_select(
                        centres[avail], [counts[i] for i in avail], n_tiles, min_sep_px
                    )
                ]
                if len(avail)
                else []
            )
            best = (chosen, level)
            if len(chosen) >= min_tiles:
                break
        sets.append(best)
        used += best[0]
    return sets


def select_tile_sets(
    centres: np.ndarray,
    counts: Sequence[int],
    n_tiles: int,
    min_sep_px: float,
    *,
    tile_px: int,
    n_sets: int = 1,
) -> list[list[int]]:
    """Return the index lists of :func:`select_tile_sets_detail` (see there)."""
    return [
        c
        for c, _ in select_tile_sets_detail(
            centres, counts, n_tiles, min_sep_px, tile_px=tile_px, n_sets=n_sets
        )
    ]


def _count_job(job: dict[str, Any]) -> int:
    """Count the bead-like peaks of one tile (worker function)."""
    rd = _open_reader(job)
    y0, x0, t = job["y0"], job["x0"], job["tile_px"]
    return count_bead_like_peaks(
        rd.read(job["ref"], job["plane"], y0, y0 + t, x0, x0 + t), job["params"]
    )


# Per-tile local fit and stain responses.
def estimate_rho(
    stack: dict[int, np.ndarray],
    model: Any,
    params: PsfSubtractParams = DEFAULT_PARAMS,
    ks: Sequence[int] = (0, 1, 2),
    max_beads: int = 150,
    half: Sequence[int] = (50, 80, 105),
) -> tuple[float, float, int]:
    """Measure the bead response of another stain relative to the reference stain.

    One calibration tile is used. Per bright single bead and plane a local robust
    regression (with offset) of the stain on the reference model is made, then a
    weighted median across beads and planes.

    Args:
        stack: Planes {z: 2-D array} of the other stain (z in ``ks``).
        model: BeadModel of the tile (``fit_tile_local``).
        params: Fixed parameters.
        ks: Planes used.
        max_beads: Number of brightest singles used.
        half: Half window per plane in ``ks``.

    Returns:
        (rho, standard error, number of bead-plane measurements); (0, inf, n) if
        fewer than 8 measurements.
    """
    src = model.src
    cand = np.where(src.is_bead & ~src.blob)[0]
    cand = cand[np.argsort(-src.a[cand])[:max_beads]]
    vals: list[float] = []
    wts: list[float] = []
    for kk, k in enumerate(ks):
        if k not in model.fits or k not in stack:
            continue
        img = stack[k]
        env = lower_envelope(img, img > 0, sub=4)
        noise = robust_noise(img, img > 0)
        sig = noise * params.tex
        plane_model = model.fits[k].model
        rad = half[kk]
        for i in cand:
            y0, y1, x0, x1 = window_bounds(img.shape, src.y[i], src.x[i], rad)
            m = plane_model[y0:y1, x0:x1].astype(np.float64)
            if m.max() < 3 * sig / np.sqrt(50.0) or (img[y0:y1, x0:x1] <= 0).any():
                continue
            r = (img[y0:y1, x0:x1] - env[y0:y1, x0:x1]).astype(np.float64)
            rho_i, ci = 0.0, float(np.median(r))
            den = 1e-9
            for _ in range(3):
                e = r - rho_i * m - ci
                w = huber1(e, sig, params.huber_k).astype(np.float64)
                mm = m - np.sum(w * m) / np.sum(w)
                den = float(np.sum(w * mm * mm))
                rho_i = float(
                    np.sum(w * mm * (r - np.sum(w * r) / np.sum(w))) / max(den, 1e-9)
                )
                ci = float(np.sum(w * (r - rho_i * m)) / np.sum(w))
            vals.append(rho_i)
            wts.append(den)
    if len(vals) < 8:
        return 0.0, float("inf"), len(vals)
    v = np.array(vals)
    w = np.array(wts)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    med = float(v[o][min(int(np.searchsorted(cw, cw[-1] / 2)), len(v) - 1)])
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    return med, 1.2533 * mad / np.sqrt(len(v)), len(vals)


def pool_rho(
    per_tile: Sequence[tuple[float, float, int]],
    params: PsfSubtractParams = DEFAULT_PARAMS,
) -> tuple[float, float, float]:
    """Pool the per-tile responses of one stain.

    The pooled value is the median over tiles, zeroed unless it is significant and not
    negligible.

    Args:
        per_tile: (rho, se, n) of every passing tile.
        params: Fixed parameters (``rho_min_sig``, ``rho_min``).

    Returns:
        (rho used, standard error, pooled median before zeroing); (0, inf, 0) if no
        tile measured the stain.
    """
    good = [(r, s) for r, s, n in per_tile if np.isfinite(s)]
    if not good:
        return 0.0, float("inf"), 0.0
    v = np.array([a for a, _ in good])
    se = np.array([b for _, b in good])
    m = float(np.median(v))
    s_ = float(
        max(
            1.2533 * np.std(v) / np.sqrt(len(v)),
            1.0 / np.sqrt(np.sum(1.0 / np.maximum(se, 1e-9) ** 2)),
        )
    )
    use = m > params.rho_min_sig * s_ and m >= params.rho_min
    return (m if use else 0.0), s_, m


def _fit_tile_job(job: dict[str, Any]) -> dict[str, Any]:
    """Fit one calibration tile locally and measure the stain responses (worker)."""
    from .model import fit_tile_local

    t0 = time.time()
    rd = _open_reader(job)
    p: PsfSubtractParams = job["params"]
    ref, dapi = job["ref"], job["dapi"]
    y0, x0, t = job["y0"], job["x0"], job["tile_px"]
    win = (y0, y0 + t, x0, x0 + t)
    pt = {k: rd.read(ref, k, *win) for k in range(4)}
    dp = {k: rd.read(dapi, k, *win) for k in range(3)}
    m = fit_tile_local(
        pt,
        {0: dp[0], 1: dp[1]},
        params=p,
        thresholds=ClassifierThresholds.from_params(p),
        min_ring_prom=job["min_ring_prom"],
    )
    info = {k: (v if isinstance(v, str) else float(v)) for k, v in m.info.items()}
    s = m.src
    cls = s.is_bead
    res: dict[str, Any] = dict(
        y0=y0,
        x0=x0,
        status=m.info["status"],
        info=info,
        n_peaks_z1=count_bead_like_peaks(pt[1], p),
        a0=s.A0[cls].copy(),
        r1_bright=s.r1[cls & (s.A0 >= 1000.0)].copy(),
        n_sel=int(m.info.get("n_sel", 0)),
        tmpl={},
        rho={},
    )
    if m.info["status"] == "ok":
        res["tmpl"] = {k: m.fits[k].tmpl.vals.copy() for k in range(4)}
        for stain in job["rho_stains"]:
            stk = (
                dp if stain == dapi else {k: rd.read(stain, k, *win) for k in range(3)}
            )
            res["rho"][stain] = estimate_rho(stk, m, p)
    res["time_s"] = time.time() - t0
    return res


def pool_library(
    tile_results: Sequence[dict[str, Any]],
    params: PsfSubtractParams = DEFAULT_PARAMS,
) -> PsfLibrary:
    """Pool the PSF library from the passing tiles.

    z0..z3 are the node-wise weighted median of the tile templates (weights: number
    of beads that defined each tile); z4..z7 are derived from z3.

    Args:
        tile_results: Results of the passing tiles (``tmpl`` {k: vals}, ``n_sel``
            weights).
        params: Fixed parameters.

    Returns:
        PsfLibrary with 8 templates.
    """
    tm = []
    for k in range(4):
        n = len(tile_results[0]["tmpl"][k])
        vals = np.array([r["tmpl"][k][:n] for r in tile_results])
        weights = np.array([max(int(r["n_sel"]), 1) for r in tile_results], float)
        order = np.argsort(vals, axis=0)
        vals_sorted = np.take_along_axis(vals, order, 0)
        cum_w = np.cumsum(weights[order], 0)
        idx = (cum_w < cum_w[-1] / 2).sum(0)
        tm.append(
            Template(
                params.ds,
                params.r_sup[k],
                np.take_along_axis(vals_sorted, idx[None], 0)[0],
            )
        )
    return PsfLibrary(derive_high_templates(tm, params))


def _spread_stats(values: Sequence[float]) -> dict[str, float]:
    """Return min, max, median, range and relative range (range / median)."""
    v = np.asarray(values, np.float64)
    med = float(np.median(v))
    span = float(v.max() - v.min())
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "median": med,
        "range": span,
        "relative_range": span / abs(med) if med else float("nan"),
    }


def library_tile_spread(
    tile_templates: dict[int, dict[int, np.ndarray]],
    library: PsfLibrary,
    params: PsfSubtractParams = DEFAULT_PARAMS,
) -> dict[str, Any]:
    """Report how much the calibration tiles disagree on the PSF library.

    For every calibration tile that fed the pooled library, the ring radius and the
    flux (relative to the same tile's z0 flux) of its own stacked template of z1..z3
    are measured, and the spread over tiles is summarised. The library is a weighted
    median of these tile templates, and the tile-to-tile spread of its flux scale is
    the largest source of the dependence of the correction on which tiles the
    calibration saw. The numbers are reported for the QC record and never raise a
    warning.

    Args:
        tile_templates: ``{tile index: {plane: template values}}`` of the passing
            tiles (planes 0..3).
        library: The pooled library (its ring radius and flux per plane are
            reported next to the spread).
        params: Fixed parameters (node spacing and support of the templates).

    Returns:
        JSON-able dict: ``planes``, ``per_tile`` (one row per tile with
        ``z<k>_ring_radius_px`` and ``z<k>_flux_rel``), ``summary`` (per quantity:
        min, max, median, range, relative range) and ``pooled`` (the library values).
    """
    per_tile: list[dict[str, Any]] = []
    for index in sorted(tile_templates):
        per = tile_templates[index]
        tmpl = {
            k: Template(params.ds, params.r_sup[k], np.asarray(per[k], np.float64))
            for k in (0, *LIBRARY_SPREAD_PLANES)
        }
        flux0 = tmpl[0].flux(1e9)
        row: dict[str, Any] = {"tile_index": int(index)}
        for k in LIBRARY_SPREAD_PLANES:
            ring = int(np.argmax(tmpl[k].nodes * tmpl[k].vals))
            row[f"z{k}_ring_radius_px"] = float(tmpl[k].nodes[ring])
            row[f"z{k}_flux_rel"] = (
                float(tmpl[k].flux(1e9) / flux0) if flux0 else float("nan")
            )
        per_tile.append(row)
    keys = [key for key in per_tile[0] if key != "tile_index"] if per_tile else []
    summary = {key: _spread_stats([row[key] for row in per_tile]) for key in keys}
    pooled_rows = library.summary()
    pooled: dict[str, float] = {}
    for k in LIBRARY_SPREAD_PLANES:
        pooled[f"z{k}_ring_radius_px"] = float(pooled_rows[k]["ring_radius_px"])
        pooled[f"z{k}_flux_rel"] = float(pooled_rows[k]["flux_rel"])
    return {
        "planes": list(LIBRARY_SPREAD_PLANES),
        "n_tiles": len(per_tile),
        "per_tile": per_tile,
        "summary": summary,
        "pooled": pooled,
        "note": "reported only: tile-to-tile spread of the PSF library",
    }


# Per-dataset classifier thresholds.
_PERMISSIVE = ClassifierThresholds(
    ring_min=-1e9, ring_max=1e9, r1_max=1e9, rd_center=0.0, rd0_center=0.0, rd_tol=1e9
)


def _reference_job(job: dict[str, Any]) -> dict[str, Any]:
    """Collect the bright candidates of one passing tile (worker function).

    The permissive detector is run; each bright candidate gets its features, whether
    it is isolated, whether it is spectrally a bead and the t-score of its raw z1 ring.
    """
    rd = _open_reader(job)
    p: PsfSubtractParams = job["params"]
    ref, dapi = job["ref"], job["dapi"]
    y0, x0, t = job["y0"], job["x0"], job["tile_px"]
    win = (y0, y0 + t, x0, x0 + t)
    pt = {k: rd.read(ref, k, *win) for k in (0, 1)}
    dp = {k: rd.read(dapi, k, *win) for k in (0, 1)}
    src = detect_sources(
        pt, dp, dog_noise=job["dog_noise"], thresholds=_PERMISSIVE, params=p
    )
    empty: dict[str, Any] = {
        k: np.zeros(0)
        for k in ("y", "x", "A0", "rring", "r1", "rD0", "rD", "tz1", "sd1rel")
    }
    empty.update(isolated=np.zeros(0, bool), spectral=np.zeros(0, bool), n_bright=0)
    if src.n == 0:
        return empty
    amplitude = np.asarray(src.A0)
    bright = np.where(amplitude >= job["a0_min_ref"])[0]
    if len(bright) == 0:
        return empty
    pts = np.c_[src.y, src.x]
    tree = cKDTree(pts)
    iso = np.ones(len(bright), bool)
    for n, i in enumerate(bright):
        for j in tree.query_ball_point(pts[i], 60.0):
            if j != i and src.A0[j] >= 0.25 * src.A0[i]:
                iso[n] = False
                break
    iy = np.clip(np.rint(src.y[bright]).astype(int), 0, t - 1)
    ix = np.clip(np.rint(src.x[bright]).astype(int), 0, t - 1)
    a0 = src.A0[bright]
    a0s = np.maximum(a0, 1.0)
    rd0 = centre_minus_bg(dp[0], iy, ix) / a0s
    th = ClassifierThresholds.from_params(p)
    sd0 = null_sd(dp[0], iy, ix)
    sd_d = null_sd(dp[0], iy, ix, img2=dp[1])
    spectral = (
        np.abs(rd0 - th.rd0_center) <= np.maximum(th.rd_tol, 3.5 * sd0 / a0s)
    ) | (
        np.abs(src.rD[bright] - th.rd_center) <= np.maximum(th.rd_tol, 3.5 * sd_d / a0s)
    )
    tz1 = ring_tscores(pt[1], src.y[bright], src.x[bright], job["r1_lib"])
    return dict(
        y=src.y[bright],
        x=src.x[bright],
        A0=a0,
        rring=np.asarray(src.rring)[bright],
        r1=src.r1[bright],
        rD0=rd0,
        rD=src.rD[bright],
        tz1=tz1,
        isolated=iso,
        spectral=spectral,
        n_bright=int(len(bright)),
        sd1rel=np.asarray(src.sd1_rel)[bright],
    )


def zprof_zmax_available(planes: Sequence[int]) -> int:
    """Return the highest plane of the rescue's z-profile test that the data has.

    The test reads planes 2..k; it is limited to ``RESCUE_ZPROF_ZMAX`` and to the
    planes the reader has, in the calibration as in production.

    Args:
        planes: Planes of the reader.

    Returns:
        The highest k <= ``RESCUE_ZPROF_ZMAX`` such that planes 2..k all exist; 0
        when plane 2 is missing (the test is off).
    """
    have = {int(z) for z in planes}
    k = 1
    while k + 1 in have and k + 1 <= RESCUE_ZPROF_ZMAX:
        k += 1
    return k if k >= 2 else 0


def derive_classifier_thresholds(
    ref: dict[str, np.ndarray],
    params: PsfSubtractParams = DEFAULT_PARAMS,
    *,
    r1_ring_px: float = 0.0,
    zprof_zmax: int = RESCUE_ZPROF_ZMAX,
) -> tuple[ClassifierThresholds, dict[str, Any]]:
    """Derive per-dataset classifier thresholds from the reference beads.

    Each threshold is the raw statistic of the reference set clamped to a safe range;
    fewer than ``MIN_REFERENCE_BEADS`` beads keep the parameter defaults.

    Args:
        ref: Reference-set features (arrays ``rring``, ``r1``, ``rD0``, ``rD`` and,
            for the rescue's texture gate, ``sd1rel``) of the isolated reference
            beads.
        params: Fixed parameters (defaults used when the set is too small).
        r1_ring_px: z1 ring radius of the PSF library (px). 0 (the default) leaves
            the rescue off; a positive value switches it on when the texture gate can
            be derived.
        zprof_zmax: Highest plane of the rescue's z-profile test (the planes
            2..zprof_zmax must exist in the data; below 2 the test is off).

    Returns:
        (thresholds, info with the raw percentiles/medians, clamps hit and the set
        size).
    """
    default = ClassifierThresholds.from_params(params)
    n = int(len(ref["rring"]))
    info: dict[str, Any] = {
        "n_reference": n,
        "used_defaults": n < MIN_REFERENCE_BEADS,
    }
    if n < MIN_REFERENCE_BEADS:
        return default, info
    p2 = float(np.percentile(ref["rring"], 2))
    p98 = float(np.percentile(ref["r1"], 98))
    m0 = float(np.median(ref["rD0"]))
    m_d = float(np.median(ref["rD"]))
    info.update(ring_p2_raw=p2, r1_p98_raw=p98, rd0_median_raw=m0, rd_median_raw=m_d)
    th = dataclasses.replace(
        default,
        ring_min=float(np.clip(p2, 0.20, 0.30)),
        r1_max=float(np.clip(p98, 0.25, 0.35)),
        rd0_center=float(np.clip(m0, 0.05, 0.20)),
        rd_center=float(np.clip(m_d, 0.05, 0.20)),
    )
    clamp_rows = [
        ("ring_min", p2, th.ring_min),
        ("r1_max", p98, th.r1_max),
        ("rd0_center", m0, th.rd0_center),
        ("rd_center", m_d, th.rd_center),
    ]
    if r1_ring_px > 0:
        # Rescue path: the ring radius and the fixed constants, then the texture gate
        # from the reference set. Without enough texture measurements the rescue stays
        # off (rescue_sd1_max = 0).
        th = dataclasses.replace(
            th,
            rescue_tz1_min=RESCUE_TZ1_MIN,
            rescue_ring_radius_px=float(r1_ring_px),
            rescue_a0_min=REFERENCE_A0_MIN,
            rescue_zprof_zmax=float(zprof_zmax) if zprof_zmax >= 2 else 0.0,
        )
        texture = np.asarray(ref.get("sd1rel", np.zeros(0)), float)
        texture = texture[np.isfinite(texture)]
        if len(texture) >= MIN_REFERENCE_BEADS:
            sd1_p98 = float(np.percentile(texture, 98))
            th = dataclasses.replace(
                th, rescue_sd1_max=float(np.clip(sd1_p98, *RESCUE_SD1_RANGE))
            )
            info.update(sd1_rel_p98_raw=sd1_p98, n_sd1_reference=int(len(texture)))
            clamp_rows.append(("rescue_sd1_max", sd1_p98, th.rescue_sd1_max))
        else:
            info["rescue_off_reason"] = (
                "fewer than MIN_REFERENCE_BEADS texture (sd1_rel) values"
            )
    info["clamped"] = [name for name, raw, val in clamp_rows if abs(raw - val) > 1e-12]
    return th, info


def confirmed_mask(
    ref: dict[str, np.ndarray], *, tz1_min: float = 3.0, require_isolated: bool = True
) -> np.ndarray:
    """Select the confirmed reference beads of one tile.

    A reference bead is bright, (isolated,) spectrally confirmed and has a coherent
    raw z1 ring.

    Args:
        ref: Output of ``_reference_job``.
        tz1_min: Minimum sector t-score of the z1 ring at the library ring radius.
        require_isolated: Keep only beads with no comparable neighbour within 60 px
            (the threshold-setting set: a clean ring ratio). The diagnostic recall
            without it includes the beads of clusters, where the ring ratio is
            disturbed.

    Returns:
        Boolean mask over the tile's bright candidates.
    """
    if len(ref["y"]) == 0:
        return np.zeros(0, bool)
    m = ref["spectral"] & (ref["tz1"] >= tz1_min)
    return m & ref["isolated"] if require_isolated else m


def focus_check(
    refs: Sequence[dict[str, np.ndarray]],
    z0_centre: float,
    z1_centre: float,
    *,
    limit: float = FOCUS_R1_MAX,
) -> dict[str, Any]:
    """Check that the beads are focused at plane 0.

    The median z1 / z0 centre ratio (``r1``) is taken over the bright (A0 >= 1,000),
    spectrally confirmed candidates of the PERMISSIVE detector, not over beads that
    already passed the classifier's own ``r1 < r1_max`` test (which would make the
    check circular). The dataset must also have its sharpest template centre at
    plane 0.

    Args:
        refs: ``_reference_job`` outputs of the passing calibration tiles.
        z0_centre, z1_centre: Centre values of the pooled z0 and z1 templates.
        limit: Median ``r1`` must stay below this.

    Returns:
        Dict with n_bright, median_r1, the isolated-only variant (information), the
        template centres, the limit and ``passed``.
    """
    r1 = np.concatenate([r["r1"][r["spectral"]] for r in refs]) if refs else np.zeros(0)
    r1_iso = (
        np.concatenate([r["r1"][r["spectral"] & r["isolated"]] for r in refs])
        if refs
        else np.zeros(0)
    )
    med = float(np.median(r1)) if r1.size else float("nan")
    return dict(
        n_bright=int(r1.size),
        median_r1=med,
        n_bright_isolated=int(r1_iso.size),
        median_r1_isolated=float(np.median(r1_iso)) if r1_iso.size else float("nan"),
        z0_centre=float(z0_centre),
        z1_centre=float(z1_centre),
        r1_limit=float(limit),
        source="permissive detector, A0 >= 1000, spectrally confirmed",
        passed=bool((not r1.size or med < limit) and z0_centre > z1_centre),
    )


def checkerboard_role(i: int, j: int) -> str:
    """Return the role of grid tile (row i, column j): a fixed checkerboard.

    Tiles alternate between ``threshold`` (sets the thresholds) and ``recall``
    (held out to measure the recall).
    """
    return "threshold" if (i + j) % 2 == 0 else "recall"


def _dedupe(
    points: np.ndarray, against: np.ndarray | None = None, tol: float = SAME_BEAD_TOL_PX
) -> np.ndarray:
    """Mask the points to keep.

    Of several points within ``tol`` only the first is kept, and points within
    ``tol`` of ``against`` are dropped. Overlapping tiles see the same bead twice, and
    a held-out bead must not be one that set the thresholds.
    """
    keep = np.ones(len(points), bool)
    if len(points) == 0:
        return keep
    if against is not None and len(against):
        d, _ = cKDTree(against).query(points)
        keep &= d > tol
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    # pairs have a < b: drop the later point of every pair
    for _, b in pairs:
        keep[b] = False
    return keep


def _dedupe_parts(
    parts: Sequence[np.ndarray],
    against: np.ndarray | None = None,
    tol: float = SAME_BEAD_TOL_PX,
) -> list[np.ndarray]:
    """Apply :func:`_dedupe` to the concatenated per-tile arrays.

    Returns:
        One keep mask per array.
    """
    sizes = [len(p) for p in parts]
    if not sum(sizes):
        return [np.zeros(n, bool) for n in sizes]
    keep = _dedupe(np.concatenate(parts), against=against, tol=tol)
    cuts = np.cumsum([0] + sizes)
    return [keep[cuts[i] : cuts[i + 1]] for i in range(len(parts))]


def _ring_stats(rring: np.ndarray, r1: np.ndarray) -> dict[str, Any]:
    """Percentiles of the z0 ring ratio (and P98 of r1) of a set of reference beads."""
    if len(rring) == 0:
        return dict(
            n=0,
            ring_p2=None,
            ring_p10=None,
            ring_median=None,
            ring_p90=None,
            r1_p98=None,
        )
    p2, p10, p50, p90 = (float(v) for v in np.percentile(rring, [2, 10, 50, 90]))
    return dict(
        n=int(len(rring)),
        ring_p2=p2,
        ring_p10=p10,
        ring_median=p50,
        ring_p90=p90,
        r1_p98=float(np.percentile(r1, 98)),
    )


def _recall_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run the production detector on one tile and count the reference beads it finds.

    Args:
        job: Tile window, global DoG noise, the final thresholds and the tile-local
            positions of the reference beads to recall (``ref_y`` / ``ref_x``, and the
            diagnostic set ``ref2_y`` / ``ref2_x`` that includes non-isolated beads;
            both may be empty or missing).

    Returns:
        Dict with n / n2 (reference beads), hit / hit2 (recalled within 2 px),
        n_classified (beads of the detector), n_rescued (of them, accepted by the
        rescue path) and n_valid (pixels).
    """
    rd = _open_reader(job)
    p: PsfSubtractParams = job["params"]
    y0, x0, t = job["y0"], job["x0"], job["tile_px"]
    win = (y0, y0 + t, x0, x0 + t)
    # The recall is measured with the production detector, which reads the planes
    # of the rescue's z-profile test.
    planes = (0, 1, *job["thresholds"].rescue_planes)
    pt = {k: rd.read(job["ref"], k, *win) for k in planes}
    dp = {k: rd.read(job["dapi"], k, *win) for k in (0, 1)}
    src = detect_sources(
        pt, dp, dog_noise=job["dog_noise"], thresholds=job["thresholds"], params=p
    )
    out = dict(
        n=0,
        hit=0,
        n_classified=int(src.is_bead.sum()),
        n_valid=int((pt[0] > 0).sum()),
        n_rescued=int(src.rescued.sum()) if src.rescued is not None else 0,
    )
    tree = (
        cKDTree(np.c_[src.y[src.is_bead], src.x[src.is_bead]])
        if src.is_bead.any()
        else None
    )
    for key, (ky, kx) in {"": ("ref_y", "ref_x"), "2": ("ref2_y", "ref2_x")}.items():
        ry, rx = job.get(ky, np.zeros(0)), job.get(kx, np.zeros(0))
        out["n" + key] = int(len(ry))
        out["hit" + key] = 0
        if len(ry) and tree is not None:
            d, _ = tree.query(np.c_[ry, rx])
            out["hit" + key] = int((d <= 2.0).sum())
    return out


def classifier_recall_warnings(
    recall: float, n_held: int, recall_with_clusters: float, n_held_with_clusters: int
) -> list[str]:
    """Return the QC warnings of the classifier's held-out recall.

    The warning rests on the cluster-inclusive recall: the isolated reference set
    misses the beads that fail in clusters, which is where a classifier problem shows.
    The isolated held-out recall is reported and warns only when it rests on enough
    beads.

    Args:
        recall: Held-out recall on the isolated reference beads of the recall tiles
            (NaN when there is none).
        n_held: Number of those beads.
        recall_with_clusters: Held-out recall including the beads of clusters.
        n_held_with_clusters: Number of those beads.

    Returns:
        Warning messages (empty when the classifier looks adequate).
    """
    out: list[str] = []
    if not n_held and not n_held_with_clusters:
        out.append(
            "held-out classifier recall could not be measured: no reference bead on "
            "the recall tiles"
        )
    if (
        n_held_with_clusters >= MIN_RECALL_WITH_CLUSTERS_REFERENCE
        and recall_with_clusters < RECALL_WITH_CLUSTERS_WARN_BELOW
    ):
        out.append(
            "classifier recall including beads in clusters is "
            f"{recall_with_clusters:.3f} (< {RECALL_WITH_CLUSTERS_WARN_BELOW:.2f}; "
            f"{n_held_with_clusters} held-out reference beads): many beads of this "
            "dataset fail the bead test (cluster-disturbed rings) and stay "
            "uncorrected"
        )
    if n_held >= MIN_RECALL_REFERENCE and recall < RECALL_WARN_BELOW:
        out.append(
            f"held-out classifier recall is {recall:.3f} (< {RECALL_WARN_BELOW:.2f}; "
            f"{n_held} isolated reference beads on the recall tiles)"
        )
    return out


# Dataset-global constants.
def _noise_job(job: dict[str, Any]) -> dict[str, Any]:
    """Measure the pixel and DoG noise of one sample tile (worker function)."""
    rd = _open_reader(job)
    y0, x0, t = job["y0"], job["x0"], job["tile_px"]
    planes = {
        k: rd.read(job["ref"], k, y0, y0 + t, x0, x0 + t)
        for k in range(job["k_fit"] + 1)
    }
    return dict(
        noise={k: robust_noise(a, a > 0) for k, a in planes.items()},
        dog=dog_noise_of(planes[0]),
    )


def estimate_global_noise(
    reader: Any,
    ref: str,
    k_fit: int,
    *,
    executor: cf.ProcessPoolExecutor | None,
    base: dict[str, Any],
    tile_px: int = 1024,
    tiles: Sequence[tuple[int, int]] | None = None,
) -> tuple[dict[int, float], float, list[tuple[int, int]]]:
    """Measure the per-plane pixel noise and the z0 DoG noise.

    The values are medians over a 4 x 4 grid of sample tiles.

    Args:
        reader: MosaicPlaneReader.
        ref: Reference stain.
        k_fit: Highest modelled plane (noise is measured on planes 0..k_fit).
        executor: Executor (or None for inline).
        base: Common job fields (images_dir, stains, planes, params).
        tile_px: Sample tile edge.
        tiles: Sample tile origins when the caller already has the grid (default:
            measured here).

    Returns:
        (noise per plane, dog_noise_z0, sample tile origins).
    """
    if tiles is None:
        tiles = sample_tiles(
            reader, stain=ref, z=0, n_grid=4, tile_px=tile_px, min_valid=0.9
        )
    tiles = [(int(y), int(x)) for y, x in tiles]
    if not tiles:
        raise BeadCalibrationError(
            "no valid pixel in the reference plane 0: nothing to calibrate"
        )
    jobs = [dict(base, y0=y, x0=x, tile_px=tile_px, k_fit=k_fit) for y, x in tiles]
    res = _map(executor, _noise_job, jobs)
    noise = {
        k: float(np.median([r["noise"][k] for r in res])) for k in range(k_fit + 1)
    }
    return noise, float(np.median([r["dog"] for r in res])), tiles


def _ell_job(job: dict[str, Any]) -> dict[str, Any]:
    """Measure the stretch of the beads of one sample tile in library mode (worker).

    The tile-local fit stacks its own template, so its stretch values are relative to
    a template that already holds the typical stretch; the fallback value used in
    library mode has to be measured against the pooled library.
    """
    from .model import fit_window_library

    rd = _open_reader(job)
    p: PsfSubtractParams = job["params"]
    t, c = job["tile_px"], job["crop_px"]
    y0 = job["y0"] + ((t - c) // 2) // 16 * 16
    x0 = job["x0"] + ((t - c) // 2) // 16 * 16
    win = (y0, y0 + c, x0, x0 + c)
    pt = {k: rd.read(job["ref"], k, *win) for k in (0, 1, 2)}
    dp = {k: rd.read(job["dapi"], k, *win) for k in (0, 1)}
    m = fit_window_library(
        pt, dp, origin_yx=(y0, x0), calibration=job["cal"], params=p, planes=[1, 2]
    )
    em = m.info.get("ell_meas_median", float("nan"))
    ok = m.info.get("status") == "ok" and np.isfinite(em)
    return dict(
        ell_meas_median=float(em) if ok else float("nan"),
        n_stretch_beads=int(m.info.get("n_stretch_beads", 0)),
    )


# Cost forecast.
def _px_per_um(images_dir: Path) -> float:
    """Read the pixel size (px per micron) next to the images, or the default."""
    try:
        row = (
            (Path(images_dir) / "micron_to_mosaic_pixel_transform.csv")
            .read_text()
            .split("\n")[0]
            .split()
        )
        v = float(row[0])
        return v if 1.0 < v < 100.0 else DEFAULT_PX_PER_UM
    except (OSError, ValueError, IndexError):
        return DEFAULT_PX_PER_UM


def count_windows(
    valid_ov: np.ndarray, *, core_px: int = FORECAST_CORE_PX, factor: int = 16
) -> int:
    """Count the ``core_px`` cores that hold any valid pixel.

    The cores lie on a grid from the mosaic origin, as the chunk driver cuts them.

    Args:
        valid_ov: Valid mask of the 1/``factor`` overview.
        core_px: Core edge (px).
        factor: Overview factor.

    Returns:
        Count of cores with at least one valid overview pixel.
    """
    t = max(core_px // factor, 1)
    h, w = valid_ov.shape
    pad = np.zeros((-(-h // t) * t, -(-w // t) * t), bool)
    pad[:h, :w] = valid_ov
    return int(
        pad.reshape(pad.shape[0] // t, t, pad.shape[1] // t, t).any(axis=(1, 3)).sum()
    )


def _interp(points: Sequence[tuple[float, float]], x: float) -> float:
    """Interpolate piecewise-linearly through ``points`` (flat outside their range)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.interp(x, xs, ys))


def gpu_window_seconds(fitted_beads_per_window: float) -> float:
    """Seconds one 4096 px core window takes on the GPU backend.

    The model is a + b x fitted beads through the two measured points.
    """
    (x0, t0), (x1, t1) = FORECAST_GPU_POINTS
    b = (t1 - t0) / (x1 - x0)
    return float(max(t0 + b * (fitted_beads_per_window - x0), 0.5 * t0))


def forecast_cost(
    *,
    density_per_px2: float,
    valid_area_px: float,
    n_workers: int,
    px_per_um: float,
    n_windows: int | None = None,
    calibration_s: float = 0.0,
    backend: Backend = "cpu",
) -> dict[str, Any]:
    """Predict the cost of the chunk pass from the bead density of the sample grid.

    CPU: beads in the cores x CPU-s per bead of the configured worker count (measured
    table, linear between the points) / (workers x measured parallel efficiency).
    GPU: windows x (a + b x fitted beads per window), fitted beads =
    ``FORECAST_HALO_FACTOR`` x beads in the cores. ``wall_hours`` is the chunk pass of
    the requested ``backend`` plus the calibration pass; the warning flag applies to
    it.

    Args:
        density_per_px2: Classified beads per mosaic pixel (sample grid, tissue
            included).
        valid_area_px: Valid mosaic area in px.
        n_workers: Worker processes of the CPU chunk pool.
        px_per_um: Pixel scale.
        n_windows: Number of 4096 px core windows with valid pixels (needed for the
            GPU figure; None = not forecast).
        calibration_s: Duration of the calibration pass (added to both backends).
        backend: "cpu" or "cuda": the backend ``wall_hours`` / ``warn`` refer to.

    Returns:
        Dict with beads/mm2, beads in the cores / fitted, the CPU and GPU figures,
        ``wall_hours``, ``warn``.
    """
    beads = density_per_px2 * valid_area_px
    fitted = beads * FORECAST_HALO_FACTOR
    nw = max(int(n_workers), 1)
    cpu_s_per_bead = _interp(FORECAST_CPU_S_PER_BEAD, nw)
    eff = _interp(FORECAST_CPU_EFFICIENCY, nw)
    cpu_s = beads * cpu_s_per_bead
    cpu_wall_h = (cpu_s / nw / eff + calibration_s) / 3600.0
    out: dict[str, Any] = dict(
        beads_per_mm2=float(density_per_px2 * px_per_um**2 * 1e6),
        n_beads_total=float(beads),
        n_beads_fitted=float(fitted),
        cpu_s_per_bead=float(cpu_s_per_bead),
        cpu_parallel_efficiency=float(eff),
        cpu_hours=float(cpu_s / 3600.0),
        cpu_wall_hours=float(cpu_wall_h),
        workers=nw,
        px_per_um=float(px_per_um),
        calibration_s=float(calibration_s),
    )
    gpu_wall_h: float | None = None
    if n_windows:
        t_win = gpu_window_seconds(fitted / n_windows)
        gpu_wall_h = float((n_windows * t_win + calibration_s) / 3600.0)
        out.update(
            n_windows=int(n_windows),
            fitted_beads_per_window=float(fitted / n_windows),
            gpu_s_per_window=float(t_win),
            gpu_wall_hours=gpu_wall_h,
        )
    use_gpu = backend == "cuda" and gpu_wall_h is not None
    wall_h = float(gpu_wall_h) if (use_gpu and gpu_wall_h is not None) else cpu_wall_h
    out.update(
        backend="cuda" if use_gpu else "cpu",
        wall_hours=float(wall_h),
        warn=bool(wall_h > FORECAST_WARN_WALL_H),
    )
    return out


# The calibration pass.
def _fmt(v: float | None) -> str:
    """Format an optional number for an error message."""
    return "n/a" if v is None else f"{v:.4f}"


def _fail(msg: str, report: dict[str, Any]) -> BeadCalibrationError:
    logger.error("[BEADS] calibration failed: %s", msg)
    report_json: dict[str, Any] = to_jsonable(report)
    return BeadCalibrationError(msg, report_json)


def calibrate_bead_model(
    reader: Any,
    *,
    config: MerscopeBeadRemovalConfig | None = None,
    z_layers: Sequence[int],
    params: PsfSubtractParams = DEFAULT_PARAMS,
    n_workers: int = 1,
    backend: Backend = "cpu",
    _tile_offset: int = 0,
) -> BeadCalibration:
    """Run the once-per-dataset calibration pass.

    Steps: validate the inputs; 1/16 overview, tissue mask and distance from tissue;
    choose up to ``calibration_tiles`` tissue-free tiles; fit every tile locally in a
    process pool; decide dataset-wide whether beads are present (majority of tiles)
    and in focus; pool the PSF library and record its tile-to-tile spread; measure the
    response of every stain; derive the classifier thresholds from a reference set of
    bright isolated beads and check them on held-out grid tiles; measure the
    dataset-global constants; forecast the cost.

    Args:
        reader: MosaicPlaneReader with the reference stain and DAPI at planes
            0..K_fit (and every z_layer) and, optionally, other stains at planes 0..2
            (their bead response is measured and recorded).
        config: Calibration settings (tile size, number, distance, gate).
        z_layers: Planes that will be corrected (all in 1..7).
        params: Fixed parameters.
        n_workers: Worker processes (1 = inline).
        backend: Backend the cost forecast is made for ("cpu" or "cuda"); the caller
            passes the backend it resolved. It changes only the forecast.
        _tile_offset: PRIVATE TEST HOOK, not a config field: a multiple of
            ``calibration_tiles``; the calibration tiles are the
            ``_tile_offset / calibration_tiles``-th successive disjoint set of the
            tile choice (0 = the production choice; 6 and 12 give the second and
            third set of 6 tiles, on other pixels), used to measure how much the
            calibration depends on which tiles it saw.

    Returns:
        BeadCalibration for the chunk pass.

    Raises:
        ValueError: ``z_layers`` empty or outside 1..7, a bad ``backend`` or a bad
            ``_tile_offset``.
        BeadCalibrationError: Missing stain/plane, no tissue-free area, no beads,
            ambiguous, beads not focused.
    """
    cfg = config or MerscopeBeadRemovalConfig()
    if backend not in ("cpu", "cuda"):
        raise ValueError(f"backend must be 'cpu' or 'cuda', got {backend!r}")
    if _tile_offset < 0 or _tile_offset % max(int(cfg.calibration_tiles), 1):
        raise ValueError(
            "_tile_offset must be a non-negative multiple of calibration_tiles "
            f"({cfg.calibration_tiles}); got {_tile_offset}"
        )
    n_sets = _tile_offset // max(int(cfg.calibration_tiles), 1) + 1
    t_start = time.time()
    timing: dict[str, float] = {}
    warnings: list[str] = []
    report: dict[str, Any] = dict(warnings=warnings)

    # Validate the inputs.
    zs = sorted({int(z) for z in z_layers})
    if not zs or zs[0] < 1 or zs[-1] > 7:
        raise ValueError(
            "z_layers must be non-empty and within 1..7 (start at plane 1 or later; "
            f"plane 0 is read for detection only); got {list(z_layers)}"
        )
    ref, dapi = cfg.reference_stain, "DAPI"
    k_fit = max(2, zs[-1])
    for s in (ref, dapi):
        if s not in reader.stains:
            raise BeadCalibrationError(
                f"stain {s} is missing from the reader (the reference stain and DAPI "
                "are required)"
            )
    need = set(range(k_fit + 1)) | set(zs)
    if not need <= set(reader.planes):
        raise BeadCalibrationError(
            f"planes {sorted(need - set(reader.planes))} are missing from the reader "
            f"(need 0..{k_fit} and z_layers)"
        )
    other_stains = [s for s in reader.stains if s != ref]
    height, width = reader.shape
    base: dict[str, Any] = dict(
        images_dir=str(reader.images_dir),
        stains=list(reader.stains),
        planes=list(reader.planes),
        params=params,
        ref=ref,
        dapi=dapi,
    )
    tile_px = int(cfg.calibration_tile_px)
    gate = float(cfg.min_ring_prominence)

    with _pool(n_workers) as ex:
        # Overview, tissue mask and distance from tissue.
        t0 = time.time()
        z_mid = min(zs, key=lambda z: (abs(z - (zs[0] + zs[-1]) / 2), z))
        dapi_ov, valid_ov = build_overview(
            reader, reference_stain=ref, dapi_stain=dapi, z_mid=z_mid
        )
        tissue, tinfo = tissue_mask_from_overview(dapi_ov, valid_ov)
        d_off = off_tissue_distance(tissue, valid_ov)
        timing["overview_s"] = time.time() - t0
        tinfo.update(
            z_mid=int(z_mid),
            overview_shape=list(valid_ov.shape),
            valid_overview_px=int(valid_ov.sum()),
        )
        report["tissue"] = tinfo
        logger.info(
            "[BEADS] overview %s, tissue fraction %.3f (threshold %.0f counts)",
            valid_ov.shape,
            tinfo["tissue_fraction"],
            tinfo["threshold_counts"],
        )
        if not valid_ov.any():
            raise _fail("no valid pixel in the reference plane 0", report)

        # Candidate tiles and greedy selection.
        t0 = time.time()
        cand = candidate_tiles(
            valid_ov,
            d_off,
            shape_px=(height, width),
            tile_px=tile_px,
            min_dist_px=cfg.min_offtissue_distance_px,
        )
        ranked = rank_tiles(cand, min_off_frac=0.95)
        low_tissue = False
        if len(ranked) < 3:
            ranked = rank_tiles(cand, min_off_frac=0.50)
            low_tissue = True
            warnings.append(
                "low_tissue_tiles: fewer than 3 tiles are >= 95 % tissue-free; the "
                "threshold was relaxed to 50 %"
            )
        tinfo.update(
            n_candidate_tiles=int(len(cand["y0"])),
            n_offtissue_ranked=int(len(ranked)),
            low_tissue_tiles=low_tissue,
        )
        if len(ranked) < 3:
            raise _fail(
                f"no tissue-free area: only {len(ranked)} tile(s) of {tile_px} px are "
                f"at least {cfg.min_offtissue_distance_px} px from tissue; the "
                "calibration needs 3 (set merscope_bead_removal false for this sample "
                "or use a sample with empty coverslip area)",
                report,
            )
        sep = 2.0 * tile_px
        counts: list[int] = []
        n_batch = 40
        chosen: list[int] = []
        while True:
            lo = len(counts)
            todo = ranked[lo : lo + n_batch]
            jobs = [
                dict(
                    base,
                    y0=int(cand["y0"][i]),
                    x0=int(cand["x0"][i]),
                    tile_px=tile_px,
                    plane=0,
                )
                for i in todo
            ]
            counts += _map(ex, _count_job, jobs)
            centres = np.array(
                [
                    [cand["y0"][i] + tile_px / 2.0, cand["x0"][i] + tile_px / 2.0]
                    for i in ranked[: len(counts)]
                ]
            )
            chosen, overlap_level = select_tile_sets_detail(
                centres,
                counts,
                int(cfg.calibration_tiles),
                sep,
                tile_px=tile_px,
                n_sets=n_sets,
            )[-1]
            if len(chosen) >= 3 or len(counts) >= len(ranked):
                break
        timing["tile_selection_s"] = time.time() - t0
        tinfo["n_counted"] = int(len(counts))
        tinfo["tile_set"] = dict(
            tile_offset=int(_tile_offset),
            set_index=int(n_sets - 1),
            overlap_with_earlier_sets=float(overlap_level),
        )
        tinfo["median_peaks_per_tile"] = float(np.median(counts))
        if len(chosen) < 3:
            raise _fail(
                f"no tissue-free area: only {len(chosen)} tissue-free tile(s) of "
                f"{tile_px} px lie at least {int(sep)} px apart; the calibration "
                "needs 3",
                report,
            )
        sel = [ranked[i] for i in chosen]
        n_tiles = len(sel)
        tiles: list[dict[str, Any]] = []
        for n, (ci, i) in enumerate(zip(chosen, sel, strict=True)):
            tiles.append(
                dict(
                    index=n,
                    y0=int(cand["y0"][i]),
                    x0=int(cand["x0"][i]),
                    tile_px=tile_px,
                    rank=int(ci),
                    off_frac=float(cand["off_frac"][i]),
                    mean_d_off=float(cand["mean_d_off"][i]),
                    n_peaks_z0=int(counts[ci]),
                )
            )
        report["tiles"] = tiles
        logger.info(
            "[BEADS] %d calibration tiles chosen (peaks %s)",
            n_tiles,
            [t["n_peaks_z0"] for t in tiles],
        )

        # Local fit of every calibration tile.
        t0 = time.time()
        rho_stains = [dapi] + [s for s in other_stains if s != dapi]
        jobs = [
            dict(
                base,
                y0=t["y0"],
                x0=t["x0"],
                tile_px=tile_px,
                min_ring_prom=gate,
                rho_stains=rho_stains,
            )
            for t in tiles
        ]
        fits = _map(ex, _fit_tile_job, jobs)
        timing["tile_fit_s"] = time.time() - t0
        for t, f in zip(tiles, fits, strict=True):
            inf = f["info"]
            t.update(
                status=f["status"],
                n_peaks_z1=int(f["n_peaks_z1"]),
                time_s=float(f["time_s"]),
                z1_ring_prominence=inf.get("z1_ring_prominence", float("nan")),
                z1_ring_radius=inf.get("z1_ring_radius", float("nan")),
                n_cand=inf.get("n_cand"),
                n_bead=inf.get("n_bead"),
                n_ring_evidence=inf.get("n_ring_evidence"),
                n_blob=inf.get("n_blob"),
                n_pruned=inf.get("n_pruned"),
                gamma_med=inf.get("gamma_med"),
                ell_median=inf.get("ell_median"),
                ell_meas_median=inf.get("ell_meas_median"),
                n_sel=f["n_sel"],
            )
            logger.info(
                "[BEADS] tile %d (%d, %d): %s prom %s",
                t["index"],
                t["y0"],
                t["x0"],
                f["status"],
                t["z1_ring_prominence"],
            )

        # Bead-presence decision.
        ok_idx = [i for i, f in enumerate(fits) if f["status"] == "ok"]
        n_pass = len(ok_idx)
        required = max(3, math.ceil(n_tiles / 2))
        prom_pass = [tiles[i]["z1_ring_prominence"] for i in ok_idx]
        prom_fail = [
            t["z1_ring_prominence"]
            for i, t in enumerate(tiles)
            if i not in ok_idx and np.isfinite(t["z1_ring_prominence"])
        ]
        n1 = [t["n_peaks_z1"] for t in tiles]
        n0 = [t["n_peaks_z0"] for t in tiles]
        decision: dict[str, Any] = dict(
            n_tiles=n_tiles,
            n_pass=n_pass,
            required=required,
            gate=gate,
            min_prom_pass=min(prom_pass) if prom_pass else None,
            margin_pass=(min(prom_pass) - gate) if prom_pass else None,
            max_prom_fail=max(prom_fail) if prom_fail else None,
            peaks_z0=n0,
            peaks_z1=n1,
        )
        report["decision"] = decision
        if n_pass < required:
            # Beads that are sharp at plane 1 instead of plane 0 leave no z0 point
            # sources: say so rather than "no beads".
            if float(np.median(n1)) >= 8 and float(np.median(n1)) > 3.0 * max(
                float(np.median(n0)), 1.0
            ):
                raise _fail(
                    "beads not focused at plane 0: bead-like sharp points are found at "
                    f"plane 1 (median {np.median(n1):.0f} per tile) but hardly at "
                    f"plane 0 (median {np.median(n0):.0f}); the correction needs beads "
                    "focused at plane 0. Set merscope_bead_removal false for this "
                    "sample.",
                    report,
                )
            if n_pass == 0:
                raise _fail(
                    f"no beads: none of the {n_tiles} tissue-free calibration tiles "
                    "shows a bead PSF (max z1 ring prominence "
                    f"{_fmt(decision['max_prom_fail'])}, gate {gate}). Set "
                    "merscope_bead_removal false for this sample.",
                    report,
                )
            raise _fail(
                f"ambiguous bead presence: {n_pass} of {n_tiles} tissue-free tiles "
                f"show a bead PSF, {required} are required. Set "
                "merscope_bead_removal false for this sample.",
                report,
            )
        passing = [fits[i] for i in ok_idx]

        # Pooled library, dataset-global constants and the sample grid (the reference
        # detector needs both).
        lib = pool_library(passing, params)
        z0c, z1c = float(lib.tmpls[0].vals[0]), float(lib.tmpls[1].vals[0])
        r1_lib = float(lib.summary()[1]["ring_radius_px"])
        t0 = time.time()
        sample_grid = sample_tile_grid(
            reader, stain=ref, z=0, n_grid=4, tile_px=GRID_TILE_PX, min_valid=0.9
        )
        noise, dog_noise, sample_tiles_xy = estimate_global_noise(
            reader,
            ref,
            k_fit,
            executor=ex,
            base=base,
            tile_px=GRID_TILE_PX,
            tiles=[(y, x) for _, _, y, x in sample_grid],
        )
        grid = (
            sample_grid
            if REFERENCE_GRID_N == 4
            else sample_tile_grid(
                reader,
                stain=ref,
                z=0,
                n_grid=REFERENCE_GRID_N,
                tile_px=GRID_TILE_PX,
                min_valid=0.9,
            )
        )
        timing["global_constants_s"] = time.time() - t0
        a0 = np.concatenate([f["a0"] for f in passing])
        if a0.size == 0:
            raise _fail("no classified bead on the passing tiles", report)
        lo, hi = np.percentile(a0, params.sel_pct)
        a0_median = float(np.median(a0))

        # Reference candidates of the permissive detector: the passing calibration
        # tiles and the sample-grid tiles.
        t0 = time.time()
        ref_jobs = [
            dict(
                base,
                y0=tiles[i]["y0"],
                x0=tiles[i]["x0"],
                tile_px=tile_px,
                dog_noise=dog_noise,
                a0_min_ref=REFERENCE_A0_MIN,
                r1_lib=r1_lib,
            )
            for i in ok_idx
        ]
        ref_jobs += [
            dict(
                base,
                y0=y,
                x0=x,
                tile_px=GRID_TILE_PX,
                dog_noise=dog_noise,
                a0_min_ref=REFERENCE_A0_MIN,
                r1_lib=r1_lib,
            )
            for _, _, y, x in grid
        ]
        refs_all = _map(ex, _reference_job, ref_jobs)
        refs_cal, refs_grid = refs_all[: len(ok_idx)], refs_all[len(ok_idx) :]
        timing["reference_s"] = time.time() - t0

        # Focus check on the permissive detector's bright spectrally confirmed
        # candidates (calibration tiles).
        focus = focus_check(refs_cal, z0c, z1c)
        decision["focus"] = focus
        if not focus["n_bright"]:
            warnings.append(
                "focus check: no bright spectrally confirmed candidate on the passing "
                "tiles; only the template centres were compared"
            )
        if not focus["passed"]:
            raise _fail(
                "beads not focused at plane 0; median z1/z0 centre ratio = "
                f"{focus['median_r1']:.3f} (limit {FOCUS_R1_MAX}), z0 template centre "
                f"{z0c:.3f} vs z1 {z1c:.3f}",
                report,
            )

        # Library bookkeeping and the tile-to-tile spread of the library.
        lib.info = dict(
            n_tiles=n_pass,
            weights=[int(f["n_sel"]) for f in passing],
            tile_indices=ok_idx,
            tile_templates={
                tiles[i]["index"]: {k: fits[i]["tmpl"][k] for k in range(4)}
                for i in ok_idx
            },
        )
        decision["library_spread"] = library_tile_spread(
            lib.info["tile_templates"], lib, params
        )

        # Stain responses.
        rho: dict[str, tuple[float, float]] = {}
        rho_raw: dict[str, Any] = {}
        for s in rho_stains:
            per_tile = [f["rho"][s] for f in passing if s in f["rho"]]
            val, se, med = pool_rho(per_tile, params)
            rho[s] = (val, se if np.isfinite(se) else 1.0e9)
            rho_raw[s] = dict(
                per_tile=[list(map(float, r)) for r in per_tile],
                pooled_median=med,
                used=val,
                se=se,
            )
        decision["rho"] = rho_raw
        for s, (v, _) in rho.items():
            if s != dapi and v != 0.0:
                warnings.append(
                    f"stain {s} shows a bead response of {v:.3f} of the reference "
                    "(recorded only; it is never corrected)"
                )

        # Classifier thresholds: the reference set is the calibration tiles plus the
        # threshold-role grid tiles; the recall is HELD OUT on the recall-role grid
        # tiles (fixed checkerboard). The grid includes tissue, so the set represents
        # the mosaic.
        t0 = time.time()
        rtiles: list[dict[str, Any]] = [
            dict(
                kind="calibration",
                role="calibration",
                index=int(i),
                y0=int(tiles[i]["y0"]),
                x0=int(tiles[i]["x0"]),
                size=tile_px,
                r=r,
            )
            for i, r in zip(ok_idx, refs_cal, strict=True)
        ] + [
            dict(
                kind="grid",
                role=checkerboard_role(gi, gj),
                grid_ij=[int(gi), int(gj)],
                y0=int(gy),
                x0=int(gx),
                size=GRID_TILE_PX,
                r=r,
            )
            for (gi, gj, gy, gx), r in zip(grid, refs_grid, strict=True)
        ]
        for t in rtiles:
            r = t["r"]
            t["m_iso"] = confirmed_mask(r)
            t["m_all"] = confirmed_mask(r, require_isolated=False)
            # All bright candidates of the tile, in mosaic coordinates.
            t["abs"] = np.c_[r["y"] + t["y0"], r["x"] + t["x0"]]
        pool_tiles = [t for t in rtiles if t["role"] in ("calibration", "threshold")]
        # Overlapping tiles see a bead twice: keep it once (first tile in the order
        # calibration, then threshold grid tiles).
        keep_pool = _dedupe_parts([t["abs"][t["m_iso"]] for t in pool_tiles])
        for t, k in zip(pool_tiles, keep_pool, strict=True):
            # Indices (into the tile's bright candidates) of the beads that set the
            # thresholds.
            t["use"] = np.where(t["m_iso"])[0][k]
        pool_pts = (
            np.concatenate([t["abs"][t["use"]] for t in pool_tiles])
            if pool_tiles
            else np.zeros((0, 2))
        )
        ref_set = {
            k: np.concatenate([t["r"][k][t["use"]] for t in pool_tiles])
            if pool_tiles
            else np.zeros(0)
            for k in ("rring", "r1", "rD0", "rD", "sd1rel")
        }
        thresholds, tinfo2 = derive_classifier_thresholds(
            ref_set,
            params,
            r1_ring_px=r1_lib,
            zprof_zmax=zprof_zmax_available(reader.planes),
        )
        if tinfo2["used_defaults"]:
            warnings.append(
                f"classifier thresholds: only {tinfo2['n_reference']} reference beads "
                f"(< {MIN_REFERENCE_BEADS}); the defaults are kept"
            )
        # Held-out reference beads: confirmed beads of the recall-role grid tiles that
        # are not a bead of the pool. ``use2`` is the diagnostic set that also holds
        # the non-isolated (cluster) beads: the threshold set cannot contain them, so
        # the isolation requirement could hide a classifier that fails in clusters.
        recall_tiles = [t for t in rtiles if t["role"] == "recall"]
        for key, mkey in (("use", "m_iso"), ("use2", "m_all")):
            keep = _dedupe_parts(
                [t["abs"][t[mkey]] for t in recall_tiles], against=pool_pts
            )
            for t, k in zip(recall_tiles, keep, strict=True):
                t[key] = np.where(t[mkey])[0][k]
        for t in rtiles:
            t.setdefault("use2", t["use"][:0])
        # Recall and density jobs: every calibration / grid tile through the
        # production detector with the final thresholds.
        rjobs = []
        for t in rtiles:
            r, u, u2 = t["r"], t["use"], t["use2"]
            rjobs.append(
                dict(
                    base,
                    y0=t["y0"],
                    x0=t["x0"],
                    tile_px=t["size"],
                    dog_noise=dog_noise,
                    thresholds=thresholds,
                    ref_y=r["y"][u],
                    ref_x=r["x"][u],
                    ref2_y=r["y"][u2],
                    ref2_x=r["x"][u2],
                )
            )
        rec = _map(ex, _recall_job, rjobs)
        per_tile_ring: list[dict[str, Any]] = []
        for t, r_ in zip(rtiles, rec, strict=True):
            u = t["use"]
            row = {
                k: t[k]
                for k in ("kind", "role", "index", "grid_ij", "y0", "x0")
                if k in t
            }
            per_tile_ring.append(
                dict(
                    row,
                    stats=_ring_stats(t["r"]["rring"][u], t["r"]["r1"][u]),
                    n_bright=int(t["r"]["n_bright"]),
                    n_confirmed=int(t["m_iso"].sum()),
                    n_reference_used=int(r_["n"]),
                    n_recalled=int(r_["hit"]),
                    n_reference_used_all=int(r_["n2"]),
                    n_recalled_all=int(r_["hit2"]),
                    n_classified=int(r_["n_classified"]),
                    n_rescued=int(r_["n_rescued"]),
                    n_valid_px=int(r_["n_valid"]),
                )
            )
        held = [t for t in per_tile_ring if t["role"] == "recall"]
        insample = [
            t for t in per_tile_ring if t["role"] in ("calibration", "threshold")
        ]
        n_held = int(sum(t["n_reference_used"] for t in held))
        recall = (
            float(sum(t["n_recalled"] for t in held) / n_held)
            if n_held
            else float("nan")
        )
        n_held2 = int(sum(t["n_reference_used_all"] for t in held))
        recall2 = (
            float(sum(t["n_recalled_all"] for t in held) / n_held2)
            if n_held2
            else float("nan")
        )
        n_in = int(sum(t["n_reference_used"] for t in insample))
        recall_in = (
            float(sum(t["n_recalled"] for t in insample) / n_in)
            if n_in
            else float("nan")
        )
        cal_ring = [
            t["r"]["rring"][t["use"]] for t in pool_tiles if t["role"] == "calibration"
        ]
        cal_ring_all = np.concatenate(cal_ring) if cal_ring else np.zeros(0)
        tinfo2.update(
            recall=recall,
            n_recall_reference=n_held,
            recall_reliable=bool(n_held >= MIN_RECALL_REFERENCE),
            recall_in_sample=recall_in,
            recall_with_clusters=recall2,
            n_recall_reference_with_clusters=n_held2,
            n_in_sample_reference=n_in,
            n_calibration_tile_reference=int(cal_ring_all.size),
            n_rescued_tiles=int(sum(t["n_rescued"] for t in per_tile_ring)),
            rescued_fraction_tiles=float(
                sum(t["n_rescued"] for t in per_tile_ring)
                / max(sum(t["n_classified"] for t in per_tile_ring), 1)
            ),
            n_grid_threshold_reference=int(
                sum(len(t["use"]) for t in pool_tiles if t["kind"] == "grid")
            ),
            n_candidates_bright=int(sum(r["n_bright"] for r in refs_all)),
            n_isolated=int(
                sum(int(r["isolated"].sum()) for r in refs_all if len(r["y"]))
            ),
            n_spectral=int(
                sum(
                    int((r["isolated"] & r["spectral"]).sum())
                    for r in refs_all
                    if len(r["y"])
                )
            ),
            ring_p2_calibration_tiles_only=float(np.percentile(cal_ring_all, 2))
            if cal_ring_all.size
            else None,
            tiles=per_tile_ring,
        )
        warnings.extend(classifier_recall_warnings(recall, n_held, recall2, n_held2))
        decision["classifier"] = tinfo2
        timing["thresholds_s"] = time.time() - t0
        grid_rows = [t for t in per_tile_ring if t["kind"] == "grid"]
        grid_beads = int(sum(t["n_classified"] for t in grid_rows))
        grid_valid = int(sum(t["n_valid_px"] for t in grid_rows))

        # Stretch fallback, measured in library mode with the constants above.
        t0 = time.time()
        pre = BeadCalibration(
            BEAD_REMOVAL_VERSION,
            params_sha256(params),
            list(cfg.stains),
            ref,
            lib,
            thresholds,
            noise,
            dog_noise,
            (float(lo), float(hi)),
            a0_median,
            0.0,
            rho,
        )
        # Measured on the sample grid (tissue included): beads under tissue are the
        # ones that lack neighbours in the tissue-free tiles, and the PSF there is
        # magnified, so the tissue-free tiles alone would be unrepresentative.
        jobs = [
            dict(base, y0=y, x0=x, tile_px=1024, crop_px=1024, cal=pre)
            for y, x in sample_tiles_xy
        ]
        ell_res = _map(ex, _ell_job, jobs)
        ell_vals = [
            r["ell_meas_median"] for r in ell_res if np.isfinite(r["ell_meas_median"])
        ]
        if ell_vals:
            ell_default = float(np.median(ell_vals))
        else:
            ell_default = float(np.median([tiles[i]["ell_median"] for i in ok_idx]))
            warnings.append(
                "stretch fallback: no library-mode stretch measurement; the "
                "tile-local median was used"
            )
        decision["ell_default"] = dict(
            value=ell_default,
            source="library mode, sample grid",
            library_mode_per_tile=[r["ell_meas_median"] for r in ell_res],
            tile_local_ell_median=[tiles[i]["ell_median"] for i in ok_idx],
            tile_local_ell_meas_median=[tiles[i]["ell_meas_median"] for i in ok_idx],
        )
        timing["ell_default_s"] = time.time() - t0

    # Cost forecast. Density = classified beads per valid pixel on the sample-grid
    # tiles (tissue included, the same detector and thresholds as the chunk pass), not
    # the tissue-free candidate tiles chosen for their bead count: the latter missed
    # the beads of the tissue area and under-counted a development mosaic by 1.48x.
    px_um = _px_per_um(reader.images_dir)
    peak_density = float(np.median(counts)) / float(tile_px * tile_px)
    if grid_valid > 0:
        dens, dens_source = (
            grid_beads / float(grid_valid),
            "sample grid (classified beads per valid pixel, tissue included)",
        )
    else:
        # No usable grid tile: the tissue-free candidates (classified / peak ratio of
        # the passing tiles).
        ratio = float(
            np.median(
                [tiles[i]["n_bead"] / max(tiles[i]["n_peaks_z0"], 1) for i in ok_idx]
            )
        )
        dens, dens_source = peak_density * ratio, "tissue-free candidate tiles"
    forecast = forecast_cost(
        density_per_px2=dens,
        valid_area_px=float(valid_ov.sum()) * 256.0,
        n_workers=int(cfg.max_workers),
        px_per_um=px_um,
        n_windows=count_windows(valid_ov),
        calibration_s=time.time() - t_start,
        backend=backend,
    )
    forecast["density_source"] = dens_source
    forecast["n_grid_tiles"] = int(len(grid_rows)) if grid_valid > 0 else 0
    forecast["beads_per_mm2_peak_count"] = peak_density * px_um**2 * 1e6
    if forecast["warn"]:
        warnings.append(
            f"cost forecast {forecast['wall_hours']:.1f} h wall on the "
            f"{forecast['backend']} backend ({forecast['workers']} CPU workers; "
            f"{forecast.get('gpu_wall_hours', float('nan')):.1f} h on the GPU) "
            f"(> {FORECAST_WARN_WALL_H} h)"
        )

    decision.update(
        status="ok",
        sample_tiles=[list(t) for t in sample_tiles_xy],
        global_constants=dict(a0_sel_pct=list(params.sel_pct)),
        z_layers=zs,
        k_fit=k_fit,
        warnings=warnings,
        timing_s=dict(timing, total_s=time.time() - t_start),
        n_workers=int(n_workers),
    )
    lib.info["rho_stains"] = rho_stains
    cal = BeadCalibration(
        version=BEAD_REMOVAL_VERSION,
        params_sha256=params_sha256(params),
        corrected_stains=list(cfg.stains),
        reference_stain=ref,
        library=lib,
        thresholds=thresholds,
        noise=noise,
        dog_noise_z0=dog_noise,
        a0_sel=(float(lo), float(hi)),
        a0_median=a0_median,
        ell_default=ell_default,
        rho=rho,
        tiles=tiles,
        decision=decision,
        tissue=tinfo,
        forecast=forecast,
    )
    logger.info(
        "[BEADS] calibration done in %.0f s: %d/%d tiles, thresholds %s, rho %s",
        time.time() - t_start,
        n_pass,
        n_tiles,
        thresholds,
        {k: round(v[0], 3) for k, v in rho.items()},
    )
    return cal


# The dataset-global sample grid.
def sample_tile_grid(
    reader: Any,
    *,
    stain: str = "PolyT",
    z: int = 0,
    n_grid: int = 4,
    tile_px: int = 1024,
    min_valid: float = 0.9,
) -> list[tuple[int, int, int, int]]:
    """Lay out an ``n_grid`` x ``n_grid`` grid of tiles over the valid area.

    The grid covers the bounding box of the valid area; tile origins are multiples
    of 16. Tiles with too little valid area are skipped.

    Args:
        reader: MosaicPlaneReader.
        stain: Stain whose non-zero pixels define the valid area (1/16 overview).
        z: Plane of that stain.
        n_grid: Grid size (production: 4 x 4).
        tile_px: Tile edge (production: 1024).
        min_valid: Tiles with a smaller valid fraction are skipped.

    Returns:
        List of (row i, column j, y0, x0), row-major.
    """
    ov = reader.overview(stain, z, 16)
    ys, xs = np.where(ov > 0)
    if ys.size == 0:
        return []
    by0, by1, bx0, bx1 = (
        ys.min() * 16,
        (ys.max() + 1) * 16,
        xs.min() * 16,
        (xs.max() + 1) * 16,
    )
    out: list[tuple[int, int, int, int]] = []
    for i in range(n_grid):
        for j in range(n_grid):
            cy = by0 + (i + 0.5) * (by1 - by0) / n_grid
            cx = bx0 + (j + 0.5) * (bx1 - bx0) / n_grid
            y0 = int(round((cy - tile_px / 2) / 16) * 16)
            x0 = int(round((cx - tile_px / 2) / 16) * 16)
            tile = reader.read(stain, z, y0, y0 + tile_px, x0, x0 + tile_px)
            if (tile > 0).mean() >= min_valid:
                out.append((i, j, y0, x0))
    return out


def sample_tiles(
    reader: Any,
    *,
    stain: str = "PolyT",
    z: int = 0,
    n_grid: int = 4,
    tile_px: int = 1024,
    min_valid: float = 0.9,
) -> list[tuple[int, int]]:
    """Return the origins (y0, x0) of the :func:`sample_tile_grid` tiles, row-major."""
    return [
        (y0, x0)
        for _, _, y0, x0 in sample_tile_grid(
            reader,
            stain=stain,
            z=z,
            n_grid=n_grid,
            tile_px=tile_px,
            min_valid=min_valid,
        )
    ]
