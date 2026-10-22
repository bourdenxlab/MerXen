"""Synthetic data for the bead-removal tests (no real data, no paths).

An analytic stand-in PSF per plane (sharp dot + ring at z0, defocused disc with a
bright rim above, radius 7 + 22.8 z), beads on a flat background with camera
noise, tissue blobs, nuclei (DAPI) and puncta (antibody stain).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import erfc

from merxen.io.bead_removal.detect import (
    ClassifierThresholds,
    dog_noise_of,
    robust_noise,
)
from merxen.io.bead_removal.params import (
    BEAD_REMOVAL_VERSION,
    BETA,
    DEFAULT_PARAMS,
    FLUX_TARGET,
    PsfSubtractParams,
    params_sha256,
)
from merxen.io.bead_removal.psf import PsfLibrary, Template

# centre / z0 centre of the defocused images
CENTRE = np.array([0.0, 0.144, 0.041, 0.019, 0.010, 0.004, 0.003, 0.001])
BG_POLYT = 1500.0
NOISE_SD = 20.0


def _parts(k: int, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (sharp part, scalable part) of the plane-k profile.

    At z0 these are the dot and the ring; above z0 the centre peak and the disc
    with its rim.
    """
    if k == 0:
        return (
            np.exp(-(r**2) / (2 * 1.5**2)),
            0.46 * np.exp(-((r - 10.7) ** 2) / (2 * 2.0**2)),
        )
    radius = 7.0 + 22.8 * k
    disc = 0.6 * 0.5 * erfc((r - radius) / (np.sqrt(2) * 2.0))
    rim = 0.4 * np.exp(-((r - radius) ** 2) / (2 * (2.0 + k) ** 2))
    return CENTRE[k] * np.exp(-(r**2) / (2 * 3.0**2)), disc + rim


@functools.cache
def _norm(k: int) -> float:
    """Scale the disc + rim so the plane flux is FLUX_TARGET[k] x the z0 flux.

    The centre peak is kept as it is.
    """
    if k == 0:
        return 1.0
    r = np.arange(0, 700, 0.05)

    def flux(profile: np.ndarray) -> float:
        return float(np.sum(profile * 2 * np.pi * r) * 0.05)

    flux0 = flux(sum(_parts(0, r)))
    sharp, wide = _parts(k, r)
    return (FLUX_TARGET[k] * flux0 - flux(sharp)) / flux(wide)


def psf_profile(k: int, r: np.ndarray) -> np.ndarray:
    """Return the response of plane k to a unit z0 point source at radius r (px)."""
    sharp, wide = _parts(k, np.asarray(r, np.float64))
    return np.clip(sharp + wide * _norm(k), 0, None)


def render_bead_psf(k: int, size: int = 401) -> np.ndarray:
    """Render the stand-in PSF of plane k (unit z0 centre amplitude), centred."""
    c = size // 2
    yy, xx = np.mgrid[:size, :size]
    return psf_profile(k, np.hypot(yy - c, xx - c)).astype(np.float32)


def library_from_generator(params: PsfSubtractParams = DEFAULT_PARAMS) -> PsfLibrary:
    """Build the PsfLibrary of the stand-in PSF's radial profiles.

    It makes ``correct_window`` testable without a calibration pass.
    """
    templates = []
    for k in range(8):
        n = int(np.ceil(params.r_sup[k] / params.ds)) + 1
        nodes = np.arange(n) * params.ds
        vals = psf_profile(k, nodes)
        taper = np.clip((params.r_sup[k] - nodes) / (params.r_sup[k] * 0.12), 0, 1)
        templates.append(
            Template(
                params.ds, params.r_sup[k], vals * (0.5 - 0.5 * np.cos(np.pi * taper))
            )
        )
    return PsfLibrary(templates)


def bead_free_constants() -> tuple[dict[int, float], float]:
    """Measure the global noise constants on a bead-free synthetic stack.

    Returns:
        (pixel noise sd per plane, DoG noise of z0), as a calibration pass would
        record them.
    """
    free = make_stack(
        n_beads=0, seed=123, size=768, blobs=False, punctum_near_bead=False
    )
    noise = {k: robust_noise(free["PolyT"][k], free["PolyT"][k] > 0) for k in range(8)}
    return noise, dog_noise_of(free["PolyT"][0])


@dataclass(frozen=True)
class SyntheticCalibration:
    """A dataset calibration of the synthetic stand-in PSF.

    It carries the attributes that ``correct_window`` reads from a calibration,
    so the window-level tests do not depend on the calibration pass (which has
    its own tests).

    Attributes:
        version: Method version.
        params_sha256: Hash of the parameter set.
        corrected_stains: Stains that are corrected.
        reference_stain: Stain the beads are fitted on.
        library: PSF library.
        thresholds: Classifier thresholds.
        noise: Pixel noise sd per plane.
        dog_noise_z0: Global DoG noise of z0.
        a0_sel: Range of z0 amplitudes of the beads that define the fit.
        a0_median: Median bead amplitude.
        ell_default: Default log-stretch.
        rho: Bead response per non-reference stain: (rho, standard error).
        decision: Free-form record of how the calibration was made.
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
    decision: dict[str, Any] = field(default_factory=dict)


def make_calibration(
    dapi_ratio: float = 0.11,
    amp: tuple[float, float] = (500.0, 6000.0),
    params: PsfSubtractParams = DEFAULT_PARAMS,
    rho_dapi: float | None = None,
) -> SyntheticCalibration:
    """Build a calibration for synthetic stacks.

    The global constants are measured on a bead-free synthetic stack and the
    library is the stand-in PSF's own profile.

    Args:
        dapi_ratio: DAPI bead response recorded as rho.
        amp: Bead z0 amplitude range of the stacks (sets the selection range).
        params: Parameter set (its hash is recorded).
        rho_dapi: Override of the DAPI rho (0 leaves DAPI unchanged).

    Returns:
        The calibration.
    """
    noise, dog_noise = bead_free_constants()
    lo = amp[0] * 0.9 * (amp[1] / amp[0]) ** 0.15
    hi = amp[0] * 0.9 * (amp[1] / amp[0]) ** 0.92
    rho = {
        "DAPI": (dapi_ratio if rho_dapi is None else rho_dapi, 0.003),
        "Anti-X": (0.0, 0.003),
    }
    return SyntheticCalibration(
        BEAD_REMOVAL_VERSION,
        params_sha256(params),
        ["PolyT", "DAPI"],
        "PolyT",
        library_from_generator(params),
        ClassifierThresholds.from_params(params),
        noise,
        dog_noise,
        (lo, hi),
        amp[0] * 0.9 * (amp[1] / amp[0]) ** 0.5,
        0.0,
        rho,
        decision={"status": "synthetic"},
    )


def _place(
    rng: np.random.Generator,
    n: int,
    size_yx: tuple[int, int],
    min_sep: float,
    avoid: list[tuple[float, float, float]],
    margin: int = 12,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    tries = 0
    while len(points) < n and tries < 200000:
        tries += 1
        y = rng.uniform(margin, size_yx[0] - margin)
        x = rng.uniform(margin, size_yx[1] - margin)
        if any(np.hypot(y - a, x - b) < r for a, b, r in avoid):
            continue
        if any(np.hypot(y - a, x - b) < min_sep for a, b in points):
            continue
        points.append((y, x))
    return np.array(points).reshape(-1, 2)


def _add_profile(
    img: np.ndarray, yc: float, xc: float, amp: float, k: int, s: float = 1.0
) -> None:
    half = int(np.ceil(220 * s)) + 1
    y0, y1 = max(0, int(yc) - half), min(img.shape[0], int(yc) + half + 1)
    x0, x1 = max(0, int(xc) - half), min(img.shape[1], int(xc) + half + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy = np.arange(y0, y1)[:, None] - yc
    xx = np.arange(x0, x1)[None, :] - xc
    r = np.hypot(yy, xx)
    img[y0:y1, x0:x1] += (amp * psf_profile(k, r / s) / (s * s)).astype(np.float32)


def _quantise(a: np.ndarray) -> np.ndarray:
    """Round to counts in 1..65535 (zero is the invalid-pixel marker)."""
    return np.clip(np.rint(a), 1, 65535).astype(np.float32)


def make_stack(
    n_planes: int = 8,
    size: int | tuple[int, int] = 1024,
    n_beads: int = 60,
    seed: int = 0,
    amp: tuple[float, float] = (500.0, 6000.0),
    tissue: bool = True,
    dapi_ratio: float = 0.11,
    focus_plane: int = 0,
    blobs: bool = True,
    punctum_near_bead: bool = True,
    tilt: float = 0.0,
    n_nuclei: int = 10,
    margin: int = 12,
    beads: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build synthetic PolyT / DAPI / Anti-X stacks with beads and tissue.

    Args:
        n_planes: Number of planes.
        size: Window edge (px) or (Y, X).
        n_beads: Number of beads (Poisson-like positions >= 30 px apart,
            log-uniform amplitudes).
        seed: Random seed.
        amp: Bead z0 amplitude range (counts).
        tissue: Add tissue (PolyT blobs, DAPI nuclei, antibody puncta).
        dapi_ratio: DAPI bead response relative to PolyT.
        focus_plane: Plane at which the beads are sharp (1 = beads not focused at
            plane 0).
        blobs: Add two PolyT tissue blobs (sd 8 px, amplitude 800) with a 260 px
            bead-free zone around each.
        punctum_near_bead: Add a PolyT punctum (sharpest at z3) 60 px from the
            first bead.
        tilt: Radius dilation ln-field across x (the within-FOV PSF stretch):
            ell = tilt * (x / X - 0.5).
        n_nuclei: Number of DAPI nuclei.
        margin: Minimum distance (px) of a random bead from the window edge.
        beads: Explicit beads (n x 3: y, x, amplitude) instead of random ones.

    Returns:
        dict with "PolyT", "DAPI" and "Anti-X" ([n_planes, Y, X] float32 in the
        uint16 range) plus ground-truth entries: "truth/PolyT" and "truth/DAPI"
        (tissue without beads, same noise), "beadlight/PolyT" (injected bead
        light per plane), "beads" (n x 3: y, x, amplitude), "blobs", "puncta"
        (n x 2 positions of the PolyT punctum), "antix_puncta".
    """
    height, width = (size, size) if isinstance(size, int) else size
    rng = np.random.default_rng(seed)
    avoid: list[tuple[float, float, float]] = []
    blob_pos = np.zeros((0, 2))
    if tissue and blobs:
        if height == width:
            blob_pos = np.array(
                [[0.14 * height, 0.15 * width], [0.86 * height, 0.85 * width]]
            )
        else:
            blob_pos = np.array(
                [[0.14 * height, 0.1 * width], [0.86 * height, 0.9 * width]]
            )
        avoid = [(float(a), float(b), 262.0) for a, b in blob_pos]
    if beads is not None:
        beads_yx = np.asarray(beads, float)[:, :2]
        amps = np.asarray(beads, float)[:, 2]
        n_beads = len(beads_yx)
    else:
        beads_yx = _place(rng, n_beads, (height, width), 30.0, avoid, margin=margin)
        n_beads = len(beads_yx)
        amps = np.exp(rng.uniform(np.log(amp[0]), np.log(amp[1]), n_beads))
    noise = rng.normal(0, NOISE_SD, (3, n_planes, height, width)).astype(np.float32)
    polyt_truth = np.full((n_planes, height, width), BG_POLYT, np.float32)
    dapi_truth = np.full((n_planes, height, width), BG_POLYT / 3, np.float32)
    antix = np.full((n_planes, height, width), BG_POLYT, np.float32)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    punctum = np.zeros((0, 2))
    antix_puncta = np.zeros((0, 2))
    if tissue:
        for by, bx in blob_pos:
            g = 800.0 * np.exp(-((yy - by) ** 2 + (xx - bx) ** 2) / (2 * 8.0**2))
            for k in range(n_planes):
                polyt_truth[k] += g.astype(np.float32)
        for ny, nx in rng.uniform([30, 30], [height - 30, width - 30], (n_nuclei, 2)):
            g = (
                3000.0 * np.exp(-((yy - ny) ** 2 + (xx - nx) ** 2) / (2 * 20.0**2))
            ).astype(np.float32)
            for k in range(1, min(n_planes, 7)):
                dapi_truth[k] += g
        antix_puncta = rng.uniform([30, 30], [height - 30, width - 30], (12, 2))
        for py, px in antix_puncta:
            for k in range(n_planes):
                d = abs(k - 3)
                sd = 1.5 * (1 + d)
                antix[k] += (
                    2500.0
                    / (1 + d) ** 2
                    * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sd**2))
                ).astype(np.float32)
        if punctum_near_bead and n_beads:
            ang = rng.uniform(0, 2 * np.pi)
            py = float(np.clip(beads_yx[0, 0] + 60 * np.sin(ang), 20, height - 20))
            px = float(np.clip(beads_yx[0, 1] + 60 * np.cos(ang), 20, width - 20))
            punctum = np.array([[py, px]])
            for k in range(n_planes):
                d = abs(k - 3)
                sd = 1.5 * (1 + d)
                polyt_truth[k] += (
                    2500.0
                    / (1 + d) ** 2
                    * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sd**2))
                ).astype(np.float32)
    bead_light = np.zeros((n_planes, height, width), np.float32)
    for (by, bx), a in zip(beads_yx, amps, strict=True):
        ell = tilt * (bx / width - 0.5)
        for k in range(n_planes):
            kk = abs(k - focus_plane)
            beta = BETA[min(kk, 7)]
            _add_profile(
                bead_light[k], by, bx, float(a), min(kk, 7), float(np.exp(beta * ell))
            )
    polyt = polyt_truth + noise[0] + bead_light
    dapi = dapi_truth + noise[1] + dapi_ratio * bead_light
    anti_x = antix + noise[2]
    return {
        "PolyT": _quantise(polyt),
        "DAPI": _quantise(dapi),
        "Anti-X": _quantise(anti_x),
        "truth/PolyT": _quantise(polyt_truth + noise[0]),
        "truth/DAPI": _quantise(dapi_truth + noise[1]),
        "beadlight/PolyT": bead_light,
        "beads": np.column_stack([beads_yx, amps]) if n_beads else np.zeros((0, 3)),
        "blobs": blob_pos,
        "puncta": punctum,
        "antix_puncta": antix_puncta,
    }
