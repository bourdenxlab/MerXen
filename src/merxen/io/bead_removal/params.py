"""Parameters and constants of the psf_subtract bead-removal method (version 1).

Every algorithm parameter is fixed in code (a frozen dataclass) and recorded in
the provenance through :func:`params_sha256`; only the on/off switch is user
facing. The values are those of the validated prototype, with the development
switches removed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import numpy as np

BEAD_REMOVAL_VERSION = 1
# Planes z0..z7 exist in the templates; a z index above 7 is refused.
NZ_MAX = 8
# Absolute grid (px) of the lower-envelope blocks: window origins must be
# multiples of this to be chunk independent.
ENVELOPE_BLOCK_PX = 16

# Plane flux / z0 flux of a bead (conserved light with a slow loss), used to
# scale the self-similar templates of planes z4..z7.
FLUX_TARGET = np.array([1.0, 1.01, 0.97, 0.90, 0.87, 0.81, 0.75, 0.67])
# d ln s_k / d ln s_1: how much the radius dilation of plane k follows the
# plane-1 dilation (defocus geometry).
BETA = np.array([1.44, 1.0, 0.62, 0.42, 0.31, 0.25, 0.21, 0.18])


@dataclass(frozen=True)
class PsfSubtractParams:
    """Fixed parameters of the empirical-PSF forced-photometry bead model.

    Attributes are grouped as in the method description: detection,
    classification defaults (the per-dataset values live in
    ``ClassifierThresholds``), pruning, templates, blobs, fitting, stretch
    field and output.
    """

    # --- detection (z0 of the reference stain)
    # DoG peak threshold in units of the (global) DoG noise.
    k_sigma: float = 5.0
    # Pixels at/above this are excluded from the fits (off: no saturation seen).
    sat_level: float = float("inf")
    # Minimum z0 amplitude (counts above local background) of a modelled bead.
    a0_min: float = 300.0
    nms_px: float = 22.0
    peak_min_distance: int = 4
    border: int = 3
    # Candidates this close to an invalid pixel (outside the mosaic) are not
    # classified.
    edge_margin_px: int = 24
    # --- classifier defaults (a calibration replaces ring_min, r1_max,
    # rd_center and rd0_center per dataset)
    # DAPI z0 / PolyT z0 amplitude ratio of a bead (spectral signature).
    rd_center: float = 0.097
    rd0_center: float = 0.106
    rd_tol: float = 0.12
    # z0 ring (r 8-15 px) / centre amplitude: beads ~0.41, tissue puncta ~0.05.
    ring_min: float = 0.30
    ring_max: float = 1.30
    # PolyT z1 / z0 centre amplitude ratio (beads are weak at the z1 centre).
    r1_max: float = 0.25
    # --- pruning (after the z1/z2 fits)
    # Consensus ring gain below this (when informative) => not a bead.
    prune_gamma: float = 0.20
    prune_gamma_blob: float = 0.08
    # A gain this high means the model cannot explain the source (tissue).
    prune_gamma_hi: float = 2.0
    # Singles: fitted z0 amplitude / detection amplitude.
    a_ratio_max: float = 2.2
    prune_info: float = 60.0
    # --- calibration-time guards (local tile fit)
    # The local template must show a z1 ring (bead tiles 0.015-0.176, bead-free
    # <= -0.004).
    min_ring_prom: float = 0.010
    # Sources with a clearly fitted z1/z2 ring needed to call a tile
    # bead-bearing.
    min_evidence: int = 8
    # --- templates (px)
    ds: float = 0.5
    r_sup: tuple[float, ...] = (30, 62, 92, 114, 132, 152, 190, 200)
    smooth_px: tuple[float, ...] = (0.6, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0)
    win_extra: int = 26
    # Percentiles of the singles used for template stacking (calibration only).
    sel_pct: tuple[float, float] = (15.0, 92.0)
    # --- blobs (extended aggregates)
    blob_q: float = 1.30
    blob_snr: float = 12.0
    # Blob candidates must have A0 above this multiple of the median bead A0
    # (global).
    blob_a_rel: float = 2.5
    blob_g_max: float = 2.4
    blob_g_sd: float = 0.6
    flux_tol: float = 1.15
    blob_half: int = 40
    rl_iter: int = 30
    r_ap: float = 18.0
    # --- fitting
    n_outer: int = 3
    n_sweeps: int = 3
    huber_k: float = 1.6
    # Noise inflation for spatially correlated tissue texture.
    tex: float = 2.0
    g_prior: float = 1.0
    g_prior_sd: tuple[float, ...] = (3.0, 0.30, 0.35, 0.30, 0.30, 0.30, 0.30, 0.30)
    g_max: float = 1.8
    # Planes >= 2: fit only sources whose peak ring amplitude exceeds this x
    # noise.
    active_snr: float = 0.5
    max_sel_hi: int = 80
    rel_tol0: float = 0.30
    rel_tol: float = 0.30
    # --- stretch field
    stretch_sigma: float = 140.0
    stretch_snr: float = 8.0
    # Candidate rule: per absolute block of this size ...
    stretch_block_px: int = 512
    # ... keep the brightest this many candidates.
    stretch_per_block: int = 2
    # Fewer samples than this within 2.5 sigma -> the dataset default stretch.
    # The prototype used 4 with up to 120 candidates per window; with the
    # absolute-grid rule (2 per 512 px block) only ~3 samples fall within 2.5
    # sigma on average, so 4 left ~84 % of the beads at the default stretch.
    # 2 keeps the prototype's result (measured).
    stretch_kmin: int = 2
    # --- other stains / output
    # Responses below 2 % of the reference stain are treated as no bead signal.
    rho_min: float = 0.02
    # |rho| must exceed this many standard errors.
    rho_min_sig: float = 3.0
    # A corrected non-reference stain never uses a response above this.
    rho_cap: float = 0.5


DEFAULT_PARAMS = PsfSubtractParams()


def params_sha256(params: PsfSubtractParams) -> str:
    """Hash every parameter value for the provenance and the calibration.

    Args:
        params: The parameter set.

    Returns:
        Hex SHA-256 of the sorted-key JSON dump of the dataclass.
    """
    blob = json.dumps(dataclasses.asdict(params), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
