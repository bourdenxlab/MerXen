"""Subtraction of the bead model and the window-level entry point.

:func:`correct_window` is the one function the chunk driver calls: it fits the
bead model on the reference stain of a window, subtracts it from every corrected
stain (scaled by that stain's bead response) and returns the corrected planes
together with the model and a statistics dictionary.
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from merxen.io.bead_removal.detect import PlaneStack, empty_sources, lower_envelope
from merxen.io.bead_removal.model import (
    BeadModel,
    WindowCalibration,
    fit_window_library,
)
from merxen.io.bead_removal.params import (
    DEFAULT_PARAMS,
    ENVELOPE_BLOCK_PX,
    NZ_MAX,
    PsfSubtractParams,
    params_sha256,
)

logger = logging.getLogger(__name__)

# Stain whose planes 0 and 1 feed the spectral part of the bead classifier
# (version 1 requires it).
SPECTRAL_STAIN = "DAPI"
# Reference pixels at or above this are counted in the QC statistics.
SATURATION_LEVEL = 65000.0


class CorrectionCalibration(WindowCalibration, Protocol):
    """What :func:`correct_window` needs from a dataset calibration.

    Adds the identity of the calibration (which stains it corrects and the
    parameter hash it was made with) and the per-stain bead response to the
    attributes of the fit.
    """

    @property
    def params_sha256(self) -> str:
        """Hash of the parameter set the calibration was made with."""

    @property
    def corrected_stains(self) -> Sequence[str]:
        """Stains that are corrected (the reference stain among them)."""

    @property
    def reference_stain(self) -> str:
        """Stain the beads are detected and fitted on."""

    @property
    def rho(self) -> Mapping[str, tuple[float, float]]:
        """Bead response of each non-reference stain: (rho, standard error)."""


class FillSettings(Protocol):
    """What :func:`correct_window` reads from the optional fill configuration."""

    @property
    def enabled(self) -> bool:
        """Whether the fill runs at all."""

    @property
    def stains(self) -> Sequence[str]:
        """Stains the fill acts on."""


def apply_bead_model(
    stack: PlaneStack,
    model: BeadModel,
    *,
    rho: float,
    stain_is_reference: bool,
    planes: Sequence[int],
) -> np.ndarray:
    """Subtract the bead model from the requested planes of one stain.

    For every plane k in ``planes``: ``delta = max(rho * M_k, 0)`` (never add
    light), then ``delta = min(delta, max(I_k - B_k, 0))`` (never go below the
    local lower envelope) and ``out_k = I_k - delta`` where the pixel is valid
    (> 0). So ``out <= I`` everywhere.

    Args:
        stack: The stain's planes (array [nz, Y, X] or {z: 2-D}); the input is
            never modified.
        model: Fitted window model (an empty model returns the planes unchanged).
        rho: Bead response of this stain relative to the reference (1.0 for the
            reference stain).
        stain_is_reference: Use the model's own envelope (reference stain) or
            estimate one from this stain (sub-sampled).
        planes: Planes to correct, in output order.

    Returns:
        float32 array [len(planes), Y, X] with the corrected planes.
    """
    first = np.asarray(stack[planes[0]])
    out = np.empty((len(planes),) + first.shape, np.float32)
    for i, k in enumerate(planes):
        img = np.asarray(stack[k], np.float32)
        if k not in model.fits or rho <= 0:
            out[i] = img
            continue
        valid = img > 0
        env = model.B[k] if stain_is_reference else lower_envelope(img, valid, sub=4)
        delta = np.maximum(rho * model.fits[k].model, 0.0)
        delta = np.minimum(delta, np.maximum(img - env, 0.0))
        out[i] = np.where(valid, img - delta, img)
    return out


def _normalise_z_layers(z_layers: Sequence[int]) -> list[int]:
    """Validate the planes to output and return them as a list of ints.

    Raises:
        ValueError: ``z_layers`` is empty, contains plane 0, or a plane above 7.
    """
    zs = [int(z) for z in z_layers]
    if not zs:
        raise ValueError("z_layers is empty")
    if min(zs) < 1:
        raise ValueError(
            "bead removal needs z_layers to start at plane 1 or later (plane 0 is "
            "the bead plane and is read for detection only); "
            f"got {sorted(zs)}. Use for example 1-6."
        )
    if max(zs) > NZ_MAX - 1:
        raise ValueError(
            f"bead removal supports planes up to {NZ_MAX - 1}; got {max(zs)}"
        )
    return zs


def _plane(stack: PlaneStack, z: int, stain: str) -> np.ndarray:
    """Return plane ``z`` of a stain's stack as a 2-D array.

    Raises:
        ValueError: The plane is missing or not 2-D.
    """
    try:
        arr = np.asarray(stack[z])
    except (KeyError, IndexError) as exc:
        raise ValueError(f"stain {stain!r} has no plane {z}") from exc
    if arr.ndim != 2:
        raise ValueError(f"stain {stain!r} plane {z} is not 2-D")
    return arr


def _has_plane(stack: PlaneStack, z: int) -> bool:
    """Return whether plane ``z`` exists in a stain's stack."""
    try:
        stack[z]
    except (KeyError, IndexError):
        return False
    return True


def _pad_planes(
    planes: dict[int, np.ndarray], pad_tl: tuple[int, int], pad_br: tuple[int, int]
) -> dict[int, np.ndarray]:
    """Pad planes with invalid zeros (top-left and bottom-right)."""
    if not any(pad_tl) and not any(pad_br):
        return planes
    pad = ((pad_tl[0], pad_br[0]), (pad_tl[1], pad_br[1]))
    return {z: np.pad(a, pad, mode="constant") for z, a in planes.items()}


def correct_window(
    stacks: Mapping[str, PlaneStack],
    *,
    origin_yx: tuple[int, int],
    calibration: CorrectionCalibration,
    z_layers: Sequence[int],
    params: PsfSubtractParams = DEFAULT_PARAMS,
    fill: FillSettings | None = None,
    core: tuple[int, int, int, int] | None = None,
) -> tuple[dict[str, np.ndarray], BeadModel, dict[str, Any]]:
    """Remove coverslip beads from one window.

    This is the function the chunk driver and the validation harness both call.
    The window must contain the whole neighbourhood of what is kept (halo
    >= 256 px around the core). All decisions use the dataset calibration;
    nothing is derived from the window's own statistics.

    Args:
        stacks: Stain name -> planes, array [nz, Y, X] (plane index = z) or
            mapping {z: 2-D}. The reference stain needs planes 0, 1, 2 and every
            plane in ``z_layers`` (and reads planes 2..zmax of the rescue's
            z-profile test when it has them); DAPI (and any other corrected stain) needs
            planes 0, 1 (detection) and ``z_layers``. A value of 0 marks an
            invalid pixel (outside the mosaic).
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0). Block grids
            are aligned to absolute coordinates: when the origin is not a
            multiple of 16 the window is padded (zeros = invalid) internally.
        calibration: Dataset calibration.
        z_layers: Planes to output; every plane must be >= 1 (plane 0 is read
            for detection only) and <= 7.
        params: Fixed parameters.
        fill: Optional fill configuration. The bounded fill runs after the
            subtraction for the stains in ``fill.stains`` when ``fill.enabled``
            (off by default); its counters go to ``stats["fill"][stain]``.
        core: Optional (y0, y1, x0, x1) window sub-region the statistics are
            restricted to (default: the whole window).

    Returns:
        (corrected, model, stats). ``corrected`` maps each stain of
        ``calibration.corrected_stains`` to a float32 array
        [len(z_layers), Y, X] (slice i = plane ``z_layers[i]``); other stains are
        not returned and are projected raw by the caller. ``model`` is the fitted
        :class:`BeadModel` (empty when no bead was detected: output = raw).
        ``stats`` holds counts, timings and per-stain, per-plane change
        statistics.

    Raises:
        ValueError: ``z_layers`` is empty, contains 0 or a plane above 7; a
            required stain or plane is missing or has the wrong shape; the
            calibration's ``params_sha256`` differs from the hash of ``params``.
    """
    t_start = time.time()
    zs = _normalise_z_layers(z_layers)
    ref = calibration.reference_stain
    corrected_stains = list(calibration.corrected_stains)
    for stain in [*corrected_stains, ref, SPECTRAL_STAIN]:
        if stain not in stacks:
            raise ValueError(
                f"stain {stain!r} is required by the bead calibration but missing "
                "from the stacks"
            )
    if calibration.params_sha256 != params_sha256(params):
        raise ValueError(
            "the bead calibration was made with a different parameter set than the "
            f"one in use (calibration {calibration.params_sha256[:12]}, params "
            f"{params_sha256(params)[:12]}); recalibrate or pass the parameters of "
            "the calibration"
        )
    k_fit = max(2, max(zs))
    need = {s: sorted({0, 1} | set(zs)) for s in corrected_stains}
    need[ref] = sorted({0, 1, 2} | set(zs))
    # The rescue's "no secondary focus" test reads reference planes 2..zmax; the
    # ones the stack has are added, so the classifier does not depend on the
    # ``z_layers`` of the call.
    need[ref] = sorted(
        set(need[ref])
        | {
            k
            for k in calibration.thresholds.rescue_planes
            if _has_plane(stacks[ref], k)
        }
    )
    need[SPECTRAL_STAIN] = sorted(set(need.get(SPECTRAL_STAIN, [])) | {0, 1})

    raw: dict[str, dict[int, np.ndarray]] = {}
    for stain, planes in need.items():
        raw[stain] = {
            k: np.asarray(_plane(stacks[stain], k, stain), np.float32) for k in planes
        }
    height, width = next(iter(raw[ref].values())).shape
    for stain, by_plane in raw.items():
        for k, arr in by_plane.items():
            if arr.shape != (height, width):
                raise ValueError(
                    f"stain {stain!r} plane {k} has shape {arr.shape}, "
                    f"expected {(height, width)}"
                )

    # The lower-envelope block grids must sit on the absolute 16 px grid: pad so
    # that the origin and the size are multiples of the block.
    oy, ox = int(origin_yx[0]), int(origin_yx[1])
    pad_tl = (oy % ENVELOPE_BLOCK_PX, ox % ENVELOPE_BLOCK_PX)
    pad_br = (
        (-(height + pad_tl[0])) % ENVELOPE_BLOCK_PX,
        (-(width + pad_tl[1])) % ENVELOPE_BLOCK_PX,
    )
    padded = any(pad_tl) or any(pad_br)
    work = {s: _pad_planes(raw[s], pad_tl, pad_br) for s in raw} if padded else raw
    origin_aligned = (oy - pad_tl[0], ox - pad_tl[1])
    window = (slice(pad_tl[0], pad_tl[0] + height), slice(pad_tl[1], pad_tl[1] + width))

    if bool((work[ref][0] > 0).any()):
        model = fit_window_library(
            work[ref],
            work[SPECTRAL_STAIN],
            origin_yx=origin_aligned,
            calibration=calibration,
            params=params,
            planes=zs,
        )
    else:
        model = BeadModel(
            empty_sources(),
            {},
            {},
            {},
            {"n_cand": 0, "n_bead": 0, "n_rescued": 0, "status": "no valid pixel"},
        )
    t_fit = time.time() - t_start

    out: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = dict(model.info)
    stats["status"] = model.info.get("status")
    y0, y1, x0, x1 = core if core is not None else (0, height, 0, width)
    per_stain: dict[str, dict[str, Any]] = {}
    stats["per_stain"] = per_stain
    for stain in corrected_stains:
        is_ref = stain == ref
        rho = (
            1.0
            if is_ref
            else min(float(calibration.rho.get(stain, (0.0, 0.0))[0]), params.rho_cap)
        )
        stack_s = work[stain]
        arr = apply_bead_model(
            stack_s, model, rho=rho, stain_is_reference=is_ref, planes=zs
        )
        # The fill inputs (model light and raw copies of every plane, about 1 GB
        # for a production window) are built only for the stains the fill acts
        # on; a disabled fill or another stain costs nothing.
        if (
            fill is not None
            and fill.enabled
            and stain in fill.stains
            and model.fits
            and rho > 0
        ):
            arr = _apply_fill(arr, model, stack_s, zs, rho, fill, stain, stats)
        sy = slice(pad_tl[0] + y0, pad_tl[0] + y1)
        sx = slice(pad_tl[1] + x0, pad_tl[1] + x1)
        raw_core = np.stack([stack_s[k][sy, sx] for k in zs])
        diff = raw_core - arr[:, sy, sx]
        per_stain[stain] = {
            "rho": rho,
            "frac_gt10": [float((np.abs(d) > 10).mean()) for d in diff],
            "frac_gt100": [float((np.abs(d) > 100).mean()) for d in diff],
            "removed_sum": [float(d.sum()) for d in diff],
            "removed_p99": [float(np.percentile(d, 99)) for d in diff],
            "max_brightening": float(max(0.0, (-diff).max())) if diff.size else 0.0,
        }
        if is_ref:
            stats["n_saturated_px"] = int(
                sum(int((r >= SATURATION_LEVEL).sum()) for r in raw_core)
            )
        out[stain] = arr[:, window[0], window[1]] if padded else arr
    stats["z_layers"] = zs
    stats["k_fit"] = k_fit
    stats["time_fit_s"] = t_fit
    stats["time_total_s"] = time.time() - t_start
    logger.debug(
        "[BEADS] window origin %s: %d beads, %.1f s",
        origin_yx,
        stats.get("n_beads_final", 0),
        stats["time_total_s"],
    )
    return out, model, stats


def _apply_fill(
    corrected: np.ndarray,
    model: BeadModel,
    stack: dict[int, np.ndarray],
    zs: list[int],
    rho: float,
    fill: FillSettings,
    stain: str,
    stats: dict[str, Any],
) -> np.ndarray:
    """Run the bounded fill on one stain after the subtraction.

    The fill may only lower pixels further, so the result is clamped to the
    subtraction result, and pixels that are invalid (0) in the raw plane stay
    unchanged.

    Args:
        corrected: Subtraction result [len(zs), Y, X].
        model: Fitted window model.
        stack: The stain's (padded) raw planes.
        zs: Planes in output order.
        rho: Bead response of the stain.
        fill: Fill configuration (also passed to the fill as its ``config``).
        stain: Stain name.
        stats: Window statistics; the fill's counters are stored in
            ``stats["fill"][stain]``.

    Returns:
        float32 array [len(zs), Y, X].
    """
    # Imported here so that a run without the fill never loads the experimental
    # module.
    fill_module = importlib.import_module("merxen.io.bead_removal.fill")
    light = np.stack(
        [np.maximum(rho * model.fits[k].model, 0.0).astype(np.float32) for k in zs]
    )
    raw_planes = np.stack([stack[k] for k in zs])
    fill_stats: dict[str, Any] = {}
    filled = fill_module.apply_bounded_fill(
        corrected,
        light,
        raw=raw_planes,
        z_layers=zs,
        config=fill,
        stain=stain,
        noise=dict(model.noise.items()),
        stats=fill_stats,
    )
    stats.setdefault("fill", {})[stain] = fill_stats
    result: np.ndarray = np.where(
        raw_planes > 0, np.minimum(np.asarray(filled, np.float32), corrected), corrected
    ).astype(np.float32)
    return result
