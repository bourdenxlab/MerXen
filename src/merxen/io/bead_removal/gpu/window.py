"""GPU end-to-end window correction.

The counterpart of ``merxen.io.bead_removal.apply.correct_window``, with an optional
device-side max projection. The hybrid mode of the fill runs on the device between the
subtraction and the projection (:mod:`.fill`); the flat mode is CPU only.
"""

# Window dimensions and matrices keep the capital letters of the formulas they port
# from the CPU modules (H, W, B, M, K, A0 ...), so the two can be read side by side.
# ruff: noqa: N803, N806

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch

from merxen.io.bead_removal.apply import (
    SATURATION_LEVEL,
    SPECTRAL_STAIN,
    CorrectionCalibration,
    _has_plane,
    _normalise_z_layers,
    _pad_planes,
    _plane,
)
from merxen.io.bead_removal.detect import empty_sources
from merxen.io.bead_removal.fill import FillParams
from merxen.io.bead_removal.model import BeadModel
from merxen.io.bead_removal.params import (
    DEFAULT_PARAMS,
    ENVELOPE_BLOCK_PX,
    PsfSubtractParams,
    params_sha256,
)

from . import ops
from .fill import fill_plane
from .fit import FitOpts
from .model import fit_window_library

logger = logging.getLogger(__name__)


def apply_bead_model(
    img: torch.Tensor, B: torch.Tensor | None, M: torch.Tensor | None, rho: float
) -> torch.Tensor:
    """Subtract the modelled bead light from a plane (device version).

    ``delta = max(rho M, 0)`` is capped at the light above the envelope,
    ``max(img - B, 0)``, and only valid pixels (``img > 0``) change.
    """
    if M is None or rho <= 0:
        return img.clone()
    valid = img > 0
    env = B if B is not None else ops.lower_envelope(img, valid, sub=4)
    delta = torch.clamp(rho * M, min=0.0)
    delta = torch.minimum(delta, torch.clamp(img - env, min=0.0))
    return torch.where(valid, img - delta, img)


def _pct(x: torch.Tensor, q: float) -> float:
    s, _ = x.reshape(-1).sort()
    n = s.numel()
    pos = q / 100.0 * (n - 1)
    lo = int(np.floor(pos))
    f = pos - lo
    return float(s[lo].double() * (1 - f) + s[min(lo + 1, n - 1)].double() * f)


def correct_window(
    stacks: Mapping[str, Any],
    *,
    origin_yx: tuple[int, int],
    calibration: CorrectionCalibration,
    z_layers: Sequence[int],
    params: PsfSubtractParams = DEFAULT_PARAMS,
    fill: FillParams | None = None,
    core: tuple[int, int, int, int] | None = None,
    opts: FitOpts | None = None,
    output: str = "numpy",
    timers: dict[str, float] | None = None,
) -> tuple[dict[str, Any], BeadModel, dict[str, Any]]:
    """GPU equivalent of ``merxen.io.bead_removal.correct_window``.

    Args:
        stacks: Stain -> planes ({z: 2-D numpy array} or [nz, Y, X]); numpy (host) or
            torch (device) planes.
        origin_yx: Absolute mosaic coordinate of window pixel (0, 0).
        calibration: Dataset calibration.
        z_layers: Planes to output (1..7).
        params: Fixed parameters.
        fill: Optional fill configuration. The hybrid fill runs on the device after
            the subtraction for the stains in ``fill.stains`` when ``fill.enabled``
            (its counters go to ``stats["fill"][stain]``, as on the CPU); the flat
            fill has no GPU implementation.
        core: Optional (y0, y1, x0, x1) the change statistics are computed on.
        opts: GPU execution options (default: the production configuration, see
            ``FitOpts``).
        output: "numpy" (corrected planes as host float32 arrays, as the CPU function),
            "device" (torch tensors), or "maxproj" (host float32 [Y, X] max projection
            over ``z_layers`` per stain: what the pipeline needs).
        timers: Optional dict receiving per-stage wall seconds.

    Returns:
        (corrected, model, stats) as the CPU function; ``model.fits[k].model`` /
        ``model.B[k]`` are device tensors.

    Raises:
        ValueError: ``z_layers`` invalid, a required stain or plane is missing, the
            calibration's ``params_sha256`` differs from the hash of ``params``, or
            ``fill`` is enabled in a mode other than hybrid.
    """
    if fill is not None and fill.enabled and fill.mode != "hybrid":
        raise ValueError(
            f"the {fill.mode} bead-removal fill has no GPU implementation; run it "
            "on the CPU backend or use the hybrid fill"
        )
    tm_ = timers if timers is not None else {}

    def tick(name: str, t0: float) -> float:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tm_[name] = tm_.get(name, 0.0) + (t1 - t0)
        tm_["peak_alloc_gb"] = max(
            tm_.get("peak_alloc_gb", 0.0), torch.cuda.max_memory_allocated() / 1e9
        )
        return t1

    t0 = t = time.perf_counter()
    zs = _normalise_z_layers(z_layers)
    ref = calibration.reference_stain
    corrected_stains = list(calibration.corrected_stains)
    for s in corrected_stains + [ref, SPECTRAL_STAIN]:
        if s not in stacks:
            raise ValueError(
                f"stain {s!r} is required by the bead calibration but missing from "
                "the stacks"
            )
    if calibration.params_sha256 != params_sha256(params):
        raise ValueError(
            "the bead calibration was made with a different parameter set than the "
            f"one in use (calibration {calibration.params_sha256[:12]}, "
            f"params {params_sha256(params)[:12]}); "
            "recalibrate or pass the parameters of the calibration"
        )
    k_fit = max(2, max(zs))
    need = {s: sorted({0, 1} | set(zs)) for s in corrected_stains}
    need[ref] = sorted({0, 1, 2} | set(zs))
    # Reference planes 2..zmax of the rescue's z-profile test (those the stack has).
    need[ref] = sorted(
        set(need[ref])
        | {
            k
            for k in calibration.thresholds.rescue_planes
            if _has_plane(stacks[ref], k)
        }
    )
    need[SPECTRAL_STAIN] = sorted(set(need.get(SPECTRAL_STAIN, [])) | {0, 1})
    raw_h: dict[str, dict[int, Any]] = {}
    for s, ks in need.items():
        raw_h[s] = {k: _plane(stacks[s], k, s) for k in ks}
    first = next(iter(raw_h[ref].values()))
    Y, X = first.shape
    oy, ox = int(origin_yx[0]), int(origin_yx[1])
    pad_tl = (oy % ENVELOPE_BLOCK_PX, ox % ENVELOPE_BLOCK_PX)
    pad_br = (
        (-(Y + pad_tl[0])) % ENVELOPE_BLOCK_PX,
        (-(X + pad_tl[1])) % ENVELOPE_BLOCK_PX,
    )
    padded = any(pad_tl) or any(pad_br)
    if padded:
        raw_h = {
            s: _pad_planes(
                {k: np.asarray(a, np.float32) for k, a in v.items()}, pad_tl, pad_br
            )
            for s, v in raw_h.items()
        }
    origin_al = (oy - pad_tl[0], ox - pad_tl[1])
    sl = (slice(pad_tl[0], pad_tl[0] + Y), slice(pad_tl[1], pad_tl[1] + X))
    dev = ops.device()
    raw: dict[str, dict[int, torch.Tensor]] = {}
    for s, v in raw_h.items():
        raw[s] = {
            k: (
                a
                if isinstance(a, torch.Tensor)
                else torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
                    dev, non_blocking=True
                )
            )
            for k, a in v.items()
        }
    t = tick("h2d", t)
    if bool((raw[ref][0] > 0).any()):
        model = fit_window_library(
            raw[ref],
            raw[SPECTRAL_STAIN],
            origin_yx=origin_al,
            calibration=calibration,
            params=params,
            planes=zs,
            opts=opts,
            timers=tm_,
        )
    else:
        model = BeadModel(
            empty_sources(),
            {},
            {},
            {},
            dict(n_cand=0, n_bead=0, n_rescued=0, status="no valid pixel"),
        )
    t_fit = time.perf_counter() - t0
    t = time.perf_counter()
    stats: dict[str, Any] = dict(model.info)
    y0, y1, x0, x1 = core if core is not None else (0, Y, 0, X)
    stats["per_stain"] = {}
    out: dict[str, Any] = {}
    for s in corrected_stains:
        is_ref = s == ref
        rho = (
            1.0
            if is_ref
            else min(float(calibration.rho.get(s, (0.0, 0.0))[0]), params.rho_cap)
        )
        cube = torch.empty(
            (len(zs),) + tuple(next(iter(raw[s].values())).shape),
            dtype=torch.float32,
            device=dev,
        )
        per: dict[str, Any] = dict(
            rho=rho,
            frac_gt10=[],
            frac_gt100=[],
            removed_sum=[],
            removed_p99=[],
            max_brightening=0.0,
        )
        # As on the CPU, the fill acts on the stains it is configured for, and only
        # where the model has fits to take light from.
        filling = (
            fill is not None
            and fill.enabled
            and s in fill.stains
            and bool(model.fits)
            and rho > 0
        )
        fill_total: dict[str, float] = {}
        fill_planes: dict[int, dict[str, float]] = {}
        for i_z, k in enumerate(zs):
            img = raw[s][k]
            fit = model.fits.get(k)
            arr = apply_bead_model(
                img,
                cast(torch.Tensor, model.B[k])
                if (is_ref and fit is not None)
                else None,
                cast(torch.Tensor, fit.model) if fit is not None else None,
                rho,
            )
            if filling and fit is not None and fill is not None:
                arr, counters = fill_plane(
                    arr,
                    torch.clamp(rho * cast(torch.Tensor, fit.model), min=0.0),
                    img,
                    plane=k,
                    params=fill,
                    noise=model.noise,
                    origin_yx=origin_al,
                )
                if counters is not None:
                    fill_planes[k] = counters
                    for name, count in counters.items():
                        fill_total[name] = fill_total.get(name, 0.0) + count
            cube[i_z] = arr
            del arr
            rc = img[pad_tl[0] + y0 : pad_tl[0] + y1, pad_tl[1] + x0 : pad_tl[1] + x1]
            cc = cube[
                i_z, pad_tl[0] + y0 : pad_tl[0] + y1, pad_tl[1] + x0 : pad_tl[1] + x1
            ]
            d = rc - cc
            ad = d.abs()
            per["frac_gt10"].append(float((ad > 10).float().mean()))
            per["frac_gt100"].append(float((ad > 100).float().mean()))
            per["removed_sum"].append(float(d.sum(dtype=torch.float64)))
            per["removed_p99"].append(_pct(d, 99))
            per["max_brightening"] = max(
                per["max_brightening"],
                float(max(0.0, float((-d).max()))) if d.numel() else 0.0,
            )
            if is_ref:
                stats["n_saturated_px"] = stats.get("n_saturated_px_acc", 0) + int(
                    (rc >= SATURATION_LEVEL).sum()
                )
                stats["n_saturated_px_acc"] = stats["n_saturated_px"]
        stats["per_stain"][s] = per
        if filling:
            stats.setdefault("fill", {})[s] = dict(
                fill_total, **({"per_plane": fill_planes} if fill_planes else {})
            )
        if padded:
            cube = cube[:, sl[0], sl[1]]
        if output == "maxproj":
            out[s] = cube.max(0).values.cpu().numpy()
        elif output == "device":
            out[s] = cube
        else:
            out[s] = cube.cpu().numpy()
    stats.pop("n_saturated_px_acc", None)
    t = tick("apply+stats+d2h", t)
    stats["z_layers"] = zs
    stats["k_fit"] = k_fit
    stats["time_fit_s"] = t_fit
    stats["time_total_s"] = time.perf_counter() - t0
    return out, model, stats
