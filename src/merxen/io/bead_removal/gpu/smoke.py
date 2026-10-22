"""Triton JIT smoke test: a 64 px window through the fused fit kernels.

The backend selection of the pipeline runs this once per process before the first real
window, so a missing C compiler, a Triton/driver mismatch or a kernel that does not
compile fails at the start of a sample with a clear message instead of hours into a
build.  The test compiles the three fused kernels (stamp rendering, robust gain/offset
fit, position step) on a synthetic bead whose answer is known, and checks the numbers,
not only that the kernels run.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from merxen.io.bead_removal.psf import Template

logger = logging.getLogger(__name__)

#: Side of the synthetic window (px).
SMOKE_SIZE = 64
_TRUE_GAIN = 1.5
_BACKGROUND = 100.0


class GpuSmokeTestError(RuntimeError):
    """The CUDA device or the fused Triton kernels are unusable (message names the
    cause)."""


def _synthetic_template(r_sup: float, ds: float) -> Template:
    """Smooth radial profile (unit centre value) that is ~0 at ``r_sup``: stands in for
    a PSF library template."""
    n = int(np.ceil(r_sup / ds)) + 1
    nodes = np.arange(n) * ds
    return Template(ds, r_sup, np.exp(-(nodes**2) / (2.0 * 2.5**2)))


def triton_smoke_test() -> dict[str, Any]:
    """Run a 64 px window through the fused Triton kernels and check the fitted gain,
    offset and rendered model.

    A noise-free window holds one point source of known gain on a flat background.  The
    test (1) renders the stamp with the add kernel, (2) refits gain and offset with the
    fit kernel (expects the known gain and a zero offset), (3) runs the position-step
    kernel (expects a bounded finite step) and (4) renders the fitted gain and compares
    the model with the image.  All comparisons are to float32-rounding level.

    Returns:
        Record with ``torch``, ``triton``, ``cuda`` versions, the device name, the wall
        seconds (JIT compile included) and the measured errors.

    Raises:
        GpuSmokeTestError: No CUDA device, Triton not importable, a kernel fails to
            compile or launch (for example no C compiler), or a result is wrong.
    """
    t0 = time.perf_counter()
    if not torch.cuda.is_available():
        raise GpuSmokeTestError("torch reports no CUDA device")
    try:
        import triton

        from . import ops, tri
        from .fit import Frame
        from .stamps import TmplDev, make_geom, point_stamps
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GpuSmokeTestError(
            f"Triton or the GPU kernels cannot be imported ({exc})"
        ) from exc
    try:
        dev = ops.device()
        r_sup, win_extra, ds, s = 12.0, 4.0, 0.25, 1.0
        tmpl = _synthetic_template(r_sup, ds)
        tm = TmplDev.from_template(tmpl)
        hb = int(np.ceil(2 * (r_sup * s + win_extra))) + 3
        pad = hb + 2
        yc, xc, amp = 31.3, 30.6, 1000.0
        f64 = lambda v: torch.tensor([v], dtype=torch.float64, device=dev)  # noqa: E731
        # image = background + true gain x stamp, built with the eager (non-Triton)
        # stamp so the kernels are checked against an independent evaluation of the same
        # profile
        geom = make_geom(
            f64(yc),
            f64(xc),
            torch.full((1,), r_sup * s + win_extra, dtype=torch.float64, device=dev),
            hb,
            hb,
            pad,
            SMOKE_SIZE + 2 * pad,
        )
        stamp = point_stamps(tm, f64(amp), f64(s), f64(yc), f64(xc), geom)
        truth = torch.zeros(
            (SMOKE_SIZE + 2 * pad) ** 2, dtype=torch.float32, device=dev
        )
        truth.index_add_(0, geom.flat.reshape(-1), (_TRUE_GAIN * stamp).reshape(-1))
        truth = truth.view(SMOKE_SIZE + 2 * pad, -1)[
            pad : pad + SMOKE_SIZE, pad : pad + SMOKE_SIZE
        ].contiguous()
        img = truth + _BACKGROUND
        frame = Frame(img, torch.full_like(img, _BACKGROUND), pad)
        tab, n_tab = tm.vals.contiguous(), tm.n
        args = (tab, ds, n_tab, r_sup, win_extra, hb)
        sig_eff = 2.0
        one = f64(1.0)
        zero = f64(0.0)
        a, sv, y, x = f64(amp), f64(s), f64(yc), f64(xc)
        tri.add_stamps(frame, *args, y, x, a, sv, one)  # model = 1 x stamp
        gi, ci = tri.fit_batch(
            frame,
            *args,
            y,
            x,
            a,
            sv,
            one,
            zero,
            one,
            f64(1e-6),
            f64(6.0),
            sig_eff,
            0.04,
            1.5,
        )
        dy, dx = tri.position_step(
            frame, *args, y, x, a, sv, one, gi, ci, sig_eff, 0.04, 1.5
        )
        tri.add_stamps(frame, *args, y, x, a, sv, gi - one)  # model = gi x stamp
        torch.cuda.synchronize()
        model = frame.crop_model()
        gain, off = float(gi[0]), float(ci[0])
        step = (float(dy[0]), float(dx[0]))
        scale = float(truth.max())
        err = float((model - truth).abs().max()) / scale
    except Exception as exc:  # compile / launch failures are reported with their cause
        raise GpuSmokeTestError(
            f"the Triton kernels failed on a {SMOKE_SIZE} px window: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    problems = []
    if not np.isfinite([gain, off, *step, err]).all():
        problems.append("non-finite result")
    if abs(gain - _TRUE_GAIN) > 0.01 * _TRUE_GAIN:
        problems.append(f"fitted gain {gain:.4f} != {_TRUE_GAIN}")
    if abs(off) > 0.5:
        problems.append(f"fitted offset {off:.3f} counts != 0")
    if max(abs(step[0]), abs(step[1])) > 0.6 + 1e-9:
        problems.append(f"position step {step} outside the bound")
    if err > 1e-3:
        problems.append(
            f"rendered model differs from the eager stamp by {err:.2e} of the peak"
        )
    if problems:
        raise GpuSmokeTestError(
            "Triton smoke test gave wrong numbers: " + "; ".join(problems)
        )
    rec = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0),
        seconds=time.perf_counter() - t0,
        gain=gain,
        offset=off,
        position_step=step,
        model_rel_err=err,
    )
    logger.info(
        "[BEADS-GPU] Triton smoke test passed in %.1f s (torch %s, triton %s)",
        rec["seconds"],
        rec["torch"],
        rec["triton"],
    )
    return rec
