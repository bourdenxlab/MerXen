"""Tests of the window-independence of the bead removal.

Window-independent envelope at invalid pixels, the edge margin, saturated beads
and the parameter-hash refusal. Synthetic data only.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from merxen.io.bead_removal import correct_window
from merxen.io.bead_removal.params import DEFAULT_PARAMS

from . import bead_synth as bs

Z = [1, 2, 3]


@pytest.fixture(scope="module")
def cal() -> bs.SyntheticCalibration:
    return bs.make_calibration()


def _run(
    st: dict[str, np.ndarray],
    cal: bs.SyntheticCalibration,
    z: list[int] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
    stacks = {k: st[k] for k in ("PolyT", "DAPI") if k in st}
    return correct_window(
        stacks,
        origin_yx=(0, 0),
        calibration=cal,
        z_layers=Z if z is None else z,
        **kwargs,
    )


def test_one_invalid_corner_pixel_leaves_the_output(
    cal: bs.SyntheticCalibration,
) -> None:
    """One invalid corner pixel moves at most 0.05 % of the pixels by > 2."""
    st = bs.make_stack(seed=23, n_beads=25, size=512)
    out0, model0, _ = _run(st, cal)
    st2 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in st.items()}
    for stain in ("PolyT", "DAPI"):
        st2[stain][:, 0, 0] = 0.0
    out1, model1, _ = _run(st2, cal)
    for stain in ("PolyT", "DAPI"):
        diff = np.abs(np.rint(out0[stain].max(0)) - np.rint(out1[stain].max(0)))
        assert (diff > 2).mean() <= 5e-4, (stain, float((diff > 2).mean()))
    assert model0.info["n_bead"] == model1.info["n_bead"]


@pytest.mark.slow
def test_windows_cut_around_an_invalid_border_and_a_hole_agree(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(size=(2048, 1024), n_beads=110, seed=6, tilt=0.12, margin=60)
    for stain in ("PolyT", "DAPI"):
        st[stain][:, :, :140] = 0  # left edge outside the mosaic
        st[stain][:, 900:960, 500:640] = 0  # an interior gap
    full, _, _ = _run(st, cal)
    halo = 256
    worst, frac10 = 0.0, 0.0
    for core0 in (0, 1024):
        arrs = {}
        for stain in ("PolyT", "DAPI"):
            window = np.zeros((8, 1024 + 2 * halo, 1024), np.float32)
            y0 = core0 - halo
            ya, yb = max(0, y0), min(2048, y0 + 1024 + 2 * halo)
            window[:, ya - y0 : yb - y0] = st[stain][:, ya:yb]
            arrs[stain] = window
        out, _, _ = correct_window(
            arrs, origin_yx=(core0 - halo, 0), calibration=cal, z_layers=Z
        )
        for i, k in enumerate(Z):
            a = out["PolyT"][i][halo : halo + 1024]
            b = full["PolyT"][i][core0 : core0 + 1024]
            valid = st["PolyT"][k][core0 : core0 + 1024] > 0
            d = np.abs(a - b)[valid]
            worst = max(worst, float(d.max()))
            frac10 = max(frac10, float((d > 10).mean()))
    assert worst <= 20
    assert frac10 <= 5e-4


def test_saturated_beads_never_brighten_and_do_not_crash(
    cal: bs.SyntheticCalibration,
) -> None:
    """Beads clipped at 65535 (flat tops) are modelled without brightening a pixel."""
    beads = np.array(
        [[150.0, 150.0, 600000.0], [150.0, 400.0, 30000.0], [400.0, 250.0, 150000.0]]
    )
    st = bs.make_stack(size=512, seed=31, beads=beads, tissue=False)
    assert int((st["PolyT"][0] >= 65535).sum()) > 20  # the z0 peaks are clipped
    out, _, stats = _run(st, cal)
    assert stats["status"] == "ok"
    for stain in ("PolyT", "DAPI"):
        for i, k in enumerate(Z):
            assert np.isfinite(out[stain][i]).all()
            assert (out[stain][i] <= st[stain][k]).all(), (stain, k)
    assert stats["per_stain"]["PolyT"]["max_brightening"] <= 0.0


def test_saturated_pixels_of_the_output_planes_are_counted(
    cal: bs.SyntheticCalibration,
) -> None:
    """Reference pixels at or above 65,000 counts in the output planes are reported."""
    beads = np.array([[250.0, 250.0, 700000.0]])
    st = bs.make_stack(size=512, seed=32, beads=beads, tissue=False)
    expected = sum(int((st["PolyT"][k] >= 65000).sum()) for k in Z)
    assert expected > 0
    _, _, stats = _run(st, cal)
    assert stats["n_saturated_px"] == expected


def test_a_calibration_made_with_other_parameters_is_refused(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=41, n_beads=5, size=512)
    other = dataclasses.replace(DEFAULT_PARAMS, k_sigma=DEFAULT_PARAMS.k_sigma + 1.0)
    with pytest.raises(ValueError, match="different parameter set"):
        _run(st, cal, params=other)
    # The matching parameters run.
    _, _, stats = _run(st, cal, params=DEFAULT_PARAMS)
    assert stats["status"] in ("ok", "no beads")


def test_a_calibration_of_other_parameters_runs_with_those_parameters() -> None:
    st = bs.make_stack(seed=41, n_beads=5, size=512)
    other = dataclasses.replace(DEFAULT_PARAMS, k_sigma=DEFAULT_PARAMS.k_sigma + 1.0)
    cal_other = bs.make_calibration(params=other)
    _, _, stats = _run(st, cal_other, params=other)
    assert stats["status"] in ("ok", "no beads")
    with pytest.raises(ValueError, match="different parameter set"):
        _run(st, cal_other)
