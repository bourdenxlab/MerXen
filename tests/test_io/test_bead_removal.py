"""Tests of the window-level bead removal (``correct_window``).

Synthetic data only. T1 shapes and refusals, T2 never brighten, T3 removal,
T4 locality, T5 tissue survives, T6 other stains untouched, T7 no beads,
T8 determinism, T9 chunk independence, T10 bead on a core boundary, T11 DAPI
response; plus the absolute-grid invariance, the padding of unaligned origins,
the fill hook and the handling of invalid pixels.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from merxen.io.bead_removal import correct_window
from merxen.io.bead_removal.apply import (
    apply_bead_model,
)
from merxen.io.bead_removal.apply import (
    correct_window as correct_window_direct,
)
from merxen.io.bead_removal.params import DEFAULT_PARAMS

from . import bead_synth as bs

Z = [1, 2, 3]
Result = tuple[dict[str, np.ndarray], Any, dict[str, Any]]


def _run(
    st: dict[str, np.ndarray],
    cal: bs.SyntheticCalibration,
    z: list[int] | None = None,
    origin: tuple[int, int] = (0, 0),
    **kwargs: Any,
) -> Result:
    stacks = {k: st[k] for k in ("PolyT", "DAPI", "Anti-X") if k in st}
    return correct_window(
        stacks,
        origin_yx=origin,
        calibration=cal,
        z_layers=Z if z is None else z,
        **kwargs,
    )


def _cropped_window(
    st: dict[str, np.ndarray], core0: int, core_rows: int, halo: int
) -> dict[str, np.ndarray]:
    """Cut the rows around a core as a window, with zeros outside the mosaic."""
    n_rows = st["PolyT"].shape[1]
    width = st["PolyT"].shape[2]
    arrs = {}
    for stain in ("PolyT", "DAPI"):
        window = np.zeros((8, core_rows + 2 * halo, width), np.float32)
        y0 = core0 - halo
        ya, yb = max(0, y0), min(n_rows, y0 + core_rows + 2 * halo)
        window[:, ya - y0 : yb - y0] = st[stain][:, ya:yb]
        arrs[stain] = window
    return arrs


@pytest.fixture(scope="module")
def cal() -> bs.SyntheticCalibration:
    return bs.make_calibration()


@pytest.fixture(scope="module")
def base(
    cal: bs.SyntheticCalibration,
) -> tuple[dict[str, np.ndarray], Result]:
    st = bs.make_stack(seed=1, n_beads=35, size=768)
    return st, _run(st, cal)


def test_the_package_exports_the_window_entry_points() -> None:
    assert correct_window is correct_window_direct


def test_t1_shapes_and_refusals(
    base: tuple[dict[str, np.ndarray], Result], cal: bs.SyntheticCalibration
) -> None:
    st, (out, _model, stats) = base
    # Only the corrected stains come back; the antibody stain is not returned.
    assert set(out) == {"PolyT", "DAPI"}
    for arr in out.values():
        assert arr.dtype == np.float32
        assert arr.shape == (len(Z), *st["PolyT"].shape[1:])
    assert stats["status"] == "ok"
    assert stats["n_beads_final"] > 25
    assert stats["z_layers"] == Z
    assert stats["k_fit"] == 3
    for bad in ([0, 1, 2], [0], [1, 8], []):
        with pytest.raises(ValueError):
            _run(st, cal, z=bad)
    with pytest.raises(ValueError, match="DAPI"):
        correct_window(
            {"PolyT": st["PolyT"]},
            origin_yx=(0, 0),
            calibration=cal,
            z_layers=Z,
        )


def test_z_layer_errors_say_what_to_do(cal: bs.SyntheticCalibration) -> None:
    st = bs.make_stack(seed=1, n_beads=5, size=256, tissue=False)
    with pytest.raises(ValueError, match="plane 1 or later"):
        _run(st, cal, z=[0, 1])
    with pytest.raises(ValueError, match="up to 7"):
        _run(st, cal, z=[1, 8])
    with pytest.raises(ValueError, match="empty"):
        _run(st, cal, z=[])


def test_a_missing_or_malformed_plane_is_refused(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=1, n_beads=5, size=256, tissue=False)
    missing = {
        "PolyT": {k: st["PolyT"][k] for k in (0, 1, 2)},
        "DAPI": {k: st["DAPI"][k] for k in (0, 1, 3)},
    }
    with pytest.raises(ValueError, match="no plane 3"):
        correct_window(missing, origin_yx=(0, 0), calibration=cal, z_layers=[3])
    flat = {"PolyT": st["PolyT"], "DAPI": {k: st["DAPI"][k][0] for k in range(4)}}
    with pytest.raises(ValueError, match="not 2-D"):
        correct_window(flat, origin_yx=(0, 0), calibration=cal, z_layers=[3])
    odd = {
        "PolyT": st["PolyT"],
        "DAPI": {k: st["DAPI"][k][:200, :200] for k in range(4)},
    }
    with pytest.raises(ValueError, match="expected"):
        correct_window(odd, origin_yx=(0, 0), calibration=cal, z_layers=[3])


def test_a_mapping_of_planes_gives_the_same_result_as_the_array(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=16, n_beads=15, size=512)
    arrays, _, _ = _run(st, cal, z=[3])
    mapping = {
        "PolyT": {k: st["PolyT"][k] for k in (0, 1, 2, 3)},
        "DAPI": {k: st["DAPI"][k] for k in (0, 1, 3)},
    }
    mapped, _, _ = correct_window(
        mapping, origin_yx=(0, 0), calibration=cal, z_layers=[3]
    )
    for stain in arrays:
        assert np.array_equal(arrays[stain], mapped[stain])


def test_t2_never_brightens(base: tuple[dict[str, np.ndarray], Result]) -> None:
    st, (out, _model, stats) = base
    for stain in ("PolyT", "DAPI"):
        for i, k in enumerate(Z):
            assert (out[stain][i] <= st[stain][k]).all(), (stain, k)
        assert stats["per_stain"][stain]["max_brightening"] == 0.0


def test_t3_removal(base: tuple[dict[str, np.ndarray], Result]) -> None:
    st, (out, _model, _stats) = base
    # z1 and z2 carry most of the bead light; z3 is dominated by the broad flat
    # disc that the lower-envelope floor and the overlapping pedestal of
    # neighbouring discs keep (a documented limit), so its bound is looser.
    bounds = {1: 0.15, 2: 0.35, 3: 0.55}
    for i, k in enumerate(Z):
        injected = np.sqrt(np.mean(st["beadlight/PolyT"][k] ** 2))
        residual = np.sqrt(np.mean((out["PolyT"][i] - st["truth/PolyT"][k]) ** 2))
        assert residual <= bounds[k] * injected, (k, residual, injected)


def test_t4_locality(cal: bs.SyntheticCalibration) -> None:
    beads = np.array(
        [[200.0, 200.0, 4000.0], [512.0, 800.0, 3000.0], [850.0, 300.0, 5000.0]]
    )
    st = bs.make_stack(size=1024, beads=beads, seed=3, punctum_near_bead=False)
    out, _, _ = _run(st, cal)
    yy, xx = np.mgrid[:1024, :1024]
    far = np.ones((1024, 1024), bool)
    for by, bx, _amp in st["beads"]:
        far &= np.hypot(yy - by, xx - bx) > 230
    assert far.sum() > 100_000
    for i, k in enumerate(Z):
        assert np.array_equal(out["PolyT"][i][far], st["PolyT"][k][far])
        assert np.array_equal(out["DAPI"][i][far], st["DAPI"][k][far])


def test_t5_tissue_survives(base: tuple[dict[str, np.ndarray], Result]) -> None:
    st, (out, _model, _stats) = base
    yy, xx = np.mgrid[:768, :768]

    def contrast(img: np.ndarray, cy: float, cx: float) -> float:
        r = np.hypot(yy - cy, xx - cx)
        return float(img[r <= 4].mean() - img[(r >= 30) & (r <= 40)].mean())

    for by, bx in st["blobs"]:
        for i, k in enumerate(Z):
            kept = contrast(out["PolyT"][i], by, bx)
            assert kept >= 0.95 * contrast(st["PolyT"][k], by, bx)

    # A punctum sharpest at z3, 60 px from a bead: its DoG contrast at z3 is
    # compared with the tissue without beads.
    py, px = st["puncta"][0]

    def dog(a: np.ndarray) -> np.ndarray:
        return gaussian_filter(a, 1.0) - gaussian_filter(a, 3.0)

    y, x = round(py), round(px)
    win = (slice(y - 3, y + 4), slice(x - 3, x + 4))
    c_out = dog(out["PolyT"][2])[win].max()  # z3 is index 2 of Z
    c_truth = dog(st["truth/PolyT"][3])[win].max()
    assert c_out >= 0.8 * c_truth, (c_out, c_truth)


def test_t6_other_stains_are_not_returned_and_inputs_not_modified(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=4, n_beads=15, size=512)
    before = {k: st[k].copy() for k in ("PolyT", "DAPI", "Anti-X")}
    out, _, _ = _run(st, cal)
    assert "Anti-X" not in out
    for k, arr in before.items():
        assert np.array_equal(st[k], arr)


def test_t7_no_beads_is_the_identity(cal: bs.SyntheticCalibration) -> None:
    st = bs.make_stack(seed=5, n_beads=0, size=512, punctum_near_bead=False)
    out, model, stats = _run(st, cal)
    assert stats["status"] == "no beads"
    assert model.fits == {}
    assert stats["n_bead"] == 0
    for stain in ("PolyT", "DAPI"):
        for i, k in enumerate(Z):
            assert np.array_equal(out[stain][i], st[stain][k])


def test_t7b_all_invalid_window(cal: bs.SyntheticCalibration) -> None:
    st = bs.make_stack(seed=5, n_beads=0, size=512)
    zero = {k: np.zeros_like(st[k]) for k in ("PolyT", "DAPI")}
    out, _, stats = correct_window(zero, origin_yx=(0, 0), calibration=cal, z_layers=Z)
    assert stats["status"] == "no valid pixel"
    assert not out["PolyT"].any()


def test_t8_determinism(
    base: tuple[dict[str, np.ndarray], Result], cal: bs.SyntheticCalibration
) -> None:
    st, (out, _model, _stats) = base
    again, _, _ = _run(st, cal)
    for stain in out:
        assert np.array_equal(out[stain], again[stain])


@pytest.mark.slow
def test_t9_chunk_independence(cal: bs.SyntheticCalibration) -> None:
    st = bs.make_stack(size=(2048, 1024), n_beads=110, seed=6, tilt=0.12, margin=60)
    full, _, _ = _run(st, cal)
    halo = 256
    worst, frac2 = 0.0, 1.0
    for core0 in (0, 1024):
        out, _, _ = correct_window(
            _cropped_window(st, core0, 1024, halo),
            origin_yx=(core0 - halo, 0),
            calibration=cal,
            z_layers=Z,
            params=DEFAULT_PARAMS,
        )
        for i in range(len(Z)):
            a = out["PolyT"][i][halo : halo + 1024]
            b = full["PolyT"][i][core0 : core0 + 1024]
            d = np.abs(a - b)
            worst = max(worst, float(d.max()))
            frac2 = min(frac2, float((d <= 2).mean()))
    assert worst <= 20
    assert frac2 >= 0.99


def test_t10_bead_on_a_core_boundary(cal: bs.SyntheticCalibration) -> None:
    beads = np.array([[1020.0, 500.0, 6000.0], [1030.0, 520.0, 6000.0]])
    st = bs.make_stack(size=(2048, 640), seed=7, beads=beads, tissue=False)
    halo, ring_radius = 256, 7.0 + 22.8
    yy, xx = np.mgrid[:2048, :640]
    for core0 in (0, 1024):
        out, _, _ = correct_window(
            _cropped_window(st, core0, 1024, halo),
            origin_yx=(core0 - halo, 0),
            calibration=cal,
            z_layers=[1],
            core=(halo, halo + 1024, 0, 1024),
        )
        for by, bx, _amp in beads:
            if not (core0 <= by < core0 + 1024):
                continue
            r = np.hypot(yy - by, xx - bx)
            ring = (np.abs(r - ring_radius) <= 4) & (yy >= core0) & (yy < core0 + 1024)
            raw_excess = (st["PolyT"][1] - st["truth/PolyT"][1])[ring].mean()
            rows = yy[ring] - core0 + halo
            residual = (
                out["PolyT"][0][rows, xx[ring]] - st["truth/PolyT"][1][ring]
            ).mean()
            assert raw_excess > 20
            assert abs(residual) <= 0.25 * raw_excess, (core0, by, residual)


def test_t11_the_dapi_response_scales_the_subtraction(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=8, n_beads=20, size=512, tissue=False)
    # The beads carry a DAPI signal: subtracting rho x model brings DAPI towards
    # the truth, and rho = 0 leaves it untouched.
    out, _, _ = _run(st, cal)

    def rms(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    assert rms(out["DAPI"][0], st["truth/DAPI"][1]) < 0.5 * rms(
        st["DAPI"][1], st["truth/DAPI"][1]
    )
    out0, _, stats0 = _run(st, bs.make_calibration(rho_dapi=0.0))
    assert np.array_equal(out0["DAPI"][0], st["DAPI"][1])
    assert stats0["per_stain"]["DAPI"]["rho"] == 0.0
    # PolyT is still corrected.
    assert not np.array_equal(out0["PolyT"][0], st["PolyT"][1])


def test_the_rho_of_a_non_reference_stain_is_capped(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=8, n_beads=12, size=512, tissue=False)
    _, _, stats = _run(st, bs.make_calibration(rho_dapi=0.9))
    assert stats["per_stain"]["DAPI"]["rho"] == DEFAULT_PARAMS.rho_cap
    assert stats["per_stain"]["PolyT"]["rho"] == 1.0


def test_absolute_grid_invariance(cal: bs.SyntheticCalibration) -> None:
    """Shifting the window by multiples of the absolute grids changes nothing.

    The grids are the 16 px envelope blocks and the 512 px stretch blocks.
    """
    st = bs.make_stack(seed=9, n_beads=20, size=512)
    a, _, _ = _run(st, cal, origin=(0, 0))
    b, _, _ = _run(st, cal, origin=(1024, 512))
    assert np.array_equal(a["PolyT"], b["PolyT"])


def test_an_unaligned_origin_is_padded_and_cropped_back(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=9, n_beads=15, size=512)
    out, _, stats = _run(st, cal, origin=(5, 11))
    assert out["PolyT"].shape == (3, 512, 512)
    assert out["DAPI"].shape == (3, 512, 512)
    for i, k in enumerate(Z):
        assert (out["PolyT"][i] <= st["PolyT"][k]).all()
    assert (out["PolyT"][0] != st["PolyT"][1]).any()
    assert stats["per_stain"]["PolyT"]["max_brightening"] == 0.0


def test_statistics_are_restricted_to_the_core(cal: bs.SyntheticCalibration) -> None:
    st = bs.make_stack(seed=15, n_beads=20, size=512)
    _, _, whole = _run(st, cal)
    out, _, part = _run(st, cal, core=(100, 400, 50, 300))
    assert out["PolyT"].shape == (3, 512, 512)
    removed_whole = whole["per_stain"]["PolyT"]["removed_sum"]
    removed_part = part["per_stain"]["PolyT"]["removed_sum"]
    assert all(p <= w + 1e-3 for p, w in zip(removed_part, removed_whole, strict=True))
    assert removed_part != removed_whole


def test_invalid_pixels_stay_unchanged(cal: bs.SyntheticCalibration) -> None:
    st = bs.make_stack(seed=13, n_beads=20, size=512)
    stacks = {k: st[k].copy() for k in ("PolyT", "DAPI")}
    for arr in stacks.values():
        arr[:, :90, :] = 0  # outside the mosaic
        arr[:, 300:330, 200:400] = 0  # an interior gap
    out, _, _ = correct_window(stacks, origin_yx=(0, 0), calibration=cal, z_layers=Z)
    for stain in out:
        for i, k in enumerate(Z):
            invalid = stacks[stain][k] == 0
            assert (out[stain][i][invalid] == 0).all()
            assert (out[stain][i] <= stacks[stain][k]).all()
    assert (out["PolyT"][0] != stacks["PolyT"][1]).any()


def test_apply_bead_model_leaves_an_empty_model_and_zero_rho_alone(
    base: tuple[dict[str, np.ndarray], Result], cal: bs.SyntheticCalibration
) -> None:
    st, (_out, model, _stats) = base
    same = apply_bead_model(
        st["DAPI"], model, rho=0.0, stain_is_reference=False, planes=[1, 2]
    )
    assert np.array_equal(same, st["DAPI"][[1, 2]])
    empty = _run(bs.make_stack(seed=5, n_beads=0, size=256), cal)[1]
    unchanged = apply_bead_model(
        st["PolyT"], empty, rho=1.0, stain_is_reference=True, planes=[1]
    )
    assert np.array_equal(unchanged, st["PolyT"][[1]])
    removed = apply_bead_model(
        st["PolyT"], model, rho=1.0, stain_is_reference=True, planes=[1, 2]
    )
    assert removed.dtype == np.float32
    assert (removed <= st["PolyT"][[1, 2]]).all()
    assert st["PolyT"].dtype == np.float32


@pytest.fixture(scope="module")
def small(
    cal: bs.SyntheticCalibration,
) -> tuple[dict[str, np.ndarray], Result]:
    st = bs.make_stack(seed=1, n_beads=15, size=512)
    return st, _run(st, cal)


def _install_fake_fill(
    monkeypatch: pytest.MonkeyPatch,
    fake: Callable[..., np.ndarray],
) -> None:
    module = types.ModuleType("merxen.io.bead_removal.fill")
    module.apply_bounded_fill = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.bead_removal.fill", module)


def test_fill_hook(
    small: tuple[dict[str, np.ndarray], Result],
    cal: bs.SyntheticCalibration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st, (out, _model, _stats) = small
    calls: dict[str, tuple[Any, ...]] = {}

    def fake(
        corrected: np.ndarray,
        model_light: np.ndarray,
        *,
        raw: np.ndarray,
        z_layers: list[int],
        config: Any,
        stain: str,
        noise: dict[int, float] | None = None,
        stats: dict[str, Any],
    ) -> np.ndarray:
        calls[stain] = (corrected.shape, model_light.shape, raw.shape, list(z_layers))
        assert (model_light >= 0).all()
        stats["cand_px"] = 7
        return corrected - 5.0

    _install_fake_fill(monkeypatch, fake)
    config = types.SimpleNamespace(enabled=True, stains=("PolyT", "DAPI"))
    out_f, _, stats_f = _run(st, cal, fill=config)
    assert set(calls) == {"PolyT", "DAPI"}
    assert calls["PolyT"][3] == Z
    assert calls["PolyT"][0] == calls["PolyT"][1] == calls["PolyT"][2]
    # The fill's counters are kept per stain.
    assert stats_f["fill"] == {"PolyT": {"cand_px": 7}, "DAPI": {"cand_px": 7}}
    assert (out_f["PolyT"] <= out["PolyT"]).all()
    assert (out_f["PolyT"] < out["PolyT"]).any()
    # Pixels that are invalid in the raw plane stay unchanged.
    holes = {k: st[k].copy() for k in ("PolyT", "DAPI")}
    for arr in holes.values():
        arr[:, 300:340, 300:340] = 0
    out_h, _, _ = _run(holes, cal, fill=config)
    assert (out_h["PolyT"][:, 300:340, 300:340] == 0).all()


def test_the_fill_result_is_clamped_to_the_subtraction(
    small: tuple[dict[str, np.ndarray], Result],
    cal: bs.SyntheticCalibration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st, (out, _model, _stats) = small

    def brighten(corrected: np.ndarray, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return corrected + 100.0

    _install_fake_fill(monkeypatch, brighten)
    config = types.SimpleNamespace(enabled=True, stains=("PolyT",))
    out_f, _, _ = _run(st, cal, fill=config)
    assert np.array_equal(out_f["PolyT"], out["PolyT"])


def test_fill_inputs_are_built_only_for_enabled_stains(
    small: tuple[dict[str, np.ndarray], Result],
    cal: bs.SyntheticCalibration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled fill, or a stain outside ``fill.stains``, never calls the fill."""
    st, (out, _model, _stats) = small
    calls: list[str] = []

    def fake(corrected: np.ndarray, *_args: Any, stain: str, **_kwargs: Any) -> Any:
        calls.append(stain)
        return corrected

    _install_fake_fill(monkeypatch, fake)
    disabled = types.SimpleNamespace(enabled=False, stains=("PolyT", "DAPI"))
    out_d, _, stats_d = _run(st, cal, fill=disabled)
    assert calls == []
    assert "fill" not in stats_d
    for stain in out:
        assert np.array_equal(out_d[stain], out[stain])
    _run(st, cal, fill=types.SimpleNamespace(enabled=True, stains=("PolyT",)))
    assert calls == ["PolyT"]
