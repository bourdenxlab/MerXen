"""Tests of the GPU port of the hybrid bead-removal fill (synthetic data only).

The numpy fill (:mod:`merxen.io.bead_removal.fill`, hybrid mode) is the
specification. These tests compare the device implementation with it:

* P1 on identical inputs (a textured plane with bead rings, several window origins)
  the two agree to float32 rounding and report the same counters, for the default
  hybrid fill (eps 1.0, dilation 3, no texture gate) and for the opt-in texture
  gate with the earlier bound and dilation;
* P2 the bounds hold on the device: never brighter, removal at most ``eps x light``
  (max filtered over the dilation), invalid pixels unchanged;
* W1 end to end (``correct_window`` on both backends, both settings): at most
  0.5 % of the pixels differ by more than 10 counts, the removed light agrees to
  1 %, the result is never brighter than the subtraction, and removal stays within
  ``eps x model``;
* C1 chunk invariance: the plane filled in two windows with different origins
  agrees beyond the support margin;
* D1 two runs are bitwise identical;
* G1 the driver on ``cuda`` with the hybrid fill agrees with the driver on ``cpu``
  and reports the fill (slow);
* E1 refusals: the flat mode, an origin off the 8 px grid, a missing noise value.

The ``gpu`` tests are opt-in (``flock <gpu lock file> pytest -m gpu``) and skip
themselves without CUDA.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy import ndimage as ndi

from merxen.io.bead_removal import correct_window as cpu_correct_window
from merxen.io.bead_removal.fill import FillParams, apply_bounded_fill, support_px

from . import bead_synth as bs
from .bead_build_fixtures import open_projection_store, real_calibration, write_mosaic
from .test_bead_fill import NOISE, TEXTURE_HYBRID, Z, make_textured_planes

needs_gpu = pytest.mark.gpu

HYBRID = FillParams(enabled=True, mode="hybrid")
# The default hybrid fill and the earlier one (texture gate, 0.3 x light, dilation 2).
SETTINGS = pytest.mark.parametrize(
    "params", [HYBRID, TEXTURE_HYBRID], ids=["default", "texture"]
)
STAINS = ("PolyT", "DAPI")
# Beads of a 704 x 1024 window with texture (the window of the numpy chunk
# invariance test of the hybrid mode).
BEADS = (
    (200, 250, 30.0, 6.0, 2500.0),
    (480, 560, 30.0, 6.0, 1800.0),
    (300, 880, 30.0, 6.0, 3000.0),
)
SIZE = (704, 1024)


@pytest.fixture(scope="module")
def cuda() -> None:
    """Skip the test unless torch can use a CUDA device (imports torch)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture(scope="module")
def gpu(cuda: None) -> ModuleType:
    """The GPU backend, loaded the way the pipeline loads it."""
    from merxen.io.bead_removal import backend

    module = backend.load_gpu_backend()
    module.ops.configure(True)
    return module


@pytest.fixture(scope="module")
def fill_module(gpu: ModuleType) -> ModuleType:
    """The device fill module."""
    import importlib

    return importlib.import_module("merxen.io.bead_removal.gpu.fill")


@pytest.fixture(scope="module")
def planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Textured planes with bead ring residuals; a strip of the window is invalid."""
    corr, light, raw = make_textured_planes(size=SIZE, seed=3, beads=BEADS)
    corr[:, :, :40] = 0
    raw[:, :, :40] = 0
    return corr, light, raw


def _device_fill(
    fill_module: ModuleType,
    corr: np.ndarray,
    light: np.ndarray,
    raw: np.ndarray,
    origin_yx: tuple[int, int],
    params: FillParams = HYBRID,
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    """Fill every plane of ``corr`` on the device."""
    import torch

    dev = torch.device("cuda")
    out = np.empty_like(corr)
    counters: dict[int, dict[str, float]] = {}
    for i, k in enumerate(Z):
        plane, ctr = fill_module.fill_plane(
            torch.from_numpy(corr[i]).to(dev),
            torch.from_numpy(light[i]).to(dev),
            torch.from_numpy(raw[i]).to(dev),
            plane=k,
            params=params,
            noise=NOISE,
            origin_yx=origin_yx,
        )
        out[i] = plane.cpu().numpy()
        assert ctr is not None
        counters[k] = ctr
    return out, counters


def _numpy_fill(
    corr: np.ndarray,
    light: np.ndarray,
    raw: np.ndarray,
    origin_yx: tuple[int, int],
    params: FillParams = HYBRID,
) -> tuple[np.ndarray, dict[str, Any]]:
    stats: dict[str, Any] = {}
    out = apply_bounded_fill(
        corr,
        light,
        raw=raw,
        z_layers=Z,
        config=params,
        stain="PolyT",
        noise=NOISE,
        stats=stats,
        origin_yx=origin_yx,
    )
    return out, stats


# ---------------------------------------------------------------------------
# P1, P2: the plane fill against the numpy fill
@needs_gpu
@SETTINGS
@pytest.mark.parametrize("origin_yx", [(0, 0), (16, 48), (4112, 80)])
def test_p1_device_fill_matches_the_numpy_fill_on_identical_inputs(
    fill_module: ModuleType,
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    origin_yx: tuple[int, int],
    params: FillParams,
) -> None:
    corr, light, raw = planes
    ref, ref_stats = _numpy_fill(corr, light, raw, origin_yx, params)
    out, counters = _device_fill(fill_module, corr, light, raw, origin_yx, params)

    assert (ref < corr).sum() > 1000  # a real test: the fill acts
    assert np.abs(out - ref).max() <= 0.05
    assert int((out < corr).sum()) == int((ref < corr).sum())
    # The texture counter exists with the texture gate only, on both backends.
    assert ("texture_dropped_px" in ref_stats) == params.texture_gate
    for k in Z:
        assert ("texture_dropped_px" in counters[k]) == params.texture_gate
    keys = ["cand_px", "changed_px"]
    if params.texture_gate:
        keys.append("texture_dropped_px")
    for key in keys:
        total = sum(c[key] for c in counters.values())
        assert total == ref_stats[key], key
        for k in Z:
            assert counters[k][key] == ref_stats["per_plane"][k][key], (k, key)
    removed = sum(c["removed_sum"] for c in counters.values())
    assert removed == pytest.approx(ref_stats["removed_sum"], rel=1e-4)
    # The numpy fill sets the unfilled counter only when it fills a crop.
    assert sum(c.get("unfilled_px", 0.0) for c in counters.values()) == ref_stats.get(
        "unfilled_px", 0.0
    )


@needs_gpu
def test_p2_device_fill_is_one_sided_and_bounded_and_leaves_invalid_pixels(
    fill_module: ModuleType, planes: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    corr, light, raw = planes
    out, _ = _device_fill(fill_module, corr, light, raw, (16, 48))

    assert out.dtype == np.float32
    assert (out <= corr).all()
    assert np.array_equal(out[raw == 0], corr[raw == 0])
    for i, k in enumerate(Z):
        udil = ndi.grey_dilation(HYBRID.eps[k] * light[i], size=2 * HYBRID.dilation + 1)
        assert (out[i] >= corr[i] - udil - 1e-3).all()


# ---------------------------------------------------------------------------
# W1: correct_window on both backends
@pytest.fixture(scope="module")
def cal() -> Any:
    return real_calibration()


@pytest.fixture(scope="module")
def synth() -> dict[str, np.ndarray]:
    """768 px window: 45 beads, tissue blobs, nuclei and an antibody stain."""
    return bs.make_stack(size=768, n_beads=45, seed=11, n_nuclei=8)


def _stacks(st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: st[k] for k in ("PolyT", "DAPI", "Anti-X")}


@needs_gpu
@SETTINGS
@pytest.mark.parametrize("origin_yx", [(0, 0), (16, 48)])
def test_w1_gpu_hybrid_window_agrees_with_the_cpu_hybrid_window(
    gpu: ModuleType,
    synth: dict[str, np.ndarray],
    cal: Any,
    origin_yx: tuple[int, int],
    params: FillParams,
) -> None:
    stacks = _stacks(synth)
    cpu, _, _ = cpu_correct_window(
        stacks, origin_yx=origin_yx, calibration=cal, z_layers=Z, fill=params
    )
    out, model, stats = gpu.correct_window(
        stacks, origin_yx=origin_yx, calibration=cal, z_layers=Z, fill=params
    )
    base, _, _ = gpu.correct_window(
        stacks, origin_yx=origin_yx, calibration=cal, z_layers=Z
    )

    raw = np.stack([synth["PolyT"][k] for k in Z]).astype(np.float64)
    diff = np.abs(out["PolyT"].astype(np.float64) - cpu["PolyT"])
    assert (diff > 10).mean() <= 0.005, f"{(diff > 10).mean():.4%} differ by > 10"
    ratio = (raw - out["PolyT"]).sum() / (raw - cpu["PolyT"]).sum()
    assert abs(ratio - 1.0) <= 0.01, f"removed-light ratio GPU/CPU {ratio:.4f}"
    # The fill acts, never brightens, and removes at most eps x model light.
    assert (out["PolyT"] < base["PolyT"]).sum() > 1000
    assert (out["PolyT"] <= base["PolyT"]).all()
    assert (out["PolyT"] <= raw).all()
    for i, k in enumerate(Z):
        light = np.maximum(model.fits[k].model.cpu().numpy(), 0.0)
        udil = ndi.grey_dilation(params.eps[k] * light, size=2 * params.dilation + 1)
        removed = base["PolyT"][i].astype(np.float64) - out["PolyT"][i]
        assert (removed <= udil + 1e-2).all(), k
    # DAPI is not a fill stain: it is untouched by the fill.
    assert np.array_equal(out["DAPI"], base["DAPI"])
    counters = stats["fill"]["PolyT"]
    assert {"cand_px", "changed_px", "removed_sum"} <= set(counters)
    assert ("texture_dropped_px" in counters) == params.texture_gate
    assert set(counters["per_plane"]) == set(Z)
    assert "DAPI" not in stats["fill"]


@needs_gpu
def test_w1_a_disabled_fill_is_bit_identical_to_no_fill(
    gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any
) -> None:
    stacks = _stacks(synth)
    base, _, _ = gpu.correct_window(
        stacks, origin_yx=(0, 0), calibration=cal, z_layers=Z
    )
    off, _, stats = gpu.correct_window(
        stacks,
        origin_yx=(0, 0),
        calibration=cal,
        z_layers=Z,
        fill=FillParams(enabled=False, mode="hybrid"),
    )

    for s in STAINS:
        assert np.array_equal(off[s], base[s])
    assert "fill" not in stats


@needs_gpu
def test_w1_maxproj_output_equals_the_projection_of_the_filled_planes(
    gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any
) -> None:
    stacks = _stacks(synth)
    planes_out, _, _ = gpu.correct_window(
        stacks, origin_yx=(0, 0), calibration=cal, z_layers=Z, fill=HYBRID
    )
    proj, _, _ = gpu.correct_window(
        stacks,
        origin_yx=(0, 0),
        calibration=cal,
        z_layers=Z,
        fill=HYBRID,
        output="maxproj",
    )

    for s in STAINS:
        assert np.array_equal(proj[s], planes_out[s].max(0))


# ---------------------------------------------------------------------------
# C1: chunk invariance
@needs_gpu
@SETTINGS
def test_c1_device_fill_is_chunk_invariant_beyond_the_support_margin(
    fill_module: ModuleType,
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    params: FillParams,
) -> None:
    """Windows whose origin is on the 16 px grid but not on the 32 px texture grid."""
    corr, light, raw = planes
    full, _ = _device_fill(fill_module, corr, light, raw, (0, 0), params)
    assert (full < corr).sum() > 1000
    margin = support_px(params)["total"] + 2
    for y0, y1, x0, x1 in [(16, 704, 48, 1024), (32, 688, 16, 976), (0, 704, 80, 1008)]:
        sub, _ = _device_fill(
            fill_module,
            np.ascontiguousarray(corr[:, y0:y1, x0:x1]),
            np.ascontiguousarray(light[:, y0:y1, x0:x1]),
            np.ascontiguousarray(raw[:, y0:y1, x0:x1]),
            (y0, x0),
            params,
        )
        ya, yb = (margin if y0 > 0 else 0), (y1 - y0 - (margin if y1 < SIZE[0] else 0))
        xa, xb = (margin if x0 > 0 else 0), (x1 - x0 - (margin if x1 < SIZE[1] else 0))
        a = sub[:, ya:yb, xa:xb]
        b = full[:, y0 + ya : y0 + yb, x0 + xa : x0 + xb]
        # Different window shapes may pick different convolution algorithms: allow
        # float32 rounding, nothing of the size of a different decision.
        assert np.abs(a - b).max() <= 0.05, (y0, y1, x0, x1)
    if not params.texture_gate:
        return
    # The origin is what aligns the texture blocks: a wrong origin changes the result.
    window = (slice(None), slice(16, None), slice(48, None))
    args = [np.ascontiguousarray(a[window]) for a in (corr, light, raw)]
    wrong, _ = _device_fill(fill_module, *args, (0, 0), params)
    right, _ = _device_fill(fill_module, *args, (16, 48), params)
    assert not np.array_equal(wrong, right)


@needs_gpu
def test_c1_gpu_hybrid_correct_window_is_chunk_invariant_inside_the_halo(
    gpu: ModuleType, cal: Any
) -> None:
    """Two windows of one mosaic, cut at different origins, agree in their overlap."""
    st = bs.make_stack(size=(1152, 1152), n_beads=70, seed=5, n_nuclei=8)
    stacks = _stacks(st)
    out_a, _, _ = gpu.correct_window(
        stacks, origin_yx=(0, 0), calibration=cal, z_layers=Z, fill=HYBRID
    )
    y0, x0 = 16, 48
    sub = {k: np.ascontiguousarray(v[:, y0:, x0:]) for k, v in stacks.items()}
    out_b, _, _ = gpu.correct_window(
        sub, origin_yx=(y0, x0), calibration=cal, z_layers=Z, fill=HYBRID
    )
    # Away from the cut edges by the halo (256 px): the fit and the fill read only
    # pixels inside both windows there.
    lo, hi = 256 + 16 + 48, 1152 - 256
    a = out_a["PolyT"][:, lo:hi, lo:hi]
    b = out_b["PolyT"][:, lo - y0 : hi - y0, lo - x0 : hi - x0]
    diff = np.abs(a.astype(np.float64) - b)
    assert (diff > 10).mean() <= 0.001, f"{(diff > 10).mean():.4%} differ by > 10"


# ---------------------------------------------------------------------------
# D1: determinism
@needs_gpu
def test_d1_two_runs_are_bitwise_identical(
    gpu: ModuleType,
    fill_module: ModuleType,
    synth: dict[str, np.ndarray],
    cal: Any,
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    corr, light, raw = planes
    first, _ = _device_fill(fill_module, corr, light, raw, (16, 48))
    second, _ = _device_fill(fill_module, corr, light, raw, (16, 48))
    assert np.array_equal(first, second)

    stacks = _stacks(synth)
    runs = [
        gpu.correct_window(
            stacks, origin_yx=(16, 48), calibration=cal, z_layers=Z, fill=HYBRID
        )[0]
        for _ in range(2)
    ]
    for s in STAINS:
        assert np.array_equal(runs[0][s], runs[1][s])


# ---------------------------------------------------------------------------
# E1: refusals and plane selection
@needs_gpu
def test_e1_the_device_fill_refuses_what_it_does_not_implement(
    gpu: ModuleType,
    fill_module: ModuleType,
    synth: dict[str, np.ndarray],
    cal: Any,
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    import torch

    corr, light, raw = (torch.from_numpy(a[0]).to("cuda") for a in planes)
    common: dict[str, Any] = dict(plane=1, noise=NOISE, origin_yx=(0, 0))
    with pytest.raises(ValueError, match="hybrid mode only"):
        fill_module.fill_plane(
            corr, light, raw, params=FillParams(enabled=True), **common
        )
    with pytest.raises(ValueError, match="multiple of 8"):
        fill_module.fill_plane(
            corr, light, raw, params=HYBRID, **{**common, "origin_yx": (4, 0)}
        )
    with pytest.raises(ValueError, match="global noise of plane 1"):
        fill_module.fill_plane(
            corr, light, raw, params=HYBRID, **{**common, "noise": {2: 37.0}}
        )
    with pytest.raises(ValueError, match="no GPU implementation"):
        gpu.correct_window(
            _stacks(synth),
            origin_yx=(0, 0),
            calibration=cal,
            z_layers=Z,
            fill=FillParams(enabled=True),
        )


@needs_gpu
def test_e1_a_plane_with_eps_zero_or_outside_planes_is_skipped(
    fill_module: ModuleType, planes: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    import dataclasses

    import torch

    corr, light, raw = (torch.from_numpy(a[0]).to("cuda") for a in planes)
    common: dict[str, Any] = dict(plane=1, noise=NOISE, origin_yx=(0, 0))
    eps = list(HYBRID.eps)
    eps[1] = 0.0
    skipped = dataclasses.replace(HYBRID, eps=tuple(eps))
    out, ctr = fill_module.fill_plane(corr, light, raw, params=skipped, **common)
    assert ctr is None and out is corr
    only_two = dataclasses.replace(HYBRID, planes=(2,))
    out, ctr = fill_module.fill_plane(corr, light, raw, params=only_two, **common)
    assert ctr is None and out is corr


# ---------------------------------------------------------------------------
# G1: the driver, cuda against cpu, with the hybrid fill
@needs_gpu
@pytest.mark.slow
def test_g1_the_cuda_driver_with_the_hybrid_fill_agrees_with_the_cpu_driver(
    cuda: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.io.bead_removal import driver

    # The outer flock of `pytest -m gpu` holds the host lock: run unlocked here.
    monkeypatch.delenv("MERXEN_GPU_LOCK_FILE", raising=False)
    shape = (1536, 1536)
    st = bs.make_stack(n_planes=4, size=shape, n_beads=70, seed=5, n_nuclei=10)
    images = tmp_path / "images"
    write_mosaic(images, st, 4)
    calibration = real_calibration()
    cal_dir = tmp_path / "cal"
    calibration.save(cal_dir)

    def _run(device: str, fill: FillParams | None, label: str) -> tuple[Any, Any]:
        store = open_projection_store(
            tmp_path / f"{label}.zarr", n_channels=2, shape=shape, chunk_px=1024
        )
        run = driver.run_bead_removal(
            images_dir=images,
            stains=list(STAINS),
            z_layers=Z,
            store=store,
            calibration=calibration,
            calibration_dir=cal_dir,
            work_dir=tmp_path / f"w_{label}",
            fill=fill,
            core_px=1024,
            max_workers=2,
            device=device,
        )
        return np.asarray(store[:]), run

    gpu_store, gpu_run = _run("cuda", HYBRID, "gpu")
    again, _ = _run("cuda", HYBRID, "gpu_again")
    plain, plain_run = _run("cuda", None, "gpu_plain")
    cpu_store, _ = _run("cpu", HYBRID, "cpu")

    assert gpu_run.device == "cuda"
    # The chunk statistics carry the counters of the device fill, as on the CPU.
    assert sum(c["fill"]["PolyT"]["cand_px"] for c in gpu_run.chunk_stats) > 0
    assert all("fill" not in c for c in plain_run.chunk_stats)
    assert np.array_equal(gpu_store, again)
    # The fill acts on the reference stain only, and never brightens it.
    assert (gpu_store[0] < plain[0]).sum() > 1000
    assert (gpu_store[0] <= plain[0]).all()
    assert np.array_equal(gpu_store[1], plain[1])
    diff = np.abs(gpu_store.astype(np.int64) - cpu_store)
    assert (diff > 10).mean() <= 0.005, f"{(diff > 10).mean():.4%} differ by > 10"
