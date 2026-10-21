"""Tests of the GPU backend of the bead removal (synthetic data only).

The numpy implementation is the specification; these tests compare the GPU
window correction with it:

* G1 ``seq`` mode (the CPU algorithm itself, brightness order) reproduces numpy
  to 0.05 counts;
* G2 colour mode (production: Triton, anchored colour grid) agrees with numpy
  statistically and never brightens;
* G3 bitwise determinism, in one process and across two processes;
* G6 the CUDA driver path writes identical stores whatever the number of reader
  threads (needs the CUDA consumer of the driver);
* the lower envelope with invalid pixels, the edge margin measured from invalid
  pixels only, the parameter-hash refusal, the removed Jacobi mode and the
  Triton smoke test.

The ``gpu`` tests are opt-in (``flock <gpu lock file> pytest -m gpu``) and skip
themselves without CUDA. Two static tests run everywhere: they keep ``torch``
and ``triton`` inside the ``gpu`` subpackage and the private CPU helpers out of
its imports.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import merxen.io.bead_removal as bead_removal_package
from merxen.io.bead_removal import correct_window as cpu_correct_window
from merxen.io.bead_removal.detect import detect_sources as cpu_detect_sources
from merxen.io.bead_removal.detect import lower_envelope as cpu_envelope
from merxen.io.bead_removal.model import beads_table
from merxen.io.bead_removal.params import DEFAULT_PARAMS

from . import bead_synth as bs
from .bead_build_fixtures import open_projection_store, real_calibration, write_mosaic

needs_gpu = pytest.mark.gpu

Z = [1, 2, 3]
STAINS = ("PolyT", "DAPI")
TESTS_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(bead_removal_package.__file__).resolve().parent

_WINDOW_SCRIPT = """
import sys

import numpy as np

from merxen.io.bead_removal import backend
from merxen.io.bead_removal.calibrate import BeadCalibration
from test_io import bead_synth as bs

gpu = backend.load_gpu_backend()
gpu.ops.configure(True)
st = bs.make_stack(size=768, n_beads=45, seed=11, n_nuclei=8)
stacks = {k: st[k] for k in ("PolyT", "DAPI", "Anti-X")}
cal = BeadCalibration.load(sys.argv[1])
out, _, _ = gpu.correct_window(
    stacks, origin_yx=(0, 0), calibration=cal, z_layers=[1, 2, 3]
)
np.savez(sys.argv[2], **out)
"""


# ---------------------------------------------------------------------------
# static guards (no GPU, no torch)
def _imported_modules(path: Path, *, module_level: bool = False) -> list[str]:
    """List the imports of a file (``module:name`` for ``from`` imports).

    Args:
        path: Python source file.
        module_level: Skip the imports inside function bodies.
    """
    names: list[str] = []

    class Collect(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            if not module_level:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            names.extend(alias.name for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            names.append(("." * node.level) + (node.module or ""))
            names.extend(f"{node.module}:{alias.name}" for alias in node.names)

    Collect().visit(ast.parse(path.read_text()))
    return names


def test_torch_and_triton_are_imported_only_inside_the_gpu_subpackage() -> None:
    outside = [
        p
        for p in PACKAGE_DIR.rglob("*.py")
        if "gpu" not in p.relative_to(PACKAGE_DIR).parts
    ]
    assert outside, "the CPU modules were not found"
    for path in outside:
        # Function-local imports (the backend probes torch when a GPU is requested)
        # are fine: importing the module must not load torch.
        for name in _imported_modules(path, module_level=True):
            root = name.split(":")[0].lstrip(".").split(".")[0]
            assert root not in {"torch", "triton"}, f"{path.name} imports {name}"
    gpu_files = sorted((PACKAGE_DIR / "gpu").glob("*.py"))
    assert {p.name for p in gpu_files} >= {
        "__init__.py",
        "detect.py",
        "fit.py",
        "model.py",
        "ops.py",
        "smoke.py",
        "stamps.py",
        "tri.py",
        "window.py",
    }
    # the GPU modules reach the envelope helpers of the CPU module through public
    # names (the fill and the constants), never through a private one
    ops_imports = _imported_modules(PACKAGE_DIR / "gpu" / "ops.py")
    assert "merxen.io.bead_removal.detect:fill_undefined_blocks" in ops_imports
    for name in ("ENVELOPE_FILTER_PCT", "ENVELOPE_GAUSS_SIGMA_BLOCKS"):
        assert f"merxen.io.bead_removal.detect:{name}" in ops_imports
    for path in gpu_files:
        assert not any(
            n.endswith(":_fill_undefined_blocks") for n in _imported_modules(path)
        ), f"{path.name} imports a private helper of the CPU envelope"


def test_the_gpu_subpackage_is_not_imported_by_the_cpu_package() -> None:
    code = (
        "import sys\n"
        "import merxen.io.bead_removal as b\n"
        "from merxen.io.bead_removal import backend, driver\n"
        "gpu = 'merxen.io.bead_removal.gpu'\n"
        "assert not any(m.startswith(gpu) for m in sys.modules)\n"
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# fixtures
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
def cal() -> Any:
    return real_calibration()


@pytest.fixture(scope="module")
def synth() -> dict[str, np.ndarray]:
    """768 px window: 45 beads, tissue blobs, nuclei and an antibody stain."""
    return bs.make_stack(size=768, n_beads=45, seed=11, n_nuclei=8)


def _stacks(st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: st[k] for k in ("PolyT", "DAPI", "Anti-X")}


@pytest.fixture(scope="module")
def numpy_result(synth: dict[str, np.ndarray], cal: Any) -> Any:
    return cpu_correct_window(
        _stacks(synth), origin_yx=(0, 0), calibration=cal, z_layers=Z
    )


@pytest.fixture(scope="module")
def colour_result(gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any) -> Any:
    # production defaults of FitOpts
    return gpu.correct_window(
        _stacks(synth), origin_yx=(0, 0), calibration=cal, z_layers=Z
    )


# ---------------------------------------------------------------------------
# G1: seq mode == numpy
@needs_gpu
@pytest.mark.slow
@pytest.mark.parametrize("triton", [True, False], ids=["triton", "torch"])
def test_g1_seq_mode_matches_numpy(
    gpu: ModuleType,
    synth: dict[str, np.ndarray],
    cal: Any,
    numpy_result: Any,
    triton: bool,
) -> None:
    ref, ref_model, ref_stats = numpy_result
    out, model, stats = gpu.correct_window(
        _stacks(synth),
        origin_yx=(0, 0),
        calibration=cal,
        z_layers=Z,
        opts=gpu.FitOpts(mode="seq", triton=triton),
    )
    # a real test: dozens of beads
    assert stats["status"] == "ok" and stats["n_beads_final"] > 25
    assert stats["n_beads_final"] == ref_stats["n_beads_final"]
    for s in STAINS:
        d = np.abs(out[s].astype(np.float64) - ref[s])
        assert d.max() <= 0.05, f"{s}: seq mode differs from numpy by {d.max():.4f}"
    assert np.array_equal(
        beads_table(model, (0, 0))[:, 7], beads_table(ref_model, (0, 0))[:, 7]
    )


# ---------------------------------------------------------------------------
# G2: colour mode vs numpy
@needs_gpu
@pytest.mark.slow
def test_g2_colour_mode_agrees_with_numpy_and_never_brightens(
    synth: dict[str, np.ndarray], numpy_result: Any, colour_result: Any
) -> None:
    ref, _, ref_stats = numpy_result
    out, _, stats = colour_result
    # the antibody stain is not corrected, so it is not returned
    assert set(out) == set(STAINS)
    n_ref = ref_stats["n_beads_final"]
    assert abs(stats["n_beads_final"] - n_ref) <= max(2, 0.03 * n_ref)
    for s in STAINS:
        raw = np.stack([synth[s][k] for k in Z])
        assert (out[s] <= raw).all(), f"{s}: a pixel got brighter"
        d = np.abs(out[s].astype(np.float64) - ref[s])
        assert (d > 10).mean() <= 0.001, f"{s}: {(d > 10).mean():.4%} differ by > 10"
    raw_p = np.stack([synth["PolyT"][k] for k in Z]).astype(np.float64)
    ratio = (raw_p - out["PolyT"]).sum() / (raw_p - ref["PolyT"]).sum()
    assert abs(ratio - 1.0) <= 0.002, f"removed-signal ratio GPU/numpy {ratio:.4f}"


@needs_gpu
def test_g2_maxproj_output_equals_projection_of_the_planes(
    gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any, colour_result: Any
) -> None:
    out_m, _, _ = gpu.correct_window(
        _stacks(synth),
        origin_yx=(0, 0),
        calibration=cal,
        z_layers=Z,
        output="maxproj",
    )
    for s in STAINS:
        assert out_m[s].shape == colour_result[0][s].shape[1:]
        assert np.array_equal(out_m[s], colour_result[0][s].max(0))


# ---------------------------------------------------------------------------
# G3: determinism
@needs_gpu
def test_g3_bitwise_deterministic_in_one_process(
    gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any, colour_result: Any
) -> None:
    again, _, _ = gpu.correct_window(
        _stacks(synth), origin_yx=(0, 0), calibration=cal, z_layers=Z
    )
    for s in STAINS:
        assert np.array_equal(again[s], colour_result[0][s])


@needs_gpu
@pytest.mark.slow
def test_g3_bitwise_deterministic_across_processes(
    cuda: None, cal: Any, colour_result: Any, tmp_path: Path
) -> None:
    cal_dir = tmp_path / "cal"
    cal.save(cal_dir)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS_DIR), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    runs = []
    for i in range(2):
        out = tmp_path / f"out{i}.npz"
        subprocess.run(
            [sys.executable, "-c", _WINDOW_SCRIPT, str(cal_dir), str(out)],
            check=True,
            env=env,
            timeout=900,
        )
        runs.append(np.load(out))
    for s in STAINS:
        assert np.array_equal(runs[0][s], runs[1][s]), f"{s}: two processes differ"
        assert np.array_equal(runs[0][s], colour_result[0][s]), (
            f"{s}: a fresh process differs from this one"
        )


# ---------------------------------------------------------------------------
# G6: the CUDA driver path does not depend on the number of reader threads
@needs_gpu
@pytest.mark.slow
def test_g6_identical_stores_for_one_and_two_reader_threads(
    cuda: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.io.bead_removal import driver

    try:
        driver.get_window_corrector("cuda")
    except ValueError:
        pytest.skip("the driver has no CUDA consumer in this version")
    shape = (1536, 2048)
    st = bs.make_stack(n_planes=4, size=shape, n_beads=70, seed=5, n_nuclei=10)
    images = tmp_path / "images"
    write_mosaic(images, st, 4)
    calibration = real_calibration()
    cal_dir = tmp_path / "cal"
    calibration.save(cal_dir)
    stores = []
    for n_threads in (1, 2):
        monkeypatch.setattr(driver, "GPU_READER_THREADS", n_threads, raising=False)
        store = open_projection_store(
            tmp_path / f"p{n_threads}.zarr",
            n_channels=2,
            shape=shape,
            chunk_px=1024,
        )
        driver.run_bead_removal(
            images_dir=images,
            stains=list(STAINS),
            z_layers=Z,
            store=store,
            calibration=calibration,
            calibration_dir=cal_dir,
            work_dir=tmp_path / f"w{n_threads}",
            core_px=1024,
            device="cuda",
        )
        stores.append(np.asarray(store[:]))
    assert stores[0].any() and np.array_equal(stores[0], stores[1])


# ---------------------------------------------------------------------------
# lower envelope with invalid pixels, edge margin, hash refusal, Jacobi, smoke test
def _envelope_cases(
    shape: tuple[int, int] = (768, 640),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img = (
        1500
        + 200 * np.sin(yy / 150.0)
        + 100 * np.cos(xx / 90.0)
        + rng.normal(0, 30, shape)
    ).astype(np.float32)
    img[rng.random(shape) < 0.001] += 4000
    full = np.ones(shape, bool)
    one = full.copy()
    one[0, 0] = False
    band = full.copy()
    band[:, :230] = False
    holes = full.copy()
    holes[80:330, 380:600] = False
    holes[700:, 500:] = False
    block = np.zeros(shape, bool)
    block[384:400, 384:400] = True
    stripes = full.copy()
    stripes[:, ::7] = False
    partial = full.copy()
    partial[3:9, 5:11] = False
    return img, {
        "all_valid": full,
        "one_pixel": one,
        "band": band,
        "holes": holes,
        "isolated_block": block,
        "stripes": stripes,
        "partial_block": partial,
        "none_valid": np.zeros(shape, bool),
    }


@needs_gpu
@pytest.mark.parametrize("sub", [1, 4])
def test_envelope_matches_numpy_for_every_validity_pattern(
    gpu: ModuleType, sub: int
) -> None:
    import torch

    img, cases = _envelope_cases()
    for name, valid in cases.items():
        im = np.where(valid, img, 0).astype(np.float32)
        ref = cpu_envelope(im, valid, sub=sub)
        got = (
            gpu.ops.lower_envelope(
                torch.from_numpy(im).cuda(), torch.from_numpy(valid).cuda(), sub=sub
            )
            .cpu()
            .numpy()
        )
        diff = np.abs(ref - got).max()
        assert diff <= 0.01, f"{name} (sub {sub}): envelope differs by {diff:.4f}"


@needs_gpu
def test_envelope_of_a_valid_pixel_does_not_depend_on_the_window(
    gpu: ModuleType,
) -> None:
    """No window-level statistic leaks into valid pixels (same block grid)."""
    import torch

    img, _ = _envelope_cases((768, 768))
    valid = np.ones((768, 768), bool)
    valid[:, :256] = False
    # an interior hole as well
    valid[300:420, 400:520] = False
    im = np.where(valid, img, 0).astype(np.float32)
    big = (
        gpu.ops.lower_envelope(
            torch.from_numpy(im).cuda(), torch.from_numpy(valid).cuda()
        )
        .cpu()
        .numpy()
    )
    # A window that starts on the same block grid but drops the right-hand 128
    # columns: valid pixels farther than the smoothing reach (4 + 8 blocks = 192 px,
    # plus the bilinear block) from the cut have the same envelope.
    cut = 640
    small = (
        gpu.ops.lower_envelope(
            torch.from_numpy(im[:, :cut]).cuda(),
            torch.from_numpy(valid[:, :cut]).cuda(),
        )
        .cpu()
        .numpy()
    )
    inner = np.zeros(valid.shape, bool)
    inner[:, : cut - 208] = valid[:, : cut - 208]
    assert inner.sum() > 100_000
    diff = np.abs(big[:, :cut][inner[:, :cut]] - small[inner[:, :cut]]).max()
    assert diff <= 0.01


@needs_gpu
def test_edge_margin_ignores_the_window_border(gpu: ModuleType, cal: Any) -> None:
    """One invalid pixel at the window corner must not change the candidates."""
    import torch

    st = bs.make_stack(
        size=640, n_beads=25, seed=21, blobs=False, punctum_near_bead=False
    )
    p0, p1, d0, d1 = (
        torch.from_numpy(st[s][k]).cuda()
        for s, k in (("PolyT", 0), ("PolyT", 1), ("DAPI", 0), ("DAPI", 1))
    )
    kw = {"dog_noise": cal.dog_noise_z0, "thresholds": cal.thresholds}
    full = gpu.detect.detect_sources(p0, p1, d0, d1, **kw)
    p0b = p0.clone()
    p0b[0, 0] = 0.0
    corner = gpu.detect.detect_sources(p0b, p1, d0, d1, **kw)
    assert corner.n == full.n and np.array_equal(corner.is_bead, full.is_bead)
    ref_corner = cpu_detect_sources(
        {0: p0b.cpu().numpy(), 1: p1.cpu().numpy()},
        {0: d0.cpu().numpy(), 1: d1.cpu().numpy()},
        **kw,
    )
    assert ref_corner.n == corner.n and np.allclose(ref_corner.y, corner.y, atol=0.05)
    assert np.array_equal(ref_corner.is_bead, corner.is_bead)


@needs_gpu
def test_window_with_invalid_band_matches_numpy_in_seq_mode(
    gpu: ModuleType, cal: Any
) -> None:
    """The invalid-pixel paths (envelope, edge margin) through the whole window."""
    st = bs.make_stack(size=640, n_beads=30, seed=31, n_nuclei=5)
    for s in ("PolyT", "DAPI", "Anti-X"):
        st[s][:, :, :150] = 0.0
        st[s][:, 600:, :] = 0.0
    stacks = _stacks(st)
    ref, _, _ = cpu_correct_window(
        stacks, origin_yx=(0, 0), calibration=cal, z_layers=Z
    )
    out, _, _ = gpu.correct_window(
        stacks,
        origin_yx=(0, 0),
        calibration=cal,
        z_layers=Z,
        opts=gpu.FitOpts(mode="seq"),
    )
    for s in STAINS:
        assert np.abs(out[s].astype(np.float64) - ref[s]).max() <= 0.05
        # invalid pixels stay invalid
        assert (out[s][:, :, :150] == 0).all()


@needs_gpu
def test_calibration_hash_mismatch_raises(
    gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any
) -> None:
    other = dataclasses.replace(DEFAULT_PARAMS, n_sweeps=DEFAULT_PARAMS.n_sweeps + 1)
    with pytest.raises(ValueError, match="different parameter set"):
        gpu.correct_window(
            _stacks(synth),
            origin_yx=(0, 0),
            calibration=cal,
            z_layers=Z,
            params=other,
        )


@needs_gpu
def test_jacobi_mode_is_gone(gpu: ModuleType) -> None:
    with pytest.raises(ValueError, match="unknown fit mode"):
        gpu.FitOpts(mode="jacobi")
    opts = gpu.FitOpts()
    # the defaults are the production configuration
    assert (opts.mode, opts.triton, opts.anchor, opts.s_cap) == (
        "color",
        True,
        True,
        1.6,
    )


@needs_gpu
def test_triton_smoke_test_passes(gpu: ModuleType) -> None:
    rec = gpu.triton_smoke_test()
    assert abs(rec["gain"] - 1.5) < 0.01
    assert rec["model_rel_err"] < 1e-3 and rec["triton"] and rec["torch"]


@needs_gpu
def test_smoke_test_reports_a_broken_kernel(
    gpu: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.io.bead_removal.gpu import tri

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Failed to find C compiler")

    monkeypatch.setattr(tri, "fit_batch", boom)
    with pytest.raises(gpu.GpuSmokeTestError, match="C compiler"):
        gpu.triton_smoke_test()


# ---------------------------------------------------------------------------
# the gated z1-ring rescue of the classifier
def _rescue_thresholds(cal: Any, zmax: float = 0.0, sd1_max: float = 0.08) -> Any:
    """Thresholds with ``ring_min`` raised so that every bead takes the rescue path."""
    ring_z1 = float(cal.library.summary()[1]["ring_radius_px"])
    return dataclasses.replace(
        cal.thresholds,
        ring_min=0.95,
        rescue_tz1_min=4.0,
        rescue_ring_radius_px=ring_z1,
        rescue_a0_min=1000.0,
        rescue_sd1_max=sd1_max,
        rescue_zprof_zmax=zmax,
    )


def _cpu_detect(st: dict[str, np.ndarray], cal: Any, thresholds: Any) -> Any:
    return cpu_detect_sources(
        {k: st["PolyT"][k] for k in range(7)},
        {0: st["DAPI"][0], 1: st["DAPI"][1]},
        dog_noise=cal.dog_noise_z0,
        thresholds=thresholds,
    )


@needs_gpu
@pytest.mark.parametrize("zmax", [0.0, 6.0], ids=["texture_gate", "z_profile"])
def test_rescue_detect_matches_numpy(gpu: ModuleType, cal: Any, zmax: float) -> None:
    """The rescue decisions of the GPU detector equal the numpy ones.

    A secondary focus at z3 on one rescued bead blocks it only when the z-profile
    test is on.
    """
    import torch

    from merxen.io.bead_removal.gpu.detect import detect_sources

    st = bs.make_stack(
        size=640,
        n_beads=25,
        seed=21,
        blobs=False,
        punctum_near_bead=False,
        tissue=False,
    )
    th = _rescue_thresholds(cal, zmax)
    first = _cpu_detect(st, cal, th)
    assert first.rescued.sum() >= 5
    k = int(np.where(first.rescued)[0][0])
    yy, xx = np.mgrid[:640, :640]
    focus = np.exp(-((yy - first.y[k]) ** 2 + (xx - first.x[k]) ** 2) / (2 * 3.0**2))
    st["PolyT"][3] += (0.6 * first.A0[k] * focus).astype(np.float32)
    ref = _cpu_detect(st, cal, th)
    assert ref.rescued.sum() == first.rescued.sum() - (1 if zmax else 0)

    planes = {k: torch.from_numpy(st["PolyT"][k]).cuda() for k in range(7)}
    dapi0, dapi1 = (torch.from_numpy(st["DAPI"][k]).cuda() for k in (0, 1))
    got = detect_sources(
        planes[0],
        planes[1],
        dapi0,
        dapi1,
        dog_noise=cal.dog_noise_z0,
        thresholds=th,
        z_planes={k: planes[k] for k in range(2, 7)},
    )
    assert got.n == ref.n
    assert np.array_equal(got.rescued, ref.rescued)
    assert np.array_equal(got.is_bead, ref.is_bead)
    assert np.allclose(got.sd1_rel, ref.sd1_rel, rtol=1e-4, atol=1e-6)
    assert np.allclose(got.t1, ref.t1, equal_nan=True, rtol=1e-4, atol=1e-4)


@needs_gpu
def test_rescue_z_profile_ignores_planes_that_are_not_given(
    gpu: ModuleType, cal: Any
) -> None:
    import torch

    from merxen.io.bead_removal.gpu.detect import detect_sources

    st = bs.make_stack(
        size=640,
        n_beads=25,
        seed=21,
        blobs=False,
        punctum_near_bead=False,
        tissue=False,
    )
    th = _rescue_thresholds(cal, 6.0)
    planes = {k: torch.from_numpy(st["PolyT"][k]).cuda() for k in range(2)}
    dapi0, dapi1 = (torch.from_numpy(st["DAPI"][k]).cuda() for k in (0, 1))
    got = detect_sources(
        planes[0],
        planes[1],
        dapi0,
        dapi1,
        dog_noise=cal.dog_noise_z0,
        thresholds=th,
    )
    ref = cpu_detect_sources(
        {k: st["PolyT"][k] for k in range(2)},
        {0: st["DAPI"][0], 1: st["DAPI"][1]},
        dog_noise=cal.dog_noise_z0,
        thresholds=th,
    )
    assert got.rescued.sum() >= 5
    assert np.array_equal(got.rescued, ref.rescued)


@needs_gpu
def test_rescue_is_off_without_the_texture_gate_in_the_gpu_detector(
    gpu: ModuleType, cal: Any
) -> None:
    import torch

    from merxen.io.bead_removal.gpu.detect import detect_sources

    st = bs.make_stack(
        size=640,
        n_beads=25,
        seed=21,
        blobs=False,
        punctum_near_bead=False,
        tissue=False,
    )
    th = _rescue_thresholds(cal, 0.0, sd1_max=0.0)
    t0, t1 = (torch.from_numpy(st["PolyT"][k]).cuda() for k in (0, 1))
    d0, d1 = (torch.from_numpy(st["DAPI"][k]).cuda() for k in (0, 1))
    got = detect_sources(t0, t1, d0, d1, dog_noise=cal.dog_noise_z0, thresholds=th)
    assert not got.rescued.any() and not got.is_bead.any()


@needs_gpu
def test_the_device_ring_tscore_is_the_cpu_one(gpu: ModuleType) -> None:
    import torch

    from merxen.io.bead_removal.detect import ring_tscores
    from merxen.io.bead_removal.gpu.detect import _ring_tscores_device

    rng = np.random.default_rng(0)
    img = rng.normal(100, 5, (200, 200)).astype(np.float32)
    yy, xx = np.mgrid[:200, :200]
    img += (30 * np.exp(-((np.hypot(yy - 80, xx - 90) - 29.8) ** 2) / 18)).astype(
        np.float32
    )
    ys = np.array([80.2, 100.0, 150.4, 199.0])  # a ring, noise, noise, off the image
    xs = np.array([90.4, 100.0, 120.0, 100.0])
    got = _ring_tscores_device(torch.from_numpy(img).cuda(), ys, xs, 29.8)
    want = ring_tscores(img, ys, xs, 29.8)
    np.testing.assert_array_equal(got, want)  # bitwise, NaN where the window leaves
    assert got[0] > 4 and np.isnan(got[3]) and np.isfinite(got[1:3]).all()


@needs_gpu
@pytest.mark.slow
@pytest.mark.parametrize("zmax", [0.0, 6.0], ids=["texture_gate", "z_profile"])
def test_g1_seq_mode_matches_numpy_with_the_rescue(
    gpu: ModuleType, synth: dict[str, np.ndarray], cal: Any, zmax: float
) -> None:
    """Whole window with every bead on the rescue path: GPU equals numpy, same set."""
    cal_r = dataclasses.replace(cal, thresholds=_rescue_thresholds(cal, zmax))
    stacks = _stacks(synth)
    ref, ref_model, ref_stats = cpu_correct_window(
        stacks, origin_yx=(0, 0), calibration=cal_r, z_layers=Z
    )
    out, model, stats = gpu.correct_window(
        stacks,
        origin_yx=(0, 0),
        calibration=cal_r,
        z_layers=Z,
        opts=gpu.FitOpts(mode="seq"),
    )
    assert ref_stats["n_rescued"] >= 10
    assert stats["n_rescued"] == ref_stats["n_rescued"]
    assert np.array_equal(model.src.rescued, ref_model.src.rescued)
    for s in STAINS:
        assert np.abs(out[s].astype(np.float64) - ref[s]).max() <= 0.05
