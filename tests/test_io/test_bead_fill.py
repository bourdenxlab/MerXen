"""Tests of the optional bounded fill of residual bead light (synthetic data only).

F1 off is bit-identical (also end to end through ``correct_window``), F2 one-sided,
F3 bounded by eps x light, F4 invalid pixels, F5 removes crest residual on flat
background, F6 bright-source restriction, F8 chunk invariance (different window
cuts, four configurations), F9 noise is required, F10 defaults and support, F11
``correct_window`` integration, F12 parameter validation, F12b only the reference
stain can run, F13 odd shapes and determinism, F14 dense coating is not filled.

Tissue damage: N1 null test on tissue (a perfect subtraction must not be changed),
N2 null test on flat background (no more than the noise deadband), N3 tissue is
left alone while the same ring on flat background is filled.

Hybrid mode: H1 fills residual ring light on textured background where flat mode
does nothing, H2 never brighter and bounded by eps x light, H3 the gate follows the
local texture sd, H4 chunk invariance across window origins that are not on the
32 px texture grid, H5 the flat hash and support are unchanged and the hybrid
support fits, H6 the mode's effective gates ignore the flat fields, H7 invalid
pixels, H8 the origin check, H9 ``correct_window`` passes the absolute origin.

Chunk invariance: B1 a faint candidate bridge to a bright source outside the
window (0.00 difference at margin 256 and at ``support_px + 2``), B2 the support of
the default configuration is at most 254 px and larger settings are refused, R2
candidates beyond the reach of a bright source are dropped.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from typing import Any

import numpy as np
import pytest
from scipy import ndimage as ndi

from merxen.io.bead_removal import correct_window, fill
from merxen.io.bead_removal.fill import (
    MAX_SUPPORT_PX,
    FillParams,
    apply_bounded_fill,
    fill_params_sha256,
    support_px,
)

from . import bead_synth as bs

NOISE = {k: 37.0 for k in range(8)}
Z = [1, 2, 3]
COUNTERS = {
    "cand_px",
    "reach_dropped_px",
    "dense_skipped_px",
    "not_flat_px",
    "unfilled_px",
    "changed_px",
    "removed_sum",
}


def _ring(
    shape: tuple[int, int], yc: float, xc: float, r0: float, width: float, amp: float
) -> np.ndarray:
    """Gaussian ring of radius ``r0`` (a punctum when ``r0`` is 0)."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    r = np.hypot(yy - yc, xx - xc)
    return (amp * np.exp(-0.5 * ((r - r0) / width) ** 2)).astype(np.float32)


def make_light(
    size: tuple[int, int] = (512, 512),
    beads: tuple[tuple[float, ...], ...] = ((256, 256, 30.0, 6.0, 1500.0),),
    n_z: int = 3,
) -> np.ndarray:
    """Model light of ring-shaped beads.

    The ring radius and the amplitude change with the plane like a defocused bead.
    """
    light = np.zeros((n_z, *size), np.float32)
    for i in range(n_z):
        for yc, xc, r0, w, a in beads:
            light[i] += _ring(size, yc, xc, r0 * (1 + 0.5 * i), w, a / (1 + 2 * i))
    return light


def make_planes(
    size: tuple[int, int] = (512, 512),
    seed: int = 0,
    beads: tuple[tuple[float, ...], ...] = ((256, 256, 30.0, 6.0, 1500.0),),
    punct: tuple[tuple[int, int], ...] = ((256, 226),),
    n_z: int = 3,
    residual: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat background with a bead ring left in the corrected plane.

    Off tissue: flat background, a ring of model light, ``residual`` x the light
    left in the corrected plane, a punctum on the ring and pixel noise.

    Returns:
        (corrected, light, raw).
    """
    rng = np.random.default_rng(seed)
    light = make_light(size, beads, n_z)
    corr = np.empty((n_z, *size), np.float32)
    for i in range(n_z):
        corr[i] = (
            1000 + rng.normal(0, 30, size).astype(np.float32) + residual * light[i]
        )
        for py, px in punct:
            corr[i] += _ring(size, py, px, 0.0, 1.6, 700.0)
    raw = corr + light
    return corr.astype(np.float32), light, raw.astype(np.float32)


def make_tissue(
    size: tuple[int, int] = (512, 512), seed: int = 1, n_z: int = 3
) -> np.ndarray:
    """Ground truth of bright, smooth tissue.

    Level about 2800 with cell-like blobs, processes (ridges) and puncta of
    100-1500 counts. It is smooth at the 6 px scale, so a local high-pass texture
    test would call it textureless (the failure mode of the first fill).
    """
    rng = np.random.default_rng(seed)
    gt = np.empty((n_z, *size), np.float32)
    for i in range(n_z):
        lvl = 2800 + ndi.gaussian_filter(rng.normal(0, 1, size), 12) * 4000
        cells = (
            ndi.gaussian_filter((rng.random(size) > 0.9995).astype(np.float32), 7)
            * 6e4
            * 40
        )
        ridges = (
            ndi.gaussian_filter((rng.random(size) > 0.99).astype(np.float32), 2.5) * 8e3
        )
        puncta = (
            ndi.gaussian_filter((rng.random(size) > 0.9996).astype(np.float32), 1.5)
            * 8e4
        )
        gt[i] = (lvl + cells + ridges + puncta).astype(np.float32)
    return gt


def _fill(
    corr: np.ndarray,
    light: np.ndarray,
    raw: np.ndarray,
    config: FillParams | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Run the fill on the reference stain with the default noise and planes."""
    cfg = FillParams(enabled=True) if config is None else config
    kwargs.setdefault("noise", NOISE)
    return apply_bounded_fill(
        corr, light, raw=raw, z_layers=Z, config=cfg, stain="PolyT", **kwargs
    )


def test_f10_defaults_off_and_support() -> None:
    p = FillParams()
    assert p.enabled is False
    sup = support_px(p)
    assert sup == {"fine": 32, "cover": 136, "reach": 64, "coarse": 144, "total": 232}
    # Below the 256 px halo (the driver asserts halo - 2).
    assert sup["total"] <= MAX_SUPPORT_PX == 254
    assert sup["reach"] == p.bright_reach_px
    assert sup["total"] >= sup["coarse"] + sup["reach"]
    assert (p.bright_reach_px, p.flat_max, p.deadband) == (64, 1.0, 2.0)
    assert fill_params_sha256(p) != fill_params_sha256(
        dataclasses.replace(p, enabled=True)
    )
    assert fill_params_sha256(p) != fill_params_sha256(
        dataclasses.replace(p, flat_max=2.0)
    )


def test_the_removed_options_are_gone() -> None:
    """No puncta protection, no texture gate and no second name for the params."""
    names = {f.name for f in dataclasses.fields(FillParams)}
    assert not {n for n in names if n.startswith("protect")}
    assert not hasattr(fill, "FillConfig")


def test_f1_off_is_bit_identical() -> None:
    corr, light, raw = make_planes()
    out = _fill(corr, light, raw, FillParams())
    assert out is corr
    assert np.array_equal(out, corr)
    assert out.dtype == np.float32
    # Without noise a disabled fill still returns the input untouched.
    assert _fill(corr, light, raw, FillParams(), noise=None) is corr


@pytest.fixture(scope="module")
def filled() -> tuple[Any, ...]:
    corr, light, raw = make_planes()
    cfg = FillParams(enabled=True)
    stats: dict[str, Any] = {}
    out = _fill(corr, light, raw, cfg, stats=stats)
    return corr, light, raw, cfg, out, stats


def test_f2_one_sided_and_f3_bounded(filled: tuple[Any, ...]) -> None:
    corr, light, _raw, cfg, out, _ = filled
    assert (out <= corr).all()
    for i, k in enumerate(Z):
        udil = ndi.grey_dilation(cfg.eps[k] * light[i], size=2 * cfg.dilation + 1)
        assert (out[i] >= corr[i] - udil - 1e-3).all()


def test_f4_invalid_pixels_unchanged() -> None:
    corr, light, raw = make_planes()
    raw = raw.copy()
    raw[:, 200:320, 200:260] = 0
    corr = corr.copy()
    corr[:, 200:320, 200:260] = 0
    out = _fill(corr, light, raw)
    assert (out[:, 200:320, 200:260] == 0).all()
    assert (out <= corr).all()


def test_f5_removes_crest_residual_on_flat_background(
    filled: tuple[Any, ...],
) -> None:
    corr, light, _raw, _cfg, out, stats = filled
    # Residual of plane 1 on the ring: the fill must remove a sizeable part of it,
    # never more than eps x light.
    ring = light[0] > 600
    before = (corr[0] - 1000)[ring].mean()
    after = (out[0] - 1000)[ring].mean()
    # At least a third of the 40 % residual is gone.
    assert after < before - 0.3 * 0.4 * light[0][ring].mean()
    assert stats["changed_px"] > 1000
    assert stats["cand_px"] > 0
    assert stats["not_flat_px"] == 0


def test_the_qc_counters_are_the_documented_set(filled: tuple[Any, ...]) -> None:
    _corr, _light, _raw, _cfg, _out, stats = filled
    assert set(stats) >= COUNTERS
    assert set(stats) == COUNTERS | {"per_plane"}
    assert sorted(stats["per_plane"]) == Z
    assert all(set(c) <= COUNTERS for c in stats["per_plane"].values())
    for key in COUNTERS - {"unfilled_px", "dense_skipped_px", "not_flat_px"}:
        assert stats[key] == pytest.approx(
            sum(c.get(key, 0.0) for c in stats["per_plane"].values())
        )


def test_n3_tissue_is_left_alone_but_the_same_ring_on_flat_background_is_filled() -> (
    None
):
    """The same ring residual on bright smooth tissue is not touched."""
    gt = make_tissue()
    rng = np.random.default_rng(2)
    light = make_light()
    tissue = (gt + 0.4 * light + rng.normal(0, 30, gt.shape)).astype(np.float32)
    stats: dict[str, Any] = {}
    out_t = _fill(tissue, light, tissue + light, stats=stats)
    assert np.array_equal(out_t, tissue)
    assert stats["not_flat_px"] > 0
    corr, light2, raw = make_planes(punct=())
    out_f = _fill(corr, light2, raw)
    assert (out_f < corr).sum() > 1000


def test_n1_null_test_on_tissue() -> None:
    """A perfect subtraction (ground truth + noise) with real-shaped model light.

    The fill may change less than 0.1 % of the ground-truth structure pixels and
    remove at most 15 counts per footprint pixel (footprint: eps x light > noise).
    """
    gt = make_tissue(size=(768, 768), seed=5)
    rng = np.random.default_rng(3)
    beads = (
        (200, 200, 30.0, 6.0, 3000.0),
        (500, 300, 30.0, 6.0, 2200.0),
        (300, 580, 30.0, 6.0, 4000.0),
        (620, 620, 30.0, 6.0, 1800.0),
    )
    light = make_light((768, 768), beads)
    obs = (gt + rng.normal(0, 37, gt.shape)).astype(np.float32)
    out = _fill(obs, light, obs + light)
    struct = np.stack([(g - ndi.median_filter(g, 31)) > 111.0 for g in gt])
    changed = out < obs - 0.5
    # The fixture has structure under the footprint.
    assert struct.sum() > 1000
    assert (changed & struct).sum() / struct.sum() < 1e-3
    footprint = 0.3 * light > NOISE[1]
    assert footprint.sum() > 5000
    assert (obs - out)[footprint].sum() / footprint.sum() <= 15.0


def test_n2_null_test_on_flat_background_stays_within_the_noise_deadband() -> None:
    """With nothing left to remove the deadband keeps the clip off the pixel noise."""
    rng = np.random.default_rng(4)
    light = make_light((512, 512))
    obs = (1800 + rng.normal(0, 37, light.shape)).astype(np.float32)
    out = _fill(obs, light, obs + light)
    footprint = 0.3 * light > NOISE[1]
    assert footprint.sum() > 5000
    # A hard clip removes about 9 counts per footprint pixel.
    assert (obs - out)[footprint].mean() <= 5.0
    hard = _fill(obs, light, obs + light, FillParams(enabled=True, deadband=0.0))
    # About 9 here: bounded by eps x light where the light is faint.
    assert (obs - hard)[footprint].mean() > 5.0


def test_f6_bright_source_restriction() -> None:
    # A bright bead and a faint one: with peak_snr the faint one is left alone and
    # the bright one is still filled.
    beads = ((120, 120, 30.0, 6.0, 1500.0), (380, 380, 30.0, 6.0, 330.0))
    corr, light, raw = make_planes(beads=beads, punct=())
    free = _fill(corr, light, raw, FillParams(enabled=True, peak_snr=0.0))
    res = _fill(corr, light, raw, FillParams(enabled=True, peak_snr=3.0))
    faint = (slice(300, 460), slice(300, 460))
    bright = (slice(40, 200), slice(40, 200))
    # Unrestricted touches the faint source (eps x 330 x 0.4 > noise).
    assert (free[0][faint] < corr[0][faint]).sum() > 0
    assert np.array_equal(res[0][faint], corr[0][faint])
    assert (res[0][bright] < corr[0][bright]).sum() > 0


def test_r2_candidates_beyond_the_reach_of_a_bright_source_are_dropped() -> None:
    """A faint band farther than ``bright_reach_px`` from a bright pixel is left."""
    size = (512, 1024)
    light = np.zeros((3, *size), np.float32)
    for i in range(3):
        light[i] += _ring(
            size, 256, 100, 30.0 * (1 + 0.5 * i), 6.0, 3000.0 / (1 + 2 * i)
        )
        # Faint band: eps x light = 48 > noise 37, below 3 x noise.
        light[i][250:262, 100:1000] += 160.0
    rng = np.random.default_rng(6)
    corr = (1000 + rng.normal(0, 30, light.shape) + 0.4 * light).astype(np.float32)
    stats: dict[str, Any] = {}
    out = _fill(corr, light, corr + light, stats=stats)
    far = (slice(None), slice(250, 262), slice(600, 1000))
    assert np.array_equal(out[far], corr[far])
    assert stats["reach_dropped_px"] > 0
    # Near the bright source it is filled.
    assert (out[:, :, :200] < corr[:, :, :200]).sum() > 500


@pytest.mark.slow
@pytest.mark.parametrize("deadband", [0.0, 2.0])
def test_b1_bridge_to_a_bright_source_outside_the_window_is_chunk_invariant(
    deadband: float,
) -> None:
    """A long faint candidate bridge connects a bright source to a far region.

    A window that excludes the bright source must give 0.00 difference to the full
    fill at margin 256 and at margin ``support_px + 2`` (a connected-component
    restriction gave 48 counts).
    """
    h, w = 1024, 1792
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    light = np.zeros((3, h, w), np.float32)
    for i in range(3):
        r = np.hypot(yy - 512, xx - 80)
        ring = np.exp(-0.5 * ((r - 30.0 * (1 + 0.5 * i)) / 6.0) ** 2)
        light[i] += 3000.0 / (1 + 2 * i) * ring
        light[i][506:518, 100:1700] += 160.0
    rng = np.random.default_rng(5)
    corr = (1000 + rng.normal(0, 30, light.shape) + 0.4 * light).astype(np.float32)
    raw = corr + light
    cfg = FillParams(enabled=True, deadband=deadband)
    full = _fill(corr, light, raw, cfg)
    assert (full < corr).sum() > 1000
    # The window starts on the 16 px grid and excludes the bright source (x < 150).
    x0 = 512
    sub = _fill(corr[:, :, x0:], light[:, :, x0:], raw[:, :, x0:], cfg)
    for margin in (256, support_px(cfg)["total"] + 2):
        assert np.abs(sub[:, :, margin:] - full[:, :, x0 + margin :]).max() == 0.0


def test_b2_support_limit_is_enforced() -> None:
    assert support_px(FillParams())["total"] <= MAX_SUPPORT_PX
    with pytest.raises(ValueError, match="bright_reach_px"):
        FillParams(bright_reach_px=65)
    with pytest.raises(ValueError, match="bright_reach_px"):
        FillParams(bright_reach_px=0)
    # Gaussian reach 48 px: support above 254.
    with pytest.raises(ValueError, match="support"):
        FillParams(fine_sigmas=(1.5, 3.0, 12.0))
    # Coarse reach 33 blocks: support above 254.
    with pytest.raises(ValueError, match="support"):
        FillParams(coarse_sigmas=(1.0, 2.0, 8.0))


def test_f9_noise_is_required() -> None:
    corr, light, raw = make_planes()
    with pytest.raises(ValueError, match="noise"):
        _fill(corr, light, raw, noise=None)
    # Planes 2 and 3 are missing.
    with pytest.raises(ValueError, match="plane 2"):
        _fill(corr, light, raw, noise={1: 37.0})


@pytest.mark.slow
@pytest.mark.parametrize(
    "cfg_kw",
    [
        {},
        {"peak_snr": 0.0, "max_cover": 1.0, "deadband": 0.0},
        {"flat_max": 1e6, "gate": 0.5, "max_cover": 0.3},
        {"flat_max": 1e6, "bright_reach_px": 64, "dilation": 3},
    ],
    ids=["default", "no_restriction", "flat_off_gate_cover", "flat_off_dilation"],
)
def test_f8_chunk_invariance(cfg_kw: dict[str, Any]) -> None:
    """A pixel's result must not depend on how the window was cut.

    Windows are shifted by multiples of 16 px and have different extents. The left
    part is tissue-like (the fill is inert there with the default flatness limit),
    the right part flat background.
    """
    size = (1280, 1536)
    beads = (
        (300, 300, 30.0, 6.0, 2500.0),
        (620, 700, 30.0, 6.0, 1800.0),
        (900, 1100, 30.0, 6.0, 3000.0),
        (350, 1250, 30.0, 6.0, 2200.0),
        (1000, 400, 30.0, 6.0, 1500.0),
        (700, 1300, 30.0, 6.0, 2600.0),
    )
    corr, light, raw = make_planes(
        size=size, beads=beads, punct=((300, 330), (620, 668), (900, 1070)), seed=3
    )
    rng = np.random.default_rng(9)
    tex = ndi.gaussian_filter(rng.normal(0, 1, size).astype(np.float32), 4) * 600
    tex[:, 800:] = 0
    corr = corr + tex[None]
    raw = raw + tex[None]
    cfg = FillParams(enabled=True, **cfg_kw)
    full = _fill(corr, light, raw, cfg)
    assert (full < corr).any()
    for y0, y1, x0, x1 in [
        (16, 1200, 48, 1456),
        (160, 1280, 0, 1280),
        (0, 1040, 256, 1536),
    ]:
        sub = _fill(
            corr[:, y0:y1, x0:x1],
            light[:, y0:y1, x0:x1],
            raw[:, y0:y1, x0:x1],
            cfg,
        )
        # The window edges are excluded.
        m = support_px(cfg)["total"] + 2
        ya, yb = (m if y0 > 0 else 0), (y1 - y0 - (m if y1 < size[0] else 0))
        xa, xb = (m if x0 > 0 else 0), (x1 - x0 - (m if x1 < size[1] else 0))
        a = sub[:, ya:yb, xa:xb]
        b = full[:, y0 + ya : y0 + yb, x0 + xa : x0 + xb]
        assert np.abs(a - b).max() == 0.0, (cfg_kw, (y0, y1, x0, x1))


@pytest.mark.slow
def test_f11_correct_window_integration() -> None:
    cal = bs.make_calibration()
    st = bs.make_stack(seed=1, n_beads=60)
    stacks = {k: st[k] for k in ("PolyT", "DAPI")}

    def run(**kwargs: Any) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
        return correct_window(
            stacks, origin_yx=(0, 0), calibration=cal, z_layers=Z, **kwargs
        )

    base, _, base_stats = run()
    off, _, _ = run(fill=FillParams())
    on, _, stats = run(fill=FillParams(enabled=True))
    for stain in base:
        # A disabled fill is bit-identical, end to end.
        assert np.array_equal(base[stain], off[stain])
        # An enabled fill is never brighter than the subtraction.
        assert (on[stain] <= base[stain]).all()
    # The fill runs on the reference stain only.
    assert "fill" not in base_stats
    assert set(stats["fill"]) == {"PolyT"}
    assert set(stats["fill"]["PolyT"]) >= COUNTERS


def test_f12_parameter_validation() -> None:
    for bad in (
        {"eps": (0.3,) * 7},
        {"eps": (0.3,) * 7 + (1.5,)},
        {"gate": 0.0},
        {"floor_frac": 0.0},
        {"dilation": -1},
        {"tau": 0.0},
        {"fine_sigmas": ()},
        {"coarse_sigmas": ()},
        {"flat_max": 0.0},
        {"deadband": -1.0},
        {"smooth": -1.0},
    ):
        with pytest.raises(ValueError):
            FillParams(**bad)


def test_f12b_only_the_reference_stain_can_be_listed() -> None:
    """No other stain has a calibrated noise floor: the params refuse to list one."""
    with pytest.raises(ValueError, match="reference stain"):
        FillParams(enabled=True, stains=("PolyT", "DAPI"))
    with pytest.raises(ValueError, match="reference stain"):
        FillParams(enabled=True, stains=("DAPI",))
    with pytest.raises(ValueError, match="reference stain"):
        FillParams(stains=())
    # A different reference stain is allowed when the stains follow it.
    alt = FillParams(stains=("DAPI",), reference_stain="DAPI")
    assert alt.stains == ("DAPI",)
    # A stain that is not listed is left untouched, also when enabled.
    corr, light, raw = make_planes()
    out = apply_bounded_fill(
        corr,
        light,
        raw=raw,
        z_layers=Z,
        config=FillParams(enabled=True),
        stain="DAPI",
        noise=NOISE,
    )
    assert out is corr


def test_f13_odd_shapes_and_determinism() -> None:
    """Shapes that are not a multiple of 8 equal the padded computation."""
    corr, light, raw = make_planes(size=(512, 512))
    cut = (slice(None), slice(0, 507), slice(0, 499))
    a = _fill(corr[cut], light[cut], raw[cut])
    assert a.shape == (3, 507, 499)
    assert (a <= corr[cut]).all()
    pad = ((0, 0), (0, 5), (0, 5))
    b = _fill(np.pad(corr[cut], pad), np.pad(light[cut], pad), np.pad(raw[cut], pad))
    assert np.array_equal(a, b[:, :507, :499])
    c = _fill(corr[cut], light[cut], raw[cut])
    assert np.array_equal(a, c)


def test_f14_dense_coating_is_not_filled() -> None:
    """With (almost) the whole plane candidate there is nothing to interpolate from."""
    size = (512, 512)
    # Bead pedestal everywhere.
    light = np.full((3, *size), 900.0, np.float32)
    rng = np.random.default_rng(4)
    corr = (1000 + rng.normal(0, 30, (3, *size))).astype(np.float32)
    raw = corr + light
    stats: dict[str, Any] = {}
    on = _fill(corr, light, raw, FillParams(enabled=True, peak_snr=0.0), stats=stats)
    assert np.array_equal(on, corr)
    assert stats["dense_skipped_px"] > 0
    off = _fill(corr, light, raw, FillParams(enabled=True, peak_snr=0.0, max_cover=1.0))
    assert (off <= corr).all()


# ---------------------------------------------------------------------------
# Hybrid mode
# ---------------------------------------------------------------------------
HYBRID = FillParams(enabled=True, mode="hybrid")
# Hash of the flat defaults before the hybrid mode existed (a recorded cache
# identity must not change).
FLAT_DEFAULT_SHA256 = "89cee507dacf718f6e34c035602e4f743b84598bbf8048c6308174bdbb71c82a"
FLAT_ENABLED_SHA256 = "aa80d327de2cfcf92db1fffc4e7f70ceb845054f0da404d6920d63f10e60ae69"


def make_textured_planes(
    size: tuple[int, int] = (512, 512),
    seed: int = 0,
    texture_amp: float = 800.0,
    beads: tuple[tuple[float, ...], ...] = ((256, 256, 30.0, 6.0, 1500.0),),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bead ring residual (0.4 x light) on a textured, tissue-like background.

    Returns:
        (corrected, light, raw).
    """
    rng = np.random.default_rng(seed)
    light = make_light(size, beads)
    tex = np.stack(
        [
            ndi.gaussian_filter(rng.normal(0, 1, size).astype(np.float32), 3)
            * texture_amp
            for _ in range(light.shape[0])
        ]
    )
    corr = (1000 + tex + rng.normal(0, 30, light.shape) + 0.4 * light).astype(
        np.float32
    )
    return corr, light, (corr + light).astype(np.float32)


@pytest.fixture(scope="module")
def hybrid_textured() -> tuple[Any, ...]:
    corr, light, raw = make_textured_planes()
    stats: dict[str, Any] = {}
    out = _fill(corr, light, raw, HYBRID, stats=stats)
    return corr, light, raw, out, stats


def test_h1_hybrid_fills_textured_background_where_flat_mode_does_nothing(
    hybrid_textured: tuple[Any, ...],
) -> None:
    corr, light, raw, out, stats = hybrid_textured
    flat_stats: dict[str, Any] = {}
    flat = _fill(corr, light, raw, FillParams(enabled=True), stats=flat_stats)
    # The flat fill is inert here: every candidate sits on non-flat surroundings.
    assert np.array_equal(flat, corr)
    assert flat_stats["not_flat_px"] > 0
    # The hybrid fill removes a sizeable part of the residual on the ring.
    assert (out < corr).sum() > 1000
    ring = light[0] > 600
    assert (corr[0] - out[0])[ring].mean() > 0.3 * 0.4 * light[0][ring].mean()
    # No flatness gate, premise guard or bright-source restriction in the counters.
    assert stats["cand_px"] > 0
    assert not {"not_flat_px", "dense_skipped_px", "reach_dropped_px"} & set(stats)
    assert "texture_dropped_px" in stats


def test_h2_hybrid_is_one_sided_and_bounded_by_eps_times_the_model(
    hybrid_textured: tuple[Any, ...],
) -> None:
    corr, light, _raw, out, _stats = hybrid_textured
    assert out.dtype == np.float32
    assert (out <= corr).all()
    for i, k in enumerate(Z):
        udil = ndi.grey_dilation(HYBRID.eps[k] * light[i], size=2 * HYBRID.dilation + 1)
        assert (out[i] >= corr[i] - udil - 1e-3).all()


def test_h3_the_hybrid_gate_is_the_local_texture_sd_not_the_global_noise() -> None:
    """On white noise of sd 100 only pixels near eps x light > ~100 are filled."""
    rng = np.random.default_rng(7)
    light = make_light((512, 512))
    corr = (1800 + rng.normal(0, 100, light.shape) + 0.4 * light).astype(np.float32)
    out = _fill(corr, light, corr + light, HYBRID)
    n_changed = 0
    for i, k in enumerate(Z):
        u = HYBRID.eps[k] * light[i]
        changed = out[i] < corr[i]
        n_changed += int(changed.sum())
        # A candidate has u above the local sd (about 100, well above the global
        # floor of 37); the mask is dilated by 2 px. With the floor as the gate,
        # the hard clip would also change pixels with u between 37 and 60.
        assert (ndi.grey_dilation(u, size=5)[changed] > 60.0).all()
    assert n_changed > 500


def test_h3b_a_textured_plane_is_filled_less_than_a_quiet_one() -> None:
    quiet = make_textured_planes(texture_amp=0.0)
    busy = make_textured_planes(texture_amp=3000.0)
    n_quiet = (_fill(*quiet, HYBRID) < quiet[0]).sum()
    n_busy = (_fill(*busy, HYBRID) < busy[0]).sum()
    assert n_busy < n_quiet


def test_h4_hybrid_chunk_invariance_for_origins_off_the_texture_grid() -> None:
    """Windows whose origin is on the 16 px grid but not on the 32 px texture grid.

    The texture blocks sit on the absolute grid, so a window that starts 16 px past
    a block edge must give the same pixels as the full mosaic.
    """
    size = (704, 1024)
    beads = (
        (200, 250, 30.0, 6.0, 2500.0),
        (480, 560, 30.0, 6.0, 1800.0),
        (300, 880, 30.0, 6.0, 3000.0),
    )
    corr, light, raw = make_textured_planes(size=size, seed=3, beads=beads)
    # Part of the mosaic is outside the data (invalid), as at a real mosaic edge.
    corr[:, :, :40] = 0
    raw[:, :, :40] = 0
    full = _fill(corr, light, raw, HYBRID, origin_yx=(0, 0))
    assert (full < corr).sum() > 1000
    margin = support_px(HYBRID)["total"] + 2
    for y0, y1, x0, x1 in [(16, 704, 48, 1024), (32, 688, 16, 976), (0, 704, 80, 1008)]:
        sub = _fill(
            corr[:, y0:y1, x0:x1],
            light[:, y0:y1, x0:x1],
            raw[:, y0:y1, x0:x1],
            HYBRID,
            origin_yx=(y0, x0),
        )
        ya, yb = (margin if y0 > 0 else 0), (y1 - y0 - (margin if y1 < size[0] else 0))
        xa, xb = (margin if x0 > 0 else 0), (x1 - x0 - (margin if x1 < size[1] else 0))
        a = sub[:, ya:yb, xa:xb]
        b = full[:, y0 + ya : y0 + yb, x0 + xa : x0 + xb]
        assert np.abs(a - b).max() == 0.0, (y0, y1, x0, x1)
    # The origin is what aligns the blocks: a wrong origin changes the result.
    window = (slice(None), slice(16, None), slice(48, None))
    wrong = _fill(corr[window], light[window], raw[window], HYBRID, origin_yx=(0, 0))
    right = _fill(corr[window], light[window], raw[window], HYBRID, origin_yx=(16, 48))
    assert not np.array_equal(wrong, right)


def test_h5_flat_hash_is_unchanged_and_the_hybrid_support_fits() -> None:
    assert fill_params_sha256(FillParams()) == FLAT_DEFAULT_SHA256
    assert fill_params_sha256(FillParams(enabled=True)) == FLAT_ENABLED_SHA256
    # The texture fields do not matter in flat mode (and are not hashed there).
    assert fill_params_sha256(FillParams(texture_block=64)) == FLAT_DEFAULT_SHA256
    assert support_px(FillParams()) == {
        "fine": 32,
        "cover": 136,
        "reach": 64,
        "coarse": 144,
        "total": 232,
    }
    # Hybrid has its own hash, which follows its texture fields.
    assert fill_params_sha256(HYBRID) != FLAT_ENABLED_SHA256
    assert fill_params_sha256(HYBRID) != fill_params_sha256(
        dataclasses.replace(HYBRID, texture_hp_sigma=5.0)
    )
    # High-pass radius 24 + (1 + 1) blocks of 32 px behind the 144 px coarse path.
    assert support_px(HYBRID) == {
        "fine": 32,
        "cover": 0,
        "reach": 88,
        "coarse": 144,
        "total": 232,
    }
    assert support_px(HYBRID)["total"] <= MAX_SUPPORT_PX
    # The original smoothing (4 blocks, 128 px) would not fit: it is refused.
    for radius in (2, 4):
        with pytest.raises(ValueError, match="support"):
            FillParams(mode="hybrid", texture_radius_blocks=radius)
    assert (
        support_px(dataclasses.replace(HYBRID, texture_radius_blocks=0))["total"] < 232
    )
    # In flat mode the texture fields are not read, so they do not count.
    assert FillParams(texture_radius_blocks=4).mode == "flat"


def test_h6_hybrid_uses_its_own_effective_gates() -> None:
    """The flat fields are ignored in hybrid mode and keep their flat defaults."""
    flat = FillParams()
    assert (flat.flat_max, flat.deadband, flat.peak_snr, flat.max_cover) == (
        1.0,
        2.0,
        3.0,
        0.6,
    )
    assert flat.mode == "flat"
    corr, light, raw = make_textured_planes()
    base = _fill(corr, light, raw, HYBRID)
    other = FillParams(
        enabled=True,
        mode="hybrid",
        flat_max=1e-6,
        deadband=5.0,
        peak_snr=10.0,
        bright_reach_px=8,
        max_cover=0.05,
    )
    assert np.array_equal(base, _fill(corr, light, raw, other))
    # Hard one-sided clip: noise on a flat background is clipped, unlike flat mode.
    rng = np.random.default_rng(4)
    flat_light = make_light((512, 512))
    obs = (1800 + rng.normal(0, 37, flat_light.shape)).astype(np.float32)
    hard = _fill(obs, flat_light, obs + flat_light, HYBRID)
    soft = _fill(obs, flat_light, obs + flat_light)
    assert (obs - hard).sum() > 5 * (obs - soft).sum()


def test_h7_hybrid_leaves_invalid_pixels_and_stains_alone() -> None:
    corr, light, raw = make_textured_planes()
    corr[:, 200:320, 200:260] = 0
    raw[:, 200:320, 200:260] = 0
    out = _fill(corr, light, raw, HYBRID)
    assert (out[:, 200:320, 200:260] == 0).all()
    assert (out <= corr).all()
    assert (out < corr).sum() > 1000
    # A stain the fill is not configured for is returned as is.
    other = apply_bounded_fill(
        corr, light, raw=raw, z_layers=Z, config=HYBRID, stain="DAPI", noise=NOISE
    )
    assert other is corr
    # Disabled stays bit-identical in hybrid mode too.
    off = dataclasses.replace(HYBRID, enabled=False)
    assert _fill(corr, light, raw, off) is corr


def test_h8_hybrid_origin_and_mode_validation() -> None:
    corr, light, raw = make_textured_planes()
    with pytest.raises(ValueError, match="origin_yx"):
        _fill(corr, light, raw, HYBRID, origin_yx=(4, 0))
    # Flat mode does not read the origin.
    assert np.array_equal(
        _fill(corr, light, raw, origin_yx=(4, 0)), _fill(corr, light, raw)
    )
    with pytest.raises(ValueError, match="mode"):
        FillParams(mode="other")  # type: ignore[arg-type]
    for bad in (
        {"texture_block": 0},
        {"texture_hp_sigma": 0.0},
        {"texture_smooth": 0.0},
        {"texture_radius_blocks": -1},
    ):
        with pytest.raises(ValueError, match="texture"):
            FillParams(mode="hybrid", **bad)


def test_h9_correct_window_passes_the_absolute_origin_to_the_hybrid_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a mode with absolute blocks gets the origin; flat settings never do."""
    calls: list[dict[str, Any]] = []

    def fake(corrected: np.ndarray, model_light: np.ndarray, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return corrected

    module = types.ModuleType("merxen.io.bead_removal.fill")
    module.apply_bounded_fill = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.bead_removal.fill", module)
    cal = bs.make_calibration()
    st = bs.make_stack(seed=1, n_beads=15, size=512)
    stacks = {k: st[k] for k in ("PolyT", "DAPI")}
    for settings, expected in (
        (types.SimpleNamespace(enabled=True, stains=("PolyT",)), None),
        (types.SimpleNamespace(enabled=True, stains=("PolyT",), mode="flat"), None),
        (
            types.SimpleNamespace(enabled=True, stains=("PolyT",), mode="hybrid"),
            (16, 32),
        ),
    ):
        calls.clear()
        correct_window(
            stacks,
            origin_yx=(21, 37),  # padded down to the 16 px grid: (16, 32)
            calibration=cal,
            z_layers=Z,
            fill=settings,
        )
        assert len(calls) == 1
        assert calls[0].get("origin_yx") == expected


def test_h10_hybrid_correct_window_is_never_brighter_than_the_subtraction() -> None:
    cal = bs.make_calibration()
    st = bs.make_stack(seed=1, n_beads=15, size=512)
    stacks = {k: st[k] for k in ("PolyT", "DAPI")}

    def run(**kwargs: Any) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
        return correct_window(
            stacks, origin_yx=(16, 48), calibration=cal, z_layers=Z, **kwargs
        )

    base, _, _ = run()
    hybrid, _, stats = run(fill=HYBRID)
    for stain in base:
        assert (hybrid[stain] <= base[stain]).all()
    assert set(stats["fill"]) == {"PolyT"}
    assert stats["fill"]["PolyT"]["cand_px"] > 0
    assert (hybrid["PolyT"] < base["PolyT"]).any()
