"""Tests of the bead calibration pass and its building blocks.

Synthetic data only. The mosaic tests write one large synthetic stack (beads on a flat
background, camera noise) with tissue painted as a smooth DAPI / PolyT plateau over
some of the 4 x 4 grid cells of 768 px; planes 0..3 (z_layers 1-3), stains PolyT, DAPI
and Anti-X. Tests that build such a mosaic are marked slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from merxen.config import MerscopeBeadRemovalConfig
from merxen.io.bead_removal import calibrate as calibrate_module
from merxen.io.bead_removal.calibrate import (
    FORECAST_CPU_S_PER_BEAD,
    FORECAST_GPU_POINTS,
    MIN_RECALL_REFERENCE,
    MIN_RECALL_WITH_CLUSTERS_REFERENCE,
    REFERENCE_GRID_N,
    BeadCalibration,
    BeadCalibrationError,
    _dedupe,
    _dedupe_parts,
    build_overview,
    calibrate_bead_model,
    candidate_tiles,
    checkerboard_role,
    classifier_recall_warnings,
    confirmed_mask,
    count_windows,
    derive_classifier_thresholds,
    focus_check,
    forecast_cost,
    gpu_window_seconds,
    greedy_select,
    library_tile_spread,
    off_tissue_distance,
    pool_rho,
    ring_tscores,
    select_tile_sets,
    select_tile_sets_detail,
    tissue_mask_from_overview,
)
from merxen.io.bead_removal.detect import ClassifierThresholds
from merxen.io.bead_removal.params import DEFAULT_PARAMS
from merxen.io.bead_removal.reader import MosaicPlaneReader

from . import bead_synth as bs

TILE = 768
GRID = 4
SIZE = TILE * GRID
ZL = [1, 2, 3]
CFG = MerscopeBeadRemovalConfig(
    enabled=True, calibration_tile_px=TILE, min_offtissue_distance_px=64
)


def _tissue_cells(kind: str) -> np.ndarray:
    """Boolean [GRID, GRID] map of the grid cells covered by tissue."""
    t = np.zeros((GRID, GRID), bool)
    if kind == "centre":
        t[1:3, 1:3] = True
    elif kind == "tight":  # tissue everywhere except two opposite corner cells
        t[:] = True
        t[0, 0] = t[GRID - 1, GRID - 1] = False
    return t


def build_mosaic(
    path: Path,
    *,
    n_beads: int,
    focus_plane: int = 0,
    tissue: str = "centre",
    seed: int = 3,
    dapi_ratio: float = 0.11,
) -> np.ndarray:
    """Write the mosaic planes (PolyT, DAPI, Anti-X; planes 0..3) of a 3072 px mosaic.

    Returns:
        The tissue truth mask.
    """
    import tifffile

    st = bs.make_stack(
        n_planes=4,
        size=SIZE,
        n_beads=n_beads,
        seed=seed,
        tissue=False,
        focus_plane=focus_plane,
        dapi_ratio=dapi_ratio,
        blobs=False,
    )
    mask = np.kron(_tissue_cells(tissue), np.ones((TILE, TILE), bool))
    rng = np.random.default_rng(seed + 100)
    plateau = gaussian_filter(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), 6)
    plateau *= 800.0 / max(float(plateau.std()), 1e-6)
    for k in range(4):
        st["DAPI"][k][mask] += 3000.0 + plateau[mask]
        st["PolyT"][k][mask] += 300.0 + 0.3 * plateau[mask]
    for stain in ("PolyT", "DAPI", "Anti-X"):
        for k in range(4):
            tifffile.imwrite(
                path / f"mosaic_{stain}_z{k}.tif",
                np.clip(np.rint(st[stain][k]), 1, 65535).astype(np.uint16),
            )
    return mask


def _reader(path: Path) -> MosaicPlaneReader:
    return MosaicPlaneReader(path, ["PolyT", "DAPI", "Anti-X"], [0, 1, 2, 3])


@pytest.fixture(scope="module")
def mosaic_beads(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    d = tmp_path_factory.mktemp("mosaic_beads")
    return d, build_mosaic(d, n_beads=540)


@pytest.fixture(scope="module")
def mosaic_free(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("mosaic_free")
    build_mosaic(d, n_beads=0)
    return d


@pytest.fixture(scope="module")
def mosaic_focus1(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("mosaic_focus1")
    build_mosaic(d, n_beads=540, focus_plane=1)
    return d


@pytest.fixture(scope="module")
def mosaic_tight(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("mosaic_tight")
    build_mosaic(d, n_beads=540, tissue="tight")
    return d


# ---------------------------------------------------------------------------
# T13: overview, tissue mask, tile choice
@pytest.mark.slow
def test_t13_overview_and_tissue_mask(
    mosaic_beads: tuple[Path, np.ndarray],
) -> None:
    d, truth = mosaic_beads
    with _reader(d) as rd:
        dapi_ov, valid_ov = build_overview(
            rd, reference_stain="PolyT", dapi_stain="DAPI", z_mid=2
        )
        mask, info = tissue_mask_from_overview(dapi_ov, valid_ov)
    assert valid_ov.all() and dapi_ov.shape == (SIZE // 16, SIZE // 16)
    truth_ov = truth[8::16, 8::16]
    iou = (mask & truth_ov).sum() / (mask | truth_ov).sum()
    assert iou >= 0.9, iou
    assert abs(info["tissue_fraction"] - truth.mean()) < 0.05
    assert 600 < info["threshold_counts"] < 3500
    dist = off_tissue_distance(mask, valid_ov)
    assert dist[~truth_ov].max() > 200 and (dist[mask] == 0).all()


def test_tissue_mask_ignores_invalid_border() -> None:
    """Zeros outside the mosaic neither become tissue nor darken tissue at the edge."""
    dapi = np.full((120, 160), 500.0, np.float32)
    dapi[:, :40] = 0
    dapi[30:90, 40:100] = 4000.0  # tissue touching the invalid margin
    valid = dapi > 0
    valid[:, :40] = False
    mask, _ = tissue_mask_from_overview(dapi, valid)
    truth = np.zeros_like(mask)
    truth[30:90, 40:100] = True
    assert (mask & truth).sum() / (mask | truth).sum() > 0.9
    assert not mask[:, :40].any()


def test_candidate_tiles_and_greedy_selection() -> None:
    valid = np.ones((256, 256), bool)
    tissue = np.zeros_like(valid)
    tissue[100:160, 100:160] = True
    d_off = off_tissue_distance(tissue, valid)
    cand = candidate_tiles(
        valid, d_off, shape_px=(4096, 4096), tile_px=1024, min_dist_px=200
    )
    assert len(cand["y0"]) == 49  # 7 x 7 positions at stride 512
    inside = (
        (cand["y0"] < 2560)
        & (cand["y0"] + 1024 > 1600)
        & (cand["x0"] < 2560)
        & (cand["x0"] + 1024 > 1600)
    )
    assert (cand["off_frac"][inside] < 0.95).all()
    assert cand["off_frac"][(cand["y0"] == 0) & (cand["x0"] == 0)][0] == 1.0
    centres = np.array([[512, 512], [512, 1024], [3584, 3584], [512, 3584]], float)
    assert greedy_select(centres, [10, 50, 40, 5], 3, 2048.0) == [1, 2, 3]
    # ties go to the better-ranked tile
    assert greedy_select(centres, [10, 10, 10, 10], 6, 2048.0) == [0, 2, 3]


def test_pool_rho_rules() -> None:
    p = DEFAULT_PARAMS
    pooled = pool_rho([(0.11, 0.003, 100), (0.115, 0.004, 90), (0.108, 0.003, 80)], p)
    assert pooled[0] == pytest.approx(0.11, abs=0.005)
    assert pool_rho([(0.01, 0.001, 100)] * 3, p)[0] == 0.0  # below rho_min
    assert pool_rho([(0.05, 0.03, 100)] * 3, p)[0] == 0.0  # not significant
    assert pool_rho([(0.0, float("inf"), 3)], p)[0] == 0.0


# ---------------------------------------------------------------------------
# T12: the calibration pass on mosaics
@pytest.mark.slow
def test_t12_calibration_with_beads(
    mosaic_beads: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    d, _ = mosaic_beads
    with _reader(d) as rd:
        cal = calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)
    dec = cal.decision
    assert dec["status"] == "ok"
    assert dec["n_pass"] >= dec["required"] >= 3 and dec["margin_pass"] > 0
    assert dec["focus"]["passed"] and dec["focus"]["median_r1"] < 0.3
    th = cal.thresholds
    assert 0.20 <= th.ring_min <= 0.30 and 0.25 <= th.r1_max <= 0.35
    assert 0.05 <= th.rd_center <= 0.20 and 0.05 <= th.rd0_center <= 0.20
    assert abs(cal.rho["DAPI"][0] - 0.11) < 0.02
    assert cal.rho["Anti-X"][0] == 0.0
    cls = dec["classifier"]
    assert cls["n_reference"] >= 50 and cls["recall"] >= 0.9
    assert len(cal.library.tmpls) == 8
    s = cal.library.summary()
    # the rescue is derived with the thresholds: the texture gate from the reference
    # set, the z-profile limited to the planes the mosaic has (0..3 here)
    assert th.rescue_active and th.rescue_tz1_min == 4.0
    assert th.rescue_ring_radius_px == pytest.approx(s[1]["ring_radius_px"])
    assert 0.02 <= th.rescue_sd1_max <= 0.08 and th.rescue_zprof_zmax == 3.0
    assert cls["n_sd1_reference"] == cls["n_reference"]
    assert cls["sd1_rel_p98_raw"] > 0
    rescued = [t["n_rescued"] for t in cls["tiles"]]
    assert cls["n_rescued_tiles"] == sum(rescued)
    assert 0.0 <= cls["rescued_fraction_tiles"] <= 1.0
    assert abs(s[1]["ring_radius_px"] - 29.8) < 3.0
    assert abs(s[3]["ring_radius_px"] - 75.4) < 6.0
    assert set(cal.noise) == {0, 1, 2, 3}
    assert all(15 < v < 25 for v in cal.noise.values())
    assert cal.a0_sel[0] < cal.a0_median < cal.a0_sel[1]
    assert abs(cal.ell_default) < 0.1
    assert len(cal.tiles) >= 3 and all(t["tile_px"] == TILE for t in cal.tiles)
    assert cal.forecast["backend"] == "cpu"
    # the library's tile-to-tile spread is recorded, one row per passing tile
    spread = dec["library_spread"]
    assert spread["n_tiles"] == dec["n_pass"] == len(spread["per_tile"])
    assert spread["summary"]["z1_ring_radius_px"]["min"] > 0
    # round trip, including the per-tile templates kept for the QC picture
    cal.save(tmp_path / "cal")
    back = BeadCalibration.load(tmp_path / "cal")
    assert back.thresholds == cal.thresholds and back.rho == cal.rho
    assert back.calibration_sha256() == cal.calibration_sha256()
    assert np.allclose(back.library.tmpls[2].vals, cal.library.tmpls[2].vals)
    assert set(back.library.info["tile_templates"]) == set(
        cal.library.info["tile_templates"]
    )
    # the result does not depend on the worker count
    with _reader(d) as rd:
        cal1 = calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=1)
    assert cal1.thresholds == cal.thresholds and cal1.rho == cal.rho
    assert np.array_equal(cal1.library.tmpls[1].vals, cal.library.tmpls[1].vals)
    assert cal1.calibration_sha256() == cal.calibration_sha256()


@pytest.mark.slow
def test_t12_backend_changes_only_the_forecast(
    mosaic_beads: tuple[Path, np.ndarray],
) -> None:
    d, _ = mosaic_beads
    with _reader(d) as rd:
        cpu = calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)
        gpu = calibrate_bead_model(
            rd, config=CFG, z_layers=ZL, n_workers=2, backend="cuda"
        )
    assert gpu.calibration_sha256() == cpu.calibration_sha256()
    assert cpu.forecast["backend"] == "cpu" and gpu.forecast["backend"] == "cuda"
    assert gpu.forecast["gpu_wall_hours"] == pytest.approx(gpu.forecast["wall_hours"])


@pytest.mark.slow
def test_t12_no_beads(mosaic_free: Path) -> None:
    with (
        _reader(mosaic_free) as rd,
        pytest.raises(BeadCalibrationError, match="no beads") as ei,
    ):
        calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)
    assert len(ei.value.report["tiles"]) >= 3
    assert "merscope_bead_removal false" in str(ei.value)
    json.dumps(ei.value.report)  # the per-tile report is JSON-able


@pytest.mark.slow
def test_t12_not_focused(mosaic_focus1: Path) -> None:
    with (
        _reader(mosaic_focus1) as rd,
        pytest.raises(BeadCalibrationError, match="plane"),
    ):
        calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)


@pytest.mark.slow
def test_t12_no_tissue_free_area(mosaic_tight: Path) -> None:
    with (
        _reader(mosaic_tight) as rd,
        pytest.raises(BeadCalibrationError, match="tissue-free"),
    ):
        calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)


def test_z_layers_validation(tmp_path: Path) -> None:
    import tifffile

    for stain in ("PolyT", "DAPI"):
        for k in range(4):
            tifffile.imwrite(
                tmp_path / f"mosaic_{stain}_z{k}.tif", np.full((64, 64), 5, np.uint16)
            )
    rd = MosaicPlaneReader(tmp_path, ["PolyT", "DAPI"], [0, 1, 2, 3])
    for bad in ([0, 1, 2], [], [1, 8]):
        with pytest.raises(ValueError, match="plane"):
            calibrate_bead_model(rd, config=CFG, z_layers=bad)
    with pytest.raises(BeadCalibrationError, match="missing"):
        calibrate_bead_model(rd, config=CFG, z_layers=[1, 2, 3, 4])


def test_backend_and_tile_offset_are_validated() -> None:
    with pytest.raises(ValueError, match="backend"):
        calibrate_bead_model(None, config=CFG, z_layers=ZL, backend="auto")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="_tile_offset"):
        # not a multiple of calibration_tiles (6)
        calibrate_bead_model(None, config=CFG, z_layers=ZL, _tile_offset=4)
    with pytest.raises(ValueError, match="_tile_offset"):
        calibrate_bead_model(None, config=CFG, z_layers=ZL, _tile_offset=-6)


# ---------------------------------------------------------------------------
# classifier reference set, focus check, held-out recall warnings
def _ref(r1: list[float], *, spectral: bool = True, isolated: bool = True) -> dict:
    n = len(r1)
    return {
        "y": np.arange(n, dtype=float),
        "x": np.zeros(n),
        "r1": np.asarray(r1, float),
        "spectral": np.full(n, spectral),
        "isolated": np.full(n, isolated),
    }


def test_focus_check_is_measured_on_permissive_candidates() -> None:
    """Candidates the classifier's own r1 < r1_max test would reject must count."""
    sharp_at_z1 = [_ref([0.4] * 60 + [0.1] * 20)]  # 75 % have a strong z1 centre
    f = focus_check(sharp_at_z1, z0_centre=1.3, z1_centre=0.2)
    assert not f["passed"] and f["n_bright"] == 80
    assert f["median_r1"] == pytest.approx(0.4)
    good = focus_check([_ref([0.1, 0.12, 0.2, 0.35, 0.11])], 1.3, 0.2)
    assert good["passed"] and good["median_r1"] == pytest.approx(0.12)
    # non-spectral candidates (tissue) are not counted; the template test still applies
    assert focus_check([_ref([0.9] * 30, spectral=False)], 1.3, 0.2)["passed"]
    assert not focus_check([_ref([0.1] * 30)], z0_centre=0.2, z1_centre=1.3)["passed"]
    assert focus_check([], 1.3, 0.2)["n_bright"] == 0


def test_checkerboard_roles_and_dedupe() -> None:
    roles = [[checkerboard_role(i, j) for j in range(4)] for i in range(4)]
    assert roles[0] == ["threshold", "recall", "threshold", "recall"]
    assert roles[1][0] == "recall"
    assert sum(r == "recall" for row in roles for r in row) == 8
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [50.0, 50.0], [51.0, 50.0], [200.0, 0.0]])
    assert _dedupe(pts).tolist() == [True, False, True, False, True]
    against = np.array([[200.5, 0.5]])
    assert _dedupe(pts, against=against).tolist() == [True, False, True, False, False]
    assert _dedupe(np.zeros((0, 2))).shape == (0,)


def test_confirmed_mask_and_dedupe_parts() -> None:
    r = {
        "y": np.arange(4.0),
        "x": np.zeros(4),
        "isolated": np.array([True, False, True, True]),
        "spectral": np.array([True, True, False, True]),
        "tz1": np.array([5.0, 5.0, 5.0, 2.0]),
    }
    assert confirmed_mask(r).tolist() == [True, False, False, False]
    assert confirmed_mask(r, require_isolated=False).tolist() == [
        True,
        True,
        False,
        False,
    ]
    empty = {
        "y": np.zeros(0),
        "x": np.zeros(0),
        "isolated": np.zeros(0, bool),
        "spectral": np.zeros(0, bool),
        "tz1": np.zeros(0),
    }
    assert confirmed_mask(empty).shape == (0,)
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[10.5, 10.0], [90.0, 90.0]])
    ka, kb = _dedupe_parts([a, b])
    assert ka.tolist() == [True, True]
    assert kb.tolist() == [False, True]  # the later tile's copy of a bead is dropped
    empty_parts = _dedupe_parts([np.zeros((0, 2)), np.zeros((0, 2))])
    assert [k.shape for k in empty_parts] == [(0,), (0,)]


def test_derive_classifier_thresholds_clamps_and_keeps_defaults() -> None:
    rng = np.random.default_rng(0)

    def ref(n: int, ring_lo: float, r1_hi: float) -> dict[str, np.ndarray]:
        return {
            "rring": rng.uniform(ring_lo, 0.9, n),
            "r1": rng.uniform(0.0, r1_hi, n),
            "rD0": np.full(n, 0.11),
            "rD": np.full(n, 0.10),
        }

    default = ClassifierThresholds.from_params(DEFAULT_PARAMS)
    th, info = derive_classifier_thresholds(ref(10, 0.4, 0.2))
    assert th == default and info["used_defaults"] and info["n_reference"] == 10
    # a ring ratio far below the allowed range is clamped to the lower limit
    th, info = derive_classifier_thresholds(ref(200, 0.05, 0.2))
    assert th.ring_min == pytest.approx(0.20) and "ring_min" in info["clamped"]
    assert 0.25 <= th.r1_max <= 0.35 and "r1_max" in info["clamped"]
    assert th.rd0_center == pytest.approx(0.11) and "rd0_center" not in info["clamped"]
    assert not info["used_defaults"] and "variant" not in info


def test_ring_tscores_separate_rings_from_flat_background() -> None:
    rng = np.random.default_rng(1)
    img = rng.normal(0, 5.0, (200, 200))
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(yy - 100.0, xx - 100.0)
    img += 60.0 * np.exp(-((r - 30.0) ** 2) / (2 * 3.0**2))
    ys = np.array([100.0, 50.0, 3.0])
    xs = np.array([100.0, 150.0, 100.0])
    t = ring_tscores(img, ys, xs, 30.0)
    assert t[0] > 10 and abs(t[1]) < 4  # ring present / flat background
    assert np.isnan(t[2])  # the window leaves the image
    # a non-finite pixel in the window gives NaN, not a score
    img2 = img.copy()
    img2[100, 100 + 20] = np.nan
    assert np.isnan(ring_tscores(img2, ys[:1], xs[:1], 30.0)[0])


def test_classifier_recall_warnings_follow_the_owner_decision() -> None:
    nan = float("nan")
    # cluster-inclusive recall below 0.75 on at least 50 beads warns
    w = classifier_recall_warnings(0.667, 6, 0.564, 188)
    assert len(w) == 1 and "including beads in clusters" in w[0] and "0.564" in w[0]
    # ... but not on fewer than 50 beads, nor at or above 0.75
    assert (
        classifier_recall_warnings(nan, 0, 0.5, MIN_RECALL_WITH_CLUSTERS_REFERENCE - 1)
        == []
    )
    assert classifier_recall_warnings(0.95, 60, 0.75, 188) == []
    # the isolated recall warns only below 0.90 and on at least 20 beads
    w = classifier_recall_warnings(0.85, MIN_RECALL_REFERENCE, 0.80, 40)
    assert len(w) == 1 and "held-out classifier recall is 0.850" in w[0]
    assert classifier_recall_warnings(0.85, MIN_RECALL_REFERENCE - 1, 0.80, 40) == []
    assert classifier_recall_warnings(0.90, 100, 0.90, 100) == []
    # both low: both warn, the cluster-inclusive one first
    w = classifier_recall_warnings(0.5, 30, 0.4, 80)
    assert len(w) == 2 and "clusters" in w[0] and "isolated" in w[1]
    # nothing to measure
    w = classifier_recall_warnings(nan, 0, nan, 0)
    assert len(w) == 1 and "could not be measured" in w[0]


# ---------------------------------------------------------------------------
# tile sets and the _tile_offset hook
def test_select_tile_sets_disjoint_and_first_set_is_production_choice() -> None:
    gy, gx = np.meshgrid(np.arange(6) * 512.0, np.arange(6) * 512.0, indexing="ij")
    centres = np.c_[gy.ravel(), gx.ravel()] + 512.0
    counts = list(np.random.default_rng(0).integers(10, 100, len(centres)))
    detail = select_tile_sets_detail(centres, counts, 3, 1024.0, tile_px=1024, n_sets=2)
    assert detail[0] == (greedy_select(centres, counts, 3, 1024.0), 0.0)
    (first, _), (second, level) = detail
    assert level == 0.0 and len(second) == 3  # the second set shares no pixel
    for i in second:
        for j in first:
            assert (
                abs(centres[i, 0] - centres[j, 0]) >= 1024
                or abs(centres[i, 1] - centres[j, 1]) >= 1024
            )
    sets = select_tile_sets(centres, counts, 3, 1024.0, tile_px=1024, n_sets=2)
    assert sets == [first, second]


def test_select_tile_sets_accepts_overlap_only_when_the_area_is_too_small() -> None:
    """One row of half-overlapping tiles holds no two disjoint sets of three."""
    centres = np.c_[np.full(10, 512.0), 512.0 + 512.0 * np.arange(10)]
    counts = list(range(10, 20))
    (first, l0), (second, l1) = select_tile_sets_detail(
        centres, counts, 3, 1024.0, tile_px=1024, n_sets=2
    )
    assert l0 == 0.0 and len(first) == 3 and len(second) >= 3
    assert not set(first) & set(second)
    assert 0.0 < l1 <= 0.5  # neighbours of an earlier tile overlap it by half
    sets3 = select_tile_sets_detail(centres, counts, 3, 1024.0, tile_px=1024, n_sets=3)
    assert len(sets3) == 3
    chosen = [i for s, _ in sets3 for i in s]
    assert len(set(chosen)) == len(chosen)


def test_reference_grid_is_the_sample_grid_by_default() -> None:
    assert REFERENCE_GRID_N == 4


# ---------------------------------------------------------------------------
# cost forecast
def test_forecast_models() -> None:
    kw = {"density_per_px2": 1.0e-4, "valid_area_px": 2.0e9, "px_per_um": 9.2593}
    f5 = forecast_cost(n_workers=5, **kw)
    assert f5["n_beads_total"] == pytest.approx(2.0e5)
    assert f5["n_beads_fitted"] == pytest.approx(2.46e5)
    assert f5["cpu_s_per_bead"] == pytest.approx(0.106)
    assert f5["cpu_hours"] == pytest.approx(2.0e5 * 0.106 / 3600)
    assert f5["cpu_wall_hours"] == pytest.approx(2.0e5 * 0.106 / 5 / 0.94 / 3600)
    # the cost per bead rises with the worker count and is interpolated
    assert forecast_cost(n_workers=2, **kw)["cpu_s_per_bead"] == pytest.approx(0.106)
    assert 0.116 < forecast_cost(n_workers=12, **kw)["cpu_s_per_bead"] < 0.25
    f16 = forecast_cost(n_workers=16, **kw)
    assert f16["cpu_s_per_bead"] == pytest.approx(dict(FORECAST_CPU_S_PER_BEAD)[16])
    # GPU: a + b x beads, exact at the two measured points
    (x0, t0), (x1, t1) = FORECAST_GPU_POINTS
    assert gpu_window_seconds(x0) == pytest.approx(t0)
    assert gpu_window_seconds(x1) == pytest.approx(t1)
    g = forecast_cost(
        n_workers=5, n_windows=150, calibration_s=120.0, backend="cuda", **kw
    )
    assert g["backend"] == "cuda"
    assert g["wall_hours"] == pytest.approx(g["gpu_wall_hours"])
    assert g["gpu_wall_hours"] < g["cpu_wall_hours"]
    expected = (150 * gpu_window_seconds(2.46e5 / 150) + 120.0) / 3600
    assert g["gpu_wall_hours"] == pytest.approx(expected)
    assert (
        forecast_cost(n_workers=5, n_windows=150, backend="cpu", **kw)["backend"]
        == "cpu"
    )
    # no window count: only the CPU figure exists
    assert forecast_cost(n_workers=5, backend="cuda", **kw)["backend"] == "cpu"
    # WARN when the forecast of the requested backend exceeds 12 h
    dense = {"density_per_px2": 7.0e-4, "valid_area_px": 3.0e9, "px_per_um": 9.2593}
    assert forecast_cost(n_workers=5, **dense)["warn"]
    assert not forecast_cost(n_workers=5, n_windows=300, backend="cuda", **dense)[
        "warn"
    ]


def test_count_windows_uses_the_chunk_grid() -> None:
    valid = np.zeros((600, 600), bool)  # 1/16 overview: 256 px per 4096 px core
    valid[0, 0] = True
    assert count_windows(valid) == 1
    valid[300, 300] = valid[599, 599] = True
    assert count_windows(valid) == 3
    assert count_windows(np.zeros((10, 10), bool)) == 0


# ---------------------------------------------------------------------------
# the calibration record: hash, round trip, library spread
def _synthetic_calibration() -> BeadCalibration:
    syn = bs.make_calibration()
    return BeadCalibration(
        version=syn.version,
        params_sha256=syn.params_sha256,
        corrected_stains=list(syn.corrected_stains),
        reference_stain=syn.reference_stain,
        library=syn.library,
        thresholds=syn.thresholds,
        noise=dict(syn.noise),
        dog_noise_z0=syn.dog_noise_z0,
        a0_sel=syn.a0_sel,
        a0_median=syn.a0_median,
        ell_default=syn.ell_default,
        rho=dict(syn.rho),
        tiles=[{"index": 0, "y0": 0, "x0": 0, "time_s": 1.5}],
        decision={"status": "synthetic", "timing_s": {"total_s": 2.0}},
    )


def test_calibration_save_load_round_trip_and_hash(tmp_path: Path) -> None:
    cal = _synthetic_calibration()
    cal.save(tmp_path / "cal")
    assert (tmp_path / "cal" / "calibration.json").exists()
    assert (tmp_path / "cal" / "psf_library.npz").exists()
    j = json.loads((tmp_path / "cal" / "calibration.json").read_text())
    assert j["calibration_sha256"] == cal.calibration_sha256()
    back = BeadCalibration.load(tmp_path / "cal")
    assert back.thresholds == cal.thresholds
    assert back.rho == cal.rho and back.noise == cal.noise
    assert back.a0_sel == cal.a0_sel and back.corrected_stains == cal.corrected_stains
    assert back.tiles == cal.tiles and back.decision == cal.decision
    for a, b in zip(back.library.tmpls, cal.library.tmpls, strict=True):
        assert np.array_equal(a.vals, b.vals)
    assert back.calibration_sha256() == cal.calibration_sha256()


def test_calibration_hash_covers_the_model_and_ignores_qc_records() -> None:
    cal = _synthetic_calibration()
    base = cal.calibration_sha256()
    assert len(base) == 64
    qc_changed = BeadCalibration(
        **{
            **cal.__dict__,
            "tiles": [],
            "decision": {"warnings": ["x"]},
            "tissue": {"a": 1},
            "forecast": {"wall_hours": 3.0},
        }
    )
    assert qc_changed.calibration_sha256() == base
    for field, value in (
        ("ell_default", cal.ell_default + 0.01),
        ("dog_noise_z0", cal.dog_noise_z0 * 1.01),
        ("a0_sel", (cal.a0_sel[0] + 1.0, cal.a0_sel[1])),
        ("rho", {**cal.rho, "DAPI": (0.5, 0.01)}),
        ("noise", {**cal.noise, 0: cal.noise[0] + 0.1}),
        ("corrected_stains", ["PolyT"]),
        ("thresholds", ClassifierThresholds(ring_min=0.25)),
    ):
        changed = BeadCalibration(**{**cal.__dict__, field: value})
        assert changed.calibration_sha256() != base, field
    lib = bs.library_from_generator()
    lib.tmpls[2].vals = lib.tmpls[2].vals * 1.001
    assert (
        BeadCalibration(**{**cal.__dict__, "library": lib}).calibration_sha256() != base
    )


def test_library_tile_spread_reports_per_tile_numbers() -> None:
    lib = bs.library_from_generator()
    scales = {0: 1.0, 1: 1.1, 2: 0.9}
    tile_templates = {
        i: {k: lib.tmpls[k].vals * (s if k > 0 else 1.0) for k in range(4)}
        for i, s in scales.items()
    }
    spread = library_tile_spread(tile_templates, lib, DEFAULT_PARAMS)
    assert spread["n_tiles"] == 3 and spread["planes"] == [1, 2, 3]
    assert [r["tile_index"] for r in spread["per_tile"]] == [0, 1, 2]
    # the ring radius does not depend on the scale; the flux scales with it
    radii = {r["z1_ring_radius_px"] for r in spread["per_tile"]}
    assert len(radii) == 1
    flux = [r["z1_flux_rel"] for r in spread["per_tile"]]
    assert flux[1] / flux[0] == pytest.approx(1.1) and flux[2] / flux[
        0
    ] == pytest.approx(0.9)
    summ = spread["summary"]["z1_flux_rel"]
    assert summ["min"] == pytest.approx(flux[2]) and summ["max"] == pytest.approx(
        flux[1]
    )
    assert summ["range"] == pytest.approx(flux[1] - flux[2])
    assert summ["relative_range"] == pytest.approx(summ["range"] / summ["median"])
    assert spread["summary"]["z1_ring_radius_px"]["range"] == 0.0
    assert set(spread["pooled"]) == {
        f"z{k}_{q}" for k in (1, 2, 3) for q in ("ring_radius_px", "flux_rel")
    }
    json.dumps(spread)


# ---------------------------------------------------------------------------
# held-out recall on mosaics whose bead focus changes across x
def _add_z0(img: np.ndarray, yc: float, xc: float, amp: float, ring_w: float) -> None:
    """Add a z0 stand-in bead: a sharp dot plus a ring at 10.7 px.

    ``ring_w`` is the relative weight of the ring; 0.46 is the library bead.
    """
    half = 40
    y0, y1 = max(0, int(yc) - half), min(img.shape[0], int(yc) + half + 1)
    x0, x1 = max(0, int(xc) - half), min(img.shape[1], int(xc) + half + 1)
    r = np.hypot(np.arange(y0, y1)[:, None] - yc, np.arange(x0, x1)[None, :] - xc)
    img[y0:y1, x0:x1] += (
        amp
        * (
            np.exp(-(r**2) / (2 * 1.5**2))
            + ring_w * np.exp(-((r - 10.7) ** 2) / (2 * 2.0**2))
        )
    ).astype(np.float32)


# The sample grid has 1024 px tiles: edge tiles of a smaller mosaic fall partly outside
# it and are skipped.
GSIZE = 4096


def build_gradient_mosaic(
    path: Path, *, ring_w_right: float, n_beads: int = 1400, seed: int = 5
) -> None:
    """Write a mosaic whose z0 ring weight falls linearly across x.

    The weight goes from 0.46 (left edge) to ``ring_w_right`` (right edge). A weaker
    ring at plane 0 gives a lower ring / centre ratio (the beads are focused slightly
    differently on that side), as on a mosaic whose focus varies: the tissue-free tiles
    the calibration prefers are on the strong-ring side. Tissue: a smooth plateau over
    the central 2048 px square.
    """
    import tifffile

    st = bs.make_stack(
        n_planes=4, size=GSIZE, n_beads=0, seed=seed, tissue=False, blobs=False
    )
    rng = np.random.default_rng(seed + 7)
    pos = bs._place(rng, n_beads, (GSIZE, GSIZE), 30.0, [])
    amps = np.exp(rng.uniform(np.log(800.0), np.log(6000.0), len(pos)))
    light = np.zeros((4, GSIZE, GSIZE), np.float32)
    for (by, bx), a in zip(pos, amps, strict=True):
        w = 0.46 + (ring_w_right - 0.46) * bx / GSIZE
        _add_z0(light[0], by, bx, float(a), w)
        for k in (1, 2, 3):
            bs._add_profile(light[k], by, bx, float(a), k)
    mask = np.zeros((GSIZE, GSIZE), bool)
    mask[1024:3072, 1024:3072] = True
    noise = np.random.default_rng(seed + 100).normal(0, 1, (GSIZE, GSIZE))
    plateau = gaussian_filter(noise.astype(np.float32), 6)
    plateau *= 800.0 / max(float(plateau.std()), 1e-6)
    for k in range(4):
        pol = st["PolyT"][k] + light[k]
        dap = st["DAPI"][k] + 0.11 * light[k]
        pol[mask] += 300.0 + 0.3 * plateau[mask]
        dap[mask] += 3000.0 + plateau[mask]
        for stain, plane in (
            ("PolyT", pol),
            ("DAPI", dap),
            ("Anti-X", st["Anti-X"][k]),
        ):
            tifffile.imwrite(
                path / f"mosaic_{stain}_z{k}.tif",
                np.clip(np.rint(plane), 1, 65535).astype(np.uint16),
            )


@pytest.fixture(scope="module")
def mosaic_gradient_mild(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("mosaic_gradient_mild")
    build_gradient_mosaic(d, ring_w_right=0.17)
    return d


@pytest.fixture(scope="module")
def mosaic_gradient_strong(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("mosaic_gradient_strong")
    build_gradient_mosaic(d, ring_w_right=0.06)
    return d


def _warning_texts(cal: BeadCalibration, needle: str) -> list[str]:
    return [w for w in cal.decision["warnings"] if needle in w]


@pytest.mark.slow
def test_heldout_recall_adapts_threshold_to_a_focus_gradient(
    mosaic_gradient_mild: Path,
) -> None:
    """Weaker rings on one side: the grid tiles lower ring_min below the clamp."""
    with _reader(mosaic_gradient_mild) as rd:
        cal = calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)
    c = cal.decision["classifier"]
    assert {t["role"] for t in c["tiles"]} == {"calibration", "threshold", "recall"}
    assert c["n_grid_threshold_reference"] > 0 and c["n_recall_reference"] >= 8
    assert c["recall_reliable"] == (c["n_recall_reference"] >= MIN_RECALL_REFERENCE)
    # the calibration tiles alone would give the 0.30 clamp
    assert cal.thresholds.ring_min < 0.29
    assert c["ring_p2_calibration_tiles_only"] > cal.thresholds.ring_min
    assert c["recall"] >= 0.90, c["recall"]
    assert not _warning_texts(cal, "held-out")
    # the cluster-inclusive recall counts at least the isolated reference beads
    assert c["recall_with_clusters"] >= 0.0
    assert c["n_recall_reference_with_clusters"] >= c["n_recall_reference"]
    # the per-tile ring-ratio distributions show the gradient
    rec = [t for t in c["tiles"] if t["role"] == "recall" and t["stats"]["n"] >= 2]
    left = [t["stats"]["ring_median"] for t in rec if t["x0"] < GSIZE / 2 - 512]
    right = [t["stats"]["ring_median"] for t in rec if t["x0"] > GSIZE / 2 - 512]
    assert left and right and np.median(left) > np.median(right) + 0.05
    # held-out reference beads are never beads that set the thresholds
    assert (
        c["n_reference"]
        == c["n_calibration_tile_reference"] + c["n_grid_threshold_reference"]
    )
    assert "variant" not in c and cal.thresholds.ring_max == DEFAULT_PARAMS.ring_max


@pytest.mark.slow
def test_heldout_recall_drops_when_the_gradient_exceeds_the_clamp(
    mosaic_gradient_strong: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rings so weak on one side that no allowed ring_min recalls them.

    This is the baseline classifier (the rescue is switched off): the held-out
    recall drops (the in-sample recall on the calibration tiles does not) and the
    QC warnings are exactly those of the owner decision.
    """
    monkeypatch.setattr(calibrate_module, "RESCUE_TZ1_MIN", 0.0)
    with _reader(mosaic_gradient_strong) as rd:
        cal = calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)
    c = cal.decision["classifier"]
    assert cal.thresholds.ring_min == pytest.approx(0.20)  # lower clamp
    assert not cal.thresholds.rescue_active and c["n_rescued_tiles"] == 0
    assert c["n_recall_reference"] >= 8 and c["recall"] < 0.90, c["recall"]
    cal_tiles = [t for t in c["tiles"] if t["role"] == "calibration"]
    in_sample = sum(t["n_recalled"] for t in cal_tiles) / sum(
        t["n_reference_used"] for t in cal_tiles
    )
    assert in_sample >= 0.95
    expected = classifier_recall_warnings(
        c["recall"],
        c["n_recall_reference"],
        c["recall_with_clusters"],
        c["n_recall_reference_with_clusters"],
    )
    got = [
        w
        for w in cal.decision["warnings"]
        if "held-out" in w or "including beads in clusters" in w
    ]
    assert got == expected
    # the isolated recall warns only when it rests on enough beads
    assert bool(_warning_texts(cal, "held-out classifier recall is")) == (
        c["n_recall_reference"] >= MIN_RECALL_REFERENCE
    )


@pytest.mark.slow
def test_the_rescue_recovers_beads_whose_z0_rings_are_too_weak(
    mosaic_gradient_strong: Path,
) -> None:
    """The same mosaic with the rescue on: weak z0 rings, but clean z1 rings.

    The beads that no allowed ring_min recalls are rescued on their z1 ring, the
    held-out recall (isolated and with clusters) rises above the warning limits
    and the calibration reports how many beads the rescue accepted.
    """
    with _reader(mosaic_gradient_strong) as rd:
        cal = calibrate_bead_model(rd, config=CFG, z_layers=ZL, n_workers=2)
    c = cal.decision["classifier"]
    assert cal.thresholds.rescue_active
    assert cal.thresholds.ring_min == pytest.approx(0.20)  # the ring test is unchanged
    assert c["n_recall_reference"] >= 8 and c["recall"] >= 0.95, c["recall"]
    assert c["recall_with_clusters"] >= 0.95
    assert c["n_rescued_tiles"] > 0 and c["rescued_fraction_tiles"] > 0.0
    assert c["rescued_fraction_tiles"] == pytest.approx(
        c["n_rescued_tiles"] / sum(t["n_classified"] for t in c["tiles"])
    )
    assert not [w for w in cal.decision["warnings"] if "recall" in w]
