"""Tests of the bead detection, the classifier and the window-independent envelope.

Synthetic data only. The envelope tests pin the contract that a valid pixel's
lower envelope never depends on the rest of the window (so chunked and
whole-mosaic runs agree), and that a fully valid window takes exactly the
original computation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from merxen.io.bead_removal import detect as det
from merxen.io.bead_removal.detect import ClassifierThresholds
from merxen.io.bead_removal.params import DEFAULT_PARAMS, ENVELOPE_BLOCK_PX

from . import bead_synth as bs


@pytest.fixture(scope="module")
def dog_noise() -> float:
    return bs.bead_free_constants()[1]


@pytest.fixture(scope="module")
def thresholds() -> ClassifierThresholds:
    return ClassifierThresholds.from_params(DEFAULT_PARAMS)


def _reference_valid_envelope(
    img: np.ndarray,
    sub: int = 1,
    block: int = 16,
    win_blocks: int = 9,
    pct: float = 20.0,
) -> np.ndarray:
    """Compute the original code path for a fully valid window.

    It is the oracle of bit-identity for the production implementation.
    """
    height, width = img.shape
    n_by, n_bx = height // block, width // block
    blocks = (
        img[: n_by * block, : n_bx * block]
        .reshape(n_by, block, n_bx, block)
        .transpose(0, 2, 1, 3)
        .reshape(n_by, n_bx, -1)
    )
    grid = np.percentile(blocks[:, :, ::sub] if sub > 1 else blocks, pct, axis=2)
    grid = ndi.percentile_filter(grid, 30, size=win_blocks, mode="nearest")
    grid = ndi.gaussian_filter(grid, 2.0, mode="nearest")
    yy = np.clip((np.arange(height) + 0.5) / block - 0.5, 0, n_by - 1)
    xx = np.clip((np.arange(width) + 0.5) / block - 0.5, 0, n_bx - 1)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    return ndi.map_coordinates(grid, [gy, gx], order=1, mode="nearest").astype(
        np.float32
    )


def _scene(shape: tuple[int, int] = (1024, 1024), seed: int = 0) -> np.ndarray:
    """Make a textured background with a gradient so that any leak is visible."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    img = 1000 + 3 * yy + 0.5 * xx + rng.normal(0, 30, shape).astype(np.float32)
    return img.astype(np.float32)


# --- envelope ---------------------------------------------------------------
def test_envelope_constants_are_consistent() -> None:
    assert det.ENVELOPE_BLOCK_PX == ENVELOPE_BLOCK_PX == 16
    # scipy truncates a Gaussian at 4 sigma; the reach is that, in blocks.
    assert round(4 * det.ENVELOPE_GAUSS_SIGMA_BLOCKS) == det.ENVELOPE_GAUSS_REACH_BLOCKS
    assert det.ENVELOPE_FILTER_PCT == 30.0


@pytest.mark.parametrize("sub", [1, 4])
def test_envelope_fully_valid_window_is_bit_identical(sub: int) -> None:
    img = _scene()
    reference = _reference_valid_envelope(img, sub=sub)
    assert np.array_equal(det.lower_envelope(img, None, sub=sub), reference)
    # An all-True mask takes the same path as no mask.
    assert np.array_equal(det.lower_envelope(img, img > 0, sub=sub), reference)


@pytest.mark.parametrize("n", [16, 64, 256])
def test_masked_percentile_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(n)
    vals = (rng.random((200, n)) * 5000).astype(np.float32)
    ok = np.ones(vals.shape, bool)
    got = det._masked_percentile_rows(vals, ok, 20.0)
    assert np.allclose(got, np.percentile(vals, 20, axis=1), rtol=1e-6, atol=0)
    ok = rng.random(vals.shape) > 0.4
    ok[0] = False  # a row without a valid entry
    got = det._masked_percentile_rows(vals, ok, 20.0)
    want = np.array(
        [
            np.percentile(row[valid], 20) if valid.any() else np.nan
            for row, valid in zip(vals, ok, strict=True)
        ]
    )
    assert np.isnan(got[0])
    assert np.allclose(got[1:], want[1:], rtol=1e-6, atol=0)


def test_nan_percentile_filter_equals_scipy_without_nan() -> None:
    rng = np.random.default_rng(1)
    grid = (rng.random((40, 50)) * 1000).astype(np.float32)
    assert np.array_equal(
        det._percentile_filter_nan(grid, 30.0, 9),
        ndi.percentile_filter(grid, 30, size=9, mode="nearest"),
    )


def test_nan_percentile_filter_ignores_nan_neighbours() -> None:
    rng = np.random.default_rng(1)
    grid = (rng.random((40, 50)) * 1000).astype(np.float32)
    grid[5:8, 5] = np.nan
    grid[:, 45:] = np.nan  # an invalid band at the array edge
    got = det._percentile_filter_nan(grid, 30.0, 9)
    padded = np.pad(grid, 4, mode="edge")
    for y, x in ((6, 6), (0, 0), (20, 44), (20, 48), (39, 49)):
        window = padded[y : y + 9, x : x + 9].ravel()
        window = np.sort(window[~np.isnan(window)])
        want = window[int(len(window) * 30 // 100)] if len(window) else np.nan
        assert (np.isnan(got[y, x]) and np.isnan(want)) or got[y, x] == want, (y, x)
    # Away from every NaN the result is scipy's.
    clean = ndi.percentile_filter(
        np.where(np.isnan(grid), 0, grid), 30, size=9, mode="nearest"
    )
    assert np.array_equal(got[20:30, 5:30], clean[20:30, 5:30])


def test_fill_undefined_blocks_propagates_within_reach_only() -> None:
    grid = np.full((9, 9), np.nan, np.float32)
    grid[4, 4] = 7.0
    filled = det.fill_undefined_blocks(grid, reach=2)
    assert filled.dtype == np.float32 and not np.isnan(filled).any()
    # Chessboard distance <= 2 takes the nearest defined value, beyond it 0.
    assert (filled[2:7, 2:7] == 7.0).all()
    assert filled[0, 0] == 0.0 and filled[4, 8] == 0.0
    # Without NaN the grid is returned as it is; with nothing defined it is 0.
    full = np.ones((4, 4), np.float32)
    assert det.fill_undefined_blocks(full, 3) is full
    assert not det.fill_undefined_blocks(np.full((4, 4), np.nan, np.float32), 3).any()


def test_fill_prefers_the_nearest_defined_block() -> None:
    grid = np.full((1, 12), np.nan)
    grid[0, 0], grid[0, 11] = 1.0, 2.0
    filled = det.fill_undefined_blocks(grid, reach=20)
    assert filled[0, 3] == 1.0 and filled[0, 8] == 2.0


def test_envelope_same_statistic_whatever_the_validity() -> None:
    """One invalid pixel moves the envelope only within the reach of its block."""
    img = _scene()
    valid = np.ones(img.shape, bool)
    valid[0, 0] = False
    diff = np.abs(
        det.lower_envelope(img, None, sub=4) - det.lower_envelope(img, valid, sub=4)
    )
    # The percentile filter and the Gaussian reach about 208 px.
    assert diff[200:, 200:].max() == 0.0
    assert diff.max() < 40


def test_envelope_of_valid_pixels_is_window_independent() -> None:
    img = _scene((1024, 1024))
    valid = np.ones(img.shape, bool)
    valid[:, :150] = False  # outside the mosaic
    valid[500:540, 600:700] = False  # an interior hole
    img = np.where(valid, img, 0).astype(np.float32)
    full = det.lower_envelope(img, valid, sub=4)
    # The same data cut at row 640: a window-level statistic would differ
    # strongly because of the gradient.
    cut = det.lower_envelope(img[:640], valid[:640], sub=4)
    reach = 13 * 16
    region = valid[: 640 - reach]
    assert np.abs(full[: 640 - reach] - cut[: 640 - reach])[region].max() < 1e-3
    # No NaN leaks into the smoothing.
    assert np.isfinite(full).all() and np.isfinite(cut).all()


def test_envelope_all_invalid_and_isolated_valid_block() -> None:
    img = np.zeros((256, 256), np.float32)
    out = det.lower_envelope(img, img > 0)
    assert np.isfinite(out).all() and not out.any()
    img[100:116, 100:116] = 500.0
    out = det.lower_envelope(img, img > 0)
    assert np.isfinite(out).all()
    # A single valid block gets its own value, not a window statistic.
    assert abs(float(out[108, 108]) - 500.0) < 1e-3


# --- small helpers ------------------------------------------------------------
def test_noise_estimates_ignore_invalid_pixels() -> None:
    rng = np.random.default_rng(0)
    img = rng.normal(1000, 20, (200, 200)).astype(np.float32)
    valid = np.ones(img.shape, bool)
    assert det.robust_noise(img, valid) == pytest.approx(20.0, rel=0.1)
    assert det.robust_noise(np.zeros((8, 8), np.float32), np.zeros((8, 8), bool)) == 1.0
    assert det.dog_noise_of(img) > 0
    assert det.dog_noise_of(np.zeros((8, 8), np.float32)) == 1.0
    # Zeros (outside the mosaic) are not valid by default and do not inflate it.
    padded = img.copy()
    padded[:, :60] = 0
    assert det.dog_noise_of(padded) == pytest.approx(det.dog_noise_of(img), rel=0.2)


def test_huber_weights_are_one_sided() -> None:
    residual = np.array([-100.0, 0.0, 1.0, 2.0, 4.0], np.float32)
    weights = det.huber1(residual, 1.0, 2.0)
    assert weights.dtype == np.float32
    assert weights.tolist() == [1.0, 1.0, 1.0, 1.0, 0.5]


def test_window_bounds_and_radius_grid() -> None:
    assert det.window_bounds((100, 80), 50.0, 40.0, 10.0) == (40, 61, 30, 51)
    # Clipped to the image on every side.
    assert det.window_bounds((100, 80), 2.0, 78.0, 10.0) == (0, 13, 68, 80)
    grid = det.radius_grid(5.0, 6.0, 3, 9, 2, 10)
    assert grid.shape == (6, 8) and grid.dtype == np.float32
    assert grid[2, 4] == 0.0
    assert grid[0, 4] == pytest.approx(2.0)


def test_nms_keeps_the_brightest_of_a_neighbourhood_in_stable_order() -> None:
    y = np.array([10.0, 12.0, 100.0, 11.0])
    x = np.array([10.0, 10.0, 100.0, 10.0])
    score = np.array([1.0, 3.0, 2.0, 3.0])
    keep = det.nms(y, x, score, 5.0)
    # Ties go to the earlier point (stable sort).
    assert keep.tolist() == [False, True, True, False]


def test_centre_minus_background_and_null_sd() -> None:
    img = np.full((101, 101), 100.0, np.float32)
    img[48:53, 48:53] += 50.0
    contrast = det.centre_minus_bg(img, np.array([50]), np.array([50]))
    assert contrast[0] == pytest.approx(50.0)
    sd = det.null_sd(img, np.array([50]), np.array([50]), n=16)
    assert sd.shape == (1,) and sd[0] >= 0.0
    flat = det.null_sd(
        np.full((300, 300), 5.0, np.float32), np.array([150]), np.array([150])
    )
    assert flat[0] == 0.0


def test_ring_ratio_sees_a_ring_and_not_a_puncta() -> None:
    yy, xx = np.mgrid[:80, :80]
    radius = np.hypot(yy - 40, xx - 40)
    ring = (1000 + 400 * np.exp(-((radius - 11) ** 2) / 8)).astype(np.float32)
    puncta = (1000 + 1000 * np.exp(-(radius**2) / 4.5)).astype(np.float32)
    ys, xs = np.array([40.0]), np.array([40.0])
    assert det.ring_ratio(ring, ys, xs)[0] > 300
    assert det.ring_ratio(puncta, ys, xs)[0] < 5


# --- detection and classification --------------------------------------------
def test_detection_classifies_the_beads_and_not_the_tissue(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    stack = bs.make_stack(seed=1, n_beads=40, size=768)
    sources = det.detect_sources(
        stack["PolyT"], stack["DAPI"], dog_noise=dog_noise, thresholds=thresholds
    )
    truth = stack["beads"][:, :2]
    distance, nearest = cKDTree(np.c_[sources.y, sources.x]).query(truth)
    assert (distance < 1.5).all()
    assert sources.is_bead[nearest].all()
    # Every classified bead is a real one: tissue is never accepted.
    assert int(sources.is_bead.sum()) == len(truth)
    assert sources.n >= len(truth)
    assert sources.rring is not None and sources.rring.shape == (sources.n,)


def test_detection_accepts_a_mapping_of_planes(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    stack = bs.make_stack(seed=2, n_beads=10, size=512)
    as_array = det.detect_sources(
        stack["PolyT"], stack["DAPI"], dog_noise=dog_noise, thresholds=thresholds
    )
    as_dict = det.detect_sources(
        {k: stack["PolyT"][k] for k in (0, 1)},
        {k: stack["DAPI"][k] for k in (0, 1)},
        dog_noise=dog_noise,
        thresholds=thresholds,
    )
    assert np.array_equal(as_array.y, as_dict.y)
    assert np.array_equal(as_array.is_bead, as_dict.is_bead)
    assert np.array_equal(as_array.A0, as_dict.A0)


def test_bright_tissue_puncta_without_a_ring_are_not_beads(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:512, :512]
    polyt = (1500 + rng.normal(0, 20, (2, 512, 512))).astype(np.float32)
    dapi = (500 + rng.normal(0, 20, (2, 512, 512))).astype(np.float32)
    for cy, cx in ((100, 100), (300, 200), (400, 400)):
        spot = 3000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5**2))
        polyt += spot.astype(np.float32)  # in focus at both planes
        dapi[0] += 0.1 * spot.astype(np.float32)
    sources = det.detect_sources(
        polyt, dapi, dog_noise=dog_noise, thresholds=thresholds
    )
    assert sources.n == 3 and not sources.is_bead.any()
    assert sources.r1 is not None and (sources.r1 > 0.9).all()


def test_no_candidates_gives_empty_sources(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    free = bs.make_stack(n_beads=0, seed=5, size=256, blobs=False)
    sources = det.detect_sources(
        free["PolyT"], free["DAPI"], dog_noise=dog_noise, thresholds=thresholds
    )
    assert sources.n == 0 and not sources.is_bead.any()
    assert det.empty_sources().n == 0


def test_a_missing_dapi_skips_the_spectral_test(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    stack = bs.make_stack(seed=3, n_beads=15, size=512, tissue=False)
    with_dapi = det.detect_sources(
        stack["PolyT"], stack["DAPI"], dog_noise=dog_noise, thresholds=thresholds
    )
    without = det.detect_sources(
        stack["PolyT"], None, dog_noise=dog_noise, thresholds=thresholds
    )
    assert np.isnan(without.rD).all()
    assert without.is_bead.sum() >= with_dapi.is_bead.sum() == 15


def test_dog_noise_defaults_to_the_window_estimate(
    thresholds: ClassifierThresholds,
) -> None:
    stack = bs.make_stack(seed=4, n_beads=10, size=512, tissue=False)
    own = det.dog_noise_of(stack["PolyT"][0])
    explicit = det.detect_sources(
        stack["PolyT"], stack["DAPI"], dog_noise=own, thresholds=thresholds
    )
    implicit = det.detect_sources(
        stack["PolyT"], stack["DAPI"], dog_noise=None, thresholds=thresholds
    )
    assert np.array_equal(explicit.y, implicit.y)


def test_classifier_thresholds_follow_the_parameters() -> None:
    thr = ClassifierThresholds.from_params(DEFAULT_PARAMS)
    assert thr.ring_min == DEFAULT_PARAMS.ring_min
    assert thr.rd0_center == DEFAULT_PARAMS.rd0_center
    assert set(thr.to_dict()) == {
        "ring_min",
        "ring_max",
        "r1_max",
        "rd_center",
        "rd0_center",
        "rd_tol",
        "rescue_tz1_min",
        "rescue_ring_radius_px",
        "rescue_a0_min",
        "rescue_sd1_max",
        "rescue_zprof_zmax",
    }
    assert not thr.rescue_active
    assert all(isinstance(v, float) for v in thr.to_dict().values())


def test_a_stricter_ring_threshold_rejects_candidates(dog_noise: float) -> None:
    stack = bs.make_stack(seed=1, n_beads=20, size=512, tissue=False)
    strict = ClassifierThresholds(ring_min=5.0)
    sources = det.detect_sources(
        stack["PolyT"], stack["DAPI"], dog_noise=dog_noise, thresholds=strict
    )
    assert sources.n >= 15 and not sources.is_bead.any()


# --- edge margin ---------------------------------------------------------------
def test_edge_margin_ignores_the_window_border(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    """One invalid corner pixel must not change the candidates away from it."""
    stack = bs.make_stack(seed=21, n_beads=40, size=768, tissue=False, margin=12)
    kwargs = {
        "dog_noise": dog_noise,
        "thresholds": thresholds,
        "params": DEFAULT_PARAMS,
    }
    polyt = {k: stack["PolyT"][k] for k in (0, 1)}
    dapi = {k: stack["DAPI"][k] for k in (0, 1)}
    before = det.detect_sources(polyt, dapi, **kwargs)
    polyt2 = {k: plane.copy() for k, plane in polyt.items()}
    polyt2[0][0, 0] = 0.0
    after = det.detect_sources(polyt2, dapi, **kwargs)

    def far(s: det.Sources) -> np.ndarray:
        return np.hypot(s.y, s.x) > DEFAULT_PARAMS.edge_margin_px + 20

    pos0 = np.round(np.c_[before.y, before.x][far(before)], 3)
    pos1 = np.round(np.c_[after.y, after.x][far(after)], 3)
    assert len(pos0) > 10 and np.array_equal(pos0, pos1)
    assert np.array_equal(before.is_bead[far(before)], after.is_bead[far(after)])
    # The comparison includes candidates nearer than the margin to the window
    # border: the border is not an invalid edge.
    border = (
        np.minimum(
            np.minimum(before.y, before.x),
            np.minimum(767 - before.y, 767 - before.x),
        )
        < DEFAULT_PARAMS.edge_margin_px
    )
    assert (border & far(before)).any()


def test_edge_margin_drops_candidates_next_to_invalid_pixels(
    dog_noise: float, thresholds: ClassifierThresholds
) -> None:
    stack = bs.make_stack(
        seed=22,
        n_beads=0,
        size=512,
        tissue=False,
        beads=np.array([[100.0, 150.0, 5000.0], [300.0, 300.0, 5000.0]]),
    )
    polyt = {k: stack["PolyT"][k].copy() for k in (0, 1)}
    dapi = {k: stack["DAPI"][k] for k in (0, 1)}
    for plane in polyt.values():
        plane[:, :140] = 0  # edge of the mosaic 10 px left of the first bead
    sources = det.detect_sources(
        polyt, dapi, dog_noise=dog_noise, thresholds=thresholds, params=DEFAULT_PARAMS
    )
    assert sources.n >= 1 and (sources.x > 150 - 5).all()
    assert (np.abs(sources.y - 300) < 3).any()
