"""Tests of the bead model: the window fit and its building blocks.

Synthetic data only.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from merxen.io.bead_removal.detect import detect_sources, empty_sources
from merxen.io.bead_removal.model import (
    BeadModel,
    PlaneFit,
    beads_table,
    derive_library_templates,
    estimate_stretch_field,
    fit_plane,
    fit_tile_local,
    fit_window_library,
    smooth_field,
    top_per_block,
)
from merxen.io.bead_removal.params import (
    BETA,
    DEFAULT_PARAMS,
    FLUX_TARGET,
)
from merxen.io.bead_removal.psf import ring_prominence

from . import bead_synth as bs


@pytest.fixture(scope="module")
def cal() -> bs.SyntheticCalibration:
    return bs.make_calibration()


@pytest.fixture(scope="module")
def fitted(
    cal: bs.SyntheticCalibration,
) -> tuple[dict[str, np.ndarray], BeadModel]:
    st = bs.make_stack(seed=1, n_beads=30, size=512, tilt=0.1)
    model = fit_window_library(
        st["PolyT"],
        st["DAPI"],
        origin_yx=(0, 0),
        calibration=cal,
        planes=[1, 2, 3],
    )
    return st, model


def test_top_per_block_keeps_the_best_per_absolute_block() -> None:
    y = np.array([10.0, 20.0, 30.0, 600.0, 610.0, 620.0, 100.0])
    x = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 100.0])
    score = np.array([5.0, 9.0, 9.0, 1.0, 2.0, 3.0, 7.0])
    cand = np.ones(7, bool)
    keep = top_per_block(score, cand, y, x, (0, 0), 512, 2)
    # Block (0, 0) holds the scores 5, 9, 9, 7: the two 9s win (the tie goes to
    # the smaller y). Block (1, 0) holds 1, 2, 3: 3 and 2 win.
    assert keep.tolist() == [False, True, True, False, True, True, False]
    # An origin shifted by whole blocks gives the same choice; a shift of 500 px
    # regroups the sources.
    assert np.array_equal(keep, top_per_block(score, cand, y, x, (1024, 512), 512, 2))
    assert not np.array_equal(keep, top_per_block(score, cand, y, x, (500, 0), 512, 2))
    assert not top_per_block(score, np.zeros(7, bool), y, x, (0, 0), 512, 2).any()


def test_top_per_block_respects_the_candidate_mask() -> None:
    y = np.array([10.0, 20.0, 30.0])
    x = np.array([10.0, 20.0, 30.0])
    score = np.array([9.0, 8.0, 7.0])
    cand = np.array([False, True, True])
    keep = top_per_block(score, cand, y, x, (0, 0), 512, 1)
    assert keep.tolist() == [False, True, False]


def test_smooth_field_is_a_weighted_median_and_nan_where_sparse() -> None:
    pts = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [5.0, 5.0], [900.0, 900.0]])
    val = np.array([1.0, 1.0, 1.0, 100.0, 7.0])
    wgt = np.ones(5)
    query = np.array([[4.0, 4.0], [900.0, 900.0], [500.0, 0.0]])
    out = smooth_field(pts, val, wgt, query, sigma=20.0, kmin=3)
    assert out[0] == 1.0  # the outlier does not move the median
    assert np.isnan(out[1])  # one sample within 2.5 sigma, kmin is 3
    assert np.isnan(out[2])
    assert smooth_field(pts[:0], val[:0], wgt[:0], query, 20.0).shape == (3,)


def test_smooth_field_without_samples_is_all_nan() -> None:
    out = smooth_field(
        np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.ones((4, 2)), sigma=10.0
    )
    assert np.isnan(out).all()


def test_the_window_fit_finds_and_models_the_beads(
    fitted: tuple[dict[str, np.ndarray], BeadModel],
) -> None:
    st, model = fitted
    info = model.info
    assert info["status"] == "ok"
    assert info["n_beads_final"] >= 25
    assert set(model.fits) == {0, 1, 2, 3}
    assert set(model.B) == {0, 1, 2, 3}
    for k, fit in model.fits.items():
        assert isinstance(fit, PlaneFit)
        assert fit.model.shape == st["PolyT"].shape[1:]
        assert fit.model.dtype == np.float32
        assert fit.g.shape == (model.src.n,)
        assert fit.tmpl is not None
        assert model.noise[k] > 0
    # The z0 amplitudes of the fit follow the injected ones.
    assert float(np.median(model.src.a[model.src.is_bead])) > 300
    for key in ("time_detect_s", "time_z0_s", "time_stretch_s", "time_s"):
        assert info[key] >= 0.0


def test_the_fit_does_not_depend_on_the_window_it_is_cut_from(
    cal: bs.SyntheticCalibration,
) -> None:
    """Only the absolute grid matters: shifting the origin by whole blocks is free."""
    st = bs.make_stack(seed=9, n_beads=12, size=384)
    a = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1, 2]
    )
    b = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(1024, 512), calibration=cal, planes=[1, 2]
    )
    for k in a.fits:
        assert np.array_equal(a.fits[k].model, b.fits[k].model)


def test_planes_above_two_are_fitted_only_when_listed(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=9, n_beads=10, size=384)
    low = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1]
    )
    high = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1, 5]
    )
    assert set(low.fits) == {0, 1, 2}
    assert set(high.fits) == {0, 1, 2, 5}


def test_a_window_without_beads_gives_an_empty_model(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=5, n_beads=0, size=512, punctum_near_bead=False)
    model = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1]
    )
    assert model.info["status"] == "no beads"
    assert model.fits == {}
    assert beads_table(model, (0, 0)).shape == (0, 8)


def test_saturation_handling_is_off_by_default(
    cal: bs.SyntheticCalibration,
) -> None:
    beads = np.array(
        [[150.0, 150.0, 600000.0], [150.0, 400.0, 30000.0], [400.0, 250.0, 150000.0]]
    )
    st = bs.make_stack(size=512, seed=31, beads=beads, tissue=False)
    assert int((st["PolyT"][0] >= 65535).sum()) > 20
    model = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1]
    )
    assert model.info["n_saturated_px"] == 0
    capped = dataclasses.replace(DEFAULT_PARAMS, sat_level=65000.0)
    sat_cal = bs.make_calibration(params=capped)
    model = fit_window_library(
        st["PolyT"],
        st["DAPI"],
        origin_yx=(0, 0),
        calibration=sat_cal,
        params=capped,
        planes=[1],
    )
    assert model.info["n_saturated_px"] > 0


def test_an_aggregate_of_beads_is_modelled_as_a_blob_and_removed(
    cal: bs.SyntheticCalibration,
) -> None:
    cluster = [
        [200.0 + dy, 200.0 + dx, 9000.0]
        for dy, dx in ((0, 0), (7, 3), (-6, 8), (4, -9), (-8, -5))
    ]
    beads = np.array([*cluster, [400.0, 400.0, 3000.0], [350.0, 150.0, 2000.0]])
    st = bs.make_stack(size=640, seed=4, beads=beads, tissue=False, margin=5)
    model = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1, 2]
    )
    # The five overlapping beads are one extended source with a source map.
    assert model.info["n_blob"] == 1
    assert model.src.blob.sum() == 1
    assert len(model.src.u) == 1
    j = int(np.where(model.src.blob)[0][0])
    assert abs(model.src.y[j] - 200) < 12
    assert abs(model.src.x[j] - 200) < 12
    for k in (1, 2):
        residual = st["PolyT"][k] - st["truth/PolyT"][k] - model.fits[k].model
        injected = st["beadlight/PolyT"][k]
        assert np.sqrt(np.mean(residual**2)) < 0.1 * np.sqrt(np.mean(injected**2))


def test_beads_table_lists_the_classified_beads_with_absolute_positions(
    fitted: tuple[dict[str, np.ndarray], BeadModel],
) -> None:
    _st, model = fitted
    table = beads_table(model, (32, 48))
    assert table.shape == (int(model.src.cand_bead.sum()), 8)  # type: ignore[union-attr]
    assert np.allclose(table[:, 0] - 32, model.src.y[model.src.cand_bead])
    assert np.allclose(table[:, 1] - 48, model.src.x[model.src.cand_bead])
    assert set(np.unique(table[:, 7])) <= {0.0, 1.0}
    # A pruned source is flagged in the last column.
    assert table[:, 7].sum() == model.info["n_pruned"]


def test_beads_table_of_an_empty_model() -> None:
    model = BeadModel(empty_sources(), {}, {}, {}, {"status": "no beads"})
    assert beads_table(model, (0, 0)).shape == (0, 8)


def test_derived_library_templates_reach_the_flux_targets(
    cal: bs.SyntheticCalibration,
) -> None:
    low = list(cal.library.tmpls[:4])
    full = derive_library_templates(low)
    assert len(full) == 8
    for k in range(4, 8):
        flux = full[k].flux(1e9)
        assert flux > 0
        assert flux == pytest.approx(FLUX_TARGET[k] * full[0].flux(1e9), rel=0.35)


def test_fit_plane_with_a_fixed_template_scales_the_gain_to_the_data(
    cal: bs.SyntheticCalibration,
) -> None:
    beads = np.array([[200.0, 200.0, 4000.0], [200.0, 420.0, 2500.0]])
    st = bs.make_stack(size=640, seed=3, beads=beads, tissue=False)
    plane = st["PolyT"]
    src = detect_sources(
        plane,
        st["DAPI"],
        dog_noise=cal.dog_noise_z0,
        thresholds=cal.thresholds,
        params=DEFAULT_PARAMS,
    )
    assert src.is_bead.sum() == 2
    env = np.full(plane.shape[1:], bs.BG_POLYT, np.float32)
    fit = fit_plane(
        plane[1],
        env,
        src,
        1,
        DEFAULT_PARAMS,
        use=src.is_bead,
        sel=src.is_bead,
        noise=cal.noise[1],
        fixed_template=cal.library.tmpls[1],
    )
    assert fit.model.max() > 0
    assert np.all(fit.g[src.is_bead] > 0.5)
    assert np.all(fit.g[src.is_bead] < 1.6)
    # The model explains the bead light: the residual shrinks.
    residual = plane[1] - env - fit.model
    assert np.abs(residual).mean() < np.abs(plane[1] - env).mean()


def test_stretch_estimation_falls_back_to_the_default_without_samples(
    cal: bs.SyntheticCalibration,
) -> None:
    st = bs.make_stack(seed=9, n_beads=3, size=512, tissue=False)
    model = fit_window_library(
        st["PolyT"], st["DAPI"], origin_yx=(0, 0), calibration=cal, planes=[1]
    )
    imgs = {k: st["PolyT"][k] for k in (0, 1, 2)}
    ell, n_measured, _median = estimate_stretch_field(
        imgs,
        {k: model.B[k] for k in (0, 1, 2)},
        model.src,
        {k: model.fits[k] for k in (0, 1, 2)},
        DEFAULT_PARAMS,
        origin_yx=(0, 0),
        ell_default=0.07,
        ks=(0, 1, 2),
    )
    assert ell.shape == (model.src.n,)
    assert np.isfinite(ell).all()
    if n_measured == 0:
        assert np.all(ell == 0.07)
    assert BETA[0] > BETA[1]


@pytest.mark.slow
def test_fit_tile_local_stacks_templates_with_a_z1_ring() -> None:
    st = bs.make_stack(seed=11, n_beads=90, tissue=False)
    model = fit_tile_local(
        {k: st["PolyT"][k] for k in range(4)}, {k: st["DAPI"][k] for k in range(2)}
    )
    assert model.info["status"] == "ok"
    assert model.info["n_ring_evidence"] >= 8
    prominence, radius = ring_prominence(model.fits[1].tmpl, 20.0, 40.0)
    assert prominence > 0.03
    assert 24 <= radius <= 34
    assert model.info["z1_ring_radius"] == radius
    free = bs.make_stack(seed=12, n_beads=0, tissue=False)
    empty = fit_tile_local(
        {k: free["PolyT"][k] for k in range(4)}, {k: free["DAPI"][k] for k in range(2)}
    )
    assert empty.info["status"] != "ok"
    assert empty.fits == {}


def test_a_tile_with_too_few_beads_is_reported_not_raised() -> None:
    st = bs.make_stack(seed=12, n_beads=3, size=512, tissue=False)
    model = fit_tile_local(
        {k: st["PolyT"][k] for k in range(4)}, {k: st["DAPI"][k] for k in range(2)}
    )
    assert model.info["status"] == "too few beads"
    assert model.fits == {}
