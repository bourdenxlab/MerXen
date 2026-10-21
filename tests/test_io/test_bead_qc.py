"""Tests of the bead-removal QC record (``bead_removal_qc/``).

Fast: the t-scores, the residual-ring score and its pooling, the per-chunk probe,
the warning rules, the tables, the crop choice, the maps, a QC folder written
from a tiny synthetic run, and the builder call (a failing QC never fails the
build). Slow: the probe results and the change overview are the same for 1 and
2 workers.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.config import MerscopeBeadRemovalConfig, MerscopeBuildConfig
from merxen.io.bead_removal import driver, qc
from merxen.io.bead_removal.calibrate import BeadCalibration
from merxen.io.bead_removal.fill import FillParams
from merxen.io.builders import merscope as merscope_module
from merxen.io.builders.merscope import (
    ProjectionBuildResult,
    write_merscope_spatialdata,
)

from . import bead_synth as bs
from .bead_build_fixtures import (
    open_projection_store,
    raw_max,
    real_calibration,
    write_mosaic,
)

ZS = [1, 2, 3]
SHAPE = (1024, 2048)
STAINS = ["PolyT", "DAPI"]
R1_PX = 28.5


def _calibration() -> BeadCalibration:
    """A synthetic calibration with the QC records a calibration pass writes."""
    cal = real_calibration()
    cal.tiles = [
        {"index": 0, "y0": 0, "x0": 0, "tile_px": 512, "status": "ok", "n_bead": 30},
        {"index": 1, "y0": 0, "x0": 1024, "tile_px": 512, "status": "ok", "n_bead": 5},
        {"index": 2, "y0": 1024, "x0": 0, "tile_px": 512, "status": "failed"},
    ]
    cal.decision = {
        "status": "ok",
        "n_tiles": 3,
        "n_pass": 2,
        "required": 3,
        "gate": 0.01,
        "margin_pass": 0.05,
        "warnings": [],
        "timing_s": {"total_s": 12.0},
        "library_spread": {
            "summary": {"z1_ring_radius_px": {"min": 28.0, "max": 29.0}}
        },
        "classifier": {
            "recall": 0.97,
            "n_recall_reference": 40,
            "recall_with_clusters": 0.9,
            "n_recall_reference_with_clusters": 60,
        },
    }
    cal.tissue = {"z_mid": 2, "threshold_counts": 100.0, "tissue_fraction": 0.4}
    cal.forecast = {"px_per_um": 9.26, "wall_hours": 1.0, "backend": "cpu"}
    return cal


# ---------------------------------------------------------------------------
# t-scores and the residual-ring score
def _bead_image(
    size: int, beads: list[tuple[int, int, float]], seed: int = 0
) -> np.ndarray:
    """Noise plus the z1 PSF of each bead (y, x, A0)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(500.0, 20.0, (size, size)).astype(np.float32)
    half = 150
    kernel = bs.render_bead_psf(1, 2 * half + 1)
    for y, x, a0 in beads:
        y0, x0 = y - half, x - half
        ya, xa = max(y0, 0), max(x0, 0)
        yb, xb = min(y0 + 2 * half + 1, size), min(x0 + 2 * half + 1, size)
        img[ya:yb, xa:xb] += a0 * kernel[ya - y0 : yb - y0, xa - x0 : xb - x0]
    return img


def test_ring_and_disc_scores_see_a_bead_and_not_noise() -> None:
    img = _bead_image(600, [(150, 150, 3000.0)])
    ys, xs = np.array([150.0, 400.0]), np.array([150.0, 420.0])
    ring = qc._scores(img, ys, xs, np.full(2, R1_PX), "ring")
    disc = qc._scores(img, ys, xs, np.full(2, R1_PX), "disc")
    assert ring[0] > 10 and disc[0] > 10
    assert abs(ring[1]) < qc.VISIBLE_T and abs(disc[1]) < qc.VISIBLE_T


def test_disc_scores_are_nan_where_the_window_leaves_the_image() -> None:
    img = np.ones((100, 100), np.float32)
    out = qc.disc_tscores(img, np.array([10.0, 50.0]), np.array([50.0, 50.0]), 20.0)
    assert np.isnan(out[0]) and out[1] == 0.0
    big = np.ones((200, 200), np.float32)
    big[100, 100] = np.nan
    assert np.isnan(qc.disc_tscores(big, np.array([100.0]), np.array([100.0]), 20.0)[0])


def _table(rows: list[tuple[float, ...]]) -> np.ndarray:
    return np.array(rows, np.float64)


def _row(
    y: float,
    x: float,
    a0: float,
    *,
    blob: float = 0.0,
    ell: float = 0.0,
    pruned: float = 0.0,
) -> tuple[float, ...]:
    return (y, x, a0, a0, blob, 1.0, ell, pruned)


def test_residual_ring_scores_pick_bright_isolated_beads_only() -> None:
    beads = [
        _row(150, 150, 3000),  # scored
        _row(400, 420, 5000),  # scored
        _row(450, 150, 2000),  # scored
        _row(300, 300, 4000),  # a neighbour within 40 px: both are excluded
        _row(300, 330, 4000),
        _row(250, 450, 1000),  # not bright enough
        _row(30, 300, 4000),  # too close to the edge
        _row(100, 450, 4000, pruned=1.0),  # not a bead
        _row(500, 480, 4000, blob=1.0),  # an aggregate
    ]
    specs = [(int(r[0]), int(r[1]), r[2]) for r in beads[:6]]
    raw = _bead_image(600, specs)
    corrected = _bead_image(600, [], seed=1)
    out = qc.residual_ring_scores(
        raw,
        corrected,
        _table(beads),
        origin_yx=(0, 0),
        r1_px=R1_PX,
        ell_default=0.0,
    )
    assert out["n_candidates"] == 3
    for mode in ("ring", "disc"):
        assert out[mode]["n"] == 3
        assert out[mode]["vis_raw"] == 3
        assert out[mode]["vis_corr"] == 0
        assert out[mode]["t_raw_median"] > out[mode]["t_corr_median"]
    assert out["control"]["n"] > 0
    # The cap keeps the brightest.
    capped = qc.residual_ring_scores(
        raw,
        corrected,
        _table(beads),
        origin_yx=(0, 0),
        r1_px=R1_PX,
        ell_default=0.0,
        max_beads=1,
    )
    assert capped["n_candidates"] == 3 and capped["ring"]["n"] == 1


def test_residual_ring_scores_use_the_fitted_stretch_and_the_origin() -> None:
    # A bead whose ring is stretched by exp(0.2) is only seen at the stretched radius.
    img = np.full((600, 600), 500.0, np.float32)
    yy, xx = np.mgrid[:600, :600]
    radius = np.hypot(yy - 300, xx - 300)
    stretch = float(np.exp(0.2))
    img += 400.0 * (np.abs(radius - R1_PX * stretch) < 3.0)
    bead = _table([_row(1300, 2300, 3000, ell=0.2)])
    kwargs: dict[str, Any] = {"origin_yx": (1000, 2000), "r1_px": R1_PX}
    right = qc.residual_ring_scores(
        img, img, bead, ell_default=0.0, n_control=0, **kwargs
    )
    unstretched = bead.copy()
    unstretched[:, 6] = 0.0
    wrong = qc.residual_ring_scores(
        img, img, unstretched, ell_default=0.0, n_control=0, **kwargs
    )
    assert right["ring"]["n"] == 1 and wrong["ring"]["n"] == 1
    assert right["ring"]["t_raw_median"] > 3 > wrong["ring"]["t_raw_median"]
    # A table without ell uses the default stretch.
    nan_ell = bead.copy()
    nan_ell[:, 6] = np.nan
    by_default = qc.residual_ring_scores(
        img, img, nan_ell, ell_default=0.2, n_control=0, **kwargs
    )
    assert by_default["ring"]["t_raw_median"] == right["ring"]["t_raw_median"]


def test_residual_ring_scores_of_a_small_or_empty_image() -> None:
    small = np.zeros((100, 100), np.float32)
    out = qc.residual_ring_scores(
        small, small, np.zeros((0, 8)), origin_yx=(0, 0), r1_px=R1_PX, ell_default=0
    )
    assert out["n_candidates"] == 0 and out["ring"]["n"] == 0
    big = _bead_image(400, [])
    empty = qc.residual_ring_scores(
        big, big, np.zeros((0, 8)), origin_yx=(0, 0), r1_px=R1_PX, ell_default=0
    )
    assert empty["n_candidates"] == 0 and empty["control"]["n"] > 0


def test_pool_ring_scores_sums_the_counts() -> None:
    def chunk(n: int, vis_corr: int) -> dict[str, Any]:
        part = {"n": n, "vis_raw": n, "vis_corr": vis_corr}
        return {
            "n_candidates": n + 2,
            "ring": dict(part),
            "disc": dict(part),
            "control": {"n": 10, "ring_vis_raw": 1, "disc_vis_raw": 0},
        }

    pooled = qc.pool_ring_scores([chunk(10, 1), None, {}, chunk(30, 0)])
    assert pooled["n_chunks"] == 2 and pooled["n_candidates"] == 44
    assert pooled["ring"]["n"] == 40
    assert pooled["ring"]["frac_corr"] == pytest.approx(1 / 40)
    assert pooled["ring"]["frac_raw"] == 1.0
    assert pooled["control"]["ring_frac_raw"] == pytest.approx(0.1)
    empty = qc.pool_ring_scores([])
    assert empty["ring"]["frac_corr"] is None


# ---------------------------------------------------------------------------
# the per-chunk probe
def test_chunk_probe_block_maximum_and_ring_scores() -> None:
    rng = np.random.default_rng(1)
    window = (64, 64 + 320)
    stacks = {
        s: {z: rng.normal(1000, 5, (320, 320)).astype(np.float32) for z in ZS}
        for s in STAINS
    }
    core = (96, 96 + 256, 96, 96 + 256)
    raw_proj = np.max([stacks["PolyT"][z][32:288, 32:288] for z in ZS], axis=0)
    corrected = np.rint(raw_proj - 7).astype(np.uint16)
    corrected[0:16, 0:16] -= 100  # one block that changed more
    dapi = np.rint(np.max([stacks["DAPI"][z][32:288, 32:288] for z in ZS], 0))
    out = qc.chunk_qc_probe(
        stacks=stacks,
        window_origin=window,
        core=core,
        z_layers=ZS,
        projections={"PolyT": corrected, "DAPI": dapi.astype(np.uint16)},
        beads=np.zeros((0, 8)),
        calibration=_calibration(),
        index=0,
    )
    change = out["change"]["PolyT"]
    assert change.shape == (16, 16) and change.dtype == np.float32
    expected = np.maximum(raw_proj - corrected.astype(np.float32), 0)
    np.testing.assert_array_equal(
        change, expected.reshape(16, 16, 16, 16).max(axis=(1, 3))
    )
    assert change[0, 0] > 100
    assert out["change"]["DAPI"].max() <= 0.5
    assert out["ring"]["ring"]["n"] == 0


def test_block_max_pads_a_partial_edge_block() -> None:
    diff = np.zeros((20, 33), np.float32)
    diff[19, 32] = 5.0
    out = qc._block_max(diff)
    assert out.shape == (2, 3) and out[1, 2] == 5.0 and out.sum() == 5.0


# ---------------------------------------------------------------------------
# warnings
def _stats(**model: Any) -> dict[str, Any]:
    return {"model": model}


def _ring(n: int, ring_corr: float, disc_corr: float) -> dict[str, Any]:
    return {
        "ring": {"n": n, "frac_corr": ring_corr, "frac_raw": 0.9},
        "disc": {"n": n, "frac_corr": disc_corr, "frac_raw": 0.9},
    }


def _codes(warnings: list[qc.QcWarning]) -> list[str]:
    return [w.code for w in warnings]


def test_no_warning_for_a_clean_build() -> None:
    assert (
        qc.collect_qc_warnings(
            calibration=_calibration(),
            chunk_stats=[_stats(n_saturated_px=0)],
            ring=_ring(100, 0.01, 0.1),
        )
        == []
    )


def test_calibration_warnings_are_passed_through_with_codes() -> None:
    cal = _calibration()
    cal.decision["warnings"] = [
        "low_tissue_tiles: fewer than 3 tiles are >= 95 % tissue-free",
        "cost forecast 14.0 h wall on the cpu backend (5 CPU workers)",
        "classifier recall including beads in clusters is 0.600 (< 0.75; 80 "
        "held-out reference beads)",
        "held-out classifier recall is 0.800 (< 0.90; 40 isolated reference beads)",
        "held-out classifier recall could not be measured: no reference bead",
        "stain Anti-X shows a bead response of 0.030 of the reference",
        "something new",
    ]
    out = qc.collect_qc_warnings(calibration=cal, chunk_stats=[], ring=None)
    assert _codes(out) == [
        "low_tissue_tiles",
        "forecast",
        "classifier_recall",
        "classifier_recall",
        "classifier_recall",
        "antibody_rho",
        "calibration",
    ]
    assert out[0].message.startswith("low_tissue_tiles")


def test_saturation_margin_fill_and_residual_ring_warnings() -> None:
    cal = _calibration()
    cal.decision["margin_pass"] = 0.0117
    stats = [
        {
            **_stats(n_saturated_px=3),
            "fill": {"PolyT": {"cand_px": 100.0, "dense_skipped_px": 60.0}},
        },
        {
            **_stats(n_saturated_px=0),
            "fill": {"PolyT": {"cand_px": 100.0, "dense_skipped_px": 20.0}},
        },
    ]
    out = qc.collect_qc_warnings(
        calibration=cal, chunk_stats=stats, ring=_ring(50, 0.08, 0.3)
    )
    assert _codes(out) == [
        "saturation",
        "calibration_margin",
        "residual_ring",
        "residual_disc",
    ]
    stats[0]["fill"]["PolyT"]["dense_skipped_px"] = 150.0
    out = qc.collect_qc_warnings(calibration=cal, chunk_stats=stats, ring=None)
    assert _codes(out) == ["saturation", "calibration_margin", "fill_guard"]
    assert "premise guard" in out[-1].message
    # Up to half of the candidates skipped is not a warning.
    stats[0]["fill"]["PolyT"]["dense_skipped_px"] = 80.0
    out = qc.collect_qc_warnings(calibration=cal, chunk_stats=stats, ring=None)
    assert "fill_guard" not in _codes(out)


def test_residual_ring_warnings_need_enough_scored_beads() -> None:
    few = _ring(qc.MIN_SCORED_FOR_WARNING - 1, 0.5, 0.9)
    assert (
        qc.collect_qc_warnings(calibration=_calibration(), chunk_stats=[], ring=few)
        == []
    )
    at_limit = _ring(qc.MIN_SCORED_FOR_WARNING, qc.RING_WARN_ABOVE, qc.DISC_WARN_ABOVE)
    assert (
        qc.collect_qc_warnings(
            calibration=_calibration(), chunk_stats=[], ring=at_limit
        )
        == []
    )


# ---------------------------------------------------------------------------
# tables
def test_beads_chunk_index_and_chunk_rows() -> None:
    cores = [(0, 100, 0, 100), (0, 100, 100, 200), (100, 200, 0, 100)]
    beads = _table(
        [_row(10, 10, 500), _row(50, 150, 500), _row(150, 50, 500), _row(900, 9, 500)]
    )
    assert qc.beads_chunk_index(beads, cores).tolist() == [0, 1, 2, -1]
    stats = [
        {
            "index": 0,
            "core": [0, 100, 0, 100],
            "model": {"n_cand": 5, "n_bead": 3, "n_saturated_px": 0},
            "per_stain": {
                "PolyT": {
                    "frac_gt10": [0.1, 0.2],
                    "frac_gt100": [0.0, 0.01],
                    "removed_sum": [1.0, 2.0],
                    "removed_p99": [3.0, 4.0],
                }
            },
            "ring_score": {
                "ring": {"n": 4, "vis_raw": 4, "vis_corr": 1},
                "disc": {"n": 4, "vis_raw": 3, "vis_corr": 0},
            },
            "wall_s": 2.0,
        },
        {"index": 1, "core": [0, 100, 100, 200], "model": {}},
    ]
    rows = qc.chunk_rows(stats, [1, 2])
    assert rows[0]["PolyT_z2_frac_gt100"] == 0.01
    assert rows[0]["ring_vis_corr"] == 1 and rows[0]["n_candidates"] == 5
    assert rows[1]["core_x0"] == 100 and "ring_n" not in rows[1]


def test_chunk_rows_gain_the_gpu_seconds_only_for_gpu_runs() -> None:
    cpu = {"index": 0, "core": [0, 1, 0, 1], "model": {}, "wall_s": 3.0}
    gpu = {**cpu, "index": 1, "gpu_s": 5.5, "device": "cuda"}
    assert "gpu_s" not in qc.chunk_rows([cpu], [1])[0]
    rows = qc.chunk_rows([gpu, gpu], [1])
    assert [r["gpu_s"] for r in rows] == [5.5, 5.5]


def test_the_qc_log_reports_the_gpu_lock_wait_and_seconds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summary: dict[str, Any] = {
        "status": "pass",
        "warnings": [],
        "n_chunks": 2,
        "n_beads_modelled": 10,
        "beads_per_mm2_a0_400": 1.0,
        "calibration": {},
        "residual_ring": {},
        "gpu": {
            "lock_wait_s": 12.5,
            "locked": True,
            "gpu_s": 11.0,
            "gpu_s_per_window": {"median": 5.5, "max": 6.0},
        },
    }
    with caplog.at_level(logging.INFO, logger=qc.logger.name):
        qc._log_summary(summary)
    text = caplog.text
    assert "waited 12.5 s for the GPU lock (lock file set)" in text
    assert "11.0 s of GPU work, 5.5 s per window (median), 6.0 s (max)" in text
    caplog.clear()
    del summary["gpu"]
    with caplog.at_level(logging.INFO, logger=qc.logger.name):
        qc._log_summary(summary)
    assert "GPU backend" not in caplog.text


def test_beads_table_is_gzipped_and_capped(tmp_path: Path) -> None:
    beads = _table([_row(10.04, 20.06, 1500), _row(30, 40, 100, pruned=1.0)])
    info = qc.write_beads_table(tmp_path / "b.csv.gz", beads, np.array([0, 1]))
    assert info == {"rows": 2, "truncated": False}
    with gzip.open(tmp_path / "b.csv.gz", "rt") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "y,x,A0,a,blob,gamma,ell,pruned,chunk"
    assert lines[1].startswith("10.0,20.1,1500") and lines[2].endswith(",1,1")
    capped = qc.write_beads_table(
        tmp_path / "c.csv.gz", beads, np.array([0, 1]), max_mb=0.0
    )
    assert capped == {"rows": 1, "truncated": True}


def test_json_output_has_no_nan(tmp_path: Path) -> None:
    qc._write_json(
        tmp_path / "x.json", {"a": float("nan"), "b": [np.inf, np.float32(1)]}
    )
    text = (tmp_path / "x.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [None, 1.0]}


# ---------------------------------------------------------------------------
# crops and maps
def test_select_qc_locations_prefers_dense_tiles_and_chunks() -> None:
    cal = _calibration()
    shape = (4096, 4096)
    cores = [(0, 2048, 0, 2048), (0, 2048, 2048, 4096), (2048, 4096, 0, 2048)]
    valid = np.ones((256, 256), bool)
    tissue = np.zeros_like(valid)
    tissue[:128, 128:] = True  # chunk 1 is tissue
    tissue[128:, :128] = True  # chunk 2 is tissue
    rng = np.random.default_rng(0)
    bright = [
        _row(float(y), float(x), 3000.0)
        for y, x in zip(
            rng.uniform(300, 1700, 40), rng.uniform(2300, 3800, 40), strict=True
        )
    ] + [
        _row(float(y), float(x), 3000.0)
        for y, x in zip(
            rng.uniform(2300, 3800, 10), rng.uniform(300, 1700, 10), strict=True
        )
    ]
    locations = qc.select_qc_locations(
        calibration=cal,
        beads=_table(bright),
        cores=cores,
        tissue=tissue,
        valid=valid,
        mosaic_shape=shape,
        crop_px=512,
    )
    kinds = [loc.kind for loc in locations]
    assert kinds == ["calibration", "calibration", "tissue", "tissue"]
    # The failed tile is never used; the densest tile comes first.
    assert (locations[0].y0, locations[0].x0) == (0, 0)
    assert (locations[1].y0, locations[1].x0) == (0, 1024)
    assert all(loc.y0 % 16 == 0 and loc.x0 % 16 == 0 for loc in locations)
    assert all(
        0 <= loc.y0 <= 4096 - 512 and 0 <= loc.x0 <= 4096 - 512 for loc in locations
    )
    for i, a in enumerate(locations):
        for b in locations[i + 1 :]:
            assert abs(a.y0 - b.y0) >= 512 or abs(a.x0 - b.x0) >= 512
    # The densest tissue chunk is the one on the right, then the lower left one.
    assert locations[2].x0 >= 2048 and locations[2].y0 < 2048
    assert locations[3].y0 >= 2048 and locations[3].x0 < 2048


def test_select_qc_locations_without_beads_or_tissue() -> None:
    cal = _calibration()
    cal.tiles = []
    out = qc.select_qc_locations(
        calibration=cal,
        beads=np.zeros((0, 8)),
        cores=[(0, 1024, 0, 1024)],
        tissue=np.zeros((64, 64), bool),
        valid=np.ones((64, 64), bool),
        mosaic_shape=(1024, 1024),
    )
    assert out == []


def test_density_and_stretch_maps() -> None:
    beads = _table(
        [
            _row(100, 100, 500, ell=0.1),
            _row(120, 140, 2000, ell=0.3),
            _row(100, 100, 5000, pruned=1.0),
            _row(900, 900, 500, ell=float("nan")),
        ]
    )
    valid = np.ones((64, 64), bool)
    valid[:32, 32:] = False
    dens = qc.bead_density_map(beads, valid, (1024, 1024), 10.0, min_a0=400.0)
    assert dens.shape == (2, 2)
    area_mm2 = (512 / 10.0 / 1000.0) ** 2
    assert dens[0, 0] == pytest.approx(2 / area_mm2)
    assert np.isnan(dens[0, 1]) and dens[1, 1] == pytest.approx(1 / area_mm2)
    bright = qc.bead_density_map(beads, valid, (1024, 1024), 10.0, min_a0=1600.0)
    assert bright[0, 0] == pytest.approx(1 / area_mm2)
    ell = qc.stretch_field_map(beads, (1024, 1024), cell=256)
    assert ell.shape == (4, 4) and ell[0, 0] == pytest.approx(0.2)
    assert np.isnan(ell[3, 3])


# ---------------------------------------------------------------------------
# a tiny synthetic build
@pytest.fixture(scope="module")
def qc_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("qc_run")
    stack = bs.make_stack(n_planes=4, size=SHAPE, n_beads=70, seed=11, n_nuclei=12)
    write_mosaic(folder / "images", stack, 4)
    calibration = _calibration()
    cal_dir = folder / "cal"
    calibration.save(cal_dir)
    store = open_projection_store(
        folder / "p.zarr", n_channels=2, shape=SHAPE, chunk_px=1024
    )
    fill = FillParams(enabled=True, stains=("PolyT",), reference_stain="PolyT")
    run = driver.run_bead_removal(
        images_dir=folder / "images",
        stains=STAINS,
        z_layers=ZS,
        store=store,
        calibration=calibration,
        calibration_dir=cal_dir,
        work_dir=folder / "work",
        fill=fill,
        core_px=1024,
        max_workers=1,
        rss_per_worker_gb=1.0,
        memory_available_gb=1000.0,
    )
    qc_dir = folder / qc.QC_DIRNAME
    summary = qc.write_bead_removal_qc(
        qc_dir,
        calibration=calibration,
        run=run,
        images_dir=folder / "images",
        projection=store,
        stainings=STAINS,
        z_layers=ZS,
        fill=fill,
        crop_px=512,
    )
    return {
        "folder": folder,
        "stack": stack,
        "store": store,
        "run": run,
        "summary": summary,
        "qc_dir": qc_dir,
        "calibration": calibration,
    }


EXPECTED_FILES = {
    "summary.json",
    "calibration.json",
    "psf_library.npz",
    "psf_montage.png",
    "overview.png",
    "beads.csv.gz",
    "bead_density.png",
    "stretch_field.png",
    "chunks.csv",
    "before_after.png",
    "before_after_dapi.png",
    "before_after_crops.npz",
    "change_overview.png",
    "residual_profiles.json",
    "residual_profiles.png",
}


def _strict_json(path: Path) -> Any:
    def _reject(token: str) -> None:
        raise ValueError(f"non-finite number {token} in {path.name}")

    return json.loads(path.read_text(), parse_constant=_reject)


def test_qc_folder_holds_every_product_and_stays_small(qc_run: dict[str, Any]) -> None:
    files = {p.name for p in qc_run["qc_dir"].iterdir()}
    assert files == EXPECTED_FILES
    total = sum(p.stat().st_size for p in qc_run["qc_dir"].iterdir())
    assert total < 50e6
    for png in qc_run["qc_dir"].glob("*.png"):
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png.name
    for name in ("summary.json", "calibration.json", "residual_profiles.json"):
        _strict_json(qc_run["qc_dir"] / name)
    assert qc_run["summary"]["errors"] == []
    assert qc_run["summary"]["total_size_mb"] == pytest.approx(total / 1e6, rel=0.01)


def test_summary_reports_identity_runtime_and_calibration(
    qc_run: dict[str, Any],
) -> None:
    summary = _strict_json(qc_run["qc_dir"] / "summary.json")
    cal: BeadCalibration = qc_run["calibration"]
    run = qc_run["run"]
    assert summary["method"] == "psf_subtract" and summary["version"] == 1
    assert summary["params_sha256"] == cal.params_sha256
    assert summary["calibration_sha256"] == cal.calibration_sha256()
    assert summary["z_layers"] == ZS and summary["corrected_stains"] == STAINS
    assert summary["n_chunks"] == run.n_chunks == 2
    beads = np.load(Path(run.work_dir) / "beads.npy")
    assert summary["n_beads_listed"] == len(beads)
    assert summary["n_beads_modelled"] == int((beads[:, 7] == 0).sum())
    assert summary["runtime_s"]["calibration"] == 12.0
    assert summary["runtime_s"]["chunks"] == pytest.approx(run.wall_s)
    assert summary["n_workers"] == 1 and summary["backend"]["device"] == "cpu"
    assert "gpu" not in summary, "the GPU record belongs to GPU runs only"
    assert summary["peak_rss_gb"]["worker"] == pytest.approx(run.peak_worker_rss_gb)
    assert summary["beads_per_mm2_a0_400"] >= summary["beads_per_mm2_a0_1600"] > 0
    assert summary["status"] == ("warn" if summary["warnings"] else "pass")
    # The decision margins, the library spread and both recalls are reported.
    assert summary["calibration"]["margin_pass"] == 0.05
    assert summary["library_spread"]["z1_ring_radius_px"]["max"] == 29.0
    assert summary["classifier"]["recall"] == 0.97
    assert summary["classifier"]["recall_with_clusters"] == 0.9
    assert set(summary["calibration"]["rho"]) == {"DAPI", "Anti-X"}
    assert summary["fill"]["enabled"] is True
    assert "cand_px" in summary["fill"]["counters"]["PolyT"]
    assert set(summary["files"]) == EXPECTED_FILES


def test_calibration_json_compares_the_library_with_its_laws(
    qc_run: dict[str, Any],
) -> None:
    record = _strict_json(qc_run["qc_dir"] / "calibration.json")
    assert [r["plane"] for r in record["library_vs_law"]] == list(range(8))
    assert record["library_vs_law"][3]["flux_target"] == pytest.approx(0.90)
    assert record["calibration_sha256"] == qc_run["calibration"].calibration_sha256()
    assert record["rho"]["DAPI"][0] == pytest.approx(0.11)
    lib = np.load(qc_run["qc_dir"] / "psf_library.npz")
    assert "vals1" in lib.files


def test_beads_and_chunks_tables_match_the_run(qc_run: dict[str, Any]) -> None:
    run = qc_run["run"]
    beads = np.load(Path(run.work_dir) / "beads.npy")
    with gzip.open(qc_run["qc_dir"] / "beads.csv.gz", "rt") as handle:
        rows = handle.read().splitlines()[1:]
    assert len(rows) == len(beads)
    chunk_of_bead = {int(r.rsplit(",", 1)[1]) for r in rows}
    assert chunk_of_bead <= {0, 1}
    with (qc_run["qc_dir"] / "chunks.csv").open() as handle:
        header = handle.readline().strip().split(",")
        n_rows = sum(1 for _ in handle)
    assert n_rows == 2
    for column in (
        "core_y0",
        "n_candidates",
        "n_beads",
        "n_blobs",
        "n_pruned",
        "gamma_med",
        "ell_sd",
        "wall_s",
        "peak_rss_gb",
        "PolyT_z1_frac_gt10",
        "PolyT_z3_frac_gt100",
        "DAPI_z2_removed_sum",
        "PolyT_z1_removed_p99",
        "polyt_px_saturated",
        "ring_vis_raw",
        "disc_vis_corr",
    ):
        assert column in header, column


def test_chunk_probe_output_is_in_the_run_record(qc_run: dict[str, Any]) -> None:
    for stats in qc_run["run"].chunk_stats:
        assert set(stats["ring_score"]) >= {"ring", "disc", "control", "n_candidates"}
        assert stats["t_qc_s"] >= 0
        json.dumps(stats)
    ring = qc_run["summary"]["residual_ring"]
    assert ring["n_chunks"] == 2
    assert ring["ring"]["n"] > 0
    assert ring["ring"]["frac_raw"] >= ring["ring"]["frac_corr"]
    assert ring["control"]["n"] > 0


def test_change_overview_is_the_block_maximum_of_raw_minus_corrected(
    qc_run: dict[str, Any],
) -> None:
    run = qc_run["run"]
    overview = qc.load_change_overview(Path(run.work_dir))
    assert set(overview) == set(STAINS)
    store = np.asarray(qc_run["store"][:])
    for index, stain in enumerate(STAINS):
        raw = raw_max(qc_run["stack"], stain, ZS).astype(np.float32)
        diff = np.maximum(raw - store[index].astype(np.float32), 0.0)
        expected = diff.reshape(SHAPE[0] // 16, 16, SHAPE[1] // 16, 16).max(axis=(1, 3))
        np.testing.assert_array_equal(overview[stain], expected)
    assert overview["PolyT"].max() > 100


def test_before_after_crops_hold_the_builds_own_output(qc_run: dict[str, Any]) -> None:
    summary = qc_run["summary"]
    locations = summary["crops"]["locations"]
    assert summary["crops"]["refit_windows"] == len(locations) >= 1
    assert {loc["kind"] for loc in locations} <= {"calibration", "tissue"}
    crops = np.load(qc_run["qc_dir"] / "before_after_crops.npz")
    store = np.asarray(qc_run["store"][:])
    for r in range(len(locations)):
        y0, x0 = crops[f"loc{r}_origin_yx"]
        raw = crops[f"loc{r}_PolyT_raw"]
        built = crops[f"loc{r}_PolyT_corrected"]
        assert raw.dtype == built.dtype == np.uint16
        h, w = raw.shape
        np.testing.assert_array_equal(built, store[0, y0 : y0 + h, x0 : x0 + w])
        np.testing.assert_array_equal(
            raw, raw_max(qc_run["stack"], "PolyT", ZS)[y0 : y0 + h, x0 : x0 + w]
        )
        assert (built <= raw).all()
    assert "loc0_DAPI_raw" in crops.files


def test_residual_profiles_show_the_bead_before_and_not_after(
    qc_run: dict[str, Any],
) -> None:
    profiles = _strict_json(qc_run["qc_dir"] / "residual_profiles.json")
    assert profiles["unit"] == "1e-3 x A0"
    groups = profiles["groups"]
    n_total = sum(g["n_beads"] for g in groups.values())
    assert n_total > 0
    for group in groups.values():
        for z, plane in group["planes"].items():
            assert int(z) in (1, 2, 3)
            assert len(plane["raw"]["profile"]) == len(profiles["radius_px"])
            assert plane["raw"]["disc"] > plane["corrected"]["disc"]
    assert qc_run["summary"]["residual_profiles"] == {
        k: g["n_beads"] for k, g in groups.items()
    }


# ---------------------------------------------------------------------------
# the builder
def _raw_folder(tmp_path: Path) -> Path:
    stack = bs.make_stack(n_planes=4, size=768, n_beads=20, seed=3, n_nuclei=6)
    raw = tmp_path / "raw"
    write_mosaic(raw / "images", stack, 4, stains=("PolyT", "DAPI", "Anti-X"))
    return raw


def _patch_build(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Fake the SpatialData reader and writer; the writer creates the zarr folder."""
    from .test_spatialdata_build import _fake_spatialdata_io

    _fake_spatialdata_io(monkeypatch)
    written: list[Path] = []

    def _write(sdata: Any, path: Path, overwrite: bool | None = None) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        written.append(Path(path))

    monkeypatch.setattr("merxen.io.builders.merscope.write_spatialdata_zarr", _write)
    monkeypatch.setattr(
        "merxen.io.bead_removal.calibrate.calibrate_bead_model",
        lambda reader, **kwargs: _calibration(),
    )
    return written


def test_builder_writes_the_qc_folder_into_the_zarr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written = _patch_build(monkeypatch)
    raw = _raw_folder(tmp_path)
    seen: dict[str, Any] = {}
    real_write = qc.write_bead_removal_qc

    def _spy(qc_dir: Path, **kwargs: Any) -> dict[str, Any]:
        seen["fill"] = kwargs["fill"]
        seen["stainings"] = kwargs["stainings"]
        return real_write(qc_dir, **kwargs)

    monkeypatch.setattr("merxen.io.bead_removal.qc.write_bead_removal_qc", _spy)

    write_merscope_spatialdata(
        input_path=raw,
        output_path=tmp_path / "source.zarr",
        build_config=MerscopeBuildConfig(
            z_layers=ZS,
            bead_removal=MerscopeBeadRemovalConfig(enabled=True, max_workers=1),
        ),
    )

    qc_dir = written[0] / qc.QC_DIRNAME
    assert {p.name for p in qc_dir.iterdir()} >= {"summary.json", "chunks.csv"}
    summary = json.loads((qc_dir / "summary.json").read_text())
    assert summary["n_chunks"] == 1 and summary["errors"] == []
    assert seen["fill"] is None and seen["stainings"]
    # The temporary projection is gone once the build is done.
    assert list(tmp_path.glob("merscope_projection_*")) == []


def _result_with_run(tmp_path: Path) -> ProjectionBuildResult:
    """A projection result of a bead build with a real (tiny) temporary store."""
    import zarr

    tmp_path.mkdir(parents=True, exist_ok=True)
    zarr.open_array(
        str(tmp_path / merscope_module._PROJECTION_STORE_NAME),
        mode="w",
        shape=(1, 32, 32),
        chunks=(1, 32, 32),
        dtype="uint16",
    )
    run = driver.BeadRemovalRunResult(
        n_chunks=0,
        n_workers=1,
        wall_s=0.0,
        worker_cpu_s=0.0,
        worker_wall_s=0.0,
        read_s=0.0,
        correct_s=0.0,
        project_s=0.0,
        parent_write_s=0.0,
        n_beads=0,
        peak_tree_rss_gb=0.0,
        peak_tree_anon_gb=0.0,
        peak_worker_rss_gb=0.0,
        peak_worker_anon_gb=0.0,
        work_dir=str(tmp_path),
    )
    return ProjectionBuildResult(
        object(),
        tmp_path,
        stainings=("PolyT",),
        bead_calibration=_calibration(),
        bead_removal=run,
    )


def test_qc_failure_is_logged_and_does_not_fail_the_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    zarr_path = tmp_path / "source.zarr"
    zarr_path.mkdir()
    (zarr_path / qc.QC_DIRNAME).mkdir()
    (zarr_path / qc.QC_DIRNAME / "stale.txt").write_text("from an earlier build")
    projection = _result_with_run(tmp_path / "temp")

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("qc exploded")

    monkeypatch.setattr("merxen.io.bead_removal.qc.write_bead_removal_qc", _boom)
    with caplog.at_level(logging.ERROR, logger=merscope_module.logger.name):
        merscope_module._write_bead_removal_qc(
            images_dir=tmp_path,
            output_path=zarr_path,
            projection=projection,
            z_layers=ZS,
            config=MerscopeBeadRemovalConfig(enabled=True),
        )
    assert any("QC failed" in r.getMessage() for r in caplog.records)
    assert any(r.exc_info for r in caplog.records)
    # An earlier build's QC never survives next to a new build.
    assert not (zarr_path / qc.QC_DIRNAME).exists()


def test_default_build_never_writes_a_qc_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from .test_spatialdata_build import _fake_spatialdata_io

    _fake_spatialdata_io(monkeypatch)
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: ProjectionBuildResult(
            object(), None, stainings=("PolyT", "DAPI")
        ),
    )
    monkeypatch.setattr(
        "merxen.io.builders.merscope.write_spatialdata_zarr",
        lambda sdata, path, overwrite=None: Path(path).mkdir(parents=True),
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        merscope_module,
        "_write_bead_removal_qc",
        lambda **kwargs: calls.append(kwargs),
    )
    write_merscope_spatialdata(
        input_path=tmp_path,
        output_path=tmp_path / "source.zarr",
        build_config=MerscopeBuildConfig(z_layers=ZS),
    )
    assert calls == []
    assert not (tmp_path / "source.zarr" / qc.QC_DIRNAME).exists()


def test_qc_helper_skips_a_missing_zarr_and_a_run_without_beads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = MerscopeBeadRemovalConfig(enabled=True)
    plain = ProjectionBuildResult(object(), tmp_path, stainings=("PolyT",))
    merscope_module._write_bead_removal_qc(
        images_dir=tmp_path,
        output_path=tmp_path / "missing.zarr",
        projection=plain,
        z_layers=ZS,
        config=config,
    )
    full = _result_with_run(tmp_path / "temp")
    with caplog.at_level(logging.WARNING, logger=merscope_module.logger.name):
        merscope_module._write_bead_removal_qc(
            images_dir=tmp_path,
            output_path=tmp_path / "missing.zarr",
            projection=full,
            z_layers=ZS,
            config=config,
        )
    assert any("does not exist" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "missing.zarr").exists()


# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_probe_results_do_not_depend_on_the_worker_count(
    qc_run: dict[str, Any], tmp_path: Path
) -> None:
    folder = qc_run["folder"]
    store = open_projection_store(
        tmp_path / "p2.zarr", n_channels=2, shape=SHAPE, chunk_px=1024
    )
    run2 = driver.run_bead_removal(
        images_dir=folder / "images",
        stains=STAINS,
        z_layers=ZS,
        store=store,
        calibration=qc_run["calibration"],
        calibration_dir=folder / "cal",
        work_dir=tmp_path / "work2",
        fill=FillParams(enabled=True, stains=("PolyT",), reference_stain="PolyT"),
        core_px=1024,
        max_workers=2,
        rss_per_worker_gb=1.0,
        memory_available_gb=1000.0,
    )
    assert run2.n_workers == 2
    one = qc_run["run"]
    assert [s["ring_score"] for s in run2.chunk_stats] == [
        s["ring_score"] for s in one.chunk_stats
    ]
    a = qc.load_change_overview(Path(one.work_dir))
    b = qc.load_change_overview(Path(run2.work_dir))
    for stain in STAINS:
        np.testing.assert_array_equal(a[stain], b[stain])
