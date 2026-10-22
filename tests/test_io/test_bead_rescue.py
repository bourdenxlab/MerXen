"""Tests of the gated z1-ring rescue of the bead classifier (synthetic data only).

The rescue accepts a candidate that fails the z0 ring test on the low side when it
is flat around (a per-dataset texture gate), has no secondary focus at z2..zmax and
shows a coherent raw z1 ring. These tests cover the rule, its gates, the per-dataset
derivation of the gate, the old-calibration contract (the rescue is off and the
hash is unchanged), the planes the readers supply and the QC record.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.io.bead_removal import calibrate as cal_mod
from merxen.io.bead_removal import correct_window, driver, qc
from merxen.io.bead_removal.calibrate import BeadCalibration
from merxen.io.bead_removal.detect import (
    ClassifierThresholds,
    Sources,
    detect_sources,
    dog_noise_of,
)

from . import bead_synth as bs
from .bead_build_fixtures import real_calibration

#: z1 ring radius of the synthetic library (px).
RING_Z1 = 7.0 + 22.8
BEADS = np.array(
    [
        [100.0, 100.0, 3000.0],
        [100.0, 300.0, 4000.0],
        [300.0, 100.0, 2500.0],
        [300.0, 300.0, 3500.0],
        [200.0, 200.0, 6000.0],
    ]
)
#: ``ring_min`` above every bead's z0 ring ratio: each bead takes the rescue path.
STRICT = dataclasses.replace(
    ClassifierThresholds(),
    ring_min=0.95,
    rescue_tz1_min=4.0,
    rescue_ring_radius_px=RING_Z1,
    rescue_sd1_max=0.08,
)


@pytest.fixture(scope="module")
def stack() -> dict[str, np.ndarray]:
    return bs.make_stack(size=420, tissue=False, seed=3, beads=BEADS, margin=40)


def _detect(
    st: dict[str, np.ndarray], th: ClassifierThresholds, n_planes: int = 7
) -> Sources:
    polyt = {k: st["PolyT"][k] for k in range(n_planes)}
    dapi = {0: st["DAPI"][0], 1: st["DAPI"][1]}
    return detect_sources(
        polyt, dapi, dog_noise=dog_noise_of(st["PolyT"][0]), thresholds=th
    )


def _near(src: Sources, y: float, x: float, radius: float = 3.0) -> np.ndarray:
    return np.hypot(src.y - y, src.x - x) < radius


def _with_secondary_focus(
    st: dict[str, np.ndarray], z: int = 3, amp_rel: float = 0.6
) -> dict[str, np.ndarray]:
    """Return the stack with a compact second focus on the bead at (200, 200)."""
    out = {k: st[k].copy() for k in ("PolyT", "DAPI")}
    yy, xx = np.mgrid[: out["PolyT"].shape[1], : out["PolyT"].shape[2]]
    blob = np.exp(-((yy - 200.0) ** 2 + (xx - 200.0) ** 2) / (2 * 3.0**2))
    out["PolyT"][z] += (amp_rel * 6000.0 * blob).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# thresholds: fields, old files, hash
def test_rescue_fields_are_off_by_default_and_round_trip() -> None:
    th = ClassifierThresholds()
    assert not th.rescue_active
    assert th.rescue_sd1_max == 0.0 and th.rescue_zprof_zmax == 0.0
    assert th.rescue_planes == ()
    assert ClassifierThresholds.from_dict(th.to_dict()) == th
    assert ClassifierThresholds.from_dict(STRICT.to_dict()) == STRICT
    assert STRICT.rescue_active


def test_rescue_needs_all_three_switches() -> None:
    assert not dataclasses.replace(STRICT, rescue_tz1_min=0.0).rescue_active
    assert not dataclasses.replace(STRICT, rescue_ring_radius_px=0.0).rescue_active
    assert not dataclasses.replace(STRICT, rescue_sd1_max=0.0).rescue_active


def test_rescue_planes_follow_the_z_profile_switch() -> None:
    assert STRICT.rescue_planes == ()
    on = dataclasses.replace(STRICT, rescue_zprof_zmax=6.0)
    assert on.rescue_planes == (2, 3, 4, 5, 6)
    assert dataclasses.replace(on, rescue_sd1_max=0.0).rescue_planes == ()


@pytest.mark.parametrize("with_antibody_field", [False, True])
def test_a_calibration_without_the_texture_gate_loads_with_the_rescue_off(
    with_antibody_field: bool,
) -> None:
    # A development-version file: the rescue constants without the texture gate
    # and the z-profile switch, and (optionally) the retired antibody-guard field.
    old = ClassifierThresholds().to_dict()
    old.update(rescue_tz1_min=4.0, rescue_ring_radius_px=29.0)
    old.pop("rescue_sd1_max")
    old.pop("rescue_zprof_zmax")
    if with_antibody_field:
        old["rescue_ab_max"] = 0.1
    loaded = ClassifierThresholds.from_dict(old)
    assert loaded.rescue_sd1_max == 0.0 and not loaded.rescue_active
    assert loaded.rescue_tz1_min == 4.0


def test_a_file_from_before_the_rescue_loads_unchanged() -> None:
    old = {
        k: v
        for k, v in ClassifierThresholds().to_dict().items()
        if not k.startswith("rescue_")
    }
    assert ClassifierThresholds.from_dict(old) == ClassifierThresholds()


def _legacy_hash(calibration: BeadCalibration) -> str:
    """Hash with the definition from before the rescue (thresholds without it)."""
    import hashlib

    digest = hashlib.sha256()
    meta = {
        "version": int(calibration.version),
        "params_sha256": calibration.params_sha256,
        "corrected_stains": list(calibration.corrected_stains),
        "reference_stain": calibration.reference_stain,
        "thresholds": {
            k: v
            for k, v in calibration.thresholds.to_dict().items()
            if not k.startswith("rescue_")
        },
        "noise": {str(k): float(v) for k, v in sorted(calibration.noise.items())},
        "dog_noise_z0": float(calibration.dog_noise_z0),
        "a0_sel": [float(calibration.a0_sel[0]), float(calibration.a0_sel[1])],
        "a0_median": float(calibration.a0_median),
        "ell_default": float(calibration.ell_default),
        "rho": {
            k: [float(v[0]), float(v[1])] for k, v in sorted(calibration.rho.items())
        },
    }
    digest.update(json.dumps(meta, sort_keys=True).encode())
    for tmpl in calibration.library.tmpls:
        digest.update(repr((float(tmpl.ds), float(tmpl.r_sup))).encode())
        digest.update(np.ascontiguousarray(tmpl.vals, dtype="<f8").tobytes())
    return digest.hexdigest()


def test_calibration_hash_is_unchanged_while_the_rescue_is_off() -> None:
    cal = real_calibration()
    assert cal.calibration_sha256() == _legacy_hash(cal)
    # constants of an inactive rescue do not change what a chunk does
    inactive = dataclasses.replace(
        cal,
        thresholds=dataclasses.replace(
            cal.thresholds, rescue_tz1_min=4.0, rescue_ring_radius_px=29.0
        ),
    )
    assert inactive.calibration_sha256() == cal.calibration_sha256()


def test_calibration_hash_changes_with_an_active_rescue() -> None:
    cal = real_calibration()
    base = cal.calibration_sha256()
    on = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.05, 6.0))
    )
    other_gate = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.06, 6.0))
    )
    no_profile = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.05, 0.0))
    )
    hashes = {
        base,
        on.calibration_sha256(),
        other_gate.calibration_sha256(),
        no_profile.calibration_sha256(),
    }
    assert len(hashes) == 4


def _rescue_kwargs(sd1_max: float, zmax: float) -> dict[str, float]:
    return {
        "rescue_tz1_min": 4.0,
        "rescue_ring_radius_px": 29.0,
        "rescue_sd1_max": sd1_max,
        "rescue_zprof_zmax": zmax,
    }


def test_the_rescue_survives_a_save_and_load(tmp_path: Path) -> None:
    cal = real_calibration()
    on = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.05, 6.0))
    )
    on.save(tmp_path)
    loaded = BeadCalibration.load(tmp_path)
    assert loaded.thresholds == on.thresholds and loaded.thresholds.rescue_active
    assert loaded.calibration_sha256() == on.calibration_sha256()
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["calibration_sha256"] == on.calibration_sha256()
    assert saved["thresholds"]["rescue_sd1_max"] == 0.05


def test_a_saved_calibration_without_rescue_fields_loads_with_the_rescue_off(
    tmp_path: Path,
) -> None:
    cal = real_calibration()
    cal.save(tmp_path)
    path = tmp_path / "calibration.json"
    data = json.loads(path.read_text())
    data["thresholds"] = {
        k: v for k, v in data["thresholds"].items() if not k.startswith("rescue_")
    }
    path.write_text(json.dumps(data))
    loaded = BeadCalibration.load(tmp_path)
    assert not loaded.thresholds.rescue_active
    assert loaded.calibration_sha256() == cal.calibration_sha256()


# ---------------------------------------------------------------------------
# the rule
def test_a_flat_ring_weak_bead_is_rescued(stack: dict[str, np.ndarray]) -> None:
    without = _detect(stack, dataclasses.replace(STRICT, rescue_sd1_max=0.0))
    assert not without.is_bead.any() and not without.rescued.any()
    src = _detect(stack, STRICT)
    for y, x, _ in BEADS:
        hit = _near(src, y, x)
        assert hit.sum() == 1
        assert src.rescued[hit].all() and src.is_bead[hit].all()
    # the diagnostics are filled for the candidates that reached the ring test
    assert src.sd1_rel is not None and len(src.sd1_rel) == src.n
    assert src.t1 is not None and np.isfinite(src.t1[src.rescued]).all()
    assert (src.t1[src.rescued] >= STRICT.rescue_tz1_min).all()


def test_a_passing_ring_test_is_not_a_rescue(stack: dict[str, np.ndarray]) -> None:
    th = dataclasses.replace(STRICT, ring_min=0.2)
    src = _detect(stack, th)
    assert src.is_bead.sum() >= len(BEADS)
    assert not src.rescued.any()


def test_a_rescue_off_run_equals_the_baseline_classifier(
    stack: dict[str, np.ndarray],
) -> None:
    base = ClassifierThresholds()
    off = dataclasses.replace(
        base, rescue_tz1_min=4.0, rescue_ring_radius_px=RING_Z1, rescue_sd1_max=0.0
    )
    a, b = _detect(stack, base), _detect(stack, off)
    assert np.array_equal(a.is_bead, b.is_bead) and np.array_equal(a.A0, b.A0)
    assert np.array_equal(a.y, b.y) and not b.rescued.any()


def test_a_compact_tissue_object_without_a_z1_ring_is_not_rescued() -> None:
    # A spectrally bead-like punctum, compact at z0, with nothing at z1: it fails
    # the z0 ring test, passes the amplitude, texture and z-profile gates, and is
    # refused by the z1 ring test.
    size = 300
    st = bs.make_stack(
        size=size, tissue=False, seed=5, beads=np.zeros((0, 3)), margin=40
    )
    yy, xx = np.mgrid[:size, :size]
    spot = np.exp(-((yy - 150.0) ** 2 + (xx - 150.0) ** 2) / (2 * 2.0**2))
    st["PolyT"][0] += (3000.0 * spot).astype(np.float32)
    st["DAPI"][0] += (0.11 * 3000.0 * spot).astype(np.float32)
    th = dataclasses.replace(STRICT, rescue_sd1_max=1e9, rescue_zprof_zmax=6.0)
    src = _detect(st, th)
    hit = _near(src, 150.0, 150.0)
    assert hit.sum() == 1
    assert not src.rescued[hit].any() and not src.is_bead[hit].any()
    assert np.isfinite(src.t1[hit]).all() and (src.t1[hit] < th.rescue_tz1_min).all()


def test_the_texture_gate_blocks_candidates_in_textured_surroundings(
    stack: dict[str, np.ndarray],
) -> None:
    src = _detect(stack, STRICT)
    rescued = np.where(src.rescued)[0]
    assert len(rescued) >= len(BEADS)
    top = float(src.sd1_rel[rescued].max())
    # just above the largest sd1_rel of the rescued set nothing changes, just below
    # the candidate with that value is blocked
    above = _detect(stack, dataclasses.replace(STRICT, rescue_sd1_max=top * 1.001))
    below = _detect(stack, dataclasses.replace(STRICT, rescue_sd1_max=top * 0.999))
    assert above.rescued.sum() == len(rescued)
    assert below.rescued.sum() == len(rescued) - 1
    assert not _detect(
        stack, dataclasses.replace(STRICT, rescue_sd1_max=1e-12)
    ).rescued.any()


def test_textured_surroundings_raise_the_gate_statistic(
    stack: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[: stack["PolyT"].shape[1], : stack["PolyT"].shape[2]]
    texture = np.zeros(yy.shape)
    for _ in range(1500):  # compact z1 structures all over the window
        y, x = rng.uniform(0, yy.shape[0]), rng.uniform(0, yy.shape[1])
        amp = rng.uniform(300, 1500)
        texture += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3.0**2))
    textured = {k: stack[k].copy() for k in ("PolyT", "DAPI")}
    textured["PolyT"][1] += texture.astype(np.float32)
    flat, rough = _detect(stack, STRICT), _detect(textured, STRICT)
    for y, x, _ in BEADS:
        f, r = _near(flat, y, x), _near(rough, y, x)
        assert rough.sd1_rel[r][0] > 2 * flat.sd1_rel[f][0]
    assert rough.rescued.sum() < flat.rescued.sum()


def test_the_z_profile_blocks_a_secondary_focus_only_when_on(
    stack: dict[str, np.ndarray],
) -> None:
    st = _with_secondary_focus(stack)
    off = _detect(st, STRICT)
    on = _detect(st, dataclasses.replace(STRICT, rescue_zprof_zmax=6.0))
    assert off.rescued[_near(off, 200.0, 200.0)].all()  # no z-profile test
    assert not on.rescued[_near(on, 200.0, 200.0)].any()  # blocked by the focus at z3
    # every other bead keeps its rescue (flat z profile)
    assert on.rescued.sum() == off.rescued.sum() - 1


def test_the_z_profile_ignores_planes_that_are_not_given(
    stack: dict[str, np.ndarray],
) -> None:
    st = _with_secondary_focus(stack, z=5)
    th = dataclasses.replace(STRICT, rescue_zprof_zmax=6.0)
    full = _detect(st, th, n_planes=7)
    short = _detect(st, th, n_planes=4)  # plane 5 is not in the stack: not tested
    assert not full.rescued[_near(full, 200.0, 200.0)].any()
    assert short.rescued[_near(short, 200.0, 200.0)].all()


def test_the_z_profile_reads_arrays_and_mappings_alike(
    stack: dict[str, np.ndarray],
) -> None:
    st = _with_secondary_focus(stack)
    th = dataclasses.replace(STRICT, rescue_zprof_zmax=6.0)
    as_map = _detect(st, th)
    as_array = detect_sources(
        st["PolyT"],
        st["DAPI"],
        dog_noise=dog_noise_of(st["PolyT"][0]),
        thresholds=th,
    )
    assert np.array_equal(as_map.rescued, as_array.rescued)


def test_a_bead_near_the_image_edge_is_not_rescued_without_a_full_ring_window() -> None:
    # the ring window leaves the image: NaN t-score, never a rescue
    st = bs.make_stack(
        size=200,
        tissue=False,
        seed=2,
        beads=np.array([[30.0, 100.0, 4000.0]]),
        margin=20,
    )
    src = _detect(st, STRICT)
    hit = _near(src, 30.0, 100.0)
    assert hit.sum() == 1 and not src.rescued[hit].any()
    assert np.isnan(src.t1[hit]).all()


# ---------------------------------------------------------------------------
# derivation of the gate
def _ref(n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "rring": rng.uniform(0.45, 0.7, n),
        "r1": rng.uniform(0.05, 0.2, n),
        "rD0": rng.normal(0.1, 0.01, n),
        "rD": rng.normal(0.09, 0.01, n),
        "sd1rel": rng.uniform(0.005, 0.05, n),
    }


def test_the_default_derivation_leaves_the_rescue_off() -> None:
    th, _ = cal_mod.derive_classifier_thresholds(_ref())
    assert not th.rescue_active and th.rescue_tz1_min == 0.0


def test_the_gate_is_the_p98_of_the_reference_set() -> None:
    ref = _ref()
    th, info = cal_mod.derive_classifier_thresholds(ref, r1_ring_px=29.0)
    p98 = float(np.percentile(ref["sd1rel"], 98))
    assert th.rescue_sd1_max == pytest.approx(np.clip(p98, *cal_mod.RESCUE_SD1_RANGE))
    assert info["sd1_rel_p98_raw"] == pytest.approx(p98)
    assert info["n_sd1_reference"] == 300
    assert th.rescue_active
    assert th.rescue_tz1_min == cal_mod.RESCUE_TZ1_MIN
    assert th.rescue_ring_radius_px == 29.0
    assert th.rescue_a0_min == cal_mod.REFERENCE_A0_MIN
    assert th.rescue_zprof_zmax == cal_mod.RESCUE_ZPROF_ZMAX
    assert th.ring_min == pytest.approx(0.30)  # the ring test itself is unchanged


def test_the_gate_is_clipped_to_the_range_fixed_in_advance() -> None:
    low, high = cal_mod.RESCUE_SD1_RANGE
    flat = dict(_ref(), sd1rel=np.full(300, 0.001))
    th, info = cal_mod.derive_classifier_thresholds(flat, r1_ring_px=29.0)
    assert th.rescue_sd1_max == pytest.approx(low)
    assert "rescue_sd1_max" in info["clamped"]
    rough = dict(_ref(), sd1rel=np.full(300, 0.5))
    th, info = cal_mod.derive_classifier_thresholds(rough, r1_ring_px=29.0)
    assert th.rescue_sd1_max == pytest.approx(high)
    assert "rescue_sd1_max" in info["clamped"]
    _, inside = cal_mod.derive_classifier_thresholds(_ref(), r1_ring_px=29.0)
    assert "rescue_sd1_max" not in inside["clamped"]


def test_no_texture_measurement_or_too_few_references_keeps_the_rescue_off() -> None:
    ref = _ref()
    ref.pop("sd1rel")
    th, info = cal_mod.derive_classifier_thresholds(ref, r1_ring_px=29.0)
    assert th.rescue_sd1_max == 0.0 and not th.rescue_active
    assert "rescue_off_reason" in info
    few = _ref(10)  # fewer than MIN_REFERENCE_BEADS: the defaults, no rescue
    th, _ = cal_mod.derive_classifier_thresholds(few, r1_ring_px=29.0)
    assert th.rescue_tz1_min == 0.0 and not th.rescue_active
    nan = dict(_ref(), sd1rel=np.full(300, np.nan))
    th, info = cal_mod.derive_classifier_thresholds(nan, r1_ring_px=29.0)
    assert not th.rescue_active and "rescue_off_reason" in info


def test_the_z_profile_planes_follow_the_reader() -> None:
    avail = cal_mod.zprof_zmax_available
    assert avail(range(8)) == cal_mod.RESCUE_ZPROF_ZMAX
    assert avail([0, 1, 2, 3]) == 3
    assert avail([0, 1]) == 0
    assert avail([0, 1, 2, 4, 5]) == 2
    th, _ = cal_mod.derive_classifier_thresholds(_ref(), r1_ring_px=29.0, zprof_zmax=3)
    assert th.rescue_zprof_zmax == 3.0
    th, _ = cal_mod.derive_classifier_thresholds(_ref(), r1_ring_px=29.0, zprof_zmax=0)
    assert th.rescue_zprof_zmax == 0.0 and th.rescue_sd1_max > 0
    assert th.rescue_active


# ---------------------------------------------------------------------------
# windows, driver and QC
def _synthetic(cal_thresholds: ClassifierThresholds) -> Any:
    return dataclasses.replace(bs.make_calibration(), thresholds=cal_thresholds)


def test_correct_window_reads_the_profile_planes_and_counts_the_rescues(
    stack: dict[str, np.ndarray],
) -> None:
    th = dataclasses.replace(STRICT, rescue_zprof_zmax=6.0)
    st = _with_secondary_focus(stack, z=5)  # z5 is not a corrected layer
    stacks = {k: st[k] for k in ("PolyT", "DAPI")}
    kwargs: dict[str, Any] = {"origin_yx": (0, 0), "z_layers": [1, 2]}
    _, _, with_profile = correct_window(stacks, calibration=_synthetic(th), **kwargs)
    no_profile = dataclasses.replace(th, rescue_zprof_zmax=0.0)
    _, _, without = correct_window(stacks, calibration=_synthetic(no_profile), **kwargs)
    assert without["n_rescued"] >= len(BEADS)
    assert with_profile["n_rescued"] == without["n_rescued"] - 1


def test_correct_window_ignores_profile_planes_the_stack_does_not_have(
    stack: dict[str, np.ndarray],
) -> None:
    th = dataclasses.replace(STRICT, rescue_zprof_zmax=6.0)
    stacks = {k: {z: stack[k][z] for z in range(4)} for k in ("PolyT", "DAPI")}
    _, _, stats = correct_window(
        stacks, origin_yx=(0, 0), calibration=_synthetic(th), z_layers=[1, 2]
    )
    assert stats["n_rescued"] >= len(BEADS)


def test_a_window_without_a_rescue_reports_none(stack: dict[str, np.ndarray]) -> None:
    stacks = {k: stack[k] for k in ("PolyT", "DAPI")}
    _, _, stats = correct_window(
        stacks,
        origin_yx=(0, 0),
        calibration=bs.make_calibration(),
        z_layers=[1, 2],
    )
    assert stats["n_rescued"] == 0


def test_the_driver_reads_the_profile_planes_only_with_an_active_rescue() -> None:
    cal = real_calibration()
    base = driver.planes_needed(cal, [1, 2])
    assert base["PolyT"] == [0, 1, 2]
    on = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.05, 6.0))
    )
    needed = driver.planes_needed(on, [1, 2])
    assert needed["PolyT"] == [0, 1, 2, 3, 4, 5, 6]
    assert needed["DAPI"] == base["DAPI"]
    no_profile = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.05, 0.0))
    )
    assert driver.planes_needed(no_profile, [1, 2])["PolyT"] == [0, 1, 2]
    gate_off = dataclasses.replace(
        cal, thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.0, 6.0))
    )
    assert driver.planes_needed(gate_off, [1, 2])["PolyT"] == [0, 1, 2]


def test_the_chunk_rows_carry_the_rescued_count() -> None:
    stats = [
        {
            "index": 0,
            "core": [0, 10, 0, 10],
            "model": {"n_cand": 5, "n_bead": 4, "n_rescued": 2},
        }
    ]
    rows = qc.chunk_rows(stats, [1, 2])
    assert rows[0]["n_rescued"] == 2 and rows[0]["n_beads"] == 4


def _calibration_with_classifier(**classifier: Any) -> BeadCalibration:
    cal = real_calibration()
    return dataclasses.replace(
        cal,
        thresholds=dataclasses.replace(cal.thresholds, **_rescue_kwargs(0.05, 6.0)),
        decision={"classifier": classifier},
    )


def test_the_qc_rescue_record_reports_counts_gate_and_risk() -> None:
    cal = _calibration_with_classifier(
        sd1_rel_p98_raw=0.0912,
        clamped=["rescue_sd1_max"],
        n_rescued_tiles=195,
        rescued_fraction_tiles=0.0277,
    )
    stats = [
        {"model": {"n_bead": 600, "n_rescued": 12}},
        {"model": {"n_bead": 400, "n_rescued": 3}},
        {"model": {"status": "no beads"}},
    ]
    record = qc._rescue_record(cal, stats)
    assert record["active"] is True
    assert record["n_rescued_windows"] == 15
    assert record["rescued_fraction"] == pytest.approx(0.015)
    assert record["rescue_sd1_max"] == 0.05 and record["rescue_zprof_zmax"] == 6.0
    assert record["sd1_rel_p98_raw"] == 0.0912 and record["sd1_rel_clamped"] is True
    assert record["n_rescued_tiles"] == 195
    assert record["rescued_fraction_tiles"] == pytest.approx(0.0277)
    assert "compact tissue object" in record["residual_risk"]
    assert "off_reason" not in record


def test_the_qc_rescue_record_of_a_dataset_without_the_rescue() -> None:
    cal = real_calibration()  # rescue off
    record = qc._rescue_record(cal, [{"model": {"n_bead": 10, "n_rescued": 0}}])
    assert record["active"] is False
    assert record["n_rescued_windows"] == 0 and record["rescued_fraction"] == 0.0
    assert "residual_risk" not in record and "off_reason" in record
    assert qc._rescue_record(cal, [])["rescued_fraction"] is None


def test_the_recall_warnings_of_the_fallback_stay_as_decided() -> None:
    # The rescue does not lift the cluster-inclusive recall warning of the
    # dense-cluster datasets, and the isolated recall warns only on enough beads.
    warn = cal_mod.classifier_recall_warnings
    assert warn(0.99, 100, 0.569, 188) and "clusters" in warn(0.99, 100, 0.569, 188)[0]
    assert not warn(0.99, 100, 0.569, 49)
    assert not warn(0.80, 19, 0.9, 100)
    assert warn(0.80, 20, 0.9, 100)
