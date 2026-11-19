"""Settings rules, class table, field fit and check payload of the landmark check."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.image_registration import (
    pull_points_through_registration,
    push_points_through_registration,
)
from merxen.manual_registration import field_fit
from merxen.manual_registration import landmark_field as landmark_field_module
from merxen.manual_registration.field_fit import (
    GROSS_P,
    GROSS_WEIGHT,
    SuggestionRecord,
    best_affine,
    build_landmark_field,
    check_payload,
    classify_pairs,
    details_payload,
    field_flow,
    field_options,
    field_sd_grid,
    resolve_store_roles,
    tissue_mask_function,
    validate_field_settings,
    window_params,
)
from merxen.manual_registration.image_evidence import (
    ChannelRoles,
    ClickCheck,
    ClickProfile,
    WindowObservations,
)
from merxen.manual_registration.landmark_field import Suggestion
from merxen.manual_registration.presets import resolve_settings
from merxen.manual_registration.register import fit_landmarks
from merxen.manual_registration.schema import (
    CHECK_CLASSES,
    CHECK_VERDICTS,
    CropRect,
    LandmarkPair,
    canonical_json,
    labelling_content_sha256,
    parse_check_attrs,
    parse_labelling_attrs,
    parse_store_attrs,
)
from merxen.manual_registration.simulate import FieldPhantom, make_field_phantom

from ._mr_helpers import (
    DIGEST_A,
    labelling_dict,
    store_attr_dict,
)

PIXEL_UM = 2.0
ROLES = ChannelRoles(
    reference={"at8": "AF568", "map2": "AF647", "dapi": "DAPI"},
    moving={"at8": "AF568", "map2": "AF647", "dapi": "DAPI"},
)


# ---------------------------------------------------------------------------
# classify_pairs: the C7 truth table
# ---------------------------------------------------------------------------


def _classify(
    *,
    verdict: str,
    p: float = 0.5,
    w: float = 1.0,
    p1: float | None = None,
    w1: float | None = None,
    enabled: bool = True,
    observed: bool = True,
    outlier_p: float = 0.01,
) -> str:
    result = classify_pairs(
        enabled=np.array([enabled]),
        observed=np.array([observed]),
        p_value=np.array([p]),
        weight=np.array([w]),
        clicks_only_p_value=np.array([p if p1 is None else p1]),
        clicks_only_weight=np.array([w if w1 is None else w1]),
        verdict=[verdict],
        outlier_p=outlier_p,
    )
    return result[0]


@pytest.mark.parametrize("verdict", CHECK_VERDICTS)
def test_disabled_and_excluded_pairs_ignore_everything_else(verdict: str) -> None:
    assert _classify(verdict=verdict, p=1e-12, enabled=False) == "disabled"
    assert _classify(verdict=verdict, p=1e-12, observed=False) == "excluded"
    assert (
        _classify(verdict=verdict, p=1e-12, enabled=False, observed=False) == "disabled"
    )


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("confirmed", "check"),
        ("contradicted", "likely_misclick"),
        ("ridge", "likely_misclick"),
        ("inconclusive", "likely_misclick"),
        ("not_checked", "likely_misclick"),
    ],
)
def test_gross_pairs(verdict: str, expected: str) -> None:
    assert _classify(verdict=verdict, p=GROSS_P / 10) == expected
    assert _classify(verdict=verdict, w=GROSS_WEIGHT / 2) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("contradicted", "likely_misclick"),
        ("confirmed", "kept_confirmed"),
        ("ridge", "check"),
        ("inconclusive", "check"),
        ("not_checked", "check"),
    ],
)
def test_flagged_pairs(verdict: str, expected: str) -> None:
    assert _classify(verdict=verdict, p=0.005) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("contradicted", "check"),
        ("confirmed", "ok"),
        ("ridge", "ok"),
        ("inconclusive", "ok"),
        ("not_checked", "ok"),
    ],
)
def test_unflagged_pairs(verdict: str, expected: str) -> None:
    assert _classify(verdict=verdict, p=0.2) == expected


def test_the_outlier_threshold_is_the_setting() -> None:
    assert _classify(verdict="not_checked", p=0.02) == "ok"
    assert _classify(verdict="not_checked", p=0.02, outlier_p=0.05) == "check"
    assert _classify(verdict="contradicted", p=0.02, outlier_p=0.05) == (
        "likely_misclick"
    )


def test_the_boundaries_are_strict() -> None:
    assert _classify(verdict="not_checked", p=0.01) == "ok"
    assert _classify(verdict="not_checked", p=GROSS_P) == "check"
    assert _classify(verdict="not_checked", w=GROSS_WEIGHT) == "ok"


def test_the_lower_p_and_weight_of_the_two_fields_decide() -> None:
    # Gross only in the clicks-only field (the windows hid it): not ok.
    assert _classify(verdict="not_checked", p=0.5, p1=1e-9) == "likely_misclick"
    assert _classify(verdict="confirmed", p=0.5, p1=1e-9) == "check"
    assert _classify(verdict="not_checked", w=1.0, w1=0.01) == "likely_misclick"
    # Flagged only in the final field.
    assert _classify(verdict="confirmed", p=0.004, p1=0.3) == "kept_confirmed"
    assert _classify(verdict="not_checked", p=0.004, p1=0.3) == "check"
    # Flagged only in the clicks-only field.
    assert _classify(verdict="contradicted", p=0.3, p1=0.004) == "likely_misclick"


def test_nan_weights_are_ignored() -> None:
    assert _classify(verdict="not_checked", w=float("nan"), w1=0.01) == (
        "likely_misclick"
    )
    assert _classify(verdict="not_checked", w=float("nan"), w1=float("nan")) == "ok"
    assert _classify(verdict="not_checked", w=0.01, w1=float("nan")) == (
        "likely_misclick"
    )


def test_class_names_are_the_contract_classes() -> None:
    seen = {
        _classify(verdict=v, p=p, enabled=e, observed=o)
        for v in CHECK_VERDICTS
        for p in (0.5, 0.005, 1e-9)
        for e in (True, False)
        for o in (True, False)
    }
    assert seen == set(CHECK_CLASSES)


# ---------------------------------------------------------------------------
# validate_field_settings
# ---------------------------------------------------------------------------


def _settings(preset: str = "cortex_serial", row: str = "") -> Any:
    return resolve_settings(
        preset,
        reference_pixel_size_um=PIXEL_UM,
        reference_channel="DAPI",
        row_settings=row,
    )


def _validate(
    settings: Any, *, preset: str, roles: Any, purpose: str = "register"
) -> None:
    validate_field_settings(
        settings,
        preset=preset,
        roles=roles,
        run_id="X_p",
        purpose=purpose,  # type: ignore[arg-type]
    )


def test_register_rules_accept_the_cortex_preset_with_roles() -> None:
    _validate(_settings(), preset="cortex_serial", roles=ROLES)
    _validate(_settings("same_section"), preset="same_section", roles=None)
    _validate(_settings("serial_section"), preset="serial_section", roles=None)


def test_register_rule_preview_needs_a_field_run() -> None:
    settings = _settings("same_section", "materialize=none")
    with pytest.raises(ValueError, match=r"^\[X_p\] .*only for robust-field runs"):
        _validate(settings, preset="same_section", roles=None)


@pytest.mark.parametrize("override", ["refine_affine=true", "nonrigid_enabled=true"])
def test_register_rule_field_runs_skip_the_refinement(override: str) -> None:
    settings = _settings("cortex_serial", override)
    with pytest.raises(ValueError, match="skip the DAPI refinement and TV-L1"):
        _validate(settings, preset="cortex_serial", roles=ROLES)


def test_register_rule_cortex_serial_needs_roles() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"preset cortex_serial needs channel roles: set the samplesheet column "
            r"channel_roles, e\.g\. at8=AF568;map2=AF647;dapi=DAPI"
        ),
    ):
        _validate(_settings(), preset="cortex_serial", roles=None)


def test_register_rule_a_field_run_with_image_evidence_needs_roles() -> None:
    row = "field_model=robust_gp;refine_affine=false;nonrigid_enabled=false"
    with pytest.raises(ValueError, match="needs channel roles"):
        _validate(_settings("same_section", row), preset="same_section", roles=None)
    off = f"{row};field_image_check=false"
    _validate(_settings("same_section", off), preset="same_section", roles=None)


def test_check_rule_needs_roles_only_for_image_evidence() -> None:
    settings = _settings()
    with pytest.raises(ValueError, match=r"--channel-roles 'at8=AF568;map2=AF647"):
        _validate(settings, preset="cortex_serial", roles=None, purpose="check")
    with pytest.raises(ValueError, match="--no-image-check --no-windows"):
        _validate(settings, preset="cortex_serial", roles=None, purpose="check")
    _validate(settings, preset="cortex_serial", roles=ROLES, purpose="check")
    off = settings.model_copy(
        update={"field_image_check": False, "field_window_observations": False}
    )
    _validate(off, preset="cortex_serial", roles=None, purpose="check")


def test_check_with_the_same_section_preset_does_not_trip_the_refinement_rule() -> None:
    settings = _settings("same_section").model_copy(
        update={"field_model": "robust_gp", "field_window_observations": True}
    )
    assert settings.refine_affine and settings.nonrigid_enabled
    _validate(settings, preset="same_section", roles=ROLES, purpose="check")


def test_resolve_store_roles_blank_and_names() -> None:
    attrs = store_attr_dict((100, 100), (100, 100), 1.0, 1.0)
    for record in attrs["images"].values():
        record["canonical_channel_names"] = ["DAPI", "AF568", "AF647"]
        record["channel_names"] = ["DAPI", "AF568", "AF647"]
    store = parse_store_attrs(attrs)
    assert resolve_store_roles("  ", store) is None
    roles = resolve_store_roles("at8=AF568;map2=AF647;dapi=DAPI", store)
    assert roles is not None and roles.reference["map2"] == "AF647"
    with pytest.raises(ValueError, match="channel roles: channel 'NOPE'"):
        resolve_store_roles("at8=NOPE;map2=AF647;dapi=DAPI", store)


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_field_options_follow_the_settings() -> None:
    options = field_options(_settings())
    assert options.components is None
    assert options.nu == 4.0
    assert dict(options.noise_bounds_um) == {
        "click": (3.0, 300.0),
        "confirmed": (3.0, 100.0),
        "window": (15.0, 100.0),
    }
    custom = _settings(
        "cortex_serial",
        "field_kernel=two_scale;field_student_t_nu=6;field_window_noise_floor_um=20",
    )
    options = field_options(custom)
    assert options.components == 2 and options.nu == 6.0
    assert options.noise_bounds_um["window"] == (20.0, 100.0)
    assert (
        field_options(_settings("cortex_serial", "field_kernel=single")).components == 1
    )


def test_window_params_follow_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    params = window_params(_settings())
    assert params.sizes_um == (600.0, 800.0)
    assert (params.step_um, params.search_um, params.thinning_um) == (
        200.0,
        300.0,
        400.0,
    )
    monkeypatch.setattr(field_fit, "WINDOW_SIZES_UM", (200.0, 300.0))
    custom = window_params(
        _settings(
            "cortex_serial",
            "field_window_step_um=150;field_window_search_um=250;"
            "field_window_min_spacing_um=0",
        )
    )
    assert custom.sizes_um == (200.0, 300.0)
    assert (custom.step_um, custom.search_um, custom.thinning_um) == (150.0, 250.0, 0.0)


# ---------------------------------------------------------------------------
# build_landmark_field on a numpy phantom
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phantom() -> FieldPhantom:
    return make_field_phantom(
        seed=11,
        n_clicks=45,
        n_windows=40,
        n_gross=2,
        n_medium=2,
        long_signal_um=60.0,
        short_signal_um=25.0,
        click_noise_um=20.0,
    )


def _inputs(
    phantom: FieldPhantom,
    *,
    disabled: tuple[int, ...] = (),
    crops: dict[str, CropRect | None] | None = None,
    row_settings: str = "",
) -> tuple[Any, Any, Any, Any]:
    """The similarity report, store attrs, labelling and settings of the phantom."""
    count = len(phantom.reference_xy_um)
    reference_px = phantom.reference_xy_um / PIXEL_UM
    moving_px = phantom.moving_xy_um / PIXEL_UM
    pairs = [
        LandmarkPair(
            id=i + 1,
            colour="#aabbcc",
            enabled=(i + 1) not in disabled,
            reference_xy_px=(float(reference_px[i, 0]), float(reference_px[i, 1])),
            moving_xy_px=(float(moving_px[i, 0]), float(moving_px[i, 1])),
        )
        for i in range(count)
    ]
    shape = (9000, 12000)
    store = parse_store_attrs(store_attr_dict(shape, shape, PIXEL_UM, PIXEL_UM))
    from merxen.manual_registration.schema import CoarseTransform

    labelling = parse_labelling_attrs(
        labelling_dict(
            pairs,
            CoarseTransform(rotation_deg=2.0, offset_um=(150.0, -80.0)),
            crops,
            ref_shape=shape,
            mov_shape=shape,
            p_r=PIXEL_UM,
            p_m=PIXEL_UM,
            grid_digest=DIGEST_A,
        )
    )
    settings = _settings(row=row_settings)
    similarity = fit_landmarks(labelling, store, settings, run_id="X_p")
    return similarity, store, labelling, settings


def test_clicks_only_fit_on_a_phantom(phantom: FieldPhantom) -> None:
    similarity, store, _labelling, settings = _inputs(phantom, disabled=(7,))
    report, context = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops={},
        image_check=False,
        windows=False,
        run_id="X_p",
    )
    assert context is None
    assert report.has_second_pass is False
    assert report.field is report.clicks_only
    assert report.field.components == 1
    count = len(phantom.reference_xy_um)
    assert report.pair_ids.shape == (count,)
    assert report.loo_vector_moving_um.shape == (count, 2)
    assert report.observed.sum() == count - 1
    assert not report.observed[6]
    assert report.pair_class[6] == "disabled"
    assert np.isnan(report.weight[6]) and np.isfinite(report.weight[0])
    assert set(report.verdict) == {"not_checked"}
    assert report.checks == {}
    assert report.windows is None
    assert report.n_windows == 0
    gross = [i for i, kind in enumerate(phantom.kind) if kind == "gross" and i != 6]
    assert gross
    for index in gross:
        assert report.pair_class[index] == "likely_misclick", index
        assert report.loo_reference_um[index] > 500.0
    clean = [i for i, kind in enumerate(phantom.kind) if kind == "clean"]
    assert np.median([report.loo_reference_um[i] for i in clean]) < 100.0
    # The disabled pair is judged against the field it is not part of.
    assert report.loo_reference_um[6] == pytest.approx(
        report.field.predict_residual(
            phantom.reference_xy_um[6:7], phantom.moving_xy_um[6:7]
        ).magnitude_reference_um[0]
    )
    assert set(report.timings_s) == {"clicks_only_fit"}
    assert report.similarity is similarity


def test_the_similarity_outlier_warning_is_replaced(phantom: FieldPhantom) -> None:
    similarity, store, _labelling, settings = _inputs(phantom)
    assert similarity.outlier.any()
    assert any("flagged as outlier" in w for w in similarity.warnings)
    report, _ = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops={},
        image_check=False,
        windows=False,
        run_id="X_p",
    )
    assert not any("flagged as outlier" in w for w in report.warnings)
    (informative,) = [w for w in report.warnings if w.startswith("rigid (similarity)")]
    n = int(similarity.outlier.sum())
    assert f"{n} pair(s) exceed the similarity outlier rule" in informative
    assert "the robust field decides" in informative


def test_crop_policy_exclusions_are_not_observations(phantom: FieldPhantom) -> None:
    centre = phantom.reference_xy_um.mean(axis=0) / PIXEL_UM
    crop = CropRect(x0=0, y0=0, x1=int(centre[0] + 3000), y1=int(centre[1] + 2500))
    similarity, store, _labelling, settings = _inputs(
        phantom,
        crops={"reference": crop, "moving": None},
        row_settings="landmarks_outside_crop=exclude",
    )
    report, _ = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops={},
        image_check=False,
        windows=False,
        run_id="X_p",
    )
    excluded = [i for i, c in enumerate(report.pair_class) if c == "excluded"]
    assert excluded
    assert not report.observed[excluded].any()
    assert np.isnan(report.weight[excluded]).all()
    assert report.observed.sum() == len(phantom.reference_xy_um) - len(excluded)


def test_image_evidence_without_roles_is_refused(phantom: FieldPhantom) -> None:
    similarity, store, _labelling, settings = _inputs(phantom)
    with pytest.raises(ValueError, match="image evidence needs channel roles"):
        build_landmark_field(
            similarity=similarity,
            store=store,
            settings=settings,
            roles=None,
            crops={},
            image_check=True,
            windows=False,
            run_id="X_p",
        )


# ---------------------------------------------------------------------------
# With a fake image stage (the WP2 functions are replaced)
# ---------------------------------------------------------------------------


def _profile(
    axis: tuple[float, float] = (1.0, 0.0), *, elongated: bool = True
) -> ClickProfile:
    unit = np.array(axis, dtype=float) / np.hypot(*axis)
    return ClickProfile(
        elongated=elongated,
        elongation=4.0 if elongated else 1.2,
        length_um=180.0,
        width_um=40.0,
        area_um2=5000.0,
        axis_reference_xy=unit,
        axis_moving_xy=unit,
        masked=elongated,
    )


def _click_check(
    pair_id: int,
    verdict: str,
    *,
    strength: str | None = None,
    direction: tuple[float, float] | None = None,
    profile: ClickProfile | None = None,
) -> ClickCheck:
    return ClickCheck(
        pair_id=pair_id,
        verdict=verdict,
        image_moving_xy_um=None,
        image_offset_um=4.0 if verdict == "confirmed" else None,
        prediction_offset_um=None,
        snap_moving_xy_um=(
            np.array([5000.0, 4000.0]) if verdict == "contradicted" else None
        ),
        ridge_direction_moving_xy=(
            None if direction is None else np.array(direction, dtype=float)
        ),
        elongation=3.0 if direction is not None else None,
        estimates=(),
        reason="fake",
        strength=strength,
        profile=profile,
    )


def _fake_window_set(phantom: FieldPhantom, count: int) -> WindowObservations:
    return WindowObservations(
        reference_xy_um=phantom.window_reference_xy_um[:count],
        moving_xy_um=phantom.window_moving_xy_um[:count],
        peak_cov_um2=phantom.window_peak_cov_um2[:count],
        elongation=np.ones(count),
        window_um=np.full(count, 600.0),
        peak=np.full(count, 0.5),
        distinct=np.full(count, 0.2),
        region=("grey_matter",) * count,
        residual_um=np.zeros((count, 2)),
        candidates=pd.DataFrame(
            {"confident": [True] * count, "accepted": [True] * count}
        ),
        nulls={},
        summary={"algorithm": "fake", "n_thinned": count},
    )


@pytest.fixture
def fake_images(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the image stage; ``state`` holds what the test wants it to return."""
    state = SimpleNamespace(checks={}, windows=None, calls=[])

    def load(*_a: Any, **_k: Any) -> dict[str, Any]:
        state.calls.append("load")
        return {"reference": SimpleNamespace(), "moving": SimpleNamespace()}

    def warp(_moving: Any, *, reference: Any, pull_um: Any) -> Any:
        state.calls.append("warp")
        state.pull_probe = pull_um(np.array([[1000.0, 2000.0]]))
        return SimpleNamespace()

    def check(*_a: Any, pair_ids: list[int], **kwargs: Any) -> list[ClickCheck]:
        state.calls.append("check")
        state.check_kwargs = kwargs
        return [state.checks.get(i, _click_check(i, "inconclusive")) for i in pair_ids]

    def windows(*_a: Any, **_k: Any) -> WindowObservations:
        state.calls.append("windows")
        return state.windows

    monkeypatch.setattr(field_fit, "load_role_images", load)

    def represent(_images: Any, params: Any = None) -> SimpleNamespace:
        state.representation_params = params
        return SimpleNamespace()

    monkeypatch.setattr(field_fit, "compute_representation", represent)
    monkeypatch.setattr(field_fit, "warp_role_images", warp)
    monkeypatch.setattr(field_fit, "check_clicks", check)
    monkeypatch.setattr(field_fit, "tissue_window_observations", windows)
    return state


def _build(
    phantom: FieldPhantom, *, image_check: bool, windows: bool, **kwargs: Any
) -> tuple[Any, Any, Any, Any, Any]:
    similarity, store, labelling, settings = _inputs(phantom, **kwargs)
    report, context = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=ROLES,
        crops={},
        image_check=image_check,
        windows=windows,
        run_id="X_p",
    )
    return report, context, store, labelling, settings


def test_windows_enter_as_window_observations(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.windows = _fake_window_set(phantom, 30)
    report, context, *_ = _build(phantom, image_check=False, windows=True)
    assert fake_images.calls == ["load", "warp", "windows"]
    assert context is not None and context.roles is ROLES
    assert report.has_second_pass is True
    assert report.field is not report.clicks_only
    assert report.n_windows == 30
    assert report.field.components == 2
    observations = report.field.observations
    count = len(phantom.reference_xy_um)
    assert observations.n == count + 30
    assert observations.obs_class[count:] == ("window",) * 30
    np.testing.assert_allclose(
        observations.fixed_cov_um2[count:], phantom.window_peak_cov_um2[:30]
    )
    assert not observations.fixed_cov_um2[:count].any()
    # C1 ids stay unique across clicks and windows.
    assert len(set(observations.obs_id.tolist())) == observations.n
    assert report.field.hyperparameters.noise_um["window"] >= 15.0
    assert set(report.verdict) == {"not_checked"}
    assert fake_images.pull_probe.shape == (1, 2)
    assert "final_fit" in report.timings_s and "tissue_windows" in report.timings_s


def test_the_window_noise_floor_setting_is_a_lower_bound(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.windows = _fake_window_set(phantom, 30)
    report, *_ = _build(
        phantom,
        image_check=False,
        windows=True,
        row_settings="field_window_noise_floor_um=40",
    )
    assert report.field.hyperparameters.noise_um["window"] >= 40.0 - 1e-9


def test_ridge_clicks_get_the_ridge_shape(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.checks = {
        3: _click_check(3, "ridge", direction=(1.0, 0.0)),
        4: _click_check(4, "ridge", direction=(0.0, 2.0)),
    }
    report, *_ = _build(phantom, image_check=True, windows=False)
    assert report.has_second_pass is True
    shapes = report.field.observations.shape_cov
    ratio = 60.0 / 27.0
    np.testing.assert_allclose(shapes[2], np.diag([ratio, 1.0 / ratio]), atol=1e-12)
    np.testing.assert_allclose(shapes[3], np.diag([1.0 / ratio, ratio]), atol=1e-12)
    np.testing.assert_allclose(shapes[0], np.eye(2))
    assert report.verdict[2] == report.verdict[3] == "ridge"
    assert report.field.components == 1


def test_vessel_profile_clicks_get_along_axis_noise_and_are_never_exempt(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.checks = {
        1: _click_check(1, "confirmed", strength="strong", profile=_profile((1, 0))),
        2: _click_check(2, "confirmed", strength="strong"),
        3: _click_check(3, "inconclusive", profile=_profile((0.0, 2.0))),
        4: _click_check(
            4, "confirmed", strength="strong", profile=_profile(elongated=False)
        ),
        5: _click_check(5, "ridge", direction=(1.0, 0.0), profile=_profile((1, 0))),
    }
    report, *_ = _build(phantom, image_check=True, windows=False)
    assert fake_images.representation_params.vessel_mask is True
    assert report.has_second_pass is True
    obs = report.field.observations
    sd2 = 45.0**2
    np.testing.assert_allclose(obs.fixed_cov_um2[0], np.diag([sd2, 0.0]))
    np.testing.assert_allclose(obs.fixed_cov_um2[2], np.diag([0.0, sd2]), atol=1e-9)
    np.testing.assert_allclose(obs.fixed_cov_um2[4], np.diag([sd2, 0.0]))
    assert not obs.fixed_cov_um2[[1, 3, 5]].any()
    # A vessel-end confirmation is never exempt; a compact profile's still is.
    assert list(report.em_exempt[:5]) == [False, True, False, True, False]
    assert list(report.vessel_profile[:6]) == [True, False, True, False, True, False]
    # The ridge shape and the vessel axis noise combine.
    ratio = 60.0 / 27.0
    np.testing.assert_allclose(obs.shape_cov[4], np.diag([ratio, 1.0 / ratio]))
    np.testing.assert_allclose(obs.shape_cov[0], np.eye(2))


def test_the_clicks_only_field_keeps_the_vessel_noise_but_not_the_rest(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    plain, *_ = _build(phantom, image_check=True, windows=False)
    # The click the clicks alone fit worst (among those not gross) sits on a
    # vessel whose axis is its own residual: its offset is a moving profile.
    residual = plain.clicks_only.loo().residual_um
    clean = [i for i, kind in enumerate(phantom.kind) if kind != "gross"]
    index = max(clean, key=lambda i: float(np.hypot(*residual[i])))
    axis = tuple(float(v) for v in residual[index])
    fake_images.checks = {
        index + 1: _click_check(
            index + 1, "confirmed", strength="strong", profile=_profile(axis)
        ),
        1: _click_check(1, "confirmed", strength="strong"),
        2: _click_check(2, "ridge", direction=(1.0, 0.0)),
    }
    report, context, *_ = _build(phantom, image_check=True, windows=False)

    clicks_only = report.clicks_only.observations
    assert report.clicks_only is not report.field
    np.testing.assert_allclose(
        clicks_only.fixed_cov_um2[index], report.field.observations.fixed_cov_um2[index]
    )
    assert clicks_only.fixed_cov_um2[index].any()
    # No ridge shape, exemption or confirmed class: those stay in the final field.
    np.testing.assert_allclose(
        clicks_only.shape_cov, np.broadcast_to(np.eye(2), clicks_only.shape_cov.shape)
    )
    assert not clicks_only.em_exempt.any()
    assert set(clicks_only.obs_class) == {"click"}
    assert report.em_exempt[0] and not report.em_exempt[index]
    # The offset along the vessel no longer looks like a misclick.
    assert report.clicks_only_p_value[index] > plain.clicks_only_p_value[index]
    # The warp prior is still the plain clicks-only fit.
    assert context is not None
    probe = np.array([[1000.0, 2000.0]])
    np.testing.assert_allclose(
        context.pull_um(probe), plain.clicks_only.mean(probe), atol=1e-9
    )


def test_the_along_axis_sd_setting_scales_the_vessel_noise_and_zero_turns_it_off(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.checks = {
        1: _click_check(1, "confirmed", strength="strong", profile=_profile((1, 0))),
    }
    report, *_ = _build(
        phantom,
        image_check=True,
        windows=False,
        row_settings="field_vessel_click_along_sd_um=30",
    )
    np.testing.assert_allclose(
        report.field.observations.fixed_cov_um2[0], np.diag([900.0, 0.0])
    )
    off, *_ = _build(
        phantom,
        image_check=True,
        windows=False,
        row_settings="field_vessel_click_along_sd_um=0;field_vessel_mask=false",
    )
    assert fake_images.representation_params.vessel_mask is False
    # Nothing changes the clicks: pass 2 is pass 1, and the click stays unexempt.
    assert off.has_second_pass is False
    assert not off.em_exempt.any()
    assert off.vessel_profile[0]


def test_inconclusive_verdicts_alone_leave_pass_two_equal_to_pass_one(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    report, *_ = _build(phantom, image_check=True, windows=False)
    assert report.has_second_pass is False
    assert report.field is report.clicks_only
    assert set(report.verdict) == {"inconclusive"}


def test_strong_confirmations_are_exempt_and_become_a_class_from_five(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    strong = [1, 2, 3, 4, 5]
    fake_images.checks = {
        i: _click_check(i, "confirmed", strength="strong") for i in strong
    }
    fake_images.checks[6] = _click_check(6, "confirmed", strength="weak")
    report, *_ = _build(phantom, image_check=True, windows=False)
    assert report.has_second_pass is True
    obs = report.field.observations
    assert obs.obs_class[:6] == ("confirmed",) * 5 + ("click",)
    assert list(obs.em_exempt[:7]) == [True] * 5 + [False, False]
    assert list(report.em_exempt[:7]) == [True] * 5 + [False, False]
    np.testing.assert_allclose(report.field.weights[:5], 1.0)
    assert report.confirm_strength[:6] == ("strong",) * 5 + ("weak",)
    assert report.confirm_strength[6] is None
    # The clicks-only field is unchanged: no exemption and no extra class.
    assert report.clicks_only.observations.obs_class == ("click",) * len(
        report.pair_ids
    )
    assert not report.clicks_only.observations.em_exempt.any()


def test_fewer_than_five_strong_confirmations_stay_clicks_but_exempt(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.checks = {
        i: _click_check(i, "confirmed", strength="strong") for i in (1, 2, 3)
    }
    report, *_ = _build(phantom, image_check=True, windows=False)
    obs = report.field.observations
    assert set(obs.obs_class) == {"click"}
    assert list(obs.em_exempt[:4]) == [True, True, True, False]


def test_trust_off_ignores_strong_confirmations(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.checks = {
        i: _click_check(i, "confirmed", strength="strong") for i in range(1, 8)
    }
    report, *_ = _build(
        phantom,
        image_check=True,
        windows=False,
        row_settings="field_trust_confirmed_clicks=false",
    )
    assert report.has_second_pass is False
    assert not report.field.observations.em_exempt.any()


def test_a_misclick_the_clicks_alone_expose_is_not_hidden_by_windows(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.windows = _fake_window_set(phantom, 40)
    gross = [i for i, kind in enumerate(phantom.kind) if kind == "gross"]
    fake_images.checks = {
        gross[0] + 1: _click_check(gross[0] + 1, "contradicted"),
    }
    report, *_ = _build(phantom, image_check=True, windows=True)
    index = gross[0]
    assert report.verdict[index] == "contradicted"
    assert report.pair_class[index] == "likely_misclick"
    assert report.clicks_only_p_value[index] < GROSS_P


# ---------------------------------------------------------------------------
# check_payload and details
# ---------------------------------------------------------------------------


def _suggestions(report: Any) -> list[SuggestionRecord]:
    field = report.field
    points = np.array([[8000.0, 6000.0], [12000.0, 9000.0]])
    covariance = field.covariance(points)
    return [
        SuggestionRecord(
            Suggestion(
                rank=i + 1,
                reference_xy_um=points[i],
                predicted_moving_xy_um=field.mean(points[i : i + 1])[0],
                sd_um=float(np.sqrt(np.trace(covariance[i]) / 2.0)),
                covariance_um2=covariance[i],
                weight=1.0 if i == 0 else 0.25,
                score=1.0,
            ),
            "grey_matter" if i == 0 else "other",
        )
        for i in range(2)
    ]


def _payload(report: Any, store: Any, labelling: Any, settings: Any) -> dict[str, Any]:
    return check_payload(
        report,
        store=store,
        labelling=labelling,
        labelling_sha=labelling_content_sha256(
            labelling_dict(
                labelling.landmarks.pairs,
                labelling.coarse,
                ref_shape=(9000, 12000),
                mov_shape=(9000, 12000),
                p_r=PIXEL_UM,
                p_m=PIXEL_UM,
            )
        ),
        labelling_source="store",
        settings=settings,
        preset="cortex_serial",
        roles=ROLES,
        suggestions=_suggestions(report),
        sd_summary={"median": 20.0, "p90": 40.0, "max": 80.0},
        outputs={
            "json": "X_cortex_serial_check.json",
            "png": "X_cortex_serial_check.png",
        },
    )


def test_check_payload_without_a_second_pass(phantom: FieldPhantom) -> None:
    similarity, store, labelling, settings = _inputs(phantom, disabled=(5,))
    report, _ = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops={},
        image_check=False,
        windows=False,
        run_id="X_p",
    )
    payload = _payload(report, store, labelling, settings)
    canonical_json(payload)
    check = parse_check_attrs(payload)
    assert check.field.loo_um_clicks_only is None
    assert check.algorithm == {
        "landmark_field": "robust_gp_matern52_v2",
        "image_evidence": None,
    }
    assert all(pair.clicks_only is None for pair in check.pairs.values())
    assert check.pairs["5"].enabled is False and check.pairs["5"].weight is None
    assert check.pairs["5"].pair_class == "disabled"
    assert check.pairs["1"].used is True and check.pairs["1"].weight is not None
    assert check.field.n_clicks == len(phantom.reference_xy_um) - 1
    assert check.field.n_windows == 0
    assert sum(check.field.class_counts.values()) == len(phantom.reference_xy_um)
    assert set(check.field.class_counts) == set(CHECK_CLASSES)
    assert set(check.field.verdict_counts) == set(CHECK_VERDICTS)
    assert check.field.hyperparameters.short_length_um is None
    assert check.suggestions[0].region == "grey_matter"
    assert check.suggestions[0].reference_xy_px == (4000.0, 3000.0)
    assert (
        check.suggestions[0].ellipse_um["major"]
        >= check.suggestions[0].ellipse_um["minor"]
    )
    assert check.channel_roles == ROLES.to_dict()
    assert check.dataset_name == store.dataset_name
    assert check.labelling_source == "store"
    assert "details" not in payload


def test_field_outlier_p_setting_reaches_the_classes_and_the_payload(
    phantom: FieldPhantom,
) -> None:
    def build(row_settings: str) -> dict[str, Any]:
        similarity, store, labelling, settings = _inputs(
            phantom, row_settings=row_settings
        )
        report, _ = build_landmark_field(
            similarity=similarity,
            store=store,
            settings=settings,
            roles=None,
            crops={},
            image_check=False,
            windows=False,
            run_id="X_p",
        )
        return _payload(report, store, labelling, settings)

    default = build("")
    loose = build("field_outlier_p=0.49")

    assert default["field"]["outlier_p"] == 0.01
    assert loose["field"]["outlier_p"] == 0.49
    assert loose["field"]["class_counts"]["ok"] < default["field"]["class_counts"]["ok"]


def test_check_payload_with_a_second_pass_and_details(
    phantom: FieldPhantom, fake_images: SimpleNamespace
) -> None:
    fake_images.windows = _fake_window_set(phantom, 25)
    gross = [i for i, kind in enumerate(phantom.kind) if kind == "gross"]
    fake_images.checks = {
        1: _click_check(1, "confirmed", strength="strong"),
        2: _click_check(2, "confirmed", strength="weak"),
        3: _click_check(3, "inconclusive", profile=_profile((0.6, 0.8))),
        gross[0] + 1: _click_check(gross[0] + 1, "contradicted"),
    }
    report, _, store, labelling, settings = _build(
        phantom, image_check=True, windows=True, disabled=(9,)
    )
    payload = _payload(report, store, labelling, settings)
    canonical_json(payload)
    check = parse_check_attrs(payload)
    assert check.algorithm["image_evidence"] == "image_evidence_bp20_v3"
    assert check.field.n_windows == 25
    assert check.field.loo_um_clicks_only is not None
    assert check.field.hyperparameters.short_length_um is not None
    assert check.field.verdict_counts["confirmed"] == 2
    first = check.pairs["1"]
    assert first.clicks_only is not None
    assert first.em_exempt is True and first.confirm_strength == "strong"
    assert check.pairs["2"].confirm_strength == "weak"
    assert check.pairs["2"].em_exempt is False
    assert check.pairs["9"].clicks_only is not None
    assert check.pairs["9"].clicks_only.weight is None
    snap = check.pairs[str(gross[0] + 1)].snap_moving_xy_px
    assert snap == (2500.0, 2000.0)
    assert check.pairs["1"].image_offset_um == 4.0
    assert check.pairs["3"].vessel_profile is True
    assert payload["pairs"]["3"]["vessel_profile"] is True
    assert check.pairs["1"].vessel_profile is False
    details = details_payload(report)
    assert set(details) == {
        "click_estimates",
        "windows",
        "field_model",
        "clicks_only_hyperparameters",
        "timings_s",
        "warnings",
    }
    assert details["windows"]["n_accepted_windows"] == 25
    assert (
        details["field_model"]["schema"] == "merxen.manual_registration.landmark_field"
    )
    assert details["click_estimates"]["1"]["reason"] == "fake"
    assert details["click_estimates"]["1"]["profile"] is None
    profile = details["click_estimates"]["3"]["profile"]
    assert profile["elongated"] is True and profile["masked"] is True
    assert profile["axis_moving_xy"] == pytest.approx([0.6, 0.8])
    canonical_json(details)


# ---------------------------------------------------------------------------
# The field as a pull flow
# ---------------------------------------------------------------------------

FLOW_P_REF = 5.5
FLOW_P_MOV = 5.0
FLOW_CROP_REF = CropRect(x0=128, y0=64, x1=128 + 1920, y1=64 + 1408)
FLOW_CROP_MOV = CropRect(x0=256, y0=192, x1=256 + 2048, y1=192 + 1536)
FLOW_GRID_UM = 33.0


@pytest.fixture(scope="module")
def smooth_field() -> landmark_field_module.LandmarkField:
    phantom = make_field_phantom(
        seed=4,
        short_signal_um=0.0,
        ridge_fraction=0.0,
        n_medium=0,
        n_gross=0,
        click_noise_um=10.0,
    )
    return landmark_field_module.fit_landmark_field(
        landmark_field_module.FieldObservations.clicks(
            phantom.reference_xy_um,
            phantom.moving_xy_um,
            np.arange(1, len(phantom.reference_xy_um) + 1),
        )
    )


def _flow_inputs(field: Any) -> dict[str, Any]:
    base = best_affine(
        field,
        crop_reference=FLOW_CROP_REF,
        reference_pixel_size_um=FLOW_P_REF,
        moving_pixel_size_um=FLOW_P_MOV,
    )
    origin_r = (FLOW_CROP_REF.x0, FLOW_CROP_REF.y0)
    origin_m = (FLOW_CROP_MOV.x0, FLOW_CROP_MOV.y0)
    from merxen.manual_registration.geometry import crop_matrix

    return {
        "base_crop": crop_matrix(
            base, reference_origin_xy=origin_r, moving_origin_xy=origin_m
        ),
        "origin_r": np.array(origin_r, dtype=float),
        "origin_m": np.array(origin_m, dtype=float),
    }


def _true_moving_crop(field: Any, reference_crop_xy: np.ndarray, inputs: Any) -> Any:
    reference_um = FLOW_P_REF * (reference_crop_xy + inputs["origin_r"])
    return field.mean(reference_um) / FLOW_P_MOV - inputs["origin_m"]


@pytest.fixture(scope="module")
def flow_case(smooth_field: Any) -> tuple[Any, dict[str, Any]]:
    inputs = _flow_inputs(smooth_field)
    flow = field_flow(
        smooth_field,
        crop_reference=FLOW_CROP_REF,
        crop_moving=FLOW_CROP_MOV,
        reference_pixel_size_um=FLOW_P_REF,
        moving_pixel_size_um=FLOW_P_MOV,
        base_matrix_crop=inputs["base_crop"],
        grid_um=FLOW_GRID_UM,
    )
    return flow, inputs


def test_field_flow_grid_follows_the_contract(
    flow_case: tuple[Any, dict[str, Any]],
) -> None:
    flow, _inputs = flow_case
    factor = 6  # round(33 / 5.5)
    assert flow.factor == factor
    assert flow.flow_yx.dtype == np.float32
    assert flow.flow_yx.shape == (2, 1408 // factor + 1 + 2, 1920 // factor + 2)
    shift = (factor - 1) / 2 - factor
    np.testing.assert_allclose(
        flow.working_to_reference,
        [[factor, 0, shift], [0, factor, shift], [0, 0, 1]],
    )
    assert flow.max_displacement_um > 0.0
    default = field_flow(
        _SmoothStub(),  # type: ignore[arg-type]
        crop_reference=FLOW_CROP_REF,
        crop_moving=FLOW_CROP_MOV,
        reference_pixel_size_um=0.345,
        moving_pixel_size_um=0.345,
        base_matrix_crop=np.eye(3),
        grid_um=11.04,
    )
    assert default.factor == 32


class _SmoothStub:
    """A field stub for the grid size rule: the identity in um."""

    @staticmethod
    def mean(positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions, dtype=float)


def test_field_flow_pull_at_the_nodes_equals_the_field(
    flow_case: tuple[Any, dict[str, Any]], smooth_field: Any
) -> None:
    flow, inputs = flow_case
    height, width = flow.flow_yx.shape[1:]
    rows, columns = np.indices((height, width), dtype=float)
    nodes = np.column_stack([columns.ravel(), rows.ravel()])
    reference_crop = (
        nodes @ flow.working_to_reference[:2, :2].T + flow.working_to_reference[:2, 2]
    )
    pulled = pull_points_through_registration(
        reference_crop,
        final_matrix=inputs["base_crop"],
        flow_yx=flow.flow_yx,
        working_to_reference=flow.working_to_reference,
    )
    expected = _true_moving_crop(smooth_field, reference_crop, inputs)
    # float32 flow: the error is the float32 rounding of the displacement.
    assert np.abs(pulled - expected).max() < 1e-6


def test_field_flow_pull_between_nodes_stays_close_to_the_field(
    flow_case: tuple[Any, dict[str, Any]], smooth_field: Any
) -> None:
    flow, inputs = flow_case
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0], [1919, 1407], size=(4000, 2))
    pulled = pull_points_through_registration(
        points,
        final_matrix=inputs["base_crop"],
        flow_yx=flow.flow_yx,
        working_to_reference=flow.working_to_reference,
    )
    expected = _true_moving_crop(smooth_field, points, inputs)
    assert np.abs(pulled - expected).max() < 0.05


def test_field_flow_borders_go_through_the_flow_not_the_bare_affine(
    flow_case: tuple[Any, dict[str, Any]], smooth_field: Any
) -> None:
    flow, inputs = flow_case
    height, width = 1408, 1920
    xs = np.arange(width, dtype=float)
    ys = np.arange(height, dtype=float)
    border = np.vstack(
        [
            np.column_stack([xs, np.zeros(width)]),
            np.column_stack([xs, np.full(width, height - 1.0)]),
            np.column_stack([np.zeros(height), ys]),
            np.column_stack([np.full(height, width - 1.0), ys]),
        ]
    )
    pulled = pull_points_through_registration(
        border,
        final_matrix=inputs["base_crop"],
        flow_yx=flow.flow_yx,
        working_to_reference=flow.working_to_reference,
    )
    bare = pull_points_through_registration(border, final_matrix=inputs["base_crop"])
    expected = _true_moving_crop(smooth_field, border, inputs)
    assert np.abs(pulled - expected).max() < 0.05
    # The flow matters here: the bare affine misses the field by far more.
    assert np.abs(bare - expected).max() > 0.2


def test_field_flow_push_returns_the_reference_also_near_the_edge(
    flow_case: tuple[Any, dict[str, Any]], smooth_field: Any
) -> None:
    flow, inputs = flow_case
    reference_crop = np.array(
        [[900.0, 700.0], [3.0, 2.0], [1916.0, 1405.0], [4.0, 1400.0], [1000.0, 4.0]]
    )
    moving_crop = _true_moving_crop(smooth_field, reference_crop, inputs)
    pushed, converged = push_points_through_registration(
        moving_crop,
        final_matrix=inputs["base_crop"],
        flow_yx=flow.flow_yx,
        working_to_reference=flow.working_to_reference,
    )
    assert converged.all()
    assert np.abs(pushed - reference_crop).max() < 0.05


def test_best_affine_of_an_affine_field_is_that_affine() -> None:
    linear = np.array([[1.02, 0.03], [-0.04, 0.97]])
    offset = np.array([120.0, -75.0])

    class AffineField:
        @staticmethod
        def mean(positions: np.ndarray) -> np.ndarray:
            return np.asarray(positions) @ linear.T + offset

    matrix = best_affine(
        AffineField(),
        crop_reference=FLOW_CROP_REF,
        reference_pixel_size_um=FLOW_P_REF,
        moving_pixel_size_um=FLOW_P_MOV,
    )
    # moving px -> reference px; compare through its action on moving points.
    reference_px = np.array([[500.0, 400.0], [1500.0, 1100.0], [300.0, 1300.0]])
    moving_px = ((reference_px * FLOW_P_REF) @ linear.T + offset) / FLOW_P_MOV
    mapped = moving_px @ matrix[:2, :2].T + matrix[:2, 2]
    np.testing.assert_allclose(mapped, reference_px, atol=1e-6)


def test_best_affine_uses_only_the_tissue_mask_when_given() -> None:
    class SplitField:
        """An affine map on the left half of the crop and a shifted one on the right."""

        @staticmethod
        def mean(positions: np.ndarray) -> np.ndarray:
            shifted = np.asarray(positions, dtype=float).copy()
            shifted[positions[:, 0] > 5000.0, 1] += 200.0
            return shifted

    def left_half(xy_um: np.ndarray) -> np.ndarray:
        return np.asarray(xy_um)[:, 0] <= 5000.0

    kwargs: dict[str, Any] = {
        "crop_reference": FLOW_CROP_REF,
        "reference_pixel_size_um": 5.0,
        "moving_pixel_size_um": 5.0,
    }
    masked = best_affine(SplitField(), mask_xy_um=left_half, **kwargs)
    np.testing.assert_allclose(masked, np.eye(3), atol=1e-9)
    unmasked = best_affine(SplitField(), **kwargs)
    assert abs(unmasked[1, 2]) > 1.0
    # A mask with almost no tissue falls back to the whole crop.
    nothing = best_affine(SplitField(), mask_xy_um=lambda xy: xy[:, 0] < -1, **kwargs)
    np.testing.assert_allclose(nothing, unmasked)


def test_field_sd_grid_shape_and_matrix(smooth_field: Any) -> None:
    sd, matrix = field_sd_grid(
        smooth_field,
        crop_reference=FLOW_CROP_REF,
        reference_pixel_size_um=FLOW_P_REF,
        grid_um=88.0,
    )
    factor = 16  # round(88 / 5.5)
    assert sd.dtype == np.float32
    assert sd.shape == (-(-1408 // factor) + 2, -(-1920 // factor) + 2)
    assert matrix[0, 0] == factor and matrix[0, 2] == (factor - 1) / 2 - factor
    assert np.isfinite(sd).all() and (sd > 0).all()


def test_tissue_mask_function_looks_up_the_nearest_working_pixel() -> None:
    tissue = np.zeros((10, 20), dtype=bool)
    tissue[2:6, 4:9] = True
    representation = SimpleNamespace(
        tissue=tissue,
        um_to_px=lambda xy: np.asarray(xy, dtype=float) / 10.0 - 1.0,
    )
    inside = tissue_mask_function(representation)  # type: ignore[arg-type]
    points_um = np.array(
        [[60.0, 40.0], [10.0, 10.0], [5000.0, 40.0], [-500.0, 40.0], [100.0, 90.0]]
    )
    np.testing.assert_array_equal(inside(points_um), [True, False, False, False, False])
