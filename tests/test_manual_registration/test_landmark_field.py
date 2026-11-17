"""Unit tests of the robust landmark field (math, EM, LOO, serialisation)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from merxen.manual_registration import landmark_field as lf
from merxen.manual_registration import schema
from merxen.manual_registration.simulate import make_field_phantom

CLICK_SIGMA = 30.0
WINDOW_SIGMA = 15.0


def _hyper(
    *,
    long_length: float = 20000.0,
    long_signal: float = 80.0,
    short_length: float | None = 1200.0,
    short_signal: float | None = 30.0,
    noise: dict[str, float] | None = None,
) -> lf.FieldHyperparameters:
    return lf.FieldHyperparameters(
        long_length_um=long_length,
        long_signal_um=long_signal,
        short_length_um=short_length,
        short_signal_um=short_signal,
        noise_um=noise or {"click": CLICK_SIGMA, "window": WINDOW_SIGMA},
    )


def _rotation(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )


def _random_shapes(rng: np.random.Generator, count: int, ratio: float) -> np.ndarray:
    shapes = np.empty((count, 2, 2))
    for index in range(count):
        rotation = _rotation(rng.uniform(0.0, 180.0))
        stretch = rng.uniform(1.0, ratio)
        shapes[index] = rotation @ np.diag([stretch, 1.0 / stretch]) @ rotation.T
    return shapes


def _mixed_observations(
    seed: int = 0, n_clicks: int = 12, n_windows: int = 8
) -> lf.FieldObservations:
    """Clicks with anisotropic shapes plus windows with fixed covariances."""
    rng = np.random.default_rng(seed)
    clicks_ref = rng.uniform(0.0, 20000.0, size=(n_clicks, 2))
    clicks_mov = (
        clicks_ref
        + np.array([100.0, -50.0])
        + 0.01 * (clicks_ref - 10000.0)
        + rng.normal(0.0, 40.0, size=(n_clicks, 2))
    )
    clicks = lf.FieldObservations.clicks(
        clicks_ref,
        clicks_mov,
        np.arange(1, n_clicks + 1),
        shape_cov=_random_shapes(rng, n_clicks, 3.0),
    )
    windows_ref = rng.uniform(0.0, 20000.0, size=(n_windows, 2))
    windows_mov = (
        windows_ref
        + np.array([100.0, -50.0])
        + 0.01 * (windows_ref - 10000.0)
        + rng.normal(0.0, 20.0, size=(n_windows, 2))
    )
    fixed = np.empty((n_windows, 2, 2))
    for index in range(n_windows):
        rotation = _rotation(rng.uniform(0.0, 180.0))
        fixed[index] = (
            rotation @ np.diag([rng.uniform(10.0, 30.0) ** 2, 5.0**2]) @ rotation.T
        )
    return clicks.concat(lf.FieldObservations.windows(windows_ref, windows_mov, fixed))


def _clean_phantom_observations(
    seed: int, *, n_clicks: int = 30, n_gross: int = 0
) -> tuple[lf.FieldObservations, list[str]]:
    phantom = make_field_phantom(
        seed=seed,
        n_clicks=n_clicks,
        short_signal_um=0.0,
        ridge_fraction=0.0,
        n_medium=0,
        n_gross=n_gross,
        click_noise_um=CLICK_SIGMA,
    )
    observations = lf.FieldObservations.clicks(
        phantom.reference_xy_um,
        phantom.moving_xy_um,
        np.arange(1, n_clicks + 1),
    )
    return observations, list(phantom.kind)


def _dense_reference(
    observations: lf.FieldObservations,
    hyper: lf.FieldHyperparameters,
    weights: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and latent covariance by explicit dense inversion."""
    count = observations.n
    ref = observations.reference_xy_um
    distance = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    components = [(hyper.long_length_um, hyper.long_signal_um)]
    if hyper.short_length_um is not None:
        components.append((hyper.short_length_um, hyper.short_signal_um))  # type: ignore[arg-type]
    kernel = sum(s**2 * lf.matern52(distance, length) for length, s in components)
    big = np.zeros((2 * count, 2 * count))
    big[:count, :count] = kernel
    big[count:, count:] = kernel
    for i in range(count):
        sigma = hyper.noise_um[observations.obs_class[i]]
        block = (
            sigma**2 * observations.shape_cov[i] + observations.fixed_cov_um2[i]
        ) / weights[i]
        big[i, i] += block[0, 0]
        big[i, count + i] += block[0, 1]
        big[count + i, i] += block[1, 0]
        big[count + i, count + i] += block[1, 1]
    big += lf.JITTER_UM2 * np.eye(2 * count)
    centre = ref[[n == "click" for n in observations.obs_class]].mean(axis=0)

    def basis(points: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.ones(len(points)), (points - centre) / lf.TREND_SCALE_UM]
        )

    design = np.zeros((2 * count, 6))
    design[:count, :3] = basis(ref)
    design[count:, 3:] = basis(ref)
    target = np.concatenate(
        [
            observations.moving_xy_um[:, 0] - ref[:, 0],
            observations.moving_xy_um[:, 1] - ref[:, 1],
        ]
    )
    inverse = np.linalg.inv(big)
    info_inverse = np.linalg.inv(design.T @ inverse @ design)
    beta = info_inverse @ design.T @ inverse @ target
    alpha = inverse @ (target - design @ beta)
    total = sum(s**2 for _, s in components)
    means = np.empty((len(queries), 2))
    covariances = np.empty((len(queries), 2, 2))
    for index, point in enumerate(queries):
        cross = sum(
            s**2 * lf.matern52(np.linalg.norm(ref - point, axis=1), length)
            for length, s in components
        )
        cross_full = np.zeros((2 * count, 2))
        cross_full[:count, 0] = cross
        cross_full[count:, 1] = cross
        h_star = np.zeros((2, 6))
        h_star[0, :3] = basis(point[None, :])[0]
        h_star[1, 3:] = basis(point[None, :])[0]
        means[index] = point + h_star @ beta + cross_full.T @ alpha
        residual_basis = h_star.T - design.T @ inverse @ cross_full
        covariances[index] = (
            total * np.eye(2)
            - cross_full.T @ inverse @ cross_full
            + residual_basis.T @ info_inverse @ residual_basis
        )
    return means, covariances


# --------------------------------------------------------------------- kernel
def test_matern52_matches_known_values() -> None:
    assert lf.matern52(np.array([0.0]), 10.0)[0] == pytest.approx(1.0)
    b = math.sqrt(5.0)
    assert lf.matern52(np.array([10.0]), 10.0)[0] == pytest.approx(
        (1.0 + b + 5.0 / 3.0) * math.exp(-b)
    )
    assert np.all(np.diff(lf.matern52(np.linspace(0.0, 100.0, 50), 10.0)) < 0.0)


def test_matern52_kernel_derivatives_match_central_differences() -> None:
    distance = np.linspace(0.0, 5000.0, 11)[None, :]
    length, signal = 1500.0, 40.0
    _, derivatives = lf._kernel_parts(distance, [length], [signal])

    def kernel_of(log_length: float, log_signal: float) -> np.ndarray:
        return np.asarray(
            math.exp(2 * log_signal) * lf.matern52(distance, math.exp(log_length))
        )

    step = 1e-6
    d_length = (
        kernel_of(math.log(length) + step, math.log(signal))
        - kernel_of(math.log(length) - step, math.log(signal))
    ) / (2 * step)
    d_signal = (
        kernel_of(math.log(length), math.log(signal) + step)
        - kernel_of(math.log(length), math.log(signal) - step)
    ) / (2 * step)
    np.testing.assert_allclose(derivatives[0], d_length, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(derivatives[1], d_signal, rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------------ the gradient
def test_reml_gradient_matches_central_differences() -> None:
    observations = _mixed_observations(seed=1, n_clicks=7, n_windows=5)
    options = lf.FieldOptions(components=2)
    centre = lf._trend_centre(observations)
    context = lf._make_context(observations, centre, 2, options)
    rng = np.random.default_rng(3)
    weights = rng.uniform(0.3, 1.0, size=observations.n)
    theta = np.log([9000.0, 120.0, 1500.0, 25.0, 35.0, 18.0])

    def value(point: np.ndarray) -> float:
        return float(lf._reml(point, context, weights)[0])

    analytic = lf._reml(theta, context, weights)[1]
    numeric = approx_fprime(theta, value, 1e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


# -------------------------------------------------------------------- prediction
def test_fixed_prediction_matches_dense_reference() -> None:
    observations = _mixed_observations(seed=2)
    hyper = _hyper()
    rng = np.random.default_rng(5)
    weights = rng.uniform(0.2, 1.0, size=observations.n)
    field = lf.landmark_field_from_hyperparameters(observations, hyper, weights)
    queries = rng.uniform(-2000.0, 22000.0, size=(20, 2))
    means, covariances = _dense_reference(observations, hyper, weights, queries)
    np.testing.assert_allclose(field.mean(queries), means, rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(
        field.covariance(queries), covariances, rtol=1e-9, atol=1e-7
    )
    np.testing.assert_allclose(
        field.sd_um(queries),
        np.sqrt(np.trace(covariances, axis1=1, axis2=2) / 2.0),
        rtol=1e-9,
    )


def test_closed_form_loo_matches_brute_force_refits() -> None:
    observations = _mixed_observations(seed=4, n_clicks=10, n_windows=6)
    hyper = _hyper()
    rng = np.random.default_rng(6)
    weights = rng.uniform(0.2, 1.0, size=observations.n)
    field = lf.landmark_field_from_hyperparameters(observations, hyper, weights)
    loo = field.loo()
    assert field.loo() is loo
    for index in range(observations.n):
        keep = np.arange(observations.n) != index
        reduced = lf.landmark_field_from_hyperparameters(
            observations.subset(keep), hyper, weights[keep]
        )
        held_out = reduced.predict_residual(
            observations.reference_xy_um[index],
            observations.moving_xy_um[index],
            obs_class=observations.obs_class[index],
            shape_cov=observations.shape_cov[index],
            fixed_cov_um2=observations.fixed_cov_um2[index],
        )
        np.testing.assert_allclose(
            loo.residual_um[index], held_out.residual_um[0], rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            loo.covariance_um2[index],
            held_out.covariance_um2[0],
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            loo.latent_covariance_um2[index],
            held_out.latent_covariance_um2[0],
            rtol=1e-8,
            atol=1e-8,
        )
        assert loo.d2[index] == pytest.approx(held_out.d2[0], rel=1e-8)
        assert loo.p_value[index] == pytest.approx(np.exp(-loo.d2[index] / 2))
        np.testing.assert_allclose(
            loo.predicted_um[index] + loo.residual_um[index],
            observations.moving_xy_um[index],
        )
    assert loo.magnitude_reference_um.shape == (observations.n,)


# ------------------------------------------------------------------------- EM
def test_em_downweights_one_gross_misclick_and_keeps_clean_weights() -> None:
    observations, kind = _clean_phantom_observations(seed=7, n_gross=1)
    field = lf.fit_landmark_field(observations)
    gross = kind.index("gross")
    assert field.converged
    assert field.weights[gross] < 0.05
    clean = np.delete(field.weights, gross)
    assert np.all((clean >= 0.5) & (clean <= 1.6))


def test_gaussian_fit_keeps_every_weight_at_one() -> None:
    observations, _ = _clean_phantom_observations(seed=7, n_gross=1)
    field = lf.fit_landmark_field(observations, lf.FieldOptions(nu=math.inf))
    np.testing.assert_array_equal(field.weights, np.ones(observations.n))
    assert field.em_iterations == 0
    assert field.to_dict()["student_t_nu"] is None


def test_em_exempt_click_keeps_weight_one_but_is_still_flagged() -> None:
    observations, _ = _clean_phantom_observations(seed=8)
    moving = observations.moving_xy_um.copy()
    offset = np.array([300.0, 0.0])
    exempt_index, plain_index = 3, 17
    moving[[exempt_index, plain_index]] += offset
    exempt = np.zeros(observations.n, dtype=bool)
    exempt[exempt_index] = True
    shifted = lf.FieldObservations.clicks(
        observations.reference_xy_um,
        moving,
        observations.obs_id,
        em_exempt=exempt,
    )
    field = lf.fit_landmark_field(shifted)
    assert field.weights[exempt_index] == 1.0
    assert field.weights[plain_index] < 0.2
    loo = field.loo()
    in_sample = np.linalg.norm(
        shifted.moving_xy_um - field.mean(shifted.reference_xy_um), axis=1
    )
    assert in_sample[exempt_index] < in_sample[plain_index]
    assert loo.p_value[exempt_index] < 0.01


def test_weight_divides_the_whole_noise_covariance() -> None:
    observations = _mixed_observations(seed=9, n_clicks=8, n_windows=6)
    base = np.ones(observations.n)
    halved = base.copy()
    window_index = observations.n - 1
    halved[window_index] = 0.5
    one = lf.landmark_field_from_hyperparameters(observations, _hyper(), base)
    two = lf.landmark_field_from_hyperparameters(observations, _hyper(), halved)
    np.testing.assert_allclose(
        two._posterior.noise_flat[window_index],
        2.0 * one._posterior.noise_flat[window_index],
    )
    other = np.delete(np.arange(observations.n), window_index)
    np.testing.assert_allclose(
        two._posterior.noise_flat[other], one._posterior.noise_flat[other]
    )
    assert np.any(observations.fixed_cov_um2[window_index] != 0.0)


def test_anisotropic_shape_constrains_the_field_across_its_long_axis() -> None:
    rng = np.random.default_rng(10)
    others = rng.uniform(0.0, 40000.0, size=(6, 2))
    others = np.vstack(
        [others, [[2000.0, 2000.0], [38000.0, 2000.0], [20000.0, 38000.0]]]
    )
    point = np.array([[20000.0, 20000.0]])
    reference = np.vstack([others, point])
    moving = reference + 50.0
    direction = np.array([math.cos(0.5), math.sin(0.5)])
    shapes = lf.isotropic_shapes(len(reference))
    shapes[-1] = lf.ridge_shape(direction, ratio=4.0)
    observations = lf.FieldObservations.clicks(
        reference, moving, np.arange(1, len(reference) + 1), shape_cov=shapes
    )
    hyper = _hyper(
        long_length=500.0,
        long_signal=200.0,
        short_length=None,
        short_signal=None,
        noise={"click": 40.0},
    )
    field = lf.landmark_field_from_hyperparameters(
        observations, hyper, np.ones(observations.n)
    )
    covariance = field.covariance(point)[0]
    values, vectors = np.linalg.eigh(covariance)
    # The noise is elongated along ``direction``: least constrained there.
    assert values[1] > 1.5 * values[0]
    assert abs(vectors[:, 1] @ direction) > 0.99


def test_window_noise_floor_holds_at_every_direction() -> None:
    phantom = make_field_phantom(
        seed=11,
        n_clicks=30,
        n_windows=40,
        ridge_fraction=0.0,
        n_medium=0,
        n_gross=0,
        window_wrong_fraction=0.0,
    )
    fixed = lf.FieldObservations.windows(
        phantom.window_reference_xy_um,
        phantom.window_moving_xy_um,
        phantom.window_peak_cov_um2,
    )
    clicks = lf.FieldObservations.clicks(
        phantom.reference_xy_um, phantom.moving_xy_um, np.arange(1, 31)
    )
    observations = clicks.concat(fixed)
    elongated = np.diag([30.0**2, 7.5**2])
    unit = lf.DEFAULT_NOISE_BOUNDS_UM["window"][0] ** 2 * np.eye(2) + elongated
    assert np.linalg.eigvalsh(unit)[0] >= 15.0**2
    field = lf.fit_landmark_field(observations)
    assert field.hyperparameters.noise_um["window"] >= 15.0
    stricter = lf.fit_landmark_field(
        observations,
        lf.FieldOptions(
            noise_bounds_um={"click": (3.0, 300.0), "window": (25.0, 100.0)}
        ),
    )
    assert stricter.hyperparameters.noise_um["window"] >= 25.0 - 1e-9
    loo_unit = field.loo().covariance_um2
    window_rows = [
        i for i, name in enumerate(observations.obs_class) if name == "window"
    ]
    floor_variance = field.hyperparameters.noise_um["window"] ** 2
    for index in window_rows:
        smallest = np.linalg.eigvalsh(loo_unit[index])[0]
        assert smallest >= floor_variance - 1e-6


def test_exactly_affine_truth_is_recovered() -> None:
    rng = np.random.default_rng(12)
    reference = rng.uniform(0.0, 20000.0, size=(25, 2))
    matrix = np.array([[1.012, -0.03], [0.02, 0.991]])
    offset = np.array([220.0, -140.0])
    moving = reference @ matrix.T + offset
    observations = lf.FieldObservations.clicks(reference, moving, np.arange(1, 26))
    field = lf.fit_landmark_field(observations)
    queries = rng.uniform(0.0, 20000.0, size=(40, 2))
    np.testing.assert_allclose(
        field.mean(queries), queries @ matrix.T + offset, atol=0.1
    )
    expected = np.eye(3)
    expected[:2, :2] = matrix
    expected[:2, 2] = offset
    np.testing.assert_allclose(field.trend_matrix(), expected, rtol=1e-6, atol=1e-6)


def test_affine_truth_with_one_micrometre_noise_is_recovered_closely() -> None:
    rng = np.random.default_rng(13)
    reference = rng.uniform(0.0, 20000.0, size=(40, 2))
    matrix = np.array([[1.012, -0.03], [0.02, 0.991]])
    offset = np.array([220.0, -140.0])
    moving = reference @ matrix.T + offset + rng.normal(0.0, 1.0, size=(40, 2))
    observations = lf.FieldObservations.clicks(reference, moving, np.arange(1, 41))
    field = lf.fit_landmark_field(observations)
    queries = rng.uniform(2000.0, 18000.0, size=(60, 2))
    error = np.linalg.norm(field.mean(queries) - (queries @ matrix.T + offset), axis=1)
    assert np.median(error) < 1.0


def test_fit_is_invariant_to_a_rigid_motion_of_all_inputs() -> None:
    observations = _mixed_observations(seed=14, n_clicks=14, n_windows=10)
    rotation = _rotation(90.0)
    translation = np.array([10000.0, 10000.0])

    def move(points: np.ndarray) -> np.ndarray:
        return np.asarray(points @ rotation.T + translation)

    moved = lf.FieldObservations(
        reference_xy_um=move(observations.reference_xy_um),
        moving_xy_um=move(observations.moving_xy_um),
        shape_cov=rotation @ observations.shape_cov @ rotation.T,
        fixed_cov_um2=rotation @ observations.fixed_cov_um2 @ rotation.T,
        obs_class=observations.obs_class,
        obs_id=observations.obs_id,
    )
    original = lf.fit_landmark_field(observations)
    rigid = lf.fit_landmark_field(moved)
    queries = np.random.default_rng(15).uniform(0.0, 20000.0, size=(30, 2))
    np.testing.assert_allclose(
        rigid.mean(move(queries)), move(original.mean(queries)), atol=1e-4
    )
    np.testing.assert_allclose(
        rigid.loo().magnitude_reference_um,
        original.loo().magnitude_reference_um,
        rtol=1e-5,
    )
    np.testing.assert_allclose(rigid.loo().p_value, original.loo().p_value, rtol=1e-5)
    np.testing.assert_allclose(rigid.weights, original.weights, rtol=1e-5)


def test_fixed_path_is_exactly_covariant_with_a_rigid_motion() -> None:
    observations = _mixed_observations(seed=16)
    rotation = _rotation(90.0)
    translation = np.array([10000.0, -3000.0])
    weights = np.random.default_rng(17).uniform(0.3, 1.0, size=observations.n)

    def move(points: np.ndarray) -> np.ndarray:
        return np.asarray(points @ rotation.T + translation)

    moved = lf.FieldObservations(
        reference_xy_um=move(observations.reference_xy_um),
        moving_xy_um=move(observations.moving_xy_um),
        shape_cov=rotation @ observations.shape_cov @ rotation.T,
        fixed_cov_um2=rotation @ observations.fixed_cov_um2 @ rotation.T,
        obs_class=observations.obs_class,
        obs_id=observations.obs_id,
    )
    original = lf.landmark_field_from_hyperparameters(observations, _hyper(), weights)
    rigid = lf.landmark_field_from_hyperparameters(moved, _hyper(), weights)
    queries = np.random.default_rng(18).uniform(0.0, 20000.0, size=(25, 2))
    np.testing.assert_allclose(
        rigid.mean(move(queries)), move(original.mean(queries)), atol=1e-6
    )
    np.testing.assert_allclose(rigid.loo().d2, original.loo().d2, rtol=1e-6)


# ----------------------------------------------------------- components and bounds
def test_components_resolve_to_one_for_clicks_only_and_two_with_windows() -> None:
    clicks_only, _ = _clean_phantom_observations(seed=19)
    assert lf.fit_landmark_field(clicks_only).components == 1
    mixed = _mixed_observations(seed=20, n_clicks=14, n_windows=10)
    mixed_field = lf.fit_landmark_field(mixed)
    assert mixed_field.components == 2
    assert mixed_field.hyperparameters.short_length_um is not None
    forced = lf.fit_landmark_field(clicks_only, lf.FieldOptions(components=2))
    assert forced.components == 2


def _within(bounds: tuple[float, float], value: float) -> bool:
    """Inside the bounds up to the rounding of exp(log(bound))."""
    return bounds[0] * (1 - 1e-9) <= value <= bounds[1] * (1 + 1e-9)


def test_fitted_hyperparameters_lie_inside_the_bounds() -> None:
    options = lf.FieldOptions()
    bounds = options.bounds
    mixed = lf.fit_landmark_field(
        _mixed_observations(seed=21, n_clicks=14, n_windows=10)
    )
    hyper = mixed.hyperparameters
    assert _within(bounds.long_length_um, hyper.long_length_um)
    assert _within(bounds.long_signal_um, hyper.long_signal_um)
    assert hyper.short_length_um is not None and hyper.short_signal_um is not None
    assert _within(bounds.short_length_um, hyper.short_length_um)
    assert _within(bounds.short_signal_um, hyper.short_signal_um)
    for name, sigma in hyper.noise_um.items():
        low, high = options.noise_bounds_um[name]
        assert _within((low, high), sigma)
    clicks_only, _ = _clean_phantom_observations(seed=22)
    single = lf.fit_landmark_field(clicks_only).hyperparameters
    assert _within(bounds.single_length_um, single.long_length_um)
    assert single.short_length_um is None and single.short_signal_um is None


def test_forced_two_components_on_single_scale_truth_drop_the_short_signal() -> None:
    observations, _ = _clean_phantom_observations(seed=23, n_clicks=40)
    field = lf.fit_landmark_field(observations, lf.FieldOptions(components=2))
    assert field.hyperparameters.short_signal_um is not None
    assert field.hyperparameters.short_signal_um <= 1.0


# ------------------------------------------------------------------- determinism
def test_fits_are_deterministic_and_the_fixed_path_is_bitwise_identical() -> None:
    observations = _mixed_observations(seed=24, n_clicks=12, n_windows=8)
    first = lf.fit_landmark_field(observations)
    second = lf.fit_landmark_field(observations)
    queries = np.random.default_rng(25).uniform(0.0, 20000.0, size=(20, 2))
    np.testing.assert_allclose(first.mean(queries), second.mean(queries), rtol=1e-12)
    np.testing.assert_allclose(first.weights, second.weights, rtol=1e-12)
    weights = np.full(observations.n, 0.8)
    one = lf.landmark_field_from_hyperparameters(observations, _hyper(), weights)
    two = lf.landmark_field_from_hyperparameters(observations, _hyper(), weights)
    assert one.digest == two.digest
    assert np.array_equal(one.mean(queries), two.mean(queries))
    assert np.array_equal(one.loo().d2, two.loo().d2)


# ----------------------------------------------------------------- serialisation
@pytest.mark.parametrize("with_windows", [False, True])
def test_to_dict_round_trip_reproduces_the_field(with_windows: bool) -> None:
    observations = _mixed_observations(
        seed=26, n_clicks=12, n_windows=8 if with_windows else 0
    )
    field = lf.fit_landmark_field(observations)
    payload = json.loads(json.dumps(field.to_dict(), allow_nan=False))
    rebuilt = lf.LandmarkField.from_dict(payload)
    queries = np.random.default_rng(27).uniform(-1000.0, 21000.0, size=(50, 2))
    np.testing.assert_allclose(rebuilt.mean(queries), field.mean(queries), rtol=1e-10)
    np.testing.assert_allclose(
        rebuilt.covariance(queries), field.covariance(queries), rtol=1e-10, atol=1e-9
    )
    assert rebuilt.digest == field.digest
    assert rebuilt.components == field.components
    assert payload["kernel"]["components"][0]["name"] == "long"


def test_infinite_nu_round_trips_through_null() -> None:
    observations, _ = _clean_phantom_observations(seed=28)
    field = lf.fit_landmark_field(observations, lf.FieldOptions(nu=math.inf))
    payload = json.loads(json.dumps(field.to_dict(), allow_nan=False))
    assert payload["student_t_nu"] is None
    assert payload["fit"]["options"]["nu"] is None
    assert lf.LandmarkField.from_dict(payload).options.nu == math.inf


def test_digest_equals_the_schema_canonical_digest() -> None:
    observations, _ = _clean_phantom_observations(seed=29)
    payload = lf.fit_landmark_field(observations).to_dict()
    body = {key: value for key, value in payload.items() if key != "digest"}
    assert payload["digest"] == schema.canonical_sha256(body)
    assert payload["schema"] == lf.FIELD_SCHEMA
    assert payload["schema_version"] == 1
    assert payload["algorithm"] == lf.FIELD_ALGORITHM
    assert payload["trend"]["scale_um"] == lf.TREND_SCALE_UM


def test_missing_em_exempt_key_reads_as_false() -> None:
    observations, _ = _clean_phantom_observations(seed=30)
    payload = json.loads(json.dumps(lf.fit_landmark_field(observations).to_dict()))
    del payload["observations"]["em_exempt"]
    payload["digest"] = lf._canonical_sha256(
        {key: value for key, value in payload.items() if key != "digest"}
    )
    rebuilt = lf.LandmarkField.from_dict(payload)
    assert not rebuilt.observations.em_exempt.any()  # type: ignore[union-attr]


def _tamper_noise(payload: dict[str, object]) -> None:
    payload["noise_um"]["click"] += 1.0  # type: ignore[index]


def _tamper_weight(payload: dict[str, object]) -> None:
    payload["observations"]["weight"][0] *= 0.5  # type: ignore[index]


def _tamper_position(payload: dict[str, object]) -> None:
    payload["observations"]["moving_xy_um"][2][0] += 1.0  # type: ignore[index]


def _tamper_digest(payload: dict[str, object]) -> None:
    payload["digest"] = "sha256:" + "0" * 64


def _tamper_version(payload: dict[str, object]) -> None:
    payload["schema_version"] = 2


def _tamper_schema(payload: dict[str, object]) -> None:
    payload["schema"] = "something.else"


@pytest.mark.parametrize(
    "tamper",
    [
        _tamper_noise,
        _tamper_weight,
        _tamper_position,
        _tamper_digest,
        _tamper_version,
        _tamper_schema,
    ],
)
def test_tampering_with_the_payload_raises_value_error(
    tamper: Callable[[dict[str, object]], None],
) -> None:
    observations, _ = _clean_phantom_observations(seed=31)
    payload = json.loads(json.dumps(lf.fit_landmark_field(observations).to_dict()))
    tamper(payload)
    with pytest.raises(ValueError):
        lf.LandmarkField.from_dict(payload)


def test_wrong_trend_coefficients_with_a_valid_digest_raise_value_error() -> None:
    observations, _ = _clean_phantom_observations(seed=32)
    payload = json.loads(json.dumps(lf.fit_landmark_field(observations).to_dict()))
    payload["trend"]["coefficients"][0][0] += 5.0
    payload["digest"] = lf._canonical_sha256(
        {key: value for key, value in payload.items() if key != "digest"}
    )
    with pytest.raises(ValueError, match="trend coefficients"):
        lf.LandmarkField.from_dict(payload)


# ------------------------------------------------------------- points not in the fit
def test_predict_residual_equals_the_loo_of_the_added_point() -> None:
    observations = _mixed_observations(seed=33, n_clicks=10, n_windows=6)
    hyper = _hyper()
    weights = np.full(observations.n, 0.9)
    field = lf.landmark_field_from_hyperparameters(observations, hyper, weights)
    point_ref = np.array([[7000.0, 12000.0]])
    point_mov = point_ref + np.array([[260.0, -40.0]])
    shape = lf.ridge_shape([1.0, 1.0])
    outside = field.predict_residual(point_ref, point_mov, shape_cov=shape)
    extra = lf.FieldObservations.clicks(
        point_ref, point_mov, [999], shape_cov=shape[None]
    )
    enlarged = lf.landmark_field_from_hyperparameters(
        observations.concat(extra), hyper, np.append(weights, 1.0)
    )
    inside = enlarged.loo()
    np.testing.assert_allclose(
        inside.residual_um[-1], outside.residual_um[0], rtol=1e-7, atol=1e-7
    )
    assert inside.d2[-1] == pytest.approx(outside.d2[0], rel=1e-7)
    assert outside.p_value[0] == pytest.approx(math.exp(-outside.d2[0] / 2))
    with pytest.raises(ValueError, match="class"):
        field.predict_residual(point_ref, point_mov, obs_class="loftr")


# ------------------------------------------------------------------- suggestions
def _suggestion_setup() -> tuple[lf.LandmarkField, np.ndarray, np.ndarray]:
    observations = _mixed_observations(seed=34, n_clicks=9, n_windows=0)
    hyper = _hyper(
        long_length=6000.0,
        long_signal=150.0,
        short_length=None,
        short_signal=None,
        noise={"click": 30.0},
    )
    field = lf.landmark_field_from_hyperparameters(
        observations, hyper, np.full(observations.n, 0.9)
    )
    axis = np.arange(0.0, 20001.0, 1000.0)
    candidates = np.array([(x, y) for y in axis for x in axis])
    weights = np.where(candidates[:, 0] < 12000.0, 1.0, 0.3)
    weights[::7] = 0.0
    return field, candidates, weights


def test_first_suggestion_is_the_argmax_of_weight_times_sd() -> None:
    field, candidates, weights = _suggestion_setup()
    picks = lf.suggest_landmarks(
        field, candidates, weights, n=1, exclusion_radius_um=0.0
    )
    sd = field.sd_um(candidates)
    score = np.where(weights > 0.0, weights * sd, -np.inf)
    assert len(picks) == 1
    np.testing.assert_array_equal(
        picks[0].reference_xy_um, candidates[int(np.argmax(score))]
    )
    assert picks[0].rank == 1
    assert picks[0].score == pytest.approx(float(score.max()))
    assert picks[0].sd_um == pytest.approx(float(sd[int(np.argmax(score))]))


def test_suggestions_respect_spacing_exclusion_and_zero_weights() -> None:
    field, candidates, weights = _suggestion_setup()
    picks = lf.suggest_landmarks(
        field,
        candidates,
        weights,
        n=12,
        min_spacing_um=2500.0,
        exclusion_radius_um=1500.0,
    )
    assert len(picks) >= 4
    positions = np.array([p.reference_xy_um for p in picks])
    for i in range(len(positions)):
        for j in range(i):
            assert np.linalg.norm(positions[i] - positions[j]) > 2500.0
    clicks = field.observations.reference_xy_um
    for position in positions:
        assert np.min(np.linalg.norm(clicks - position, axis=1)) > 1500.0
        index = int(np.flatnonzero(np.all(candidates == position, axis=1))[0])
        assert weights[index] > 0.0
    assert [p.rank for p in picks] == list(range(1, len(picks) + 1))


def test_sequential_suggestion_sd_equals_a_refit_with_hypothetical_clicks() -> None:
    field, candidates, weights = _suggestion_setup()
    picks = lf.suggest_landmarks(
        field,
        candidates,
        weights,
        n=4,
        min_spacing_um=1000.0,
        exclusion_radius_um=500.0,
    )
    assert len(picks) == 4
    observations = field.observations
    hyper = field.hyperparameters
    for count in range(1, len(picks)):
        earlier = np.array([p.reference_xy_um for p in picks[:count]])
        hypothetical = lf.FieldObservations.clicks(
            earlier, earlier + 10.0, np.arange(900, 900 + count)
        )
        refit = lf.landmark_field_from_hyperparameters(
            observations.concat(hypothetical),
            hyper,
            np.concatenate([field.weights, np.ones(count)]),
            field.options,
        )
        expected = refit.covariance(picks[count].reference_xy_um[None, :])[0]
        np.testing.assert_allclose(
            picks[count].covariance_um2, expected, rtol=1e-6, atol=1e-6
        )
        assert picks[count].sd_um == pytest.approx(
            math.sqrt(np.trace(expected) / 2.0), rel=1e-6
        )


def test_suggestions_stop_when_no_candidate_is_left() -> None:
    field, candidates, _ = _suggestion_setup()
    assert lf.suggest_landmarks(field, candidates, np.zeros(len(candidates))) == []
    few = lf.suggest_landmarks(
        field,
        candidates[:3] + 500.0,
        np.ones(3),
        n=10,
        min_spacing_um=10.0,
        exclusion_radius_um=0.0,
    )
    assert len(few) == 3


# -------------------------------------------------------------------- grids, misc
def test_evaluate_on_grid_matches_pointwise_mean_and_masks_with_nan() -> None:
    field, _, _ = _suggestion_setup()
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    grid = lf.evaluate_on_grid(
        field, origin_um=(1000.0, 2000.0), step_um=500.0, shape_yx=(4, 5), mask=mask
    )
    assert grid.mean_um.shape == (4, 5, 2)
    assert grid.sd_um is not None and grid.sd_um.dtype == np.float32
    assert np.isnan(grid.mean_um[0]).all() and np.isnan(grid.sd_um[0]).all()
    node = np.array([[1000.0 + 500.0 * 2, 2000.0 + 500.0 * 1]])
    np.testing.assert_allclose(grid.mean_um[1, 2], field.mean(node)[0])
    assert grid.sd_um[1, 2] == pytest.approx(float(field.sd_um(node)[0]), rel=1e-5)
    plain = lf.evaluate_on_grid(
        field, origin_um=(0.0, 0.0), step_um=100.0, shape_yx=(2, 2), with_sd=False
    )
    assert plain.sd_um is None and not np.isnan(plain.mean_um).any()


def test_jacobian_of_an_affine_field_is_the_trend_matrix() -> None:
    rng = np.random.default_rng(35)
    reference = rng.uniform(0.0, 20000.0, size=(20, 2))
    matrix = np.array([[1.01, -0.02], [0.03, 0.99]])
    moving = reference @ matrix.T + 100.0
    field = lf.fit_landmark_field(
        lf.FieldObservations.clicks(reference, moving, np.arange(1, 21))
    )
    jacobian = field.jacobian(np.array([[5000.0, 6000.0]]))[0]
    np.testing.assert_allclose(jacobian, matrix, atol=1e-5)


def test_ridge_and_normalise_shapes_have_unit_determinant() -> None:
    shape = lf.ridge_shape([3.0, 4.0])
    assert np.linalg.det(shape) == pytest.approx(1.0)
    values, vectors = np.linalg.eigh(shape)
    assert values[1] == pytest.approx(lf.RIDGE_SHAPE_RATIO)
    assert abs(vectors[:, 1] @ np.array([0.6, 0.8])) == pytest.approx(1.0)
    scaled = lf.normalise_shape(np.array([[4.0, 0.0], [0.0, 1.0]]))
    assert np.linalg.det(scaled) == pytest.approx(1.0)
    assert lf.isotropic_shapes(3).shape == (3, 2, 2)
    with pytest.raises(ValueError):
        lf.ridge_shape([0.0, 0.0])
    with pytest.raises(ValueError):
        lf.normalise_shape(np.zeros((2, 2)))


def test_concat_and_subset_carry_every_field() -> None:
    observations = _mixed_observations(seed=36, n_clicks=6, n_windows=4)
    exempt = np.zeros(observations.n, dtype=bool)
    exempt[2] = True
    flagged = replace(observations, em_exempt=exempt)
    subset = flagged.subset(np.array([2, 7, 9]))
    assert subset.obs_class == ("click", "window", "window")
    assert subset.em_exempt.tolist() == [True, False, False]  # type: ignore[union-attr]
    again = subset.concat(subset)
    assert again.n == 6 and again.em_exempt.tolist()[0:4] == [True, False, False, True]  # type: ignore[union-attr]


def test_robust_affine_weights_expose_a_gross_outlier() -> None:
    observations, kind = _clean_phantom_observations(seed=37, n_gross=1)
    weights = lf.robust_affine_weights(
        observations, nu=4.0, noise_guess_um={"click": 30.0}
    )
    assert weights[kind.index("gross")] < 0.05


# ---------------------------------------------------------------------- validation
def _clicks(count: int = 8) -> lf.FieldObservations:
    rng = np.random.default_rng(40)
    reference = rng.uniform(0.0, 10000.0, size=(count, 2))
    return lf.FieldObservations.clicks(
        reference, reference + 50.0, np.arange(1, count + 1)
    )


def _fit_or_build(observations: lf.FieldObservations) -> None:
    lf.fit_landmark_field(observations)


def test_validation_rejects_non_finite_positions() -> None:
    clicks = _clicks()
    moving = clicks.moving_xy_um.copy()
    moving[3, 1] = np.nan
    bad = replace(clicks, moving_xy_um=moving)
    with pytest.raises(ValueError, match=r"observation 3.*not finite"):
        _fit_or_build(bad)


def test_validation_rejects_wrong_shapes() -> None:
    clicks = _clicks()
    with pytest.raises(ValueError, match="shape_cov"):
        replace(clicks, shape_cov=np.zeros((7, 2, 2)))
    with pytest.raises(ValueError, match="reference_xy_um"):
        replace(clicks, reference_xy_um=np.zeros((8, 3)))


def test_validation_rejects_determinant_other_than_one() -> None:
    clicks = _clicks()
    shapes = clicks.shape_cov.copy()
    shapes[2] = 2.0 * np.eye(2)
    with pytest.raises(ValueError, match=r"observation 2.*determinant"):
        _fit_or_build(replace(clicks, shape_cov=shapes))


def test_validation_rejects_anisotropy_above_sixteen() -> None:
    clicks = _clicks()
    shapes = clicks.shape_cov.copy()
    shapes[5] = np.diag([5.0, 0.2])
    with pytest.raises(ValueError, match=r"observation 5.*anisotropy"):
        _fit_or_build(replace(clicks, shape_cov=shapes))


def test_validation_rejects_asymmetric_or_negative_fixed_covariance() -> None:
    clicks = _clicks()
    fixed = clicks.fixed_cov_um2.copy()
    fixed[1] = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"observation 1.*symmetric"):
        _fit_or_build(replace(clicks, fixed_cov_um2=fixed))
    fixed = clicks.fixed_cov_um2.copy()
    fixed[4] = np.diag([-5.0, 3.0])
    with pytest.raises(ValueError, match=r"observation 4.*semi-definite"):
        _fit_or_build(replace(clicks, fixed_cov_um2=fixed))


def test_validation_rejects_unknown_class_few_observations_and_collinear_points() -> (
    None
):
    clicks = _clicks()
    classes = list(clicks.obs_class)
    classes[6] = "loftr"
    with pytest.raises(ValueError, match=r"observation 6.*loftr"):
        _fit_or_build(replace(clicks, obs_class=tuple(classes)))
    with pytest.raises(ValueError, match="at least 4"):
        _fit_or_build(clicks.subset(np.array([0, 1, 2])))
    line = np.column_stack([np.linspace(0.0, 9000.0, 8), np.linspace(0.0, 9000.0, 8)])
    collinear = lf.FieldObservations.clicks(line, line + 20.0, np.arange(1, 9))
    with pytest.raises(ValueError, match="collinear"):
        _fit_or_build(collinear)


def test_bad_weights_raise_value_error() -> None:
    clicks = _clicks()
    with pytest.raises(ValueError, match="weight"):
        lf.landmark_field_from_hyperparameters(
            clicks, _hyper(short_length=None, short_signal=None), np.zeros(clicks.n)
        )
