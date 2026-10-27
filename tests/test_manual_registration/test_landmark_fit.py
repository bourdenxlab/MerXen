"""Tests for the similarity fits, leave-one-out residuals and robust trimming."""

from __future__ import annotations

import numpy as np
import pytest

from merxen.manual_registration.geometry import (
    apply_matrix_xy,
    coarse_matrix_xy,
    rotation_matrix_xy,
    translation_xy,
    wrap_deg,
)
from merxen.manual_registration.landmark_fit import (
    RobustFitOptions,
    RobustSimilarityFit,
    fit_similarity,
    leave_one_out_residuals,
    robust_similarity,
    spread_ratio,
    umeyama_similarity,
)
from merxen.manual_registration.schema import CoarseTransform

P = 0.34511  # moving and reference pixel size (um/px)
TRUE = coarse_matrix_xy(
    CoarseTransform(rotation_deg=179, offset_um=(150, -80)),
    moving_shape_yx=(48000, 72000),
    reference_shape_yx=(61000, 80000),
    moving_pixel_size_um=P,
    reference_pixel_size_um=P,
)
MOVING = np.array(
    [
        (5000, 4000),
        (30000, 6000),
        (65000, 9000),
        (12000, 24000),
        (40000, 22000),
        (68000, 30000),
        (8000, 42000),
        (36000, 44000),
        (60000, 40000),
        (20000, 12000),
    ],
    dtype=float,
)
REFERENCE = np.array(
    [
        (71081.083, 50804.36),
        (46116.596, 48370.654),
        (11176.784, 44760.977),
        (64428.598, 30684.939),
        (36517.762, 32197.367),
        (8542.641, 23707.218),
        (68745.332, 12759.59),
        (40781.701, 10270.827),
        (16716.747, 13849.06),
        (56219.587, 42546.592),
    ]
)
NOISE = np.array(
    [
        (1.2, -0.8),
        (-2.0, 1.5),
        (0.5, 2.2),
        (-1.4, -1.1),
        (2.5, 0.3),
        (-0.6, -2.4),
        (1.8, 1.0),
        (-1.0, 0.6),
        (0.2, -1.7),
        (-2.2, 2.0),
    ]
)


def _with_offsets(offsets_um: dict[int, tuple[float, float]]) -> np.ndarray:
    reference = REFERENCE.copy()
    for index, (dx, dy) in offsets_um.items():
        reference[index] += np.array([dx, dy]) / P
    return reference


def test_appendix_reference_points_are_reproducible() -> None:
    noise = NOISE.copy()
    noise[4, 0] += 40.0 / P
    expected = np.round(apply_matrix_xy(TRUE, MOVING) + noise, 3)
    np.testing.assert_allclose(expected, REFERENCE, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        TRUE[:2],
        [[-0.999848, 0.017452, 76009.3121], [-0.017452, -0.999848, 54891.812518]],
        atol=2e-6,
        rtol=0,
    )


def test_leave_one_out_flags_only_the_displaced_pair() -> None:
    loo = leave_one_out_residuals(MOVING, REFERENCE, reflection=False) * P
    np.testing.assert_allclose(
        loo,
        [
            3.8451,
            5.459,
            6.4563,
            4.6403,
            41.0328,
            7.0818,
            3.4237,
            5.1563,
            6.1226,
            5.2289,
        ],
        atol=1e-3,
    )
    flagged = (loo > 10.0) & (loo > 3 * np.median(loo))
    assert np.flatnonzero(flagged).tolist() == [4]


def test_robust_similarity_with_default_options() -> None:
    result = robust_similarity(
        MOVING,
        REFERENCE,
        reflection=False,
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
        options=RobustFitOptions(),
    )
    assert np.flatnonzero(~result.used).tolist() == [4]
    assert np.flatnonzero(result.outlier).tolist() == [4]
    np.testing.assert_allclose(
        result.fit.matrix[:2, :2],
        [[-0.999861, 0.017468], [-0.017468, -0.999861]],
        atol=1e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.fit.matrix[:2, 2], [76009.005205, 54892.785195], atol=1e-5, rtol=0
    )
    assert result.fit.rotation_deg == pytest.approx(178.99913, abs=1e-5)
    assert result.fit.scale == pytest.approx(1.00001339, abs=1e-8)
    assert result.fit.reflected is False
    assert result.scale_ratio == pytest.approx(1.00001339, abs=1e-8)
    np.testing.assert_allclose(
        result.residuals_um,
        [
            0.7702,
            0.6055,
            0.9654,
            0.7053,
            41.0328,
            0.668,
            0.5918,
            0.4061,
            0.4827,
            0.8113,
        ],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        result.loo_um,
        [1.0707, 0.7173, 1.3361, 0.8578, 41.0328, 0.9301, 0.7922, 0.49, 0.6404, 0.9614],
        atol=1e-3,
    )
    assert result.chosen_class_median_um == pytest.approx(0.6680, abs=1e-3)
    assert result.other_class_median_um == pytest.approx(8278.383, abs=1e-2)
    assert len(result.fit.residuals_px) == 9


def test_all_pairs_fits_in_both_reflection_classes() -> None:
    proper = fit_similarity(MOVING, REFERENCE, reflection=False)
    mirrored = fit_similarity(MOVING, REFERENCE, reflection=True)
    assert np.median(proper.residuals_px) * P == pytest.approx(4.5655, abs=1e-3)
    assert np.median(mirrored.residuals_px) * P == pytest.approx(8131.7391, abs=1e-3)


def test_without_trimming_all_pairs_are_used() -> None:
    options = RobustFitOptions(trim_outliers=False)
    result = robust_similarity(
        MOVING,
        REFERENCE,
        reflection=False,
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
        options=options,
    )
    first_pass = leave_one_out_residuals(MOVING, REFERENCE, reflection=False) * P
    assert result.used.all()
    assert np.flatnonzero(result.outlier).tolist() == [4]
    np.testing.assert_allclose(result.loo_um, first_pass, atol=1e-9)


def test_trimming_stops_at_the_maximum_outlier_fraction() -> None:
    reference = _with_offsets({1: (0.0, 60.0), 7: (0.0, 80.0)})
    result = robust_similarity(
        MOVING,
        reference,
        reflection=False,
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
    )
    # max_trim = floor(0.25 * 10) = 2: pair 7 then pair 1, so pair 4 stays.
    assert set(np.flatnonzero(~result.used).tolist()) == {1, 7}
    assert result.used[4]
    assert set(np.flatnonzero(result.outlier).tolist()) == {1, 7}
    assert result.residuals_um[4] == pytest.approx(35.624, abs=1e-2)


def test_trimming_stops_at_the_minimum_used_pairs() -> None:
    reference = _with_offsets({7: (0.0, 80.0)})
    limited = robust_similarity(
        MOVING,
        reference,
        reflection=False,
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
        options=RobustFitOptions(min_used_pairs=9),
    )
    assert np.flatnonzero(~limited.used).tolist() == [7]
    assert limited.used[4]
    assert limited.residuals_um[4] == pytest.approx(36.2314, abs=1e-2)
    default = robust_similarity(
        MOVING,
        reference,
        reflection=False,
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
    )
    assert set(np.flatnonzero(~default.used).tolist()) == {4, 7}


@pytest.mark.parametrize("theta", [0, 73, 180, -125])
@pytest.mark.parametrize("scale", [0.6, 1.0, 1.6])
@pytest.mark.parametrize("reflection", [False, True])
def test_umeyama_recovers_the_truth(
    theta: float, scale: float, reflection: bool
) -> None:
    index = [0, 73, 180, -125].index(theta) * 6 + [0.6, 1.0, 1.6].index(scale) * 2
    rng = np.random.default_rng(1000 + index + int(reflection))
    truth = (
        translation_xy(1500, -700)
        @ rotation_matrix_xy(theta)
        @ np.diag([-1.0 if reflection else 1.0, 1.0, 1.0])
        @ np.diag([scale, scale, 1.0])
    )
    src = rng.uniform(0, 40000, size=(20, 2))
    dst = apply_matrix_xy(truth, src)
    exact = umeyama_similarity(src, dst, reflection=reflection)
    np.testing.assert_allclose(exact, truth, rtol=1e-9, atol=1e-6)

    noisy = dst + rng.normal(0.0, 0.5, size=dst.shape)
    fit = fit_similarity(src, noisy, reflection=reflection)
    other = fit_similarity(src, noisy, reflection=not reflection)
    assert abs(wrap_deg(fit.rotation_deg - theta)) < 0.01
    assert abs(fit.scale / scale - 1) < 1e-4
    assert fit.reflected == reflection
    assert np.median(fit.residuals_px) < 1.5
    assert np.median(other.residuals_px) > 50 * np.median(fit.residuals_px)


def test_fit_errors() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        umeyama_similarity([[0, 0]], [[1, 1]], reflection=False)
    with pytest.raises(ValueError, match="identical"):
        umeyama_similarity(
            [[1, 1], [1, 1], [1, 1]], [[0, 0], [1, 1], [2, 2]], reflection=False
        )
    with pytest.raises(ValueError, match="same length"):
        fit_similarity([[0, 0], [1, 1]], [[0, 0]], reflection=False)
    with pytest.raises(ValueError, match="finite"):
        fit_similarity([[0, 0], [1, np.nan]], [[0, 0], [1, 1]], reflection=False)
    with pytest.raises(ValueError):
        robust_similarity(
            [[0, 0]],
            [[0, 0]],
            reflection=False,
            moving_pixel_size_um=1.0,
            reference_pixel_size_um=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        robust_similarity(
            MOVING,
            REFERENCE,
            reflection=False,
            moving_pixel_size_um=0.0,
            reference_pixel_size_um=1.0,
        )


def test_leave_one_out_with_two_pairs_is_nan() -> None:
    loo = leave_one_out_residuals([[0, 0], [5, 1]], [[1, 1], [6, 2]], reflection=False)
    assert loo.shape == (2,)
    assert np.isnan(loo).all()


def test_spread_ratio() -> None:
    assert spread_ratio([(0, 0), (1, 0), (0, 1), (1, 1)]) == pytest.approx(1.0)
    assert spread_ratio([(0, 0), (4, 0), (0, 1), (4, 1)]) == pytest.approx(0.25)
    assert spread_ratio([(0, 0), (1, 1), (2, 2)]) < 1e-12
    assert spread_ratio([(3, 4)]) == 0.0


def test_nan_loo_is_ignored_by_the_median_and_the_flagging() -> None:
    """A NaN leave-one-out (degenerate sub-fit) must not stop trimming."""
    result = _nan_loo_case(trim_outliers=True)
    assert np.flatnonzero(~result.used).tolist() == [5]
    assert np.flatnonzero(result.outlier).tolist() == [5]
    assert np.isnan(result.loo_um[6])
    untrimmed = _nan_loo_case(trim_outliers=False)
    assert untrimmed.used.all()
    assert np.flatnonzero(untrimmed.outlier).tolist() == [5]


def _nan_loo_case(*, trim_outliers: bool) -> RobustSimilarityFit:
    moving = np.array([(1000.0, 1000.0)] * 6 + [(9000.0, 4000.0)])
    reference = apply_matrix_xy(
        translation_xy(500, -300) @ rotation_matrix_xy(30), moving
    )
    noise_px = np.array(
        [(0.4, -0.2), (-0.3, 0.5), (0.1, 0.2), (-0.2, -0.4), (0.3, 0.1)]
    )
    reference[:5] += noise_px
    reference[5, 0] += 200.0
    return robust_similarity(
        moving,
        reference,
        reflection=False,
        moving_pixel_size_um=0.5,
        reference_pixel_size_um=0.5,
        options=RobustFitOptions(trim_outliers=trim_outliers),
    )
