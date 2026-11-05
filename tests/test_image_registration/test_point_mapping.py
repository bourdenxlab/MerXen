"""Tests for pulling and pushing points through a final registration."""

from __future__ import annotations

import numpy as np
import pytest

from merxen.image_registration import (
    _nonrigid_pull_function,
    pull_points_through_registration,
    push_points_through_registration,
)

_Y, _X = np.mgrid[0:200, 0:260]
FLOW = np.stack([3.0 * np.sin(_X / 40.0), 2.5 * np.cos(_Y / 35.0)]).astype(np.float32)
WORKING_TO_REFERENCE = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
FINAL_MATRIX = np.array([[1.2, 0.3, 50.0], [-0.3, 1.2, 80.0], [0.0, 0.0, 1.0]])


def _reference_points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack(
        [rng.uniform(100.0, 900.0, 20), rng.uniform(100.0, 700.0, 20)]
    )


def _moving_points() -> np.ndarray:
    reference = _reference_points()
    homogeneous = np.column_stack([reference, np.ones(len(reference))])
    return (np.linalg.inv(FINAL_MATRIX) @ homogeneous.T).T[:, :2]


def _affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def _pull(points: np.ndarray) -> np.ndarray:
    return pull_points_through_registration(
        points,
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
    )


def test_pull_matches_the_warp_pull_function() -> None:
    reference = _reference_points()

    pulled = _pull(reference)
    affine_only = pull_points_through_registration(reference, final_matrix=FINAL_MATRIX)

    expected = _nonrigid_pull_function(FINAL_MATRIX, FLOW, WORKING_TO_REFERENCE)(
        reference
    )
    assert np.array_equal(pulled, expected)
    assert np.allclose(
        affine_only, _affine(reference, np.linalg.inv(FINAL_MATRIX)), atol=1e-12, rtol=0
    )


def test_push_inverts_pull() -> None:
    moving = _moving_points()

    pushed, converged = push_points_through_registration(
        moving,
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
    )
    tight, _ = push_points_through_registration(
        moving,
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
        tolerance_px=1e-8,
        max_iterations=200,
    )

    assert converged.all()
    assert np.abs(_pull(pushed) - moving).max() < 1e-3
    assert np.abs(_pull(tight) - moving).max() < 1e-6


def test_push_differs_from_affine_inside_the_grid() -> None:
    moving = _moving_points()

    pushed, _ = push_points_through_registration(
        moving,
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
    )

    assert np.abs(pushed - _affine(moving, FINAL_MATRIX)).max() > 1.0


def test_push_outside_the_flow_grid_is_affine() -> None:
    reference = np.array([[-100.0, -100.0], [2000.0, 50.0], [50.0, 1500.0]])
    homogeneous = np.column_stack([reference, np.ones(len(reference))])
    moving = (np.linalg.inv(FINAL_MATRIX) @ homogeneous.T).T[:, :2]

    pushed, converged = push_points_through_registration(
        moving,
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
    )

    assert np.allclose(pushed, _affine(moving, FINAL_MATRIX), atol=1e-9, rtol=0)
    assert converged.all()


def test_push_without_flow_is_the_affine() -> None:
    moving = _moving_points()

    pushed, converged = push_points_through_registration(
        moving, final_matrix=FINAL_MATRIX
    )

    assert np.allclose(pushed, _affine(moving, FINAL_MATRIX), atol=1e-12, rtol=0)
    assert converged.all()


def test_push_reports_non_convergence() -> None:
    pushed, converged = push_points_through_registration(
        _moving_points(),
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
        max_iterations=1,
        tolerance_px=1e-12,
    )

    assert not converged.all()
    assert np.isfinite(pushed).all()


def test_point_mapping_validation() -> None:
    points = _moving_points()

    with pytest.raises(ValueError, match="given together"):
        pull_points_through_registration(
            points, final_matrix=FINAL_MATRIX, flow_yx=FLOW
        )
    with pytest.raises(ValueError, match="given together"):
        push_points_through_registration(
            points,
            final_matrix=FINAL_MATRIX,
            working_to_reference=WORKING_TO_REFERENCE,
        )
    for bad in (np.zeros(3), np.array([[1.0, np.nan]])):
        with pytest.raises(ValueError, match="points"):
            pull_points_through_registration(bad, final_matrix=FINAL_MATRIX)
        with pytest.raises(ValueError, match="points"):
            push_points_through_registration(bad, final_matrix=FINAL_MATRIX)
    with pytest.raises(ValueError, match="max_iterations"):
        push_points_through_registration(
            points, final_matrix=FINAL_MATRIX, max_iterations=0
        )
    with pytest.raises(ValueError, match="tolerance_px"):
        push_points_through_registration(
            points, final_matrix=FINAL_MATRIX, tolerance_px=0.0
        )
    with pytest.raises(ValueError, match="flow_yx"):
        pull_points_through_registration(
            points,
            final_matrix=FINAL_MATRIX,
            flow_yx=FLOW[0],
            working_to_reference=WORKING_TO_REFERENCE,
        )

    empty = np.zeros((0, 2))
    pushed, converged = push_points_through_registration(
        empty,
        final_matrix=FINAL_MATRIX,
        flow_yx=FLOW,
        working_to_reference=WORKING_TO_REFERENCE,
    )
    assert pushed.shape == (0, 2)
    assert converged.shape == (0,)
    assert _pull(empty).shape == (0, 2)
