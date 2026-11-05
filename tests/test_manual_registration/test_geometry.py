"""Tests for the coarse-transform, crop and frame maths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from merxen.manual_registration.geometry import (
    SWAP,
    apply_matrix_xy,
    coarse_from_matrix,
    coarse_matrix_xy,
    crop_matrix,
    flipped_horizontal,
    flipped_vertical,
    full_matrix,
    image_centre_xy,
    napari_affine_rc,
    points_outside_crop,
    rotated,
    rotation_matrix_xy,
    shifted,
    similarity_parameters,
    snap_crop,
    translation_xy,
    wrap_deg,
)
from merxen.manual_registration.schema import CoarseTransform, CropRect


@dataclass(frozen=True)
class Case:
    name: str
    coarse: CoarseTransform
    moving: tuple[int, int]
    reference: tuple[int, int]
    p_m: float
    p_r: float
    rows: list[list[float]]

    def matrix(self) -> np.ndarray:
        return coarse_matrix_xy(
            self.coarse,
            moving_shape_yx=self.moving,
            reference_shape_yx=self.reference,
            moving_pixel_size_um=self.p_m,
            reference_pixel_size_um=self.p_r,
        )

    def kwargs(self) -> dict[str, object]:
        return {
            "moving_shape_yx": self.moving,
            "reference_shape_yx": self.reference,
            "moving_pixel_size_um": self.p_m,
            "reference_pixel_size_um": self.p_r,
        }


CASES = [
    Case(
        "identity",
        CoarseTransform(),
        (1000, 2000),
        (1000, 2000),
        0.5,
        0.5,
        [[1, 0, 0], [0, 1, 0]],
    ),
    Case(
        "rotated_and_scaled",
        CoarseTransform(rotation_deg=30, offset_um=(12.5, -40)),
        (1001, 1500),
        (2000, 3001),
        0.3434,
        0.2125,
        [
            [1.3994970525, 0.808, 105.9004885513],
            [-0.808, 1.3994970525, 717.1121796245],
        ],
    ),
    Case(
        "mirrored",
        CoarseTransform(rotation_deg=-125, mirror_x=True, offset_um=(-100, 250)),
        (48000, 72000),
        (61000, 80000),
        0.34511,
        0.34511,
        [
            [0.5735764364, -0.8191520443, 38720.5118821327],
            [-0.8191520443, -0.5735764364, 74478.5184134923],
        ],
    ),
    Case(
        "near_half_turn",
        CoarseTransform(rotation_deg=178.5, offset_um=(120, -35.5)),
        (31347, 33453),
        (51313, 47618),
        0.3434,
        0.2125,
        [
            [-1.6154462372, 0.0423019485, 50730.1612067993],
            [-0.0423019485, -1.6154462372, 51515.3724415215],
        ],
    ),
    Case(
        "hand_computed",
        CoarseTransform(rotation_deg=90, mirror_x=True),
        (3, 4),
        (5, 6),
        1.0,
        1.0,
        [[0, 1, 1.5], [1, 0, 0.5]],
    ),
]


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0, 0),
        (180, 180),
        (-180, 180),
        (190, -170),
        (-190, 170),
        (540, 180),
        (-540, 180),
        (359.5, -0.5),
        (720, 0),
    ],
)
def test_wrap_deg_table(angle: float, expected: float) -> None:
    assert wrap_deg(angle) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_wrap_deg_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        wrap_deg(bad)


def test_rotation_matrix_is_counter_clockwise_on_screen() -> None:
    mapped = apply_matrix_xy(rotation_matrix_xy(90), (1.0, 0.0))
    np.testing.assert_allclose(mapped, (0.0, -1.0), atol=1e-15)


def test_image_centre_and_validation() -> None:
    assert image_centre_xy((3, 4)) == (1.5, 1.0)
    with pytest.raises(ValueError):
        image_centre_xy((0, 4))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_coarse_matrix_matches_the_reference_values(case: Case) -> None:
    matrix = case.matrix()
    np.testing.assert_allclose(matrix[:2], case.rows, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(matrix[2], [0.0, 0.0, 1.0])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_coarse_matrix_invariants(case: Case) -> None:
    matrix = case.matrix()
    c_mov = image_centre_xy(case.moving)
    c_ref = np.array(image_centre_xy(case.reference))
    expected = c_ref + np.array(case.coarse.offset_um) / case.p_r
    np.testing.assert_allclose(apply_matrix_xy(matrix, c_mov), expected, atol=1e-9)
    det = np.linalg.det(matrix[:2, :2])
    assert np.sign(det) == (-1 if case.coarse.mirror_x else 1)
    assert abs(det) == pytest.approx((case.p_m / case.p_r) ** 2, rel=1e-12)


def test_coarse_matrix_rejects_non_positive_pixel_sizes() -> None:
    with pytest.raises(ValueError):
        coarse_matrix_xy(
            CoarseTransform(),
            moving_shape_yx=(10, 10),
            reference_shape_yx=(10, 10),
            moving_pixel_size_um=0.0,
            reference_pixel_size_um=1.0,
        )


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0, 0.0])
@pytest.mark.parametrize("which", ["moving", "reference"])
def test_pixel_sizes_must_be_finite_and_positive(which: str, bad: float) -> None:
    sizes = {"moving_pixel_size_um": 1.0, "reference_pixel_size_um": 1.0}
    sizes[f"{which}_pixel_size_um"] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        coarse_matrix_xy(
            CoarseTransform(),
            moving_shape_yx=(10, 10),
            reference_shape_yx=(10, 10),
            **sizes,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        coarse_from_matrix(
            np.eye(3),
            moving_shape_yx=(10, 10),
            reference_shape_yx=(10, 10),
            **sizes,
        )


STATE_CASE = CASES[1]  # shapes and pixel sizes of the flip/rotate identities


def _conjugated(matrix: np.ndarray, centre: np.ndarray) -> np.ndarray:
    return translation_xy(*centre) @ matrix @ translation_xy(*-centre)


@pytest.mark.parametrize("theta", [0, 30, -125, 178.5, 180])
@pytest.mark.parametrize("mirror", [False, True])
def test_flip_and_rotate_identities(theta: float, mirror: bool) -> None:
    coarse = CoarseTransform(rotation_deg=theta, mirror_x=mirror, offset_um=(12.5, -40))
    kwargs = STATE_CASE.kwargs()

    def matrix_of(state: CoarseTransform) -> np.ndarray:
        return coarse_matrix_xy(state, **kwargs)  # type: ignore[arg-type]

    base = matrix_of(coarse)
    centre = np.array(image_centre_xy(STATE_CASE.reference)) + (
        np.array(coarse.offset_um) / STATE_CASE.p_r
    )
    checks = [
        (flipped_horizontal(coarse), np.diag([-1.0, 1.0, 1.0])),
        (flipped_vertical(coarse), np.diag([1.0, -1.0, 1.0])),
        (rotated(coarse, 37), rotation_matrix_xy(37)),
        (rotated(coarse, -200), rotation_matrix_xy(-200)),
    ]
    for state, operator in checks:
        np.testing.assert_allclose(
            matrix_of(state),
            _conjugated(operator, centre) @ base,
            rtol=1e-12,
            atol=1e-6,
        )
    for flip in (flipped_horizontal, flipped_vertical):
        twice = flip(flip(coarse))
        assert wrap_deg(twice.rotation_deg - coarse.rotation_deg) == pytest.approx(
            0.0, abs=1e-9
        )
        assert twice.mirror_x == coarse.mirror_x


def test_shifted_adds_to_the_offset() -> None:
    coarse = CoarseTransform(offset_um=(1.0, 2.0))
    assert shifted(coarse, 10.0, -5.0).offset_um == (11.0, -3.0)
    assert coarse.offset_um == (1.0, 2.0)


@pytest.mark.parametrize("case", CASES[:4], ids=lambda case: case.name)
def test_coarse_from_matrix_round_trip_on_cases(case: Case) -> None:
    _assert_round_trip(case.coarse, **case.kwargs())


def test_coarse_from_matrix_round_trip_on_random_states() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        coarse = CoarseTransform(
            rotation_deg=float(180.0 - rng.uniform(0.0, 360.0)),
            mirror_x=bool(rng.integers(0, 2)),
            offset_um=(
                float(rng.uniform(-5000, 5000)),
                float(rng.uniform(-5000, 5000)),
            ),
        )
        kwargs = {
            "moving_shape_yx": (
                int(rng.integers(500, 5000)),
                int(rng.integers(500, 5000)),
            ),
            "reference_shape_yx": (
                int(rng.integers(500, 5000)),
                int(rng.integers(500, 5000)),
            ),
            "moving_pixel_size_um": float(rng.uniform(0.2, 1.0)),
            "reference_pixel_size_um": float(rng.uniform(0.2, 1.0)),
        }
        _assert_round_trip(coarse, **kwargs)  # type: ignore[arg-type]


def _assert_round_trip(coarse: CoarseTransform, **kwargs: object) -> None:
    matrix = coarse_matrix_xy(coarse, **kwargs)  # type: ignore[arg-type]
    recovered, scale_ratio = coarse_from_matrix(matrix, **kwargs)  # type: ignore[arg-type]
    assert abs(wrap_deg(recovered.rotation_deg - coarse.rotation_deg)) < 1e-9
    assert recovered.mirror_x == coarse.mirror_x
    np.testing.assert_allclose(recovered.offset_um, coarse.offset_um, atol=1e-6)
    assert scale_ratio == pytest.approx(1.0, abs=1e-12)
    scaled = matrix.copy()
    scaled[:2, :2] *= 1.01
    _, scaled_ratio = coarse_from_matrix(scaled, **kwargs)  # type: ignore[arg-type]
    assert scaled_ratio == pytest.approx(1.01, abs=1e-12)


def test_similarity_parameters_rejects_a_singular_matrix() -> None:
    with pytest.raises(ValueError, match="singular"):
        similarity_parameters(np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0, 0, 1]]))


def test_similarity_parameters_of_a_proper_rotation() -> None:
    scale, angle, reflected = similarity_parameters(rotation_matrix_xy(33.0) * 2.0)
    assert scale == pytest.approx(2.0, rel=1e-12)
    assert angle == pytest.approx(33.0, abs=1e-9)
    assert not reflected


SHAPE = (55012, 76544)


@pytest.mark.parametrize(
    ("crop", "shape", "multiple", "expected"),
    [
        (None, SHAPE, 64, (0, 0, 76544, 55012)),
        ((1000, 500, 71000, 50500), SHAPE, 64, (1024, 512, 70976, 50496)),
        ((0, 0, 76544, 55012), SHAPE, 64, (0, 0, 76544, 55012)),
        ((63, 65, 76543, 55011), SHAPE, 64, (64, 128, 76480, 54976)),
        ((10, 0, 1090, 5000), (6000, 6000), 64, (64, 0, 1088, 4992)),
        ((3, 5, 1500, 1700), (2000, 2000), 1, (3, 5, 1500, 1700)),
    ],
    ids=["none", "interior", "edge", "near_edge", "exact_minimum", "multiple_one"],
)
def test_snap_crop_snaps_inward(
    crop: tuple[int, int, int, int] | None,
    shape: tuple[int, int],
    multiple: int,
    expected: tuple[int, int, int, int],
) -> None:
    rect = (
        None
        if crop is None
        else CropRect(x0=crop[0], y0=crop[1], x1=crop[2], y1=crop[3])
    )
    assert snap_crop(rect, shape, multiple=multiple).as_tuple() == expected


def test_snap_crop_rejects_too_small_and_out_of_image_crops() -> None:
    with pytest.raises(ValueError, match="below the minimum"):
        snap_crop(CropRect(x0=65, y0=0, x1=1090, y1=5000), (6000, 6000))
    with pytest.raises(ValueError, match="exceeds the image"):
        snap_crop(CropRect(x0=0, y0=0, x1=6001, y1=10), (6000, 6000))
    with pytest.raises(ValueError):
        snap_crop(None, (10, 10), multiple=0)


def test_crop_matrix_example_and_round_trip() -> None:
    matrix_full = CASES[2].matrix()
    o_r, o_m = (1024.0, 512.0), (64.0, 128.0)
    cropped = crop_matrix(matrix_full, reference_origin_xy=o_r, moving_origin_xy=o_m)
    np.testing.assert_allclose(
        cropped[:2],
        [
            [0.5735764364, -0.8191520443, 37628.369312],
            [-0.8191520443, -0.5735764364, 73840.674899],
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        full_matrix(cropped, reference_origin_xy=o_r, moving_origin_xy=o_m),
        matrix_full,
        atol=1e-9,
    )
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 70000, size=(10, 2))
    np.testing.assert_allclose(
        apply_matrix_xy(matrix_full, points) - np.array(o_r),
        apply_matrix_xy(cropped, points - np.array(o_m)),
        atol=1e-6,
    )


def test_crop_matrix_second_case_translation_only() -> None:
    matrix_full = CASES[1].matrix()
    cropped = crop_matrix(
        matrix_full, reference_origin_xy=(0.0, 0.0), moving_origin_xy=(64.0, 0.0)
    )
    np.testing.assert_allclose(cropped[:2, :2], matrix_full[:2, :2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        cropped[:2, 2], matrix_full[:2, 2] + matrix_full[:2, 0] * 64.0, atol=1e-9
    )


def test_points_outside_crop_boundaries() -> None:
    crop = CropRect(x0=10, y0=20, x1=30, y1=40)
    points = [
        (9.5, 20),
        (9.4999, 25),
        (29.4999, 39.4999),
        (29.5, 30),
        (20, 39.5),
        (20, 19.5),
    ]
    outside = points_outside_crop(points, crop)
    assert outside.dtype == bool
    assert outside.shape == (6,)
    assert outside.tolist() == [False, True, False, True, True, False]
    assert not points_outside_crop(points, None).any()
    assert points_outside_crop(points, None).shape == (6,)


def test_napari_affine_hand_case_and_property() -> None:
    reference = np.diag([0.5, 0.5, 1.0])
    affine = napari_affine_rc(reference, CASES[4].matrix())
    np.testing.assert_allclose(
        affine, [[0, 0.5, 0.25], [0.5, 0, 0.75], [0, 0, 1]], atol=1e-12
    )
    reference = np.diag([0.34511, 0.34511, 1.0])
    matrix = CASES[2].matrix()
    affine = napari_affine_rc(reference, matrix)
    m_x, m_y = 12345.0, 6789.0
    lhs = affine @ np.array([m_y, m_x, 1.0])
    rhs = reference @ (SWAP @ (matrix @ np.array([m_x, m_y, 1.0])))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_apply_matrix_xy_shapes_and_purity() -> None:
    matrix = CASES[1].matrix()
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    before = points.copy()
    many = apply_matrix_xy(matrix, points)
    single = apply_matrix_xy(matrix, points[0])
    assert many.shape == (2, 2)
    assert single.shape == (2,)
    np.testing.assert_allclose(many[0], single)
    np.testing.assert_array_equal(points, before)
    with pytest.raises(ValueError):
        apply_matrix_xy(np.eye(2), points)
    with pytest.raises(ValueError):
        apply_matrix_xy(matrix, np.zeros((3, 3)))
