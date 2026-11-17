"""Synthetic pairs, landmark simulation and labelling injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import threshold_otsu

from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration.geometry import coarse_matrix_xy
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    labelling_content_sha256,
    pair_colour,
    parse_labelling_attrs,
    read_root_attrs,
    write_root_attr,
)
from merxen.manual_registration.simulate import (
    SyntheticPair,
    inject_labelling,
    make_field_phantom,
    make_synthetic_pair,
    sample_tissue_points,
    simulate_landmark_pairs,
)

from ._mr_helpers import fake_store, make_pairs, store_attr_dict

SHAPE = (512, 512)


def _dapi(path: Path) -> np.ndarray:
    info = read_ome_tiff_info(path)
    return open_ome_tiff_level(info, 0).values[0].astype(np.float64)


def _truth_pearson(pair: SyntheticPair, *, smooth_um: float = 0.0) -> float:
    reference = _dapi(pair.reference_path)
    moving = _dapi(pair.moving_path)
    inverse = np.linalg.inv(pair.true_matrix)
    height, width = pair.reference_shape_yx
    rows, cols = np.mgrid[0:height:4, 0:width:4]
    points = np.column_stack([cols.ravel(), rows.ravel()]).astype(float)
    source = points @ inverse[:2, :2].T + inverse[:2, 2]
    if smooth_um > 0:
        reference = gaussian_filter(reference, smooth_um / pair.reference_pixel_size_um)
        moving = gaussian_filter(moving, smooth_um / pair.moving_pixel_size_um)
    sampled = map_coordinates(moving, [source[:, 1], source[:, 0]], order=1, cval=0)
    expected = reference[rows.ravel(), cols.ravel()]
    tissue = reference[rows.ravel(), cols.ravel()] > 100
    return float(np.corrcoef(expected[tissue], sampled[tissue])[0, 1])


@pytest.fixture(scope="module")
def plain_pair(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPair:
    return make_synthetic_pair(
        tmp_path_factory.mktemp("plain"), reference_shape_yx=SHAPE, warp_amplitude_um=0
    )


def test_files_and_ground_truth(plain_pair: SyntheticPair) -> None:
    reference = read_ome_tiff_info(plain_pair.reference_path)
    moving = read_ome_tiff_info(plain_pair.moving_path)
    assert reference.pixel_size_um == (2.0, 2.0)
    assert moving.pixel_size_um == (1.6, 1.6)
    assert reference.channel_names == moving.channel_names == ("DAPI", "AF568")
    assert len(reference.level_shapes_cyx) >= 2 and len(moving.level_shapes_cyx) >= 2
    assert reference.dtype == "uint16"
    assert plain_pair.moving_shape_yx == (640, 640)
    assert plain_pair.reference_shape_yx == SHAPE
    expected = coarse_matrix_xy(
        CoarseTransform(rotation_deg=178.0, mirror_x=False, offset_um=(0.0, 0.0)),
        moving_shape_yx=(640, 640),
        reference_shape_yx=SHAPE,
        moving_pixel_size_um=1.6,
        reference_pixel_size_um=2.0,
    )
    np.testing.assert_allclose(plain_pair.true_matrix, expected, atol=1e-12)


def test_rendering_follows_the_true_matrix(
    plain_pair: SyntheticPair, tmp_path: Path
) -> None:
    assert _truth_pearson(plain_pair) > 0.9
    mirrored = make_synthetic_pair(
        tmp_path, reference_shape_yx=SHAPE, warp_amplitude_um=0, mirror_x=True
    )
    assert np.linalg.det(mirrored.true_matrix[:2, :2]) < 0
    assert _truth_pearson(mirrored) > 0.9


def test_independent_nuclei_share_only_the_low_frequency_structure(
    tmp_path: Path,
) -> None:
    pair = make_synthetic_pair(
        tmp_path,
        reference_shape_yx=SHAPE,
        warp_amplitude_um=0,
        independent_nuclei=True,
    )
    assert _truth_pearson(pair) < 0.6
    assert _truth_pearson(pair, smooth_um=40.0) > 0.7


def test_pair_is_deterministic_in_the_seed(tmp_path: Path) -> None:
    first = make_synthetic_pair(tmp_path / "a", reference_shape_yx=SHAPE, seed=3)
    second = make_synthetic_pair(tmp_path / "b", reference_shape_yx=SHAPE, seed=3)
    other = make_synthetic_pair(tmp_path / "c", reference_shape_yx=SHAPE, seed=4)
    assert np.array_equal(_dapi(first.moving_path), _dapi(second.moving_path))
    assert not np.array_equal(_dapi(first.moving_path), _dapi(other.moving_path))


def test_tissue_points_are_inside_separated_and_deterministic(
    plain_pair: SyntheticPair,
) -> None:
    dapi = _dapi(plain_pair.reference_path)
    to_level0 = np.eye(3)
    points = sample_tissue_points(dapi, to_level0, n=16, seed=1, min_separation_px=40.0)
    assert points.shape == (16, 2)
    columns = points[:, 0].round().astype(int)
    rows = points[:, 1].round().astype(int)
    smoothed = gaussian_filter(np.log1p(dapi), 2)
    assert (smoothed[rows, columns] > threshold_otsu(smoothed)).all()
    distances = np.linalg.norm(points[:, None] - points[None], axis=2)
    assert distances[np.triu_indices(16, 1)].min() >= 40.0
    again = sample_tissue_points(dapi, to_level0, n=16, seed=1, min_separation_px=40.0)
    other = sample_tissue_points(dapi, to_level0, n=16, seed=2, min_separation_px=40.0)
    assert np.array_equal(points, again)
    assert not np.array_equal(points, other)
    with pytest.raises(ValueError, match="tissue points"):
        sample_tissue_points(dapi, to_level0, n=16, seed=1, min_separation_px=1000.0)


def test_simulated_pairs_follow_the_map() -> None:
    moving = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])

    def to_reference(points: np.ndarray) -> np.ndarray:
        return points * 2.0 + 1.0

    pairs = simulate_landmark_pairs(
        moving,
        to_reference,
        noise_um=0.0,
        reference_pixel_size_um=0.5,
        seed=0,
        outliers={3: (30.0, 0.0)},
        disabled=(4,),
    )
    plain = to_reference(moving)
    assert [p.id for p in pairs] == [1, 2, 3, 4]
    np.testing.assert_allclose(pairs[0].reference_xy_px, plain[0], atol=5e-4)
    assert pairs[2].reference_xy_px[0] == pytest.approx(plain[2, 0] + 60.0, abs=5e-4)
    assert pairs[2].reference_xy_px[1] == pytest.approx(plain[2, 1], abs=5e-4)
    assert [p.enabled for p in pairs] == [True, True, True, False]
    assert [p.colour for p in pairs] == [pair_colour(i, 0) for i in (1, 2, 3, 4)]
    assert pairs[1].moving_xy_px == (30.0, 40.0)


def test_simulated_noise_is_deterministic() -> None:
    moving = np.arange(20.0).reshape(10, 2)
    kwargs = {"noise_um": 1.0, "reference_pixel_size_um": 0.5}
    first = simulate_landmark_pairs(moving, lambda p: p, seed=5, **kwargs)
    second = simulate_landmark_pairs(moving, lambda p: p, seed=5, **kwargs)
    other = simulate_landmark_pairs(moving, lambda p: p, seed=6, **kwargs)
    assert first == second
    assert first != other


def _prepared(directory: Path, kind: str = "prepared") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    attrs = store_attr_dict((3000, 4000), (2800, 3600), 0.5, 0.4, kind=kind)
    return fake_store(
        directory / "prepared.zarr", {STORE_ATTR: attrs, "other": {"k": 1}}
    )


def test_inject_labelling_increments_the_revision_and_validates(
    tmp_path: Path,
) -> None:
    store = _prepared(tmp_path / "s")
    moving = [(100.0 * i + 50, 80.0 * i + 30) for i in range(10)]
    reference = [(x * 0.8 + 10, y * 0.8 + 20) for x, y in moving]
    pairs = make_pairs(moving, reference)
    assert inject_labelling(store, pairs, crops={"reference": None}) == 1
    assert inject_labelling(store, pairs, status="draft") == 2
    attrs = read_root_attrs(store)
    payload = attrs[LABELLING_ATTR]
    parsed = parse_labelling_attrs(payload)
    assert parsed.revision == 2 and parsed.status == "draft"
    assert payload["content_sha256"] == labelling_content_sha256(payload)
    assert attrs["other"] == {"k": 1}
    assert payload["landmarks"]["next_id"] == 11
    assert payload["images"]["moving"]["shape_yx"] == [2800, 3600]
    with pytest.raises(ValueError, match="status"):
        inject_labelling(store, pairs, status="bogus")


def test_inject_labelling_recovers_the_coarse_transform(tmp_path: Path) -> None:
    true = CoarseTransform(rotation_deg=37.5, mirror_x=True, offset_um=(25.0, -10.0))
    matrix = coarse_matrix_xy(
        true,
        moving_shape_yx=(2800, 3600),
        reference_shape_yx=(3000, 4000),
        moving_pixel_size_um=0.4,
        reference_pixel_size_um=0.5,
    )
    rng = np.random.default_rng(0)
    moving = rng.uniform(100, 2500, (12, 2))
    reference = moving @ matrix[:2, :2].T + matrix[:2, 2]
    store = _prepared(tmp_path / "s")
    inject_labelling(store, make_pairs(moving, reference))
    coarse = parse_labelling_attrs(read_root_attrs(store)[LABELLING_ATTR]).coarse
    assert coarse.mirror_x is True
    assert coarse.rotation_deg == pytest.approx(37.5, abs=0.01)
    assert coarse.offset_um == pytest.approx((25.0, -10.0), abs=0.01)


def test_inject_labelling_needs_a_prepared_store_and_stores_crops(
    tmp_path: Path,
) -> None:
    result_store = _prepared(tmp_path / "result", kind="result")
    with pytest.raises(ValueError, match="prepared"):
        inject_labelling(result_store, make_pairs([(1.0, 1.0)], [(1.0, 1.0)]))
    store = _prepared(tmp_path / "prepared")
    crop = {"moving": CropRect(x0=0, y0=0, x1=64, y1=64)}
    pairs = make_pairs([(1.0, 1.0), (5.0, 1.0), (1.0, 5.0)], [(2.0, 2.0)] * 3)
    inject_labelling(store, pairs, coarse=CoarseTransform(), crops=crop)
    payload = read_root_attrs(store)[LABELLING_ATTR]
    assert payload["crops"]["moving"] == {"x0": 0, "y0": 0, "x1": 64, "y1": 64}
    assert payload["crops"]["reference"] is None


def test_inject_labelling_never_lowers_next_id_and_keeps_the_colour_seed(
    tmp_path: Path,
) -> None:
    store = _prepared(tmp_path / "s")
    moving = [(100.0 * i + 50, 80.0 * i + 30) for i in range(10)]
    reference = [(x * 0.8 + 10, y * 0.8 + 20) for x, y in moving]
    pairs = make_pairs(moving, reference)
    inject_labelling(store, pairs)
    payload = read_root_attrs(store)[LABELLING_ATTR]
    assert payload["landmarks"]["next_id"] == 11
    assert payload["landmarks"]["colour_seed"] == 0
    # The viewer's counter moved on (ids 11..24 were placed and deleted).
    payload["landmarks"]["next_id"] = 25
    payload["landmarks"]["colour_seed"] = 3
    payload["content_sha256"] = labelling_content_sha256(payload)
    with spatialdata_write_lock(store):
        write_root_attr(store, LABELLING_ATTR, payload)

    inject_labelling(store, pairs[:5])  # fewer pairs: next_id must not drop to 6
    landmarks = read_root_attrs(store)[LABELLING_ATTR]["landmarks"]
    assert landmarks["next_id"] == 25
    assert landmarks["colour_seed"] == 3
    assert len(landmarks["pairs"]) == 5

    many = make_pairs(moving * 4, reference * 4)  # ids up to 40: above the counter
    inject_labelling(store, many)
    assert read_root_attrs(store)[LABELLING_ATTR]["landmarks"]["next_id"] == 41


# ---------------------------------------------------------------- field phantom
def _inside_ellipse(points: np.ndarray, extent: tuple[float, float]) -> np.ndarray:
    centre = np.array(extent) / 2.0
    semi = np.array([0.48 * extent[0], 0.45 * extent[1]])
    return np.asarray(np.sum(((points - centre) / semi) ** 2, axis=1) <= 1.0 + 1e-9)


def test_field_phantom_is_deterministic_per_seed() -> None:
    first = make_field_phantom(seed=3, n_windows=20)
    second = make_field_phantom(seed=3, n_windows=20)
    other = make_field_phantom(seed=4, n_windows=20)
    np.testing.assert_array_equal(first.moving_xy_um, second.moving_xy_um)
    np.testing.assert_array_equal(first.window_moving_xy_um, second.window_moving_xy_um)
    assert first.kind == second.kind
    assert not np.array_equal(first.reference_xy_um, other.reference_xy_um)


def test_field_phantom_kind_counts_and_ridge_directions() -> None:
    phantom = make_field_phantom(
        seed=5, n_clicks=66, n_gross=3, n_medium=5, ridge_fraction=0.5
    )
    kinds = list(phantom.kind)
    assert kinds.count("gross") == 3 and kinds.count("medium") == 5
    assert kinds.count("ridge") == 29 and kinds.count("clean") == 29
    norms = np.linalg.norm(phantom.ridge_direction_xy, axis=1)
    np.testing.assert_allclose(norms[np.array(kinds) == "ridge"], 1.0)
    assert np.all(norms[np.array(kinds) != "ridge"] == 0.0)
    gross = np.array(kinds) == "gross"
    displacement = np.linalg.norm(
        phantom.moving_xy_um - phantom.true_moving_xy_um, axis=1
    )
    np.testing.assert_allclose(displacement[gross], 1000.0)


def test_field_phantom_respects_minimum_click_spacing_and_tissue() -> None:
    phantom = make_field_phantom(seed=6, min_click_spacing_um=400.0)
    points = phantom.reference_xy_um
    distance = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    assert distance.min() >= 400.0
    assert _inside_ellipse(points, (22000.0, 16000.0)).all()
    assert _inside_ellipse(phantom.grid_xy_um, (22000.0, 16000.0)).all()
    np.testing.assert_allclose(
        phantom.true_field(phantom.grid_xy_um), phantom.true_grid_moving_xy_um
    )


def test_field_phantom_random_field_variance_matches_the_kernel() -> None:
    # Residual of the truth after removing its affine part has variance
    # s_L^2 + s_S^2 per axis; averaged over seeds and points it is within 10 %.
    variances = []
    for seed in range(40):
        phantom = make_field_phantom(
            seed=seed,
            n_clicks=4,
            n_gross=0,
            n_medium=0,
            rotation_deg=0.0,
            scale_xy=(1.0, 1.0),
            translation_um=(0.0, 0.0),
            long_length_um=2000.0,
            long_signal_um=100.0,
            short_length_um=500.0,
            short_signal_um=60.0,
            grid_step_um=2000.0,
        )
        residual = phantom.true_grid_moving_xy_um - phantom.grid_xy_um
        variances.append(float(np.mean(residual**2)))
    expected = 100.0**2 + 60.0**2
    assert abs(np.mean(variances) / expected - 1.0) < 0.1


def test_field_phantom_single_scale_truth_when_short_signal_is_zero() -> None:
    single = make_field_phantom(
        seed=7, short_signal_um=0.0, n_clicks=5, n_gross=0, n_medium=0
    )
    smooth = make_field_phantom(
        seed=7,
        short_signal_um=0.0,
        short_length_um=50.0,
        n_clicks=5,
        n_gross=0,
        n_medium=0,
    )
    # With no short component its length scale cannot matter.
    np.testing.assert_allclose(
        single.true_grid_moving_xy_um, smooth.true_grid_moving_xy_um
    )


def test_field_phantom_window_peak_covariances_are_valid() -> None:
    phantom = make_field_phantom(seed=8, n_windows=60, window_wrong_fraction=0.05)
    peak = phantom.window_peak_cov_um2
    assert peak.shape == (60, 2, 2)
    np.testing.assert_allclose(peak, np.swapaxes(peak, 1, 2))
    stds = np.sqrt(np.linalg.eigvalsh(peak))
    assert np.all(stds[:, 1] >= 5.0 - 1e-9) and np.all(stds[:, 1] <= 30.0 + 1e-9)
    assert np.all(stds[:, 0] >= 4.0 - 1e-9) and np.all(stds[:, 0] <= 9.0 + 1e-9)
    assert int(phantom.window_wrong.sum()) == 3
    assert _inside_ellipse(phantom.window_reference_xy_um, (22000.0, 16000.0)).all()
    with pytest.raises(ValueError, match="n_windows"):
        make_field_phantom(seed=8, n_windows=5000)
    with pytest.raises(ValueError, match="n_gross"):
        make_field_phantom(seed=8, n_clicks=5, n_gross=4, n_medium=4)
