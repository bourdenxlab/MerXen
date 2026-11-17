"""Tests of the image evidence module (channel roles, representations, checks)."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from merxen.manual_registration import image_evidence as ie
from merxen.manual_registration.config import (
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
)
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.schema import (
    STORE_ATTR,
    CropRect,
    StoreAttrs,
    parse_store_attrs,
    read_root_attrs,
)
from merxen.manual_registration.simulate import make_synthetic_pair

from ._image_phantom import (
    MOVING_ORIGIN_UM,
    Phantom,
    make_phantom,
    replace_with_texture,
    shift_patch,
)
from ._scripts import load_script

derive_script = load_script("derive_test_landmarks")


def _blob(shape: int, centre_xy: tuple[float, float], sigma: float) -> np.ndarray:
    rows, columns = np.mgrid[:shape, :shape]
    return np.asarray(
        np.exp(
            -((columns - centre_xy[0]) ** 2 + (rows - centre_xy[1]) ** 2)
            / (2 * sigma**2)
        ),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Helpers moved from the test-landmark script
# ---------------------------------------------------------------------------


def test_script_aliases_are_the_package_functions() -> None:
    assert derive_script._bandpass is ie.bandpass_log
    assert derive_script._parabolic_peak is ie.parabolic_peak
    assert derive_script._match_window is ie.match_window


def test_a_peak_on_the_search_border_is_not_confident() -> None:
    search = 20
    window = _blob(60, (30.0, 30.0), 12.0)

    inside = _blob(100, (50.0 + 10.0, 50.0), 12.0)
    peak, second, x, y, on_border = ie.match_window(inside, window, exclusion_px=8)
    assert not on_border and peak > 0.9 and second < peak
    assert x - search == pytest.approx(10.0, abs=0.1)
    assert y - search == pytest.approx(0.0, abs=0.1)

    # 35 px to the right is beyond the +-20 px search: the best correlation within
    # the search is a high one on its border, which says nothing about the match.
    beyond = _blob(100, (50.0 + 35.0, 50.0), 12.0)
    peak, _, x, _, on_border = ie.match_window(beyond, window, exclusion_px=8)
    assert on_border and peak > 0.3
    assert x == 2 * search


@pytest.mark.parametrize(("shift_x", "shift_y"), [(10.4, -3.3), (-7.25, 5.6)])
def test_match_window_recovers_a_fractional_shift(
    shift_x: float, shift_y: float
) -> None:
    search = 20
    window = _blob(60, (30.0, 30.0), 12.0)
    region = _blob(100, (50.0 + shift_x, 50.0 + shift_y), 12.0)

    peak, _, x, y, on_border = ie.match_window(region, window, exclusion_px=8)

    assert not on_border and peak > 0.9
    assert x - search == pytest.approx(shift_x, abs=0.05)
    assert y - search == pytest.approx(shift_y, abs=0.05)


def test_parabolic_peak_is_the_vertex_of_a_sampled_parabola() -> None:
    samples = 5.0 - (np.arange(8) - 3.3) ** 2

    assert ie.parabolic_peak(samples, 3) == pytest.approx(3.3)
    assert ie.parabolic_peak(samples[::-1], 4) == pytest.approx(7 - 3.3)
    assert ie.parabolic_peak(samples, 0) == 0.0
    assert ie.parabolic_peak(samples, 7) == 7.0
    assert ie.parabolic_peak(np.ones(5), 2) == 2.0


# ---------------------------------------------------------------------------
# Channel roles (contract C6)
# ---------------------------------------------------------------------------

_CHANNELS = ["DAPI", "AF568", "AF647"]


def test_parse_channel_roles_reads_unscoped_and_scoped_items() -> None:
    parsed = ie.parse_channel_roles(
        " at8 = AF568 ; map2=AF647;; reference:dapi=DAPI ;moving:dapi=DAPI2 "
    )

    assert parsed == {
        None: {"at8": "AF568", "map2": "AF647"},
        "reference": {"dapi": "DAPI"},
        "moving": {"dapi": "DAPI2"},
    }
    assert ie.parse_channel_roles("") == {}
    assert ie.parse_channel_roles(" ; ") == {}


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("at8=AF568;at8=AF647", "given twice"),
        ("reference:at8=A;reference:at8=B", "given twice"),
        ("nuclei=DAPI", "unknown role"),
        ("left:dapi=DAPI", "unknown scope"),
        ("dapi=", "empty channel"),
        ("dapi", "exactly one '='"),
        ("dapi=A=B", "exactly one '='"),
        ("dapi=A:B", "not allowed"),
        ("reference:moving:dapi=A", "more than one ':'"),
    ],
)
def test_parse_channel_roles_rejects_malformed_items(text: str, fragment: str) -> None:
    with pytest.raises(ValueError, match="^channel roles: ") as excinfo:
        ie.parse_channel_roles(text)
    assert fragment in str(excinfo.value)


def test_unscoped_roles_apply_to_both_images_and_scoped_ones_override() -> None:
    roles = ie.resolve_channel_roles(
        "at8=AF568;map2=AF647;dapi=DAPI;moving:at8=AF647;moving:map2=AF568",
        reference_channels=_CHANNELS,
        moving_channels=_CHANNELS,
    )

    assert roles.reference == {"at8": "AF568", "map2": "AF647", "dapi": "DAPI"}
    assert roles.moving == {"at8": "AF647", "map2": "AF568", "dapi": "DAPI"}
    assert roles.to_dict() == {
        "reference": {"at8": "AF568", "dapi": "DAPI", "map2": "AF647"},
        "moving": {"at8": "AF647", "dapi": "DAPI", "map2": "AF568"},
    }


def test_resolve_channel_roles_checks_names_and_required_roles() -> None:
    with pytest.raises(ValueError, match="channel roles: .*'af568'.*DAPI") as excinfo:
        ie.resolve_channel_roles(
            "at8=af568;map2=AF647;dapi=DAPI",
            reference_channels=_CHANNELS,
            moving_channels=_CHANNELS,
        )
    assert "reference image" in str(excinfo.value)
    with pytest.raises(ValueError, match="no channel for role.*'map2'.*moving"):
        ie.resolve_channel_roles(
            "at8=AF568;dapi=DAPI;reference:map2=AF647",
            reference_channels=_CHANNELS,
            moving_channels=_CHANNELS,
        )
    # A role that is not required may stay unresolved.
    roles = ie.resolve_channel_roles(
        "dapi=DAPI",
        reference_channels=_CHANNELS,
        moving_channels=_CHANNELS,
        required=("dapi",),
    )
    assert roles.reference == {"dapi": "DAPI"}


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def _small_role_images(
    *,
    gradient: float = 0.0,
    soma_amplitude: float = 0.0,
    stripes_deg: float | None = None,
    seed: int = 0,
) -> ie.RoleImages:
    """A 512 px tissue disc with smooth structure on glass, at 2.76 um/px."""
    size = 512
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    tissue = ((xx - 256) ** 2 + (yy - 256) ** 2 < 220**2).astype(np.float32)
    structure = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 6.0)
    structure = (structure / structure.std()).astype(np.float32)
    ramp = gradient * (xx - 256) / 256.0
    log_image = math.log(20.0) + (math.log(300.0) - math.log(20.0)) * tissue
    log_image = log_image + tissue * (0.5 * structure + ramp)
    blobs = np.zeros((size, size), np.float32)
    if soma_amplitude:
        for x, y in rng.integers(120, 392, (12, 2)):
            blobs[y, x] = 1.0
        blobs = ndimage.gaussian_filter(blobs, 1.2) * 30.0 * soma_amplitude
    images = {
        "dapi": np.exp(log_image + 0.02 * rng.normal(size=(size, size))),
        "at8": np.exp(log_image + tissue * blobs),
        "map2": np.exp(log_image),
    }
    if stripes_deg is not None:
        angle = math.radians(stripes_deg)
        phase = (xx * math.cos(angle) + yy * math.sin(angle)) * 2 * math.pi / 20.0
        images["map2"] = np.exp(
            math.log(20.0)
            + (math.log(300.0) - math.log(20.0)) * tissue
            + tissue * 0.6 * np.sin(phase)
        )
    return ie.RoleImages(
        images={k: v.astype(np.float32) for k, v in images.items()},
        to_level0=np.eye(3),
        pixel_size_um=2.76,
        level0_pixel_size_um=2.76,
    )


def test_band_is_insensitive_to_a_linear_illumination_gradient() -> None:
    flat = ie.compute_representation(_small_role_images())
    tilted = ie.compute_representation(_small_role_images(gradient=0.8))

    interior = ndimage.binary_erosion(flat.tissue & tilted.tissue, iterations=40)
    for role_index in range(3):
        a = flat.bands[role_index][interior]
        b = tilted.bands[role_index][interior]
        assert np.corrcoef(a, b)[0, 1] > 0.97


def test_band_has_unit_robust_sd_over_tissue() -> None:
    representation = ie.compute_representation(_small_role_images(seed=3))

    assert representation.bands.shape == (3, 512, 512)
    assert representation.bands.dtype == np.float32
    for band in representation.bands:
        values = band[representation.tissue]
        mad = np.median(np.abs(values - np.median(values)))
        assert 1.4826 * mad == pytest.approx(1.0, abs=1e-3)
    assert 0.3 < representation.tissue.mean() < 0.6


def test_soma_mask_replaces_small_bright_blobs_and_keeps_broad_structure() -> None:
    clean = ie.compute_representation(_small_role_images(seed=4))
    bright = _small_role_images(seed=4, soma_amplitude=1.0)
    masked = ie.compute_representation(bright)
    unmasked = ie.compute_representation(
        bright, ie.RepresentationParams(soma_mask=False)
    )

    interior = ndimage.binary_erosion(clean.tissue, iterations=40)
    at8 = ie.ROLES.index("at8")
    error_masked = np.abs(masked.bands[at8] - clean.bands[at8])[interior].mean()
    error_unmasked = np.abs(unmasked.bands[at8] - clean.bands[at8])[interior].mean()
    assert error_masked < 0.5 * error_unmasked
    assert (
        np.corrcoef(masked.bands[at8][interior], clean.bands[at8][interior])[0, 1] > 0.9
    )


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 75.0, 130.0])
def test_orientation_of_a_striped_pattern_matches_its_angle(angle_deg: float) -> None:
    representation = ie.compute_representation(
        _small_role_images(stripes_deg=angle_deg)
    )

    vector = representation.orientation[:, 200:312, 200:312]
    mean = vector.reshape(2, -1).mean(axis=1)
    # The vector holds the double angle of the gradient (normal to the stripes).
    measured = math.degrees(0.5 * math.atan2(mean[1], mean[0]))
    difference = (measured - angle_deg + 90.0) % 180.0 - 90.0
    assert abs(difference) < 3.0
    assert np.hypot(*mean) > 0.5  # strongly coherent
    assert not representation.orientation[:, ~representation.tissue].any()


def test_compute_representation_requires_every_role() -> None:
    images = _small_role_images()
    partial = ie.RoleImages(
        images={"dapi": images.images["dapi"]},
        to_level0=images.to_level0,
        pixel_size_um=2.76,
        level0_pixel_size_um=2.76,
    )
    with pytest.raises(ValueError, match="at8"):
        ie.compute_representation(partial)


def test_representation_converts_between_um_and_working_px() -> None:
    images = _small_role_images()
    images = ie.RoleImages(
        images=images.images,
        to_level0=np.array([[2.0, 0.0, 10.5], [0.0, 2.0, 4.5], [0.0, 0.0, 1.0]]),
        pixel_size_um=2.76,
        level0_pixel_size_um=1.38,
    )
    representation = ie.compute_representation(images)

    um = representation.px_to_um([[0.0, 0.0], [3.0, 5.0]])
    assert um == pytest.approx(np.array([[10.5, 4.5], [16.5, 14.5]]) * 1.38)
    assert representation.um_to_px(um) == pytest.approx(np.array([[0, 0], [3, 5.0]]))


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_ncc_map_with_identical_channels_equals_the_single_channel_map() -> None:
    rng = np.random.default_rng(0)
    search = ndimage.gaussian_filter(rng.normal(size=(120, 140)), 3.0).astype(
        np.float32
    )
    template = search[30:70, 40:90].copy()

    single = ie.ncc_map(template[None], search[None])
    triple = ie.ncc_map(np.stack([template] * 3), np.stack([search] * 3))

    assert single.shape == (120 - 40 + 1, 140 - 50 + 1)
    assert triple == pytest.approx(single, abs=1e-5)
    assert single[30, 40] == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="at most 4"):
        ie.ncc_map(np.zeros((5, 4, 4), np.float32), np.zeros((5, 8, 8), np.float32))


# ---------------------------------------------------------------------------
# Loading real-format role images
# ---------------------------------------------------------------------------


def _prepared_three_channel_pair(tmp_path: Path) -> tuple[StoreAttrs, Path]:
    pair = make_synthetic_pair(
        tmp_path / "images",
        reference_shape_yx=(2048, 2048),
        reference_pixel_size_um=1.38,
        moving_pixel_size_um=1.38,
        channel_names=("DAPI", "AF568", "AF647"),
        warp_amplitude_um=0.0,
    )
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name="SYN1",
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(path=pair.moving_path),
            prepared_dir=tmp_path / "prepared",
            output_dir=tmp_path / "prepare_out",
            workers=1,
        )
    )
    attrs = parse_store_attrs(read_root_attrs(outputs["store"])[STORE_ATTR])
    return attrs, outputs["store"]


@pytest.mark.integration
def test_load_role_images_reads_the_named_channels_on_the_cropped_working_grid(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_three_channel_pair(tmp_path)
    crops = {
        "reference": CropRect(x0=64, y0=128, x1=1984, y1=1920),
        "moving": CropRect(x0=128, y0=64, x1=2048, y1=1856),
    }
    roles = ie.resolve_channel_roles(
        "at8=AF568;map2=AF647;dapi=DAPI",
        reference_channels=["DAPI", "AF568", "AF647"],
        moving_channels=["DAPI", "AF568", "AF647"],
    )
    swapped = ie.resolve_channel_roles(
        "at8=AF647;map2=AF568;dapi=DAPI",
        reference_channels=["DAPI", "AF568", "AF647"],
        moving_channels=["DAPI", "AF568", "AF647"],
    )

    images = ie.load_role_images(store, roles, crops=crops)
    other = ie.load_role_images(store, swapped, crops=crops)

    for key, crop in crops.items():
        loaded = images[key]
        # 2.76 um working pixels are 2 level-0 pixels of 1.38 um.
        assert loaded.shape_yx == (crop.height() // 2, crop.width() // 2)
        assert loaded.pixel_size_um == pytest.approx(2.76)
        assert loaded.level0_pixel_size_um == pytest.approx(1.38)
        for row, column in ((0, 0), (5, 9), (loaded.shape_yx[0] - 1, 3)):
            level0 = loaded.to_level0 @ np.array([column, row, 1.0])
            expected = np.array([crop.x0 + 2 * column + 0.5, crop.y0 + 2 * row + 0.5])
            assert level0[:2] == pytest.approx(expected, abs=1e-9)
        assert set(loaded.images) == {"at8", "map2", "dapi"}
        assert all(a.dtype == np.float32 for a in loaded.images.values())
        # The roles pick the channels by name: swapping two roles swaps the images.
        assert np.array_equal(loaded.images["at8"], other[key].images["map2"])
        assert np.array_equal(loaded.images["map2"], other[key].images["at8"])
        assert np.array_equal(loaded.images["dapi"], other[key].images["dapi"])
        assert not np.array_equal(loaded.images["at8"], loaded.images["map2"])
        assert not np.array_equal(loaded.images["at8"], loaded.images["dapi"])


@pytest.mark.integration
def test_coarse_role_images_read_the_registration_channel_without_roles(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_three_channel_pair(tmp_path)
    crops = {
        "reference": CropRect(x0=0, y0=0, x1=2048, y1=2048),
        "moving": CropRect(x0=0, y0=0, x1=2048, y1=2048),
    }

    coarse = ie.coarse_role_images(store, None, crops=crops, pixel_size_um=22.0)

    for key in ("reference", "moving"):
        assert set(coarse[key].images) == {"dapi"}
        # 22 um over 1.38 um is a block of 16 level-0 pixels.
        assert coarse[key].shape_yx == (128, 128)
        assert coarse[key].pixel_size_um == pytest.approx(22.08)


# ---------------------------------------------------------------------------
# Peak shape and covariance
# ---------------------------------------------------------------------------
def _anisotropic_gaussian(
    shape: tuple[int, int],
    centre_xy: tuple[float, float],
    sigma_xy: tuple[float, float],
) -> np.ndarray:
    rows, columns = np.mgrid[: shape[0], : shape[1]]
    return np.exp(
        -((columns - centre_xy[0]) ** 2) / (2 * sigma_xy[0] ** 2)
        - (rows - centre_xy[1]) ** 2 / (2 * sigma_xy[1] ** 2)
    )


@pytest.mark.parametrize("angle_deg", [0.0, 35.0, 90.0])
def test_peak_shape_finds_the_direction_of_an_elongated_ridge(
    angle_deg: float,
) -> None:
    rows, columns = np.mgrid[:201, :201]
    angle = math.radians(angle_deg)
    along = (columns - 100) * math.cos(angle) + (rows - 100) * math.sin(angle)
    across = -(columns - 100) * math.sin(angle) + (rows - 100) * math.cos(angle)
    scores = np.exp(-(along**2) / (2 * 30.0**2) - across**2 / (2 * 5.0**2))

    shape = ie.peak_shape(scores, (100, 100), pixel_size_um=2.76)

    assert shape.elongation > 2.0
    difference = (math.degrees(shape.angle_rad) - angle_deg + 90.0) % 180.0 - 90.0
    assert abs(difference) < 5.0
    assert shape.cov_um2.shape == (2, 2)


def test_peak_shape_of_a_round_peak_is_not_elongated() -> None:
    scores = _anisotropic_gaussian((201, 201), (100, 100), (8.0, 8.0))

    assert ie.peak_shape(scores, (100, 100), pixel_size_um=2.76).elongation < 1.2
    assert (
        ie.peak_shape(np.zeros((50, 50)), (25, 25), pixel_size_um=2.76).elongation
        == 1.0
    )


def test_peak_covariance_of_an_anisotropic_peak_has_the_expected_moments() -> None:
    pixel = 2.76
    sigma_x, sigma_y, drop = 40.0, 15.0, 0.05
    scores = _anisotropic_gaussian((301, 301), (150.0, 150.0), (sigma_x, sigma_y))

    cov = ie.peak_covariance(scores, (150.0, 150.0), pixel_size_um=pixel, drop=drop)

    # With u = x / sigma_x and v = y / sigma_y the weights (G - 1 + drop)_+ are
    # radially symmetric, so cov = sigma^2 * a with a = E[u^2] of the weights.
    grid = np.linspace(-0.6, 0.6, 1201)
    uu, vv = np.meshgrid(grid, grid)
    weights = np.clip(np.exp(-(uu**2 + vv**2) / 2) - (1 - drop), 0, None)
    expected = float((weights * uu**2).sum() / weights.sum())
    assert cov[0, 0] == pytest.approx(expected * (sigma_x * pixel) ** 2, rel=0.1)
    assert cov[1, 1] == pytest.approx(expected * (sigma_y * pixel) ** 2, rel=0.1)
    assert abs(cov[0, 1]) < 0.05 * cov[0, 0]


def test_peak_covariance_of_a_flat_map_is_zero() -> None:
    assert not ie.peak_covariance(
        np.full((101, 101), 0.3), (50, 50), pixel_size_um=2.76
    ).any()
    assert not ie.peak_covariance(
        np.zeros((101, 101)), (50, 50), pixel_size_um=2.76
    ).any()


# ---------------------------------------------------------------------------
# Phantom fixtures
# ---------------------------------------------------------------------------


@functools.cache
def _phantom(
    seed: int, fine_detail: bool = True, warp_wavelength_um: float = 1500.0
) -> Phantom:
    """A phantom, built once per process (it is never modified)."""
    return make_phantom(
        seed, fine_detail=fine_detail, warp_wavelength_um=warp_wavelength_um
    )


# A window of 600-800 um averages the warp over its extent, so its displacement
# differs from the one at its centre by about A k^2 w^2 / 24: a 1.5 mm sinusoid
# would bias it by tens of um whatever the matching. The window tests use 3 mm.
_WINDOW_WAVELENGTH_UM = 3000.0


@dataclass(frozen=True, eq=False)
class _Pair:
    phantom: Phantom
    reference: ie.Representation
    moving: ie.Representation


def _build_pair(
    phantom: Phantom, pull: Callable[[np.ndarray], np.ndarray] | None = None
) -> _Pair:
    reference = ie.compute_representation(phantom.reference)
    warped = ie.warp_role_images(
        phantom.moving, reference=phantom.reference, pull_um=pull or phantom.pull
    )
    return _Pair(phantom, reference, ie.compute_representation(warped))


@pytest.fixture(scope="module")
def pair() -> _Pair:
    """The phantom warped by its exact pull (what a perfect field would give)."""
    return _build_pair(_phantom(0))


@pytest.fixture(scope="module")
def other_pair() -> _Pair:
    """An unrelated phantom, for nulls of independent images."""
    return _build_pair(_phantom(5))


@pytest.fixture(scope="module")
def click_nulls(pair: _Pair) -> dict[tuple[str, float], ie.NullModel]:
    params = ie.ClickCheckParams()
    nulls: dict[tuple[str, float], ie.NullModel] = {}
    for window_um in params.windows_um:
        built = ie._null_models(
            pair.reference,
            pair.moving,
            representations=("bp20", "orientation", "fused"),
            window_um=window_um,
            search_um=params.search_um,
            n_pairs=150,
            min_distance_um=2000.0,
            seed=0,
        )
        nulls.update({(name, window_um): model for name, model in built.items()})
    return nulls


def _deep_tissue_points(pair: _Pair, count: int, seed: int) -> np.ndarray:
    """Reference um of points far from the tissue edge."""
    interior = ndimage.binary_erosion(pair.reference.tissue, iterations=300)
    rows, columns = np.nonzero(interior)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rows), size=count, replace=False)
    return pair.reference.px_to_um(np.stack([columns[chosen], rows[chosen]], axis=1))


def test_warp_role_images_reproduces_the_reference_structure() -> None:
    phantom = _phantom(0, fine_detail=False)
    reference = ie.compute_representation(phantom.reference)
    warped_images = ie.warp_role_images(
        phantom.moving, reference=phantom.reference, pull_um=phantom.pull
    )
    warped = ie.compute_representation(warped_images)

    assert warped_images.shape_yx == phantom.reference.shape_yx
    assert np.array_equal(warped_images.to_level0, phantom.reference.to_level0)
    inside = ndimage.binary_erosion(reference.tissue & warped.tissue, iterations=60)
    for role_index in range(3):
        a = reference.bands[role_index][inside]
        b = warped.bands[role_index][inside]
        assert np.corrcoef(a, b)[0, 1] > 0.9
    # The affine prior alone leaves the sinusoid (40 um) unexplained: worse.
    prior_images = ie.warp_role_images(
        phantom.moving, reference=phantom.reference, pull_um=phantom.prior
    )
    prior = ie.compute_representation(prior_images)
    assert np.corrcoef(reference.bands[2][inside], prior.bands[2][inside])[0, 1] < 0.9


def test_warp_role_images_leaves_uncovered_area_empty() -> None:
    images = ie.RoleImages(
        images={"dapi": np.full((40, 60), 100.0, np.float32)},
        to_level0=np.eye(3),
        pixel_size_um=2.0,
        level0_pixel_size_um=2.0,
    )

    def shifted(points_um: np.ndarray) -> np.ndarray:
        return np.asarray(points_um) + np.array([50.0, 0.0])

    warped = ie.warp_role_images(
        images, reference=images, pull_um=shifted, lattice_px=7
    )

    # Reference px x maps to moving px x + 25: columns 35 and beyond are outside.
    assert warped.images["dapi"][:, :30].min() == pytest.approx(100.0)
    assert warped.images["dapi"][:, 36:].max() == 0.0


# ---------------------------------------------------------------------------
# Empirical null
# ---------------------------------------------------------------------------


def test_empirical_null_of_independent_images_stays_low_and_is_deterministic(
    pair: _Pair, other_pair: _Pair
) -> None:
    kwargs: dict[str, Any] = {
        "representation": "bp20",
        "window_um": 600.0,
        "search_um": 250.0,
        "n_pairs": 200,
        "seed": 3,
    }

    first = ie.empirical_null(pair.reference, other_pair.reference, **kwargs)
    again = ie.empirical_null(pair.reference, other_pair.reference, **kwargs)
    different = ie.empirical_null(
        pair.reference, other_pair.reference, **{**kwargs, "seed": 4}
    )

    assert first == again
    assert first != different
    assert first.n > 100
    assert first.peak_p99 < 0.35
    assert 0.0 < first.peak_p95 <= first.peak_p99
    assert 0.0 < first.distinct_p95 < 0.3


def test_empirical_null_rejects_an_unknown_representation(pair: _Pair) -> None:
    with pytest.raises(ValueError, match="unknown representation"):
        ie.empirical_null(
            pair.reference,
            pair.moving,
            representation="nuclei",
            window_um=600.0,
            search_um=250.0,
            n_pairs=5,
        )


# ---------------------------------------------------------------------------
# Click checks
# ---------------------------------------------------------------------------


def _check(
    pair: _Pair,
    nulls: dict[tuple[str, float], ie.NullModel],
    reference_um: np.ndarray,
    moving_um: np.ndarray,
    *,
    predicted_um: np.ndarray | None = None,
    pull: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[ie.ClickCheck]:
    reference_um = np.atleast_2d(reference_um)
    moving_um = np.atleast_2d(moving_um)
    return ie.check_clicks(
        pair.reference,
        pair.moving,
        pair_ids=list(range(1, len(reference_um) + 1)),
        reference_xy_um=reference_um,
        moving_xy_um=moving_um,
        predicted_moving_xy_um=moving_um if predicted_um is None else predicted_um,
        pull_um=pull or pair.phantom.pull,
        nulls=nulls,
    )


def test_correct_clicks_are_confirmed_strongly_within_a_few_um(
    pair: _Pair, click_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    reference_um = _deep_tissue_points(pair, 10, seed=1)
    truth = pair.phantom.pull(reference_um)

    checks = _check(pair, click_nulls, reference_um, truth)

    assert [c.verdict for c in checks] == ["confirmed"] * 10
    assert all(c.strength == "strong" for c in checks)
    assert all(
        c.image_offset_um is not None and c.image_offset_um <= 5.0 for c in checks
    )
    assert all(
        c.snap_moving_xy_um is None and c.ridge_direction_moving_xy is None
        for c in checks
    )


def test_a_click_moved_120_um_is_contradicted_with_a_snap_at_the_truth(
    pair: _Pair, click_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    reference_um = _deep_tissue_points(pair, 6, seed=2)
    truth = pair.phantom.pull(reference_um)
    direction = np.array([0.8, 0.6])

    checks = _check(
        pair, click_nulls, reference_um, truth + 120.0 * direction, predicted_um=truth
    )

    assert [c.verdict for c in checks] == ["contradicted"] * 6
    for check, true_position in zip(checks, truth, strict=True):
        assert check.snap_moving_xy_um is not None
        assert np.hypot(*(check.snap_moving_xy_um - true_position)) <= 10.0
        assert check.strength is None


def test_a_click_outside_the_image_is_not_checked(
    pair: _Pair, click_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    inside = _deep_tissue_points(pair, 1, seed=3)
    reference_um = np.vstack([inside, [[-500.0, -500.0]]])

    checks = _check(pair, click_nulls, reference_um, pair.phantom.pull(reference_um))

    assert checks[0].verdict == "confirmed"
    assert checks[1].verdict == "not_checked"
    assert checks[1].estimates == ()
    assert checks[1].image_offset_um is None


def test_check_clicks_requires_matching_lengths_and_grids(pair: _Pair) -> None:
    with pytest.raises(ValueError, match="same length"):
        ie.check_clicks(
            pair.reference,
            pair.moving,
            pair_ids=[1, 2],
            reference_xy_um=np.zeros((1, 2)),
            moving_xy_um=np.zeros((1, 2)),
            predicted_moving_xy_um=np.zeros((1, 2)),
            pull_um=pair.phantom.pull,
        )
    assert (
        ie.check_clicks(
            pair.reference,
            pair.moving,
            pair_ids=[],
            reference_xy_um=np.zeros((0, 2)),
            moving_xy_um=np.zeros((0, 2)),
            predicted_moving_xy_um=np.zeros((0, 2)),
            pull_um=pair.phantom.pull,
        )
        == []
    )


# A straight tube in a cleared region, and a cleared region without one.
_TUBE_Y_UM = 2800.0
_RIDGE_BOXES = ((2200.0, 2300.0, 3600.0, 3300.0), (1700.0, 3600.0, 3000.0, 4400.0))


@pytest.fixture(scope="module")
def ridge_pair() -> _Pair:
    phantom = make_phantom(
        3,
        clear_boxes_um=_RIDGE_BOXES,
        extra_tubes_um=[(1500.0, _TUBE_Y_UM, 4300.0, _TUBE_Y_UM, 20.0)],
    )
    return _build_pair(phantom)


@pytest.fixture(scope="module")
def ridge_nulls(ridge_pair: _Pair) -> dict[tuple[str, float], ie.NullModel]:
    nulls: dict[tuple[str, float], ie.NullModel] = {}
    for window_um in ie.ClickCheckParams().windows_um:
        built = ie._null_models(
            ridge_pair.reference,
            ridge_pair.moving,
            representations=("bp20", "orientation", "fused"),
            window_um=window_um,
            search_um=250.0,
            n_pairs=150,
            min_distance_um=2000.0,
            seed=0,
        )
        nulls.update({(name, window_um): model for name, model in built.items()})
    return nulls


def test_a_click_moved_along_a_straight_tube_is_a_ridge(
    ridge_pair: _Pair, ridge_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    pull = ridge_pair.phantom.pull
    on_tube = np.array([[2900.0, _TUBE_Y_UM]])
    truth = pull(on_tube)
    step = 5.0
    tube_direction = (pull(on_tube + [[step, 0.0]]) - pull(on_tube - [[step, 0.0]]))[0]
    tube_direction = tube_direction / np.hypot(*tube_direction)

    (check,) = _check(
        ridge_pair,
        ridge_nulls,
        on_tube,
        truth + 80.0 * tube_direction,
        predicted_um=truth,
    )

    assert check.verdict == "ridge"
    assert check.ridge_direction_moving_xy is not None
    cosine = abs(float(check.ridge_direction_moving_xy @ tube_direction))
    assert math.degrees(math.acos(min(cosine, 1.0))) < 10.0
    assert check.elongation is not None and check.elongation >= 2.0
    assert check.snap_moving_xy_um is None and check.strength is None


def test_a_click_moved_across_a_straight_tube_is_not_a_ridge(
    ridge_pair: _Pair, ridge_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    pull = ridge_pair.phantom.pull
    on_tube = np.array([[2900.0, _TUBE_Y_UM]])
    truth = pull(on_tube)
    step = 5.0
    tube_direction = (pull(on_tube + [[step, 0.0]]) - pull(on_tube - [[step, 0.0]]))[0]
    tube_direction = tube_direction / np.hypot(*tube_direction)
    normal = np.array([-tube_direction[1], tube_direction[0]])

    (check,) = _check(
        ridge_pair,
        ridge_nulls,
        on_tube,
        truth + 80.0 * normal,
        predicted_um=truth,
    )

    # The click is off the structure: the verdict depends on where it was placed.
    assert check.verdict in ("inconclusive", "contradicted")
    assert check.ridge_direction_moving_xy is None


def test_correct_clicks_on_a_tube_are_not_contradicted(
    ridge_pair: _Pair, ridge_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    on_tube = np.array([[2900.0, _TUBE_Y_UM], [3300.0, _TUBE_Y_UM]])

    checks = _check(ridge_pair, ridge_nulls, on_tube, ridge_pair.phantom.pull(on_tube))

    # An end-to-end guard on the verdict. The rule that ridge-shaped band-pass
    # estimates never vote for a contradiction is covered on hand-made estimates
    # (test_ridge_shaped_estimates_never_vote_for_a_contradiction): the phantom's
    # tube gives no confident ridge-shaped band-pass estimate to exclude.
    assert all(c.verdict != "contradicted" for c in checks)


def test_a_click_in_a_featureless_region_is_inconclusive(
    ridge_pair: _Pair, ridge_nulls: dict[tuple[str, float], ie.NullModel]
) -> None:
    empty = np.array([[2350.0, 4000.0]])

    (check,) = _check(ridge_pair, ridge_nulls, empty, ridge_pair.phantom.pull(empty))

    assert check.verdict == "inconclusive"
    assert check.snap_moving_xy_um is None


# The verdict rules on hand-made estimates (amendment A2).


def _estimate(
    representation: str,
    window_um: float,
    offset_um: tuple[float, float],
    *,
    confident: bool = True,
    elongation: float = 1.2,
) -> ie.ClickEstimate:
    return ie.ClickEstimate(
        representation=representation,
        window_um=window_um,
        peak=0.9,
        distinct=0.3,
        on_border=False,
        confident=confident,
        elongation=elongation,
        image_moving_xy_um=np.array(offset_um) + np.array([1000.0, 2000.0]),
    )


def _assess(*estimates: ie.ClickEstimate) -> ie.ClickCheck:
    click = np.array([1000.0, 2000.0])
    return ie._assess(7, estimates, click, click, ie.ClickCheckParams())


def test_a_fused_small_window_confirmation_without_dissent_is_strong() -> None:
    check = _assess(
        _estimate("fused", 300.0, (5.0, 0.0)),
        _estimate("bp20", 300.0, (4.0, 3.0)),
        _estimate("orientation", 600.0, (200.0, 0.0), confident=False),
    )

    assert (check.verdict, check.strength) == ("confirmed", "strong")
    assert check.image_offset_um == pytest.approx(5.0)


def test_a_confirmation_with_a_confident_distant_band_estimate_is_weak() -> None:
    check = _assess(
        _estimate("fused", 300.0, (5.0, 0.0)),
        _estimate("bp20", 300.0, (250.0, 0.0)),
    )

    assert (check.verdict, check.strength) == ("confirmed", "weak")
    # A ridge-shaped dissenting estimate does not make it weak.
    ridge = _assess(
        _estimate("fused", 300.0, (5.0, 0.0)),
        _estimate("bp20", 300.0, (250.0, 0.0), elongation=3.0),
    )
    assert (ridge.verdict, ridge.strength) == ("confirmed", "strong")
    # Neither does a confirmation that comes from the large window only.
    large = _assess(_estimate("fused", 600.0, (5.0, 0.0)))
    assert (large.verdict, large.strength) == ("confirmed", "weak")


def test_two_agreeing_distant_estimates_contradict_with_a_snap() -> None:
    check = _assess(
        _estimate("bp20", 300.0, (150.0, 100.0)),
        _estimate("orientation", 600.0, (160.0, 90.0)),
        _estimate("fused", 300.0, (400.0, 0.0), confident=False),
    )

    assert check.verdict == "contradicted"
    assert check.snap_moving_xy_um == pytest.approx([1155.0, 2095.0])
    assert check.strength is None


def test_a_contradiction_from_large_windows_only_is_inconclusive() -> None:
    check = _assess(
        _estimate("bp20", 600.0, (150.0, 100.0)),
        _estimate("orientation", 600.0, (155.0, 95.0)),
    )

    assert (check.verdict, check.reason) == ("inconclusive", "large_window_only")


def test_a_contradiction_from_orientation_only_is_inconclusive() -> None:
    check = _assess(
        _estimate("orientation", 300.0, (150.0, 100.0)),
        _estimate("orientation", 600.0, (155.0, 95.0)),
    )

    assert (check.verdict, check.reason) == ("inconclusive", "orientation_only")


def test_ridge_shaped_estimates_never_vote_for_a_contradiction() -> None:
    check = _assess(
        _estimate("bp20", 300.0, (150.0, 100.0), elongation=3.0),
        _estimate("bp20", 600.0, (155.0, 95.0), elongation=2.5),
    )

    assert check.verdict == "inconclusive"


def test_a_click_that_is_both_confirmed_and_contradicted_is_conflicting() -> None:
    check = _assess(
        _estimate("fused", 300.0, (5.0, 0.0)),
        _estimate("bp20", 300.0, (150.0, 100.0)),
        _estimate("bp20", 600.0, (152.0, 101.0)),
    )

    assert (check.verdict, check.reason) == ("inconclusive", "conflicting")
    # The image's own opinion survives: the confirming fused estimate.
    assert check.image_offset_um == pytest.approx(5.0)
    assert check.snap_moving_xy_um is None


def test_an_inconclusive_click_keeps_the_best_confident_fused_estimate() -> None:
    check = _assess(
        _estimate("fused", 300.0, (30.0, 0.0)),
        _estimate("fused", 600.0, (-35.0, 0.0), confident=False),
    )

    assert (check.verdict, check.reason) == ("inconclusive", "no_agreement")
    assert check.image_moving_xy_um == pytest.approx([1030.0, 2000.0])
    assert check.image_offset_um == pytest.approx(30.0)
    assert check.prediction_offset_um == pytest.approx(30.0)
    assert check.snap_moving_xy_um is None


def test_an_inconclusive_click_without_a_confident_fused_estimate_has_no_position() -> (
    None
):
    check = _assess(_estimate("bp20", 300.0, (150.0, 0.0)))

    assert check.verdict == "inconclusive"
    assert check.image_moving_xy_um is None and check.image_offset_um is None


def test_a_chain_of_estimates_agreeing_through_a_middle_one_does_not_contradict() -> (
    None
):
    # 172 and 150 agree, 172 and 195 agree, but 150 and 195 are 45 um apart (more
    # than agree_um): no pairwise-agreeing set has both a band-pass estimate and
    # one at the smallest window.
    check = _assess(
        _estimate("orientation", 600.0, (172.0, 0.0)),
        _estimate("bp20", 600.0, (150.0, 0.0)),
        _estimate("orientation", 300.0, (195.0, 0.0)),
    )

    assert check.verdict == "inconclusive"
    assert check.reason in ("orientation_only", "large_window_only")
    assert check.snap_moving_xy_um is None


def test_the_largest_agreeing_group_sets_the_snap_and_ties_go_to_the_prediction() -> (
    None
):
    click = np.array([1000.0, 2000.0])
    estimates = (
        _estimate("bp20", 300.0, (150.0, 0.0)),
        _estimate("bp20", 600.0, (155.0, 0.0)),
        _estimate("bp20", 300.0, (-150.0, 0.0)),
        _estimate("bp20", 600.0, (-155.0, 0.0)),
    )
    near_left = click + [-150.0, 0.0]

    check = ie._assess(7, estimates, click, near_left, ie.ClickCheckParams())

    assert check.verdict == "contradicted"
    assert check.snap_moving_xy_um == pytest.approx([847.5, 2000.0])


# ---------------------------------------------------------------------------
# Tissue-scale window observations
# ---------------------------------------------------------------------------

_WINDOW_PARAMS = ie.WindowParams(null_pairs=150, workers=4)


@pytest.fixture(scope="module")
def prior_pair() -> _Pair:
    """The phantom warped by its affine part only (the sinusoid is left to find)."""
    phantom = _phantom(0, warp_wavelength_um=_WINDOW_WAVELENGTH_UM)
    return _build_pair(phantom, pull=phantom.prior)


@pytest.fixture(scope="module")
def observations(prior_pair: _Pair) -> ie.WindowObservations:
    return ie.tissue_window_observations(
        prior_pair.reference,
        prior_pair.moving,
        pull_um=prior_pair.phantom.prior,
        params=_WINDOW_PARAMS,
    )


def test_window_observations_recover_the_warp_the_prior_misses(
    prior_pair: _Pair, observations: ie.WindowObservations
) -> None:
    truth = prior_pair.phantom.pull(observations.reference_xy_um)
    error = np.hypot(*(observations.moving_xy_um - truth).T)
    candidates = observations.candidates
    summary = observations.summary

    assert len(error) > 20
    assert float(np.median(error)) <= 8.0
    assert candidates["accepted"].mean() > 0.3
    assert summary["n_thinned"] == len(error)
    assert summary["n_merged"] >= summary["n_thinned"]
    assert set(summary["by_size"]) == {"600", "800"}
    for entry in summary["by_size"].values():
        assert entry["null_p99"] >= entry["null_p95"] > 0.0
        assert entry["accepted"] <= entry["confident"] <= entry["candidates"]
    assert summary["runtime_s"] > 0.0
    assert observations.residual_um.shape == (len(error), 2)
    assert set(observations.region) <= {"grey_matter", "white_matter", "edge"}
    # The residuals are what the prior misses: tens of um, from the sinusoid.
    assert np.hypot(*observations.residual_um.T).max() > 10.0


def test_kept_windows_are_at_least_the_thinning_distance_apart(
    observations: ie.WindowObservations,
) -> None:
    positions = observations.reference_xy_um
    distances = np.hypot(
        positions[:, None, 0] - positions[None, :, 0],
        positions[:, None, 1] - positions[None, :, 1],
    )
    np.fill_diagonal(distances, np.inf)

    assert distances.min() >= 400.0 - 1e-6
    assert observations.candidates["kept"].sum() == len(positions)
    # Larger windows win the merge: a kept 600 um window has no accepted 800 um
    # window within the merge radius.
    accepted = observations.candidates[observations.candidates["accepted"]]
    large = accepted[accepted["window_um"] == 800.0][
        ["reference_x_um", "reference_y_um"]
    ]
    for index in np.flatnonzero(observations.window_um == 600.0):
        gaps = np.hypot(*(large.to_numpy() - observations.reference_xy_um[index]).T)
        assert gaps.size == 0 or gaps.min() > 300.0


def test_window_covariances_are_symmetric_positive_semidefinite(
    observations: ie.WindowObservations,
) -> None:
    covariance = observations.peak_cov_um2

    assert covariance.shape == (len(observations.reference_xy_um), 2, 2)
    assert np.allclose(covariance, np.swapaxes(covariance, 1, 2))
    assert np.linalg.eigvalsh(covariance).min() >= -1e-9
    assert observations.elongation.shape == (len(covariance),)
    assert (observations.elongation >= 1.0).all()
    sd = observations.summary["peak_sd_um"]
    assert 0.0 < sd["minor"]["median"] <= sd["major"]["median"] < 100.0


def test_the_covariance_of_a_window_on_a_tube_follows_the_tube(
    ridge_pair: _Pair,
) -> None:
    pixel = ridge_pair.reference.pixel_size_um
    on_tube = ridge_pair.reference.um_to_px([[2900.0, _TUBE_Y_UM]])[0]
    centre = (int(round(on_tube[0])), int(round(on_tube[1])))
    geometry = ie._geometry(600.0, 300.0, pixel)

    scores = ie._correlate(
        ridge_pair.reference, ridge_pair.moving, centre, centre, geometry
    )["bp20"]
    cov = ie.peak_covariance(
        scores,
        np.unravel_index(scores.argmax(), scores.shape)[::-1],
        pixel_size_um=pixel,
        drop=0.05,
        radius_um=100.0,
    )

    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    assert math.degrees(math.atan2(abs(major[1]), abs(major[0]))) < 10.0
    assert values[1] > 3.0 * values[0]


def test_window_covariances_are_reported_in_the_moving_frame(
    prior_pair: _Pair, observations: ie.WindowObservations
) -> None:
    phantom = prior_pair.phantom
    reference, pixel = prior_pair.reference, prior_pair.reference.pixel_size_um
    # The prior is the phantom's 35 degree rotation, so the Jacobian is far from
    # the identity: a covariance left in the reference frame would be wrong.
    step = 5.0
    probe = observations.reference_xy_um[:1]
    jacobian = np.stack(
        [
            (
                phantom.prior(probe + [[step, 0.0]])
                - phantom.prior(probe - [[step, 0.0]])
            )[0],
            (
                phantom.prior(probe + [[0.0, step]])
                - phantom.prior(probe - [[0.0, step]])
            )[0],
        ],
        axis=1,
    ) / (2.0 * step)
    assert abs(math.degrees(math.atan2(jacobian[1, 0], jacobian[0, 0]))) > 25.0

    checked = 0
    for index in np.argsort(-observations.elongation)[:3]:
        centre_um = observations.reference_xy_um[index]
        centre = tuple(int(round(v)) for v in reference.um_to_px(centre_um))
        geometry = ie._geometry(float(observations.window_um[index]), 300.0, pixel)
        scores = ie._correlate(
            reference,
            prior_pair.moving,
            centre,
            centre,
            geometry,
            use_orientation=False,
        )["bp20"]
        peak_x, peak_y, *_ = ie.summarise_scores(
            scores, exclusion_px=int(round(50.0 / pixel))
        )[2:]
        in_reference = ie.peak_covariance(
            scores,
            (peak_x, peak_y),
            pixel_size_um=pixel,
            drop=0.05,
            radius_um=100.0,
        )
        expected = jacobian @ in_reference @ jacobian.T

        assert np.allclose(
            observations.peak_cov_um2[index], expected, rtol=1e-3, atol=1e-6
        )
        # The two frames differ materially, so the check discriminates.
        assert not np.allclose(in_reference, expected, rtol=0.2, atol=1e-6)
        checked += 1
    assert checked == 3


def test_windows_over_unrelated_texture_are_rejected() -> None:
    phantom = _phantom(0, warp_wavelength_um=_WINDOW_WAVELENGTH_UM)
    box = (2000.0, 2000.0, 4200.0, 4200.0)
    textured = Phantom(
        reference=phantom.reference,
        moving=replace_with_texture(phantom.moving, box),
        pull=phantom.pull,
        prior=phantom.prior,
        tubes_um=phantom.tubes_um,
    )
    pair = _build_pair(textured, pull=phantom.prior)

    result = ie.tissue_window_observations(
        pair.reference, pair.moving, pull_um=phantom.prior, params=_WINDOW_PARAMS
    )

    assert len(result.reference_xy_um) > 5  # the rest of the tissue still matches
    inner = np.array([box[0] + 450, box[1] + 450, box[2] - 450, box[3] - 450])
    inside = (
        (result.moving_xy_um[:, 0] > inner[0])
        & (result.moving_xy_um[:, 0] < inner[2])
        & (result.moving_xy_um[:, 1] > inner[1])
        & (result.moving_xy_um[:, 1] < inner[3])
    )
    assert not inside.any()


def test_confident_windows_that_disagree_with_their_neighbours_are_not_accepted() -> (
    None
):
    phantom = _phantom(0, warp_wavelength_um=_WINDOW_WAVELENGTH_UM)
    centre = np.array([2800.0, 2800.0])
    half = 350.0
    moving_centre = phantom.pull(centre)
    box = (
        float(moving_centre[0] - half),
        float(moving_centre[1] - half),
        float(moving_centre[0] + half),
        float(moving_centre[1] + half),
    )
    # The patch shows the same tissue 90 um away: windows inside it match the
    # reference confidently, but at a displacement that its neighbours outside do
    # not share.
    shifted = Phantom(
        reference=phantom.reference,
        moving=shift_patch(phantom.moving, box, (90.0, 0.0)),
        pull=phantom.pull,
        prior=phantom.prior,
        tubes_um=phantom.tubes_um,
    )
    pair = _build_pair(shifted, pull=phantom.prior)

    result = ie.tissue_window_observations(
        pair.reference, pair.moving, pull_um=phantom.prior, params=_WINDOW_PARAMS
    )

    candidates = result.candidates
    inside = (np.abs(candidates["reference_x_um"] - centre[0]) < half) & (
        np.abs(candidates["reference_y_um"] - centre[1]) < half
    )
    rejected = candidates["confident"] & ~candidates["accepted"] & inside
    assert rejected.any()
    assert (candidates.loc[rejected, "neighbour_deviation_um"] > 30.0).all()
    assert not (candidates["kept"] & rejected).any()
    truth = phantom.pull(result.reference_xy_um)
    assert np.hypot(*(result.moving_xy_um - truth).T).max() <= 20.0


def test_threaded_and_serial_window_runs_are_identical(prior_pair: _Pair) -> None:
    kwargs: dict[str, Any] = {
        "pull_um": prior_pair.phantom.prior,
    }
    small = {"sizes_um": (600.0,), "null_pairs": 30, "step_um": 400.0}

    serial = ie.tissue_window_observations(
        prior_pair.reference,
        prior_pair.moving,
        params=ie.WindowParams(workers=1, **small),
        **kwargs,
    )
    threaded = ie.tissue_window_observations(
        prior_pair.reference,
        prior_pair.moving,
        params=ie.WindowParams(workers=4, **small),
        **kwargs,
    )

    assert serial.nulls == threaded.nulls
    assert np.array_equal(serial.reference_xy_um, threaded.reference_xy_um)
    assert np.array_equal(serial.moving_xy_um, threaded.moving_xy_um)
    assert np.array_equal(serial.peak_cov_um2, threaded.peak_cov_um2)
    assert serial.candidates.equals(threaded.candidates)


def test_window_threads_restore_the_opencv_thread_count(prior_pair: _Pair) -> None:
    import cv2

    before = cv2.getNumThreads()
    ie.tissue_window_observations(
        prior_pair.reference,
        prior_pair.moving,
        pull_um=prior_pair.phantom.prior,
        params=ie.WindowParams(
            sizes_um=(600.0,), null_pairs=10, step_um=800.0, workers=3
        ),
    )

    assert cv2.getNumThreads() == before


# ---------------------------------------------------------------------------
# Target weights
# ---------------------------------------------------------------------------


def _coarse(images: ie.RoleImages, factor: int = 8) -> ie.RoleImages:
    """Block-average role images, with the pixel-centre affine of the blocks."""
    out = {}
    for role, image in images.images.items():
        height, width = (s // factor * factor for s in image.shape)
        out[role] = (
            image[:height, :width]
            .reshape(height // factor, factor, width // factor, factor)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )
    centre = (factor - 1) / 2.0
    return ie.RoleImages(
        images=out,
        # Coarse px -> fine px first, then the image's own working px -> level 0.
        to_level0=images.to_level0
        @ np.array([[factor, 0, centre], [0, factor, centre], [0, 0, 1.0]]),
        pixel_size_um=images.pixel_size_um * factor,
        level0_pixel_size_um=images.level0_pixel_size_um,
    )


def test_target_weights_sit_on_shared_tissue_and_favour_grey_matter(
    pair: _Pair,
) -> None:
    phantom = pair.phantom
    reference, moving = _coarse(phantom.reference), _coarse(phantom.moving)

    weights = ie.target_weight_map(reference, moving, pull_um=phantom.prior)

    assert len(weights.centres_um) == len(weights.weights) == len(weights.region)
    assert len(weights.centres_um) > 100
    assert set(np.unique(weights.weights)) == {0.25, 1.0}
    assert all(
        w == (1.0 if r == "grey_matter" else 0.25)
        for w, r in zip(weights.weights, weights.region, strict=True)
    )
    # Spacing: every 100 um, so at most (size / 100)^2 candidates.
    assert len(weights.centres_um) <= (2048 * 2.76 / 100.0 + 1) ** 2
    # Shared tissue: the points lie in the full-resolution reference tissue (up to
    # the coarse mask's resolution) and their pull inside the moving image.
    reference_px = np.rint(pair.reference.um_to_px(weights.centres_um)).astype(int)
    on_tissue = pair.reference.tissue[reference_px[:, 1], reference_px[:, 0]]
    assert on_tissue.mean() > 0.95
    local = phantom.prior(weights.centres_um) - MOVING_ORIGIN_UM
    assert ((local > 0) & (local < 2048 * 2.76)).all()
    # Moving-only tissue is excluded: a moving image with its tissue blanked
    # leaves no candidates.
    empty = ie.RoleImages(
        images={r: np.full_like(a, 20.0) for r, a in moving.images.items()},
        to_level0=moving.to_level0,
        pixel_size_um=moving.pixel_size_um,
        level0_pixel_size_um=moving.level0_pixel_size_um,
    )
    nothing = ie.target_weight_map(reference, empty, pull_um=phantom.prior)
    assert len(nothing.centres_um) == 0 and nothing.region == ()


def test_target_weights_without_roles_are_all_other(pair: _Pair) -> None:
    phantom = pair.phantom
    reference = _coarse(phantom.reference)
    moving = _coarse(phantom.moving)

    def only_dapi(images: ie.RoleImages) -> ie.RoleImages:
        return ie.RoleImages(
            images={"dapi": images.images["dapi"]},
            to_level0=images.to_level0,
            pixel_size_um=images.pixel_size_um,
            level0_pixel_size_um=images.level0_pixel_size_um,
        )

    with_roles = ie.target_weight_map(reference, moving, pull_um=phantom.prior)
    without = ie.target_weight_map(
        only_dapi(reference), only_dapi(moving), pull_um=phantom.prior
    )

    assert set(without.region) == {"other"}
    assert set(np.unique(without.weights)) == {0.25}
    # The ranking of candidates is unchanged: the same centres.
    assert np.array_equal(without.centres_um, with_roles.centres_um)
