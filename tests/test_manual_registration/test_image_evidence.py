"""Tests of the image evidence module (channel roles, representations, checks)."""

from __future__ import annotations

import math
from pathlib import Path

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
