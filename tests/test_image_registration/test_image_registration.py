"""Tests for registering external images onto the mask grid."""

from __future__ import annotations

import json
import math
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
import pytest
import spatialdata as sd
import tifffile
import xarray as xr
from scipy.ndimage import gaussian_filter, shift
from spatialdata.models import Image2DModel, TableModel

from merxen.config import ImageRegistrationConfig, RegisteredImageSpec
from merxen.image_registration import (
    _read_working_channel,
    check_matrix_scale,
    fit_affine_displacement,
    load_alignment_matrix,
    measure_local_shifts,
    run_image_registration,
)
from merxen.io.image_source import REGISTERED_IMAGES_ATTR
from merxen.viewer_cache.build import ViewerCacheParams, _resolve_base_image
from merxen.viewer_cache.format import derived_image_pyramid_cache_key

REFERENCE_SIZE = 768
IMAGE_SIZE = 640
IMAGE_PIXEL_SIZE_UM = 1.25
# True image-pixel -> reference-pixel affine: 90 degree rotation and 1.25x scale,
# the same geometry as the P7513 post-Xenium IF.
TRUE_MATRIX = np.array(
    [
        [0.0, IMAGE_PIXEL_SIZE_UM, -20.0],
        [-IMAGE_PIXEL_SIZE_UM, 0.0, 780.0],
        [0.0, 0.0, 1.0],
    ]
)
MARKER_ROWS = slice(300, 340)
MARKER_COLS = slice(500, 540)


def _write_matrix(path: Path, matrix: np.ndarray) -> Path:
    np.savetxt(path, matrix, delimiter=",")
    return path


def _nuclei_field(rng: np.random.Generator) -> np.ndarray:
    """Return a DAPI-like image: blurred nuclei on a tissue background."""
    image = np.zeros((REFERENCE_SIZE, REFERENCE_SIZE), dtype=np.float64)
    tissue = (slice(40, REFERENCE_SIZE - 40), slice(40, REFERENCE_SIZE - 40))
    image[tissue] = 80.0
    count = 2500
    ys = rng.integers(45, REFERENCE_SIZE - 45, count)
    xs = rng.integers(45, REFERENCE_SIZE - 45, count)
    seeds = np.zeros_like(image)
    seeds[ys, xs] = rng.uniform(800.0, 1600.0, count)
    image += gaussian_filter(seeds, 2.0) * 2.0 * math.pi * 4.0
    return image + rng.normal(0.0, 5.0, image.shape).clip(0.0)


@pytest.fixture
def registration_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write a reference zarr and an IF OME-TIFF related by ``TRUE_MATRIX``."""
    rng = np.random.default_rng(7)
    reference_dapi = _nuclei_field(rng)
    marker = np.zeros_like(reference_dapi)
    marker[MARKER_ROWS, MARKER_COLS] = 1000.0
    reference = np.stack([reference_dapi, reference_dapi * 0.5]).astype(np.uint16)
    sdata = sd.SpatialData(
        images={
            "morphology_focus": Image2DModel.parse(
                reference,
                dims=("c", "y", "x"),
                c_coords=["DAPI", "18S"],
                scale_factors=[2, 2],
            )
        },
        tables={"table": TableModel.parse(ad.AnnData(X=np.ones((3, 2))))},
    )
    zarr_path = tmp_path / "latest_spatialdata.zarr"
    sdata.write(zarr_path)

    def to_image(channel: np.ndarray) -> np.ndarray:
        # WARP_INVERSE_MAP: image pixel (x, y) samples the reference at TRUE(x, y).
        return cv2.warpAffine(
            channel.astype(np.float32),
            TRUE_MATRIX[:2],
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )

    image = np.stack([to_image(reference_dapi), to_image(marker)]).astype(np.uint16)
    image_path = tmp_path / "post_if.ome.tif"
    tifffile.imwrite(
        image_path,
        image,
        tile=(64, 64),
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["DAPI", "AF568"]},
            "PhysicalSizeX": IMAGE_PIXEL_SIZE_UM,
            "PhysicalSizeY": IMAGE_PIXEL_SIZE_UM,
        },
    )

    # Explorer-like matrix: the true one plus a 0.4 degree rotation about the
    # tissue centre and a (3, -2) pixel offset, as a manual alignment might leave.
    angle = math.radians(0.4)
    centre = REFERENCE_SIZE / 2.0
    rotate = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    about_centre = (
        np.array([[1, 0, centre + 3.0], [0, 1, centre - 2.0], [0, 0, 1]])
        @ rotate
        @ np.array([[1, 0, -centre], [0, 1, -centre], [0, 0, 1]])
    )
    return {
        "zarr": zarr_path,
        "image": image_path,
        "true_matrix": _write_matrix(tmp_path / "true.csv", TRUE_MATRIX),
        "explorer_matrix": _write_matrix(
            tmp_path / "explorer.csv",
            about_centre @ TRUE_MATRIX,
        ),
        "output_dir": tmp_path / "register_images_out",
    }


@pytest.fixture
def small_viewer_pyramids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the viewer pyramid build for test-sized images (default min is 1024)."""

    def _params(**kwargs: object) -> ViewerCacheParams:
        return ViewerCacheParams(min_size=64, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("merxen.image_registration.ViewerCacheParams", _params)


def _config(
    inputs: dict[str, Path],
    *,
    matrix: str = "explorer_matrix",
    image_key: str = "post_if",
    **overrides: object,
) -> ImageRegistrationConfig:
    settings: dict[str, object] = {
        "dataset_name": "P1_XENIUM",
        "platform": "XENIUM",
        "latest_zarr_path": inputs["zarr"],
        "output_dir": inputs["output_dir"],
        "images": [
            RegisteredImageSpec(
                image_key=image_key,
                image_path=inputs["image"],
                alignment_matrix_path=inputs[matrix],
                channel_names=["DAPI", "p62"],
            )
        ],
        "reference_pixel_size_um": 1.0,
        "registration_pixel_size_um": 2.0,
        "refinement_window_um": 64.0,
        "tile_size": 256,
        "chunk_size": 256,
    }
    settings.update(overrides)
    return ImageRegistrationConfig.model_validate(settings)


def _corner_error_px(matrix: np.ndarray) -> float:
    corners = np.array(
        [
            [0, 0, 1],
            [IMAGE_SIZE, 0, 1],
            [0, IMAGE_SIZE, 1],
            [IMAGE_SIZE, IMAGE_SIZE, 1],
        ],
        dtype=np.float64,
    ).T
    return float(np.abs((np.asarray(matrix) - TRUE_MATRIX) @ corners)[:2].max())


def test_load_alignment_matrix_accepts_explorer_csv(tmp_path: Path) -> None:
    """The comma-separated 3x3 file written by Xenium Explorer loads as-is."""
    path = tmp_path / "matrix.csv"
    path.write_text(
        "0.015618826142675095,1.615313853507129,-1418.8935033299726\n"
        "-1.615313853507129,0.015618826142675095,49651.47174083347\n"
        "0,0,1\n"
    )

    matrix = load_alignment_matrix(path)

    assert matrix.shape == (3, 3)
    assert matrix[1, 2] == pytest.approx(49651.47174083347)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1,0,0\n0,1,0\n", "3x3"),
        ("1,0,0\n0,1,0\n0.1,0,1\n", "not affine"),
        ("0,0,5\n0,0,5\n0,0,1\n", "singular"),
        ("a,b,c\n0,1,0\n0,0,1\n", "non-numeric"),
    ],
)
def test_load_alignment_matrix_rejects_invalid_files(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    """Malformed, projective, or singular matrices fail before any warp."""
    path = tmp_path / "matrix.csv"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        load_alignment_matrix(path)


def test_matrix_scale_matches_pixel_sizes_of_p7513() -> None:
    """The real P7513 matrix agrees with 0.343 um IF and 0.2125 um Xenium pixels."""
    matrix = np.array(
        [
            [0.015618826142675095, 1.615313853507129, -1418.89],
            [-1.615313853507129, 0.015618826142675095, 49651.47],
            [0.0, 0.0, 1.0],
        ]
    )

    check = check_matrix_scale(
        matrix,
        image_pixel_size_um=0.343374282,
        reference_pixel_size_um=0.2125,
        tolerance=0.02,
    )

    assert check.status == "ok"
    assert abs(check.relative_error or 1.0) < 0.001


@pytest.mark.parametrize(
    ("scale", "hint"),
    [(1.0 / 1.6159, "inverted"), (0.343374282, "image-pixel to micron")],
)
def test_matrix_scale_mismatch_names_the_likely_convention(
    scale: float,
    hint: str,
) -> None:
    """Inverted or micron matrices are rejected with a diagnosis."""
    matrix = np.diag([scale, scale, 1.0])

    with pytest.raises(ValueError, match=hint):
        check_matrix_scale(
            matrix,
            image_pixel_size_um=0.343374282,
            reference_pixel_size_um=0.2125,
            tolerance=0.02,
        )


def test_matrix_scale_check_is_skipped_without_pixel_size() -> None:
    """Images without OME calibration cannot be checked and are not rejected."""
    check = check_matrix_scale(
        np.diag([2.0, 2.0, 1.0]),
        image_pixel_size_um=None,
        reference_pixel_size_um=0.2125,
        tolerance=0.02,
    )

    assert check.status == "skipped_no_pixel_size"
    assert check.observed_scale == pytest.approx(2.0)


def test_measure_local_shifts_recovers_a_known_translation() -> None:
    """Content drawn shifted by +s must be reported as needing a -s shift."""
    reference = _nuclei_field(np.random.default_rng(3))
    moving = shift(reference, (2.0, -3.0), order=1)
    valid = np.zeros(reference.shape, dtype=bool)
    valid[60:-60, 60:-60] = True

    shifts = measure_local_shifts(
        np.log1p(reference),
        np.log1p(moving),
        valid,
        window_px=96,
    )

    assert len(shifts.shifts_xy) > 20
    # Windowed cross-correlation is biased ~0.1 px towards zero shift because the
    # overlap shrinks as the shift grows; 0.2 px is ~0.2 um on the working grid.
    np.testing.assert_allclose(
        np.median(shifts.shifts_xy, axis=0), [3.0, -2.0], atol=0.2
    )


def test_fit_affine_displacement_ignores_outlier_windows() -> None:
    """A few windows with local deformation must not bias the global fit."""
    rng = np.random.default_rng(1)
    centers = rng.uniform(0.0, 1000.0, (200, 2))
    linear = np.array([[0.002, -0.004], [0.004, 0.002]])
    translation = np.array([1.5, -0.7])
    shifts = centers @ linear.T + translation + rng.normal(0.0, 0.05, (200, 2))
    shifts[:10] += 15.0

    correction, inliers = fit_affine_displacement(centers, shifts)

    np.testing.assert_allclose(correction[:2, :2] - np.eye(2), linear, atol=2e-4)
    np.testing.assert_allclose(correction[:2, 2], translation, atol=0.1)
    assert not inliers[:10].any()


@pytest.mark.usefixtures("small_viewer_pyramids")
def test_registration_with_exact_matrix_places_marker_on_grid(
    registration_inputs: dict[str, Path],
) -> None:
    """The warped image lands on the reference grid where the matrix says."""
    config = _config(registration_inputs, matrix="true_matrix", refine_affine=False)

    outputs = run_image_registration(config)

    sdata = sd.read_zarr(registration_inputs["zarr"])
    image = sdata.images["post_if"]
    assert image.dims == ("c", "y", "x")
    assert image.shape == (2, REFERENCE_SIZE, REFERENCE_SIZE)
    assert list(image.coords["c"].values) == ["DAPI", "p62"]
    marker = image.sel(c="p62").values
    assert marker[305:335, 505:535].mean() > 950
    assert marker[100:140, 100:140].max() == 0
    dapi = image.sel(c="DAPI").values[60:-60, 60:-60].astype(float)
    reference = (
        sdata.images["morphology_focus"]["scale0"]
        .ds["image"]
        .sel(c="DAPI")
        .values[60:-60, 60:-60]
        .astype(float)
    )
    assert np.corrcoef(dapi.ravel(), reference.ravel())[0, 1] > 0.98
    assert "table" in sdata.tables
    assert derived_image_pyramid_cache_key("post_if", 4) in sdata.images

    registry = sdata.attrs[REGISTERED_IMAGES_ATTR]
    assert registry["post_if"]["status"] == "complete"
    assert registry["post_if"]["refinement_status"] == "disabled"
    summary = json.loads(outputs["post_if_summary"].read_text())
    assert summary["matrix_scale_check"]["status"] == "ok"
    assert Path(summary["outputs"]["qc_plot"]).exists()
    assert Path(summary["outputs"]["local_shifts"]).exists()


def test_refinement_recovers_the_true_matrix(
    registration_inputs: dict[str, Path],
) -> None:
    """A slightly-off Explorer matrix is corrected by the DAPI affine fit."""
    config = _config(registration_inputs)
    explorer = load_alignment_matrix(registration_inputs["explorer_matrix"])
    assert _corner_error_px(explorer) > 4.0

    outputs = run_image_registration(config)

    summary = json.loads(outputs["post_if_summary"].read_text())
    assert summary["refinement"]["status"] == "applied"
    assert _corner_error_px(np.asarray(summary["final_matrix"])) < 0.5
    assert (
        summary["residuals_after"]["median_um"]
        < summary["residuals_before"]["median_um"]
    )
    assert summary["residuals_after"]["median_um"] < 0.5


def test_refinement_beyond_limits_keeps_the_explorer_matrix(
    registration_inputs: dict[str, Path],
) -> None:
    """A correction larger than the safety limits is reported, not applied."""
    config = _config(registration_inputs, max_refinement_shift_um=0.5)

    outputs = run_image_registration(config)

    summary = json.loads(outputs["post_if_summary"].read_text())
    explorer = load_alignment_matrix(registration_inputs["explorer_matrix"])
    assert summary["refinement"]["status"] == "rejected_exceeds_limits"
    assert "shift" in summary["refinement"]["reason"]
    np.testing.assert_allclose(summary["final_matrix"], explorer)


def test_unchanged_registration_is_skipped_on_rerun(
    registration_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical inputs reuse the stored image instead of warping again."""
    config = _config(registration_inputs, refine_affine=False)
    run_image_registration(config)

    def _fail(**_: object) -> None:
        raise AssertionError("image was warped again")

    monkeypatch.setattr("merxen.image_registration.materialize_warped_image", _fail)
    run_image_registration(config)

    with pytest.raises(AssertionError, match="warped again"):
        run_image_registration(config, force_rerun=True)


@pytest.mark.usefixtures("small_viewer_pyramids")
def test_images_dropped_from_config_are_removed(
    registration_inputs: dict[str, Path],
) -> None:
    """A registered image no longer configured must not reach quantification."""
    run_image_registration(_config(registration_inputs, refine_affine=False))
    pyramid_key = derived_image_pyramid_cache_key("post_if", 4)
    assert pyramid_key in sd.read_zarr(registration_inputs["zarr"]).images

    outputs = run_image_registration(
        _config(registration_inputs, image_key="post_if_v2", refine_affine=False)
    )

    sdata = sd.read_zarr(registration_inputs["zarr"])
    assert "post_if" not in sdata.images
    assert pyramid_key not in sdata.images
    assert "post_if_v2" in sdata.images
    assert set(sdata.attrs[REGISTERED_IMAGES_ATTR]) == {"post_if_v2"}
    combined = json.loads(outputs["summary"].read_text())
    assert combined["removed_image_keys"] == ["post_if"]


def test_mismatched_matrix_fails_before_writing(
    registration_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An inverted matrix is rejected without adding an image to the zarr."""
    inverted = _write_matrix(tmp_path / "inverted.csv", np.linalg.inv(TRUE_MATRIX))
    config = _config({**registration_inputs, "explorer_matrix": inverted})

    with pytest.raises(ValueError, match="inverted"):
        run_image_registration(config)

    assert "post_if" not in sd.read_zarr(registration_inputs["zarr"]).images


def test_viewer_cache_base_image_skips_registered_images() -> None:
    """A registered key sorting before the instrument image is never the base."""
    image = Image2DModel.parse(
        np.zeros((1, 8, 8), dtype=np.uint16), dims=("c", "y", "x")
    )
    sdata = sd.SpatialData(images={"aaa_if": image, "morphology_focus": image})
    sdata.attrs[REGISTERED_IMAGES_ATTR] = {"aaa_if": {"status": "complete"}}

    key, *_ = _resolve_base_image(sdata)

    assert key == "morphology_focus"


def test_config_rejects_duplicate_or_reserved_image_keys(tmp_path: Path) -> None:
    """Keys must be unique and must not replace the reference image."""
    spec = {"image_path": tmp_path / "a.tif", "alignment_matrix_path": tmp_path / "m"}
    base = {
        "dataset_name": "P1_XENIUM",
        "platform": "XENIUM",
        "latest_zarr_path": tmp_path / "z.zarr",
        "output_dir": tmp_path,
    }
    with pytest.raises(ValueError, match="unique"):
        ImageRegistrationConfig.model_validate(
            {**base, "images": [{**spec, "image_key": "a"}, {**spec, "image_key": "a"}]}
        )
    with pytest.raises(ValueError, match="reference image"):
        ImageRegistrationConfig.model_validate(
            {**base, "images": [{**spec, "image_key": "morphology_focus"}]}
        )
    with pytest.raises(ValueError, match="pattern"):
        ImageRegistrationConfig.model_validate(
            {**base, "images": [{**spec, "image_key": "_napari_compare_x"}]}
        )


@pytest.mark.parametrize(
    ("target_factor", "expected_shape", "expected_scale"),
    [(4.0, (249, 249), 1000 / 249), (8.0, (124, 124), 2 * 1000 / 249)],
)
def test_working_level_tolerates_odd_pyramid_sizes(
    target_factor: float,
    expected_shape: tuple[int, int],
    expected_scale: float,
) -> None:
    """A level 4.016x smaller must count as the 4x level, not fall back to 2x."""
    shapes = [(1000, 1000), (500, 500), (249, 249)]
    levels = [
        xr.DataArray(np.ones((1, *shape), dtype=np.uint16), dims=("c", "y", "x"))
        for shape in shapes
    ]

    working = _read_working_channel(
        open_level=lambda index: levels[index],
        level_shapes_yx=shapes,
        channel_index=0,
        target_factor=target_factor,
    )

    assert working.pixels.shape == expected_shape
    assert working.to_level0[0, 0] == pytest.approx(expected_scale)


def _write_ome(
    path: Path,
    image: np.ndarray,
    *,
    names: list[str] | None,
    pixel_size_um: float | None = IMAGE_PIXEL_SIZE_UM,
    **kwargs: object,
) -> Path:
    metadata: dict[str, object] = {"axes": kwargs.pop("axes", "CYX")}
    if names is not None:
        metadata["Channel"] = {"Name": names}
    if pixel_size_um is not None:
        metadata.update(
            {"PhysicalSizeX": pixel_size_um, "PhysicalSizeY": pixel_size_um}
        )
    tifffile.imwrite(path, image, metadata=metadata, **kwargs)  # type: ignore[arg-type]
    return path


def _reference_channels(zarr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    reference = sd.read_zarr(zarr_path).images["morphology_focus"]["scale0"]
    dapi = reference.ds["image"].sel(c="DAPI").values.astype(np.float32)
    marker = np.zeros_like(dapi)
    marker[MARKER_ROWS, MARKER_COLS] = 1000.0
    return dapi, marker


def _translate(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def test_partial_coverage_registers_inside_and_zero_fills_outside(
    registration_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An IF crop covering part of the section lands only on its footprint."""
    full = tifffile.imread(registration_inputs["image"])
    crop_path = _write_ome(
        tmp_path / "crop.ome.tif", full[:, 350:550, 300:500], names=["DAPI", "AF568"]
    )
    crop_matrix = _write_matrix(
        tmp_path / "crop.csv", TRUE_MATRIX @ _translate(300, 350)
    )
    config = _config(
        {**registration_inputs, "image": crop_path, "crop_matrix": crop_matrix},
        matrix="crop_matrix",
    )

    outputs = run_image_registration(config)

    image = sd.read_zarr(registration_inputs["zarr"]).images["post_if"]
    marker = image.sel(c="p62").values
    dapi = image.sel(c="DAPI").values
    assert marker[305:335, 505:535].mean() > 950
    assert dapi[600:700, 100:200].max() == 0
    assert dapi[200:380, 450:650].mean() > 50
    summary = json.loads(outputs["post_if_summary"].read_text())
    assert summary["refinement"]["status"] in {
        "applied",
        "skipped_insufficient_windows",
    }
    assert 0.05 < summary["output"]["nonzero_fraction"] < 0.3


def test_reflected_matrix_is_registered_and_refined(
    registration_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A mirror (negative determinant) matrix is valid and refinable."""
    flip = np.array([[-1.25, 0.0, 780.0], [0.0, 1.25, -20.0], [0.0, 0.0, 1.0]])
    dapi, marker = _reference_channels(registration_inputs["zarr"])
    image = np.stack(
        [
            cv2.warpAffine(
                channel,
                flip[:2],
                (IMAGE_SIZE, IMAGE_SIZE),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
            for channel in (dapi, marker)
        ]
    ).astype(np.uint16)
    image_path = _write_ome(tmp_path / "flip.ome.tif", image, names=["DAPI", "AF568"])
    explorer = _write_matrix(tmp_path / "flip.csv", _translate(3.0, -2.0) @ flip)
    config = _config(
        {**registration_inputs, "image": image_path, "flip_matrix": explorer},
        matrix="flip_matrix",
    )

    outputs = run_image_registration(config)

    summary = json.loads(outputs["post_if_summary"].read_text())
    final = np.asarray(summary["final_matrix"])
    assert summary["refinement"]["status"] == "applied"
    assert np.linalg.det(final[:2, :2]) < 0
    assert np.abs(final - flip)[:2, 2].max() < 0.5
    registered = sd.read_zarr(registration_inputs["zarr"]).images["post_if"]
    assert registered.sel(c="p62").values[305:335, 505:535].mean() > 950


def test_rgb_image_without_registration_channel_skips_refinement(
    registration_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An H&E-style RGB image registers with the matrix alone and keeps uint8."""
    full = tifffile.imread(registration_inputs["image"])
    rgb = np.repeat((full[0] / 8).clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)
    image_path = _write_ome(
        tmp_path / "he.ome.tif", rgb, names=None, axes="YXS", photometric="rgb"
    )
    spec = RegisteredImageSpec(
        image_key="he",
        image_path=image_path,
        alignment_matrix_path=registration_inputs["true_matrix"],
        registration_channel=None,
    )
    config = _config(registration_inputs).model_copy(update={"images": [spec]})

    outputs = run_image_registration(config)

    summary = json.loads(outputs["he_summary"].read_text())
    image = sd.read_zarr(registration_inputs["zarr"]).images["he"]
    assert summary["refinement"]["status"] == "skipped_no_registration_channel"
    assert summary["residuals_after"] == {}
    assert set(summary["outputs"]) == {"summary"}
    assert image.dtype == np.uint8
    assert list(image.coords["c"].values) == ["c0", "c1", "c2"]


def test_plain_tiff_without_pixel_size_skips_scale_check(
    registration_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Without OME calibration the matrix scale is trusted and still refined."""
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, tifffile.imread(registration_inputs["image"]))
    config = _config({**registration_inputs, "image": plain})

    outputs = run_image_registration(config)

    summary = json.loads(outputs["post_if_summary"].read_text())
    assert summary["matrix_scale_check"]["status"] == "skipped_no_pixel_size"
    assert summary["image_pixel_size_um"] == pytest.approx(
        IMAGE_PIXEL_SIZE_UM, rel=0.02
    )
    assert summary["refinement"]["status"] == "applied"
    assert _corner_error_px(np.asarray(summary["final_matrix"])) < 0.5


def test_key_of_an_existing_instrument_image_is_refused(
    registration_inputs: dict[str, Path],
) -> None:
    """Registering under the name of an instrument image must not replace it."""
    sdata = sd.read_zarr(registration_inputs["zarr"])
    sdata.images["other_stain"] = Image2DModel.parse(
        np.full((1, 8, 8), 7, dtype=np.uint16), dims=("c", "y", "x")
    )
    sdata.write_element("other_stain")
    config = _config(registration_inputs, image_key="other_stain")

    with pytest.raises(ValueError, match="did not register"):
        run_image_registration(config)

    kept = sd.read_zarr(registration_inputs["zarr"]).images["other_stain"]
    assert kept.shape == (1, 8, 8)


def test_missing_registration_channel_is_reported(
    registration_inputs: dict[str, Path],
) -> None:
    config = _config(registration_inputs)
    spec = config.images[0].model_copy(update={"registration_channel": "Hoechst"})

    with pytest.raises(ValueError, match="'Hoechst' not found"):
        run_image_registration(config.model_copy(update={"images": [spec]}))


def test_channel_name_count_must_match_the_image(
    registration_inputs: dict[str, Path],
) -> None:
    config = _config(registration_inputs)
    spec = config.images[0].model_copy(update={"channel_names": ["DAPI", "p62", "AT8"]})

    with pytest.raises(ValueError, match="3 channel name"):
        run_image_registration(config.model_copy(update={"images": [spec]}))


def test_rerun_in_a_fresh_output_dir_reuses_the_stored_registration(
    registration_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A new Nextflow work dir must not force the full-resolution warp again.

    Nextflow passes a relative output_dir resolved inside each task's work
    directory, so both runs use the same relative path from different cwds.
    """
    first_work, second_work = tmp_path / "work_a", tmp_path / "work_b"
    first_work.mkdir()
    second_work.mkdir()
    relative = {**registration_inputs, "output_dir": Path("register_images_out")}
    monkeypatch.chdir(first_work)
    first = run_image_registration(_config(relative, refine_affine=False))

    def _fail(**_: object) -> None:
        raise AssertionError("image was warped again")

    monkeypatch.setattr("merxen.image_registration.materialize_warped_image", _fail)
    monkeypatch.chdir(second_work)
    second = run_image_registration(_config(relative, refine_affine=False))

    reused = json.loads((second_work / second["post_if_summary"]).read_text())
    original = json.loads((first_work / first["post_if_summary"]).read_text())
    assert reused["reused"] is True
    assert reused["final_matrix"] == original["final_matrix"]
    assert set(reused["outputs"]) == {"summary", "qc_plot", "local_shifts"}
    for path in reused["outputs"].values():
        assert Path(path).is_relative_to(second_work / "register_images_out")
        assert Path(path).exists()
    assert Path(reused["outputs"]["qc_plot"]).with_suffix(".pdf").exists()
