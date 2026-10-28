"""Tests for the registration core extracted from ``REGISTER_IMAGES``."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import dask.array as da
import matplotlib.figure
import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from scipy.ndimage import gaussian_filter, shift
from spatialdata.transformations import Identity

import merxen.image_registration as registration_module
from merxen.alignment.image_warp import ImageWarpResult
from merxen.config import RegistrationSettings
from merxen.image_registration import (
    MovingImage,
    ReferenceGrid,
    RegistrationTuning,
    WorkingImage,
    _plot_nonrigid_qc,
    _plot_registration_qc,
    _read_working_channel,
    _reference_grid,
    _ReferenceGrid,
    _refine_nonrigid_registration,
    _refine_registration,
    _WorkingImage,
    load_alignment_matrix,
    measure_registration_overlay,
    moving_image_from_ome_tiff,
    read_working_channel,
    reference_grid,
    register_onto_reference,
    registration_output_paths,
    write_alignment_matrix,
)
from merxen.io.ome_tiff import read_ome_tiff_info
from merxen.io.spatialdata_io import spatialdata_write_lock

from .test_image_registration import (
    IMAGE_PIXEL_SIZE_UM,
    _config,
    _write_registration_inputs,
)


def _synthetic_dapi(size: int = 512, seed: int = 3) -> np.ndarray:
    """Return a DAPI-like field: Gaussian nuclei on a tissue background."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float64)
    image[24 : size - 24, 24 : size - 24] = 80.0
    count = 1500
    ys = rng.integers(30, size - 30, count)
    xs = rng.integers(30, size - 30, count)
    seeds = np.zeros_like(image)
    seeds[ys, xs] = rng.uniform(800.0, 1600.0, count)
    image += gaussian_filter(seeds, 2.0) * 2.0 * math.pi * 4.0
    image += rng.normal(0.0, 5.0, image.shape).clip(0.0)
    return image.astype(np.float32)


def _levels(image: np.ndarray) -> list[xr.DataArray]:
    arrays = [image]
    height, width = image.shape
    for factor in (2, 4):
        arrays.append(
            image.reshape(height // factor, factor, width // factor, factor).mean(
                axis=(1, 3)
            )
        )
    return [
        xr.DataArray(
            array[None].astype(np.float32),
            dims=("c", "y", "x"),
            coords={"c": ["DAPI"]},
        )
        for array in arrays
    ]


def _in_memory_reference(
    image: np.ndarray,
    factors: tuple[tuple[float, float], ...] | None = None,
) -> ReferenceGrid:
    return ReferenceGrid(
        key="reference",
        levels=_levels(image),
        transformations={"global": Identity()},
        coordinate_system="global",
        level_factors_xy=factors,
    )


def _in_memory_moving(
    image: np.ndarray,
    *,
    key: str = "moving",
    pixel_size_um: float = 1.0,
) -> MovingImage:
    levels = _levels(image)
    return MovingImage(
        image_key=key,
        channel_names=("DAPI",),
        pixel_size_um=pixel_size_um,
        level_shapes_yx=tuple(
            (int(level.sizes["y"]), int(level.sizes["x"])) for level in levels
        ),
        open_level=lambda index: levels[index],
        level0=levels[0],
    )


def _settings(**overrides: Any) -> RegistrationSettings:
    values: dict[str, Any] = {
        "reference_pixel_size_um": 1.0,
        "registration_pixel_size_um": 2.0,
        "refinement_window_um": 64.0,
        "min_refinement_windows": 5,
        "tile_size": 256,
        "chunk_size": 256,
        "build_viewer_pyramid": False,
    }
    values.update(overrides)
    return RegistrationSettings(**values)


def test_public_aliases(tmp_path: Path) -> None:
    assert _ReferenceGrid is ReferenceGrid
    assert WorkingImage is _WorkingImage
    assert _read_working_channel is read_working_channel
    assert registration_module._output_paths is registration_output_paths

    paths = registration_output_paths(tmp_path, "P1_XENIUM", "post_if")

    assert paths["summary"].name == "p1_xenium_post_if_registration_summary.json"
    assert set(paths) == {
        "summary",
        "qc_plot",
        "local_shifts",
        "nonrigid_flow",
        "nonrigid_qc_plot",
        "registration_quality",
    }


def test_reference_grid_matches_private_wrapper(tmp_path: Path) -> None:
    inputs = _write_registration_inputs(tmp_path)
    sdata = sd.read_zarr(inputs["zarr"], selection=("images",))
    config = _config(inputs)

    public = reference_grid(sdata, "morphology_focus", dataset_name="P1")
    private = _reference_grid(sdata, config)

    assert public.key == private.key
    assert public.coordinate_system == private.coordinate_system
    assert sorted(public.transformations) == sorted(private.transformations)
    assert public.level_factors_xy is None
    assert len(public.levels) == len(private.levels) == 3
    for first, second in zip(public.levels, private.levels, strict=True):
        assert np.array_equal(first.values, second.values)
    with pytest.raises(
        KeyError,
        match=(
            r"\[P1\] Reference image 'nope' not found; "
            r"images: \['morphology_focus'\]"
        ),
    ):
        reference_grid(sdata, "nope", dataset_name="P1")


def _ones_levels(shapes: list[tuple[int, int]]) -> list[xr.DataArray]:
    return [
        xr.DataArray(np.ones((1, *shape), dtype=np.float32), dims=("c", "y", "x"))
        for shape in shapes
    ]


def test_exact_level_factors_are_used() -> None:
    shapes = [(1000, 1000), (501, 501), (251, 251)]
    levels = _ones_levels(shapes)
    factors = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]

    exact = read_working_channel(
        open_level=lambda index: levels[index],
        level_shapes_yx=shapes,
        channel_index=0,
        target_factor=4.0,
        level_factors_xy=factors,
    )
    approximate = read_working_channel(
        open_level=lambda index: levels[index],
        level_shapes_yx=shapes,
        channel_index=0,
        target_factor=4.0,
    )
    coarser = read_working_channel(
        open_level=lambda index: levels[index],
        level_shapes_yx=shapes,
        channel_index=0,
        target_factor=8.0,
        level_factors_xy=factors,
    )

    assert exact.pixels.shape == (251, 251)
    assert exact.to_level0[0, 0] == 4.0
    assert exact.to_level0[0, 2] == 1.5
    assert approximate.to_level0[0, 0] == pytest.approx(1000 / 251)
    assert coarser.pixels.shape == (125, 125)
    assert coarser.to_level0[0, 0] == 8.0
    assert coarser.to_level0[0, 2] == 3.5


@pytest.mark.parametrize(
    ("target_factor", "expected_shape"),
    [(4.0, (249, 249)), (8.0, (124, 124))],
)
def test_level_factors_none_is_the_existing_path(
    target_factor: float, expected_shape: tuple[int, int]
) -> None:
    shapes = [(1000, 1000), (500, 500), (249, 249)]
    levels = _ones_levels(shapes)

    implicit = read_working_channel(
        open_level=lambda index: levels[index],
        level_shapes_yx=shapes,
        channel_index=0,
        target_factor=target_factor,
    )
    explicit = read_working_channel(
        open_level=lambda index: levels[index],
        level_shapes_yx=shapes,
        channel_index=0,
        target_factor=target_factor,
        level_factors_xy=None,
    )

    assert implicit.pixels.shape == expected_shape
    assert np.array_equal(implicit.pixels, explicit.pixels)
    assert np.array_equal(implicit.to_level0, explicit.to_level0)


@pytest.mark.parametrize(
    "factors",
    [
        [(1.0, 1.0), (2.0, 2.0)],
        [(2.0, 2.0), (2.0, 2.0), (4.0, 4.0)],
        [(1.0, 1.0), (0.0, 2.0), (4.0, 4.0)],
    ],
)
def test_invalid_level_factors_are_rejected(
    factors: list[tuple[float, float]], tmp_path: Path
) -> None:
    shapes = [(1000, 1000), (500, 500), (250, 250)]
    levels = _ones_levels(shapes)

    with pytest.raises(ValueError, match="level_factors_xy"):
        read_working_channel(
            open_level=lambda index: levels[index],
            level_shapes_yx=shapes,
            channel_index=0,
            target_factor=2.0,
            level_factors_xy=factors,
        )
    with pytest.raises(ValueError, match="level_factors_xy"):
        ReferenceGrid(
            key="reference",
            levels=levels,
            transformations={"global": Identity()},
            coordinate_system="global",
            level_factors_xy=tuple(factors),
        )


def test_numpy_scalar_level_factors_are_accepted() -> None:
    shapes = [(1000, 1000), (500, 500)]
    levels = _ones_levels(shapes)

    for factors in (
        [(1.0, 1.0), (np.int64(2), np.int64(2))],
        [(np.float32(1.0), np.float32(1.0)), (np.float32(2.0), np.float32(2.0))],
    ):
        result = read_working_channel(
            open_level=lambda index: levels[index],
            level_shapes_yx=shapes,
            channel_index=0,
            target_factor=2.0,
            level_factors_xy=factors,
        )
        assert result.pixels.shape == (500, 500)
        assert result.to_level0[0, 0] == 2.0
    ReferenceGrid(
        key="reference",
        levels=levels,
        transformations={"global": Identity()},
        coordinate_system="global",
        level_factors_xy=((1.0, 1.0), (np.int64(2), np.int64(2))),
    )
    with pytest.raises(ValueError, match="level_factors_xy"):
        ReferenceGrid(
            key="reference",
            levels=levels,
            transformations={"global": Identity()},
            coordinate_system="global",
            level_factors_xy=((1.0, 1.0), (np.bool_(True), np.int64(2))),
        )


def _holed_moving() -> tuple[np.ndarray, np.ndarray]:
    reference = _synthetic_dapi()
    rng = np.random.default_rng(5)
    moving = reference.copy()
    moving[176:336, 176:336] = rng.normal(20.0, 5.0, (160, 160)).clip(0.0)
    return reference, moving.astype(np.float32)


def _centres_in_level0(window_centres_xy: np.ndarray) -> np.ndarray:
    return window_centres_xy * 2.0 + 0.5


def test_moving_tissue_requirement_drops_windows_over_a_hole() -> None:
    reference, moving = _holed_moving()
    grid = _in_memory_reference(reference)
    image = _in_memory_moving(moving)
    settings = _settings(refine_affine=False)

    def measure(require: bool) -> Any:
        return _refine_registration(
            reference=grid,
            moving=image,
            registration_channel="DAPI",
            matrix=np.eye(3),
            settings=settings,
            tuning=RegistrationTuning(
                refinement_min_window_quality=-1.0,
                refinement_require_moving_tissue=require,
            ),
        )

    without = measure(False)
    with_requirement = measure(True)

    assert (
        with_requirement.before.shifts.n_candidate_windows
        < without.before.shifts.n_candidate_windows
    )
    assert len(with_requirement.before.shifts.centers_xy) < len(
        without.before.shifts.centers_xy
    )

    def inside(refinement: Any) -> int:
        centres = _centres_in_level0(refinement.before.shifts.centers_xy)
        within = (centres >= 200).all(axis=1) & (centres < 312).all(axis=1)
        return int(within.sum())

    assert inside(with_requirement) == 0
    assert inside(without) >= 1


def test_window_quality_threshold_is_tunable() -> None:
    reference = _synthetic_dapi()
    rng = np.random.default_rng(11)
    moving = (reference + rng.normal(0.0, 25.0, reference.shape)).clip(0.0)
    grid = _in_memory_reference(reference)
    image = _in_memory_moving(moving.astype(np.float32))
    settings = _settings(refine_affine=False)

    def windows(threshold: float) -> int:
        refinement = _refine_registration(
            reference=grid,
            moving=image,
            registration_channel="DAPI",
            matrix=np.eye(3),
            settings=settings,
            tuning=RegistrationTuning(refinement_min_window_quality=threshold),
        )
        assert refinement.before is not None
        return len(refinement.before.shifts.centers_xy)

    default = measure_registration_overlay(
        reference=grid,
        moving=image,
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=settings,
    )

    assert windows(0.99) < windows(0.3)
    assert windows(0.3) == default["n_windows"]


# The registration pixel size is 2 um on a 1 um reference grid (see ``_settings``).
_WORKING_SIZE = 256


def _wrong_scale_fit(
    centers_xy: np.ndarray, shifts_xy: np.ndarray, **_kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """A within-limits correction that is wrong: 1 % scale about the image centre."""
    centre = _WORKING_SIZE / 2.0
    correction = np.array(
        [[1.01, 0.0, -0.01 * centre], [0.0, 1.01, -0.01 * centre], [0.0, 0.0, 1.0]]
    )
    return correction, np.ones(len(centers_xy), dtype=bool)


def _refine_identical_images(
    monkeypatch: pytest.MonkeyPatch, tuning: RegistrationTuning
) -> Any:
    """Refine two identical images (the identity is already the best matrix)."""
    monkeypatch.setattr(
        registration_module, "fit_affine_displacement", _wrong_scale_fit
    )
    image = _synthetic_dapi()
    return _refine_registration(
        reference=_in_memory_reference(image),
        moving=_in_memory_moving(image),
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=_settings(),
        tuning=tuning,
    )


def test_refinement_that_lowers_the_correlation_is_rejected(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tuning = RegistrationTuning(
        refinement_min_correlation_gain=0.0, qc_initial_matrix_label="Landmark matrix"
    )

    with caplog.at_level("WARNING", logger="merxen.image_registration"):
        refinement = _refine_identical_images(monkeypatch, tuning)

    assert refinement.status == "rejected_no_correlation_gain"
    np.testing.assert_array_equal(refinement.matrix, np.eye(3))
    assert refinement.after is refinement.before
    before = refinement.correction["pearson_before"]
    candidate = refinement.correction["pearson_candidate"]
    assert candidate < before
    assert re.fullmatch(
        rf"DAPI correlation {before:.4f} -> {candidate:.4f} "
        rf"\(gain {candidate - before:.4f} < 0\.0\)",
        refinement.reason,
    )
    # The correction itself was inside the limits and is still reported.
    assert refinement.correction["n_inlier_windows"] > 0
    assert 0.0 < refinement.correction["scale_change"] < 0.02
    assert f"keeping the Landmark matrix ({refinement.reason})" in caplog.text, (
        caplog.text
    )


def test_the_reason_names_the_registration_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registration_module, "fit_affine_displacement", _wrong_scale_fit
    )
    image = _synthetic_dapi()
    levels = _levels(image)
    moving = MovingImage(
        image_key="moving",
        channel_names=("Hoechst",),
        pixel_size_um=1.0,
        level_shapes_yx=tuple(
            (int(level.sizes["y"]), int(level.sizes["x"])) for level in levels
        ),
        open_level=lambda index: levels[index],
        level0=levels[0].assign_coords(c=["Hoechst"]),
    )

    refinement = _refine_registration(
        reference=_in_memory_reference(image),
        moving=moving,
        registration_channel="Hoechst",
        matrix=np.eye(3),
        settings=_settings(),
        tuning=RegistrationTuning(refinement_min_correlation_gain=0.0),
    )

    assert refinement.status == "rejected_no_correlation_gain"
    assert refinement.reason.startswith("Hoechst correlation ")
    assert "DAPI" not in refinement.reason


def test_a_gain_of_exactly_the_threshold_is_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary: ``gain >= min`` applies, so a zero gain passes a 0.0 threshold."""

    def identity_fit(
        centers_xy: np.ndarray, shifts_xy: np.ndarray, **_kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.eye(3), np.ones(len(centers_xy), dtype=bool)

    monkeypatch.setattr(registration_module, "fit_affine_displacement", identity_fit)
    image = _synthetic_dapi()

    refinement = _refine_registration(
        reference=_in_memory_reference(image),
        moving=_in_memory_moving(image),
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=_settings(),
        tuning=RegistrationTuning(refinement_min_correlation_gain=0.0),
    )

    correction = refinement.correction
    # The precondition that makes this the boundary: not a hair above or below.
    assert correction["pearson_candidate"] - correction["pearson_before"] == 0.0
    assert refinement.status == "applied"
    assert refinement.reason is None
    assert refinement.after is not refinement.before


def test_refinement_inside_the_limits_applies_without_a_gain_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refinement = _refine_identical_images(monkeypatch, RegistrationTuning())

    # Today's behaviour: the wrong correction is applied, and the summary has no
    # Pearson keys, so REGISTER_IMAGES summaries are unchanged.
    assert refinement.status == "applied"
    assert not np.allclose(refinement.matrix, np.eye(3))
    assert refinement.after is not refinement.before
    assert "pearson_before" not in refinement.correction
    assert "pearson_candidate" not in refinement.correction
    assert set(refinement.correction) == {
        "max_shift_um",
        "rotation_deg",
        "scale_change",
        "n_windows",
        "n_inlier_windows",
    }


def test_a_lenient_gain_threshold_applies_the_correction_and_records_pearson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refinement = _refine_identical_images(
        monkeypatch, RegistrationTuning(refinement_min_correlation_gain=-0.5)
    )

    assert refinement.status == "applied"
    assert refinement.reason is None
    correction = refinement.correction
    assert correction["pearson_candidate"] < correction["pearson_before"]
    assert refinement.after is not refinement.before
    assert refinement.after.pearson_log == correction["pearson_candidate"]


def test_limit_violations_are_reported_before_the_gain_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registration_module,
        "fit_affine_displacement",
        lambda centers, shifts, **_: (
            np.array([[1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0]]),
            np.ones(len(centers), dtype=bool),
        ),
    )
    image = _synthetic_dapi()
    limited = _refine_registration(
        reference=_in_memory_reference(image),
        moving=_in_memory_moving(image),
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=_settings(),
        tuning=RegistrationTuning(refinement_min_correlation_gain=0.0),
    )

    assert limited.status == "rejected_exceeds_limits"
    assert "scale change" in limited.reason
    # The limits come first: the candidate was never measured, so no Pearson keys.
    assert "pearson_candidate" not in limited.correction


def test_a_nan_correlation_fails_the_gain_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registration_module, "_masked_log_pearson", lambda *a, **k: float("nan")
    )
    refinement = _refine_identical_images(
        monkeypatch, RegistrationTuning(refinement_min_correlation_gain=-2.0)
    )

    assert refinement.status == "rejected_no_correlation_gain"
    assert "gain undefined" in refinement.reason
    np.testing.assert_array_equal(refinement.matrix, np.eye(3))
    assert math.isnan(refinement.correction["pearson_before"])
    stats = registration_module._jsonable(refinement.correction)
    assert stats["pearson_before"] is None and stats["pearson_candidate"] is None


def test_a_refinement_that_truly_improves_the_overlay_applies_with_the_gain_check() -> (
    None
):
    reference = _synthetic_dapi()
    moving = shift(reference, (3, -2), order=1).astype(np.float32)

    refinement = _refine_registration(
        reference=_in_memory_reference(reference),
        moving=_in_memory_moving(moving),
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=_settings(),
        tuning=RegistrationTuning(refinement_min_correlation_gain=0.0),
    )

    assert refinement.status == "applied"
    assert (
        refinement.correction["pearson_candidate"]
        > refinement.correction["pearson_before"]
    )
    assert refinement.after.pearson_log == refinement.correction["pearson_candidate"]
    assert refinement.before.pearson_log == refinement.correction["pearson_before"]
    assert not np.allclose(refinement.matrix, np.eye(3))


def test_measure_registration_overlay_returns_stats_keys(tmp_path: Path) -> None:
    reference = _synthetic_dapi()
    moving = shift(reference, (3, -2), order=1)

    stats = measure_registration_overlay(
        reference=_in_memory_reference(reference),
        moving=_in_memory_moving(moving.astype(np.float32)),
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=_settings(),
    )

    assert set(stats) == {
        "working_pixel_size_um",
        "n_windows",
        "n_candidate_windows",
        "pearson_log_registration_channel",
        "median_um",
        "p90_um",
        "p99_um",
        "max_um",
    }
    assert stats["working_pixel_size_um"] == 2.0
    assert stats["median_um"] == pytest.approx(math.hypot(3, 2), abs=0.5)
    assert list(tmp_path.iterdir()) == []


def test_measure_registration_overlay_needs_the_channel() -> None:
    reference = _synthetic_dapi()

    with pytest.raises(ValueError, match="Channel 'p62' not found in image 'moving'"):
        measure_registration_overlay(
            reference=_in_memory_reference(reference),
            moving=_in_memory_moving(reference),
            registration_channel="p62",
            matrix=np.eye(3),
            settings=_settings(),
        )


def test_measure_registration_overlay_rejects_no_channel() -> None:
    reference = _synthetic_dapi()

    with pytest.raises(ValueError, match="registration channel"):
        measure_registration_overlay(
            reference=_in_memory_reference(reference),
            moving=_in_memory_moving(reference),
            registration_channel=None,  # type: ignore[arg-type]
            matrix=np.eye(3),
            settings=_settings(),
        )


_NONRIGID_OVERRIDES: dict[str, Any] = {
    "nonrigid_enabled": True,
    "nonrigid_pixel_size_um": 2.0,
    "nonrigid_smoothing_sigma_um": 6.0,
    "nonrigid_max_displacement_um": 12.0,
    "nonrigid_max_cycle_error_um": 3.0,
    "build_viewer_pyramid": False,
}


@pytest.mark.integration
def test_core_matches_run_image_registration(tmp_path: Path) -> None:
    """The core, called directly, reproduces what REGISTER_IMAGES wrote."""
    inputs_a = _write_registration_inputs(tmp_path / "a")
    inputs_b = _write_registration_inputs(tmp_path / "b")
    # tifffile embeds a random OME UUID; everything else must be identical.
    assert _without_uuid(inputs_a["image"]) == _without_uuid(inputs_b["image"])

    config_a = _config(inputs_a, **_NONRIGID_OVERRIDES)
    outputs = registration_module.run_image_registration(config_a)
    summary = json.loads(outputs["post_if_summary"].read_text())
    warped_a = _read_s0(inputs_a["zarr"])

    config_b = _config(inputs_b, **_NONRIGID_OVERRIDES)
    store_path_b = inputs_b["zarr"]
    sdata = sd.read_zarr(store_path_b, selection=("images",))
    reference = reference_grid(sdata, "morphology_focus", dataset_name="P1_XENIUM")
    info = read_ome_tiff_info(inputs_b["image"])
    moving = moving_image_from_ome_tiff(
        info,
        image_key="post_if",
        channel_names=["DAPI", "p62"],
        pixel_size_um=IMAGE_PIXEL_SIZE_UM,
    )
    paths = registration_output_paths(tmp_path / "core_out", "P1_XENIUM", "post_if")
    with spatialdata_write_lock(store_path_b.resolve()):
        outcome = register_onto_reference(
            sdata_obj=sdata,
            zarr_path=store_path_b,
            reference=reference,
            moving=moving,
            output_key="post_if",
            matrix=load_alignment_matrix(inputs_b["explorer_matrix"]),
            registration_channel="DAPI",
            settings=config_b,
            paths=paths,
            dataset_name="P1_XENIUM",
        )
    warped_b = _read_s0(store_path_b)

    assert warped_a.dtype == warped_b.dtype
    assert np.array_equal(warped_a, warped_b)
    assert outcome.final_matrix.tolist() == summary["final_matrix"]
    assert outcome.refinement.status == summary["refinement"]["status"]
    assert outcome.nonrigid.status == summary["nonrigid"]["status"]
    assert outcome.nonrigid.result is not None
    expected_nonrigid = {
        key: value
        for key, value in summary["nonrigid"].items()
        if key not in {"enabled", "algorithm", "status", "reason"}
    }
    assert json.dumps(
        registration_module._jsonable(outcome.nonrigid.result.stats), sort_keys=True
    ) == json.dumps(expected_nonrigid, sort_keys=True)
    assert outcome.overlay_summary()["before"] == summary["residuals_before"]
    assert outcome.overlay_summary()["after"] == summary["residuals_after"]
    nan_aware = {"local_correlation", "cycle_error_px"}
    with (
        np.load(summary["outputs"]["nonrigid_flow"]) as first,
        np.load(paths["nonrigid_flow"]) as second,
    ):
        assert sorted(first.files) == sorted(second.files)
        for name in first.files:
            assert np.array_equal(
                first[name], second[name], equal_nan=name in nan_aware
            ), name
    assert (
        Path(summary["outputs"]["local_shifts"]).read_bytes()
        == paths["local_shifts"].read_bytes()
    )
    assert summary["nonrigid"]["status"] == "applied"


def _read_s0(store: Path) -> np.ndarray:
    return np.asarray(da.from_zarr(str(store / "images" / "post_if" / "s0")).compute())


def _without_uuid(path: Path) -> bytes:
    return re.sub(rb"urn:uuid:[0-9a-f-]{36}", b"urn:uuid:", path.read_bytes())


def _forbid_lock(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("core took the lock")


@pytest.mark.integration
def test_core_never_takes_the_store_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_registration_inputs(tmp_path)
    config = _config(inputs, refine_affine=False, build_viewer_pyramid=False)
    sdata = sd.read_zarr(inputs["zarr"], selection=("images",))
    reference = reference_grid(sdata, "morphology_focus", dataset_name="P1_XENIUM")
    moving = moving_image_from_ome_tiff(
        read_ome_tiff_info(inputs["image"]),
        image_key="post_if",
        channel_names=["DAPI", "p62"],
        pixel_size_um=IMAGE_PIXEL_SIZE_UM,
    )
    monkeypatch.setattr(
        "merxen.image_registration.spatialdata_write_lock", _forbid_lock
    )
    monkeypatch.setattr("merxen.io.spatialdata_io.spatialdata_write_lock", _forbid_lock)

    outcome = register_onto_reference(
        sdata_obj=sdata,
        zarr_path=inputs["zarr"],
        reference=reference,
        moving=moving,
        output_key="post_if",
        matrix=load_alignment_matrix(inputs["explorer_matrix"]),
        registration_channel=None,
        settings=config,
        paths=registration_output_paths(tmp_path / "out", "P1_XENIUM", "post_if"),
        dataset_name="P1_XENIUM",
    )

    assert outcome.warp.output_key == "post_if"
    assert "post_if" in sd.read_zarr(inputs["zarr"], selection=("images",)).images


def test_core_uses_monkeypatched_module_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_materialize(**kwargs: Any) -> ImageWarpResult:
        calls.append(kwargs)
        return ImageWarpResult(
            output_key="x",
            shape_cyx=(1, 64, 64),
            chunks_cyx=(1, 64, 64),
            dtype="uint16",
            channels=("DAPI",),
            tile_count=1,
            nonzero_fraction=1.0,
            tile_seams_validated=True,
        )

    monkeypatch.setattr(
        "merxen.image_registration.materialize_warped_image", fake_materialize
    )
    image = _synthetic_dapi(64)
    reference = _in_memory_reference(image)
    moving = _in_memory_moving(image)
    matrix = np.array([[1.1, 0.2, 5.0], [-0.2, 0.9, -3.0], [0.0, 0.0, 1.0]])

    outcome = register_onto_reference(
        sdata_obj=object(),
        zarr_path=tmp_path / "store.zarr",
        reference=reference,
        moving=moving,
        output_key="out",
        matrix=matrix,
        registration_channel=None,
        settings=_settings(),
        paths=registration_output_paths(tmp_path, "P1", "out"),
        dataset_name="P1",
    )

    assert len(calls) == 1
    assert calls[0]["output_key"] == "out"
    assert calls[0]["output_shape_rc"] == reference.shape_yx
    assert calls[0]["source_image"] is moving.level0
    assert calls[0]["transform"] is reference.transformations["global"]
    assert outcome.refinement.status == "skipped_no_registration_channel"
    assert outcome.nonrigid.status == "disabled"
    assert outcome.flow_yx is None
    assert outcome.flow_working_to_reference is None
    expected = np.linalg.inv(matrix) @ np.array([10.0, 20.0, 1.0])
    assert np.allclose(outcome.pull_mapping(np.array([[10.0, 20.0]])), [expected[:2]])


def test_core_requires_the_qc_paths(tmp_path: Path) -> None:
    image = _synthetic_dapi(64)

    with pytest.raises(ValueError, match="paths is missing"):
        register_onto_reference(
            sdata_obj=object(),
            zarr_path=tmp_path / "store.zarr",
            reference=_in_memory_reference(image),
            moving=_in_memory_moving(image),
            output_key="out",
            matrix=np.eye(3),
            registration_channel=None,
            settings=_settings(),
            paths={"qc_plot": tmp_path / "qc.png"},
            dataset_name="P1",
        )


def test_qc_labels_follow_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figures: list[matplotlib.figure.Figure] = []

    def capture(
        fig: matplotlib.figure.Figure, *_args: object, **_kwargs: object
    ) -> None:
        figures.append(fig)

    monkeypatch.setattr("merxen.image_registration.save_figure", capture)
    reference_image = _synthetic_dapi(256)
    moving_image = shift(reference_image, (2, -1), order=1).astype(np.float32)
    grid = _in_memory_reference(reference_image)
    image = _in_memory_moving(moving_image)
    settings = _settings(**{**_NONRIGID_OVERRIDES, "refine_affine": False})
    refinement = _refine_registration(
        reference=grid,
        moving=image,
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=settings,
        tuning=RegistrationTuning(),
    )
    nonrigid = _refine_nonrigid_registration(
        reference=grid,
        moving=image,
        registration_channel="DAPI",
        matrix=np.eye(3),
        settings=settings,
        tuning=RegistrationTuning(),
        nonrigid_tuning=None,
    )

    def render(matrix: str, reference: str, moving: str) -> tuple[str, str]:
        figures.clear()
        _plot_registration_qc(
            refinement, tmp_path / "qc.png", "title", initial_matrix_label=matrix
        )
        _plot_nonrigid_qc(
            nonrigid,
            tmp_path / "nonrigid.png",
            "title",
            reference_label=reference,
            moving_label=moving,
        )
        legend = figures[0].axes[2].get_legend()
        assert legend is not None
        return legend.get_texts()[0].get_text(), figures[1].axes[0].get_title()

    custom_legend, custom_title = render(
        "Landmark matrix", "Reference DAPI", "Moving DAPI"
    )
    default_legend, default_title = render(
        RegistrationTuning().qc_initial_matrix_label,
        RegistrationTuning().qc_reference_label,
        RegistrationTuning().qc_moving_label,
    )

    assert custom_title == "Affine: Reference DAPI (green) / Moving DAPI (magenta)"
    assert custom_legend.startswith("Landmark matrix (median")
    assert default_title == "Affine: Xenium DAPI (green) / IF DAPI (magenta)"
    assert default_legend.startswith("Explorer matrix (median")


_ROUND_TRIP_MATRIX = np.array(
    [
        [-0.9998476951563913, 0.01745240643728351, 76009.31210000001],
        [-0.01745240643728351, -0.9998476951563913, 54891.812518],
        [0.0, 0.0, 1.0],
    ]
)


def test_write_alignment_matrix_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "matrix.csv"

    returned = write_alignment_matrix(path, _ROUND_TRIP_MATRIX)

    assert returned == path
    assert np.array_equal(load_alignment_matrix(path), _ROUND_TRIP_MATRIX)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert all(len(line.split(",")) == 3 for line in lines)
    assert not [p for p in path.parent.iterdir() if ".tmp-" in p.name]

    other = _ROUND_TRIP_MATRIX.copy()
    other[0, 2] = 1.0 / 3.0
    write_alignment_matrix(path, other)
    assert np.array_equal(load_alignment_matrix(path), other)
    assert not [p for p in path.parent.iterdir() if ".tmp-" in p.name]


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]]),
        np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, float("nan")], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
    ids=["not-affine", "singular", "2x3", "nan"],
)
def test_write_alignment_matrix_rejects_bad_matrices(
    tmp_path: Path, matrix: np.ndarray
) -> None:
    path = tmp_path / "bad.csv"

    with pytest.raises(ValueError, match="bad.csv"):
        write_alignment_matrix(path, matrix)

    assert list(tmp_path.iterdir()) == []


def test_write_alignment_matrix_cleans_up_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "matrix.csv"

    def broken(_path: Path) -> np.ndarray:
        return np.eye(3)

    monkeypatch.setattr("merxen.image_registration.load_alignment_matrix", broken)

    with pytest.raises(ValueError, match="round trip"):
        write_alignment_matrix(path, _ROUND_TRIP_MATRIX)

    assert list(tmp_path.iterdir()) == []


def test_tuning_validation() -> None:
    for quality in (1.5, float("nan")):
        with pytest.raises(ValueError, match="refinement_min_window_quality"):
            RegistrationTuning(refinement_min_window_quality=quality)
    with pytest.raises(ValueError, match="qc_moving_label"):
        RegistrationTuning(qc_moving_label="")
    assert RegistrationTuning().refinement_min_correlation_gain is None
    for gain in (None, -2.0, 0.0, 0, 0.01, 2.0):
        tuning = RegistrationTuning(refinement_min_correlation_gain=gain)
        assert tuning.refinement_min_correlation_gain == gain
    for bad_gain in (float("nan"), float("inf"), -float("inf"), 2.01, -2.01, True, "0"):
        with pytest.raises(ValueError, match="refinement_min_correlation_gain"):
            RegistrationTuning(
                refinement_min_correlation_gain=bad_gain  # type: ignore[arg-type]
            )

    level0 = _levels(_synthetic_dapi(64))[0]
    shapes = ((64, 64),)
    for kwargs in (
        {"pixel_size_um": 0.0},
        {"channel_names": ("DAPI", "extra")},
    ):
        values: dict[str, Any] = {
            "image_key": "moving",
            "channel_names": ("DAPI",),
            "pixel_size_um": 1.0,
            "level_shapes_yx": shapes,
            "open_level": (lambda _index: level0),
            "level0": level0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match="image 'moving'"):
            MovingImage(**values)

    with pytest.raises(ValueError, match="has no levels"):
        ReferenceGrid(
            key="reference",
            levels=[],
            transformations={"global": Identity()},
            coordinate_system="global",
        )
    with pytest.raises(ValueError, match="not in transformations"):
        ReferenceGrid(
            key="reference",
            levels=[level0],
            transformations={"global": Identity()},
            coordinate_system="other",
        )
