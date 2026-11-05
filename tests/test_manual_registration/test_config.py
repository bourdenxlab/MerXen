"""Tests for the manual-registration run configuration models."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from merxen.config import ImageRegistrationConfig, RegistrationSettings
from merxen.manual_registration.config import (
    PRESET_NAMES,
    ManualRegistrationCompareConfig,
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
    ManualRegistrationRunConfig,
    ManualRegistrationSettings,
)

# serial_section v1 values (provisional); the real preset table lives with the
# register step, this copy only proves they validate against the settings model.
SERIAL_V1: dict[str, Any] = {
    "matrix_scale_tolerance": 0.05,
    "registration_pixel_size_um": 6.0,
    "refinement_window_um": 1200.0,
    "min_refinement_windows": 12,
    "max_refinement_shift_um": 200.0,
    "max_refinement_rotation_deg": 5.0,
    "max_refinement_scale_change": 0.05,
    "refinement_min_window_quality": 0.2,
    "refinement_require_moving_tissue": True,
    "nonrigid_enabled": True,
    "nonrigid_pixel_size_um": 12.0,
    "nonrigid_smoothing_sigma_um": 100.0,
    "nonrigid_max_displacement_um": 200.0,
    "nonrigid_max_cycle_error_um": 30.0,
    "nonrigid_min_local_correlation": 0.15,
    "nonrigid_min_jacobian": 0.67,
    "nonrigid_max_jacobian": 1.5,
    "nonrigid_highpass_sigma_um": 120.0,
    "nonrigid_local_correlation_sigma_um": 80.0,
    "nonrigid_local_residual_tiles": False,
    "nonrigid_local_residual_tile_um": 1500.0,
    "landmark_outlier_floor_um": 30.0,
    "nucleus_qc": True,
}


def _image(**overrides: Any) -> dict[str, Any]:
    return {"path": "/path/to/image.ome.tif", **overrides}


def _prepare(**overrides: Any) -> dict[str, Any]:
    return {
        "dataset_name": "LCM1",
        "reference": _image(),
        "moving": _image(),
        "prepared_dir": "/path/to/prepared",
        "output_dir": "prepare_out",
        **overrides,
    }


def _run(**overrides: Any) -> dict[str, Any]:
    return {
        "dataset_name": "LCM1",
        "preset": "serial_section",
        "prepared_store_path": "/path/to/LCM1_prepared.zarr",
        "result_dir": "/path/to/registered/serial_section",
        "output_dir": "register_out",
        **overrides,
    }


def test_settings_defaults_equal_image_registration_defaults() -> None:
    manual = ManualRegistrationSettings()
    registration = ImageRegistrationConfig.model_construct()
    shared = set(RegistrationSettings.model_fields)
    assert shared <= set(ManualRegistrationSettings.model_fields)
    for name in shared - {"nonrigid_enabled"}:
        assert getattr(manual, name) == getattr(registration, name), name
    assert manual.nonrigid_enabled is True
    assert registration.nonrigid_enabled is False


def test_new_setting_defaults() -> None:
    settings = ManualRegistrationSettings()
    assert settings.refinement_min_window_quality == 0.3
    assert settings.refinement_require_moving_tissue is False
    assert settings.refinement_min_correlation_gain == 0.0
    assert settings.nonrigid_highpass_sigma_um == 25.0
    assert settings.nonrigid_local_correlation_sigma_um == 15.0
    assert settings.nonrigid_local_residual_tiles is True
    assert settings.nonrigid_local_residual_tile_um == 220.0
    assert settings.landmark_min_pairs == 8
    assert settings.landmark_min_used_pairs == 6
    assert settings.landmark_outlier_floor_um == 10.0
    assert settings.landmark_outlier_k == 3.0
    assert settings.landmark_max_outlier_fraction == 0.25
    assert settings.landmark_outlier_policy == "exclude"
    assert settings.landmarks_outside_crop == "use"
    assert settings.landmark_min_spread_ratio == 0.05
    assert settings.landmark_reflection_check_ratio == 0.5
    assert settings.crop_snap_px == 64
    assert settings.crop_min_size_px == 1024
    assert settings.tissue_qc_pixel_size_um == 10.0
    assert settings.tissue_qc_sigma_um == 20.0
    assert settings.tissue_qc_highpass_sigma_um == 200.0
    assert settings.nucleus_qc is False
    assert settings.nucleus_qc_pixel_size_um == 1.7
    assert settings.nucleus_qc_window_um == 200.0


@pytest.mark.parametrize("value", [-2.0, -0.1, 0.0, 0.05, 2.0, "0.1", "-2"])
def test_correlation_gain_accepts_numbers_in_range(value: float | str) -> None:
    settings = ManualRegistrationSettings(refinement_min_correlation_gain=value)  # type: ignore[arg-type]
    assert settings.refinement_min_correlation_gain == float(value)


@pytest.mark.parametrize("value", [-2.01, 2.01, float("nan"), float("inf"), "none"])
def test_correlation_gain_rejects_other_values(value: float | str) -> None:
    with pytest.raises(ValidationError, match="refinement_min_correlation_gain"):
        ManualRegistrationSettings(refinement_min_correlation_gain=value)  # type: ignore[arg-type]


def test_settings_forbid_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ManualRegistrationSettings.model_validate({"unknown_knob": 1})


@pytest.mark.parametrize("value", [1, 64, 128])
def test_crop_snap_px_accepts_powers_of_two(value: int) -> None:
    assert ManualRegistrationSettings(crop_snap_px=value).crop_snap_px == value


def test_crop_snap_px_rejects_other_values() -> None:
    with pytest.raises(ValidationError, match="power of two"):
        ManualRegistrationSettings(crop_snap_px=48)


def test_settings_cross_field_checks() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        ManualRegistrationSettings(landmark_min_used_pairs=9, landmark_min_pairs=8)
    with pytest.raises(ValidationError):
        ManualRegistrationSettings(landmark_max_outlier_fraction=0.5)


def test_serial_section_v1_values_validate() -> None:
    settings = ManualRegistrationSettings(**SERIAL_V1)
    assert settings.landmark_outlier_floor_um == 30.0
    assert settings.nucleus_qc is True


def test_image_input_validation() -> None:
    for bad in ([], ["a", "a"], ["a", ""]):
        with pytest.raises(ValidationError):
            ManualRegistrationImageInput.model_validate(_image(channel_names=bad))
    stripped = ManualRegistrationImageInput.model_validate(
        _image(channel_names=[" DAPI ", "AF568"])
    )
    assert stripped.channel_names == ["DAPI", "AF568"]
    with pytest.raises(ValidationError, match="registration_channel must not be blank"):
        ManualRegistrationImageInput.model_validate(_image(registration_channel="  "))
    with pytest.raises(ValidationError, match="scene must not be blank"):
        ManualRegistrationImageInput.model_validate(_image(scene=" "))
    with pytest.raises(ValidationError):
        ManualRegistrationImageInput.model_validate(_image(pixel_size_um=0))
    with pytest.raises(ValidationError):
        ManualRegistrationImageInput.model_validate(_image(unknown=1))
    assert ManualRegistrationImageInput.model_validate(_image(scene=" S1 ")).scene == (
        "S1"
    )


def test_prepare_config_validation() -> None:
    config = ManualRegistrationPrepareConfig.model_validate(_prepare())
    assert config.store_path == Path("/path/to/prepared") / "LCM1_prepared.zarr"
    with pytest.raises(ValidationError):
        ManualRegistrationPrepareConfig.model_validate(_prepare(dataset_name="bad id"))
    with pytest.raises(ValidationError, match="multiple of tile_size"):
        ManualRegistrationPrepareConfig.model_validate(_prepare(band_height=3000))
    with pytest.raises(ValidationError, match="multiple of 16"):
        ManualRegistrationPrepareConfig.model_validate(_prepare(tile_size=1000))


def test_run_config_names() -> None:
    config = ManualRegistrationRunConfig.model_validate(_run())
    assert config.run_name == "serial_section"
    assert config.run_id == "LCM1_serial_section"
    assert config.result_store_path == (
        Path("/path/to/registered/serial_section")
        / "LCM1_serial_section_registered.zarr"
    )
    labelled = ManualRegistrationRunConfig.model_validate(_run(run_label="try2"))
    assert labelled.run_name == "serial_section_try2"
    assert labelled.run_id == "LCM1_serial_section_try2"
    same = ManualRegistrationRunConfig.model_validate(_run(preset="same_section"))
    assert same.run_id != config.run_id


def test_run_config_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        ManualRegistrationRunConfig.model_validate(_run(preset="other"))
    with pytest.raises(ValidationError):
        ManualRegistrationRunConfig.model_validate(_run(run_label="bad label"))
    assert set(PRESET_NAMES) == {"same_section", "serial_section"}


def test_compare_config_needs_two_summaries() -> None:
    payload = {
        "dataset_name": "LCM1",
        "output_dir": "/path/to/registered",
        "task_output_dir": "compare_out",
    }
    with pytest.raises(ValidationError):
        ManualRegistrationCompareConfig.model_validate(
            {**payload, "summary_paths": ["/path/to/a.json"]}
        )
    config = ManualRegistrationCompareConfig.model_validate(
        {**payload, "summary_paths": ["/path/to/a.json", "/path/to/b.json"]}
    )
    assert len(config.summary_paths) == 2


def test_geometry_and_fit_import_neither_zarr_nor_spatialdata() -> None:
    code = (
        "import sys, merxen.manual_registration.geometry, "
        "merxen.manual_registration.landmark_fit; "
        "assert 'zarr' not in sys.modules and 'spatialdata' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
