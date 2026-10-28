"""Presets, the override grammar and settings resolution."""

from __future__ import annotations

import subprocess
import sys

import pytest

from merxen.config import ImageRegistrationConfig, RegistrationSettings
from merxen.image_registration import RegistrationTuning
from merxen.manual_registration.config import ManualRegistrationSettings
from merxen.manual_registration.presets import (
    FORBIDDEN_OVERRIDES,
    PRESET_VERSIONS,
    PRESETS,
    parse_settings_overrides,
    registration_tuning,
    resolve_settings,
)
from merxen.nonrigid_registration import NonrigidTuning

# The serial_section table is unchanged since v1: v2 only inherits the new
# refinement_min_correlation_gain default, so the key must not appear here.
SERIAL_V1 = {
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


def _resolve(preset: str, **kwargs: str) -> ManualRegistrationSettings:
    return resolve_settings(
        preset, reference_pixel_size_um=0.2125, reference_channel="DAPI", **kwargs
    )


def test_preset_tables_are_pinned() -> None:
    assert PRESETS["same_section"] == {}
    assert set(PRESET_VERSIONS) == set(PRESETS)
    assert PRESET_VERSIONS == {
        "same_section": "same_section_v2",
        "serial_section": "serial_section_v2",
    }
    fields = set(ManualRegistrationSettings.model_fields)
    assert set(PRESETS["serial_section"]) <= fields
    assert PRESETS["serial_section"] == SERIAL_V1
    assert "refinement_min_correlation_gain" not in PRESETS["serial_section"]
    assert {"reference_pixel_size_um", "reference_channel"} == FORBIDDEN_OVERRIDES


def test_same_section_equals_register_images_defaults() -> None:
    settings = _resolve("same_section")
    for name in RegistrationSettings.model_fields:
        expected = ImageRegistrationConfig.model_fields[name].default
        if name == "nonrigid_enabled":
            assert settings.nonrigid_enabled is True
            assert expected is False
        elif name == "reference_pixel_size_um":
            assert settings.reference_pixel_size_um == 0.2125
        else:
            assert getattr(settings, name) == expected, name


def test_same_section_differs_from_register_images_in_the_correlation_gain() -> None:
    # REGISTER_IMAGES has no such setting (adding it to RegistrationSettings would
    # change the provenance of every registered image), so it applies every
    # refinement inside the limits.
    assert "refinement_min_correlation_gain" not in RegistrationSettings.model_fields
    assert "refinement_min_correlation_gain" in ManualRegistrationSettings.model_fields
    assert RegistrationTuning().refinement_min_correlation_gain is None


@pytest.mark.parametrize("preset", ["same_section", "serial_section"])
def test_both_presets_carry_the_correlation_gain_default(preset: str) -> None:
    assert _resolve(preset).refinement_min_correlation_gain == 0.0
    tuning = registration_tuning(_resolve(preset))
    assert tuning.refinement_min_correlation_gain == 0.0


def test_registration_tuning_passes_the_correlation_gain_through() -> None:
    for text, expected in (
        ("0.01", 0.01),
        ("-0.5", -0.5),
        ("2", 2.0),
        # -2 is the lowest value the settings accept and disables the check.
        ("-2", None),
        ("-2.0", None),
    ):
        settings = _resolve(
            "serial_section",
            row_settings=f"refinement_min_correlation_gain={text}",
        )
        tuning = registration_tuning(settings)
        assert tuning.refinement_min_correlation_gain == expected, text


def test_correlation_gain_overrides_follow_the_scope_precedence() -> None:
    key = "refinement_min_correlation_gain"
    glob = f"{key}=0.1;serial_section:{key}=0.2"
    assert (
        _resolve("same_section", global_settings=glob).refinement_min_correlation_gain
        == 0.1
    )
    assert (
        _resolve("serial_section", global_settings=glob).refinement_min_correlation_gain
        == 0.2
    )
    row = f"same_section:{key}=-2"
    settings = _resolve("same_section", global_settings=glob, row_settings=row)
    assert settings.refinement_min_correlation_gain == -2.0
    assert registration_tuning(settings).refinement_min_correlation_gain is None


@pytest.mark.parametrize("value", ["-2.5", "2.5", "nan", "inf", "none", "off", "abc"])
def test_correlation_gain_override_errors(value: str) -> None:
    # The grammar has no None: the check is disabled with -2, anything outside
    # [-2, 2] or not a finite number is refused.
    with pytest.raises(ValueError, match="refinement_min_correlation_gain") as excinfo:
        _resolve(
            "same_section", row_settings=f"refinement_min_correlation_gain={value}"
        )
    assert str(excinfo.value).startswith("registration settings")


def test_precedence_of_override_scopes() -> None:
    key = "nonrigid_smoothing_sigma_um"

    def value(**kwargs: str) -> float:
        return getattr(_resolve("serial_section", **kwargs), key)

    assert value() == 100.0
    assert value(global_settings=f"{key}=110") == 110.0
    glob = f"{key}=110;serial_section:{key}=120"
    assert value(global_settings=glob) == 120.0
    assert value(global_settings=glob, row_settings=f"{key}=130") == 130.0
    row = f"{key}=130;serial_section:{key}=140"
    assert value(global_settings=glob, row_settings=row) == 140.0
    # Another preset's prefix is validated but not applied.
    assert value(row_settings=f"same_section:{key}=999") == 100.0


def test_values_are_coerced_by_pydantic() -> None:
    settings = _resolve(
        "same_section",
        row_settings="nucleus_qc=true;min_refinement_windows=15;landmark_outlier_policy=keep",
    )
    assert settings.nucleus_qc is True
    assert settings.min_refinement_windows == 15
    assert settings.landmark_outlier_policy == "keep"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("foo=1", "unknown key"),
        ("reference_pixel_size_um=0.3", "samplesheet"),
        ("serial_section:reference_channel=X", "samplesheet"),
        ("nonrigid_smoothing_sigma_um", "key=value"),
        ("bogus:refine_affine=false", "unknown preset"),
        ("a=1;a=2", "twice"),
        ("min_refinement_windows=2", "invalid"),
        ("=1", "empty key"),
        ("a=", "empty key or value"),
        ("A=1", "invalid key"),
    ],
)
def test_override_errors(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match) as excinfo:
        _resolve("same_section", row_settings=text)
    assert str(excinfo.value).startswith("registration settings")


def test_global_settings_errors_name_their_source() -> None:
    with pytest.raises(ValueError, match="global settings"):
        _resolve("same_section", global_settings="foo=1")


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="preset"):
        _resolve("sideways")


def test_parse_settings_overrides_strips_and_scopes() -> None:
    assert parse_settings_overrides(" a=1 ; serial_section:b = x ;") == {
        None: {"a": "1"},
        "serial_section": {"b": "x"},
    }
    assert parse_settings_overrides("") == {}
    assert parse_settings_overrides("  ;  ") == {}


def test_registration_tuning_follows_settings() -> None:
    tuning = registration_tuning(_resolve("serial_section"))
    assert tuning.refinement_min_window_quality == 0.2
    assert tuning.refinement_require_moving_tissue is True
    assert tuning.nonrigid == NonrigidTuning(120.0, 80.0, False, 1500.0)
    assert tuning.qc_reference_label == "Reference DAPI"
    assert tuning.qc_moving_label == "Moving DAPI"
    assert tuning.qc_initial_matrix_label == "Landmark matrix"
    same = registration_tuning(_resolve("same_section"))
    assert same.nonrigid == NonrigidTuning()


@pytest.mark.parametrize("value", ["1", "16", "32"])
def test_crop_snap_px_below_the_pyramid_limit_is_refused_up_front(value: str) -> None:
    # Powers of two pass the settings model, but the pyramid levels are cropped at
    # factors dividing 64, so a finer snap would fail late inside the core.
    with pytest.raises(ValueError, match="registration settings: crop_snap_px") as exc:
        _resolve("same_section", row_settings=f"crop_snap_px={value}")
    assert "at least 64" in str(exc.value)
    with pytest.raises(ValueError, match="crop_snap_px"):
        _resolve("serial_section", global_settings=f"crop_snap_px={value}")


@pytest.mark.parametrize("value", [64, 128, 512])
def test_crop_snap_px_of_64_or_more_is_allowed(value: int) -> None:
    settings = _resolve("same_section", row_settings=f"crop_snap_px={value}")
    assert settings.crop_snap_px == value
    assert _resolve("same_section").crop_snap_px == 64


def test_presets_import_stays_light() -> None:
    # The module docstring promises no SpatialData or plotting import.
    code = (
        "import sys\n"
        "import merxen.manual_registration.presets\n"
        "names = ('spatialdata', 'matplotlib.pyplot')\n"
        "heavy = [m for m in names if m in sys.modules]\n"
        "print(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
