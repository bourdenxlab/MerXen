"""Tests for the pipeline configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from merxen.config import (
    MerscopeBeadRemovalConfig,
    MerscopeBuildConfig,
    SpatialDataBuildConfig,
)


def test_merscope_build_config_has_no_bead_removal_by_default() -> None:
    """Existing build configs must validate unchanged, without the new key."""
    config = MerscopeBuildConfig()

    assert config.bead_removal is None


def test_bead_removal_defaults_apply_when_only_enabled_is_given() -> None:
    config = MerscopeBuildConfig.model_validate({"bead_removal": {"enabled": True}})

    bead = config.bead_removal
    assert bead is not None
    assert bead.enabled is True
    assert bead.method == "psf_subtract"
    assert bead.stains == ["PolyT", "DAPI"]
    assert bead.reference_stain == "PolyT"
    assert bead.detection_plane == 0
    assert bead.halo_px == 256
    assert bead.max_workers == 5
    assert bead.calibration_tiles == 6
    assert bead.calibration_tile_px == 2048
    assert bead.min_offtissue_distance_px == 500
    assert bead.min_ring_prominence == pytest.approx(0.010)
    # The fill is on by default, in hybrid mode.
    assert bead.fill is True
    assert bead.fill_mode == "hybrid"
    assert bead.device == "cpu"


def test_bead_removal_is_disabled_by_default_when_constructed_directly() -> None:
    assert MerscopeBeadRemovalConfig().enabled is False


def test_bead_removal_rejects_unknown_nested_key() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MerscopeBuildConfig.model_validate(
            {"bead_removal": {"enabled": True, "halo": 300}}
        )


def test_bead_removal_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        MerscopeBeadRemovalConfig.model_validate({"method": "median_filter"})


def test_bead_removal_rejects_reference_stain_outside_stains() -> None:
    with pytest.raises(ValidationError, match="reference_stain"):
        MerscopeBeadRemovalConfig(stains=["DAPI"], reference_stain="PolyT")


def test_bead_removal_accepts_custom_stains_with_matching_reference() -> None:
    config = MerscopeBeadRemovalConfig(stains=["DAPI"], reference_stain="DAPI")

    assert config.stains == ["DAPI"]


@pytest.mark.parametrize("detection_plane", [1, 3, -1])
def test_bead_removal_rejects_detection_plane_other_than_zero(
    detection_plane: int,
) -> None:
    with pytest.raises(ValidationError, match="detection_plane"):
        MerscopeBeadRemovalConfig(detection_plane=detection_plane)


@pytest.mark.parametrize("halo_px", [0, 240, 255, 257, 265])
def test_bead_removal_rejects_halo_below_256_or_off_grid(halo_px: int) -> None:
    with pytest.raises(ValidationError, match="halo_px"):
        MerscopeBeadRemovalConfig(halo_px=halo_px)


@pytest.mark.parametrize("halo_px", [256, 272, 512])
def test_bead_removal_accepts_halo_on_the_16_px_grid(halo_px: int) -> None:
    assert MerscopeBeadRemovalConfig(halo_px=halo_px).halo_px == halo_px


@pytest.mark.parametrize("max_workers", [0, -2])
def test_bead_removal_rejects_non_positive_max_workers(max_workers: int) -> None:
    with pytest.raises(ValidationError, match="max_workers"):
        MerscopeBeadRemovalConfig(max_workers=max_workers)


def test_bead_removal_fill_requires_enabled_when_given_explicitly() -> None:
    with pytest.raises(ValidationError, match="fill requires enabled"):
        MerscopeBeadRemovalConfig(enabled=False, fill=True)


def test_bead_removal_fill_default_does_not_need_enabled() -> None:
    """The default config is valid with removal off although the fill defaults on."""
    config = MerscopeBeadRemovalConfig()

    assert config.enabled is False
    assert config.fill is True
    assert MerscopeBeadRemovalConfig(enabled=False, fill=False).fill is False


def test_bead_removal_fill_is_accepted_when_enabled() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=True)

    assert config.fill is True


def test_bead_removal_fill_and_its_mode_default_to_hybrid_on() -> None:
    assert MerscopeBeadRemovalConfig().fill_mode == "hybrid"
    config = MerscopeBeadRemovalConfig(enabled=True)
    assert config.fill is True
    assert config.fill_mode == "hybrid"
    assert MerscopeBeadRemovalConfig(enabled=True, fill=True).fill_mode == "hybrid"


def test_bead_removal_fill_false_switches_the_fill_off() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=False)

    assert config.fill is False
    # The default mode is ignored with the fill off, not refused.
    assert config.fill_mode == "hybrid"


@pytest.mark.parametrize("mode", ["flat", "hybrid"])
def test_bead_removal_fill_mode_requires_the_fill_when_given_explicitly(
    mode: str,
) -> None:
    with pytest.raises(ValidationError, match=f"fill_mode '{mode}' requires fill"):
        MerscopeBeadRemovalConfig(
            enabled=True,
            fill=False,
            fill_mode=mode,  # type: ignore[arg-type]
        )


def test_bead_removal_accepts_the_flat_fill_mode() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=True, fill_mode="flat")

    assert config.fill_mode == "flat"
    assert MerscopeBeadRemovalConfig(enabled=True, fill_mode="flat").fill is True


def test_bead_removal_rejects_an_unknown_fill_mode() -> None:
    with pytest.raises(ValidationError, match="fill_mode"):
        MerscopeBeadRemovalConfig(
            enabled=True,
            fill=True,
            fill_mode="aggressive",  # type: ignore[arg-type]
        )


def test_spatialdata_build_config_loads_bead_removal_from_json(
    tmp_path: Path,
) -> None:
    """The JSON shape Nextflow writes must round-trip into the nested model."""
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "z_layers": [1, 2, 3],
            "bead_removal": {"enabled": True, "fill": True},
        },
    }

    config = SpatialDataBuildConfig.model_validate(raw)

    assert config.merscope.z_layers == [1, 2, 3]
    assert config.merscope.bead_removal is not None
    assert config.merscope.bead_removal.fill is True


def test_spatialdata_build_config_loads_the_fill_settings_the_workflow_writes(
    tmp_path: Path,
) -> None:
    """The workflow writes ``fill`` and, with the fill on, ``fill_mode``."""
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "z_layers": [1, 2, 3],
            "bead_removal": {"enabled": True, "fill": True, "fill_mode": "hybrid"},
        },
    }

    config = SpatialDataBuildConfig.model_validate(raw)

    assert config.merscope.bead_removal is not None
    assert config.merscope.bead_removal.fill_mode == "hybrid"


def test_spatialdata_build_config_loads_a_bead_build_without_the_fill(
    tmp_path: Path,
) -> None:
    """``--merscope_bead_removal_fill false`` writes ``fill: false`` and no mode."""
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "z_layers": [1, 2, 3],
            "bead_removal": {"enabled": True, "fill": False},
        },
    }

    config = SpatialDataBuildConfig.model_validate(raw)

    assert config.merscope.bead_removal is not None
    assert config.merscope.bead_removal.fill is False


def test_spatialdata_build_config_refuses_a_fill_mode_without_the_fill(
    tmp_path: Path,
) -> None:
    raw = {
        "dataset_name": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "input_path": str(tmp_path / "raw"),
        "output_path": "spatialdata_out/source_spatialdata.zarr",
        "merscope": {
            "bead_removal": {"enabled": True, "fill": False, "fill_mode": "hybrid"},
        },
    }

    with pytest.raises(ValidationError, match="requires fill=True"):
        SpatialDataBuildConfig.model_validate(raw)
