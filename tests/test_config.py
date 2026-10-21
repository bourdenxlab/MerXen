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
    assert bead.fill is False


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


def test_bead_removal_fill_requires_enabled() -> None:
    with pytest.raises(ValidationError, match="fill requires enabled"):
        MerscopeBeadRemovalConfig(enabled=False, fill=True)


def test_bead_removal_fill_is_accepted_when_enabled() -> None:
    config = MerscopeBeadRemovalConfig(enabled=True, fill=True)

    assert config.fill is True


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
