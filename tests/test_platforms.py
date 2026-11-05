"""Tests for the platform registry (``merxen.platforms``)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import get_args

import pytest

from merxen.config import DatasetConfig, SpatialDataBuildConfig
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME
from merxen.platforms import (
    ORIGINAL_CELL_SHAPE_KEYS,
    ORIGINAL_NUCLEUS_SHAPE_KEYS,
    PLATFORM_SPECS,
    PlatformName,
    PlatformSpec,
    UnknownPlatformError,
    find_platform_spec,
    find_um_per_px,
    platform_spec,
    spec_file_pixel_size,
    supported_platforms,
)


def test_registry_holds_merscope_xenium_then_atera() -> None:
    """The registry order is the order every caller iterates in."""
    assert tuple(PLATFORM_SPECS) == ("MERSCOPE", "XENIUM", "ATERA")
    assert supported_platforms() == ("MERSCOPE", "XENIUM", "ATERA")
    for name, spec in PLATFORM_SPECS.items():
        assert spec.name == name


def test_registry_is_read_only() -> None:
    """Neither the mapping nor an entry can be changed at runtime."""
    with pytest.raises(TypeError):
        PLATFORM_SPECS["ATERA"] = PLATFORM_SPECS["XENIUM"]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        PLATFORM_SPECS["XENIUM"].name = "ATERA"  # type: ignore[misc]


def test_merscope_fields_equal_the_literals_they_replace() -> None:
    """MERSCOPE values match the former literals of the call sites."""
    spec = platform_spec("MERSCOPE")
    assert spec == PlatformSpec(
        name="MERSCOPE",
        family="MERSCOPE",
        native_image_key="MERSCOPE_z_projection",
        original_cell_shape_key="merscope_cell_boundaries",
        original_nucleus_shape_key=None,
        default_channels=("DAPI", "PolyT"),
        quality_column_alias="transcript_score",
        fallback_um_per_px=0.108,
    )
    assert spec.native_image_key == MERSCOPE_ZPROJ_IMAGE_NAME
    assert spec.original_shape_keys == ("merscope_cell_boundaries",)


def test_xenium_fields_equal_the_literals_they_replace() -> None:
    """XENIUM values match the former literals of the call sites."""
    spec = platform_spec("XENIUM")
    assert spec == PlatformSpec(
        name="XENIUM",
        family="10X",
        native_image_key="morphology_focus",
        original_cell_shape_key="xenium_cell_boundaries",
        original_nucleus_shape_key="xenium_nucleus",
        default_channels=("DAPI", "18S"),
        quality_column_alias=None,
        fallback_um_per_px=0.2125,
        nucleus_ids_are_cell_ids=True,
        spec_sidecar_names=("experiment.xenium", "specs.json", "specs/specs.json"),
        mask_pixel_convention="center",
    )
    assert spec.original_shape_keys == ("xenium_cell_boundaries", "xenium_nucleus")


def test_atera_fields() -> None:
    """ATERA is 10x geometry with per-nucleus IDs and corner-convention masks."""
    spec = platform_spec("ATERA")
    assert spec == PlatformSpec(
        name="ATERA",
        family="10X",
        native_image_key="morphology_focus",
        original_cell_shape_key="atera_cell_boundaries",
        original_nucleus_shape_key="atera_nucleus",
        default_channels=("DAPI", "18S"),
        quality_column_alias=None,
        fallback_um_per_px=0.21249222,
        nucleus_ids_are_cell_ids=False,
        spec_sidecar_names=("atera/experiment.spatial", "experiment.spatial"),
        mask_pixel_convention="corner",
    )
    assert spec.original_shape_keys == ("atera_cell_boundaries", "atera_nucleus")


def test_merscope_keeps_the_field_defaults() -> None:
    """MERSCOPE has no spec sidecar and no separate nucleus namespace."""
    spec = platform_spec("MERSCOPE")
    assert spec.nucleus_ids_are_cell_ids
    assert spec.spec_sidecar_names == ()
    assert spec.mask_pixel_convention == "center"


def test_original_shape_key_sets() -> None:
    """The derived key sets hold exactly the registered original layers."""
    assert (
        frozenset(
            {
                "merscope_cell_boundaries",
                "xenium_cell_boundaries",
                "atera_cell_boundaries",
            }
        )
        == ORIGINAL_CELL_SHAPE_KEYS
    )
    assert frozenset({"xenium_nucleus", "atera_nucleus"}) == ORIGINAL_NUCLEUS_SHAPE_KEYS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MERSCOPE", "MERSCOPE"),
        ("merscope", "MERSCOPE"),
        (" Xenium ", "XENIUM"),
        ("XENIUM", "XENIUM"),
        ("atera", "ATERA"),
    ],
)
def test_lookup_ignores_case_and_whitespace(raw: str, expected: str) -> None:
    """Lookups normalise like the former ``str(p).upper()`` comparisons."""
    assert platform_spec(raw).name == expected
    found = find_platform_spec(raw)
    assert found is not None and found.name == expected


@pytest.mark.parametrize("raw", ["COSMX", "", "XENIUM_aligned", "10X", "unknown"])
def test_unknown_platform_raises_naming_the_supported_values(raw: str) -> None:
    """An unknown platform raises instead of falling into an else-arm."""
    with pytest.raises(UnknownPlatformError) as excinfo:
        platform_spec(raw)
    message = str(excinfo.value)
    assert repr(raw) in message
    assert "Supported: MERSCOPE, XENIUM, ATERA" in message
    assert find_platform_spec(raw) is None


def test_unknown_platform_error_is_a_value_error_not_a_missing_file() -> None:
    """Fallbacks that catch FileNotFoundError must not swallow the error."""
    assert issubclass(UnknownPlatformError, ValueError)
    assert not issubclass(UnknownPlatformError, FileNotFoundError)


def test_find_platform_spec_accepts_none() -> None:
    """The tolerant lookup returns None for a missing label."""
    assert find_platform_spec(None) is None


def test_platform_name_alias_matches_the_config_literals() -> None:
    """The build config takes every platform; downstream Literals are staged.

    Plan §3.2: a ``platform`` Literal is widened to ATERA only when its stage
    handles it, so segmentation (``DatasetConfig``) still lists MERSCOPE and
    XENIUM only.
    """
    names = get_args(PlatformName)
    assert names == tuple(PLATFORM_SPECS)
    build = SpatialDataBuildConfig.model_fields["platform"].annotation
    assert get_args(build) == names
    dataset = DatasetConfig.model_fields["platform"].annotation
    assert get_args(dataset) == ("MERSCOPE", "XENIUM")


def _write_spec(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def test_find_um_per_px_reads_the_atera_sidecar_first(tmp_path: Path) -> None:
    """A built store holds the manifest under ``atera/``; a raw folder at the root."""
    store = tmp_path / "store.zarr"
    _write_spec(store / "atera" / "experiment.spatial", {"pixel_size": 0.21249222})
    _write_spec(store / "experiment.spatial", {"pixel_size": 0.5})
    assert find_um_per_px(store, "ATERA") == 0.21249222
    raw = tmp_path / "raw"
    _write_spec(raw / "experiment.spatial", {"pixel_size": 0.25})
    assert find_um_per_px(raw, "atera") == 0.25


def test_find_um_per_px_follows_the_xenium_sidecar_order(tmp_path: Path) -> None:
    _write_spec(tmp_path / "specs" / "specs.json", {"pixel_size": 0.3})
    assert find_um_per_px(tmp_path, "XENIUM") == 0.3
    _write_spec(tmp_path / "specs.json", {"microns_per_pixel": 0.25})
    assert find_um_per_px(tmp_path, "XENIUM") == 0.25
    _write_spec(tmp_path / "experiment.xenium", {"pixel_size": 0.2125})
    assert find_um_per_px(tmp_path, "XENIUM") == 0.2125


def test_find_um_per_px_raises_without_a_spec(tmp_path: Path) -> None:
    _write_spec(tmp_path / "experiment.spatial", {"no_pixel_size": 1})
    with pytest.raises(FileNotFoundError, match="ATERA pixel size"):
        find_um_per_px(tmp_path, "ATERA")
    with pytest.raises(FileNotFoundError, match="no spec file"):
        find_um_per_px(tmp_path, "MERSCOPE")
    with pytest.raises(UnknownPlatformError):
        find_um_per_px(tmp_path, "COSMX")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"pixel_size": 0.5, "microns_per_pixel": 0.25}', 0.5),
        ('{"microns_per_pixel": "0.25"}', 0.25),
        ('{"pixel_size": "wide"}', None),
        ("[1, 2]", None),
        ("not json", None),
    ],
)
def test_spec_file_pixel_size(
    tmp_path: Path, content: str, expected: float | None
) -> None:
    path = tmp_path / "spec.json"
    path.write_text(content)
    assert spec_file_pixel_size(path) == expected
    assert spec_file_pixel_size(tmp_path / "missing.json") is None
