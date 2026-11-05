"""Tests for the platform registry (``merxen.platforms``)."""

from __future__ import annotations

import dataclasses
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
    platform_spec,
    supported_platforms,
)


def test_registry_holds_merscope_then_xenium() -> None:
    """The registry order is the order every caller iterates in."""
    assert tuple(PLATFORM_SPECS) == ("MERSCOPE", "XENIUM")
    assert supported_platforms() == ("MERSCOPE", "XENIUM")
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
    )
    assert spec.original_shape_keys == ("xenium_cell_boundaries", "xenium_nucleus")


def test_original_shape_key_sets() -> None:
    """The derived key sets hold exactly the registered original layers."""
    assert (
        frozenset({"merscope_cell_boundaries", "xenium_cell_boundaries"})
        == ORIGINAL_CELL_SHAPE_KEYS
    )
    assert frozenset({"xenium_nucleus"}) == ORIGINAL_NUCLEUS_SHAPE_KEYS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MERSCOPE", "MERSCOPE"),
        ("merscope", "MERSCOPE"),
        (" Xenium ", "XENIUM"),
        ("XENIUM", "XENIUM"),
    ],
)
def test_lookup_ignores_case_and_whitespace(raw: str, expected: str) -> None:
    """Lookups normalise like the former ``str(p).upper()`` comparisons."""
    assert platform_spec(raw).name == expected
    found = find_platform_spec(raw)
    assert found is not None and found.name == expected


@pytest.mark.parametrize("raw", ["ATERA", "", "XENIUM_aligned", "10X", "unknown"])
def test_unknown_platform_raises_naming_the_supported_values(raw: str) -> None:
    """An unknown platform raises instead of falling into an else-arm."""
    with pytest.raises(UnknownPlatformError) as excinfo:
        platform_spec(raw)
    message = str(excinfo.value)
    assert repr(raw) in message
    assert "Supported: MERSCOPE, XENIUM" in message
    assert find_platform_spec(raw) is None


def test_unknown_platform_error_is_a_value_error_not_a_missing_file() -> None:
    """Fallbacks that catch FileNotFoundError must not swallow the error."""
    assert issubclass(UnknownPlatformError, ValueError)
    assert not issubclass(UnknownPlatformError, FileNotFoundError)


def test_find_platform_spec_accepts_none() -> None:
    """The tolerant lookup returns None for a missing label."""
    assert find_platform_spec(None) is None


def test_platform_name_alias_matches_the_config_literals() -> None:
    """The alias, the registry and the unchanged config Literals agree."""
    names = get_args(PlatformName)
    assert names == tuple(PLATFORM_SPECS)
    for model in (SpatialDataBuildConfig, DatasetConfig):
        annotation = model.model_fields["platform"].annotation
        assert get_args(annotation) == names
