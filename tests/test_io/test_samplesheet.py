"""Tests for samplesheet parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen.io.samplesheet import parse_samplesheet, validate_samplesheet


def test_parse_samplesheet_supports_raw_dirs_and_cached_spatialdata(
    tmp_path: Path,
) -> None:
    """The parser should understand the new raw-folder plus cache-path schema."""
    merscope_dir = tmp_path / "merscope_raw"
    merscope_dir.mkdir()
    xenium_dir = tmp_path / "xenium_raw"
    xenium_dir.mkdir()
    csv_path = tmp_path / "samplesheet.csv"
    csv_path.write_text(
        (
            "pair_id,merscope_dir,merscope_spatialdata_path,merscope_z_range,"
            "merscope_channels,xenium_dir,xenium_spatialdata_path,xenium_channels\n"
            f"P1,{merscope_dir},,1-6,\"DAPI,PolyT\",{xenium_dir},,\"DAPI,18S\"\n"
        )
    )

    pairs = parse_samplesheet(csv_path)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.pair_id == "P1"
    assert pair.merscope_dir == merscope_dir
    assert pair.merscope_spatialdata_path is None
    assert pair.merscope_z_range == (1, 6)
    assert pair.merscope_channels == ["DAPI", "PolyT"]
    assert pair.xenium_dir == xenium_dir
    assert pair.xenium_spatialdata_path is None


def test_parse_samplesheet_supports_legacy_merscope_zarr_alias(
    tmp_path: Path,
) -> None:
    """Legacy `merscope_zarr_path` should map onto the new cache-path field."""
    merscope_zarr = tmp_path / "merscope_source.zarr"
    merscope_zarr.mkdir()
    xenium_dir = tmp_path / "xenium_raw"
    xenium_dir.mkdir()
    csv_path = tmp_path / "legacy_samplesheet.csv"
    csv_path.write_text(
        (
            "pair_id,merscope_zarr_path,xenium_dir\n"
            f"P1,{merscope_zarr},{xenium_dir}\n"
        )
    )

    pairs = parse_samplesheet(csv_path)

    assert len(pairs) == 1
    assert pairs[0].merscope_spatialdata_path == merscope_zarr
    assert pairs[0].merscope_dir is None


def test_validate_samplesheet_accepts_existing_cached_zarr_without_raw_dir(
    tmp_path: Path,
) -> None:
    """Validation should pass when an existing cached zarr is supplied alone."""
    merscope_zarr = tmp_path / "merscope_source.zarr"
    merscope_zarr.mkdir()
    xenium_zarr = tmp_path / "xenium_source.zarr"
    xenium_zarr.mkdir()
    csv_path = tmp_path / "samplesheet.csv"
    csv_path.write_text(
        (
            "pair_id,merscope_spatialdata_path,xenium_spatialdata_path\n"
            f"P1,{merscope_zarr},{xenium_zarr}\n"
        )
    )

    pairs = parse_samplesheet(csv_path)

    validate_samplesheet(pairs)


def test_validate_samplesheet_requires_platform_source_or_cache(
    tmp_path: Path,
) -> None:
    """Validation should fail when neither raw folders nor cached zarrs exist."""
    csv_path = tmp_path / "invalid_samplesheet.csv"
    csv_path.write_text("pair_id\nP1\n")

    pairs = parse_samplesheet(csv_path)

    with pytest.raises(FileNotFoundError, match="Provide either merscope_dir"):
        validate_samplesheet(pairs)
