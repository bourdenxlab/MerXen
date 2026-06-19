"""Tests for samplesheet parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen.io.samplesheet import (
    parse_samplesheet,
    required_platforms_for_mode,
    validate_samplesheet,
)


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
        "pair_id,merscope_dir,merscope_spatialdata_path,merscope_z_range,"
        "merscope_channels,xenium_dir,xenium_spatialdata_path,xenium_channels\n"
        f'P1,{merscope_dir},,1-6,"DAPI,PolyT",{xenium_dir},,"DAPI,18S"\n'
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
        f"pair_id,merscope_zarr_path,xenium_dir\nP1,{merscope_zarr},{xenium_dir}\n"
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
        "pair_id,merscope_spatialdata_path,xenium_spatialdata_path\n"
        f"P1,{merscope_zarr},{xenium_zarr}\n"
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


def test_validate_samplesheet_allows_merscope_only_mode(tmp_path: Path) -> None:
    """MERSCOPE-only mode should not require Xenium input columns."""
    merscope_dir = tmp_path / "merscope_raw"
    merscope_dir.mkdir()
    csv_path = tmp_path / "merscope_only.csv"
    csv_path.write_text(f"pair_id,merscope_dir,xenium_dir\nP1,{merscope_dir},\n")

    pairs = parse_samplesheet(csv_path)

    validate_samplesheet(pairs, analysis_mode="merscope")


def test_validate_samplesheet_allows_xenium_only_mode_with_blank_merscope_fields(
    tmp_path: Path,
) -> None:
    """Xenium-only mode should tolerate blank MERSCOPE-specific optional fields."""
    xenium_dir = tmp_path / "xenium_raw"
    xenium_dir.mkdir()
    csv_path = tmp_path / "xenium_only.csv"
    csv_path.write_text(
        "pair_id,merscope_dir,merscope_z_range,merscope_voxel_layers,xenium_dir\n"
        f"P1,,,,{xenium_dir}\n"
    )

    pairs = parse_samplesheet(csv_path)

    assert pairs[0].merscope_z_range == (0, 6)
    assert pairs[0].merscope_voxel_layers == 7
    validate_samplesheet(pairs, analysis_mode="xenium")


def test_validate_samplesheet_uses_row_level_analysis_mode(
    tmp_path: Path,
) -> None:
    """Rows can opt into different platform requirements in one samplesheet."""
    merscope_dir = tmp_path / "merscope_raw"
    merscope_dir.mkdir()
    xenium_dir = tmp_path / "xenium_raw"
    xenium_dir.mkdir()
    csv_path = tmp_path / "mixed_mode.csv"
    csv_path.write_text(
        "pair_id,analysis_mode,enable_alignment,merscope_dir,xenium_dir,"
        "analysis_segmentation,start_stage,stop_stage,only_stage\n"
        f"P1,merscope,false,{merscope_dir},,reseg,segment,enrich,\n"
        f"P2,xenium,true,,{xenium_dir},original_seg,,,qc\n"
    )

    pairs = parse_samplesheet(csv_path)

    assert pairs[0].analysis_mode == "merscope"
    assert pairs[0].enable_alignment is False
    assert pairs[0].analysis_segmentation == "reseg"
    assert pairs[0].start_stage == "segment"
    assert pairs[0].stop_stage == "enrich"
    assert pairs[1].analysis_mode == "xenium"
    assert pairs[1].enable_alignment is True
    assert pairs[1].analysis_segmentation == "original_seg"
    assert pairs[1].only_stage == "qc"
    validate_samplesheet(pairs)


def test_required_platforms_for_mode_rejects_unknown_mode() -> None:
    """Unknown analysis modes should fail with a clear validation error."""
    with pytest.raises(ValueError, match="Unknown analysis_mode"):
        required_platforms_for_mode("other")
