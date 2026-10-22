"""Tests for samplesheet parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen.io.samplesheet import (
    parse_samplesheet,
    required_platforms_for_mode,
    validate_samplesheet,
)


def test_parse_samplesheet_preserves_optional_model_selectors(tmp_path: Path) -> None:
    """Keep registry hashes as strings and blank platform settings unset."""
    csv_path = tmp_path / "samplesheet.csv"
    csv_path.write_text(
        "pair_id,xenium_cellpose_model,merscope_cellpose_model\n"
        f"P1,{'a' * 64},\n"
        "P2,/path/to/model-bundle,/path/to/merscope-model\n"
    )
    first, second = parse_samplesheet(csv_path)
    assert first.xenium_cellpose_model == "a" * 64
    assert first.merscope_cellpose_model is None
    assert second.xenium_cellpose_model == "/path/to/model-bundle"
    assert second.merscope_cellpose_model == "/path/to/merscope-model"


def test_parse_samplesheet_preserves_pciseq_overrides(tmp_path: Path) -> None:
    """Blank overrides inherit pipeline settings and false remains disabled."""
    csv_path = tmp_path / "samplesheet.csv"
    reference = tmp_path / "reference.csv"
    csv_path.write_text(
        "pair_id,pciseq_enabled,pciseq_reference_path\n"
        f"P1,true,{reference}\nP2,false,\nP3,,\n"
    )

    enabled, disabled, inherited = parse_samplesheet(csv_path)

    assert enabled.pciseq_enabled is True
    assert enabled.pciseq_reference_path == reference
    assert disabled.pciseq_enabled is False
    assert disabled.pciseq_reference_path is None
    assert inherited.pciseq_enabled is None


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


def test_validate_samplesheet_allows_direct_vzg2_for_merscope(
    tmp_path: Path,
) -> None:
    """The existing merscope_dir column should accept a direct VZG2 file."""
    archive_path = tmp_path / "sample.vzg2"
    archive_path.touch()
    csv_path = tmp_path / "vzg2.csv"
    csv_path.write_text(
        f"pair_id,analysis_mode,merscope_dir\nP1,merscope,{archive_path}\n"
    )

    pairs = parse_samplesheet(csv_path)

    assert pairs[0].merscope_dir == archive_path
    validate_samplesheet(pairs)


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
        "analysis_segmentation,spatial_gene_analysis_enabled,"
        "spatial_gene_analysis_transcript_analysis_enabled,"
        "start_stage,stop_stage,only_stage\n"
        f"P1,merscope,false,{merscope_dir},,reseg,false,false,segment,enrich,\n"
        f"P2,xenium,true,,{xenium_dir},original_seg,true,false,,,qc\n"
    )

    pairs = parse_samplesheet(csv_path)

    assert pairs[0].analysis_mode == "merscope"
    assert pairs[0].enable_alignment is False
    assert pairs[0].analysis_segmentation == "reseg"
    assert pairs[0].spatial_gene_analysis_enabled is False
    assert pairs[0].spatial_gene_analysis_transcript_analysis_enabled is False
    assert pairs[0].start_stage == "segment"
    assert pairs[0].stop_stage == "enrich"
    assert pairs[1].analysis_mode == "xenium"
    assert pairs[1].enable_alignment is True
    assert pairs[1].analysis_segmentation == "original_seg"
    assert pairs[1].spatial_gene_analysis_enabled is True
    assert pairs[1].spatial_gene_analysis_transcript_analysis_enabled is False
    assert pairs[1].only_stage == "qc"
    validate_samplesheet(pairs)


def test_parse_samplesheet_supports_mender_row_overrides(tmp_path: Path) -> None:
    """MENDER enablement and segmentation selection can be set per row."""
    csv_path = tmp_path / "mender.csv"
    csv_path.write_text(
        'pair_id,mender_enabled,mender_segmentations\nP1,true,"reseg,proseg_hybrid"\n'
    )

    pair = parse_samplesheet(csv_path)[0]

    assert pair.mender_enabled is True
    assert pair.mender_segmentations == "reseg,proseg_hybrid"


def test_required_platforms_for_mode_rejects_unknown_mode() -> None:
    """Unknown analysis modes should fail with a clear validation error."""
    with pytest.raises(ValueError, match="Unknown analysis_mode"):
        required_platforms_for_mode("other")


def test_parse_samplesheet_reads_merscope_bead_removal_column(tmp_path: Path) -> None:
    """The bead-removal column parses true, false and blank (inherit) values."""
    csv_path = tmp_path / "beads.csv"
    csv_path.write_text(
        "pair_id,merscope_bead_removal\nP1,true\nP2,false\nP3,\nP4,Yes\nP5,0\n"
    )

    pairs = parse_samplesheet(csv_path)

    assert [pair.merscope_bead_removal for pair in pairs] == [
        True,
        False,
        None,
        True,
        False,
    ]


def test_parse_samplesheet_defaults_bead_removal_to_none_without_column(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "no_beads.csv"
    csv_path.write_text("pair_id\nP1\n")

    assert parse_samplesheet(csv_path)[0].merscope_bead_removal is None


def test_parse_samplesheet_rejects_invalid_bead_removal_value(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_beads.csv"
    csv_path.write_text("pair_id,merscope_bead_removal\nP1,maybe\n")

    with pytest.raises(ValueError, match="Invalid boolean"):
        parse_samplesheet(csv_path)


def test_samplesheet_docs_describe_the_bead_removal_column() -> None:
    docs = Path(__file__).resolve().parents[2] / "docs" / "samplesheet.md"

    assert "`merscope_bead_removal`" in docs.read_text()


def test_parse_samplesheet_supports_gaston_row_overrides(tmp_path: Path) -> None:
    """Every GASTON control can be supplied at samplesheet row scope."""
    csv_path = tmp_path / "gaston.csv"
    csv_path.write_text(
        "pair_id,gaston_enabled,gaston_segmentations,gaston_use_gpu,"
        "gaston_n_restarts,gaston_epochs,gaston_checkpoint_interval,"
        "gaston_hidden_spatial,gaston_hidden_expression,gaston_optimizer,"
        "gaston_glmpca_dimensions,gaston_glmpca_penalty,"
        "gaston_glmpca_iterations,gaston_domain_mode,gaston_num_domains,"
        "gaston_min_domains,gaston_max_domains,gaston_domain_buckets,"
        "gaston_auto_k_fallback,gaston_write_spatialdata_table,"
        "gaston_keep_seed_models,gaston_keep_checkpoints,gaston_max_genes\n"
        'P1,true,"reseg,proseg_hybrid",false,2,20,5,"8,4","9,3",adam,'
        "3,2.5,12,fixed,4,2,6,100,3,true,true,all,500\n"
    )

    pair = parse_samplesheet(csv_path)[0]

    assert pair.gaston_enabled is True
    assert pair.gaston_segmentations == "reseg,proseg_hybrid"
    assert pair.gaston_use_gpu is False
    assert pair.gaston_n_restarts == 2
    assert pair.gaston_epochs == 20
    assert pair.gaston_checkpoint_interval == 5
    assert pair.gaston_hidden_spatial == [8, 4]
    assert pair.gaston_hidden_expression == [9, 3]
    assert pair.gaston_optimizer == "adam"
    assert pair.gaston_glmpca_dimensions == 3
    assert pair.gaston_glmpca_penalty == 2.5
    assert pair.gaston_glmpca_iterations == 12
    assert pair.gaston_domain_mode == "fixed"
    assert pair.gaston_num_domains == 4
    assert pair.gaston_min_domains == 2
    assert pair.gaston_max_domains == 6
    assert pair.gaston_domain_buckets == 100
    assert pair.gaston_auto_k_fallback == 3
    assert pair.gaston_write_spatialdata_table is True
    assert pair.gaston_keep_seed_models is True
    assert pair.gaston_keep_checkpoints == "all"
    assert pair.gaston_max_genes == 500
