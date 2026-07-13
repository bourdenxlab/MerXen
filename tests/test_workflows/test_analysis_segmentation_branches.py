"""Workflow text checks for analysis segmentation branch wiring."""

from __future__ import annotations

from pathlib import Path


def test_nextflow_exposes_analysis_segmentation_branches() -> None:
    """The pipeline should route downstream modules through segmentation branches."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        'analysis_segmentation = "both"',
        "normalizeAnalysisSegmentation",
        "rowSampleSettings",
        "settings.analysis_segmentations",
        '"table_MOSAIK_proseg"',
        '"table_original"',
        '"merscope_cell_boundaries"',
        '"xenium_cell_boundaries"',
        "samplesJsonForSegmentation",
    ]:
        assert expected in main_text or expected in config_text


def test_downstream_modules_publish_under_segmentation_branch() -> None:
    """Branch-specific downstream modules should include segmentation in outputs."""
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = repo_root / "workflows" / "modules"

    expectations = {
        "qc.nf": "/${segmentation}/qc",
        "comparison.nf": "/${segmentation}/comparison",
        "visualization.nf": "/${segmentation}/visualization",
        "spatial_gene_analysis.nf": "/${segmentation}/spatial_gene_analysis",
        "clustering_squidpy.nf": "/${segmentation}/clustering_squidpy",
        "mapmycells.nf": "/${segmentation}/mapmycells",
    }
    for filename, expected in expectations.items():
        assert expected in (module_dir / filename).read_text()


def test_nextflow_uses_row_level_settings_and_continues_after_task_errors() -> None:
    """Workflow settings should be row-scoped and task failures non-terminal."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        'rowFieldOrDefault(row, "analysis_mode", params.analysis_mode)',
        (
            'rowFieldOrDefault(row, "analysis_segmentation", '
            "params.analysis_segmentation)"
        ),
        'rowFieldOrDefault(row, "enable_alignment", params.enable_alignment)',
        'rowStartStageRaw = chooseField(row, ["start_stage"])',
        'rowStopStageRaw = chooseField(row, ["stop_stage"])',
        'rowOnlyStageRaw = chooseField(row, ["only_stage"])',
        'tuple("${pairId}|${platform}|${segmentation}", settings)',
        "settings.enable_alignment",
        "settings.run_compare",
        "settings.run_visualize",
        "settings.run_spatial_gene_analysis",
        "settings.run_clustering_squidpy",
    ]:
        assert expected in main_text

    assert 'errorStrategy = "ignore"' in config_text
    assert "failOnIgnore = true" in config_text


def test_mask_image_quantification_stage_is_wired_before_qc() -> None:
    """Image quantification should run after enrichment and feed downstream zarrs."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (
        repo_root / "workflows" / "modules" / "mask_image_quantification.nf"
    ).read_text()

    for expected in [
        (
            "include { MASK_IMAGE_QUANTIFICATION } from "
            '"./modules/mask_image_quantification"'
        ),
        '"mask_image_quantification": "mask_image_quantification"',
        '"image_quantification": "mask_image_quantification"',
        '"quantify_images": "mask_image_quantification"',
        'stages += ["mask_image_quantification"]',
        "settings.run_mask_image_quantification",
        "need_quantified_zarrs: runMaskImageQuantification",
        "MASK_IMAGE_QUANTIFICATION(",
        "downstream_zarrs_ch = enriched_downstream_zarrs_ch.mix(quantified_zarrs_ch)",
        "qc_inputs_ch = analysis_ready_zarrs_ch",
        "analysis_without_qc_ch = analysis_ready_zarrs_ch",
        "merscope_zarr_ch = analysis_ready_zarrs_ch",
        "xenium_zarr_ch = analysis_ready_zarrs_ch",
    ]:
        assert expected in main_text

    for expected in [
        "mask_image_quantification_enabled = true",
        "mask_image_quantification_max_forks",
        'withName: "MASK_IMAGE_QUANTIFICATION"',
    ]:
        assert expected in config_text

    for expected in [
        "process MASK_IMAGE_QUANTIFICATION",
        "mask_image_quantification_input_mask.npy",
        "merxen mask-image-quantification",
        "mask_image_quantification_out",
    ]:
        assert expected in module_text


def test_compute_cortical_depth_stage_is_wired_after_clustering() -> None:
    """Cortical depth should run after clustering when cluster labels are needed."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (
        repo_root / "workflows" / "modules" / "compute_cortical_depth.nf"
    ).read_text()

    for expected in [
        'include { COMPUTE_CORTICAL_DEPTH } from "./modules/compute_cortical_depth"',
        '"compute_cortical_depth": "compute_cortical_depth"',
        '"cortical_depth": "compute_cortical_depth"',
        'stages += ["compute_cortical_depth"]',
        "settings.run_compute_cortical_depth",
        "appendCorticalDepthPreflightChecks",
        "corticalDepthAnnotationPath",
        "corticalDepthConfigForPlatform",
        "COMPUTE_CORTICAL_DEPTH(",
        "analysis_ready_zarrs_ch",
        "qc_inputs_ch = analysis_ready_zarrs_ch",
        "analysis_without_qc_ch = analysis_ready_zarrs_ch",
        "merscope_zarr_ch = analysis_ready_zarrs_ch",
        "xenium_zarr_ch = analysis_ready_zarrs_ch",
    ]:
        assert expected in main_text

    for expected in [
        "cortical_depth_enabled = false",
        "cortical_depth_raster_resolution_um = 5.0",
        "cortical_depth_streamline_spacing_um = 50.0",
        'withName: "COMPUTE_CORTICAL_DEPTH"',
    ]:
        assert expected in config_text

    for expected in [
        "process COMPUTE_CORTICAL_DEPTH",
        "compute_cortical_depth_out",
        "cortical_depth_config.json",
        "merxen compute-cortical-depth",
    ]:
        assert expected in module_text


def test_segment_bootstraps_proseg_from_configured_paths() -> None:
    """The workflow should resolve ProSeg from config instead of a required flag."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (
        repo_root / "workflows" / "modules" / "proseg_bootstrap.nf"
    ).read_text()
    segmentation_module_text = (
        repo_root / "workflows" / "modules" / "segmentation.nf"
    ).read_text()

    for expected in [
        'include { ENSURE_PROSEG } from "./modules/proseg_bootstrap"',
        "proseg_trigger_ch = segment_meta_ch.map { true }.take(1)",
        "proseg_path_ch = ENSURE_PROSEG(proseg_trigger_ch)",
        "SEGMENT(segment_inputs_ch, proseg_path_ch)",
    ]:
        assert expected in main_text

    for expected in [
        "proseg_search_paths",
        "/usr/bin/proseg",
        "proseg_install_path",
        "proseg_auto_install = true",
        'proseg_cargo_package = "proseg"',
    ]:
        assert expected in config_text

    for expected in [
        "process ENSURE_PROSEG",
        "cargo install",
        "sudo -v",
        "proseg_path.txt",
    ]:
        assert expected in module_text

    for expected in [
        "workflow SEGMENT",
        "cellpose_results = CELLPOSE_SEGMENT(segment_inputs)",
        ".combine(proseg_path)",
        "segment_results = PROSEG_SEGMENT(proseg_inputs)",
        '--proseg-binary "\\$(cat "${proseg_path_file}")"',
    ]:
        assert expected in segmentation_module_text


def test_spatial_gene_analysis_stage_is_wired_after_visualization() -> None:
    """Spatial gene analysis should run after visualization and before clustering."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (
        repo_root / "workflows" / "modules" / "spatial_gene_analysis.nf"
    ).read_text()

    for expected in [
        'include { SPATIAL_GENE_ANALYSIS } from "./modules/spatial_gene_analysis"',
        '"spatial_gene_analysis": "spatial_gene_analysis"',
        '"spatial_autocorrelation": "spatial_gene_analysis"',
        ('stages += ["visualize", "spatial_gene_analysis", "clustering_squidpy"]'),
        'stages += ["mapmycells"]',
        "settings.run_spatial_gene_analysis",
        "spatial_gene_analysis_after_visualize_ch",
        "spatial_gene_analysis_done_ch",
        "clustering_after_spatial_gene_analysis_ch",
        "SPATIAL_GENE_ANALYSIS(",
    ]:
        assert expected in main_text

    for expected in [
        "spatial_gene_analysis_n_neighbors = 6",
        "spatial_gene_analysis_top_n = 10",
        "spatial_gene_analysis_max_forks = 4",
        'withName: "SPATIAL_GENE_ANALYSIS"',
    ]:
        assert expected in config_text

    for expected in [
        "process SPATIAL_GENE_ANALYSIS",
        "spatial_gene_analysis_config.json",
        "spatial_gene_analysis_out",
        "merxen spatial-gene-analysis",
        '"n_neighbors"',
        '"top_n"',
    ]:
        assert expected in module_text
