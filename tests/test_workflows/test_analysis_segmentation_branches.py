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
        "settings.run_clustering_squidpy",
    ]:
        assert expected in main_text

    assert 'errorStrategy = "ignore"' in config_text
    assert "failOnIgnore = true" in config_text


def test_segment_bootstraps_proseg_from_configured_paths() -> None:
    """The workflow should resolve ProSeg from config instead of a required flag."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (
        repo_root / "workflows" / "modules" / "proseg_bootstrap.nf"
    ).read_text()

    for expected in [
        'include { ENSURE_PROSEG } from "./modules/proseg_bootstrap"',
        "proseg_trigger_ch = segment_meta_ch.map { true }.take(1)",
        "proseg_path_ch = ENSURE_PROSEG(proseg_trigger_ch)",
        ".combine(proseg_path_ch)",
        "binary_path: prosegBinaryPath",
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
