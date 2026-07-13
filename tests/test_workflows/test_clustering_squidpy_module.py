"""Workflow text smoke tests for the clustering_squidpy Nextflow module."""

from __future__ import annotations

from pathlib import Path


def test_clustering_squidpy_nextflow_json_includes_hierarchical_fields() -> None:
    """The generated stage JSON should expose hierarchical settings."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (
        repo_root / "workflows" / "modules" / "clustering_squidpy.nf"
    ).read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        '"hierarchical_enabled"',
        '"broad_round"',
        '"subcluster_round"',
        '"neuron_split_round"',
        '"neuron_subcluster_round"',
        '"broad_annotation"',
        '"spatial_scatter_point_size"',
        '"write_spatialdata_table"',
        "clustering_squidpy_gpu_vram_monitor = true",
        "clustering_squidpy_gpu_vram_monitor_interval_seconds = 2",
        "clustering_squidpy_write_spatialdata_table = true",
        "clustering_squidpy_max_forks = 4",
        "merxen.monitoring.gpu_vram",
        "clustering_compute_out/gpu_vram",
        "clustering_squidpy_hierarchical_enabled = true",
        "clustering_squidpy_spatial_scatter_point_size = 2.0",
    ]:
        assert expected in module_text or expected in config_text


def test_workflow_preflight_checks_reference_files_before_task_inputs() -> None:
    """Stage-aware preflight checks should guard reference-backed stages."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        "runPreflightChecks(row, settings, params)",
        "preflight_done_ch = sample_rows_raw_ch",
        ".combine(preflight_done_ch)",
        "settings.run_clustering_squidpy && hierarchicalEnabled",
        "params.clustering_squidpy_broad_marker_lookup_path",
        "params.clustering_squidpy_broad_taxonomy_metadata_path",
        "params.clustering_squidpy_broad_cluster_membership_path",
        "settings.run_mapmycells",
        "params.mapmycells_marker_lookup_path",
        "params.mapmycells_precomputed_stats_path",
        "Preflight checks failed for sample",
        "alignment_max_forks = 1",
        "cellpose_segment_max_forks = 1",
        "proseg_segment_max_forks = 2",
    ]:
        assert expected in main_text or expected in config_text


def test_mapmycells_nextflow_exposes_wmb_cross_species_settings() -> None:
    """Nextflow should pass atlas, species, download, and gene-mapper settings."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (repo_root / "workflows/modules/mapmycells.nf").read_text()
    main_text = (repo_root / "workflows/main.nf").read_text()
    config_text = (repo_root / "workflows/nextflow.config").read_text()

    for expected in [
        '"reference_atlas"',
        '"query_species"',
        '"auto_download_references"',
        '"gene_mapping_db_path"',
        'mapmycells_reference_atlas = "whb"',
        'mapmycells_query_species = "human"',
        "mapmycells_auto_download_references = true",
        "mapmycells_gene_mapping_db_path = null",
        'mapMyCellsReferenceAtlas in ["whb", "wmb"]',
        'mapMyCellsQuerySpecies in ["human", "mouse"]',
    ]:
        assert (
            expected in module_text or expected in main_text or expected in config_text
        )


def test_gpu_processes_share_local_lock() -> None:
    """GPU-heavy local processes should not overlap on one workstation GPU."""
    repo_root = Path(__file__).resolve().parents[2]
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        "gpu_process_lock_enabled = true",
        "gpu_process_lock_file",
        "Waiting for MerXen GPU process lock",
        "flock 9",
        'withName: "CELLPOSE_SEGMENT"',
        'withName: "ALIGN"',
        'withName: "CLUSTERING_SQUIDPY_COMPUTE"',
        "params.cellpose_gpu",
        "params.alignment_device",
        "params.clustering_squidpy_use_gpu",
    ]:
        assert expected in config_text


def test_clustering_gpu_compute_is_isolated_from_spatialdata_io() -> None:
    """Only prepare/finalize should touch SpatialData around GPU compute."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (
        repo_root / "workflows" / "modules" / "clustering_squidpy.nf"
    ).read_text()
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for process_name in [
        "CLUSTERING_SQUIDPY_PREPARE",
        "CLUSTERING_SQUIDPY_COMPUTE",
        "CLUSTERING_SQUIDPY_FINALIZE",
    ]:
        assert f"process {process_name}" in module_text
        assert process_name in main_text

    assert "clustering_squidpy_gpu_conda" in config_text
    assert "environment.clustering-gpu.yml" in config_text
    assert 'withName: "CLUSTERING_SQUIDPY_COMPUTE"' in config_text
    assert "clustering_prepared_ch = CLUSTERING_SQUIDPY_PREPARE" in main_text
    assert "clustering_computed_ch = CLUSTERING_SQUIDPY_COMPUTE" in main_text
    assert "CLUSTERING_SQUIDPY_FINALIZE(clustering_computed_ch)" in main_text
