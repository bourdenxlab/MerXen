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
        "clustering_squidpy_out/gpu_vram",
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
        "segment_max_forks = 1",
    ]:
        assert expected in main_text or expected in config_text


def test_gpu_processes_share_local_lock() -> None:
    """GPU-heavy local processes should not overlap on one workstation GPU."""
    repo_root = Path(__file__).resolve().parents[2]
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        "gpu_process_lock_enabled = true",
        "gpu_process_lock_file",
        "Waiting for MerXen GPU process lock",
        "flock 9",
        'withName: "SEGMENT"',
        'withName: "ALIGN"',
        'withName: "CLUSTERING_SQUIDPY"',
        "params.cellpose_gpu",
        "params.alignment_device",
        "params.clustering_squidpy_use_gpu",
    ]:
        assert expected in config_text
