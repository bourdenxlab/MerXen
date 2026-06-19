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
        "clustering_squidpy_gpu_vram_monitor = true",
        "clustering_squidpy_gpu_vram_monitor_interval_seconds = 2",
        "merxen.monitoring.gpu_vram",
        "clustering_squidpy_out/gpu_vram",
        "clustering_squidpy_hierarchical_enabled = true",
        "clustering_squidpy_spatial_scatter_point_size = 2.0",
    ]:
        assert expected in module_text or expected in config_text
