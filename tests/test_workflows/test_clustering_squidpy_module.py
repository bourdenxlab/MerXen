"""Workflow text smoke tests for the clustering_squidpy Nextflow module."""

from __future__ import annotations

import re
from pathlib import Path


def test_clustering_squidpy_nextflow_json_includes_hierarchical_fields(
    combined_config_text: str,
) -> None:
    """The generated stage JSON should expose hierarchical settings."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (
        repo_root / "workflows" / "modules" / "clustering_squidpy.nf"
    ).read_text()
    config_text = combined_config_text

    for expected in [
        '"hierarchical_enabled"',
        '"broad_round"',
        '"subcluster_round"',
        '"neuron_split_round"',
        '"neuron_subcluster_round"',
        '"broad_annotation"',
        '"auto_download_reference"',
        '"spatial_scatter_point_size"',
        '"write_spatialdata_table"',
        "clustering_squidpy_gpu_vram_monitor = true",
        "clustering_squidpy_gpu_vram_monitor_interval_seconds = 2",
        "clustering_squidpy_write_spatialdata_table = true",
        "clustering_squidpy_max_forks = 4",
        "merxen.monitoring.gpu_vram",
        "clustering_compute_out/gpu_vram",
        "clustering_squidpy_hierarchical_enabled = null",
        "clustering_squidpy_spatial_scatter_point_size = 2.0",
    ]:
        assert expected in module_text or expected in config_text


def test_workflow_preflight_checks_reference_files_before_task_inputs(
    combined_config_text: str,
) -> None:
    """Stage-aware preflight checks should guard reference-backed stages."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = combined_config_text

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


def test_mapmycells_nextflow_exposes_wmb_cross_species_settings(
    combined_config_text: str,
) -> None:
    """Nextflow should pass atlas, species, download, and gene-mapper settings."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (repo_root / "workflows/modules/mapmycells.nf").read_text()
    main_text = (repo_root / "workflows/main.nf").read_text()
    config_text = combined_config_text

    for expected in [
        '"reference_atlas"',
        '"query_species"',
        '"auto_download_references"',
        '"gene_mapping_db_path"',
        "mapmycells_reference_atlas = null",
        "mapmycells_query_species = null",
        "mapmycells_auto_download_references = true",
        "mapmycells_gene_mapping_db_path = null",
        'mapMyCellsReferenceAtlas in ["whb", "wmb"]',
        'mapMyCellsQuerySpecies in ["human", "mouse"]',
    ]:
        assert (
            expected in module_text or expected in main_text or expected in config_text
        )


def test_mapmycells_n_processors_default_is_a_valid_integer(
    dwight_config_text: str,
) -> None:
    """The base default must validate as an int outside the Dwight profile."""
    repo_root = Path(__file__).resolve().parents[2]
    base_text = (repo_root / "workflows/nextflow.config").read_text()

    assert "mapmycells_n_processors = 4" in base_text
    assert "mapmycells_n_processors = null" not in base_text
    assert "mapmycells_n_processors = 8" in dwight_config_text


def test_workflow_exposes_species_aware_mouse_defaults(
    combined_config_text: str,
) -> None:
    """Mouse mode should choose WMB and retain intentional stage defaults."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (repo_root / "workflows/modules/mapmycells.nf").read_text()
    clustering_text = (
        repo_root / "workflows/modules/clustering_squidpy.nf"
    ).read_text()
    main_text = (repo_root / "workflows/main.nf").read_text()

    for expected in [
        'species = "human"',
        'species == "human"',
        'pipelineSpecies == "mouse" ? "wmb" : "whb"',
        'pipelineSpecies == "mouse" ? "whole_brain" : "both"',
        'referenceAtlas == "wmb" ? "CCN20230722_CLAS"',
        'referenceAtlas = settings.species == "mouse" ? "wmb" : "whb"',
        "mecr_auto_download_reference = true",
        "clustering_squidpy_broad_auto_download_reference = true",
        "params.clustering_squidpy_whb_marker_lookup_path",
        "params.mapmycells_whb_marker_lookup_path",
    ]:
        assert expected in "\n".join(
            [combined_config_text, module_text, clustering_text, main_text]
        )


def test_gpu_processes_share_local_lock(combined_config_text: str) -> None:
    """GPU-heavy local processes should not overlap on one workstation GPU."""
    config_text = combined_config_text

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


def test_clustering_gpu_compute_is_isolated_from_spatialdata_io(
    combined_config_text: str,
) -> None:
    """Only prepare/finalize should touch SpatialData around GPU compute."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (
        repo_root / "workflows" / "modules" / "clustering_squidpy.nf"
    ).read_text()
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = combined_config_text

    for process_name in [
        "CLUSTERING_SQUIDPY_PREPARE",
        "CLUSTERING_SQUIDPY_COMPUTE",
        "CLUSTERING_SQUIDPY_FINALIZE",
    ]:
        assert f"process {process_name}" in module_text
        assert process_name in main_text

    assert "clustering_squidpy_gpu_conda" in config_text
    assert "envs/environment.clustering-gpu.yml" in config_text
    assert 'withName: "CLUSTERING_SQUIDPY_COMPUTE"' in config_text
    assert "clustering_prepared_ch = CLUSTERING_SQUIDPY_PREPARE" in main_text
    assert "clustering_computed_ch = CLUSTERING_SQUIDPY_COMPUTE" in main_text
    assert "CLUSTERING_SQUIDPY_FINALIZE(clustering_computed_ch)" in main_text


def _process_block(module_text: str, name: str) -> str:
    start = module_text.index(f"process {name} {{")
    end = module_text.find("\nprocess ", start + 1)
    return module_text[start : end if end >= 0 else len(module_text)]


def test_compute_cpu_runs_map_first_in_the_main_env_without_gpu(
    combined_config_text: str,
) -> None:
    """COMPUTE_CPU (plan §3.5): main env, CPU only, no GPU queue or lock.

    The legacy COMPUTE (GPU env, GPU lock on Dwight) keeps its own selectors,
    which match its name exactly; no selector reaches COMPUTE_CPU but its own.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (
        repo_root / "workflows" / "modules" / "clustering_squidpy.nf"
    ).read_text()
    annotation_config = (
        repo_root / "workflows" / "conf" / "annotation.config"
    ).read_text()
    dwight_annotation = (
        repo_root / "workflows" / "conf" / "dwight.annotation.config"
    ).read_text()
    block = _process_block(module_text, "CLUSTERING_SQUIDPY_COMPUTE_CPU")

    assert 'cache "deep"' in block
    assert 'export CUDA_VISIBLE_DEVICES=""' in block
    assert 'export PYTHONPATH="${projectDir}/../src:' in block
    assert "python -m merxen.clustering_squidpy_stages compute" in block
    assert "AnnotationReferences.computeArguments(compute_spec)" in block
    assert 'stageAs: "compute_inputs/annotation_resolve_out/*"' in block
    for token in ("gpu", "vram", "--nv", "conda", "container", "queue"):
        assert token not in block.lower(), token
    # FINALIZE's input shape.
    assert 'path("clustering_compute_out")' in block
    selectors = [
        name
        for name in re.findall(
            r'withName:\s*"([^"]+)"', combined_config_text + annotation_config
        )
        if name.startswith("CLUSTERING_SQUIDPY_COMPUTE")
    ]
    assert set(selectors) == {
        "CLUSTERING_SQUIDPY_COMPUTE",
        "CLUSTERING_SQUIDPY_COMPUTE_CPU",
    }
    cpu = annotation_config[
        annotation_config.index('withName: "CLUSTERING_SQUIDPY_COMPUTE_CPU"') :
    ]
    cpu = cpu[: cpu.index("}")]
    assert "cpus = 8" in cpu and 'memory = "32 GB"' in cpu
    for token in ("conda", "container", "queue", "clusterOptions", "beforeScript"):
        assert token not in cpu
    dwight = dwight_annotation[
        dwight_annotation.index('withName: "CLUSTERING_SQUIDPY_COMPUTE_CPU"') :
    ]
    dwight = dwight[: dwight.index("}")]
    assert "maxForks = params.clustering_squidpy_max_forks" in dwight
    assert "beforeScript" not in dwight
