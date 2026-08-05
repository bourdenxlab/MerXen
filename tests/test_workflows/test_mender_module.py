"""Workflow contract tests for the terminal MENDER stage."""

from __future__ import annotations

from pathlib import Path


def _texts() -> tuple[str, str, str, str]:
    root = Path(__file__).resolve().parents[2]
    return (
        (root / "workflows" / "main.nf").read_text(),
        (root / "workflows" / "modules" / "mender.nf").read_text(),
        (root / "workflows" / "nextflow.config").read_text(),
        (root / "workflows" / "conf" / "dwight.config").read_text(),
    )


def test_mender_defaults_and_segmentation_selection_are_wired() -> None:
    main_text, _module_text, config_text, _dwight_text = _texts()
    for expected in [
        "def normalizeMenderSegmentations",
        '"all": ["reseg", "original_seg", "proseg_mask", "proseg_hybrid"]',
        '"reseg": ["reseg"]',
        '"original_seg": ["original_seg"]',
        '"proseg_mask": ["proseg_mask"]',
        '"proseg_hybrid": ["proseg_hybrid"]',
        "required_clustering_segmentations: requiredClusteringSegmentations",
        "analysis_input_segmentations: analysisInputSegmentations",
        "settings.analysis_input_segmentations.collect",
    ]:
        assert expected in main_text
    for expected in [
        "mender_enabled = false",
        'mender_segmentations = ["proseg_hybrid"]',
        'mender_cell_state_key = "hierarchical_cluster"',
        'mender_missing_state_policy = "error"',
        'mender_nn_mode = "radius"',
        "mender_radius_um = 20.0",
        "mender_n_scales = 5",
        'mender_count_rep = "s"',
        "mender_include_self = false",
        'mender_clustering_mode = "resolution"',
        "mender_leiden_resolution = 0.8",
        "mender_target_k = null",
        "mender_random_seed = 666",
        "mender_run_umap = true",
        "mender_write_spatialdata_table = true",
    ]:
        assert expected in config_text


def test_mender_is_opt_in_and_extends_the_historical_stop() -> None:
    main_text, _module_text, _config_text, _dwight_text = _texts()
    for expected in [
        '"mender": "mender"',
        'stages += ["mender"]',
        'stopStage = "mender"',
        'def runMender = stageInRange("mender"',
        "run_mender: runMender",
        "Pass --mender_enabled true to use MENDER",
        "MENDER clustering prerequisites require",
        "clustering_squidpy_hierarchical_enabled true",
        "autoExtendedToMender",
        "mender_bypass_terminal_gate",
    ]:
        assert expected in main_text
    stage_order = main_text[
        main_text.index("def activeStageOrder") : main_text.index("def validateStage")
    ]
    assert stage_order.index('stages += ["mapmycells"]') < stage_order.index(
        'stages += ["mender"]'
    )


def test_mender_process_graph_is_isolated_and_cpu_only() -> None:
    main_text, module_text, config_text, _dwight_text = _texts()
    for process_name in (
        "MENDER_PREPARE",
        "MENDER_COMPUTE",
        "MENDER_FINALIZE",
        "MENDER_IMPORT",
    ):
        assert f"process {process_name}" in module_text
        assert process_name in main_text
        assert f'withName: "{process_name}"' in config_text
    assert "mender_prepared_ch = MENDER_PREPARE" in main_text
    assert "mender_computed_ch = MENDER_COMPUTE" in main_text
    assert "mender_finalized_ch = MENDER_FINALIZE" in main_text
    assert "MENDER_IMPORT(mender_finalized_ch)" in main_text
    compute_block = module_text[
        module_text.index("process MENDER_COMPUTE") : module_text.index(
            "process MENDER_FINALIZE"
        )
    ]
    assert 'export CUDA_VISIBLE_DEVICES=""' in compute_block
    assert 'export PYTHONPATH="${projectDir}/../src:' in compute_block
    assert "merxen.mender_compute" in compute_block
    assert "source_clustered.h5ad" not in compute_block
    assert "read_zarr" not in compute_block
    assert "--nv" not in module_text
    assert "clusterOptions = ''" in config_text
    assert "mender_conda" in config_text
    assert "environment.mender.yml" in config_text
    assert "mender_container" in config_text
    assert '"${params.outdir}/${pair_id}/${segmentation}/mender"' in module_text
    assert 'pattern: "mender_out/**"' in module_text
    assert 'path("spatialdata_import_manifest.json")' in module_text


def test_mender_uses_per_pair_release_and_independent_acquisition_tasks() -> None:
    main_text, _module_text, _config_text, _dwight_text = _texts()
    for expected in [
        "def currentPairTerminalStage(settings)",
        "def currentPairTerminalExpectedCount(settings, terminalStage)",
        "pair_terminal_specs_ch",
        "tuple(groupKey(pairId, expectedCount as int), true)",
        "pair_terminal_token_ch",
        ".combine(pair_terminal_token_ch, by: 0)",
        "mender_current_artifacts_ch",
        "mender_published_artifacts_ch",
        "mender_inputs_ch = mender_artifacts_ch",
        '"${pairId}|${segmentation}|${platform}"',
        "samples.collect { sample ->",
        "MENDER_PREPARE",
    ]:
        assert expected in main_text
    mender_block = main_text[
        main_text.index("pair_terminal_specs_ch") : main_text.index(
            "MENDER_IMPORT(mender_finalized_ch)"
        )
    ]
    assert "gaston" not in mender_block.lower()


def test_mender_published_restart_requires_h5ad_and_latest_spatialdata() -> None:
    main_text, module_text, _config_text, _dwight_text = _texts()
    published_block = main_text[
        main_text.index("mender_published_artifacts_ch") : main_text.index(
            "mender_artifacts_ch"
        )
    ]
    assert "settings.mender_published_output_mode" in published_block
    assert '"latest/latest_spatialdata.zarr"' in published_block
    assert '"${sampleId}_clustered.h5ad"' in published_block
    assert "appendMenderPreflightChecks" in main_text
    assert 'latestZarr.resolve("tables").resolve(tableKey)' in main_text
    assert "source_spatialdata_table" in module_text
    assert "native_shape_key" in module_text
    finalize_block = module_text[
        module_text.index("process MENDER_FINALIZE") : module_text.index(
            "process MENDER_IMPORT"
        )
    ]
    assert 'path(clustered_h5ad, stageAs: "source_clustered.h5ad")' in finalize_block


def test_mender_resources_fit_dwight_cpu_budget() -> None:
    _main_text, _module_text, config_text, dwight_text = _texts()
    expected_resources = {
        "MENDER_PREPARE": ("4", "64 GB"),
        "MENDER_COMPUTE": ("8", "192 GB"),
        "MENDER_FINALIZE": ("8", "64 GB"),
        "MENDER_IMPORT": ("4", "48 GB"),
    }
    for process_name, (cpus, memory) in expected_resources.items():
        block_start = config_text.index(f'withName: "{process_name}"')
        block = config_text[block_start : config_text.index("}", block_start)]
        assert f"cpus = {cpus}" in block
        assert f'memory = "{memory}"' in block
    for expected in [
        "mender_prepare_max_forks = 4",
        "mender_compute_max_forks = 2",
        "mender_finalize_max_forks = 4",
        "mender_import_max_forks = 1",
    ]:
        assert expected in dwight_text
    assert 'memory = "640 GB"' in dwight_text


def test_mender_environment_pins_repository_commit_and_old_stack() -> None:
    root = Path(__file__).resolve().parents[2]
    environment_text = (root / "environment.mender.yml").read_text()
    dockerfile_text = (root / "Dockerfile.mender").read_text()
    for expected in [
        "python=3.9",
        "anndata==0.9.1",
        "scanpy==1.9.3",
        "squidpy==1.2.3",
        "pytest==7.4.4",
        "b29dc5ea352a2762cb7bf49d44ee661f0009f694",
    ]:
        assert expected in environment_text
    assert "environment.mender.yml" in dockerfile_text
    assert 'CUDA_VISIBLE_DEVICES=""' in dockerfile_text
    assert "nvidia" not in dockerfile_text.lower()
