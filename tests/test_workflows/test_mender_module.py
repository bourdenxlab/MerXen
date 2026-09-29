"""Workflow contract tests for the terminal MENDER stage.

The string tests pin the MENDER wiring. When ``nextflow`` is installed, a
harness also includes ``rowSampleSettings``, ``currentPairTerminalStage``,
``currentPairTerminalExpectedCount``, ``appendClusteringSquidpyPreflightChecks``
and ``corticalDepthConfigForPlatform`` from ``workflows/main.nf`` and runs
them on samplesheet rows in both clustering modes (plan §3.1, §13.2): in
map_first MAPMYCELLS runs only when explicitly requested, so the MENDER
barrier never waits for it, and the legacy clustering preflight and the
cortical-depth table configs of legacy rows are unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")


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
    assert "envs/environment.mender.yml" in config_text
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
    environment_text = (root / "envs" / "environment.mender.yml").read_text()
    dockerfile_text = (root / "containers" / "Dockerfile.mender").read_text()
    for expected in [
        "python=3.9",
        "anndata==0.9.1",
        "scanpy==1.9.3",
        "squidpy==1.2.3",
        "pytest==7.4.4",
        "b29dc5ea352a2762cb7bf49d44ee661f0009f694",
    ]:
        assert expected in environment_text
    assert "envs/environment.mender.yml" in dockerfile_text
    assert 'CUDA_VISIBLE_DEVICES=""' in dockerfile_text
    assert "nvidia" not in dockerfile_text.lower()


HARNESS_MAIN = """
include {
    rowSampleSettings;
    currentPairTerminalStage;
    currentPairTerminalExpectedCount;
    appendClusteringSquidpyPreflightChecks;
    corticalDepthConfigForPlatform
} from '__MAIN__'

workflow {
    def cases = new groovy.json.JsonSlurperClassic().parse(new File(params.cases))
    def results = [:]
    cases.each { name, testCase ->
        def runParams = [:] + params + testCase.params
        try {
            def settings = rowSampleSettings(testCase.row, runParams)
            def terminal = currentPairTerminalStage(settings)
            def clusteringErrors = []
            appendClusteringSquidpyPreflightChecks(
                clusteringErrors, settings, runParams
            )
            def depth = corticalDepthConfigForPlatform(
                testCase.row, settings.pair_id, "MERSCOPE",
                settings.analysis_segmentations, runParams,
            )
            results[name] = [value: [
                mode: settings.clustering_squidpy_mode ?: "legacy",
                annotation_keys: settings.keySet().findAll { key ->
                    key in AnnotationSettings.KEYS
                }.sort(),
                run_mapmycells: settings.run_mapmycells,
                run_mender: settings.run_mender,
                terminal: terminal,
                count: currentPairTerminalExpectedCount(settings, terminal),
                clustering_errors: clusteringErrors,
                depth_tables: depth.tables,
            ]]
        } catch (Exception error) {
            results[name] = [error: error.message]
        }
    }
    new File(params.out).text = groovy.json.JsonOutput.toJson(results)
}
"""

ROW = {
    "pair_id": "P1",
    "analysis_mode": "paired",
    "analysis_segmentation": "proseg_hybrid",
    "merscope_dir": "/nonexistent/m",
    "xenium_dir": "/nonexistent/x",
}
MAP_FIRST = {"clustering_squidpy_mode": "map_first"}
# Legacy clustering preflight inputs that do not exist: legacy rows must
# report them, map_first rows must not read them (hook H4).
MISSING_LEGACY_MARKERS = {
    "clustering_squidpy_broad_marker_lookup_path": "/nonexistent/markers.json",
    "clustering_squidpy_broad_taxonomy_metadata_path": "/nonexistent/terms.csv",
}
TERMINAL_CASES: dict[str, dict[str, Any]] = {
    "legacy|stop-mender": {"stop_stage": "mender", "mender_enabled": True},
    "legacy|default-mender": {"mender_enabled": True},
    "map_first|stop-mender": {
        **MAP_FIRST,
        "stop_stage": "mender",
        "mender_enabled": True,
    },
    "map_first|stop-mender-skip": {
        **MAP_FIRST,
        "stop_stage": "mender",
        "mender_enabled": True,
        "annotation_mode_mapmycells_stage": "skip",
    },
    "map_first|default-mender": {**MAP_FIRST, "mender_enabled": True},
    "map_first|depth-mender": {
        **MAP_FIRST,
        "stop_stage": "mender",
        "mender_enabled": True,
        "cortical_depth_enabled": True,
    },
    "map_first|stop-mapmycells": {**MAP_FIRST, "stop_stage": "mapmycells"},
    "map_first|stop-mapmycells-skip": {
        **MAP_FIRST,
        "stop_stage": "mapmycells",
        "annotation_mode_mapmycells_stage": "skip",
    },
    "legacy|preflight": {**MISSING_LEGACY_MARKERS, "annotation_panel_mode": "bogus"},
    "map_first|preflight": {
        **MAP_FIRST,
        **MISSING_LEGACY_MARKERS,
        "annotation_panel_mode": "bogus",
    },
}


@pytest.fixture(scope="module")
def main_nf_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the main.nf functions on every case in one Nextflow run."""
    assert NEXTFLOW is not None
    root = tmp_path_factory.mktemp("main_nf_functions")
    (root / "lib").mkdir()
    for source in (REPO_ROOT / "workflows" / "lib").glob("*.groovy"):
        shutil.copy(source, root / "lib" / source.name)
    (root / "main.nf").write_text(
        HARNESS_MAIN.replace("__MAIN__", str(REPO_ROOT / "workflows" / "main.nf"))
    )
    cases = {
        name: {"row": ROW, "params": params} for name, params in TERMINAL_CASES.items()
    }
    (root / "cases.json").write_text(json.dumps(cases))
    (root / "nextflow.config").write_text(
        f"includeConfig '{REPO_ROOT / 'workflows' / 'nextflow.config'}'\n"
        f"params.cases = '{root / 'cases.json'}'\n"
        f"params.out = '{root / 'results.json'}'\n"
        "params.samplesheet = 'unused.csv'\n"
        f"params.outdir = '{root / 'results'}'\n"
    )
    env = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true", "NXF_ANSI_LOG": "false"}
    completed = subprocess.run(
        [NEXTFLOW, "-log", str(root / "nextflow.log"), "run", "main.nf"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results: dict[str, Any] = json.loads((root / "results.json").read_text())
    return results


def _case(results: dict[str, Any], name: str) -> dict[str, Any]:
    assert "error" not in results[name], results[name]
    value: dict[str, Any] = results[name]["value"]
    return value


@needs_nextflow
def test_map_first_never_waits_for_mapmycells(main_nf_results: dict[str, Any]) -> None:
    """runMapMyCells is false in map_first, so the MENDER barrier counts clustering."""
    legacy = _case(main_nf_results, "legacy|stop-mender")
    assert legacy["mode"] == "legacy"
    assert legacy["run_mapmycells"] is True
    assert legacy["terminal"] == "mapmycells"
    for name in (
        "map_first|stop-mender",
        "map_first|stop-mender-skip",
        "map_first|default-mender",
    ):
        case = _case(main_nf_results, name)
        assert case["mode"] == "map_first", name
        assert case["run_mender"] is True, name
        assert case["run_mapmycells"] is False, name
        assert case["terminal"] == "clustering_squidpy", name
        # One FINALIZE output per required clustering segmentation.
        assert case["count"] == 1, name
    depth = _case(main_nf_results, "map_first|depth-mender")
    assert depth["run_mapmycells"] is False
    assert depth["terminal"] == "compute_cortical_depth"
    assert depth["count"] == 2


@needs_nextflow
def test_map_first_runs_legacy_mapmycells_only_when_asked(
    main_nf_results: dict[str, Any],
) -> None:
    """--stop_stage mapmycells with the stage kept legacy (plan §3.1)."""
    assert _case(main_nf_results, "map_first|stop-mapmycells")["run_mapmycells"] is True
    assert (
        _case(main_nf_results, "map_first|stop-mapmycells-skip")["run_mapmycells"]
        is False
    )


@needs_nextflow
def test_legacy_rows_get_no_annotation_settings(
    main_nf_results: dict[str, Any],
) -> None:
    """A legacy row's settings (VALIDATE_ANALYSIS_LAYER's task input) are unchanged."""
    assert _case(main_nf_results, "legacy|default-mender")["annotation_keys"] == []
    keys = _case(main_nf_results, "map_first|default-mender")["annotation_keys"]
    assert keys == sorted(
        [
            "clustering_squidpy_mode",
            "clustering_squidpy_table_key_suffix",
            "annotation_anatomical_region",
            "annotation_mouse_section_regions",
            "annotation_mode_mapmycells_stage",
            "mender_unassigned_state_policy",
        ]
    )


@needs_nextflow
def test_preflight_gates_the_legacy_checks_on_the_mode(
    main_nf_results: dict[str, Any],
) -> None:
    """Hook H4: legacy marker checks run for legacy rows only."""
    legacy = "\n".join(_case(main_nf_results, "legacy|preflight")["clustering_errors"])
    assert "CLUSTERING_SQUIDPY broad marker lookup" in legacy
    assert "annotation_panel_mode" not in legacy
    map_first = "\n".join(
        _case(main_nf_results, "map_first|preflight")["clustering_errors"]
    )
    assert "CLUSTERING_SQUIDPY broad marker lookup" not in map_first
    assert (
        "Unknown annotation_panel_mode 'bogus' for P1 (human, "
        "clustering_squidpy_mode map_first)"
    ) in map_first
    assert "not available yet" not in map_first


@needs_nextflow
def test_cortical_depth_tables_carry_the_suffix_of_map_first_rows(
    main_nf_results: dict[str, Any],
) -> None:
    legacy = _case(main_nf_results, "legacy|default-mender")["depth_tables"]
    assert legacy == [
        {
            "segmentation": "proseg_hybrid",
            "table_key": "table_MOSAIK_proseg_hybrid",
            "shape_key": "MOSAIK_proseg_hybrid",
        }
    ]
    map_first = _case(main_nf_results, "map_first|default-mender")["depth_tables"]
    assert map_first == [{**legacy[0], "clustered_table_key_suffix": "mapfirst"}]
