"""ATERA rows in the workflow: params, preflight and the BUILD task (plan §4.4, §7.3).

The text tests run in the fast tier. The ``nextflow`` tests run two harnesses
built from the Groovy functions of ``workflows/main.nf``: one resolves ATERA and
mixed samplesheet rows through ``rowSampleSettings`` and ``runPreflightChecks``;
the other runs the real ``BUILD_SPATIALDATA`` process (its pinned ``script:``
text) on a synthetic Atera bundle with the build JSON the workflow renders.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from merxen.io.builders.atera import BUILD_MARKER_NAME, read_build_marker

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parents[1]
_WORKFLOWS = _REPO_ROOT / "workflows"
_PREFLIGHT_TAIL = _HERE / "harness" / "atera_preflight.nf"
_BUILD_TAIL = _HERE / "harness" / "atera_build.nf"
_CELLPOSE_PARAMS = (
    "atera_cellpose_model",
    "atera_cellpose_cellprob",
    "atera_cellpose_flow_threshold",
    "atera_cellpose_diameter",
    "atera_cellpose_final_max_area_um2",
    "atera_mask_max_eccentricity",
)


def _golden_module() -> ModuleType:
    """The golden test module, for its harness helpers (no sibling import)."""
    name = "_merxen_platform_config_golden"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _HERE / "test_platform_config_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config_text() -> str:
    return (_WORKFLOWS / "nextflow.config").read_text()


def _main_text() -> str:
    return (_WORKFLOWS / "main.nf").read_text()


# --------------------------------------------------------------------------- #
# Fast tier: text checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "param",
    [
        *_CELLPOSE_PARAMS,
        "default_atera_voxel_layers",
        "atera_min_qv",
        "atera_build_cpus",
        "atera_build_memory_gb",
        "atera_build_max_rss_gb",
    ],
)
def test_atera_params_are_declared(param: str) -> None:
    """CellposeSettings reads the six names for every active platform."""
    assert f"    {param} = " in _config_text()


def test_build_resources_are_atera_only_closures() -> None:
    text = _config_text()
    start = text.index('withName: "BUILD_SPATIALDATA" {')
    block = text[start : text.index("withName:", start + 10)]
    assert "cpus = { platform == 'ATERA' ? (params.atera_build_cpus as int) : 8 }" in (
        block
    )
    assert ": 80.GB" in block
    assert "(params.atera_build_memory_gb as int).GB * task.attempt" in block
    assert "task.exitStatus in [137, 140]" in block
    assert "platform != 'ATERA' &&" in block


def test_build_config_has_an_atera_arm_with_atera_keys_only() -> None:
    main = _main_text()
    start = main.index("def buildConfigForPlatform(")
    body = main[start : main.index("\n}\n", start)]
    assert 'if (platform == "ATERA") {' in body
    assert body.count("atera: ateraBuildOptions(params)") == 1
    # The MERSCOPE and XENIUM maps gained no atera key.
    for arm in ('if (platform == "MERSCOPE")', 'if (platform == "XENIUM")'):
        arm_end = body.index("]\n    }", body.index(arm))
        assert "atera" not in body[body.index(arm) : arm_end], arm


def test_preflight_checks_atera_stages_first() -> None:
    main = _main_text()
    start = main.index("def runPreflightChecks(")
    body = main[start : main.index("\n}\n", start)]
    assert body.index("appendAteraPreflightChecks") < body.index(
        "CellposeSettings.preflight"
    )


def test_native_params_are_declared() -> None:
    text = _config_text()
    assert "    segmentation_source = null\n" in text
    assert '    native_latest_link_mode = "hardlink"\n' in text
    assert "    atera_clustering_hvg_n_top_genes = 2000\n" in text


def test_native_resources_are_atera_only_closures() -> None:
    """Directives are not hashed; XENIUM/MERSCOPE keep their former values."""
    text = _config_text()
    expected = {
        "NATIVE_LATEST": ('memory = "32 GB"',),
        "ENRICH": ("memory = { platform == 'ATERA' ? 64.GB : 300.GB }",),
        "QC": ("memory = { platform == 'ATERA' ? 48.GB : 24.GB }",),
        "VISUALIZE": (
            "memory = { samples_json.toString().contains('\"ATERA\"') "
            "? 64.GB : 32.GB }",
        ),
        "CLUSTERING_SQUIDPY_COMPUTE": (
            "memory = { samples_json.toString().contains('\"ATERA\"') "
            "? 96.GB : 32.GB }",
        ),
    }
    for name, lines in expected.items():
        start = text.index(f'withName: "{name}" {{')
        block = text[start : text.index("withName:", start + 10)]
        for line in lines:
            assert line in block, (name, line)


# --------------------------------------------------------------------------- #
# Nextflow: preflight on single and mixed samplesheets
# --------------------------------------------------------------------------- #


def _run(
    work_dir: Path, script: str, *params: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    (work_dir / "h.nf").write_text(script)
    (work_dir / "lib").symlink_to(_WORKFLOWS / "lib", target_is_directory=True)
    return subprocess.run(
        [
            nextflow,
            "run",
            "h.nf",
            "-c",
            str(_WORKFLOWS / "nextflow.config"),
            "-ansi-log",
            "false",
            *params,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=env or _golden_module().hermetic_env(),
        timeout=600,
        check=False,
    )


def _preflight_cases(work_dir: Path, *params: str) -> dict[str, dict[str, Any]]:
    script = "\n".join(
        [*_golden_module().workflow_functions(), _PREFLIGHT_TAIL.read_text()]
    )
    result = _run(work_dir, script, "--outdir", "results", *params)
    assert result.returncode == 0 and "DONE" in result.stdout, (
        f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    )
    cases = {}
    for line in result.stdout.splitlines():
        if line.startswith("CASE\t"):
            _, case_id, value = line.split("\t", 2)
            cases[case_id] = json.loads(value)
    return cases


_ATERA_FULL_STAGES = [
    "build_spatialdata",
    "segment",
    "enrich",
    "build_viewer_caches",
    "mask_image_quantification",
    "qc",
    "visualize",
    "spatial_gene_analysis",
    "clustering_squidpy",
]

_REFUSALS = {
    "atera/stop_mapmycells": "standalone MapMyCells is not covered for ATERA rows",
    "atera/only_mapmycells": "standalone MapMyCells is not covered for ATERA rows",
    "atera/start_mapmycells": "standalone MapMyCells is not covered for ATERA rows",
    "atera/pciseq": "pciSeq is off for ATERA rows",
    "atera/gaston": "GASTON is not covered for ATERA rows",
    "atera/mecr": "MECR is off for ATERA rows until it streams its reference",
    "atera/transcript_analysis": "transcript spatial patterns are off for ATERA rows",
    "atera/cortical_depth": "off for non-cortical ATERA sections",
    "atera/distance_from_object": "off for non-cortical ATERA sections",
    "atera/alignment": "AT-10",
    "atera/cellpose_only": (
        "segmentation_source=cellpose_only is not available for ATERA rows yet; "
        "ATERA supports: native"
    ),
    "atera/unknown_source": "Unknown segmentation_source 'magic'",
    "atera/analysis_reseg": (
        "analysis_segmentation=reseg for PAIR_A is not available with "
        "segmentation_source=native"
    ),
    "atera/mender_hybrid": (
        "mender_segmentations=proseg_hybrid for PAIR_A is not available with "
        "segmentation_source=native"
    ),
    "atera/segment_nuclei": "keep the vendor nuclei and run no Cellpose nuclei stage",
    "atera/only_pciseq": "'pciseq' is not active for this run. ATERA rows run only",
    "atera/stop_gaston": "'gaston' is not active for this run. ATERA rows run only",
    "atera/only_mecr": "'mecr' is not active for this run. ATERA rows run only",
    "mixed/xenium_native": (
        "segmentation_source=native is not available for XENIUM rows yet; "
        "XENIUM supports: cellpose_proseg"
    ),
}


def _clustering_reference_params(tmp_path: Path) -> list[str]:
    """Reference files the clustering preflight checks (contents unread)."""
    marker = tmp_path / "markers.json"
    taxonomy = tmp_path / "cluster_annotation_term.csv"
    marker.write_text("{}")
    taxonomy.write_text("")
    return [
        "--clustering_squidpy_broad_marker_lookup_path",
        str(marker),
        "--clustering_squidpy_broad_taxonomy_metadata_path",
        str(taxonomy),
    ]


@pytest.fixture(scope="module")
def preflight_cases(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Any]]:
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is unavailable")
    root = tmp_path_factory.mktemp("atera_preflight")
    work_dir = root / "launch"
    work_dir.mkdir()
    return _preflight_cases(work_dir, *_clustering_reference_params(root))


@pytest.mark.nextflow
def test_atera_rows_run_the_native_route(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    full = preflight_cases["atera/no_stop_stage"]
    assert full["result"] == "OK", full
    assert full["active_platforms"] == ["ATERA"]
    assert full["segmentation_source"] == "native"
    assert full["selected_stages"] == _ATERA_FULL_STAGES
    assert "segment_nuclei" not in full["stage_order"]
    assert full["run_native_segment"] is True
    assert full["run_segment_nuclei"] is False
    assert full["analysis_segmentations"] == ["original_seg"]
    # MECR and the transcript patterns default off for ATERA rows.
    assert full["run_mecr"] is False
    assert full["transcript_analysis"] is False
    mender = preflight_cases["atera/mender"]
    assert mender["result"] == "OK", mender
    assert mender["selected_stages"] == [*_ATERA_FULL_STAGES, "mender"]
    # The stop the refusal message suggests: an explicit stop_stage=mender passes
    # over the standalone MapMyCells stage instead of selecting it.
    stop_mender = preflight_cases["atera/stop_mender"]
    assert stop_mender["result"] == "OK", stop_mender
    assert stop_mender["selected_stages"] == [*_ATERA_FULL_STAGES, "mender"]
    assert stop_mender["run_mapmycells"] is False
    # The global mender default (proseg_hybrid) resolves to the vendor layer.
    assert mender["mender_segmentations"] == ["original_seg"]
    assert (
        "mender_segmentations proseg_hybrid resolves to original_seg "
        "(segmentation_source=native has original_seg)"
    ) in mender["setting_notes"]
    assert any(
        note.startswith("analysis_segmentation reseg, original_seg")
        for note in full["setting_notes"]
    )
    for case in (
        "atera/stop_build",
        "atera/only_segment",
        "atera/start_enrich",
        "atera/segmentation_source_native",
        "atera/analysis_original",
        "atera/analysis_all",
        "atera/mecr_off",
    ):
        record = preflight_cases[case]
        assert record["result"] == "OK", (case, record)
        assert record["segmentation_source"] == "native"
        assert record["analysis_segmentations"] == ["original_seg"]
    assert preflight_cases["atera/only_segment"]["selected_stages"] == ["segment"]
    assert preflight_cases["atera/start_enrich"]["selected_stages"] == ["enrich"]


@pytest.mark.nextflow
@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_atera_refusals(preflight_cases: dict[str, dict[str, Any]], case: str) -> None:
    record = preflight_cases[case]
    assert record.get("error_class") == "java.lang.IllegalArgumentException", record
    assert _REFUSALS[case] in record["message"], record["message"]
    # No hint sends an ATERA row towards a switch the preflight then refuses.
    assert "Pass --" not in record["message"], record["message"]


_ROW_SWITCH_REFUSALS = {
    "atera/pciseq": "pciseq_enabled",
    "atera/gaston": "gaston_enabled",
    "atera/mecr": "mecr_enabled",
    "atera/transcript_analysis": "spatial_gene_analysis_transcript_analysis_enabled",
    "atera/cortical_depth": "cortical_depth_enabled",
    "atera/distance_from_object": "distance_from_object_enabled",
}


@pytest.mark.nextflow
@pytest.mark.parametrize("case", sorted(_ROW_SWITCH_REFUSALS))
def test_row_switch_refusals_name_the_row_setting(
    preflight_cases: dict[str, dict[str, Any]], case: str
) -> None:
    message = preflight_cases[case]["message"]
    field = _ROW_SWITCH_REFUSALS[case]
    assert f"remove {field}" in message or f"and {field}" in message, message
    assert "from row PAIR_A" in message, message


@pytest.mark.nextflow
def test_mixed_samplesheet_keeps_the_xenium_route(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    for case in ("mixed/xenium", "mixed/xenium_enrich"):
        record = preflight_cases[case]
        assert record["result"] == "OK", record
        assert record["active_platforms"] == ["XENIUM"]
        assert record["segmentation_source"] == "cellpose_proseg"
        assert record["setting_notes"] == []
        assert record["analysis_segmentations"] == [
            "reseg",
            "original_seg",
            "proseg_mask",
            "proseg_hybrid",
        ]
    assert preflight_cases["mixed/xenium_enrich"]["selected_stages"] == [
        "build_spatialdata",
        "segment_nuclei",
        "segment",
        "enrich",
    ]
    assert preflight_cases["mixed/atera"]["result"] == "OK"
    (empty,) = preflight_cases["atera/no_stage"]["errors"]
    assert "selects no stage" in empty


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_a_global_build_only_stop_also_passes(tmp_path: Path) -> None:
    cases = _preflight_cases(tmp_path, "--stop_stage", "build_spatialdata")
    assert cases["atera/no_stop_stage"]["result"] == "OK"
    assert cases["atera/no_stop_stage"]["selected_stages"] == ["build_spatialdata"]
    assert cases["mixed/atera"]["result"] == "OK"


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_global_mecr_and_transcripts_are_ignored_for_atera(tmp_path: Path) -> None:
    cases = _preflight_cases(
        tmp_path,
        "--stop_stage",
        "enrich",
        "--mecr_enabled",
        "true",
        "--spatial_gene_analysis_transcript_analysis_enabled",
        "true",
    )
    record = cases["atera/no_stop_stage"]
    assert record["result"] == "OK", record
    assert record["run_mecr"] is False and record["transcript_analysis"] is False
    assert "mecr" not in record["stage_order"]


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_global_off_stage_switches_are_ignored_for_atera(tmp_path: Path) -> None:
    """Global switches of off stages are noted and ignored on ATERA rows.

    A mixed samplesheet keeps them for its XENIUM and MERSCOPE rows; only a
    value set on the ATERA row itself is refused (``test_atera_refusals``).
    """
    cases = _preflight_cases(
        tmp_path,
        "--stop_stage",
        "enrich",
        "--pciseq_enabled",
        "true",
        "--gaston_enabled",
        "true",
        "--cortical_depth_enabled",
        "true",
        "--distance_from_object_enabled",
        "true",
    )
    for case in ("atera/no_stop_stage", "mixed/atera"):
        record = cases[case]
        assert record["result"] == "OK", (case, record)
        for stage in (
            "pciseq",
            "gaston",
            "compute_cortical_depth",
            "distance_from_object",
        ):
            assert stage not in record["stage_order"], (case, stage)
        for field, label in (
            ("pciseq_enabled", "pciSeq"),
            ("gaston_enabled", "GASTON"),
            ("cortical_depth_enabled", "cortical depth"),
            ("distance_from_object_enabled", "distance from object"),
        ):
            assert (
                f"{label} is off for ATERA rows; the global {field} is ignored"
            ) in record["setting_notes"], (case, record["setting_notes"])
    # The XENIUM row of the mixed samplesheet keeps the global switches.
    xenium = cases["mixed/xenium_enrich"]
    assert xenium["result"] == "OK", xenium
    for stage in ("pciseq", "gaston", "compute_cortical_depth", "distance_from_object"):
        assert stage in xenium["stage_order"], stage
    assert xenium["setting_notes"] == []


# --------------------------------------------------------------------------- #
# Nextflow: the BUILD_SPATIALDATA task on a synthetic bundle
# --------------------------------------------------------------------------- #


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_build_task_builds_an_atera_row(tmp_path: Path) -> None:
    sys.path.insert(0, str(_REPO_ROOT / "tests" / "test_io"))
    try:
        from atera_synth import write_atera_bundle
    finally:
        sys.path.pop(0)

    synth = write_atera_bundle(tmp_path / "data" / "roi")
    work_dir = tmp_path / "launch"
    work_dir.mkdir()
    (work_dir / "modules").symlink_to(_WORKFLOWS / "modules", target_is_directory=True)
    include = 'include { BUILD_SPATIALDATA } from "./modules/spatialdata_build"\n'
    script = "\n".join(
        [include, *_golden_module().workflow_functions(), _BUILD_TAIL.read_text()]
    )
    env = _golden_module().hermetic_env()
    # The task runs `merxen` from this checkout and this interpreter's env.
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    results = tmp_path / "results"
    result = _run(
        work_dir,
        script,
        "--outdir",
        str(results),
        "--atera_dir",
        str(synth.root),
        "--atera_build_cpus",
        "2",
        "--atera_build_memory_gb",
        "4",
        "--atera_build_max_rss_gb",
        "32",
        env=env,
    )
    assert result.returncode == 0 and "BUILT\t" in result.stdout, (
        f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    )
    config_line = next(
        line for line in result.stdout.splitlines() if line.startswith("CONFIG\t")
    )
    config = json.loads(config_line.split("\t", 1)[1])
    assert config["platform"] == "ATERA"
    assert config["dataset_name"] == "PAIR_A_ATERA"
    assert config["atera"] == {"min_qv": 20.0, "species": "human", "max_rss_gb": 32.0}
    store = (
        results
        / "PAIR_A"
        / "atera"
        / "spatialdata"
        / "spatialdata_out"
        / "source_spatialdata.zarr"
    )
    marker = read_build_marker(store)
    assert marker is not None, f"no {BUILD_MARKER_NAME} in the published store"
    assert (store / "atera" / "experiment.spatial").read_bytes() == (
        synth.root / "experiment.spatial"
    ).read_bytes()
