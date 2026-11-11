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
    "mecr",
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
    "atera/transcript_missing_annotation": (
        "SPATIAL_GENE_ANALYSIS PAIR_A:ATERA combined pia/tissue-edge annotation GeoJSON"
    ),
    "atera/cortical_depth": "off for non-cortical ATERA sections",
    "atera/distance_from_object": "off for non-cortical ATERA sections",
    "atera/alignment": "AT-10",
    "atera/cellpose_only": (
        "segmentation_source=cellpose_only is not available for ATERA rows yet; "
        "ATERA supports: native, native_proseg, cellpose_proseg"
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
    "atera/map_first_row": (
        "map_first annotation requires --annotation_hippocampal_resources "
        "for ATERA rows"
    ),
    "cellpose/map_first_only_clustering": (
        "map_first annotation requires --annotation_hippocampal_resources"
    ),
    "cellpose/map_first_only_mender": (
        "map_first annotation requires --annotation_hippocampal_resources"
    ),
    "mixed/xenium_native": (
        "segmentation_source=native is not available for XENIUM rows yet; "
        "XENIUM supports: cellpose_proseg"
    ),
}


def _clustering_reference_params(tmp_path: Path) -> list[str]:
    """Reference files the clustering and MECR preflights check (contents unread).

    MECR runs by default on human ATERA rows (AT-8), so its WHB reference
    files are checked like the clustering ones.
    """
    marker = tmp_path / "markers.json"
    taxonomy = tmp_path / "cluster_annotation_term.csv"
    marker.write_text("{}")
    taxonomy.write_text("")
    params = [
        "--clustering_squidpy_broad_marker_lookup_path",
        str(marker),
        "--clustering_squidpy_broad_taxonomy_metadata_path",
        str(taxonomy),
    ]
    for name in (
        "neurons_h5ad",
        "nonneurons_h5ad",
        "cell_metadata",
        "taxonomy_metadata",
        "cluster_membership",
    ):
        path = tmp_path / f"whb_{name}"
        path.write_text("")
        params += [f"--mecr_whb_{name}_path", str(path)]
    return params


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
    # MECR and transcript patterns run on ATERA rows using streaming engines.
    assert full["run_mecr"] is True
    assert full["transcript_analysis"] is True
    for case in ("atera/transcript_analysis", "atera/transcript_analysis_off"):
        record = preflight_cases[case]
        assert record["result"] == "OK", (case, record)
        assert record["setting_notes"] == full["setting_notes"]
    assert preflight_cases["atera/transcript_analysis"]["transcript_analysis"] is True
    assert (
        preflight_cases["atera/transcript_analysis_off"]["transcript_analysis"] is False
    )
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
        "atera/mecr",
        "atera/only_mecr",
    ):
        record = preflight_cases[case]
        assert record["result"] == "OK", (case, record)
        assert record["segmentation_source"] == "native"
        assert record["analysis_segmentations"] == ["original_seg"]
    assert preflight_cases["atera/only_segment"]["selected_stages"] == ["segment"]
    # MECR (Atera plan AT-8): a row may switch it on, off, or run it alone.
    assert preflight_cases["atera/mecr"]["run_mecr"] is True
    assert preflight_cases["atera/only_mecr"]["selected_stages"] == ["mecr"]
    assert preflight_cases["atera/mecr_off"]["run_mecr"] is False
    assert "mecr" not in preflight_cases["atera/mecr_off"]["stage_order"]
    assert preflight_cases["atera/start_enrich"]["selected_stages"] == ["enrich"]


@pytest.mark.nextflow
def test_atera_rows_cluster_in_legacy_mode(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    """map_first annotation (RCA) is gated off for ATERA rows.

    Human runs map_first by default since the RCA flip (M8), but its panel,
    reference and gate steps are built for the MERSCOPE/XENIUM panels, so an
    ATERA row without its own clustering_squidpy_mode clusters in legacy
    mode, and its settings gain no annotation key (its task hashes are the
    pre-merge ones). A row that names map_first is refused
    (``test_atera_refusals``).
    """
    for case in (
        "atera/no_stop_stage",
        "atera/mender",
        "atera/legacy_row",
        "mixed/atera",
    ):
        record = preflight_cases[case]
        assert record["result"] == "OK", (case, record)
        assert record["clustering_squidpy_mode"] == "legacy", (case, record)
    # The XENIUM row of the same samplesheet keeps the human default.
    assert preflight_cases["mixed/xenium"]["clustering_squidpy_mode"] == "map_first"


_ATERA_LEGACY_NOTE = (
    "cluster in legacy mode: map_first annotation is not covered for ATERA rows "
    "yet (AT-11, RCA on Atera)"
)


@pytest.mark.nextflow
def test_atera_rows_raise_no_legacy_deprecation_warning(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    """The launch log never calls an ATERA row's legacy mode deprecated.

    Neither a row held in legacy mode nor one that names legacy itself (as the
    map_first refusal advises) is a deprecated choice. The mode summary marks a
    held row "(ATERA)", not "(row)"; a XENIUM legacy row of the same
    samplesheet still raises the warning.
    """
    summary = "Clustering mode per samplesheet row (human; run mode map_first): "
    expected = {
        "log/atera": [
            summary + "PAIR_A=legacy (ATERA)",
            f"ATERA rows PAIR_A {_ATERA_LEGACY_NOTE}",
        ],
        "log/atera_legacy_row": [summary + "PAIR_A=legacy (row)"],
        "log/mixed": [
            summary + "PAIR_B=map_first, PAIR_C=legacy (ATERA)",
            f"ATERA rows PAIR_C {_ATERA_LEGACY_NOTE}",
        ],
        "log/mixed_atera_legacy_row": [
            summary + "PAIR_B=map_first, PAIR_A=legacy (row)"
        ],
        "log/mixed_xenium_legacy_row": [
            summary + "PAIR_B=map_first, PAIR_C=legacy (ATERA), PAIR_D=legacy (row)",
            f"ATERA rows PAIR_C {_ATERA_LEGACY_NOTE}",
        ],
    }
    for case, info in expected.items():
        record = preflight_cases[case]
        assert record["info"] == info, (case, record)
        if case == "log/mixed_xenium_legacy_row":
            (warning,) = record["warn"]
            assert warning.startswith("Human legacy clustering"), warning
        else:
            assert record["warn"] == [], (case, record)


@pytest.mark.nextflow
def test_annotation_prepare_only_leaves_atera_rows_out(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    """--annotation_prepare_only builds no reference bundle for an ATERA row."""
    assert preflight_cases["prepare_only/atera"] == {"left_out": True}
    assert preflight_cases["prepare_only/xenium"] == {"left_out": False}


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_global_legacy_mode_warns_only_for_non_atera_rows(tmp_path: Path) -> None:
    """--clustering_squidpy_mode legacy is deprecated only where it acts."""
    cases = _preflight_cases(
        tmp_path, "--stop_stage", "enrich", "--clustering_squidpy_mode", "legacy"
    )
    summary = "Clustering mode per samplesheet row (human; run mode legacy): "
    assert cases["log/atera"]["info"][0] == summary + "PAIR_A=legacy (ATERA)"
    assert cases["log/atera"]["warn"] == []
    assert cases["log/atera_legacy_row"]["warn"] == []
    (warning,) = cases["log/mixed"]["warn"]
    assert warning.startswith("Human legacy clustering"), warning


@pytest.mark.nextflow
def test_early_map_first_atera_stages_need_no_annotation_resources(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    """Execute the full preflight, including the annotation library checks."""
    for stage in ("build_spatialdata", "segment", "enrich", "train_cellpose"):
        record = preflight_cases[f"cellpose/map_first_only_{stage}"]
        assert record.get("result") == "OK", record
        assert record["selected_stages"] == [stage]
        assert record["clustering_squidpy_mode"] == "map_first"


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
def test_global_mecr_and_transcripts_apply_for_atera(
    tmp_path: Path,
) -> None:
    """Both global analysis switches apply to ATERA rows."""
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
    assert record["run_mecr"] is False  # The stop at enrich is before MECR.
    assert record["transcript_analysis"] is True
    assert "mecr" in record["stage_order"]
    assert not any("mecr_enabled" in note for note in record["setting_notes"])
    (tmp_path / "off").mkdir()
    off = _preflight_cases(
        tmp_path / "off", "--stop_stage", "enrich", "--mecr_enabled", "false"
    )
    assert "mecr" not in off["atera/no_stop_stage"]["stage_order"]


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


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_global_map_first_mode_is_ignored_for_atera(tmp_path: Path) -> None:
    """An explicit --clustering_squidpy_mode map_first leaves ATERA rows legacy."""
    cases = _preflight_cases(
        tmp_path,
        "--stop_stage",
        "enrich",
        "--clustering_squidpy_mode",
        "map_first",
    )
    for case in ("atera/no_stop_stage", "mixed/atera"):
        record = cases[case]
        assert record["result"] == "OK", (case, record)
        assert record["clustering_squidpy_mode"] == "legacy", (case, record)
    assert cases["mixed/xenium_enrich"]["clustering_squidpy_mode"] == "map_first"


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


# --------------------------------------------------------------------------- #
# AT-9: the ATERA ProSeg route (segmentation_source=native_proseg)
# --------------------------------------------------------------------------- #

_PROSEG_REFUSALS = {
    "proseg/hybrid": (
        "analysis_segmentation=proseg_hybrid for PAIR_A is not available with "
        "segmentation_source=native_proseg; available: proseg_vendor, original_seg"
    ),
    # AT-7 renamed the vendor-prior layer: reseg is the Cellpose-prior layer.
    "proseg/reseg_only": (
        "analysis_segmentation=reseg for PAIR_A is not available with "
        "segmentation_source=native_proseg; available: proseg_vendor, original_seg"
    ),
    "proseg/mender_hybrid": (
        "mender_segmentations=proseg_hybrid for PAIR_A is not available with "
        "segmentation_source=native_proseg"
    ),
    "proseg/segment_nuclei": (
        "Rows with segmentation_source=native_proseg keep the vendor nuclei"
    ),
    "proseg/xenium": (
        "segmentation_source=native_proseg is not available for XENIUM rows yet; "
        "XENIUM supports: cellpose_proseg"
    ),
}


@pytest.mark.nextflow
def test_native_proseg_rows_run_both_layers(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    full = preflight_cases["proseg/default"]
    assert full["result"] == "OK", full
    assert full["segmentation_source"] == "native_proseg"
    assert full["selected_stages"] == _ATERA_FULL_STAGES
    assert "segment_nuclei" not in full["stage_order"]
    assert full["run_native_segment"] is True and full["run_atera_proseg"] is True
    assert full["run_atera_cellpose"] is False
    assert full["analysis_segmentations"] == ["proseg_vendor", "original_seg"]
    mender = preflight_cases["proseg/all_mender"]
    assert mender["result"] == "OK", mender
    assert mender["analysis_segmentations"] == ["proseg_vendor", "original_seg"]
    # The global MENDER default (proseg_hybrid) resolves to both layers.
    assert mender["mender_segmentations"] == ["proseg_vendor", "original_seg"]
    assert (
        "mender_segmentations proseg_hybrid resolves to proseg_vendor, original_seg "
        "(segmentation_source=native_proseg has proseg_vendor, original_seg)"
    ) in mender["setting_notes"]
    assert preflight_cases["proseg/stop_segment"]["selected_stages"] == [
        "build_spatialdata",
        "segment",
    ]
    start = preflight_cases["proseg/start_enrich"]
    assert start["selected_stages"] == ["enrich"]
    assert start["run_atera_proseg"] is False
    assert preflight_cases["proseg/alias"]["segmentation_source"] == "native_proseg"
    assert preflight_cases["proseg/vendor_only"]["analysis_segmentations"] == [
        "proseg_vendor"
    ]
    # The native route is unchanged by the second source.
    assert preflight_cases["atera/no_stop_stage"]["run_atera_proseg"] is False


@pytest.mark.nextflow
@pytest.mark.parametrize("case", sorted(_PROSEG_REFUSALS))
def test_native_proseg_refusals(
    preflight_cases: dict[str, dict[str, Any]], case: str
) -> None:
    record = preflight_cases[case]
    assert record.get("error_class") == "java.lang.IllegalArgumentException", record
    assert _PROSEG_REFUSALS[case] in record["message"], record["message"]


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        (("--atera_proseg_halo_um", "50"), "below the minimum of 100 microns"),
        (("--atera_proseg_tile_um", "150"), "must exceed twice the halo"),
        (
            ("--atera_proseg_samples", "50", "--atera_proseg_recorded_samples", "60"),
            "exceeds atera_proseg_samples=50",
        ),
        (("--atera_proseg_prior_weight", "0"), "atera_proseg_prior_weight must be"),
        (("--proseg_version", "3.3.0"), "verified with ProSeg 3.2.0 only"),
        (
            ("--atera_proseg_tile_subset", "c2_r1"),
            "atera_proseg_tile_subset (c2_r1) is a check run",
        ),
    ],
)
def test_native_proseg_param_refusals(
    tmp_path: Path, params: tuple[str, ...], message: str
) -> None:
    references = _clustering_reference_params(tmp_path)
    launch = tmp_path / "launch"
    launch.mkdir()
    cases = _preflight_cases(launch, *params, *references)
    record = cases["proseg/default"]
    assert record.get("error_class") == "java.lang.IllegalArgumentException", record
    assert message in record["message"], record["message"]
    # Native rows ignore every ATERA ProSeg param.
    assert cases["atera/no_stop_stage"]["result"] == "OK"
    if "--atera_proseg_tile_subset" in params:
        assert cases["proseg/stop_segment"]["result"] == "OK"
    if "--proseg_version" in params:
        allowed_launch = tmp_path / "allowed"
        allowed_launch.mkdir()
        allowed = _preflight_cases(
            allowed_launch,
            *params,
            "--atera_proseg_allow_untested_version",
            "true",
            *references,
        )
        assert allowed["proseg/default"]["result"] == "OK"


# --------------------------------------------------------------------------- #
# AT-7: Cellpose (DAPI + 18S) and Cellpose-prior tiled ProSeg on ATERA
# --------------------------------------------------------------------------- #

_CELLPOSE_REFUSALS = {
    "cellpose/hybrid": (
        "analysis_segmentation=proseg_hybrid for PAIR_A is not available with "
        "segmentation_source=cellpose_proseg; available: reseg, original_seg, "
        "proseg_mask"
    ),
    "cellpose/vendor_layer": (
        "analysis_segmentation=proseg_vendor for PAIR_A is not available with "
        "segmentation_source=cellpose_proseg"
    ),
    "cellpose/mender_hybrid": (
        "mender_segmentations=proseg_hybrid for PAIR_A is not available with "
        "segmentation_source=cellpose_proseg"
    ),
    "cellpose/segment_nuclei": "ATERA rows keep the vendor nuclei",
    "cellpose/blank_channels": "atera_channels of PAIR_A names no channel",
}


@pytest.mark.nextflow
def test_atera_cellpose_rows_run_three_layers(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    full = preflight_cases["cellpose/default"]
    assert full["result"] == "OK", full
    assert full["segmentation_source"] == "cellpose_proseg"
    assert full["selected_stages"] == _ATERA_FULL_STAGES
    assert "segment_nuclei" not in full["stage_order"]
    assert full["run_native_segment"] is True
    assert full["run_atera_cellpose"] is True and full["run_atera_proseg"] is True
    assert full["run_segment_nuclei"] is False
    # "all": every layer the route makes; ATERA has no hybrid layer.
    assert full["analysis_segmentations"] == ["reseg", "original_seg", "proseg_mask"]
    assert (
        "analysis_segmentation reseg, original_seg, proseg_mask, proseg_hybrid "
        "resolves to reseg, original_seg, proseg_mask (segmentation_source="
        "cellpose_proseg has reseg, original_seg, proseg_mask)"
    ) in full["setting_notes"]
    mender = preflight_cases["cellpose/all_mender"]
    assert mender["result"] == "OK", mender
    assert mender["mender_segmentations"] == ["reseg", "original_seg", "proseg_mask"]
    assert preflight_cases["cellpose/alias"]["segmentation_source"] == (
        "cellpose_proseg"
    )
    assert preflight_cases["cellpose/stop_segment"]["selected_stages"] == [
        "build_spatialdata",
        "segment",
    ]
    start = preflight_cases["cellpose/start_enrich"]
    assert start["selected_stages"] == ["enrich"]
    assert start["run_atera_cellpose"] is False and start["run_atera_proseg"] is False
    assert preflight_cases["cellpose/mask_only"]["analysis_segmentations"] == [
        "proseg_mask"
    ]
    assert preflight_cases["cellpose/channels"]["result"] == "OK"
    # The native routes run no ATERA Cellpose.
    for case in ("atera/no_stop_stage", "proseg/default"):
        assert preflight_cases[case]["run_atera_cellpose"] is False


@pytest.mark.nextflow
@pytest.mark.parametrize("case", sorted(_CELLPOSE_REFUSALS))
def test_atera_cellpose_refusals(
    preflight_cases: dict[str, dict[str, Any]], case: str
) -> None:
    record = preflight_cases[case]
    assert record.get("error_class") == "java.lang.IllegalArgumentException", record
    assert _CELLPOSE_REFUSALS[case] in record["message"], record["message"]


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            ("--atera_cellpose_proseg_prior_weight", "0"),
            "atera_cellpose_proseg_prior_weight must be positive",
        ),
        (("--atera_proseg_halo_um", "50"), "below the minimum of 100 microns"),
        (("--proseg_version", "3.3.0"), "verified with ProSeg 3.2.0 only"),
    ],
)
def test_atera_cellpose_param_refusals(
    tmp_path: Path, params: tuple[str, ...], message: str
) -> None:
    references = _clustering_reference_params(tmp_path)
    launch = tmp_path / "launch"
    launch.mkdir()
    cases = _preflight_cases(launch, *params, *references)
    record = cases["cellpose/default"]
    assert record.get("error_class") == "java.lang.IllegalArgumentException", record
    assert message in record["message"], record["message"]
    assert cases["atera/no_stop_stage"]["result"] == "OK"


@pytest.mark.nextflow
def test_hippocampal_resources_enable_only_the_explicit_human_domain(
    preflight_cases: dict[str, dict[str, Any]],
) -> None:
    for case in ("hippocampal/explicit", "hippocampal/default"):
        record = preflight_cases[case]
        assert record == {
            "result": "OK",
            "mode": "map_first",
            "suffix": "rcahippocampal",
            "region": "hippocampus",
        }
    assert "hippocampus" in preflight_cases["hippocampal/missing_region"]["message"]
    assert "human" in preflight_cases["hippocampal/mouse"]["message"]
    assert preflight_cases["hippocampal/xenium_unchanged"] == {
        "result": "OK",
        "mode": "map_first",
        "suffix": "",
        "region": "frontal_cortex",
    }
