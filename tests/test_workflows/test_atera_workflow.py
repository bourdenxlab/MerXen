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
    assert body.index("appendAteraStagePreflightChecks") < body.index(
        "CellposeSettings.preflight"
    )


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


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_atera_rows_must_stop_at_the_build(tmp_path: Path) -> None:
    cases = _preflight_cases(tmp_path)
    for ok in ("atera/stop_build", "atera/only_build", "mixed/atera_stop_build"):
        assert cases[ok]["result"] == "OK", cases[ok]
        assert cases[ok]["active_platforms"] == ["ATERA"]
        assert cases[ok]["selected_stages"] == ["build_spatialdata"]
    for failing in (
        "atera/no_stop_stage",
        "atera/only_segment",
        "atera/stop_segment",
        "mixed/atera",
    ):
        record = cases[failing]
        assert record.get("error_class") == "java.lang.IllegalArgumentException"
        assert "ATERA rows run only build_spatialdata" in record["message"]
        assert "Set stop_stage=build_spatialdata" in record["message"]
    # The other row of a mixed samplesheet passes on its own; the workflow
    # still aborts the run, because it checks every row before any task.
    assert cases["mixed/xenium"]["result"] == "OK"
    assert cases["mixed/xenium"]["active_platforms"] == ["XENIUM"]
    assert "PAIR_C" in cases["mixed/atera"]["message"]
    (empty,) = cases["atera/no_stage"]["errors"]
    assert "this row selects no stage" in empty


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_a_global_build_only_stop_also_passes(tmp_path: Path) -> None:
    cases = _preflight_cases(tmp_path, "--stop_stage", "build_spatialdata")
    assert cases["atera/no_stop_stage"]["result"] == "OK"
    assert cases["mixed/atera"]["result"] == "OK"


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
