"""Regression tests for the default Dwight workstation execution profile."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _resolved_config(
    workflow_dir: Path,
    *,
    profile: str | None,
) -> dict[str, Any]:
    nextflow = shutil.which("nextflow")
    if nextflow is None:
        raise RuntimeError("nextflow is unavailable")
    command = [nextflow, "config", "-o", "json"]
    if profile is not None:
        command.extend(["-profile", profile])
    command.append(str(workflow_dir))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _dwight_signature(config: dict[str, Any]) -> dict[str, Any]:
    params = config["params"]
    process = config["process"]
    local_executor = config["executor"]
    viewer = process["withName:VIEWER_CACHE"]
    return {
        "cellpose_gpu": params["cellpose_gpu"],
        "proseg_binary": params["proseg_binary"],
        "proseg_num_threads": params["proseg_num_threads"],
        "max_ram_gb": params["max_ram_gb"],
        "warn_ram_gb": params["warn_ram_gb"],
        "mask_image_quantification_max_forks": params[
            "mask_image_quantification_max_forks"
        ],
        "viewer_cache_max_forks": params["viewer_cache_max_forks"],
        "gpu_process_lock_file": params["gpu_process_lock_file"],
        "process_executor": process["executor"],
        "local_executor_cpus": local_executor["cpus"],
        "local_executor_memory": local_executor["memory"],
        "viewer_cache_memory": viewer["memory"],
        "viewer_cache_process_max_forks": viewer["maxForks"],
    }


def test_standard_and_dwight_profiles_include_the_workstation_config() -> None:
    """The implicit default and explicit aliases should load one shared profile."""
    repo_root = Path(__file__).resolve().parents[2]
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    include_pattern = r"""includeConfig\s+["']conf/dwight\.config["']"""
    assert re.search(
        rf"\bstandard\s*\{{\s*{include_pattern}",
        config_text,
    )
    assert re.search(
        rf"\bdwight\s*\{{\s*{include_pattern}",
        config_text,
    )


def test_dwight_declares_local_executor_capacity_not_process_defaults(
    dwight_config_text: str,
) -> None:
    """Host capacity should constrain scheduling without inflating every task."""
    assert 'process.executor = "local"' in dwight_config_text
    assert re.search(
        r"executor\s*\{\s*cpus\s*=\s*72" r'\s*memory\s*=\s*"640 GB"',
        dwight_config_text,
    )
    assert "process.cpus = 72" not in dwight_config_text
    assert 'process.memory = "640 GB"' not in dwight_config_text


def test_dwight_gpu_processes_use_one_fixed_host_lock(
    combined_config_text: str,
    dwight_config_text: str,
) -> None:
    """Every local GPU process should contend on the same host-level lock file."""
    assert 'gpu_process_lock_file = "/tmp/merxen-dwight-gpu.lock"' in dwight_config_text
    for process_name in (
        "CELLPOSE_SEGMENT",
        "CELLPOSE_NUCLEI_SEGMENT",
        "ALIGN",
        "CLUSTERING_SQUIDPY_COMPUTE",
        "GASTON_TRAIN",
    ):
        assert f'withName: "{process_name}"' in dwight_config_text
    # Five tasks hold the lock for their whole run; BUILD_SPATIALDATA only
    # exports the path (see the bead-removal test below).
    assert (
        dwight_config_text.count(
            'MERXEN_GPU_LOCK_FILE="${params.gpu_process_lock_file}"'
        )
        == 6
    )
    assert dwight_config_text.count('exec 9<"\\${MERXEN_GPU_LOCK_FILE}"') == 5
    assert 'exec 9>"\\${MERXEN_GPU_LOCK_FILE}"' not in dwight_config_text
    assert dwight_config_text.count("set -o noclobber") == 5
    assert "${PWD}/.merxen_gpu.lock" not in combined_config_text


def _process_block(config_text: str, process_name: str) -> str:
    """Return the ``withName`` block of one process in the process scope."""
    start = config_text.index(f'withName: "{process_name}"')
    end = config_text.find("withName:", start + 1)
    return config_text[start:end]


def test_dwight_bead_removal_runs_on_the_validated_gpu(
    dwight_config_text: str,
) -> None:
    """The workstation runs the bead correction on the GPU it was validated on."""
    assert 'merscope_bead_removal_device = "cuda"' in dwight_config_text
    assert 'merscope_bead_removal_device = "cpu"' not in dwight_config_text
    assert "torch 2.14 / Triton 3.8" in dwight_config_text


def test_dwight_build_exports_the_gpu_lock_path_without_taking_the_lock(
    dwight_config_text: str,
) -> None:
    """The build holds no lock; the Python GPU phase takes it around its loop."""
    block = _process_block(dwight_config_text, "BUILD_SPATIALDATA")

    assert "maxForks = params.build_spatialdata_max_forks" in block
    assert "beforeScript" in block
    assert '"${params.gpu_process_lock_enabled}" == "true"' in block
    assert 'export MERXEN_GPU_LOCK_FILE="${params.gpu_process_lock_file}"' in block
    assert "flock" not in block
    assert "exec 9" not in block


def test_gpu_profile_passes_the_gpu_into_the_build_container() -> None:
    """Without --nv the cuda device fails inside the container (loudly)."""
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "workflows" / "nextflow.config").read_text()
    gpu_profile = text[text.index("    gpu {") :]
    gpu_profile = gpu_profile[: gpu_profile.index("// compute profiles")]

    for process_name in (
        "CELLPOSE_SEGMENT",
        "ALIGN",
        "CLUSTERING_SQUIDPY_COMPUTE",
    ):
        block = _process_block(gpu_profile, process_name)
        assert "--nv" in block.split("}")[0]


def test_gpu_profile_adds_nv_to_the_build_only_for_the_gpu_device() -> None:
    """The gpu profile keeps --nv off CPU-only containers, the default build too."""
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "workflows" / "nextflow.config").read_text()
    gpu_profile = text[text.index("    gpu {") :]
    gpu_profile = gpu_profile[: gpu_profile.index("// compute profiles")]
    block = _process_block(gpu_profile, "BUILD_SPATIALDATA")

    assert "BUILD_SPATIALDATA|ALIGN" not in gpu_profile
    assert "containerOptions = {" in block
    assert "!= 'cpu'" in block
    # Only the flat fill (CPU only) keeps the GPU out of the container.
    assert "!flatFill" in block and "'flat'" in block
    assert "? '--nv' : ''" in block
    # Nothing else is added for the build: no clean environment, no new binds.
    assert "-C" not in block and "--no-home" not in block


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_resolved_bead_removal_device_lock_export_and_gpu_container() -> None:
    """Nextflow resolves the cpu device, the lock export and the --nv option."""
    workflow_dir = Path(__file__).resolve().parents[2] / "workflows"
    dwight = _resolved_config(workflow_dir, profile="dwight")
    gpu = _resolved_config(workflow_dir, profile="gpu")

    assert dwight["params"]["merscope_bead_removal_device"] == "cuda"
    assert gpu["params"]["merscope_bead_removal_device"] == "cpu"
    script = dwight["process"]["withName:BUILD_SPATIALDATA"]["beforeScript"]
    assert (
        f'export MERXEN_GPU_LOCK_FILE="{dwight["params"]["gpu_process_lock_file"]}"'
        in script
    )
    assert "flock" not in script
    assert gpu["process"]["withName:ALIGN"]["containerOptions"].endswith("--nv")
    # The build gets --nv (and a GPU queue) only for the GPU device: closures.
    build = gpu["process"]["withName:BUILD_SPATIALDATA"]
    assert "'--nv'" in build["containerOptions"]
    assert "'gpu' : 'htc'" in build["queue"]
    assert "--gpus-per-node=1" in build["clusterOptions"]


_DIRECTIVE_PROBE = """
process BUILD_SPATIALDATA {
    input:
    val x
    output:
    stdout
    script:
    \"\"\"
    echo "RESULT queue=${task.queue} cluster=[${task.clusterOptions}] nv=\\
${task.containerOptions.contains('--nv')}"
    \"\"\"
}
workflow { BUILD_SPATIALDATA(1) | view }
"""


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], "queue=htc cluster=[] nv=false"),
        (["--merscope_bead_removal_device", "cpu"], "queue=htc cluster=[] nv=false"),
        (
            ["--merscope_bead_removal_device", "cuda"],
            "queue=gpu cluster=[--gpus-per-node=1] nv=true",
        ),
        (["--merscope_bead_removal_device", "auto"], "queue=htc cluster=[] nv=true"),
        # The default hybrid fill runs on the GPU, so it keeps the GPU request.
        (
            [
                "--merscope_bead_removal_device",
                "cuda",
                "--merscope_bead_removal_fill",
                "true",
            ],
            "queue=gpu cluster=[--gpus-per-node=1] nv=true",
        ),
        # The flat fill is CPU only: no GPU queue, no --nv.
        (
            [
                "--merscope_bead_removal_device",
                "cuda",
                "--merscope_bead_removal_fill",
                "true",
                "--merscope_bead_removal_fill_mode",
                "flat",
            ],
            "queue=htc cluster=[] nv=false",
        ),
        # Without the fill the mode does not matter.
        (
            [
                "--merscope_bead_removal_device",
                "cuda",
                "--merscope_bead_removal_fill",
                "false",
                "--merscope_bead_removal_fill_mode",
                "flat",
            ],
            "queue=gpu cluster=[--gpus-per-node=1] nv=true",
        ),
    ],
)
def test_build_gets_a_gpu_queue_and_nv_only_for_the_gpu_device(
    tmp_path: Path, extra_args: list[str], expected: str
) -> None:
    """The default build stays a CPU task in the gpu profile; cuda asks for a GPU."""
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    config = Path(__file__).resolve().parents[2] / "workflows" / "nextflow.config"
    probe = tmp_path / "probe.nf"
    probe.write_text(_DIRECTIVE_PROBE)
    completed = subprocess.run(
        [
            nextflow,
            "run",
            str(probe),
            "-c",
            str(config),
            "-profile",
            "gpu",
            "--outdir",
            str(tmp_path / "out"),
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        env={**os.environ, "NXF_OFFLINE": "true"},
    )

    assert f"RESULT {expected}" in completed.stdout


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_default_standard_and_dwight_profiles_resolve_equivalently() -> None:
    """Nextflow should resolve the same Dwight resources through all entry points."""
    repo_root = Path(__file__).resolve().parents[2]
    workflow_dir = repo_root / "workflows"
    resolved = {
        name: _resolved_config(workflow_dir, profile=profile)
        for name, profile in (
            ("default", None),
            ("standard", "standard"),
            ("dwight", "dwight"),
        )
    }

    signatures = {name: _dwight_signature(config) for name, config in resolved.items()}
    assert signatures["default"] == signatures["standard"] == signatures["dwight"]
    assert signatures["default"] == {
        "cellpose_gpu": True,
        "proseg_binary": "/usr/local/bin/proseg",
        "proseg_num_threads": 32,
        "max_ram_gb": 640,
        "warn_ram_gb": 600,
        "mask_image_quantification_max_forks": 3,
        "viewer_cache_max_forks": 9,
        "gpu_process_lock_file": "/tmp/merxen-dwight-gpu.lock",
        "process_executor": "local",
        "local_executor_cpus": 72,
        "local_executor_memory": "640 GB",
        "viewer_cache_memory": "60 GB",
        "viewer_cache_process_max_forks": 9,
    }

    lock_path = signatures["default"]["gpu_process_lock_file"]
    for config in resolved.values():
        process = config["process"]
        assert "cpus" not in process
        assert "memory" not in process
        for process_name in (
            "CELLPOSE_SEGMENT",
            "CELLPOSE_NUCLEI_SEGMENT",
            "ALIGN",
            "CLUSTERING_SQUIDPY_COMPUTE",
        ):
            script = process[f"withName:{process_name}"]["beforeScript"]
            assert f'MERXEN_GPU_LOCK_FILE="{lock_path}"' in script
            assert 'exec 9<"${MERXEN_GPU_LOCK_FILE}"' in script


def test_dwight_sends_task_temporary_files_to_srv_storage() -> None:
    """Task TMPDIR must point at /srv/storage, never the small root disk's /tmp."""
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "workflows" / "conf" / "dwight.config").read_text()
    assert re.search(r'task_tmp_dir\s*=\s*"/srv/storage/MerXen/tmp"', text)
    assert re.search(r"\benv\s*\{\s*TMPDIR\s*=\s*params\.task_tmp_dir\s*\}", text)


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_dwight_resolves_task_tmpdir_to_srv_storage() -> None:
    """The resolved dwight profile should export TMPDIR on /srv/storage to tasks."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved = _resolved_config(repo_root / "workflows", profile="dwight")
    assert resolved["env"]["TMPDIR"] == "/srv/storage/MerXen/tmp"
    assert resolved["params"]["task_tmp_dir"] == "/srv/storage/MerXen/tmp"
