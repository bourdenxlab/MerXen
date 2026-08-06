"""Regression tests for the default Dwight workstation execution profile."""

from __future__ import annotations

import json
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
    assert (
        dwight_config_text.count(
            'MERXEN_GPU_LOCK_FILE="${params.gpu_process_lock_file}"'
        )
        == 5
    )
    assert "${PWD}/.merxen_gpu.lock" not in combined_config_text


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
