"""Preview the optional acceptance workflow without scheduling GPU work."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / "workflows" / "acceptance" / "hitl0_tile_check.nf"


def _preview(
    tmp_path: Path, protected: Path, output: Path
) -> subprocess.CompletedProcess:
    args = {
        "source_root": REPO_ROOT,
        "python": sys.executable,
        "config": tmp_path / "config.json",
        "source_zarr": tmp_path / "source.zarr",
        "baseline_mask": protected / "mask.npy",
        "baseline_stats": protected / "stats.json",
        "protected_root": protected,
        "gpu_lock_file": tmp_path / "gpu.lock",
        "image_key": "native",
        "outdir": output,
    }
    command = ["nextflow", "run", str(WORKFLOW), "-preview", "-ansi-log", "false"]
    for name, value in args.items():
        command.extend([f"--{name}", str(value)])
    return subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"},
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
@pytest.mark.parametrize("use_symlink", [False, True])
def test_preview_rejects_protected_output_before_scheduling(
    tmp_path: Path, use_symlink: bool
) -> None:
    """A symlink alias must not let publishDir write into a baseline directory."""
    protected = tmp_path / "published"
    protected.mkdir()
    if use_symlink:
        alias = tmp_path / "alias"
        alias.symlink_to(protected, target_is_directory=True)
        output = alias / "trial"
    else:
        output = protected / "trial"
    result = _preview(tmp_path, protected, output)
    assert result.returncode != 0
    assert "outside --protected_root" in result.stdout + result.stderr
    assert list(protected.iterdir()) == []


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_preview_accepts_separate_output_without_running_gpu(tmp_path: Path) -> None:
    """Preview should compile the harness without reading images or loading a model."""
    output = tmp_path / "trial"
    result = _preview(tmp_path, tmp_path / "published", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not output.exists()
