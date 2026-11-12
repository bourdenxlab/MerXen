"""Frozen preparations resume only the fit and publication processes."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_resume(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("nextflow") is None:
        pytest.skip("Nextflow is unavailable")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    native = tmp_path / "native.zarr"
    native.mkdir()
    source = tmp_path / "source.h5ad"
    source.touch()
    identity = {
        "pair_id": "sample",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
    }
    (prepared / "manifest.json").write_text(
        json.dumps({**identity, "spot_count": 3, "cell_count": 2})
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {**identity, "source_h5ad": str(source), "spatialdata_path": str(native)}
        )
    )
    limits = tmp_path / "limits.config"
    limits.write_text("process.resourceLimits = [cpus: 1, memory: 1.GB]\n")
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(ROOT / "workflows/pciseq_resume.nf"),
            "-c",
            str(limits),
            "-without-conda",
            "-ansi-log",
            "false",
            "-stub-run",
            "--pciseq_prepared_dir",
            str(prepared),
            "--pciseq_prepared_config",
            str(config),
            "--outdir",
            str(tmp_path / "results"),
            *arguments,
        ],
        cwd=tmp_path,
        env={**os.environ, "MERXEN_DWIGHT_SITE_CONFIG": "none"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.nextflow
@pytest.mark.parametrize(
    ("arguments", "compute"),
    [
        ((), "PCISEQ_COMPUTE"),
        (("--pciseq_memory_mode", "batched"), "PCISEQ_COMPUTE_BATCHED"),
    ],
)
def test_prepared_resume_skips_reference_and_mask_preparation(
    tmp_path: Path, arguments: tuple[str, ...], compute: str
) -> None:
    """Only the fit and publication run; the memory mode picks the fit process.

    The default keeps PCISEQ_COMPUTE, whose script text (and so every cached
    task) is unchanged; the experimental batched mode has its own process.
    """
    result = _run_resume(tmp_path, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    # The trace lists every task the run executed. Console wording differs
    # between Nextflow releases, so it is not used to count processes.
    trace = tmp_path / "results/nextflow/trace.tsv"
    with trace.open(newline="") as handle:
        executed = sorted(
            row["process"] for row in csv.DictReader(handle, delimiter="\t")
        )
    assert executed == sorted([compute, "PCISEQ_FINALIZE"])
    assert (
        tmp_path
        / "results/sample/proseg_hybrid/pciseq/pciseq_out/merscope/stub_manifest.json"
    ).is_file()


@pytest.mark.nextflow
def test_prepared_resume_rejects_invalid_memory_settings(tmp_path: Path) -> None:
    """Mode and batch sizes are refused before the compute task starts."""
    result = _run_resume(
        tmp_path,
        "--pciseq_memory_mode",
        "Batched",
        "--pciseq_spot_batch_size",
        "0",
        "--pciseq_cell_batch_size",
        "2e4",
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "pciseq_memory_mode must be standard or batched" in output
    assert "pciseq_spot_batch_size must be a positive integer" in output
    assert "pciseq_cell_batch_size must be a positive integer" in output
    assert not list((tmp_path / "work").glob("*/*"))
