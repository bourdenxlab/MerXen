"""Frozen preparations resume only the fit and publication processes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.nextflow
def test_prepared_resume_skips_reference_and_mask_preparation(tmp_path: Path) -> None:
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
    result = subprocess.run(
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
        ],
        cwd=tmp_path,
        env={**os.environ, "MERXEN_DWIGHT_SITE_CONFIG": "none"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    submitted = [
        line for line in result.stdout.splitlines() if "Submitted process" in line
    ]
    assert len(submitted) == 2
    assert any("PCISEQ_COMPUTE" in line for line in submitted)
    assert any("PCISEQ_FINALIZE" in line for line in submitted)
    assert not any(
        "PCISEQ_REFERENCE" in line or "PCISEQ_PREPARE" in line for line in submitted
    )
    assert (
        tmp_path
        / "results/sample/proseg_hybrid/pciseq/pciseq_out/merscope/stub_manifest.json"
    ).is_file()
