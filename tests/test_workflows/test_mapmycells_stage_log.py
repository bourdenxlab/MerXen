"""The launch log names only the stages a map_first row really runs.

A mouse row runs map_first annotation, where the standalone mapmycells stage
is skipped unless ``--annotation_mode_mapmycells_stage legacy``. The row
settings keep mapmycells in ``selected_stages`` (they are part of task hashes),
so the log line and the preflight message must drop it, and a row that names
mapmycells as its stop stage must be warned that the stage does not run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_map_first_row_logs_skipped_mapmycells_stage(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    sheet = tmp_path / "samplesheet.csv"
    sheet.write_text(
        "pair_id,analysis_mode,merscope_dir,merscope_z_range,stop_stage\n"
        "PAIR_STOP_MMC,merscope,raw,1-6,mapmycells\n"
        "PAIR_STOP_CLUSTER,merscope,raw,1-6,clustering_squidpy\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MERXEN_")}
    env.update(
        {
            "MERXEN_SITE_CONFIG": "/dev/null",
            "MERXEN_DWIGHT_SITE_CONFIG": "/dev/null",
            "NXF_OFFLINE": "true",
            "NXF_ANSI_LOG": "false",
        }
    )
    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_WORKFLOWS / "main.nf"),
            "--samplesheet",
            str(sheet),
            "--outdir",
            str(tmp_path / "results"),
            "--species",
            "mouse",
            "-w",
            str(tmp_path / "work"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    # The rows point at an empty raw folder, so the preflight refuses them.
    assert result.returncode != 0, output
    run_log = (tmp_path / ".nextflow.log").read_text()
    stages = dict(
        re.findall(r"Sample (\w+): .*; selected stages=(.*)$", run_log, re.MULTILINE)
    )
    assert set(stages) == {"PAIR_STOP_MMC", "PAIR_STOP_CLUSTER"}, output
    for line in stages.values():
        assert line.endswith("-> clustering_squidpy"), line
        assert "mapmycells" not in line, line
    warned = set(
        re.findall(
            r"Sample (\w+): the standalone mapmycells stage does not run", run_log
        )
    )
    assert warned == {"PAIR_STOP_MMC"}, output
    for message in re.findall(r"Preflight checks failed for sample .*", run_log):
        assert "mapmycells" not in message, message
    assert not list((tmp_path / "work").glob("*/*")), "a task started:\n" + output
