"""``nextflow lint`` finds no errors in the pipeline scripts and configs.

The linter resolves the Groovy classes in ``lib/`` from its project
directory, which defaults to the launch directory. Run from the repository
root without ``-project-dir``, it cannot see ``workflows/lib`` and reports
every ``Annotation*`` class used in ``main.nf`` as not defined, so the
canonical command is::

    nextflow lint -project-dir workflows workflows/

``scripts/run_ci_checks.sh`` runs the same command when Nextflow is
installed. Warnings are allowed; errors are not.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
NEXTFLOW = shutil.which("nextflow")


@pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")
def test_nextflow_lint_reports_no_errors(tmp_path: Path) -> None:
    """Lint ``workflows/`` with its project dir, as the CI script does."""
    assert NEXTFLOW is not None
    env = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true", "NXF_ANSI_LOG": "false"}
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(tmp_path / "nextflow.log"),
            "lint",
            "-project-dir",
            str(WORKFLOWS),
            "-o",
            "json",
            str(WORKFLOWS),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = completed.stdout
    report = json.loads(output[output.index("{") :])

    errors = [
        f"{Path(item['filename']).relative_to(REPO_ROOT)}:{item['startLine']}: "
        f"{item['message']}"
        for item in report["errors"]
    ]
    assert report["summary"]["errors"] == 0, "\n".join(errors) or completed.stderr
    assert report["summary"]["filesWithoutErrors"] > 0
