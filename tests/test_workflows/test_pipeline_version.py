"""The MerXen release a run records (workflows/lib/PipelineVersion.groovy).

``main.nf`` logs the release and appends it to
``<outdir>/nextflow/merxen_versions.jsonl`` (docs/versioning.md). The harness
calls the same class against the real ``workflows/nextflow.config``, so the
manifest version it reports is the one bump-my-version maintains.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merxen import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"

HARNESS = """
workflow {
    def info = PipelineVersion.info(workflow, nextflow.version.toString())
    println "BANNER\\t" + PipelineVersion.banner(info)
    println "PREVIEW\\t" + PipelineVersion.isPreview(workflow)
    if (!PipelineVersion.isPreview(workflow)) {
        def problem = PipelineVersion.record(info, file(params.outdir).toAbsolutePath())
        println "PROBLEM\\t" + (problem ?: "")
    }
}
"""


def _nextflow() -> str:
    nextflow = shutil.which("nextflow")
    if nextflow is None:
        pytest.skip("nextflow is not installed")
    return nextflow


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "h.nf").write_text(HARNESS)
    (project / "lib").symlink_to(WORKFLOWS / "lib", target_is_directory=True)
    return project


def _run(project: Path, outdir: Path, *extra: str) -> dict[str, str]:
    result = subprocess.run(
        [
            _nextflow(),
            "run",
            "h.nf",
            "-c",
            str(WORKFLOWS / "nextflow.config"),
            "--outdir",
            str(outdir),
            "-ansi-log",
            "false",
            *extra,
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={
            "PATH": "/usr/bin:/bin:" + str(Path(_nextflow()).parent),
            "HOME": str(project),
            "NXF_HOME": str(Path.home() / ".nextflow"),
            "NXF_OFFLINE": "true",
            "MERXEN_SITE_CONFIG": "none",
            "MERXEN_DWIGHT_SITE_CONFIG": "none",
        },
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return dict(
        line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line
    )


@pytest.mark.nextflow
def test_a_run_outside_git_records_the_manifest_version(tmp_path: Path) -> None:
    project = _project(tmp_path)
    outdir = tmp_path / "results"
    first = _run(project, outdir)
    assert first["BANNER"] == f"MerXen {__version__}"
    assert first["PROBLEM"] == ""
    _run(project, outdir, "-resume")
    lines = (outdir / "nextflow" / "merxen_versions.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert {r["merxen_version"] for r in records} == {__version__}
    assert [r["resume"] for r in records] == [False, True]
    assert records[0]["git_describe"] is None
    assert records[0]["nextflow_version"]


@pytest.mark.nextflow
def test_a_checkout_is_described_and_dirty_edits_show(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _git(project, "init", "-q")
    _git(project, "add", "h.nf")
    _git(project, "commit", "-q", "-m", "init")
    _git(project, "tag", "v9.9.9")
    clean = _run(project, tmp_path / "clean")
    assert clean["BANNER"] == f"MerXen {__version__} (checkout v9.9.9)"
    (project / "h.nf").write_text(HARNESS + "\n// edited\n")
    dirty = _run(project, tmp_path / "dirty")
    assert dirty["BANNER"] == f"MerXen {__version__} (checkout v9.9.9-dirty)"


@pytest.mark.nextflow
def test_preview_writes_no_record(tmp_path: Path) -> None:
    project = _project(tmp_path)
    outdir = tmp_path / "results"
    output = _run(project, outdir, "-preview")
    assert output["PREVIEW"] == "true"
    assert not (outdir / "nextflow" / "merxen_versions.jsonl").exists()


def test_main_nf_logs_and_records_the_version_first() -> None:
    """The version is logged before any launch check can stop the run."""
    main = (WORKFLOWS / "main.nf").read_text()
    body = main[main.index("\nworkflow {\n") :]
    assert body.index("PipelineVersion.banner") < body.index("SiteConfigCheck")
    assert "PipelineVersion.record(" in body
    assert "PipelineVersion.isPreview(workflow)" in body
