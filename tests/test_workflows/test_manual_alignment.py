"""Render the real manual alignment module and check bounded-stage selection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NEXTFLOW = shutil.which("nextflow")


@pytest.mark.nextflow
@pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")
def test_manual_module_stages_uploaded_json_and_does_not_select_analysis(
    tmp_path: Path,
) -> None:
    """Run its shell with a stand-in CLI; the Python backend has real-Zarr tests."""
    assert NEXTFLOW is not None
    (tmp_path / "lib").symlink_to(ROOT / "workflows/lib", target_is_directory=True)
    store = tmp_path / "DAPI pair.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    landmarks = tmp_path / "uploaded root.json"
    landmarks.write_text('{"finished": true}')
    binary = tmp_path / "bin"
    binary.mkdir()
    cli = binary / "merxen"
    cli.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "mkdir -p align_out\ncp align_config.json align_out/rendered.json\n"
        "cp landmarks.json align_out/upload.json\n"
    )
    cli.chmod(0o755)
    harness = tmp_path / "harness.nf"
    harness.write_text(
        "include { MANUAL_ALIGN } from '"
        + str(ROOT / "workflows/modules/manual_alignment")
        + "'\ninclude { rowSampleSettings; "
        "appendAlignmentAnnotationPreflightChecks } from '"
        + str(ROOT / "workflows/main")
        + "'\nworkflow {\n"
        "  def row = [pair_id: 'HD1', alignment_manual_store: params.store, "
        "alignment_manual_landmarks: params.landmarks]\n"
        "  def settings = rowSampleSettings(row, params)\n"
        "  def errors = []\n"
        "  appendAlignmentAnnotationPreflightChecks(errors, row, settings, params)\n"
        "  new File(params.selection).text = groovy.json.JsonOutput.toJson(["
        "stages: settings.selected_stages, segment: settings.run_segment, "
        "cluster: settings.run_clustering_squidpy, errors: errors])\n"
        "  MANUAL_ALIGN(channel.of(tuple('HD1', 'native moving', 'native fixed', "
        "file(params.store), file(params.landmarks))))\n}\n"
    )
    options = {
        "store": str(store),
        "landmarks": str(landmarks),
        "outdir": str(tmp_path / "out"),
        "selection": str(tmp_path / "selection.json"),
        "alignment_backend": "manual_landmarks",
        "enable_alignment": True,
        "analysis_mode": "paired",
        "species": "human",
        "start_stage": "align",
        "stop_stage": "align_qc",
        "clustering_squidpy_mode": "legacy",
        "alignment_manual_settings": "landmark_outlier_policy=keep",
    }
    parameters = tmp_path / "params.json"
    parameters.write_text(json.dumps(options))
    result = subprocess.run(
        [
            NEXTFLOW,
            "run",
            str(harness),
            "-c",
            str(ROOT / "workflows/nextflow.config"),
            "-params-file",
            str(parameters),
            "-work-dir",
            str(tmp_path / "work"),
            "-ansi-log",
            "false",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "NXF_DISABLE_CHECK_LATEST": "true",
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection == {
        "stages": ["align", "align_qc"],
        "segment": False,
        "cluster": False,
        "errors": [],
    }
    output = tmp_path / "out/HD1/alignment/align_out"
    config = json.loads((output / "rendered.json").read_text())
    assert config["backend"] == "manual_landmarks"
    assert config["manual"]["prepared_store"] == store.name
    assert config["manual"]["labelling_path"] == "landmarks.json"
    assert config["manual"]["settings"] == options["alignment_manual_settings"]
    assert json.loads((output / "upload.json").read_text()) == {"finished": True}
