"""The direct pciSeq command keeps computation in its requested interpreter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from merxen.cli.run_pciseq import pciseq_command, publish_pciseq_viewer_command


def test_pciseq_cli_runs_compute_in_isolated_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.analysis import pciseq
    from merxen.cli import run_pciseq

    config_path = tmp_path / "config.json"
    output = tmp_path / "output"
    interpreter = tmp_path / "runtime-python"
    interpreter.touch()
    config_path.write_text(
        json.dumps(
            {
                "pair_id": "pair",
                "sample_id": "sample",
                "platform": "XENIUM",
                "segmentation": "reseg",
                "source_h5ad": "clustered.h5ad",
                "spatialdata_path": "latest_spatialdata.zarr",
                "reference_path": "reference.csv",
                "output_dir": str(output),
            }
        )
    )
    events: list[str] = []

    def prepare(config: Any, destination: Path) -> Path:
        events.append("prepare")
        assert destination == output / "prepared"
        return destination / "manifest.json"

    def compute(arguments: list[str], **kwargs: Any) -> None:
        events.append("compute")
        assert arguments[:3] == [str(interpreter), "-m", "merxen.pciseq_compute"]
        assert kwargs["check"] is True
        assert (
            Path(run_pciseq.__file__).resolve().parents[2].as_posix()
            in kwargs["env"]["PYTHONPATH"]
        )

    def finalize(
        config: Any, prepared: Path, computed: Path, destination: Path
    ) -> dict:
        events.append("finalize")
        assert prepared == output / "prepared"
        assert computed == output / "computed"
        assert destination == output
        return {"annotated_h5ad": output / "annotated.h5ad"}

    monkeypatch.setattr(pciseq, "prepare_pciseq", prepare)
    monkeypatch.setattr(pciseq, "finalize_pciseq", finalize)
    monkeypatch.setattr(run_pciseq.subprocess, "run", compute)
    result = CliRunner().invoke(
        pciseq_command,
        ["--config", str(config_path), "--compute-python", str(interpreter)],
    )
    assert result.exit_code == 0, result.output
    assert events == ["prepare", "compute", "finalize"]
    assert "annotated.h5ad" in result.output

    def fail_compute(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, "runtime-python")

    events.clear()
    monkeypatch.setattr(run_pciseq.subprocess, "run", fail_compute)
    result = CliRunner().invoke(pciseq_command, ["--config", str(config_path)])
    assert isinstance(result.exception, subprocess.CalledProcessError)
    assert events == ["prepare"]


def test_publish_completed_pciseq_does_not_rerun_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.analysis import pciseq_viewer
    from merxen.cli import main, run_pciseq

    results_dir = tmp_path / "completed"
    results_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "pair_id": "pair",
                "sample_id": "sample",
                "platform": "XENIUM",
                "segmentation": "proseg_hybrid",
                "source_h5ad": "clustered.h5ad",
                "spatialdata_path": "latest_spatialdata.zarr",
                "reference_path": "reference.csv",
            }
        )
    )
    calls = []

    def publish(config: Any, directory: Path) -> dict:
        calls.append((config.sample_id, config.segmentation, directory))
        return {"source_key": "pciseq_proseg_hybrid", "native_spots": 10}

    def unexpected_model(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Publishing completed assignments must not invoke computation")

    monkeypatch.setattr(pciseq_viewer, "publish_pciseq_viewer", publish)
    monkeypatch.setattr(run_pciseq.subprocess, "run", unexpected_model)
    result = CliRunner().invoke(
        publish_pciseq_viewer_command,
        ["--config", str(config_path), "--results-dir", str(results_dir)],
    )
    assert result.exit_code == 0, result.output
    assert calls == [("sample", "proseg_hybrid", results_dir)]
    assert json.loads(result.output)["source_key"] == "pciseq_proseg_hybrid"
    assert "publish-pciseq-viewer" in main.commands
