"""The manual-registration-prepare command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from test_manual_registration.test_prepare import (
    MOV,
    MOV_NAMES,
    _config,
    _inputs,
    _labelling,
    _write_strip,
)

from merxen.cli import run_manual_registration_prepare as cli_module
from merxen.cli.run_manual_registration_prepare import (
    manual_registration_prepare_command,
)
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration.schema import LABELLING_ATTR, write_root_attr


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "config.json"
    path.write_text(_config(tmp_path, **overrides).model_dump_json())
    return path


@pytest.mark.integration
def test_cli_builds_then_reports_unchanged(tmp_path: Path) -> None:
    _inputs(tmp_path)
    config = _write_config(tmp_path)
    runner = CliRunner()

    first = runner.invoke(
        manual_registration_prepare_command, ["--config", str(config)]
    )

    assert first.exit_code == 0, first.output
    assert "Prepared store (built)" in first.output
    assert "Open in the viewer" in first.output
    assert "- moving:" in first.output
    assert "(written by prepare)" in first.output
    assert "(used in place)" in first.output

    second = runner.invoke(
        manual_registration_prepare_command, ["--config", str(config)]
    )
    assert second.exit_code == 0, second.output
    assert "Prepared store (unchanged)" in second.output


@pytest.mark.integration
def test_cli_refusal_is_a_clean_error(tmp_path: Path) -> None:
    _, mov = _inputs(tmp_path)
    config = _write_config(tmp_path)
    runner = CliRunner()
    runner.invoke(manual_registration_prepare_command, ["--config", str(config)])
    store = tmp_path / "prepared" / "T1_prepared.zarr"
    with spatialdata_write_lock(store):
        write_root_attr(store, LABELLING_ATTR, _labelling(store))
    other = np.random.default_rng(3).integers(0, 4000, MOV.shape, dtype=np.uint16)
    _write_strip(mov, other, MOV_NAMES, 0.4)

    result = runner.invoke(
        manual_registration_prepare_command, ["--config", str(config)]
    )

    assert result.exit_code == 1
    assert result.output.startswith("Error: [T1]")
    assert isinstance(result.exception, SystemExit)


def test_cli_force_flag_sets_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.json"
    record = {
        "ome_tiff": {"path": "x.ome.tif", "written_by_prepare": False},
        "channel_names": ["DAPI"],
    }
    summary.write_text(
        json.dumps(
            {
                "status": "built",
                "images": {"reference": record, "moving": record},
                "warnings": ["something to note"],
            }
        )
    )
    seen: list[bool] = []

    def fake_run(cfg: Any) -> dict[str, Path]:
        seen.append(cfg.force)
        return {"store": tmp_path / "store.zarr", "summary": summary}

    monkeypatch.setattr(cli_module, "run_manual_registration_prepare", fake_run)
    config = _write_config(tmp_path)
    runner = CliRunner()

    forced = runner.invoke(
        manual_registration_prepare_command, ["--config", str(config), "--force"]
    )
    plain = runner.invoke(
        manual_registration_prepare_command, ["--config", str(config)]
    )

    assert forced.exit_code == 0, forced.output
    assert plain.exit_code == 0, plain.output
    assert seen == [True, False]
    assert "Warning: something to note" in forced.output
