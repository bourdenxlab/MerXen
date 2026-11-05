"""CLI commands of the register step (CliRunner, no ``cli/__init__`` needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from test_manual_registration._mr_helpers import a2_store

from merxen.cli import run_manual_registration_register as cli
from merxen.cli.run_manual_registration_register import (
    manual_registration_compare_command,
    manual_registration_fit_command,
    manual_registration_register_command,
)


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(manual_registration_fit_command, list(args))


def test_fit_prints_the_table_and_writes_json(tmp_path: Path) -> None:
    store = a2_store(tmp_path)
    out_json = tmp_path / "fit.json"
    result = _invoke(
        "--store", str(store), "--preset", "same_section", "--json", str(out_json)
    )
    assert result.exit_code == 0, result.output
    header = next(line for line in result.output.splitlines() if "residual_um" in line)
    assert header.split() == [
        "id",
        "enabled",
        "used",
        "outlier",
        "outside_crop",
        "residual_um",
        "loo_um",
    ]
    rows = [
        line
        for line in result.output.splitlines()
        if line.strip() and line.split()[0].isdigit()
    ]
    assert len(rows) == 10
    assert "flagged as outlier and excluded: 5" in result.output
    assert "outlier floor: 10 µm (preset same_section); viewer: not saved" in (
        result.output
    )
    payload = json.loads(out_json.read_text())
    assert payload["fit"]["n_used"] == 9
    assert len(payload["landmarks"]) == 10
    assert payload["landmarks"][4]["outlier"] is True


def test_fit_accepts_a_draft_labelling(tmp_path: Path) -> None:
    store = a2_store(tmp_path, status="draft")
    result = _invoke("--store", str(store), "--preset", "same_section")
    assert result.exit_code == 0, result.output
    assert "labelling status: draft" in result.output


def test_fit_exits_one_on_a_fit_error(tmp_path: Path) -> None:
    store = a2_store(tmp_path, disabled=(1, 2, 3))
    out_json = tmp_path / "fit.json"
    result = _invoke(
        "--store", str(store), "--preset", "same_section", "--json", str(out_json)
    )
    assert result.exit_code == 1
    assert "7 enabled" in result.output
    assert "error" in json.loads(out_json.read_text())


def test_fit_rejects_unknown_settings(tmp_path: Path) -> None:
    store = a2_store(tmp_path)
    result = _invoke(
        "--store", str(store), "--preset", "same_section", "--settings", "foo=1"
    )
    assert result.exit_code != 0
    assert "unknown key" in result.output


def test_register_command_passes_the_config_and_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(config: Any, *, force_rerun: bool = False) -> dict[str, Path]:
        seen["config"] = config
        seen["force"] = force_rerun
        return {
            "result_store": tmp_path / "x.zarr",
            "summary_json": tmp_path / "s.json",
        }

    monkeypatch.setattr(cli, "run_manual_registration", fake_run)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset_name": "SYN1",
                "preset": "serial_section",
                "run_label": "a",
                "prepared_store_path": str(tmp_path / "p.zarr"),
                "result_dir": str(tmp_path / "r"),
                "output_dir": str(tmp_path / "o"),
                "row_settings": "nucleus_qc=false",
            }
        )
    )
    result = CliRunner().invoke(
        manual_registration_register_command,
        ["--config", str(config), "--force-rerun"],
    )
    assert result.exit_code == 0, result.output
    assert seen["force"] is True
    assert seen["config"].run_id == "SYN1_serial_section_a"
    assert seen["config"].row_settings == "nucleus_qc=false"
    assert "Manual registration complete:" in result.output
    assert f"- result_store: {tmp_path / 'x.zarr'}" in result.output


def test_compare_command_echoes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_compare(config: Any) -> dict[str, Path]:
        seen["config"] = config
        return {"comparison_csv": tmp_path / "c.csv"}

    monkeypatch.setattr(cli, "compare_runs", fake_compare)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset_name": "SYN1",
                "summary_paths": [str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                "output_dir": str(tmp_path / "o"),
                "task_output_dir": str(tmp_path / "t"),
            }
        )
    )
    result = CliRunner().invoke(
        manual_registration_compare_command, ["--config", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert seen["config"].dataset_name == "SYN1"
    assert len(seen["config"].summary_paths) == 2
    assert f"- comparison_csv: {tmp_path / 'c.csv'}" in result.output
