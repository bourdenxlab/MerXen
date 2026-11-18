"""CLI commands of the register step (CliRunner, no ``cli/__init__`` needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from test_manual_registration._mr_helpers import a2_store
from test_manual_registration.test_field_fit import (
    _inputs,
    _payload,
    _suggestions,
)

from merxen.cli import run_manual_registration_register as cli
from merxen.cli.run_manual_registration_register import (
    manual_registration_check_command,
    manual_registration_compare_command,
    manual_registration_fit_command,
    manual_registration_register_command,
)
from merxen.manual_registration.check import CheckRun, StaleLabellingError
from merxen.manual_registration.config import ManualRegistrationCheckConfig
from merxen.manual_registration.field_fit import build_landmark_field
from merxen.manual_registration.simulate import FieldPhantom, make_field_phantom


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


# ---------------------------------------------------------------------------
# manual-registration-check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phantom() -> FieldPhantom:
    return make_field_phantom(seed=11, n_clicks=45, n_gross=2, n_medium=2)


def _check(*args: str) -> Any:
    return CliRunner().invoke(manual_registration_check_command, list(args))


def _check_config_file(tmp_path: Path, store: Path) -> Path:
    config = ManualRegistrationCheckConfig(
        dataset_name="SYN1",
        prepared_store_path=store,
        output_dir=tmp_path / "check",
        n_suggestions=3,
    )
    path = tmp_path / "check_config.json"
    path.write_text(config.model_dump_json())
    return path


def _finished_run(phantom: FieldPhantom, tmp_path: Path, *, attr: bool) -> CheckRun:
    similarity, store, labelling, settings = _inputs(phantom, disabled=(5,))
    report, _ = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops={},
        image_check=False,
        windows=False,
        run_id="X_p",
    )
    return CheckRun(
        paths={"json": tmp_path / "a.json", "png": tmp_path / "a.png"},
        payload=_payload(report, store, labelling, settings),
        report=report,
        suggestions=_suggestions(report),
        attr_written=attr,
        warnings=list(report.warnings),
    )


def test_check_config_excludes_every_other_option(tmp_path: Path) -> None:
    store = a2_store(tmp_path)
    config = _check_config_file(tmp_path, store)
    result = _check("--config", str(config), "--store", str(store))
    assert result.exit_code == 2
    assert "--config excludes --store" in result.output
    result = _check("--config", str(config), "--no-windows", "--suggestions", "3")
    assert result.exit_code == 2
    assert "--no-windows" in result.output and "--suggestions" in result.output


def test_check_direct_mode_needs_store_and_out_dir(tmp_path: Path) -> None:
    store = a2_store(tmp_path)
    assert _check("--out-dir", str(tmp_path)).exit_code == 2
    result = _check("--store", str(store))
    assert result.exit_code == 2
    assert "--store and --out-dir are required" in result.output


def test_check_direct_mode_takes_the_dataset_name_from_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phantom: FieldPhantom
) -> None:
    store = a2_store(tmp_path)
    seen: list[ManualRegistrationCheckConfig] = []

    def fake(config: ManualRegistrationCheckConfig) -> CheckRun:
        seen.append(config)
        return _finished_run(phantom, tmp_path, attr=False)

    monkeypatch.setattr(cli, "execute_check", fake)
    upload = tmp_path / "upload.json"
    upload.write_text("{}")
    result = _check(
        "--store",
        str(store),
        "--out-dir",
        str(tmp_path / "out"),
        "--labelling",
        str(upload),
        "--preset",
        "same_section",
        "--settings",
        "field_outlier_p=0.02",
        "--channel-roles",
        "at8=AF568;map2=AF647;dapi=DAPI",
        "--no-image-check",
        "--windows",
        "--no-write-attr",
        "--suggestions",
        "7",
    )
    assert result.exit_code == 0, result.output
    (config,) = seen
    assert config.dataset_name == "SYN1"
    assert config.preset == "same_section"
    assert config.labelling_path == upload
    assert config.row_settings == "field_outlier_p=0.02"
    assert config.channel_roles == "at8=AF568;map2=AF647;dapi=DAPI"
    assert config.image_check is False and config.windows is True
    assert config.write_attr is False and config.n_suggestions == 7
    assert config.output_dir == tmp_path / "out"


def test_check_defaults_in_direct_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phantom: FieldPhantom
) -> None:
    store = a2_store(tmp_path)
    seen: list[ManualRegistrationCheckConfig] = []

    def fake(config: ManualRegistrationCheckConfig) -> CheckRun:
        seen.append(config)
        return _finished_run(phantom, tmp_path, attr=True)

    monkeypatch.setattr(cli, "execute_check", fake)
    result = _check("--store", str(store), "--out-dir", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    config = seen[0]
    assert config.preset == "cortex_serial" and config.labelling_path is None
    assert config.image_check is None and config.windows is None
    assert config.write_attr is True and config.n_suggestions == 20


def test_check_config_mode_runs_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phantom: FieldPhantom
) -> None:
    store = a2_store(tmp_path)
    seen: list[ManualRegistrationCheckConfig] = []

    def fake(config: ManualRegistrationCheckConfig) -> CheckRun:
        seen.append(config)
        return _finished_run(phantom, tmp_path, attr=True)

    monkeypatch.setattr(cli, "execute_check", fake)
    result = _check("--config", str(_check_config_file(tmp_path, store)))
    assert result.exit_code == 0, result.output
    assert seen[0].n_suggestions == 3 and seen[0].dataset_name == "SYN1"


def test_check_prints_the_table_summary_suggestions_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phantom: FieldPhantom
) -> None:
    store = a2_store(tmp_path)
    run = _finished_run(phantom, tmp_path, attr=True)
    monkeypatch.setattr(cli, "execute_check", lambda _config: run)
    result = _check("--store", str(store), "--out-dir", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    header = next(line for line in lines if "rigid_LOO_um" in line)
    assert header.split() == [
        "id",
        "enabled",
        "class",
        "verdict",
        "LOO_um",
        "p",
        "weight",
        "rigid_LOO_um",
    ]
    rows = [
        ln for ln in lines if ln.split() and ln.split()[1:2] in (["True"], ["False"])
    ]
    assert len(rows) == len(phantom.reference_xy_um)
    disabled = next(row for row in rows if row.split()[0] == "5")
    assert "disabled" in disabled and disabled.split()[1] == "False"
    assert any("likely_misclick" in row for row in rows)
    summary = next(line for line in lines if line.startswith("field:"))
    assert "clicks, 0 windows" in summary and "LOO median/p90 um" in summary
    assert "classes {" in summary and "sd_um 20.0/40.0" in summary
    assert "suggested next clicks" in result.output
    assert f"- json: {tmp_path / 'a.json'}" in result.output
    assert "check attribute written: yes" in result.output


def test_check_exits_three_when_the_labelling_went_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phantom: FieldPhantom
) -> None:
    store = a2_store(tmp_path)
    error = StaleLabellingError("the labelling changed while the check ran")
    error.run = _finished_run(phantom, tmp_path, attr=False)

    def stale(_config: Any) -> CheckRun:
        raise error

    monkeypatch.setattr(cli, "execute_check", stale)
    result = _check("--store", str(store), "--out-dir", str(tmp_path / "out"))
    assert result.exit_code == 3
    assert "the labelling changed while the check ran" in result.output
    assert "check attribute written: no" in result.output


def test_check_exits_one_without_roles_when_the_image_check_is_on(
    tmp_path: Path,
) -> None:
    store = a2_store(tmp_path)
    result = _check("--store", str(store), "--out-dir", str(tmp_path / "out"))
    assert result.exit_code == 1
    assert "the image check and tissue windows need channel roles" in result.output
    assert "--channel-roles 'at8=AF568;map2=AF647;dapi=DAPI'" in result.output
    assert "[SYN1_cortex_serial]" in result.output


def test_check_exits_one_on_unusable_landmarks(tmp_path: Path) -> None:
    store = a2_store(tmp_path, disabled=(1, 2, 3, 4))
    result = _check(
        "--store",
        str(store),
        "--out-dir",
        str(tmp_path / "out"),
        "--no-image-check",
        "--no-windows",
    )
    assert result.exit_code == 1
    assert "6 enabled landmark pair(s); at least 8 are required" in result.output


def test_check_exits_one_on_a_labelling_file_without_a_labelling(
    tmp_path: Path,
) -> None:
    store = a2_store(tmp_path)
    upload = tmp_path / "upload.json"
    upload.write_text('{"zarr_format": 3}')
    result = _check(
        "--store",
        str(store),
        "--out-dir",
        str(tmp_path / "out"),
        "--labelling",
        str(upload),
        "--no-image-check",
        "--no-windows",
    )
    assert result.exit_code == 1
    assert "upload.json" in result.output and "no labelling found" in result.output
