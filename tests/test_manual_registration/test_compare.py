"""Comparison of several runs of one pair."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from merxen.manual_registration.compare import (
    COMPARISON_COLUMNS,
    compare_runs,
    comparison_row,
)
from merxen.manual_registration.config import ManualRegistrationCompareConfig
from merxen.manual_registration.schema import RESULT_SCHEMA


def _summary(
    run_id: str, preset: str, *, nucleus: bool, tissue: float
) -> dict[str, Any]:
    stage = {"n": 3, "median": 1.0, "p90": 2.0, "max": 3.0}
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "dataset_name": "SYN1",
        "run_id": run_id,
        "preset": preset,
        "run_label": None,
        "fit": {
            "n_enabled": 12,
            "n_used": 11,
            "rms_um": 4.5,
            "scale_ratio": 1.001,
            "rotation_deg": 178.9,
        },
        "refinement": {
            "status": "applied",
            "max_shift_um": 3.0,
            "rotation_deg": 0.1,
            "scale_change": 0.002,
        },
        "nonrigid": {
            "status": "applied",
            "confident_fraction": 0.4,
            "pearson_before": 0.5,
            "pearson_after": 0.6,
            "final_flow": {"min_jacobian": 0.8, "max_jacobian": 1.2},
        },
        "tissue_qc": {
            "pearson": {
                "coarse": 0.1,
                "landmark": 0.5,
                "affine": 0.6,
                "final": tissue,
            }
        },
        "residual_summary_um": {
            "used": {
                s: {**stage, "median": 1.5} for s in ("landmark", "affine", "final")
            },
            "held_out": {
                s: {**stage, "median": 2.5} for s in ("landmark", "affine", "final")
            },
        },
        "nucleus_qc": (
            {
                "status": "measured",
                "median_um": 6.5,
                "pearson_log_registration_channel": 0.3,
            }
            if nucleus
            else {"status": "disabled"}
        ),
        "resources": {"wall_time_s": 12.5, "peak_rss_gb": 1.5},
        "warnings": [f"warning of {run_id}"],
        "outputs": {"result_store": "does/not/exist.zarr"},
    }


def _write(tmp_path: Path, *summaries: dict[str, Any]) -> list[Path]:
    paths = []
    for summary in summaries:
        path = tmp_path / f"{summary['run_id']}.json"
        path.write_text(json.dumps(summary))
        paths.append(path)
    return paths


def _config(tmp_path: Path, paths: list[Path]) -> ManualRegistrationCompareConfig:
    return ManualRegistrationCompareConfig(
        dataset_name="SYN1",
        summary_paths=paths,
        output_dir=tmp_path / "durable",
        task_output_dir=tmp_path / "task",
    )


@pytest.fixture
def two_runs(tmp_path: Path) -> tuple[Path, list[Path]]:
    paths = _write(
        tmp_path,
        _summary("SYN1_serial_section", "serial_section", nucleus=False, tissue=0.7),
        _summary("SYN1_same_section", "same_section", nucleus=True, tissue=0.65),
    )
    return tmp_path, paths


def test_csv_columns_rows_and_values(two_runs: tuple[Path, list[Path]]) -> None:
    tmp_path, paths = two_runs
    outputs = compare_runs(_config(tmp_path, paths))
    with outputs["comparison_csv"].open() as handle:
        text = handle.read()
    assert text.splitlines()[0] == ",".join(COMPARISON_COLUMNS)
    rows = list(csv.DictReader(text.splitlines()))
    assert [row["run_id"] for row in rows] == [
        "SYN1_same_section",
        "SYN1_serial_section",
    ]
    assert rows[0]["nucleus_qc_median_um"] == "6.5"
    assert rows[1]["nucleus_qc_median_um"] == ""
    assert rows[0]["jacobian_min"] == "0.8"
    assert rows[0]["tissue_pearson_final"] == "0.65"
    assert rows[0]["held_out_median_final_um"] == "2.5"
    assert rows[0]["used_median_landmark_um"] == "1.5"
    assert rows[0]["wall_time_s"] == "12.5"


def test_json_and_figures_in_both_directories(
    two_runs: tuple[Path, list[Path]],
) -> None:
    tmp_path, paths = two_runs
    outputs = compare_runs(_config(tmp_path, paths))
    payload = json.loads(outputs["comparison_json"].read_text())
    assert payload["schema"] == "merxen.manual_registration.comparison"
    assert payload["dataset_name"] == "SYN1"
    assert payload["runs"][0]["warnings"] == ["warning of SYN1_same_section"]
    assert payload["runs"][1]["warnings"] == ["warning of SYN1_serial_section"]
    for directory in (tmp_path / "durable", tmp_path / "task"):
        for suffix in (".png", ".pdf", ".csv", ".json"):
            assert (directory / f"SYN1_run_comparison{suffix}").stat().st_size > 0
    assert set(outputs) == {"comparison_json", "comparison_csv", "comparison_png"}
    assert outputs["comparison_png"].parent == tmp_path / "durable"


def test_invalid_inputs_are_refused(tmp_path: Path) -> None:
    good = _summary("SYN1_same_section", "same_section", nucleus=False, tissue=0.5)
    other = _summary("SYN1_serial_section", "serial_section", nucleus=False, tissue=0.5)
    wrong_dataset = {**other, "dataset_name": "SYN2"}
    with pytest.raises(ValueError, match="SYN2"):
        compare_runs(_config(tmp_path, _write(tmp_path, good, wrong_dataset)))
    duplicate = {**other, "run_id": good["run_id"]}
    dup_dir = tmp_path / "dup"
    dup_dir.mkdir()
    paths = [dup_dir / "a.json", dup_dir / "b.json"]
    paths[0].write_text(json.dumps(good))
    paths[1].write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="unique"):
        compare_runs(_config(tmp_path, paths))
    failed = {**other, "status": "failed"}
    with pytest.raises(ValueError, match="not complete"):
        compare_runs(_config(tmp_path, _write(tmp_path, good, failed)))
    missing = _config(tmp_path, [*_write(tmp_path, good), tmp_path / "absent.json"])
    with pytest.raises(FileNotFoundError, match="absent.json"):
        compare_runs(missing)


def test_comparison_row_tolerates_missing_blocks() -> None:
    row = comparison_row({"run_id": "SYN1_same_section"})
    assert list(row) == list(COMPARISON_COLUMNS)
    assert row["run_id"] == "SYN1_same_section"
    assert row["fit_rms_um"] is None and row["jacobian_max"] is None
