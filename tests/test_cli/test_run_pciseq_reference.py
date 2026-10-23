"""Reference preparation CLI keeps source selection separate from model compute."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from merxen.cli.run_pciseq_reference import prepare_pciseq_reference_command


def test_reference_cli_passes_validated_species_regions_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.analysis import pciseq_reference

    destination = tmp_path / "reference_out"
    config_path = tmp_path / "reference.json"
    config_path.write_text(
        json.dumps(
            {
                "species": "mouse",
                "region_labels": ["MOp"],
                "output_dir": str(destination),
            }
        )
    )

    def prepare(config: Any, output_dir: Path) -> dict[str, Path]:
        assert config.source == "wmb"
        assert config.region_labels == ["MOp"]
        assert output_dir == destination
        return {"reference_csv": destination / "reference.csv"}

    monkeypatch.setattr(pciseq_reference, "prepare_pciseq_reference", prepare)
    result = CliRunner().invoke(
        prepare_pciseq_reference_command, ["--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output
    assert "reference.csv" in result.output
