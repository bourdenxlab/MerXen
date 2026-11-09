"""The annotation-pilot command validates its config and reports the archive."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from merxen.cli import main, run_cellpose_hitl
from merxen.segmentation.hitl.export import HitlExportConfig, HitlExportResult


def test_export_command_is_registered_and_passes_validated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = HitlExportConfig(
        source_zarr=tmp_path / "source.zarr",
        baseline_stats_path=tmp_path / "stats.json",
        output_dir=tmp_path / "pilot",
        protected_roots=[tmp_path / "published"],
    )
    path = tmp_path / "pilot.json"
    path.write_text(config.model_dump_json())
    received = []

    def export(settings: HitlExportConfig) -> HitlExportResult:
        received.append(settings)
        return HitlExportResult(
            settings.output_dir, settings.output_dir / "pilot.tar.gz", 6
        )

    monkeypatch.setattr(run_cellpose_hitl, "export_annotation_pilot", export)
    result = CliRunner().invoke(main, ["cellpose-hitl-export", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert received == [config]
    assert "Exported 6 annotation crops" in result.output


def test_prepare_rois_command_is_registered_and_writes_only_requested_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = HitlExportConfig(
        source_zarr=tmp_path / "source.zarr",
        baseline_stats_path=tmp_path / "stats.json",
        output_dir=tmp_path / "pilot",
        protected_roots=[tmp_path / "published"],
    )
    path = tmp_path / "pilot.json"
    path.write_text(config.model_dump_json())
    output = tmp_path / "context.json"
    received = []

    def prepare(settings: HitlExportConfig, destination: Path) -> None:
        received.append((settings, destination))
        destination.write_text("{}")

    monkeypatch.setattr(run_cellpose_hitl, "prepare_viewer_roi_context", prepare)
    result = CliRunner().invoke(
        main,
        ["cellpose-hitl-prepare-rois", "--config", str(path), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert received == [(config, output)]
    assert output.exists() and not config.output_dir.exists()
