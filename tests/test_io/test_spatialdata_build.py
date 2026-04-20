"""Tests for SpatialData build-step orchestration."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from merxen.config import SpatialDataBuildConfig
from merxen.io.builders.pipeline import build_spatialdata_artifact


def test_build_spatialdata_artifact_reuses_existing_persistent_zarr(
    tmp_path: Path,
) -> None:
    """A persistent cached zarr should be staged without rebuilding."""
    persistent_zarr = tmp_path / "cache" / "source.zarr"
    persistent_zarr.mkdir(parents=True)
    output_path = tmp_path / "staged" / "source.zarr"
    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=tmp_path / "missing_raw_dir",
        output_path=output_path,
        persistent_output_path=persistent_zarr,
    )

    out = build_spatialdata_artifact(cfg)

    assert out == output_path
    assert output_path.exists()
    assert output_path.resolve() == persistent_zarr.resolve()


def test_build_spatialdata_artifact_force_requires_raw_input(
    tmp_path: Path,
) -> None:
    """Force rebuild should fail if only an existing zarr was supplied."""
    persistent_zarr = tmp_path / "cache" / "source.zarr"
    persistent_zarr.mkdir(parents=True)
    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=persistent_zarr,
        output_path=tmp_path / "staged" / "source.zarr",
        persistent_output_path=persistent_zarr,
    )

    with pytest.raises(ValueError, match="force-rerun"):
        build_spatialdata_artifact(cfg, force_rerun=True)


def test_build_spatialdata_artifact_dispatches_to_platform_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The orchestrator should call the correct platform writer for raw input."""
    raw_dir = tmp_path / "merscope_raw"
    raw_dir.mkdir()
    output_path = tmp_path / "stage" / "source.zarr"
    calls: list[tuple[Path, Path]] = []

    fake_module = types.ModuleType("merxen.io.builders.merscope")

    def _fake_write_merscope_spatialdata(**kwargs: object) -> Path:
        input_path = Path(kwargs["input_path"])
        final_output_path = Path(kwargs["output_path"])
        final_output_path.mkdir(parents=True)
        calls.append((input_path, final_output_path))
        return final_output_path

    fake_module.write_merscope_spatialdata = _fake_write_merscope_spatialdata  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "merxen.io.builders.merscope", fake_module)

    cfg = SpatialDataBuildConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        input_path=raw_dir,
        output_path=output_path,
    )

    out = build_spatialdata_artifact(cfg, force_rerun=True)

    assert out == output_path
    assert calls == [(raw_dir, output_path)]
