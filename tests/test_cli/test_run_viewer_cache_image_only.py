"""The build-viewer-caches command: image-only mode and the unchanged main mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from merxen.cli import run_viewer_cache
from merxen.cli.run_viewer_cache import build_viewer_caches_command
from merxen.io.spatialdata_schema import stamp_merxen_schema


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.integration
def test_image_only_config_builds_both_pyramids(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    sdata = SpatialData(
        images={
            "reference": Image2DModel.parse(
                rng.integers(0, 4000, (2, 1300, 1100), dtype=np.uint16),
                dims=("c", "y", "x"),
                c_coords=["DAPI", "AF568"],
                transformations={"global": Scale([0.5, 0.5], axes=("x", "y"))},
            ),
            "moving": Image2DModel.parse(
                rng.integers(0, 4000, (3, 1200, 1250), dtype=np.uint16),
                dims=("c", "y", "x"),
                c_coords=["DAPI", "AF568", "AF647"],
                transformations={"global": Scale([0.4, 0.4], axes=("x", "y"))},
            ),
        }
    )
    stamp_merxen_schema(sdata)
    store = tmp_path / "store.zarr"
    sdata.write(store)
    out = tmp_path / "out"
    config = _write_config(
        tmp_path / "config.json",
        {
            "dataset_name": "T1_prepared",
            "latest_zarr_path": str(store),
            "output_dir": str(out),
            "image_only": True,
            "image_keys": ["reference", "moving"],
            "min_size": 256,
        },
    )
    runner = CliRunner()

    result = runner.invoke(build_viewer_caches_command, ["--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "image_pyramid[reference]: built" in result.output
    summary_path = out / "t1_prepared_viewer_cache_summary.json"
    assert json.loads(summary_path.read_text()) == {
        "image_pyramids": {"moving": "built", "reference": "built"}
    }

    rerun = runner.invoke(build_viewer_caches_command, ["--config", str(config)])
    assert "image_pyramid[reference]: skipped" in rerun.output
    forced = runner.invoke(
        build_viewer_caches_command, ["--config", str(config), "--force-rerun"]
    )
    assert "image_pyramid[reference]: built" in forced.output


def test_main_pipeline_mode_still_calls_build_viewer_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"masks": {}, "image_pyramid": "skipped"}

    def must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("image-only builder called in main-pipeline mode")

    monkeypatch.setattr(run_viewer_cache, "build_viewer_caches", fake_build)
    monkeypatch.setattr(run_viewer_cache, "build_image_pyramid_caches", must_not_run)
    out = tmp_path / "out"
    config = _write_config(
        tmp_path / "config.json",
        {
            "dataset_name": "T1",
            "platform": "XENIUM",
            "latest_zarr_path": str(tmp_path / "latest.zarr"),
            "original_data_path": str(tmp_path / "original"),
            "output_dir": str(out),
        },
    )

    result = CliRunner().invoke(build_viewer_caches_command, ["--config", str(config)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["platform"] == "XENIUM"
    assert calls[0]["original_data_path"] == tmp_path / "original"
    assert "image_pyramid: skipped" in result.output
    assert (out / "t1_viewer_cache_summary.json").exists()


def test_image_only_mode_does_not_need_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_build(zarr_path: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((zarr_path, kwargs))
        return {"image_pyramids": {}}

    monkeypatch.setattr(run_viewer_cache, "build_image_pyramid_caches", fake_build)
    config = _write_config(
        tmp_path / "config.json",
        {
            "dataset_name": "T1_prepared",
            "latest_zarr_path": str(tmp_path / "latest.zarr"),
            "output_dir": str(tmp_path / "out"),
            "image_only": True,
            "image_keys": ["reference", "moving"],
        },
    )

    result = CliRunner().invoke(build_viewer_caches_command, ["--config", str(config)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][1]["image_keys"] == ["reference", "moving"]
    assert calls[0][1]["params"].force is False
