"""NATIVE_LATEST entry points exist with their final signatures (AT-6 WP-F).

WP-A replaces the bodies and this file with ``test_native_latest.py``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from merxen.cli import main
from merxen.config import NativeLatestConfig
from merxen.segmentation import native


def test_signatures_are_final() -> None:
    build = inspect.signature(native.build_native_latest)
    assert list(build.parameters) == ["cfg"]
    ensure = inspect.signature(native.ensure_native_elements)
    assert list(ensure.parameters) == ["latest_path", "source_path"]


def test_stubs_raise_until_wp_a(tmp_path: Path) -> None:
    cfg = NativeLatestConfig(
        dataset_name="PAIR_A_ATERA",
        platform="ATERA",
        source_zarr_path=tmp_path / "native_source.zarr",
        output_dir=tmp_path / "native_out",
        persistent_latest_zarr_path=tmp_path / "latest.zarr",
    )
    with pytest.raises(NotImplementedError, match="WP-A"):
        native.build_native_latest(cfg)
    with pytest.raises(NotImplementedError, match="WP-A"):
        native.ensure_native_elements(tmp_path / "latest.zarr", tmp_path / "src.zarr")


def test_cli_command_is_registered(tmp_path: Path) -> None:
    assert "native-latest" in main.commands
    config = tmp_path / "native_latest_config.json"
    config.write_text(
        json.dumps(
            {
                "dataset_name": "PAIR_A_ATERA",
                "platform": "ATERA",
                "source_zarr_path": "native_source.zarr",
                "output_dir": "native_out",
                "persistent_latest_zarr_path": "latest.zarr",
            }
        )
    )
    result = CliRunner().invoke(main, ["native-latest", "--config", str(config)])
    assert isinstance(result.exception, NotImplementedError)


def test_renames_match_the_registry() -> None:
    from merxen.platforms import platform_spec

    spec = platform_spec("ATERA")
    assert spec.original_cell_shape_key == native.NATIVE_CELL_SHAPES_KEY
    assert spec.original_nucleus_shape_key == native.NATIVE_NUCLEUS_SHAPES_KEY
    renamed = {target for _, target in native.SOURCE_ELEMENT_RENAMES.values()}
    assert renamed == {
        spec.original_cell_shape_key,
        spec.original_nucleus_shape_key,
        native.NATIVE_TABLE_KEY,
    }
