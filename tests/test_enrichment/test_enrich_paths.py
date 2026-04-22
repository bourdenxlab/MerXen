"""Path-handling tests for enrichment staging helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from merxen.enrichment.enrich import (
    MERSCOPE_OLD_SHAPE_NAME,
    MERSCOPE_ZPROJ_IMAGE_NAME,
    MOSAIK_CELLPOSE_SHAPE_NAME,
    MOSAIK_PROSEG_SHAPE_NAME,
    ORIGINAL_TABLE_NAME,
    _is_already_enriched,
    _remove_path,
)


def test_remove_path_unlinks_directory_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    """Removing a staged symlink should not delete the upstream directory."""
    target = tmp_path / "target.zarr"
    target.mkdir()
    (target / "marker.txt").write_text("keep me")

    staged = tmp_path / "staged.zarr"
    staged.symlink_to(target, target_is_directory=True)

    _remove_path(staged)

    assert not staged.exists()
    assert not staged.is_symlink()
    assert target.exists()
    assert (target / "marker.txt").read_text() == "keep me"


def test_remove_path_deletes_real_directory_tree(tmp_path: Path) -> None:
    """Real directories should still be removed recursively."""
    out_dir = tmp_path / "out.zarr"
    out_dir.mkdir()
    (out_dir / "marker.txt").write_text("remove me")

    _remove_path(out_dir)

    assert not out_dir.exists()


def test_remove_path_ignores_missing_entries_during_rmtree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A partially-disappeared tree should not crash cleanup."""
    out_dir = tmp_path / "out.zarr"
    out_dir.mkdir()

    def _fake_rmtree(path: Path) -> None:
        raise FileNotFoundError("simulated race while removing a child entry")

    monkeypatch.setattr(shutil, "rmtree", _fake_rmtree)

    _remove_path(out_dir)


def test_is_already_enriched_checks_platform_specific_merscope_layers() -> None:
    """MERSCOPE completeness should depend on platform-specific shapes/images."""
    sdata = SimpleNamespace(
        shapes={
            MOSAIK_PROSEG_SHAPE_NAME: object(),
            MOSAIK_CELLPOSE_SHAPE_NAME: object(),
            MERSCOPE_OLD_SHAPE_NAME: object(),
        },
        tables={ORIGINAL_TABLE_NAME: object()},
        images={MERSCOPE_ZPROJ_IMAGE_NAME: object()},
    )

    assert _is_already_enriched(sdata, "MERSCOPE")

    del sdata.images[MERSCOPE_ZPROJ_IMAGE_NAME]

    assert not _is_already_enriched(sdata, "MERSCOPE")
