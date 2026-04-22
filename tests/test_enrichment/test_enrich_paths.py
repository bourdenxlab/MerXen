"""Path-handling tests for enrichment staging helpers."""

from __future__ import annotations

from pathlib import Path

from merxen.enrichment.enrich import _remove_path


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
