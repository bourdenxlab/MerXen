"""Small helpers for staging and cleaning filesystem paths."""

from __future__ import annotations

import shutil
from pathlib import Path


def remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory tree if it exists."""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            return
        return
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def stage_existing_output(source_path: Path, output_path: Path) -> None:
    """Expose an existing file or directory at a new path."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path == source_path:
        return

    remove_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.symlink_to(
            source_path,
            target_is_directory=source_path.is_dir(),
        )
    except OSError:
        if source_path.is_dir():
            shutil.copytree(source_path, output_path, symlinks=True)
        else:
            shutil.copy2(source_path, output_path)
