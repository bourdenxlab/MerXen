"""Small helpers for staging and cleaning filesystem paths."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

INCOMPLETE_MARKER_SUFFIX = ".merxen-incomplete"


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


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry change (a rename or unlink) to disk, best effort."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: Path, text: str) -> Path:
    """Write a text file so that a reader never sees part of it.

    The text goes to a hidden temporary file in the same directory, is flushed to
    disk, and then replaces ``path`` in one rename. A task killed mid-write leaves
    the earlier file (or none), never a truncated one.

    Args:
        path: Destination file.
        text: Full file contents.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)
    return path


def incomplete_marker_path(target: Path, *, name: str | None = None) -> Path:
    """Return the hidden "write in progress" marker that sits beside an output.

    A step writes the marker before it starts changing a persistent output and
    removes it after its last write, so an output with a marker beside it was
    left by an interrupted (killed or failed) run and must not be reused. Outputs
    written before markers existed have none and are reused as before.

    Args:
        target: The output file or directory the marker describes.
        name: Marker stem, when one marker covers several outputs in a folder
            (default: the target's own name).

    Returns:
        ``<folder>/.<name><INCOMPLETE_MARKER_SUFFIX>``.
    """
    target = Path(target)
    return target.with_name(f".{name or target.name}{INCOMPLETE_MARKER_SUFFIX}")


def mark_incomplete(marker: Path, *, step: str) -> Path:
    """Record, durably, that ``step`` is about to write the outputs of ``marker``.

    Args:
        marker: Path from :func:`incomplete_marker_path`.
        step: Name of the writing step, for the person reading the marker.

    Returns:
        The marker path.
    """
    payload = {
        "step": step,
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meaning": (
            "Written before this step changes its outputs and removed after its "
            "last write. If it is still here, the run was interrupted and the "
            "outputs beside it are incomplete; the next run discards them."
        ),
    }
    return write_text_atomic(Path(marker), json.dumps(payload, indent=2) + "\n")


def clear_incomplete(marker: Path) -> None:
    """Remove a marker after the step's last write (no-op when it is absent)."""
    marker = Path(marker)
    marker.unlink(missing_ok=True)
    _fsync_directory(marker.parent)
