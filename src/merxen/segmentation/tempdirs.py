"""Where segmentation steps put large temporary files."""

from __future__ import annotations

import os
from pathlib import Path

MERXEN_TMPDIR_ENV = "MERXEN_TMPDIR"


def large_temp_root(explicit: Path | str | None = None) -> Path:
    """Return the directory that large temporary files should be created in.

    Segmentation steps write multi-gigabyte intermediates (partitioned
    transcript parquets, transcript CSVs, mask arrays). The system temporary
    directory is often on a small root disk, and on 2026-09-30 it filled up
    and failed two ProSeg tasks. These files therefore go, in order of
    precedence, to ``explicit``, to ``$MERXEN_TMPDIR``, or to the current
    working directory. Inside a Nextflow task that is the task's own work
    directory, on the same disk as its outputs and cleaned up with it.

    Args:
        explicit: Directory chosen by the caller, if any.

    Returns:
        The directory, created if it did not exist.
    """
    if explicit is not None:
        root = Path(explicit)
    elif os.environ.get(MERXEN_TMPDIR_ENV):
        root = Path(os.environ[MERXEN_TMPDIR_ENV])
    else:
        root = Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    return root
